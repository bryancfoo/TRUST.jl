# The block-structured Jacobian: FitJacobian's storage, the per-bin forward-mode dual pass, the
# chain globals' closed forms, jvp!/jtvp!/gradient!, and the rule that J is never formed densely
# outside a test. Step 3.2b, chunk 6 of Phase 3. Spec D17-D21.
#
# The chain, exactly as measure.jl's apply! implements it (D17): M = B + g .* (K_lambda * P *
# transpose(K_t)), P the ideal (pre-instrument) streak. A per-bin entry (variable v, bin j)
# therefore has dM[:,n]/dtheta_(v,j) = g .* (K_lambda * dP[:,j]/dtheta_(v,j)) * K_t[n,j]: the
# ideal streak's Jacobian is block-diagonal in the bins, but the temporal IRF spreads block j
# over every measured column n with K_t[n,j] non-negligible, so J^T J is block-BANDED, not
# block-diagonal (Pitfall 5).

import ForwardDiff
using LinearAlgebra: mul!, dot

"""
    JACOBIAN_CHUNK

The ForwardDiff chunk width used for every per-bin dual pass: `8`. One constant for the whole
problem (not per bin), because the dual type `Dual{Tag,FT,C}` is part of every cache's type and
of every method the per-bin kernel specializes on -- a per-bin `C` would force a fresh
specialization (and hence a recompile of the kernel) per distinct block width. Eight was chosen
against `SuperGaussian`'s per-chunk construction cost (Phase 1.SG); D43 item 1 measures the
Jacobian's cost at `C = 4, 8, 12` and reports whether 8 is the right constant.

Spec D18.
"""
const JACOBIAN_CHUNK = 8

struct _JacobianTag end

"""
    FitJacobian{FT<:Real}

The assembled Jacobian's storage, block by block (D17): never a dense `N_pixels x N_theta`
matrix outside a test.

- `blocks[j]` is `N_lambda x p_j`: `dP[:,j] / d(bin-j entries)`, `p_j = length(bin_entries[j])`.
- `kernel_globals` is `N_lambda x N_t x n_kernel_global`: `dP / d(kernel global)`, one whole
  `N_lambda x N_t` slice per scalar fitted variable that affects `P` (bound to a plasma leaf, or
  to an in-kernel effect's parameter -- `KSmearing`, `GradientSmearing`, `WavelengthCalibration`).
- `chain_globals[c]` is one `N_lambda x N_t` matrix per scalar fitted variable bound to a
  chain-stage effect parameter (`SpectralIRF`/`TemporalIRF`'s `sigma`, `Sensitivity`'s `scale`,
  or a background coefficient): `dM / d(chain global)`, already the closed form times the
  logit's `dx/dy` (D3), so every entry of `chain_globals` is already in theta-space.
- `kernel_global_idx[k]` / `chain_global_idx[c]` are the theta index each column of
  `kernel_globals`/`chain_globals` belongs to -- not in D17's own struct sketch (which, like
  D15's `FitProblem` box before it, omitted a field `jvp!`/`jtvp!` need); added here for the same
  reason D15's `pixel_index` was, and flagged in the chunk 6 report.
- `response` is `g = scale .* response_curve` (or `scale` alone with no curve; `1` with no
  `Sensitivity` at all), length `N_lambda`.
- `Klambda`/`Kt` are the value-only convolution matrices `K_lambda`/`K_t` (`nothing` when that
  IRF is absent, checked once per product rather than once per element, D17).

A variable may be both a kernel global and a chain global (D11); its column is then the sum the
two lists separately contribute in `jvp!`/`jtvp!` (JT8) -- no special-casing is written for it,
which is deliberate (Pitfall 7).

`reg_rows`/`reg_cols`/`reg_coeffs` are the regularizer's own closed-form Jacobian entries (Step
3.3, D25), as parallel sparse-triple vectors: `∂rho_row/∂theta_col = coeff`, `row` already in the
combined `npixels + reg.nrows` numbering (`row` is `problem.npixels` less than the row's own
index into `r_full`, i.e. `jvp!`/`jtvp!` add `problem.npixels` when indexing `out`/`w`). Empty
when `problem.regularizer === nothing`. Not part of D17's original sketch (written before the
regularizer existed); added here the same way `kernel_global_idx`/`chain_global_idx` were,
flagged in the chunk 7 report.

Spec D17, D25.
"""
struct FitJacobian{FT<:Real}
    blocks::Vector{Matrix{FT}}
    kernel_globals::Array{FT,3}
    chain_globals::Vector{Matrix{FT}}
    kernel_global_idx::Vector{Int}
    chain_global_idx::Vector{Int}
    response::Vector{FT}
    Klambda::Union{Nothing,Matrix{FT}}
    Kt::Union{Nothing,Matrix{FT}}
    reg_rows::Vector{Int}
    reg_cols::Vector{Int}
    reg_coeffs::Vector{FT}
end

# ------------------------------------------------------------------------------------------
# D18: the per-bin dual pass -- seeding, chunking, splitting
# ------------------------------------------------------------------------------------------

"""
    _seed_dual(::Type{Dual{Tg,FTv,C}}, value, pos::Int, lo::Int) -> Dual{Tg,FTv,C}

`value` as a dual of the given type: a one-hot partial at local chunk position `pos - lo` when
`pos` falls in this sweep's window `(lo, lo+C]`, all-zero partials otherwise. `pos` is the
entry's position in the bin's combined seed order (bin-local entries first, D18); `lo = sweep *
C`. Building a fresh `Partials` tuple every call (rather than mutating one) is what avoids a
stale partial surviving from a previous sweep or a previous bin (Pitfall 6).
"""
@inline function _seed_dual(::Type{ForwardDiff.Dual{Tg,FTv,C}}, value, pos::Int, lo::Int) where {Tg,FTv,C}
    v = FTv(value)
    if lo < pos <= lo + C
        k = pos - lo
        return ForwardDiff.Dual{Tg}(v, ForwardDiff.Partials(ntuple(m -> FTv(m == k), Val(C))))
    else
        return ForwardDiff.Dual{Tg}(v, ForwardDiff.Partials(ntuple(_ -> zero(FTv), Val(C))))
    end
end

# Allocation-free rebuild of the in-kernel effects' one bound scalar parameter each (D18's "do
# not seed chain globals here": only KSmearing/GradientSmearing/WavelengthCalibration ever appear
# here, since a chain-stage effect's parameter never enters P). Written by hand, not via
# parameters.jl's `_rebuild_effect`, which allocates a temporary Vector per field (fine in
# `unflatten`, not fine here: JT11 requires this per-bin pass to allocate zero bytes once warmed).
# WavelengthCalibration is not yet bindable through this path in chunk 3 (its two-parameter
# rebuild needs D11(b)'s tuple-of-bindings-per-effect, Step 4.A.1b); the generic ::Nothing
# fallback covers it, exactly as it covered every other unbound effect through Phase 3.
_rebuild_kernel_effect(e::AbstractInstrumentEffect, ::Nothing, vals) = e
_rebuild_kernel_effect(e::KSmearing, b::Binding, vals) = reconstruct(e, (half_angle=vals[b.slot] * b.factor,))
_rebuild_kernel_effect(e::GradientSmearing, b::Binding, vals) = reconstruct(e, (width=vals[b.slot] * b.factor,))

_rebuild_kernel_tuple(::Tuple{}, ::Tuple{}, vals) = ()
function _rebuild_kernel_tuple(effects::Tuple, bindings::Tuple, vals)
    e = first(effects)
    b = first(bindings)
    return (_rebuild_kernel_effect(e, b, vals),
        _rebuild_kernel_tuple(Base.tail(effects), Base.tail(bindings), vals)...)
end

"""
    _jacobian_bin!(block, kernel_globals, j, work_trace, vals, col_cache, pspec, nvar, G,
                   kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
                   probe0, collection0, lambda0, prefactor_flag, theta)

One bin's dual pass (D18): for each sweep, seed the bin's own entries and every kernel global
into `vals` (through the logit, so the dual arithmetic carries `dx/dy` for free -- D3's
`_dx_dy` is needed only for the chain globals, computed separately), evaluate the dependents,
write the leaf rows for bin `j` into `work_trace`, rebuild the in-kernel effects that are bound,
run the kernel, and split the result's partials into `block` and `kernel_globals[:, j, :]`.

Written as its own named function (not a closure inside `foreach_bin!`'s do-block) so that the
`@allocated` measurement of JT11 is behind a genuine function barrier consuming only what the
loop needs, per `conventions.md` 5.2 and D43's "the function barrier cuts both ways" box.

Spec D18.
"""
function _jacobian_bin!(block::AbstractMatrix{FT}, kernel_globals::AbstractArray{FT,3}, j::Int,
    work_trace, vals::Vector{DFT}, col_cache, pspec::ParameterSpec, nvar::Int, G::Int,
    kernel_global_idx::Vector{Int}, kernel_global_of_var::Vector{Int}, kernel_effects0::Tuple,
    kernel_bindings::Tuple, probe0, collection0, lambda0, prefactor_flag::Bool,
    theta::AbstractVector{FT}) where {FT,DFT<:ForwardDiff.Dual}
    C = ForwardDiff.npartials(DFT)
    bj = pspec.bin_entries[j]
    bs = pspec.bin_slots[j]
    p_j = length(bj)
    n_j = p_j + G
    nsweeps = n_j == 0 ? 0 : cld(n_j, C)
    T = DFT

    for s in 0:(nsweeps-1)
        lo = s * C
        @inbounds for i in 1:nvar
            v = pspec.variables[i]
            if !v.vary
                vals[i] = convert(T, v.init)
            elseif !v.time_varying
                gk = kernel_global_of_var[i]
                if gk == 0
                    vals[i] = _from_unconstrained(convert(T, theta[pspec.offsets[i]]), v.lower, v.upper)
                else
                    y = _seed_dual(T, theta[pspec.offsets[i]], p_j + gk, lo)
                    vals[i] = _from_unconstrained(y, v.lower, v.upper)
                end
            else
                vals[i] = T(NaN)
            end
        end
        @inbounds for k in 1:p_j
            idx = bj[k]
            slot = bs[k]
            v = pspec.variables[slot]
            y = _seed_dual(T, theta[idx], k, lo)
            vals[slot] = _from_unconstrained(y, v.lower, v.upper)
        end
        tj = convert(T, pspec.axis_boundary[j])
        _eval_dependents!(vals, pspec.dependents, tj, nvar + 1)
        for b in pspec.bindings
            b.kind === :leaf || continue
            _slot_defined_at(b.slot, j, pspec.variables, nvar, pspec.intervals) || continue
            work_trace.values[b.row, j] = vals[b.slot] * b.factor
        end

        dual_kernel_effects = _rebuild_kernel_tuple(kernel_effects0, kernel_bindings, vals)

        p = plasma(work_trace, j)
        col = col_cache.column
        fill!(col, zero(eltype(col)))
        _accumulate!(col, p, probe0, collection0, lambda0, dual_kernel_effects, one(eltype(col)), prefactor_flag)

        @inbounds for k in 1:p_j
            pos = k
            (lo < pos <= lo + C) || continue
            kk = pos - lo
            for l in eachindex(col)
                block[l, k] = ForwardDiff.partials(col[l])[kk]
            end
        end
        @inbounds for gk in 1:G
            pos = p_j + gk
            (lo < pos <= lo + C) || continue
            kk = pos - lo
            for l in eachindex(col)
                kernel_globals[l, j, gk] = ForwardDiff.partials(col[l])[kk]
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# D19: the chain globals' closed forms
# ------------------------------------------------------------------------------------------

"""
    _scaled_wavelength(lambda) -> Vector

`s = (lambda .- lam0) ./ dlam`, `lam0 = (first+last)/2`, `dlam = (last-first)/2`: the scaled
wavelength `PolynomialBackground`/`RationalBackground`'s own `_prepare` computes
(`src/instrument/background.jl`), reproduced here rather than exposed there, since only the
final background vector is (D19: "written against the code").
"""
function _scaled_wavelength(lambda::AbstractVector)
    lam0 = (first(lambda) + last(lambda)) / 2
    dlam = (last(lambda) - first(lambda)) / 2
    return (lambda .- lam0) ./ dlam
end

function _rational_denom(e::RationalBackground{P0,Q0}, s::AbstractVector) where {P0,Q0}
    FT = promote_type(eltype(e.denominator), eltype(s))
    denom = Vector{FT}(undef, length(s))
    @inbounds for l in eachindex(s)
        acc = one(FT)
        spow = s[l]
        for k in 1:Q0
            acc += e.denominator[k] * spow
            spow *= s[l]
        end
        denom[l] = acc
    end
    return denom
end

function _rational_numer(e::RationalBackground{P0}, s::AbstractVector) where {P0}
    FT = promote_type(eltype(e.numerator), eltype(s))
    numer = Vector{FT}(undef, length(s))
    @inbounds for l in eachindex(s)
        acc = zero(FT)
        spow = one(FT)
        for k in 1:P0
            acc += e.numerator[k] * spow
            spow *= s[l]
        end
        numer[l] = acc
    end
    return numer
end

_chain_global_column(::Sensitivity, b::Binding, lambda, P, Q, response, curve0) = curve0 .* Q

function _chain_global_column(::ConstantBackground, b::Binding, lambda, P, Q, response, curve0)
    return ones(eltype(Q), size(P, 1), size(P, 2))
end

function _chain_global_column(::PolynomialBackground, b::Binding, lambda, P, Q, response, curve0)
    s = _scaled_wavelength(lambda)
    colvec = s .^ (b.element - 1)
    return repeat(colvec, outer=(1, size(P, 2)))
end

function _chain_global_column(e::RationalBackground, b::Binding, lambda, P, Q, response, curve0)
    s = _scaled_wavelength(lambda)
    denom = _rational_denom(e, s)
    colvec = if b.field === :numerator
        (s .^ (b.element - 1)) ./ denom
    else
        numer = _rational_numer(e, s)
        -(numer .* (s .^ b.element)) ./ (denom .^ 2)
    end
    return repeat(colvec, outer=(1, size(P, 2)))
end

_chain_global_column(e::ImageBackground, b::Binding, lambda, P, Q, response, curve0) = e.image

"""
    _chain_global_spectral_sigma(e, lambda, P, Kt, response) -> Matrix

`dM/dsigma_lambda = response .* ((dK_lambda/dsigma) * P * transpose(K_t))` (D19), `dK_lambda /
dsigma` from `ForwardDiff.partials.(_convolution_matrix(GaussianKernel(Dual(sigma,1)), lambda),
1)` -- never hand-derived (`_convolution_matrix` is generic in its element type, XP12).
"""
function _chain_global_spectral_sigma(e::SpectralIRF{<:GaussianKernel}, lambda::AbstractVector,
    P::AbstractMatrix, Kt::Union{Nothing,AbstractMatrix}, response::AbstractVector)
    sigma0 = e.kernel.sigma
    dsig = ForwardDiff.Dual(sigma0, one(sigma0))
    Kd = _convolution_matrix(GaussianKernel(dsig), lambda)
    dK = ForwardDiff.partials.(Kd, 1)
    inner = dK * P
    KtT = Kt === nothing ? inner : inner * transpose(Kt)
    return response .* KtT
end

"""
    _chain_global_temporal_sigma(e, t, P, Klambda, response) -> Matrix

`dM/dsigma_t = response .* (K_lambda * P * transpose(dK_t/dsigma))` (D19), the temporal
counterpart of `_chain_global_spectral_sigma`.
"""
function _chain_global_temporal_sigma(e::TemporalIRF{<:GaussianKernel}, t::AbstractVector,
    P::AbstractMatrix, Klambda::Union{Nothing,AbstractMatrix}, response::AbstractVector)
    sigma0 = e.kernel.sigma
    dsig = ForwardDiff.Dual(sigma0, one(sigma0))
    Kd = _convolution_matrix(GaussianKernel(dsig), t)
    dKt = ForwardDiff.partials.(Kd, 1)
    KP = Klambda === nothing ? P : Klambda * P
    return response .* (KP * transpose(dKt))
end

"""
    _jacobian_shared(expt) -> (response, curve0, Klambda, Kt)

The value-only quantities every chain global and `jvp!`/`jtvp!` share, built once per theta from
`expt`'s already-unflattened, already-`_prepare`d effects (never re-derived): `response = g =
scale .* curve` (`curve` the interpolated calibration response, or all-ones with no curve or no
`Sensitivity` at all); `curve0`, the same curve **without** `scale` (Sensitivity's own chain
global needs this, since `M`'s dependence on `scale` is exactly `curve0 .* (K_lambda P K_t^T)`);
`Klambda`/`Kt`, the value convolution matrices or `nothing`. Effects are told apart by
`_chain_order`, never by `isa` on a concrete effect type (`conventions.md` Section 4).
"""
function _jacobian_shared(expt::Experiment, nlambda::Int, ::Type{FT}) where {FT<:Real}
    Klambda = nothing
    Kt = nothing
    curve0 = ones(FT, nlambda)
    scale = one(FT)
    for (i, e) in enumerate(expt.effects)
        co = _chain_order(e)
        if co == 2
            Klambda = expt.operators[i]
        elseif co == 3
            Kt = expt.operators[i]
        elseif co == 4
            scale = FT(e.scale)
            op = expt.operators[i]
            op === nothing || (curve0 = FT.(op))
        end
    end
    response = scale .* curve0
    return response, curve0, Klambda, Kt
end

"""
    _build_chain_globals(problem, expt, theta, P, Q, response, curve0, ::Type{FT}) -> (chain_globals, chain_global_idx)

One `N_lambda x N_t` matrix per binding of a scalar fitted variable to a chain-stage (2 to 5)
effect parameter (D19), each already multiplied by the logit's `dx/dy` (D3) so it is in
theta-space, matching `blocks`/`kernel_globals`. Only a **directly fitted** variable is
supported (`b.slot <= nvar`): a dependent bound to a chain-stage parameter would need the chain
rule through the dependent's own expression, which is not exercised by any test problem of this
phase (D46) and is refused here rather than silently mishandled -- see the chunk 6 report.
"""
function _build_chain_globals(problem::FitProblem, expt::Experiment, theta::AbstractVector{FT},
    P::AbstractMatrix, Q::AbstractMatrix, response::AbstractVector, curve0::AbstractVector,
    Klambda, Kt) where {FT<:Real}
    pspec = problem.pspec
    nvar = length(pspec.variables)
    lambda = expt.lambda
    t = expt.t

    chain_globals = Matrix{FT}[]
    chain_global_idx = Int[]

    for b in pspec.bindings
        b.kind === :effect || continue
        e = expt.effects[b.effect]
        co = _chain_order(e)
        co in 2:5 || continue

        b.slot <= nvar || throw(ArgumentError(
            "jacobian: \"$(b.name)\" binds a dependent to a chain-stage instrument parameter " *
            "(effect $(b.effect), field :$(b.field)); this phase's Jacobian supports only a " *
            "directly fitted variable bound to a chain-stage parameter"))

        v = pspec.variables[b.slot]
        x = _from_unconstrained(theta[pspec.offsets[b.slot]], v.lower, v.upper)
        chainfactor = _dx_dy(x, v.lower, v.upper)

        col = if co == 2
            _chain_global_spectral_sigma(e, lambda, P, Kt, response)
        elseif co == 3
            _chain_global_temporal_sigma(e, t, P, Klambda, response)
        else
            _chain_global_column(e, b, lambda, P, Q, response, curve0)
        end
        push!(chain_globals, FT.(col .* chainfactor))
        push!(chain_global_idx, pspec.offsets[b.slot])
    end

    return chain_globals, chain_global_idx
end

"""
    _apply_irfs_value(P, expt) -> Matrix

`K_lambda * P * transpose(K_t)` (whichever of the two IRFs are present), reusing `measure.jl`'s
own `apply!` methods so this is never a second, independently-written implementation of the
convolution step (D19: "the code wins"). Value-only (`expt` is the plain, already-unflattened
experiment); Sensitivity and the backgrounds are deliberately not applied here, since this is
exactly the "pre-sensitivity" streak `Sensitivity.scale`'s own chain global needs.
"""
function _apply_irfs_value(P::AbstractMatrix, expt::Experiment)
    cur = copy(P)
    buf = similar(P)
    for (i, e) in enumerate(expt.effects)
        _chain_order(e) in (2, 3) || continue
        op = expt.operators[i]
        apply!(buf, cur, e, op)
        cur, buf = buf, cur
    end
    return cur
end

# ------------------------------------------------------------------------------------------
# D17/D18/D19 assembled: jacobian
# ------------------------------------------------------------------------------------------

"""
    jacobian(problem::FitProblem, theta::AbstractVector) -> FitJacobian

Assembles the block-structured Jacobian at `theta` (D17-D19): one forward-mode dual pass of the
per-bin kernel per bin per sweep for the per-bin entries and the kernel globals (D18), plus the
chain globals' closed forms (D19). `J` itself is never formed; `jvp!`/`jtvp!` consume
`FitJacobian`'s fields directly.

`chunk` overrides `JACOBIAN_CHUNK` (default); not part of D17-D21's own signature, added so
JT12 can assert the assembled Jacobian agrees with itself across `C = 4, 8, 12` without
redefining the `const` -- Level 3's addition, flagged under rule 11 zero in the chunk 6 report.

Spec D17, D18, D19.
"""
function jacobian(problem::FitProblem, theta::AbstractVector{FT};
    chunk::Integer=JACOBIAN_CHUNK) where {FT<:Real}
    pspec = problem.pspec
    nvar = length(pspec.variables)
    nt = nbins(pspec.trace)
    nlambda = length(problem.experiment.lambda)

    expt = unflatten(problem.experiment, pspec, theta)

    kernel_effects0 = _kernel_effects(expt)
    nkernel_eff = length(kernel_effects0)

    kernel_effect_binding = Union{Nothing,Binding}[nothing for _ in 1:nkernel_eff]
    for b in pspec.bindings
        if b.kind === :effect && 1 <= b.effect <= nkernel_eff && _chain_order(expt.effects[b.effect]) == 1
            kernel_effect_binding[b.effect] = b
        end
    end
    kernel_bindings = Tuple(kernel_effect_binding)

    kernel_global_var = Int[]
    for i in 1:nvar
        v = pspec.variables[i]
        (v.vary && !v.time_varying) || continue
        affects_P = false
        for b in pspec.bindings
            b.slot == i || continue
            if b.kind === :leaf
                affects_P = true
            elseif b.kind === :effect && 1 <= b.effect <= nkernel_eff && _chain_order(expt.effects[b.effect]) == 1
                affects_P = true
            end
        end
        affects_P && push!(kernel_global_var, i)
    end
    G = length(kernel_global_var)
    kernel_global_idx = [pspec.offsets[i] for i in kernel_global_var]
    kernel_global_of_var = zeros(Int, nvar)
    for (gk, i) in enumerate(kernel_global_var)
        kernel_global_of_var[i] = gk
    end

    max_p = nt == 0 ? 0 : maximum(j -> length(pspec.bin_entries[j]), 1:nt)
    C = max(1, min(max_p + G, Int(chunk)))
    DFT = ForwardDiff.Dual{_JacobianTag,FT,C}

    blocks = Vector{Matrix{FT}}(undef, nt)
    for j in 1:nt
        blocks[j] = Matrix{FT}(undef, nlambda, length(pspec.bin_entries[j]))
    end
    kernel_globals = Array{FT,3}(undef, nlambda, nt, G)

    probe0, collection0, lambda0 = expt.probe, expt.collection, expt.lambda
    prefactor_flag = _prefactor_flag(expt.prefactor)

    cache_factory = () -> (_widen_trace(pspec.trace, DFT), Vector{DFT}(undef, pspec.nslots),
        _make_cache(DFT, nlambda))

    foreach_bin!(nt, cache_factory; scheduler=problem.scheduler) do j, cache
        work_trace, vals, col_cache = cache
        _jacobian_bin!(blocks[j], kernel_globals, j, work_trace, vals, col_cache, pspec, nvar, G,
            kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
            probe0, collection0, lambda0, prefactor_flag, theta)
    end

    response, curve0, Klambda, Kt = _jacobian_shared(expt, nlambda, FT)
    P = ideal_streak(expt)
    Q = _apply_irfs_value(P, expt)
    chain_globals, chain_global_idx = _build_chain_globals(problem, expt, theta, P, Q, response,
        curve0, Klambda, Kt)

    if problem.regularizer === nothing
        reg_rows, reg_cols, reg_coeffs = Int[], Int[], FT[]
    else
        _, reg_rows, reg_cols, reg_coeffs_f64 = _regularizer_closed_form(problem.regularizer, pspec, theta)
        reg_coeffs = FT.(reg_coeffs_f64)
    end

    return FitJacobian{FT}(blocks, kernel_globals, chain_globals, kernel_global_idx,
        chain_global_idx, response, Klambda, Kt, reg_rows, reg_cols, reg_coeffs)
end

# ------------------------------------------------------------------------------------------
# D20: jvp!, jtvp!, gradient!
# ------------------------------------------------------------------------------------------

"""
    jvp!(out, problem::FitProblem, Jac::FitJacobian, v::AbstractVector) -> out

`out = J_full * v`, `out` of length `problem.npixels` when `problem.regularizer === nothing`,
`problem.npixels + problem.regularizer.nrows` when it is not (Step 3.3, D25: the regularizer's
rows are extra rows appended after the data rows, `out[npixels+row] = sum over the row's nonzero
columns of Jac.reg_coeffs[.] * v[Jac.reg_cols[.]]`, `Jac`'s own closed-form entries). Takes
`problem` as well as `Jac` (not given an explicit signature by D20's abbreviated header;
`bin_entries`, `pixel_index` and `sigma` all live on `problem`/`problem.pspec`, not on
`FitJacobian` -- Level 3's choice, flagged under rule 11 zero in the chunk 6 report, the same gap
D15's `FitProblem` box had for `pixel_index`).

Spec D20, D25.
"""
function jvp!(out::AbstractVector, problem::FitProblem, Jac::FitJacobian{FT}, v::AbstractVector) where {FT}
    pspec = problem.pspec
    nlambda, nt, G = size(Jac.kernel_globals)
    Q = zeros(FT, nlambda, nt)
    for j in 1:nt
        bj = pspec.bin_entries[j]
        isempty(bj) && continue
        @views mul!(Q[:, j], Jac.blocks[j], v[bj], true, true)
    end
    for (k, idx) in enumerate(Jac.kernel_global_idx)
        vk = v[idx]
        iszero(vk) && continue
        @views Q .+= vk .* Jac.kernel_globals[:, :, k]
    end
    Wl = Jac.Klambda === nothing ? Q : Jac.Klambda * Q
    W = Jac.Kt === nothing ? Wl : Wl * transpose(Jac.Kt)
    W .*= Jac.response
    for (c, idx) in enumerate(Jac.chain_global_idx)
        vc = v[idx]
        iszero(vc) && continue
        W .+= vc .* Jac.chain_globals[c]
    end
    idx = problem.pixel_index
    sig = problem.sigma
    @inbounds for k in eachindex(idx)
        i = idx[k]
        out[k] = W[i] / sig[i]
    end

    if problem.regularizer !== nothing
        nrows = problem.regularizer.nrows
        fill!(view(out, (problem.npixels+1):(problem.npixels+nrows)), zero(FT))
        @inbounds for m in eachindex(Jac.reg_rows)
            out[problem.npixels+Jac.reg_rows[m]] += Jac.reg_coeffs[m] * v[Jac.reg_cols[m]]
        end
    end
    return out
end

"""
    jtvp!(out, problem::FitProblem, Jac::FitJacobian, w::AbstractVector) -> out

`out = transpose(J_full) * w`, `out` of length `problem.pspec.ntheta` (D20). `w` is
`problem.npixels` long with no regularizer, `problem.npixels + problem.regularizer.nrows` long
with one -- the data half reads only `w[1:npixels]` (via `problem.pixel_index`, unaffected by the
extra length) and the regularizer half (Step 3.3, D25) adds
`out[col] += Jac.reg_coeffs[.] * w[npixels + Jac.reg_rows[.]]` over `Jac`'s closed-form entries.
Step 4 is `transpose(K_lambda)`
on the **left**, `K_t` **not** transposed on the right -- both change side relative to the
forward direction (`jvp!`'s step 4), and getting one wrong produces a `Jtw` that is plausible and
fails only where the kernels are asymmetric (Pitfall 5, JT3).

Spec D20, D25.
"""
function jtvp!(out::AbstractVector, problem::FitProblem, Jac::FitJacobian{FT}, w::AbstractVector) where {FT}
    pspec = problem.pspec
    nlambda, nt, G = size(Jac.kernel_globals)
    Z = zeros(FT, nlambda, nt)
    idx = problem.pixel_index
    sig = problem.sigma
    @inbounds for k in eachindex(idx)
        i = idx[k]
        Z[i] = w[k] / sig[i]
    end

    fill!(out, zero(FT))
    for (c, tidx) in enumerate(Jac.chain_global_idx)
        out[tidx] += dot(Jac.chain_globals[c], Z)
    end

    Y = Z .* Jac.response
    U1 = Jac.Klambda === nothing ? Y : transpose(Jac.Klambda) * Y
    U = Jac.Kt === nothing ? U1 : U1 * Jac.Kt

    for (k, tidx) in enumerate(Jac.kernel_global_idx)
        out[tidx] += dot(view(Jac.kernel_globals, :, :, k), U)
    end
    for j in 1:nt
        bj = pspec.bin_entries[j]
        isempty(bj) && continue
        @views out[bj] .+= transpose(Jac.blocks[j]) * U[:, j]
    end

    if problem.regularizer !== nothing
        @inbounds for m in eachindex(Jac.reg_rows)
            out[Jac.reg_cols[m]] += Jac.reg_coeffs[m] * w[problem.npixels+Jac.reg_rows[m]]
        end
    end
    return out
end

"""
    gradient!(g, problem::FitProblem, theta::AbstractVector) -> g

`grad_theta O = 2 * transpose(J_full) * r_full` (D26): `r_full` is `_full_residuals(problem,
theta)` (`residuals.jl`), the data residuals with the regularizer's rows appended when
`problem.regularizer !== nothing` (Step 3.3, D25) or `residuals(problem, theta)` unchanged when
it is `nothing`; `Jac`'s extended `jtvp!` (above) then contracts both halves in one call. Never
forms `J` densely to do it (JT10).

Spec D20, D25, D26.
"""
function gradient!(g::AbstractVector, problem::FitProblem, theta::AbstractVector)
    r = _full_residuals(problem, theta)
    Jac = jacobian(problem, theta)
    jtvp!(g, problem, Jac, r)
    g .*= 2
    return g
end

# ------------------------------------------------------------------------------------------
# Step 3.C.1, D16: the Gauss-Newton diagonal diag(J_full^T J_full) in y, one pass
# ------------------------------------------------------------------------------------------

"""
    _weighted_sumabs2(w, u) -> Real

`sum(w[r] * u[r]^2 for r in eachindex(w, u))`, allocation-light (no temporary array for `u.^2`):
`_gauss_newton_diagonal`'s innermost reduction, called once per per-bin theta entry and once per
kernel global.
"""
function _weighted_sumabs2(w::AbstractVector, u::AbstractVector)
    s = zero(promote_type(eltype(w), eltype(u)))
    @inbounds for r in eachindex(w, u)
        s += w[r] * u[r]^2
    end
    return s
end

"""
    _gauss_newton_diagonal(problem::FitProblem, Jac::FitJacobian) -> Vector{Float64}

`d_i = sum_rows (d r_full / d y_i)^2` (D16), the diagonal of `J_full^T J_full` in the unconstrained
`y` -- the diagonal preconditioner Step 3.C.3 would use, computed in one pass over `Jac`'s blocks
and the two kinds of global columns, **never from a dense `J`** (807 MB on the full problem against
a few MB of blocks). Follows `jvp!`'s own propagation exactly (D16 pitfall: "a diagonal that
disagrees with them is the diagonal of a different matrix"): a per-bin or kernel-global column of
`J_full` is `response .* ((Klambda-or-I) * Q * (Kt-or-I)^T) ./ sigma` before the mask, restricted to
`problem.pixel_index`, and `Q`'s only nonzero column(s) are the ones `e_i` touches -- so `blocks[j]`
and `kernel_globals[:,:,gk]` cannot be squared directly (as D16's own compressed English sketch
reads if taken literally): the temporal IRF `Kt` spreads a single bin's column over every other
bin's pixels (Pitfall 5, `JT6`), and `response`/`sigma` are not uniformly 1 on any deck this phase
measures (P1's own `Sensitivity(scale = 1e-6)` alone is a 1e-12 factor). This function instead:

- builds `invsig2[r,c] = 1/sigma[r,c]^2` at mask-true pixels and `0` elsewhere (`problem.sigma` and
  `problem.pixel_index`, the one definition of the pixel order, D16), and `respsig2 = response.^2
  .* invsig2`;
- for each bin `j`: `U = Klambda === nothing ? blocks[j] : Klambda * blocks[j]` (one matmul, D17's
  chain), then, for each of its entries `k` (`e_i`'s only nonzero `Q` column is `j`, so the entry's
  residual-space column is `U[:,k]` alone, spread over every `c` by `Kt[c,j]` when `Kt` is present):
  `Kt === nothing` gives `d_i = dot(respsig2[:,j], U[:,k].^2)`; `Kt` present gives
  `d_i = dot(U[:,k].^2, respsig2 * Kt[:,j].^2)` (the `Kt`-spread column, precomputed once per bin
  since it does not depend on `k`);
  a per-bin-then-per-column loop, `_weighted_sumabs2` the reduction;
- for each kernel global `gk` (`Q = kernel_globals[:,:,gk]`, already nonzero over every bin, D18):
  `W = (Kt-or-I)^T` applied to `(Klambda-or-I) * Q`, exactly `jvp!`'s own two matmuls, then
  `d += sum(respsig2 .* W.^2)`;
- for each chain global `c` (`chain_globals[c]`, already the full `dM/dtheta_c` including its own
  `Klambda`/`Kt`/`response` where the chain rule needs them, D19): `d += sum(invsig2 .*
  chain_globals[c].^2)` -- no further propagation, since `jvp!` adds it to `W` directly;
- for each regularizer triple `(row, col, coeff)` (`Jac.reg_rows`/`reg_cols`/`reg_coeffs`, D25,
  already in theta-space, independent of the pixel/sigma weighting): `d[col] += coeff^2`.

A theta index reachable from more than one of these (a variable that is both a kernel global and a
chain global, D11; a column a regularizer term also touches) accumulates (`+=`), never overwrites.

Nothing at Step 3.C.1 passes this to an optimizer; it is measured and printed (D16). Whether it
becomes a preconditioner is Step 3.C.3's question. Not part of D16's own signature (given as a
one-line summary, not a full algorithm); Level 3's implementation, verified against CV2's dense
oracle (`jvp!` on unit vectors) and flagged under rule 11 zero.

Spec D16.
"""
function _gauss_newton_diagonal(problem::FitProblem, Jac::FitJacobian)
    pspec = problem.pspec
    nlambda, nt, G = size(Jac.kernel_globals)
    d = zeros(Float64, pspec.ntheta)

    idx = problem.pixel_index
    sig = problem.sigma
    invsig2 = zeros(Float64, nlambda, nt)
    @inbounds for k in eachindex(idx)
        i = idx[k]
        invsig2[i] = 1.0 / Float64(sig[i])^2
    end
    response2 = Float64.(Jac.response) .^ 2
    respsig2 = response2 .* invsig2

    # --- per-bin blocks ---
    for j in 1:nt
        bj = pspec.bin_entries[j]
        isempty(bj) && continue
        U = Jac.Klambda === nothing ? Jac.blocks[j] : Jac.Klambda * Jac.blocks[j]
        if Jac.Kt === nothing
            wcol = view(respsig2, :, j)
            for (k, theta_idx) in enumerate(bj)
                d[theta_idx] += _weighted_sumabs2(wcol, view(U, :, k))
            end
        else
            colvec = respsig2 * abs2.(view(Jac.Kt, :, j))
            for (k, theta_idx) in enumerate(bj)
                d[theta_idx] += _weighted_sumabs2(colvec, view(U, :, k))
            end
        end
    end

    # --- kernel globals ---
    for gk in 1:G
        Q = Jac.kernel_globals[:, :, gk]
        Wl = Jac.Klambda === nothing ? Q : Jac.Klambda * Q
        W = Jac.Kt === nothing ? Wl : Wl * transpose(Jac.Kt)
        d[Jac.kernel_global_idx[gk]] += sum(respsig2 .* abs2.(W))
    end

    # --- chain globals ---
    for (c, theta_idx) in enumerate(Jac.chain_global_idx)
        d[theta_idx] += sum(invsig2 .* abs2.(Jac.chain_globals[c]))
    end

    # --- regularizer rows ---
    @inbounds for m in eachindex(Jac.reg_rows)
        d[Jac.reg_cols[m]] += Float64(Jac.reg_coeffs[m])^2
    end

    return d
end

# ------------------------------------------------------------------------------------------
# Step 3.C.3, D37: the floored diagonal -- the preconditioner's guard
# ------------------------------------------------------------------------------------------

"""
    PRECOND_FLOOR_REL

The relative floor `_gauss_newton_floor` applies to `_gauss_newton_diagonal`'s output before it is
used as a preconditioner (D37): `1e-8`, so the preconditioner's dynamic range is capped at
`1/PRECOND_FLOOR_REL = 1e8`. On the same footing as `BOUND_PROXIMITY_TOL`/`SATURATION_FRACTION`
(`parameters.jl`), not exported.

Spec D37.
"""
const PRECOND_FLOOR_REL = 1e-8

"""
    _gauss_newton_floor(d::AbstractVector{<:Real}) -> Union{Nothing,Vector{Float64}}

`max.(d, PRECOND_FLOOR_REL * maximum(d))` (D37), floored **relative** to `d`'s own maximum, never
absolute -- `d` carries the problem's own χ² scale, and a fixed absolute floor would mean a
different thing on every problem. `EPW`'s Gauss-Newton diagonal has **82 exactly-zero entries** at
its saturated point (chunk 5's report) and `1/(2*0)` is `Inf`; an `Inf` in a preconditioner's `P⁻¹`
is an infinite step, so this must run before anything divides by `d`.

Returns `nothing` when `maximum(d)` is zero or non-finite (every entry zero, or a `NaN`/`Inf`
entry present, since `Base.max` propagates both through `maximum`'s pairwise reduction) -- a
degenerate diagonal that cannot be floored relative to itself. The caller's own fallback (D37):
at the very first build this means running unpreconditioned; at a later refresh it means keeping
the previous preconditioner.

Spec D37.
"""
function _gauss_newton_floor(d::AbstractVector{<:Real})
    dmax = maximum(d)
    (isfinite(dmax) && dmax > 0) || return nothing
    floor_val = PRECOND_FLOOR_REL * dmax
    return max.(Float64.(d), floor_val)
end
