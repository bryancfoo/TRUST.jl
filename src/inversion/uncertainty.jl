# The uncertainty reports, first half -- Step 4.A.4a, chunk 9 of Phase 4.A. Spec D43-D47,
# dev/phase4A_instructions.md Section 10. Three things: the minimum (normal_matrices, gn_decrease,
# polish), the Laplace covariance (laplace), and the calibration scan (calibration_scan). Every
# report checks it is at a minimum and refuses loudly otherwise (D43, D45, D46) -- a joint fit's
# own "converged" is not proof of that (chunk 8's F2).

using LinearAlgebra: cholesky, Symmetric, dot

# ------------------------------------------------------------------------------------------
# D43: the theta layout, the free entries, the normal matrices, the Gauss-Newton decrease
# ------------------------------------------------------------------------------------------

"""
    _theta_layout(pspec::ParameterSpec) -> (entry_var::Vector{Int}, entry_bin::Vector{Int})

For each entry of theta: the index into `pspec.variables` it belongs to, and its bin (`0` for a
scalar/global entry). Read off `pspec.bin_entries`/`bin_slots` and `pspec.global_entries`/
`global_slots`, never assumed contiguous from `pspec.offsets` (D43).
"""
function _theta_layout(pspec::ParameterSpec)
    entry_var = zeros(Int, pspec.ntheta)
    entry_bin = zeros(Int, pspec.ntheta)
    for (j, entries) in enumerate(pspec.bin_entries)
        slots = pspec.bin_slots[j]
        for (k, idx) in enumerate(entries)
            entry_var[idx] = slots[k]
            entry_bin[idx] = j
        end
    end
    for (k, idx) in enumerate(pspec.global_entries)
        entry_var[idx] = pspec.global_slots[k]
    end
    return entry_var, entry_bin
end

"""
    _free_entries(pspec::ParameterSpec, theta::AbstractVector) -> BitVector

`true` at every theta entry whose **x**-space value is not within `BOUND_PROXIMITY_TOL` of its
variable's bound (D43). Near-bound entries are excluded from every solve `normal_matrices`,
`gn_decrease`, `polish` and `laplace` build below: there `dx/dy -> 0`, their column of `J` vanishes,
and `JᵀJ` is singular to round-off (Pitfall 33; Level 2 measured eigenvalues of -2.2e-10 with them
left in).
"""
function _free_entries(pspec::ParameterSpec, theta::AbstractVector)
    entry_var, _ = _theta_layout(pspec)
    free = trues(pspec.ntheta)
    for e in 1:pspec.ntheta
        v = pspec.variables[entry_var[e]]
        x = _from_unconstrained(theta[e], v.lower, v.upper)
        free[e] = !_near_bound(x, v.lower, v.upper)
    end
    return free
end

"""
    normal_matrices(problem::AbstractFitProblem, theta::AbstractVector) -> (JtJ, JdtJd)

`JᵀJ` and `J_dᵀJ_d` (`J_d` the data rows alone), both `ntheta x ntheta` `Matrix{Float64}`,
assembled from `ntheta` `jvp!`/`jtvp!` products -- **`J` is never formed densely** (Pitfall 13):

    for k in 1:ntheta                        # v = e_k
        jvp!(col, problem, J, v)             # col = J e_k, all rows
        jtvp!(out, problem, J, col); JtJ[:, k] .= out
        col[problem.npixels+1:end] .= 0.0    # data rows only
        jtvp!(out, problem, J, col); JdtJd[:, k] .= out

Symmetrized (`(A + Aᵀ)/2`) to cancel round-off asymmetry from the two `jtvp!` passes. Cost is
`ntheta` `jvp!` and `2*ntheta` `jtvp!` calls (D43; Level 2 measured 0.6-0.8 s at `ntheta = 128` on
the full-size two-channel problem, Section 12 row 13).

With no regularizer `problem.npixels == nrows`, so the "data rows only" zeroing is a no-op over an
empty range and `JdtJd` comes out **bitwise** equal to `JtJ` (UN2) without any special-casing.

Spec D43.
"""
function normal_matrices(problem::AbstractFitProblem, theta::AbstractVector)
    pspec = problem.pspec
    ntheta = pspec.ntheta
    nrows = problem.npixels + (problem.regularizer === nothing ? 0 : problem.regularizer.nrows)
    Jac = jacobian(problem, theta)

    JtJ = zeros(Float64, ntheta, ntheta)
    JdtJd = zeros(Float64, ntheta, ntheta)
    v = zeros(Float64, ntheta)
    col = Vector{Float64}(undef, nrows)
    out = Vector{Float64}(undef, ntheta)

    for k in 1:ntheta
        v[k] = 1.0
        jvp!(col, problem, Jac, v)
        jtvp!(out, problem, Jac, col)
        JtJ[:, k] .= out
        if problem.regularizer !== nothing
            view(col, (problem.npixels+1):nrows) .= 0.0
        end
        jtvp!(out, problem, Jac, col)
        JdtJd[:, k] .= out
        v[k] = 0.0
    end

    JtJ = (JtJ + JtJ') / 2
    JdtJd = (JdtJd + JdtJd') / 2
    return JtJ, JdtJd
end

"""
    GN_DECREASE_WARN

`1e-2`: the Gauss-Newton predicted decrease above which `fit(deck)` warns that its result may not
be a minimum (OQ-41 option 1, Phase 4.B spec D5). In `𝒪`'s own units a change of 1 is one σ, so
this reads "one more Gauss-Newton step would gain more than a hundredth of a σ" (OQ-36), the same
number `laplace`'s `max_decrease` defaults to. A report, never a status: `converged` is unchanged.
"""
const GN_DECREASE_WARN = 1e-2

"""
    gn_decrease(problem::AbstractFitProblem, theta::AbstractVector) -> Float64

The decrease of `𝒪` a full Gauss-Newton step from `theta` would achieve, restricted to the free
(not near-bound) entries: with `g = gradient!`'s `∇_y 𝒪` and `f = _free_entries(problem.pspec,
theta)`,

    gn_decrease = g[f]' * (JtJ[f,f] \\ g[f]) / 4

(a Cholesky solve). Since `𝒪 = rᵀr` and `g = 2Jᵀr`, the Gauss-Newton step is `Δ = -(JᵀJ)⁻¹g/2` and
the quadratic model's own predicted decrease at that step is exactly `gᵀ(JᵀJ)⁻¹g/4` -- **in 𝒪's own
units, where a change of 1 is one σ** (OQ-36), so this threshold means the same thing on every
problem.

Spec D43.
"""
function gn_decrease(problem::AbstractFitProblem, theta::AbstractVector)
    pspec = problem.pspec
    JtJ, _ = normal_matrices(problem, theta)
    f = _free_entries(pspec, theta)
    g = zeros(Float64, pspec.ntheta)
    gradient!(g, problem, theta)
    gf = g[f]
    return dot(gf, cholesky(Symmetric(JtJ[f, f])) \ gf) / 4
end

"""
    polish(problem::AbstractFitProblem, result::FitResult; tol::Real = 1e-6,
           maxsteps::Integer = 10) -> (; result::FitResult, decreases::Vector{Float64})

From `result.theta`, repeats up to `maxsteps` times (D43): recompute the free set, `JtJ` and
`g = gradient!`'s `∇_y 𝒪`, record `gn_decrease` in `decreases`, stop if it is `<= tol`; otherwise
take the Gauss-Newton step `s[f] = -(JtJ[f,f] \\ g[f]) / 2` (zero elsewhere) with backtracking
(`alpha = 1`, halved while `𝒪` increases, down to `1/1024`). If no `alpha` lowers `𝒪`, the loop
stops (with an `@warn`, since a caller must not silently read `decreases[end]` as a step actually
taken -- not part of D43's own signature, Level 3's choice under rule 11 zero).

The returned `result` is `_build_result` at the polished theta: the original solve's `status`,
`converged`, `iterations`, `f_calls`, `g_calls`, `termination` and `optimizer` are **unchanged**;
`elapsed` includes the polish; the trajectory gains one row per **accepted** polish step, numbered
after the solve's last. No new status symbol.

Spec D43.
"""
function polish(problem::AbstractFitProblem, result::FitResult; tol::Real=1e-6, maxsteps::Integer=10)
    t0 = time()
    pspec = problem.pspec
    theta = copy(result.theta)
    trajectory = copy(result.trajectory)
    decreases = Float64[]

    r = _full_residuals(problem, theta)
    Ocur = sum(abs2, r)
    last_iter = isempty(trajectory) ? -1.0 : trajectory[end].iteration

    for step in 1:maxsteps
        f = _free_entries(pspec, theta)
        JtJ, _ = normal_matrices(problem, theta)
        g = zeros(Float64, pspec.ntheta)
        gradient!(g, problem, theta)
        gf = g[f]
        Jff = cholesky(Symmetric(JtJ[f, f]))
        dec = dot(gf, Jff \ gf) / 4
        push!(decreases, dec)
        dec <= tol && break

        s = zeros(Float64, pspec.ntheta)
        s[f] = -(Jff \ gf) / 2

        alpha = 1.0
        accepted = false
        theta_new = theta
        r_new = r
        Onew = Ocur
        while alpha >= 1 / 1024
            theta_new = theta .+ alpha .* s
            r_new = _full_residuals(problem, theta_new)
            Onew = sum(abs2, r_new)
            if Onew <= Ocur
                accepted = true
                break
            end
            alpha /= 2
        end

        if !accepted
            @warn "polish: backtracking found no step that lowers the objective" step gn_decrease = dec objective = Ocur
            break
        end

        theta = theta_new
        r = r_new
        Ocur = Onew
        last_iter += 1
        push!(trajectory, (iteration=last_iter, objective=Onew,
            chi2=_chi2_from_r(problem, r), R=_R_value(problem, theta)))
    end

    elapsed = result.elapsed + (time() - t0)
    polished = _build_result(problem, theta, Ocur, trajectory, result.converged, result.status,
        result.iterations, result.f_calls, result.g_calls, elapsed, result.optimizer;
        termination=result.termination)

    return (; result=polished, decreases=decreases)
end

# ------------------------------------------------------------------------------------------
# D45: the Laplace covariance
# ------------------------------------------------------------------------------------------

"""
    LaplaceResult

The outcome of `laplace` (D45): `theta` (the point it was computed at), `decrease` (the
Gauss-Newton predicted decrease there, `<= max_decrease` by construction), `free` (the `BitVector`
`_free_entries` returned), `cov_y_post`/`cov_y_samp` (the `ntheta x ntheta` posterior and sampling
covariances in `y`, `NaN` on every row/column of a non-free entry), and, per fitted variable
(`vary == true` only), `sigma_post`/`sigma_samp` (in the variable's own unit, the same shape as
`FitResult.values[name]`) and `near_bound` (`.!free` in the same layout).

Spec D45.
"""
struct LaplaceResult
    theta::Vector{Float64}
    decrease::Float64
    free::BitVector
    cov_y_post::Matrix{Float64}
    cov_y_samp::Matrix{Float64}
    sigma_post::Dict{String,Vector{Float64}}
    sigma_samp::Dict{String,Vector{Float64}}
    near_bound::Dict{String,BitVector}
end

"""
    laplace(problem::AbstractFitProblem, theta::AbstractVector; max_decrease::Real = 1e-2) -> LaplaceResult

The Laplace covariance at the minimizer `theta` (D45, OQ-24), from the Gauss-Newton Hessian
`JᵀJ`:

    Cov_y^post = (JᵀJ)⁻¹
    Cov_y^samp = (JᵀJ)⁻¹ J_dᵀJ_d (JᵀJ)⁻¹
    sigma_x    = |dx/dy| * sqrt(diag(Cov_x))

**Refuses to run at a non-minimum** (Pitfall 34): if `gn_decrease(problem, theta) > max_decrease`,
throws naming the decrease found. `max_decrease = 1e-2` is OQ-36's "a hundredth of a sigma"; the
remedy is `polish`, never a looser threshold. Near-bound entries are excluded from the solve
(`_free_entries`) -- their row/column of both covariances is `NaN`, and their `sigma_*` is `NaN`
with `near_bound[name]` set (Pitfall 33).

**The sampling form is the one comparable to a bootstrap or the per-slice scatter, not the
posterior form** (Pitfall 35): with no regularizer `JdtJd == JtJ` bitwise and the two agree (UN2);
with one they differ (UN4).

Spec D45.
"""
function laplace(problem::AbstractFitProblem, theta::AbstractVector; max_decrease::Real=1e-2)
    pspec = problem.pspec
    d = gn_decrease(problem, theta)
    d > max_decrease && throw(ArgumentError(
        "laplace: theta is not at a minimum (Gauss-Newton predicted decrease = $d > $max_decrease); " *
        "run polish first"))

    JtJ, JdtJd = normal_matrices(problem, theta)
    f = _free_entries(pspec, theta)

    cov_y_post = fill(NaN, pspec.ntheta, pspec.ntheta)
    cov_y_samp = fill(NaN, pspec.ntheta, pspec.ntheta)
    C = inv(cholesky(Symmetric(JtJ[f, f])))
    cov_y_post[f, f] .= C
    cov_y_samp[f, f] .= C * JdtJd[f, f] * C

    entry_var, entry_bin = _theta_layout(pspec)
    nt = nbins(pspec.trace)

    sigma_post = Dict{String,Vector{Float64}}()
    sigma_samp = Dict{String,Vector{Float64}}()
    near_bound_out = Dict{String,BitVector}()
    for v in pspec.variables
        v.vary || continue
        n = v.time_varying ? nt : 1
        sigma_post[v.name] = fill(NaN, n)
        sigma_samp[v.name] = fill(NaN, n)
        near_bound_out[v.name] = falses(n)
    end

    for e in 1:pspec.ntheta
        v = pspec.variables[entry_var[e]]
        pos = v.time_varying ? entry_bin[e] : 1
        near_bound_out[v.name][pos] = !f[e]
        f[e] || continue
        x = _from_unconstrained(theta[e], v.lower, v.upper)
        dxdy = abs(_dx_dy(x, v.lower, v.upper))
        sigma_post[v.name][pos] = dxdy * sqrt(cov_y_post[e, e])
        sigma_samp[v.name][pos] = dxdy * sqrt(cov_y_samp[e, e])
    end

    return LaplaceResult(Vector{Float64}(theta), d, f, cov_y_post, cov_y_samp, sigma_post,
        sigma_samp, near_bound_out)
end

function Base.show(io::IO, lap::LaplaceResult)
    print(io, "LaplaceResult(decrease = ", lap.decrease, ", ", count(lap.free), "/",
        length(lap.free), " free)")
end

function Base.show(io::IO, ::MIME"text/plain", lap::LaplaceResult)
    println(io, "LaplaceResult")
    @printf(io, "  decrease: %.6g\n", lap.decrease)
    println(io, "  free: ", count(lap.free), " / ", length(lap.free))
    for name in sort(collect(keys(lap.sigma_post)))
        nb = count(lap.near_bound[name])
        @printf(io, "  %s: sigma_samp = %s (near_bound: %d)\n", name, lap.sigma_samp[name], nb)
    end
end

# ------------------------------------------------------------------------------------------
# D46: the calibration scan
# ------------------------------------------------------------------------------------------

"""
    CalibrationScanResult

The outcome of `calibration_scan` (D46): `shifts_nm` (sorted ascending), `channel`,
`base_offset_nm` (the scanned channel's own offset before any shift, in nm); `chi2` (the weighted
total), `objective`, `decrease` and `status`, one entry per shift; `delta::Dict{String,Matrix}`,
per variable of `problem.pspec.variables`, `size = (length(values[name]), nshifts)`, the polished
values minus the delta-zero values, in the variable's own unit; `s_sys::Dict{String,Vector}`,
`maximum(abs, delta; dims = 2)` flattened; `results`, every point's own `FitResult`.

Spec D46.
"""
struct CalibrationScanResult
    shifts_nm::Vector{Float64}
    channel::Int
    base_offset_nm::Float64
    chi2::Vector{Float64}
    objective::Vector{Float64}
    decrease::Vector{Float64}
    status::Vector{Symbol}
    delta::Dict{String,Matrix{Float64}}
    s_sys::Dict{String,Vector{Float64}}
    results::Vector{FitResult}
end

"""
    calibration_scan(problem::AbstractFitProblem, theta_star::AbstractVector,
                     shifts_nm::AbstractVector{<:Real}; channel::Integer = 1,
                     optimizer::AbstractOptimizer = LevenbergMarquardt(),
                     tol::Real = 1e-6) -> CalibrationScanResult

The systematic-uncertainty report (D46, OQ-24): the regularized fit repeated with `channel`'s own
`WavelengthCalibration.offset` fixed at `theta_star`'s own offset plus each of `shifts_nm` (nm,
sorted ascending, must include `0.0`), warm-started from `theta_star`, every point polished to
`tol`.

`theta_star` must itself be a polished minimum (`gn_decrease(problem, theta_star) <= tol`), else a
loud error naming the decrease found and saying to polish first (Pitfall 34) -- `calibration_scan`
refuses to run at a non-minimum exactly as `laplace` does.

**The target is `channel`'s one `WavelengthCalibration`** (`_chain_class`, never `isa`,
`conventions.md` Section 4): a loud error names the channel if it has none, or if any binding of
`problem.pspec` fixes that effect's `:offset` field in that channel (Pitfall 36 -- the scan fixes
the offset, it never fits it).

**Per shift `delta` (nm)**: that channel's experiment is rebuilt with the offset replaced through
`reconstruct` and `Base.setindex` on the effects tuple, **the same trace object** (Pitfall 7); that
channel's `FitProblem` is rebuilt with the keyword constructor (same data, `channel`, `name`); for
a `JointFitProblem`, a new `JointFitProblem` with the same weights and regularizer. The point is
then `polish(new_problem, solve(new_problem, optimizer; start = theta_star); tol = tol)`.

**The `delta = 0` point returns `theta_star` bitwise** (CS1): the rebuild at `delta = 0` is
numerically the identity, so `solve` either meets `g_abstol` immediately or rejects its first step
and stops by D44, and the subsequent `polish` (already at `gn_decrease <= tol`) takes no step.

Spec D46.
"""
function calibration_scan(problem::AbstractFitProblem, theta_star::AbstractVector,
    shifts_nm::AbstractVector{<:Real}; channel::Integer=1,
    optimizer::AbstractOptimizer=LevenbergMarquardt(), tol::Real=1e-6)

    d0 = gn_decrease(problem, theta_star)
    d0 <= tol || throw(ArgumentError(
        "calibration_scan: theta_star is not a polished minimum (Gauss-Newton predicted decrease " *
        "= $d0 > $tol); run polish first"))

    channels = problem isa JointFitProblem ? problem.channels : (problem,)
    (1 <= channel <= length(channels)) || throw(ArgumentError(
        "calibration_scan: channel $channel is out of range 1:$(length(channels))"))
    ch = channels[channel]
    expt = ch.experiment

    calib_idx = findall(e -> _chain_class(e) === :WavelengthCalibration, expt.effects)
    isempty(calib_idx) && throw(ArgumentError(
        "calibration_scan: channel $channel (\"$(ch.name)\") has no WavelengthCalibration effect"))
    wc_idx = only(calib_idx)

    pspec = problem.pspec
    for b in pspec.bindings
        (b.kind === :effect && b.channel == channel && b.effect == wc_idx && b.field === :offset) &&
            throw(ArgumentError(
                "calibration_scan: channel $channel's WavelengthCalibration offset is bound to " *
                "\"$(b.name)\"; the scan fixes the offset and cannot also fit it"))
    end

    shifts_sorted = sort(collect(Float64, shifts_nm))
    any(iszero, shifts_sorted) || throw(ArgumentError(
        "calibration_scan: shifts_nm must include 0.0; got $shifts_sorted"))

    wc = expt.effects[wc_idx]
    base_offset_nm = wc.offset * 1e9
    nshifts = length(shifts_sorted)

    results = Vector{FitResult}(undef, nshifts)
    decreases = Vector{Float64}(undef, nshifts)

    for (k, delta) in enumerate(shifts_sorted)
        new_wc = reconstruct(wc, (offset=wc.offset + delta * 1e-9, dispersion_scale=wc.dispersion_scale))
        new_effects = Base.setindex(expt.effects, new_wc, wc_idx)
        new_expt = Experiment(; trace=expt.trace, probe=expt.probe, collection=expt.collection,
            lambda_m=expt.lambda, t_s=expt.t, effects=new_effects, prefactor=expt.prefactor,
            gain=expt.gain)
        new_ch = FitProblem(new_expt, ch.pspec, ch.data; regularizer=ch.regularizer,
            scheduler=ch.scheduler, channel=ch.channel, name=ch.name)
        new_problem = if problem isa JointFitProblem
            new_channels = ntuple(length(channels)) do c
                c == channel ? new_ch : channels[c]
            end
            JointFitProblem(new_channels; weights=problem.weights, regularizer=problem.regularizer)
        else
            new_ch
        end

        fit0 = solve(new_problem, optimizer; start=theta_star)
        polished = polish(new_problem, fit0; tol=tol)
        results[k] = polished.result
        decreases[k] = polished.decreases[end]
    end

    zero_idx = findfirst(iszero, shifts_sorted)
    chi2v = [r.chi2 for r in results]
    objv = [r.objective for r in results]
    statusv = [r.status for r in results]

    delta = Dict{String,Matrix{Float64}}()
    for v in pspec.variables
        base = results[zero_idx].values[v.name]
        m = Matrix{Float64}(undef, length(base), nshifts)
        for k in 1:nshifts
            m[:, k] = results[k].values[v.name] .- base
        end
        delta[v.name] = m
    end
    s_sys = Dict{String,Vector{Float64}}(name => vec(maximum(abs, m; dims=2)) for (name, m) in delta)

    return CalibrationScanResult(shifts_sorted, Int(channel), base_offset_nm, chi2v, objv,
        decreases, statusv, delta, s_sys, results)
end

function Base.show(io::IO, scan::CalibrationScanResult)
    print(io, "CalibrationScanResult(channel = ", scan.channel, ", ", length(scan.shifts_nm),
        " shifts around ", scan.base_offset_nm, " nm)")
end

function Base.show(io::IO, ::MIME"text/plain", scan::CalibrationScanResult)
    println(io, "CalibrationScanResult")
    println(io, "  channel: ", scan.channel, " (base offset ", scan.base_offset_nm, " nm)")
    for (k, s) in enumerate(scan.shifts_nm)
        @printf(io, "  shift %+.4g nm: chi2 = %.6g, status = :%s, decrease = %.3g\n", s,
            scan.chi2[k], scan.status[k], scan.decrease[k])
    end
end

# ------------------------------------------------------------------------------------------
# D51: the per-slice scatter -- _detrended_spread, and ScatterResult (slice_scatter itself is
# in config/scatter.jl, which needs Deck and build_problem's three stages, D50)
# ------------------------------------------------------------------------------------------

"""
    _detrended_spread(t::AbstractVector, y::AbstractVector) -> Float64

The detrended sample spread of per-slice estimates `y` at times `t` (D51 step 6, Section 10's
"the per-slice scatter"): a least-squares line is removed first, on the **centred** axis
`tau = t .- mean(t)` so the closed form is exact and well-conditioned,

    ybar = mean(y);  c = sum(tau .* y) / sum(tau.^2);  e = y .- ybar .- c .* tau
    _detrended_spread = sqrt(sum(e.^2) / (m - 2))      (m = length(t))

`m - 2` degrees of freedom, one each for the line's slope and intercept; `NaN` for `m <= 2`, where
none are left. **Never a plain standard deviation** (which removes no trend and divides by
`m - 1`) **and never divided by `sqrt(m)`** (Pitfall 40): this is the scatter of a single-slice
estimate, the point of the report, not a standard error of the mean.

Spec D51 step 6.
"""
function _detrended_spread(t::AbstractVector, y::AbstractVector)
    m = length(t)
    length(y) == m || throw(ArgumentError(
        "_detrended_spread: t and y must have the same length; got $m and $(length(y))"))
    m <= 2 && return NaN
    tbar = sum(t) / m
    tau = t .- tbar
    ybar = sum(y) / m
    c = sum(tau .* y) / sum(abs2, tau)
    e = y .- ybar .- c .* tau
    return sqrt(sum(abs2, e) / (m - 2))
end

"""
    _quantile_sorted(x_sorted::AbstractVector, q::Real) -> Float64

The `q` quantile of the values `x_sorted`, already sorted ascending (Phase 4.N spec D6,
[OQ-64](../../dev/open_questions.md#oq-64) item 3): Hyndman and Fan's type 7, the definition
`Statistics.quantile` and numpy share. With `m = length(x_sorted)`,

    h = (m - 1) q + 1;   quantile = x[⌊h⌋] + (h - ⌊h⌋) (x[⌊h⌋ + 1] - x[⌊h⌋]),   x[m] when ⌊h⌋ = m

defined for every `m >= 1` (one value gives itself at every `q`). **Evaluated in `Statistics.quantile`'s
own order** (Statistics 1.11.5, the test environment's; the user's choice of 2026-10-15, Phase 4.N
summary U1), so that it equals that function bitwise: `h = fma(m, q, 1 - q)`, `j = clamp(⌊h⌋, 1, m - 1)`,
`γ = clamp(h - j, 0, 1)`, and `(1 - γ) x[j] + γ x[j + 1]`, unless the two values are finite and `≈`, when
it is `x[j] + γ (x[j + 1] - x[j])`. The same quantile as the formula above, rounded as that package
rounds it. A private helper, so that the package takes no `Statistics` dependency
([OQ-1](../../dev/open_questions.md#oq-1)); the caller sorts once for its three quantiles.

Phase 4.N spec D6.
"""
function _quantile_sorted(x_sorted::AbstractVector, q::Real)
    m = length(x_sorted)
    m >= 1 || throw(ArgumentError("_quantile_sorted: no value to take a quantile of"))
    p = Float64(q)
    h = fma(m, p, 1.0 - p)
    j = clamp(trunc(Int, h), 1, m - 1)
    γ = clamp(h - j, 0.0, 1.0)
    a = m == 1 ? Float64(x_sorted[1]) : Float64(x_sorted[j])
    b = m == 1 ? Float64(x_sorted[1]) : Float64(x_sorted[j+1])
    if isfinite(a) && isfinite(b) && a ≈ b
        return a + γ * (b - a)
    end
    return (1.0 - γ) * a + γ * b
end

"""
    ScatterResult

The outcome of `slice_scatter` (D51, `config/scatter.jl`): the per-slice statistical uncertainty
report, a fine-grid **joint** fit at single-slice resolution.

- `theta`: `copy(result.theta)`, the θ\\* `slice_scatter` was computed at (`write_result`'s own
  check that it was not computed at a different θ than the result it is written beside, D52).
- `m`: fine windows per coarse bin, one entry per `[[fit.regions]]` region (Pitfall 39).
- `t_fine`: the fine axis (s), D39's pooled mean, shared by every channel, length `N_fine`.
- `coarse_bin`: fine window `j`'s own coarse bin, length `N_fine`.
- `counts`: `N_fine x nchannels`, each channel's own slice count in that fine window.
- `status`, `decrease`: each fine window's own fit status and final `gn_decrease`, length `N_fine`.
- Per **free** variable (`vary && time_varying`, Pitfall 42; a scalar or a frozen variable has no
  entry here):
  - `values`: length `N_fine`, in the variable's own unit; `NaN` in a window whose coarse bin lies
    outside the variable's interval, where it was frozen at its `init` (Phase 4.N spec D3);
  - `near_bound`: length `N_fine`, `false` where `values` is `NaN`;
  - `s_stat`: length = the number of coarse bins, `_detrended_spread` over each bin's own fine
    windows, **never divided by `sqrt(m)`** (Pitfall 40); `NaN` for a bin outside the interval;
  - `n_near_bound`: length = the number of coarse bins, the count of `near_bound` over the bin's
    own fine windows; `0` for a bin outside the interval.
- Per channel, in channel order (Phase 4.M spec D13, [OQ-63](../../dev/open_questions.md#oq-63) item
  5), each `(N_λ, N_fine)`, column `j` fine window `j` as its one-bin fit saw it: `data`, `sigma` and
  `mask`, the window's averaged image, σ and used-pixel mask; `model`, the model at the window's
  solution.
- `chi2`: length `N_fine`, each window's χ² at its solution (the weighted total over channels).
- `held` (Phase 4.N spec D4, [OQ-64](../../dev/open_questions.md#oq-64) item 2): per variable the
  deck's `[fit.hold]` names, length `N_fine`, the value each window held it at; `NaN` in a window whose
  coarse bin lies outside the variable's interval. Empty without a `[fit.hold]` table.
- `p16`, `median`, `p84` (Phase 4.N spec D6, OQ-64 item 3): per free variable, length = the number of
  coarse bins, the 16th, 50th and 84th percentiles of the bin's own fine windows' `values`
  (`_quantile_sorted`, type 7), over every window whatever its `status`; `NaN` for a bin outside the
  variable's interval.

The eleven-argument positional constructor (every field but the last nine) and the sixteen-argument
one (every field but the last four) stay, as outer methods filling the later fields empty, so a
`ScatterResult` built by hand keeps compiling; `write_result` writes none of the nine (the result
file's `scatter/*` is unchanged), `write_readable` writes all of them.

Spec D51; Phase 4.M spec D13; Phase 4.N spec D4, D6.
"""
struct ScatterResult
    theta::Vector{Float64}
    m::Vector{Int}
    t_fine::Vector{Float64}
    coarse_bin::Vector{Int}
    counts::Matrix{Int}
    status::Vector{Symbol}
    decrease::Vector{Float64}
    values::Dict{String,Vector{Float64}}
    near_bound::Dict{String,BitVector}
    s_stat::Dict{String,Vector{Float64}}
    n_near_bound::Dict{String,Vector{Int}}
    data::Vector{Matrix{Float64}}
    sigma::Vector{Matrix{Float64}}
    mask::Vector{Matrix{Bool}}
    model::Vector{Matrix{Float64}}
    chi2::Vector{Float64}
    held::Dict{String,Vector{Float64}}
    p16::Dict{String,Vector{Float64}}
    median::Dict{String,Vector{Float64}}
    p84::Dict{String,Vector{Float64}}
end

ScatterResult(theta, m, t_fine, coarse_bin, counts, status, decrease, values, near_bound, s_stat,
    n_near_bound) = ScatterResult(theta, m, t_fine, coarse_bin, counts, status, decrease, values,
    near_bound, s_stat, n_near_bound, Matrix{Float64}[], Matrix{Float64}[], Matrix{Bool}[],
    Matrix{Float64}[], Float64[])

ScatterResult(theta, m, t_fine, coarse_bin, counts, status, decrease, values, near_bound, s_stat,
    n_near_bound, data, sigma, mask, model, chi2) = ScatterResult(theta, m, t_fine, coarse_bin, counts,
    status, decrease, values, near_bound, s_stat, n_near_bound, data, sigma, mask, model, chi2,
    Dict{String,Vector{Float64}}(), Dict{String,Vector{Float64}}(), Dict{String,Vector{Float64}}(),
    Dict{String,Vector{Float64}}())

"""
    _median_finite(v::AbstractVector) -> Float64

The median of `v`'s finite entries, `NaN` if none (Level 3's helper for `ScatterResult`'s
`text/plain` summary only, flagged under rule 11 zero: `Statistics` is not a dependency of this
package, and a print-only median does not warrant adding one).
"""
function _median_finite(v::AbstractVector)
    finite = filter(isfinite, v)
    isempty(finite) && return NaN
    s = sort(finite)
    n = length(s)
    return isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end

function Base.show(io::IO, sc::ScatterResult)
    print(io, "ScatterResult(", length(sc.t_fine), " fine windows over ", length(sc.m),
        " region(s), ", length(sc.values), " free variable(s))")
end

function Base.show(io::IO, ::MIME"text/plain", sc::ScatterResult)
    println(io, "ScatterResult")
    println(io, "  m (fine windows per coarse bin, per region): ", sc.m)
    println(io, "  fine windows: ", length(sc.t_fine))
    counts_by_status = Dict{Symbol,Int}()
    for st in sc.status
        counts_by_status[st] = get(counts_by_status, st, 0) + 1
    end
    println(io, "  statuses: ", counts_by_status)
    for name in sort(collect(keys(sc.s_stat)))
        @printf(io, "  %s: median s_stat = %.6g\n", name, _median_finite(sc.s_stat[name]))
    end
end
