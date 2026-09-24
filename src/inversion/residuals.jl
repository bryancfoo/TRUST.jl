# FitProblem, residuals, residuals! and chi2 -- the objective's data half (Step 3.2a, chunk 5 of
# Phase 3). This is the one place the pixel order and the sigma divide are defined: every
# Jacobian row, jvp! and jtvp! from Step 3.2b onward reuses FitProblem.pixel_index rather than
# rederiving the order. Spec D15, D16.
#
# JointFitProblem (Step 4.A.3a, chunk 6 of Phase 4.A, D31, D34): the multi-channel problem, one
# FitProblem per channel over one shared ParameterSpec. A single-channel FitProblem is channel 1,
# name "main", and every Phase 2, 3, 3.C and 4.A row passes through it unchanged (Pitfall 22).

"""
    AbstractFitProblem

The fit-problem interface (`conventions.md` Section 4): `FitProblem` (one streak) and
`JointFitProblem` (several channels sharing one trace and one `ParameterSpec`) both subtype it,
and every optimizer, `weight_scan` and the products below (`residuals`, `chi2`, `jacobian`,
`jvp!`, `jtvp!`, `gradient!`, `objective`) dispatch on it rather than on the concrete type.

Spec D31.
"""
abstract type AbstractFitProblem end

"""
    FitProblem{FT<:Real,E<:Experiment,P<:ParameterSpec,DA<:StreakData{FT},R,S,M}

The data half of a fit: the working `Experiment` (built from `pspec`'s own, already-rebuilt
trace -- `unflatten`'s precondition, D12), the `ParameterSpec`, the measured `StreakData`, an
optional `Regularizer` (`nothing` before Step 3.3), the materialized pixel mask, the pixel count,
the frozen `sigma`, the column-major linear indices of the mask-true pixels (`pixel_index`,
computed once here and reused unchanged by every Jacobian row and every `jvp!`/`jtvp!` from Step
3.2b onward), the `scheduler` passed through to `measure`, and (Step 4.A.3a, D31) `channel`, the
1-based channel this problem occupies in a `JointFitProblem` (`1` alone for a single-channel fit),
and `name`, the channel's name (`"main"` alone for a single-channel fit).

**`sigma` is user-supplied and frozen** ([OQ-14](../../dev/open_questions.md#oq-14), the user
2026-09-12): the constructor requires `data.sigma !== nothing` and nothing downstream
recomputes it, so the chi2 weights do not move when a fitted `Sensitivity.scale` changes the
model's own prediction (RS6). **`mask[l,j] == true` means the pixel is used** -- the opposite of
the commoner convention (`StreakData`'s own docstring; `conventions.md` Section 4).

Built with `FitProblem(experiment, pspec, data; regularizer = nothing, scheduler = :auto,
channel = 1, name = "main", allow_empty_mask = false)`. The constructor checks, in this order:
`data.sigma !== nothing`; `experiment.trace === pspec.trace` (A17, 2026-09-13 -- a spec built
over a different trace object leaves chi2 insensitive to theta silently); `name` matches
`^[a-z][a-z0-9_]*\$` (D31: it becomes a result-file group at chunk 7); the materialized mask is
not empty, unless `allow_empty_mask = true` (D31, D35: the one way a zero-pixel channel exists,
for the masked-channel test); the axes of `data` agree with `axes(experiment)` to `rtol = 1e-12`.

Spec D15, D31.
"""
struct FitProblem{FT<:Real,E<:Experiment,P<:ParameterSpec,DA<:StreakData{FT},R,S,M} <: AbstractFitProblem
    experiment::E
    pspec::P
    data::DA
    regularizer::R
    mask::M
    npixels::Int
    sigma::S
    pixel_index::Vector{Int}
    scheduler::Symbol
    channel::Int
    name::String
end

"""
    _check_axes_match(data::StreakData, expt::Experiment)

`data.lambda`/`data.t` against `axes(expt) = (lambda, t)`, elementwise to `rtol = 1e-12`. A loud
error names the axis and the first disagreeing bin (D15, D39) -- checked wavelength axis first,
then time.
"""
function _check_axes_match(data::StreakData, expt::Experiment)
    elambda, et = axes(expt)
    length(data.lambda) == length(elambda) || throw(ArgumentError(
        "FitProblem: data has $(length(data.lambda)) wavelength bins but the experiment has $(length(elambda))"))
    for l in eachindex(elambda)
        isapprox(data.lambda[l], elambda[l]; rtol=1e-12) || throw(ArgumentError(
            "FitProblem: data disagrees with the experiment's wavelength axis at bin $l"))
    end
    length(data.t) == length(et) || throw(ArgumentError(
        "FitProblem: data has $(length(data.t)) time bins but the experiment has $(length(et))"))
    for j in eachindex(et)
        isapprox(data.t[j], et[j]; rtol=1e-12) || throw(ArgumentError(
            "FitProblem: data disagrees with the experiment's time axis at bin $j"))
    end
    return nothing
end

function FitProblem(experiment::Experiment, pspec::ParameterSpec, data::StreakData;
    regularizer=nothing, scheduler::Symbol=:auto, channel::Int=1, name::AbstractString="main",
    allow_empty_mask::Bool=false)
    data.sigma === nothing && throw(ArgumentError(
        "FitProblem: data has no sigma; make one with sigma(model, streak)"))

    experiment.trace === pspec.trace || throw(ArgumentError(
        "FitProblem: the experiment's trace is not this ParameterSpec's own rebuilt trace " *
        "(unflatten writes into pspec.trace, so experiment.trace must be === pspec.trace; " *
        "build the experiment from pspec.trace once, e.g. Experiment(; trace = pspec.trace, ...))"))

    occursin(r"^[a-z][a-z0-9_]*$", name) || throw(ArgumentError(
        "FitProblem: name must match ^[a-z][a-z0-9_]*\$ (it becomes a result-file group); got \"$name\""))

    mask = data.mask === nothing ? trues(size(data.image)) : data.mask
    npixels = count(mask)
    (npixels == 0 && !allow_empty_mask) && throw(ArgumentError("FitProblem: the mask excludes every pixel"))

    _check_axes_match(data, experiment)

    pixel_index = findall(vec(mask))
    return FitProblem(experiment, pspec, data, regularizer, mask, npixels, data.sigma,
        pixel_index, scheduler, channel, String(name))
end

"""
    residuals(problem::FitProblem, theta::AbstractVector) -> Vector

`r_k = (M[l,j] - D[l,j]) / sigma[l,j]` for the `k`-th mask-true pixel, in column-major order
(`l` fastest, then `j`) -- `M = measure(unflatten(problem.experiment, problem.pspec, theta))`,
`D = problem.data.image`. foo2026 Eq. 7's residual, before the sum of squares.

Generic in `eltype(theta)` (`conventions.md` 5.1): with a dual `theta` this returns a dual
vector, which is what the brute-force Jacobian oracle of Step 3.2b differentiates.

Spec D16.
"""
function residuals(problem::FitProblem, theta::AbstractVector)
    r = Vector{eltype(theta)}(undef, problem.npixels)
    residuals!(r, problem, theta)
    return r
end

"""
    residuals!(r::AbstractVector, problem::FitProblem, theta::AbstractVector) -> r

In-place `residuals`: fills `r[k] = (M[i] - D[i]) / sigma[i]` for `i = problem.pixel_index[k]`,
`M = measure(unflatten(problem.experiment, problem.pspec, theta); scheduler = problem.scheduler)`.
`problem.pixel_index` is the one definition of the pixel order (D16); every Jacobian row and
every `jvp!`/`jtvp!` from Step 3.2b onward reuses it unchanged.

Spec D16.
"""
function residuals!(r::AbstractVector, problem::FitProblem, theta::AbstractVector)
    expt = unflatten(problem.experiment, problem.pspec, theta; channel=problem.channel)
    M = measure(expt; scheduler=problem.scheduler)
    D = problem.data.image
    sig = problem.sigma
    idx = problem.pixel_index
    @inbounds for k in eachindex(idx)
        i = idx[k]
        r[k] = (M[i] - D[i]) / sig[i]
    end
    return r
end

"""
    chi2(problem::FitProblem, theta::AbstractVector) -> Real

`sum(abs2, residuals(problem, theta))` -- foo2026 Eq. 7. `abs2`-summed, not `dot(r, r)`: the two
differ in reassociation.

Spec D16.
"""
chi2(problem::FitProblem, theta::AbstractVector) = sum(abs2, residuals(problem, theta))

"""
    _full_residuals(problem::FitProblem, theta::AbstractVector) -> Vector

`r_full = [r; rho_1; rho_2; ...]` (D25): the data residuals of `residuals`, followed by every
regularizer term's rows, `rho_row = sqrt(Lambda * w_term) * (op N(u - ref))_row`, when
`problem.regularizer !== nothing` (Step 3.3); `residuals(problem, theta)` unchanged when it is
`nothing` (every step before Step 3.3, and `objective`'s `problem.regularizer === nothing` case,
Step 3.4). Generic in `eltype(theta)`, so this is what RG12/RG13/RG19's `ForwardDiff.jacobian`
oracle differentiates for the regularizer rows' closed-form Jacobian (`conventions.md` 8.2), and
what OB2 sums the squares of against `objective`.
"""
function _full_residuals(problem::FitProblem, theta::AbstractVector)
    r = residuals(problem, theta)
    problem.regularizer === nothing && return r
    rr = _regularizer_residual_rows(problem.regularizer, problem.pspec, theta)
    return vcat(r, rr)
end

# ------------------------------------------------------------------------------------------
# JointFitProblem (Step 4.A.3a, chunk 6 of Phase 4.A, D31, D34)
# ------------------------------------------------------------------------------------------

"""
    JointFitProblem{P<:ParameterSpec,CH<:Tuple,R}

The multi-channel problem: one `FitProblem` per channel (`channels`, in channel order, each with
its own `channel`/`name`), sharing **one** `ParameterSpec` (`pspec`) and **one** `Regularizer`
(`regularizer`, `nothing` or a `Regularizer`; a channel's own must be `nothing`, D31). `weights[c]`
is `w_c` (OQ-35): a channel's residual rows are scaled by `sqrt(w_c)` unless `w_c == 1.0`.
`npixels` is `sum(ch.npixels for ch in channels)`; `row_ranges[c]` is channel `c`'s rows in the
data half of `r_full`, `(offset_c+1):(offset_c+n_c)` with `offset_c = sum(n_1 .. n_{c-1})` (an
empty range is legal, D35's masked channel).

Built with `JointFitProblem(channels; weights = ones(length(channels)), regularizer = nothing)`,
`channels` a `Tuple` of `FitProblem`s. The constructor checks, in this order, each a loud error
naming what failed: at least one channel; `length(weights) == length(channels)` and every weight
finite and `> 0`; every channel's `pspec === channels[1].pspec`; `channels[c].channel == c`; every
channel's own `regularizer === nothing`; the channels' `name`s distinct; `npixels > 0`.

**A single-channel problem must not move a bit** (Pitfall 22): `FitProblem`'s own methods are
never edited to accommodate this type, and every product below is exactly the single-channel code
run once per live channel (D34).

Spec D31, D34.
"""
struct JointFitProblem{P<:ParameterSpec,CH<:Tuple,R} <: AbstractFitProblem
    channels::CH
    weights::Vector{Float64}
    pspec::P
    regularizer::R
    npixels::Int
    row_ranges::Vector{UnitRange{Int}}
end

"""
    JointFitProblem(channels::Tuple{Vararg{FitProblem}}; weights = ones(length(channels)),
                    regularizer = nothing) -> JointFitProblem

Spec D31.
"""
function JointFitProblem(channels::Tuple{Vararg{FitProblem}};
    weights::AbstractVector{<:Real}=ones(length(channels)), regularizer=nothing)
    isempty(channels) && throw(ArgumentError("JointFitProblem: at least one channel is required"))

    ws = collect(Float64, weights)
    length(ws) == length(channels) || throw(ArgumentError(
        "JointFitProblem: got $(length(ws)) weights for $(length(channels)) channels"))
    all(w -> isfinite(w) && w > 0, ws) || throw(ArgumentError(
        "JointFitProblem: every weight must be finite and positive; got $ws"))

    pspec = channels[1].pspec
    for c in 2:length(channels)
        channels[c].pspec === pspec || throw(ArgumentError(
            "JointFitProblem: channel $c's pspec is not === channel 1's; every channel must " *
            "share one ParameterSpec"))
    end

    for (c, ch) in enumerate(channels)
        ch.channel == c || throw(ArgumentError(
            "JointFitProblem: channel $c's FitProblem has channel = $(ch.channel); expected $c"))
    end

    for (c, ch) in enumerate(channels)
        ch.regularizer === nothing || throw(ArgumentError(
            "JointFitProblem: channel $c (\"$(ch.name)\") carries its own regularizer; the " *
            "joint problem's one regularizer is applied once for every channel, never per channel"))
    end

    names = [ch.name for ch in channels]
    length(names) == length(unique(names)) || throw(ArgumentError(
        "JointFitProblem: channel names must be distinct; got $names"))

    npixels = sum(ch.npixels for ch in channels)
    npixels > 0 || throw(ArgumentError("JointFitProblem: every channel is empty (0 total pixels)"))

    row_ranges = Vector{UnitRange{Int}}(undef, length(channels))
    offset = 0
    for (c, ch) in enumerate(channels)
        row_ranges[c] = (offset+1):(offset+ch.npixels)
        offset += ch.npixels
    end

    return JointFitProblem(channels, ws, pspec, regularizer, npixels, row_ranges)
end

"""
    channel_names(jp::JointFitProblem) -> Vector{String}

`[ch.name for ch in jp.channels]`, in channel order. Not part of D31's own signature; Level 3's
helper, flagged under rule 11 zero.
"""
channel_names(jp::JointFitProblem) = [ch.name for ch in jp.channels]

"""
    residuals!(r::AbstractVector, jp::JointFitProblem, theta::AbstractVector) -> r

For each channel `c` with `n_c > 0` (D34 rule 1: a fully masked channel is skipped, an economy,
never a correctness requirement -- see the module note above), `residuals!` into `r`'s own
`jp.row_ranges[c]` by **exactly** `FitProblem`'s own method, then scaled by `sqrt(w_c)` unless
`w_c == 1.0` (D34 rule 3: `1.0 * x === x`, so no scaling is applied at all when every weight is 1,
which is what keeps D35's masked-channel test bitwise).

Spec D34.
"""
function residuals!(r::AbstractVector, jp::JointFitProblem, theta::AbstractVector)
    for (c, ch) in enumerate(jp.channels)
        ch.npixels == 0 && continue
        rc = view(r, jp.row_ranges[c])
        residuals!(rc, ch, theta)
        w = jp.weights[c]
        w == 1.0 || (rc .*= sqrt(w))
    end
    return r
end

"""
    residuals(jp::JointFitProblem, theta::AbstractVector) -> Vector

Allocates `jp.npixels` and calls `residuals!`.

Spec D34.
"""
function residuals(jp::JointFitProblem, theta::AbstractVector)
    r = Vector{eltype(theta)}(undef, jp.npixels)
    residuals!(r, jp, theta)
    return r
end

"""
    chi2(jp::JointFitProblem, theta::AbstractVector) -> Real

The **weighted** total `sum(abs2, residuals(jp, theta)) == sum(w_c * chi2_c)` -- what the
optimizer minimizes. The unweighted per-channel `chi2_c` is `ChannelSummary.chi2` (D36, Pitfall
26).

Spec D34.
"""
chi2(jp::JointFitProblem, theta::AbstractVector) = sum(abs2, residuals(jp, theta))

"""
    _full_residuals(jp::JointFitProblem, theta::AbstractVector) -> Vector

`residuals(jp, theta)` followed by the regularizer's rows when `jp.regularizer !== nothing`,
computed by **exactly** `FitProblem`'s own `_regularizer_residual_rows(reg, pspec, theta)` call
(D34 rule 4: this is what makes the regularizer's contribution to D35's masked-channel test
bitwise -- the same function, the same `reg`, the same `pspec`, the same `theta`).

Spec D34.
"""
function _full_residuals(jp::JointFitProblem, theta::AbstractVector)
    r = residuals(jp, theta)
    jp.regularizer === nothing && return r
    rr = _regularizer_residual_rows(jp.regularizer, jp.pspec, theta)
    return vcat(r, rr)
end

# --- show (spec D40: "every exported type", FitProblem is not in D40's own worked list) ---

function Base.show(io::IO, jp::JointFitProblem)
    print(io, "JointFitProblem(", length(jp.channels), " channels, $(jp.npixels) pixels, ",
        "ntheta = $(jp.pspec.ntheta))")
end

function Base.show(io::IO, ::MIME"text/plain", jp::JointFitProblem)
    println(io, "JointFitProblem")
    println(io, "  channels: ", length(jp.channels))
    for (c, ch) in enumerate(jp.channels)
        nl, nt = size(ch.data.image)
        @printf(io, "    %s: weight = %.6g, %d x %d, %d pixels used\n", ch.name, jp.weights[c],
            nl, nt, ch.npixels)
    end
    println(io, "  ntheta: ", jp.pspec.ntheta)
    println(io, "  regularizer: ", jp.regularizer === nothing ? "none" : "present")
end

function Base.show(io::IO, problem::FitProblem)
    nl, nt = size(problem.data.image)
    print(io, "FitProblem($nl x $nt, $(problem.npixels) pixels, ntheta = $(problem.pspec.ntheta))")
end

function Base.show(io::IO, ::MIME"text/plain", problem::FitProblem)
    nl, nt = size(problem.data.image)
    println(io, "FitProblem")
    @printf(io, "  streak: %d x %d, %d pixels used (of %d)\n", nl, nt, problem.npixels, nl * nt)
    println(io, "  ntheta: ", problem.pspec.ntheta)
    println(io, "  regularizer: ", problem.regularizer === nothing ? "none" : "present")
    println(io, "  scheduler: ", problem.scheduler)
end
