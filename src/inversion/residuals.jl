# FitProblem, residuals, residuals! and chi2 -- the objective's data half (Step 3.2a, chunk 5 of
# Phase 3). This is the one place the pixel order and the sigma divide are defined: every
# Jacobian row, jvp! and jtvp! from Step 3.2b onward reuses FitProblem.pixel_index rather than
# rederiving the order. Spec D15, D16.

"""
    FitProblem{FT<:Real,E<:Experiment,P<:ParameterSpec,DA<:StreakData{FT},R,S,M}

The data half of a fit: the working `Experiment` (built from `pspec`'s own, already-rebuilt
trace -- `unflatten`'s precondition, D12), the `ParameterSpec`, the measured `StreakData`, an
optional `Regularizer` (`nothing` before Step 3.3), the materialized pixel mask, the pixel count,
the frozen `sigma`, the column-major linear indices of the mask-true pixels (`pixel_index`,
computed once here and reused unchanged by every Jacobian row and every `jvp!`/`jtvp!` from Step
3.2b onward), and the `scheduler` passed through to `measure`.

**`sigma` is user-supplied and frozen** ([OQ-14](../../dev/open_questions.md#oq-14), the user
2026-09-12): the constructor requires `data.sigma !== nothing` and nothing downstream
recomputes it, so the chi2 weights do not move when a fitted `Sensitivity.scale` changes the
model's own prediction (RS6). **`mask[l,j] == true` means the pixel is used** -- the opposite of
the commoner convention (`StreakData`'s own docstring; `conventions.md` Section 4).

Built with `FitProblem(experiment, pspec, data; regularizer = nothing, scheduler = :auto)`. The
constructor checks, in this order: `data.sigma !== nothing`; `experiment.trace === pspec.trace`
(A17, 2026-09-13 -- a spec built over a different trace object leaves chi2 insensitive to theta
silently); the materialized mask is not empty; the axes of `data` agree with `axes(experiment)`
to `rtol = 1e-12`.

Spec D15.
"""
struct FitProblem{FT<:Real,E<:Experiment,P<:ParameterSpec,DA<:StreakData{FT},R,S,M}
    experiment::E
    pspec::P
    data::DA
    regularizer::R
    mask::M
    npixels::Int
    sigma::S
    pixel_index::Vector{Int}
    scheduler::Symbol
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
    regularizer=nothing, scheduler::Symbol=:auto)
    data.sigma === nothing && throw(ArgumentError(
        "FitProblem: data has no sigma; make one with sigma(model, streak)"))

    experiment.trace === pspec.trace || throw(ArgumentError(
        "FitProblem: the experiment's trace is not this ParameterSpec's own rebuilt trace " *
        "(unflatten writes into pspec.trace, so experiment.trace must be === pspec.trace; " *
        "build the experiment from pspec.trace once, e.g. Experiment(; trace = pspec.trace, ...))"))

    mask = data.mask === nothing ? trues(size(data.image)) : data.mask
    npixels = count(mask)
    npixels == 0 && throw(ArgumentError("FitProblem: the mask excludes every pixel"))

    _check_axes_match(data, experiment)

    pixel_index = findall(vec(mask))
    return FitProblem(experiment, pspec, data, regularizer, mask, npixels, data.sigma,
        pixel_index, scheduler)
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
    expt = unflatten(problem.experiment, problem.pspec, theta)
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

# --- show (spec D40: "every exported type", FitProblem is not in D40's own worked list) ---

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
