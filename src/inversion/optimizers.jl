# The four optimizers behind one `solve`, and the one new dependency of the phase -- Step 3.4b,
# chunk 9 of Phase 3. Spec D27-D30, OQ-20.
#
# All four work in the unconstrained `y` of the logit transform (D3), so none of them needs a
# projection: no iterate can leave its bounds by construction. `import Optim`, never `using
# Optim` (Pitfall 21): `Optim.LBFGS` and this file's own exported `LBFGS` are different types
# with the same name, and none of Optim's status accessors is exported.

import Optim
using LinearAlgebra: dot, norm
using Printf: @sprintf
using Random: Xoshiro, randn, randn!

# ------------------------------------------------------------------------------------------
# AbstractOptimizer
# ------------------------------------------------------------------------------------------

"""
    AbstractOptimizer

The optimizer interface (`conventions.md` Section 4): every concrete optimizer implements
`solve(problem::FitProblem, optimizer; start = flatten(problem.pspec)) -> FitResult`, working
entirely in the unconstrained `y` of the logit transform (D3, [OQ-17](../../dev/open_questions.md#oq-17))
so that no optimizer needs a projection -- no iterate can leave its bounds by construction.
Nothing outside `solve` dispatches on a concrete optimizer type (D30).

Spec D27-D30.
"""
abstract type AbstractOptimizer end

# Shared helpers, used by more than one optimizer below.

"""
    _residuals_jacobian(problem, theta) -> (r, Jac)

One `_full_residuals` and one `jacobian` evaluation at the same `theta` (D16, D17-D19): the
shared pass `LevenbergMarquardt` and `NoisyGradientDescent` each build their own gradient
(`jtvp!`) and their trajectory's `chi2`/`R` split from, without a second residual pass. Not part
of D28/D29's own signatures; Level 3's helper, flagged under rule 11 zero.
"""
function _residuals_jacobian(problem::AbstractFitProblem, theta::AbstractVector)
    r = _full_residuals(problem, theta)
    Jac = jacobian(problem, theta)
    return r, Jac
end

"""
    _chi2_from_r(problem, r) -> Real

`sum(abs2, r[1:npixels])`, the data half of an already-computed `r = _full_residuals(problem,
theta)` -- exactly `chi2(problem, theta)` (D16), read off `r` rather than recomputed with a
second `measure` pass.
"""
_chi2_from_r(problem::AbstractFitProblem, r::AbstractVector) = sum(abs2, view(r, 1:problem.npixels))

"""
    _R_value(problem, theta) -> Real

`0.0` with no regularizer, `_regularizer_value(problem.regularizer, problem.pspec, theta)`
otherwise (D25: Λ-free). `r_full`'s own regularizer rows already carry `sqrt(Λ w_term)`, so `R`
cannot be read off `r` the way `chi2` can and needs this separate, but cheap (no `measure` call),
evaluation.
"""
function _R_value(problem::AbstractFitProblem, theta::AbstractVector)
    problem.regularizer === nothing && return 0.0
    return _regularizer_value(problem.regularizer, problem.pspec, theta)
end

"""
    _chi2_and_R(problem, theta) -> (chi2, R)

`(chi2(problem, theta), _R_value(problem, theta))`, for an optimizer (`LBFGS`) that does not
already have `r_full` in hand at the iterate being recorded.
"""
_chi2_and_R(problem::AbstractFitProblem, theta::AbstractVector) = (chi2(problem, theta), _R_value(problem, theta))

"""
    _eval_all!(g, problem, theta) -> (objective, chi2, R)

Fills `g` with `grad_theta O = 2 Jᵀ_full r_full` and returns `(𝒪, chi2, R)` at `theta`, from one
shared `_full_residuals`/`jacobian` pass -- the same work `_objective_and_gradient!` (D26) does,
inlined here so `NoisyGradientDescent`'s own step can also read off `chi2` and `R` for its
trajectory, which the opaque `_objective_and_gradient!` does not expose. Not part of D29's own
signature; Level 3's helper, flagged under rule 11 zero.
"""
function _eval_all!(g::AbstractVector, problem::AbstractFitProblem, theta::AbstractVector)
    r, Jac = _residuals_jacobian(problem, theta)
    jtvp!(g, problem, Jac, r)
    g .*= 2
    return sum(abs2, r), _chi2_from_r(problem, r), _R_value(problem, theta)
end

const _TRAJECTORY_ROW = NamedTuple{(:iteration, :objective, :chi2, :R),NTuple{4,Float64}}

"""
    _count_near_bound(pspec, y) -> Int

The count of fitted coordinates whose **x**-space value (`_from_unconstrained(y[i], v.lower,
v.upper)`) is within `BOUND_PROXIMITY_TOL` of a bound (`_near_bound`, the same helper
`FitResult.near_bound` uses), walking every per-bin entry (`pspec.bin_entries`/`bin_slots`) and
every global entry (`pspec.global_entries`/`global_slots`) exactly once -- together the whole of
`y` (`_build_index_map`, `parameters.jl`). Step 3.C.1's `g_trace` hook (D13); not part of D27's own
signature, Level 3's helper, flagged under rule 11 zero.
"""
function _count_near_bound(pspec::ParameterSpec, y::AbstractVector)
    n = 0
    for (j, entries) in enumerate(pspec.bin_entries)
        slots = pspec.bin_slots[j]
        for (k, idx) in enumerate(entries)
            v = pspec.variables[slots[k]]
            x = _from_unconstrained(y[idx], v.lower, v.upper)
            _near_bound(x, v.lower, v.upper) && (n += 1)
        end
    end
    for (k, idx) in enumerate(pspec.global_entries)
        v = pspec.variables[pspec.global_slots[k]]
        x = _from_unconstrained(y[idx], v.lower, v.upper)
        _near_bound(x, v.lower, v.upper) && (n += 1)
    end
    return n
end

"""
    _build_result(problem, theta, objective_value, trajectory, converged, status, iterations,
                  f_calls, g_calls, elapsed, name; termination = :none) -> FitResult

Assembles the `FitResult` every `solve` method below returns (D26): `values`/`near_bound` from
`_value_table`/`_near_bound` (D13) over every entry of `pspec.variables` (frozen or not, OB5) and
every dependent, a scalar variable's entry a length-1 vector (D26); `chi2`, `R` and `term_values`
freshly evaluated at `theta` (D25). **`objective` is the caller's own already-computed value, not
recomputed as `chi2 + weight * R`**: the two agree only to `1e-14` relative (OB1, OB2), not
bitwise, while OP7 and OP10 require the returned `objective` to be `===` the trajectory's own
recorded value at the same iterate -- so it must be threaded through, not rebuilt from the split.
`termination` defaults to `:none`, which is what `LevenbergMarquardt` and `NoisyGradientDescent`
get (Step 3.C.1, D13); `LBFGS` and `OptimizerSequence` pass their own. Not part of D26-D30's own
signatures; Level 3's helper, shared by all four `solve` methods below and flagged under rule 11
zero.

**Widened to `problem::AbstractFitProblem` (Step 4.A.3a, D36)**: `values`/`near_bound`/`Rval`/
`term_values`/`weight` are unchanged, since they read only `pspec`/`theta`/`reg`; `c2 =
chi2(problem, theta)` is the weighted total for a `JointFitProblem` (Pitfall 26); `experiments[c] =
unflatten(ch.experiment, pspec, theta; channel = ch.channel)` for every channel (`(problem,)` for
a plain `FitProblem`), and one `ChannelSummary` per channel from `_channel_nparams` (D36) -- no
Jacobian call.
"""
function _build_result(problem::AbstractFitProblem, theta::AbstractVector{Float64}, objective_value::Float64,
    trajectory::Vector{_TRAJECTORY_ROW}, converged::Bool, status::Symbol, iterations::Int,
    f_calls::Int, g_calls::Int, elapsed::Float64, name::AbstractString; termination::Symbol=:none)
    pspec = problem.pspec
    nvar = length(pspec.variables)
    vtable = _value_table(pspec, theta)

    values = Dict{String,Vector{Float64}}()
    near_bound = Dict{String,BitVector}()
    for (i, v) in enumerate(pspec.variables)
        row = vec(vtable[i, :])
        vals = v.time_varying ? row : [row[1]]
        values[v.name] = vals
        near_bound[v.name] = BitVector(_near_bound.(vals, v.lower, v.upper))
    end
    for (k, d) in enumerate(pspec.dependents)
        values[d.name] = vec(vtable[nvar+k, :])
    end

    reg = problem.regularizer
    if reg === nothing
        Rval = 0.0
        weight = 0.0
        term_values = Float64[]
    else
        raw = _regularizer_raw_rows(reg, pspec, theta)
        term_values = [term.weight * sum(abs2, view(raw, term.rows)) for term in reg.terms]
        Rval = sum(term_values)
        weight = reg.weight
    end
    c2 = chi2(problem, theta)

    channels_tuple = problem isa JointFitProblem ? problem.channels : (problem,)
    weights_vec = problem isa JointFitProblem ? problem.weights : [1.0]

    experiments = ntuple(length(channels_tuple)) do c
        ch = channels_tuple[c]
        unflatten(ch.experiment, pspec, theta; channel=ch.channel)
    end

    channel_summaries = ChannelSummary[]
    for (c, ch) in enumerate(channels_tuple)
        chi2_c = chi2(ch, theta)
        p_c = _channel_nparams(pspec, ch.channel)
        n_c = ch.npixels
        reduced = n_c > p_c ? chi2_c / (n_c - p_c) : NaN
        reduced_simple = n_c == 0 ? NaN : chi2_c / n_c
        push!(channel_summaries, ChannelSummary(ch.name, weights_vec[c], chi2_c, n_c, p_c, reduced, reduced_simple))
    end

    return FitResult(Vector{Float64}(theta), experiments[1], pspec, values, near_bound, c2, Rval,
        objective_value, weight, term_values, problem.npixels, trajectory, converged, status,
        termination, iterations, f_calls, g_calls, elapsed, String(name), experiments, channel_summaries)
end

# ------------------------------------------------------------------------------------------
# Phase 4.K: the progress lines (OQ-61)
# ------------------------------------------------------------------------------------------

"""
    PROGRESS_WINDOW

`20`: how many recorded rows back a progress line's `dchi2` reaches (Phase 4.K spec D6,
[OQ-61](../../dev/open_questions.md#oq-61) item 3: the user's "d(chi^2) over the last 20
iterations"). A module constant, not a deck key.
"""
const PROGRESS_WINDOW = 20

"""
    _progress_header(io, name, problem, cap, every, g_abstol) -> nothing

The line a `solve` with `progress = every > 0` prints before its first iteration (Phase 4.K spec D6,
[OQ-61](../../dev/open_questions.md#oq-61) item 3):

    progress <name>: start npixels=<N> ntheta=<M> cap=<cap> every=<every> g_abstol=<g>

`g_abstol` (`%.2e`) is the threshold the optimizer's gradient stop compares `gmax` with; `nothing`
(the explorer, which has no gradient stop) leaves the field out. One `print` of the assembled line,
then `flush` (`conventions.md` 8.6 (iv)).
"""
function _progress_header(io::IO, name::AbstractString, problem::AbstractFitProblem, cap::Integer,
    every::Integer, g_abstol::Union{Nothing,Real})
    gtext = g_abstol === nothing ? "" : @sprintf(" g_abstol=%.2e", g_abstol)
    line = @sprintf("progress %s: start npixels=%d ntheta=%d cap=%d every=%d%s\n", name,
        problem.npixels, problem.pspec.ntheta, cap, every, gtext)
    print(io, line)
    flush(io)
    return nothing
end

"""
    _progress_line(io, name, problem, trajectory, i, y, gmax, t0; acc, rej, mu, best, noise) -> nothing

The line for the recorded row of index `i` (`0` the start; the row is `trajectory[i + 1]`, already
recorded), Phase 4.K spec D6, [OQ-61](../../dev/open_questions.md#oq-61) item 3:

    progress <name>: it=<it> chi2=<χ²> chi2/n=<χ²/N> R=<R> obj=<𝒪> dchi2=<Δ> rate=<Δ/w> gmax=<g> nb=<b>/<M> t=<t>s s/it=<t/it>

followed by ` acc=<a> rej=<r> mu=<μ>` when `mu` is given (`LevenbergMarquardt`) or ` best=<𝒪_best>
noise=<s>` when `best` is given (`NoisyGradientDescent`). `it` is the row's `iteration`; `chi2`, `R`
and `obj` the row's own; `chi2/n` its χ² over `problem.npixels`; `dchi2` its χ² minus that of the row
`w = min(i, PROGRESS_WINDOW)` rows earlier, and `rate = dchi2 / w`, both `0.0` at `i = 0`; `gmax` the
caller's ∞-norm of ∇_y 𝒪 at `y`, as its own stop sees it; `nb` `_count_near_bound(problem.pspec, y)`
over `ntheta`; `t` the seconds since `t0`; `s/it` `t / it`, left out at `it = 0`. Formats: `%.6g` for
χ², 𝒪 and `best`, `%.4f` for χ²/n, `%.4g` for R, `%+.3e` for `dchi2` and `rate`, `%.2e` for `gmax`,
`mu` and `noise`, `%.1f` for `t`, `%.2f` for `s/it`. It reads the trajectory, `y` and the numbers it is
given and evaluates nothing of the problem but `_count_near_bound`, a pure walk over `y`. One `print`
of the assembled line, then `flush`.
"""
function _progress_line(io::IO, name::AbstractString, problem::AbstractFitProblem,
    trajectory::AbstractVector{_TRAJECTORY_ROW}, i::Integer, y::AbstractVector, gmax::Real, t0::Real;
    acc::Union{Nothing,Integer}=nothing, rej::Union{Nothing,Integer}=nothing,
    mu::Union{Nothing,Real}=nothing, best::Union{Nothing,Real}=nothing,
    noise::Union{Nothing,Real}=nothing)
    row = trajectory[i+1]
    w = min(i, PROGRESS_WINDOW)
    dchi2 = w == 0 ? 0.0 : row.chi2 - trajectory[i-w+1].chi2
    rate = w == 0 ? 0.0 : dchi2 / w
    t = time() - t0
    it = Int(row.iteration)
    sit = it == 0 ? "" : @sprintf(" s/it=%.2f", t / it)
    tail = mu !== nothing ? @sprintf(" acc=%d rej=%d mu=%.2e", acc, rej, mu) :
           best !== nothing ? @sprintf(" best=%.6g noise=%.2e", best, noise) : ""
    line = @sprintf("progress %s: it=%d chi2=%.6g chi2/n=%.4f R=%.4g obj=%.6g dchi2=%+.3e rate=%+.3e gmax=%.2e nb=%d/%d t=%.1fs%s%s\n",
        name, it, row.chi2, row.chi2 / problem.npixels, row.R, row.objective, dchi2, rate, gmax,
        _count_near_bound(problem.pspec, y), problem.pspec.ntheta, t, sit, tail)
    print(io, line)
    flush(io)
    return nothing
end

"""
    _progress_final(io, result::FitResult) -> nothing

The line a `solve` with `progress > 0` prints at every stop, once it has its `FitResult` (Phase 4.K
spec D6, [OQ-61](../../dev/open_questions.md#oq-61) item 3):

    progress <name>: done status=<status> converged=<bool> it=<it> chi2=<χ²> chi2/n=<χ²/N> obj=<𝒪> t=<t>s

`name` the result's `optimizer`; `it` the last recorded row's `iteration`; `chi2`, `chi2/n`, `obj` and
`t` the result's own `chi2`, `chi2 / npixels`, `objective` and `elapsed`: the fit as returned, which for
`NoisyGradientDescent` is its best iterate, not its last row. One `print`, then `flush`.
"""
function _progress_final(io::IO, result::FitResult)
    line = @sprintf("progress %s: done status=%s converged=%s it=%d chi2=%.6g chi2/n=%.4f obj=%.6g t=%.1fs\n",
        result.optimizer, result.status, result.converged, Int(result.trajectory[end].iteration),
        result.chi2, result.chi2 / result.npixels, result.objective, result.elapsed)
    print(io, line)
    flush(io)
    return nothing
end

# ------------------------------------------------------------------------------------------
# D27: LBFGS
# ------------------------------------------------------------------------------------------

"""
    LBFGS(; m = 10, f_abstol_per_pixel = 1e-4, patience = 5, g_abstol_per_pixel = 1e-6,
          x_reltol = 0.0, iterations = 20_000, time_limit = NaN,
          precondition = true, precond_refresh = 25, progress = 0)

Optim.jl's L-BFGS, the polisher (foo2026 Section III.D phase 2; [OQ-20](../../dev/open_questions.md#oq-20)).
`m` is Optim's LBFGS memory. **The convergence criterion is stated per pixel** (Step 3.C.2, D30),
because `𝒪` is not comparable between problems or resolutions: `chi2` per pixel is about 1 at a
good fit on every problem, so `f_abstol_per_pixel` and `g_abstol_per_pixel` mean the same thing
everywhere, while an absolute threshold on `𝒪` would be a different threshold on every problem and
every resolution. `solve` derives the absolute thresholds `f_abstol = f_abstol_per_pixel *
problem.npixels` and `g_abstol = g_abstol_per_pixel * problem.npixels` and passes those to Optim.

**The primary test is a decrease of `𝒪` smaller than `f_abstol` for `patience` consecutive
iterations** (`successive_f_tol = patience - 1` in Optim's own semantics, D14: `k` requires `k + 1`
qualifying iterations). This is the mechanism that already converges a fit whose gradient never
gets close to `g_abstol` (D30, measured on multi-flow and shock). The **secondary** test is the
usual gradient norm, `g_abstol`, raised from Optim's own `1e-8` default to a per-pixel figure five
orders above it -- still far below the smallest gradient any non-saturated fit reached, and now
reachable by a real minimum rather than only by numerical noise.

`iterations = 20_000` (D32): the worst cut-down problem measured (shock) had not converged by the
*old* criterion at 10 000 iterations; under the per-pixel criterion the criterion, not the cap, is
expected to end every fit, and this is twice the largest cap ever run. `time_limit` in seconds,
`NaN` (off) by default.

**The saturation trap.** When the starting gradient is large in some direction, Optim's default
line search can accept a **first step that lands the logit fully in its saturated region**
(`_logistic(y)` is exactly `1.0` for `y ≳ 37`, D13). There `dx/dy`, and so the true gradient with
respect to every saturated coordinate, is **exactly zero**, the Wolfe curvature condition is
satisfied trivially by `0 >= (a negative number)`, and `Optim.g_converged` would then report
`status = :gradient_norm` truthfully on a fit that is garbage. **`solve` now catches this itself**
(D31): a gradient-norm stop where at least `SATURATION_FRACTION` of `theta`'s coordinates are
within `BOUND_PROXIMITY_TOL` of a bound is reported as `status = :saturated`, `converged = false`,
never as a converged fit. **No value of `g_abstol_per_pixel` fixes this on its own** -- EPW's own
saturated stop starts with a y-space gradient already far below even the old, far tighter
`g_abstol`, so raising the threshold only makes the trap easier to fall into, not harder; the
saturation check is what makes it safe to keep a gradient test at all.

**The preconditioner (Step 3.C.3, D36, D37).** When `precondition` (default `true`), `solve`
builds a diagonal Gauss-Newton preconditioner from `_gauss_newton_diagonal` -- `d_i = sum_rows
(dr_full/dy_i)^2`, computed once at the start from `jacobian(problem, y0)` and passed to
`Optim.LBFGS` as `P = Optim.InverseDiagonal(1 ./ (2 .* dg))`, `dg = _gauss_newton_floor(d)` (D37's
relative floor, `PRECOND_FLOOR_REL = 1e-8`). **`P` is a Hessian approximation and Optim applies
`P⁻¹∇E`** (D14, measured on this package's resolved Optim 2.3.1: `Optim.LBFGS`'s own
`scaleinvH0 = (P === nothing)`, so supplying `P` also switches off Nocedal's initial scaling,
changing the *first* step's length as well as every later one), so the quantity built here is
`2d`, never `d` itself. `precond_refresh` (default `25`) rebuilds `P` from a fresh `jacobian` every
that many attempted iterations (through a `precondprep!` closure Optim itself calls once per outer
loop pass); a degenerate diagonal at a refresh (`_gauss_newton_floor` returning `nothing`) leaves
`P` unchanged, and a degenerate diagonal at the very first build runs the fit unpreconditioned
(`P = nothing`, `scaleinvH0 = true`, Optim's own default) rather than dividing by nothing.

**`progress`** (Phase 4.K spec D4, [OQ-61](../../dev/open_questions.md#oq-61)): `0`, the default, prints
nothing; `n > 0` makes `solve` print a header, a line at every `n`th recorded iteration (the start
included) and a final line to its `progress_io`, each flushed (`solve`'s docstring). Refused below 0.

Spec D27, D30, D31, D32, D36, D37; Phase 4.K spec D4.
"""
Base.@kwdef struct LBFGS <: AbstractOptimizer
    m::Int = 10
    f_abstol_per_pixel::Float64 = 1e-4
    patience::Int = 5
    g_abstol_per_pixel::Float64 = 1e-6
    x_reltol::Float64 = 0.0
    iterations::Int = 20_000
    time_limit::Float64 = NaN
    precondition::Bool = true
    precond_refresh::Int = 25
    progress::Int = 0
    function LBFGS(m, f_abstol_per_pixel, patience, g_abstol_per_pixel, x_reltol, iterations,
        time_limit, precondition, precond_refresh, progress)
        progress >= 0 || throw(ArgumentError("LBFGS: progress must be an integer ≥ 0; got $progress"))
        return new(m, f_abstol_per_pixel, patience, g_abstol_per_pixel, x_reltol, iterations,
            time_limit, precondition, precond_refresh, progress)
    end
end

"""
    solve(problem::FitProblem, opt::LBFGS; start = flatten(problem.pspec),
          g_trace::Union{Nothing,Vector} = nothing, progress_io::IO = stdout) -> FitResult

Runs Optim's L-BFGS on `objective`/`_objective_and_gradient!` (D26) through
`Optim.OnceDifferentiable(f, g!, fg!, start)` with a **two**-argument `fg!(G, y) -> F` (D2, D27):
Optim then calls only `fg!`, halving the work of the fit relative to separate `f`/`g!` calls.
`theta`/`y` are the unconstrained variables of the logit transform (D3). The trajectory is
recorded through `Optim.Options(callback = ...)`, one `(iteration, 𝒪, chi2, R)` per outer
iteration including iteration 0 (Level 2's recommendation, D27, so "iteration" means the same
thing across all four optimizers) -- at the cost of one extra `chi2`/`R` evaluation per recorded
iteration, since Optim's own trace and callback state expose only the combined `𝒪`, never the
split.

**The criterion (D30).** `f_abstol = opt.f_abstol_per_pixel * problem.npixels`, `g_abstol =
opt.g_abstol_per_pixel * problem.npixels`, `successive_f_tol = opt.patience - 1` (D14's measured
semantics: `k` needs `k + 1` qualifying iterations, so `patience` itself is off by one from what
Optim is actually asked for) and `f_reltol = 0.0` (D30 rejects a relative test, which would demand
different flatness of every problem for no physical reason). Status is read off Optim's own
accessors, none of which is exported (Pitfall 21), in an order that matters: `Optim.g_converged`
is checked **first**, together with the saturation guard below, so a saturated stop can never be
relabelled by a later branch --
`Optim.g_converged` (and saturated) -> `:saturated`, `Optim.g_converged` (and not saturated) ->
`:gradient_norm`, `Optim.f_converged` -> `:f_tolerance`, `Optim.x_converged` -> `:x_tolerance`,
`Optim.iteration_limit_reached` -> `:iterations`, otherwise `:failed`. `FitResult.termination` is
Optim's own finer code, `Symbol(string(Optim.termination_code(res)))` (D13, D14).

**The saturation guard (D31).** `near_bound_fraction = _count_near_bound(problem.pspec, theta) /
problem.pspec.ntheta`; when `Optim.g_converged(res)` is true **and** `near_bound_fraction >=
SATURATION_FRACTION`, `status = :saturated` and `converged = false` regardless of what
`Optim.converged(res)` itself says -- `Optim.g_converged` is truthfully reporting a y-space
gradient near zero, but one driven there by `dx/dy -> 0` at a saturated logit, not by proximity to
a minimum. **`:converged_at_start` (D17) is layered on top of that**, and only on top of it: when
`Optim.iterations(res) == 0` and the run converged by any of Optim's own tests, the status becomes
`:converged_at_start` -- **unless the saturation check already said `:saturated`, which takes
precedence** (EPW's own two exact-1.0 scan points are both: the honest label is `:saturated`).

A status of `:iterations` or `:saturated` -- the fit did not converge, or "converged" in a way that
is not to be trusted -- emits `@warn` naming the optimizer, `problem.npixels`, the iteration count
and the gradient norm at the returned point (D34).

`g_trace`, given a `Vector`, has one named tuple `(iteration, gnorm_inf, gnorm_2, near_bound)`
pushed per recorded iteration, from inside the same callback the trajectory uses and on the same
counter, so `g_trace[k]` and `trajectory[k]` are the same iteration (Step 3.C.1, D13):
`gnorm_inf`/`gnorm_2` from `state.g_x` (the gradient at `state.x`, D14 -- `LBFGSState` has no
`iteration` field, which is why the counter `solve` already keeps is reused rather than read off
`state`); `near_bound` is `_count_near_bound(problem.pspec, state.x)`, the count of coordinates
whose **x**-space value is within `BOUND_PROXIMITY_TOL` of a bound. A test hook that changes
nothing about the default call shape, on the same footing as `LevenbergMarquardt`'s `mu_trace` and
`NoisyGradientDescent`'s `y_trace`.

`precond_trace`, given a `Vector`, has one named tuple `(iteration, floor_bound)` pushed each time
the preconditioner is actually (re)built -- the initial build (`iteration = 0`) and every
subsequent refresh (Step 3.C.3, D37) -- `floor_bound` the count of `_gauss_newton_diagonal`'s
entries `_gauss_newton_floor` raised at that build (D37: "count and report how many entries the
floor binds on at each refresh"). A degenerate refresh that leaves `P` untouched pushes nothing.
Not part of D36/D37's own signature (given as a reporting requirement, not a hook); Level 3's
addition, on the same footing as `g_trace` above, flagged under rule 11 zero.

**The progress lines** (Phase 4.K spec D5 to D7, [OQ-61](../../dev/open_questions.md#oq-61)): with
`opt.progress = n > 0`, a header before the preconditioner's first build, a line from the callback at
every recorded iteration whose index is a multiple of `n` (the start included) and a final line once the
`FitResult` is built, each one `print` and one `flush` to `progress_io`, which is looked up at the call
(`stdout` by default, so a `redirect_stdout` around the call catches it). `gmax` is
`maximum(abs, state.g_x)`, the quantity Optim's gradient stop compares with `g_abstol`; `nb` counts the
near-bound entries of `state.x`. Nothing is evaluated for them that the fit does not already compute,
`_count_near_bound` aside, so every `FitResult` field but `elapsed` is the same, bitwise, with the lines
on and off.

Spec D27, D13, D30, D31, D34, D36, D37; Phase 4.K spec D5 to D7.
"""
function solve(problem::AbstractFitProblem, opt::LBFGS; start::AbstractVector=flatten(problem.pspec),
    g_trace::Union{Nothing,Vector}=nothing, precond_trace::Union{Nothing,Vector}=nothing,
    progress_io::IO=stdout)
    t0 = time()
    y0 = Vector{Float64}(start)

    trajectory = _TRAJECTORY_ROW[]
    iter_counter = Ref(0)

    f(y) = objective(problem, y)
    g!(G, y) = (_objective_and_gradient!(G, problem, y); G)
    fg!(G, y) = _objective_and_gradient!(G, problem, y)

    function _lbfgs_callback(state)
        chi2v, Rval = _chi2_and_R(problem, state.x)
        push!(trajectory, (iteration=Float64(iter_counter[]), objective=Float64(state.f_x),
            chi2=Float64(chi2v), R=Float64(Rval)))
        if g_trace !== nothing
            push!(g_trace, (iteration=iter_counter[], gnorm_inf=maximum(abs, state.g_x),
                gnorm_2=sqrt(sum(abs2, state.g_x)),
                near_bound=_count_near_bound(problem.pspec, state.x)))
        end
        if opt.progress > 0 && iter_counter[] % opt.progress == 0
            _progress_line(progress_io, _optimizer_short_name(opt), problem, trajectory, iter_counter[],
                state.x, maximum(abs, state.g_x), t0)
        end
        iter_counter[] += 1
        return false
    end

    f_abstol = opt.f_abstol_per_pixel * problem.npixels
    g_abstol = opt.g_abstol_per_pixel * problem.npixels
    opt.progress > 0 && _progress_header(progress_io, _optimizer_short_name(opt), problem,
        opt.iterations, opt.progress, g_abstol)

    od = Optim.OnceDifferentiable(f, g!, fg!, y0)

    # The preconditioner (D36, D37): P ~ nabla^2 E, Optim applies P^-1 grad E, so the quantity
    # filled is 2d and Optim.InverseDiagonal stores 1 ./ (2d) -- D14, measured, not re-derived.
    # `refresh_count` counts precondprep! calls (one per Optim.LBFGS outer-loop pass, confirmed
    # against this package's resolved Optim 2.3.1: `update_state!` calls `_apply_precondprep`
    # once per `while` iteration, `src/multivariate/solvers/first_order/l_bfgs.jl`); the diagonal
    # is rebuilt only when that count is a multiple of `opt.precond_refresh`, otherwise `P` is
    # left untouched, matching "rebuild every precond_refresh accepted iterations" (D37).
    lbfgs_method = if opt.precondition
        d0 = _gauss_newton_diagonal(problem, jacobian(problem, y0))
        dg0 = _gauss_newton_floor(d0)
        if dg0 === nothing
            Optim.LBFGS(; m=opt.m)   # degenerate at the very first build: run unpreconditioned
        else
            if precond_trace !== nothing
                nbound0 = count(<(PRECOND_FLOOR_REL * maximum(d0)), d0)
                push!(precond_trace, (iteration=0, floor_bound=nbound0))
            end
            P0 = Optim.InverseDiagonal(1 ./ (2 .* dg0))
            refresh_count = Ref(0)
            function _lbfgs_precondprep!(P, x)
                refresh_count[] += 1
                mod(refresh_count[], opt.precond_refresh) == 0 || return nothing
                d = _gauss_newton_diagonal(problem, jacobian(problem, x))
                dg = _gauss_newton_floor(d)
                dg === nothing && return nothing   # degenerate refresh: keep the previous P
                P.diag .= 1 ./ (2 .* dg)
                if precond_trace !== nothing
                    nbound = count(<(PRECOND_FLOOR_REL * maximum(d)), d)
                    push!(precond_trace, (iteration=refresh_count[], floor_bound=nbound))
                end
                return nothing
            end
            Optim.LBFGS(; m=opt.m, P=P0, precondprep=_lbfgs_precondprep!)
        end
    else
        Optim.LBFGS(; m=opt.m)
    end

    res = Optim.optimize(od, y0, lbfgs_method,
        Optim.Options(; f_abstol=f_abstol, successive_f_tol=opt.patience - 1, g_abstol=g_abstol,
            x_reltol=opt.x_reltol, f_reltol=0.0, iterations=opt.iterations,
            time_limit=opt.time_limit, store_trace=true, extended_trace=true,
            callback=_lbfgs_callback))

    termination = Symbol(string(Optim.termination_code(res)))
    theta = Optim.minimizer(res)

    nb = _count_near_bound(problem.pspec, theta)
    near_bound_fraction = nb / problem.pspec.ntheta
    status = if Optim.g_converged(res) && near_bound_fraction >= SATURATION_FRACTION
        :saturated
    elseif Optim.g_converged(res)
        :gradient_norm
    elseif Optim.f_converged(res)
        :f_tolerance
    elseif Optim.x_converged(res)
        :x_tolerance
    elseif Optim.iteration_limit_reached(res)
        :iterations
    else
        :failed
    end
    if status !== :saturated && Optim.iterations(res) == 0 && Optim.converged(res)
        status = :converged_at_start
    end
    converged = status === :saturated ? false : Optim.converged(res)

    if status === :iterations || status === :saturated
        gfinal = zeros(Float64, problem.pspec.ntheta)
        _objective_and_gradient!(gfinal, problem, theta)
        @warn "LBFGS did not converge to a trustworthy minimum" optimizer = "lbfgs" status problem.npixels iterations = Optim.iterations(res) gnorm_inf = maximum(abs, gfinal)
    end

    elapsed = time() - t0
    result = _build_result(problem, theta, Optim.minimum(res), trajectory, converged,
        status, Optim.iterations(res), Optim.f_calls(res), Optim.g_calls(res), elapsed, "lbfgs";
        termination=termination)
    opt.progress > 0 && _progress_final(progress_io, result)
    return result
end

Base.show(io::IO, opt::LBFGS) = print(io, "LBFGS(m = ", opt.m,
    ", f_abstol_per_pixel = ", opt.f_abstol_per_pixel, ", patience = ", opt.patience,
    ", g_abstol_per_pixel = ", opt.g_abstol_per_pixel, ", iterations = ", opt.iterations, ")")

# ------------------------------------------------------------------------------------------
# D28: LevenbergMarquardt
# ------------------------------------------------------------------------------------------

"""
    LevenbergMarquardt(; iterations = 2000, g_abstol = 1e-8, x_reltol = 1e-10, mu_init = 1e-3,
                        cgls_tol = 1e-6, cgls_maxiter = 2000, seed = 20260908,
                        precondition = true, progress = 0, lm_rtol = 0.0, lm_rtol_window = 50)

Levenberg-Marquardt, expected fastest near the minimum for a least-squares objective
([OQ-20](../../dev/open_questions.md#oq-20)). `iterations` caps the outer loop; `g_abstol` stops
on `‖grad_y O‖_inf <= g_abstol`; `x_reltol` stops on `‖delta‖ <= x_reltol * (‖y‖ + x_reltol)`;
`mu_init` scales the one-`jvp!` curvature estimate that seeds `mu` (`mu0 = mu_init * s`, D28);
`cgls_tol`/`cgls_maxiter` bound the inner CGLS solve of the damped normal equations; `seed` is
`Xoshiro`'s seed for the one random direction `mu0`'s estimate is drawn along, so two runs are
bitwise identical (D28, OP16).

**`iterations` and `cgls_maxiter` are raised from 200/100 (D32)**: multi-flow's cut-down problem
saturated a 2000-iteration outer cap *and* its inner CGLS cap simultaneously at the old defaults,
with a mean of 941.6 and a max of exactly 1000 CGLS iterations per outer step at
`cgls_maxiter = 1000` -- the shipped `100` was an order below what one outer step needs. `g_abstol`
is left at Optim's-style `1e-8` this step; LM's convergence test is its own, and no deck in this
phase runs it (Step 3.C.2 leaves it alone rather than harmonizing it on no measurement of its own).
**At these raised defaults a realistic problem is a minutes-to-tens-of-minutes call**: 763 s
measured on a 128 x 32 problem at `iterations = 2000, cgls_maxiter = 1000`, and raising
`cgls_maxiter` to 2000 can double an outer step's inner work again.

**`precondition`** (D41, settling the asymmetry D37 left open): `true` runs CGLS Jacobi-
preconditioned on the floored Gauss-Newton diagonal, exactly as the "CGLS is Jacobi-preconditioned"
paragraph of `solve`'s own docstring below describes; `false` passes `precond = nothing` to
`_cgls`, which chunk 7's own measurement (OP14) found bitwise identical to the unpreconditioned
solve. Deck-settable like every other field, on the same footing as `LBFGS`'s own
`precondition::Bool`.

**`progress`** (Phase 4.K spec D4, [OQ-61](../../dev/open_questions.md#oq-61)): `0`, the default, prints
nothing; `n > 0` makes `solve` print a header, a line at every `n`th accepted step (the start included)
and a final line to its `progress_io`, each flushed (`solve`'s docstring). Refused below 0.

**`lm_rtol` and `lm_rtol_window`** (Phase 4.L spec D4, [OQ-62](../../dev/open_questions.md#oq-62)): the
stall test. `lm_rtol = 0.0`, the default, is off, and the fit is exactly what it was without it. With
`lm_rtol > 0`, `solve` ends the fit, converged, with `status = :lm_rtol` when the objective 𝒪 has moved
by at most `lm_rtol * |𝒪|` over the last `lm_rtol_window` accepted steps (`solve`'s docstring), so a
fit that has stopped improving ends there instead of at `iterations`. The window counts accepted steps,
the trajectory's rows, not attempted ones. `lm_rtol` must be a finite number ≥ 0 and `lm_rtol_window`
an integer ≥ 1; anything else is refused.

Spec D28, D32, D41; Phase 4.K spec D4; Phase 4.L spec D4.
"""
Base.@kwdef struct LevenbergMarquardt <: AbstractOptimizer
    iterations::Int = 2000
    g_abstol::Float64 = 1e-8
    x_reltol::Float64 = 1e-10
    mu_init::Float64 = 1e-3
    cgls_tol::Float64 = 1e-6
    cgls_maxiter::Int = 2000
    seed::Int = 20260908
    precondition::Bool = true
    progress::Int = 0
    lm_rtol::Float64 = 0.0
    lm_rtol_window::Int = 50
    function LevenbergMarquardt(iterations, g_abstol, x_reltol, mu_init, cgls_tol, cgls_maxiter, seed,
        precondition, progress, lm_rtol, lm_rtol_window)
        progress >= 0 || throw(ArgumentError(
            "LevenbergMarquardt: progress must be an integer ≥ 0; got $progress"))
        (isfinite(lm_rtol) && lm_rtol >= 0) || throw(ArgumentError(
            "LevenbergMarquardt: lm_rtol must be a finite number ≥ 0; got $lm_rtol"))
        lm_rtol_window >= 1 || throw(ArgumentError(
            "LevenbergMarquardt: lm_rtol_window must be an integer ≥ 1; got $lm_rtol_window"))
        return new(iterations, g_abstol, x_reltol, mu_init, cgls_tol, cgls_maxiter, seed, precondition,
            progress, lm_rtol, lm_rtol_window)
    end
end

"""
    _cgls(problem, Jac, r, gvec, mu, tol, maxiter; precond = nothing) -> (delta, niter)

CGLS on the damped least-squares system `min_delta ‖[J; sqrt(mu) I] delta + [r; 0]‖^2`,
matrix-free through `jvp!`/`jtvp!` alone (D28): solves `(JᵀJ + mu I) delta = -gvec` for `delta`,
`gvec = Jᵀ r` already computed by the caller, stopping when the normal-equation residual's norm
squared falls to `tol^2` of its starting value or `maxiter` is reached. `niter` is how many inner
iterations actually ran (D43 item 9). `gam0 == 0` (an exactly-zero `gvec`) returns `delta = 0`
immediately rather than dividing by it. Not part of D28's own signature (given as an inline
algorithm, not a named function); Level 3's extraction, flagged under rule 11 zero.

**`precond` (Step 3.C.3, D37): Jacobi preconditioning**, the same floored Gauss-Newton diagonal
`LBFGS`'s own preconditioner uses (`_gauss_newton_floor(_gauss_newton_diagonal(problem, Jac))`,
`nothing` when the caller chose not to build one, or on a degenerate diagonal). Standard
preconditioned CG on the SPD system `(JᵀJ + mu I) delta = -gvec` with `M = Diagonal(precond)`:
`A` is still applied to the search direction `p` directly (`q = J p`, `dot(q, q) + mu * dot(p, p)`
is `pᵀ A p` regardless of preconditioning, so `alpha`'s formula is unchanged), and only the
residual-to-direction step is preconditioned, `z = gnew ./ precond` in place of `gnew` itself, in
both the initial direction and `gam`'s reduction and in the Fletcher-Reeves-style update of `p`
(`gam1/gam` with `gam = dot(r, z)`, the standard choice for SPD PCG -- corrected from an earlier
"Polak-Ribiere-style" label, D41).
With `precond === nothing` every one of those becomes `z === gnew`/`p0 === -gvec` and the method is
bitwise the unpreconditioned CGLS above. A preconditioner that changes the converged `delta` (to
more than solver-noise tolerance) is a bug (PC4); one that only changes `niter` is the point.

**`problem`/`Jac` widened (Step 4.A.3a, D34)**: `Jac` is untyped rather than pinned to
`FitJacobian`, since `_cgls` merely passes the pair through to `jvp!`/`jtvp!`, which dispatch on
whichever matching `(FitProblem, FitJacobian)` or `(JointFitProblem, JointFitJacobian)` pair is
actually passed. Left untyped rather than `Jac::Union{FitJacobian,JointFitJacobian}` since a third
problem type would otherwise need editing this signature too.

Spec D28, D37, D34.
"""
function _cgls(problem::AbstractFitProblem, Jac, r::AbstractVector, gvec::AbstractVector,
    mu::Real, tol::Real, maxiter::Integer; precond::Union{Nothing,AbstractVector}=nothing)
    ntheta = length(gvec)
    delta = zeros(Float64, ntheta)
    s = -r
    gnew0 = -gvec
    z0 = precond === nothing ? gnew0 : gnew0 ./ precond
    p = copy(z0)
    gam = dot(gnew0, z0)
    gam0 = gam
    niter = 0
    if gam0 > 0
        q = Vector{Float64}(undef, length(r))
        gnew = Vector{Float64}(undef, ntheta)
        for k in 1:maxiter
            niter = k
            jvp!(q, problem, Jac, p)
            alpha = gam / (dot(q, q) + mu * dot(p, p))
            delta .+= alpha .* p
            s .-= alpha .* q
            jtvp!(gnew, problem, Jac, s)
            gnew .-= mu .* delta
            z = precond === nothing ? gnew : gnew ./ precond
            gam1 = dot(gnew, z)
            gam1 <= tol^2 * gam0 && break
            p = z .+ (gam1 / gam) .* p
            gam = gam1
        end
    end
    return delta, niter
end

"""
    solve(problem::FitProblem, opt::LevenbergMarquardt; start = flatten(problem.pspec),
          cgls_iterations::Union{Nothing,Vector{Int}} = nothing, progress_io::IO = stdout) -> FitResult

Levenberg-Marquardt on the unconstrained `y` of the logit transform, on top of Step 3.2's block
Jacobian (D28). Working with `F(y) = 1/2 ‖r_full‖^2 = 𝒪/2` and `gvec = Jᵀ_full r_full`, each trial
step solves the damped normal equations `(JᵀJ + mu I) delta = -gvec` by `_cgls` above, matrix-free
through `jvp!`/`jtvp!` -- `J` is never formed and `JᵀJ` never assembled. **Identity damping, not
`diag(JᵀJ)`**: the logit already makes every `y` O(1), so the scale invariance `diag(JᵀJ)` buys is
already present, and computing it matrix-free would cost `N_theta` Jacobian-vector products
against CGLS's typical few dozen.

**CGLS is Jacobi-preconditioned when `opt.precondition` for the damped system `_cgls` actually
solves, `(JᵀJ + mu I) delta = -gvec`** (Step 4.A.0, D22 item 3, correcting Step 3.C.3's D37: the
diagonal passed used to be `dg` alone, which is Jacobi for `JᵀJ`, not for `JᵀJ + mu I`, and was
worst exactly where LM spends its rejected steps, at large `mu`). `dg =
_gauss_newton_floor(_gauss_newton_diagonal(problem, Jac))`, the same floored Gauss-Newton diagonal
`LBFGS`'s own preconditioner uses, recomputed from `Jac` whenever `Jac` itself is (re)assembled --
once at the start and again on every **accepted** step, never on a rejected one, since `Jac` is
not rebuilt then either. **The vector actually passed to `_cgls` is `dg .+ mu`, rebuilt every
trial step** (accepted or rejected) since `mu` changes on both, and the `+ mu` is applied after
`dg`'s own floor, never before it. A degenerate diagonal (`_gauss_newton_floor` returning
`nothing`) runs that call's `_cgls` unpreconditioned, and so does `opt.precondition = false`. This
is what chunk 5's own diagnosis measured CGLS needing: a mean of 98.2 of 100 inner iterations on
the 128 x 32 cut-down problem against 8.7 on the toy problem P2 -- the same severe conditioning
(`spread(d)` of 1.4e3-2.8e3 at the *start*) that motivates `LBFGS`'s own `P`. **This changes the
CGLS iteration count, not the converged `delta`** (PC4).

`mu`'s update is Nielsen's (Madsen, Nielsen and Tingleff Section 3.2): with predicted reduction
`pred = 1/2 dot(delta, mu * delta - gvec)` and `rho = (Fold - Fnew) / pred`, a step is accepted
when `rho > 0` (written so `NaN` takes the reject branch: `rho > 0` does, `!(rho <= 0)` does not),
giving `mu *= max(1/3, 1 - (2 rho - 1)^3)` and resetting the growth factor `nu` to `2.0`; a
rejected step gives `mu *= nu` and `nu *= 2.0`. **The Jacobian is reassembled only on an accepted
step and reused across rejected ones**, since a rejected step did not move `y` -- the single most
expensive avoidable thing in this file.

`mu0 = opt.mu_init * s`, `s = ‖J u‖^2 / ‖u‖^2` a one-`jvp!` Rayleigh-quotient estimate of the mean
curvature of `JᵀJ` along one direction `u` drawn from `Xoshiro(opt.seed)` (`s == 0` falls back to
`mu = opt.mu_init`).

Termination, checked before each trial step: `‖grad_y O‖_inf <= opt.g_abstol` gives
`:gradient_norm`; after an accepted step, `‖delta‖ <= opt.x_reltol * (‖y‖ + opt.x_reltol)` gives
`:x_tolerance`; exhausting `opt.iterations` gives `:iterations`. **A rejected step whose predicted
reduction `pred` is at or below round-off (`2*pred <= 1e-12 * max(2F, 1.0)`) also gives
`:f_tolerance`** (D44): without this, a start already at a minimum (a restart at its own converged
theta) rejects every trial step forever, doubling `mu` without bound until an overflowed `mu`
reaches CGLS as a `NaN`. Every **accepted** iterate appends `(k, 𝒪, chi2, R)` to the trajectory,
`k` the outer loop's own counter, so a run with rejected steps has gaps in `k`.

**The stall test** (Phase 4.L spec D5 and D6, [OQ-62](../../dev/open_questions.md#oq-62)): with
`opt.lm_rtol > 0`, checked before each trial step, immediately after the gradient test, with `i` the
accepted steps so far (`length(trajectory) - 1`) and `w = opt.lm_rtol_window`: when `i >= w` and
`abs(trajectory[i + 1 - w].objective - trajectory[i + 1].objective) <= opt.lm_rtol *
abs(trajectory[i + 1].objective)`, the fit ends with `status = :lm_rtol` and `converged = true`: 𝒪
has moved by at most `lm_rtol` of itself over the last `w` accepted steps. The window counts accepted
steps, the trajectory's rows, so a run of rejected steps, which leaves 𝒪 unchanged, never passes for a
stall. Every other stop fires first, where it fires without the key: `:gradient_norm` at the same loop
top, `:x_tolerance` and `:f_tolerance` inside the previous trial step, `:iterations` at the loop's end;
so the test first holds right after an accepted step, and `result.iterations` is then that step's `k`,
the last row's `iteration`. The test evaluates nothing and changes nothing, so a fit it ends is the fit
without it, bitwise, up to that row. `lm_rtol = 0.0`, the default, never reads the trajectory.
`FitResult.termination` is `:none` for this stop, as for every `LevenbergMarquardt` stop.

`cgls_iterations`, given a `Vector{Int}`, has one entry per attempted outer iteration (accepted
or not) pushed onto it: the number of CGLS iterations that trial step took (D43 item 9's
measurement hook). `mu_trace`, given a `Vector{Float64}`, has one entry per attempted outer
iteration pushed onto it: `mu` **going into** that iteration's decision (so `mu_trace[k+1]`
against `mu_trace[k]` shows what iteration `k`'s accept/reject did to it), letting OP15 check
Nielsen's update directly. Neither is part of D28's own signature; Level 3's additions, flagged
under rule 11 zero, the same footing as `jacobian`'s own `chunk` keyword (chunk 6).

**The progress lines** (Phase 4.K spec D5 to D7, [OQ-61](../../dev/open_questions.md#oq-61)): with
`opt.progress = n > 0`, a header before the first Jacobian, a line at every recorded row (the start and
each accepted step) whose index is a multiple of `n`, and a final line once the `FitResult` is built,
each one `print` and one `flush` to `progress_io`, which is looked up at the call (`stdout` by default).
On a row, `it` is the outer counter `k`, with gaps where steps were rejected; `acc` the accepted steps
so far (the row's index) and `rej = k - acc` the rejected ones, so `acc + rej = it`; `mu` the damping
going into the next trial step; `gmax` `2 maximum(abs, gvec)`, the quantity the gradient stop compares
with `opt.g_abstol`. Nothing is evaluated for them that the fit does not already compute,
`_count_near_bound` aside, so every `FitResult` field but `elapsed` is the same, bitwise, with the lines
on and off.

Spec D28; Phase 4.K spec D5 to D7; Phase 4.L spec D5 and D6.
"""
function solve(problem::AbstractFitProblem, opt::LevenbergMarquardt;
    start::AbstractVector=flatten(problem.pspec),
    cgls_iterations::Union{Nothing,Vector{Int}}=nothing,
    mu_trace::Union{Nothing,Vector{Float64}}=nothing, progress_io::IO=stdout)
    t0 = time()
    y = Vector{Float64}(start)
    ntheta = length(y)
    opt.progress > 0 && _progress_header(progress_io, _optimizer_short_name(opt), problem,
        opt.iterations, opt.progress, opt.g_abstol)

    r, Jac = _residuals_jacobian(problem, y)
    F = 0.5 * sum(abs2, r)
    gvec = zeros(Float64, ntheta)
    jtvp!(gvec, problem, Jac, r)
    f_calls_count = 1
    g_calls_count = 1
    dg = opt.precondition ? _gauss_newton_floor(_gauss_newton_diagonal(problem, Jac)) : nothing

    rng = Xoshiro(opt.seed)
    u = randn(rng, ntheta)
    q0 = Vector{Float64}(undef, length(r))
    jvp!(q0, problem, Jac, u)
    s_est = dot(q0, q0) / dot(u, u)
    mu = s_est == 0 ? opt.mu_init : opt.mu_init * s_est
    nu = 2.0

    trajectory = _TRAJECTORY_ROW[(iteration=0.0, objective=2 * F, chi2=_chi2_from_r(problem, r),
        R=_R_value(problem, y))]
    opt.progress > 0 && _progress_line(progress_io, _optimizer_short_name(opt), problem, trajectory, 0,
        y, maximum(abs, gvec) * 2, t0; acc=0, rej=0, mu=mu)

    status = :iterations
    converged = false
    iters_run = 0

    for k in 1:opt.iterations
        if maximum(abs, gvec) * 2 <= opt.g_abstol
            status = :gradient_norm
            converged = true
            break
        end
        # The stall test (Phase 4.L spec D5, OQ-62 item 1): the objective's change over the last
        # lm_rtol_window accepted steps, two rows of the trajectory apart; off at lm_rtol = 0.0.
        if opt.lm_rtol > 0
            i = length(trajectory) - 1
            w = opt.lm_rtol_window
            if i >= w
                O_now = trajectory[i+1].objective
                O_then = trajectory[i+1-w].objective
                if abs(O_then - O_now) <= opt.lm_rtol * abs(O_now)
                    status = :lm_rtol
                    converged = true
                    break
                end
            end
        end
        iters_run = k
        mu_trace === nothing || push!(mu_trace, mu)   # mu going into this iteration's decision

        # Jacobi preconditioner for (JᵀJ + mu I): dg + mu, dg floored once per Jacobian build
        # (D22 item 3) -- rebuilt here every trial step since mu changes on both an accepted
        # and a rejected step, not only when Jac is rebuilt.
        precond = dg === nothing ? nothing : dg .+ mu
        delta, niter = _cgls(problem, Jac, r, gvec, mu, opt.cgls_tol, opt.cgls_maxiter; precond=precond)
        cgls_iterations === nothing || push!(cgls_iterations, niter)

        y_new = y .+ delta
        r_new = _full_residuals(problem, y_new)
        f_calls_count += 1
        Fnew = 0.5 * sum(abs2, r_new)
        pred = 0.5 * dot(delta, mu .* delta .- gvec)
        rho = (F - Fnew) / pred

        if rho > 0
            xtol_ok = norm(delta) <= opt.x_reltol * (norm(y_new) + opt.x_reltol)
            y = y_new
            r = r_new
            F = Fnew
            Jac = jacobian(problem, y)
            g_calls_count += 1
            jtvp!(gvec, problem, Jac, r)
            dg = opt.precondition ? _gauss_newton_floor(_gauss_newton_diagonal(problem, Jac)) : nothing
            mu *= max(1 / 3, 1 - (2 * rho - 1)^3)
            nu = 2.0

            push!(trajectory, (iteration=Float64(k), objective=2 * F,
                chi2=_chi2_from_r(problem, r), R=_R_value(problem, y)))
            if opt.progress > 0
                i = length(trajectory) - 1
                i % opt.progress == 0 && _progress_line(progress_io, _optimizer_short_name(opt), problem,
                    trajectory, i, y, maximum(abs, gvec) * 2, t0; acc=i, rej=k - i, mu=mu)
            end

            if xtol_ok
                status = :x_tolerance
                converged = true
                break
            end
        else
            if 2 * pred <= 1e-12 * max(2 * F, 1.0)
                # The model predicts no decrease above round-off (D44): every trial step from
                # here is rejected and mu would otherwise double without bound (33 rejections
                # reach mu = 9e161 on the MC fixture, and the next CGLS's 2000-iteration cap
                # returns a NaN). theta is already a minimum to CGLS's own precision.
                status = :f_tolerance
                converged = true
                break
            end
            mu *= nu
            nu *= 2.0
        end
    end

    elapsed = time() - t0
    result = _build_result(problem, y, 2 * F, trajectory, converged, status, iters_run,
        f_calls_count, g_calls_count, elapsed, "levenberg_marquardt")
    opt.progress > 0 && _progress_final(progress_io, result)
    return result
end

Base.show(io::IO, opt::LevenbergMarquardt) = print(io, "LevenbergMarquardt(iterations = ",
    opt.iterations, ", g_abstol = ", opt.g_abstol, ", mu_init = ", opt.mu_init, ")")

# ------------------------------------------------------------------------------------------
# D29: NoisyGradientDescent
# ------------------------------------------------------------------------------------------

"""
    NoisyGradientDescent(; steps = 2000, step_size = 1.0e-1, noise = 0.1, noise_decay = 500.0,
                          seed = 1, progress = 0)

Normalized gradient descent on `𝒪` in the unconstrained `y` of the logit transform, bumped by
Gaussian noise of decaying amplitude, the explorer of foo2026 Section III.D's two-phase routine
([OQ-20](../../dev/open_questions.md#oq-20), [OQ-37](../../dev/open_questions.md#oq-37)):

    ghat_k  = grad_y O(y_k) / ||grad_y O(y_k)||_2      (the raw gradient if the norm underflows)
    y_{k+1} = y_k - alpha * ghat_k + s0 * exp(-k / tau) * xi_k,   xi_k ~ N(0, I),

`alpha = step_size`, `s0 = noise`, `tau = noise_decay` in steps, `seed` for `Xoshiro`. **Not
SGLD, and never called SGLD** (`conventions.md` Section 11): the gradient here is the full
`grad_y O`, never a minibatch estimate, and the noise decays on a fixed schedule of its own
rather than being tied to the step size the way SGLD's is -- this method aims at the basin, not
at sampling the posterior, and makes no sampling claim. Writing `s_k^2 = 2 alpha T_k` identifies a
temperature `T_k` decaying to zero; that is a remark for the reader, not a mechanism the code
implements.

**`steps`, `noise`, `noise_decay` and `seed` are [OQ-22](../../dev/open_questions.md#oq-22)'s
deck sketch, verbatim, and stay provisional** ([OQ-20](../../dev/open_questions.md#oq-20)).

**`step_size` is a step *length* in `y` (Step 3.C.3, D38), not a multiplier on the raw gradient.**
Before this step, `alpha` multiplied `grad_y O` directly, so its natural scale was a property of
the problem's χ² and bound widths, not a length -- which is why the old default (`1.0e-8`,
measured 2026-09-14, D29, D43 item 3) moved P1 (D46) at all: `|grad_y O|` there is of order `1e6`
(`d(chi2)/dy = d(chi2)/dx * dx/dy`, and `dx/dy` carries the bound width, e.g. 750 eV for `T_D` on
`(50, 800)`), so `1e-8 * 1e6` is an `O(1e-2)` step. **The old default must not survive the
normalization**: `1.0e-8` as a step *length* is a fit that never moves, on any problem, regardless
of its gradient's raw scale -- exactly the regression OP19 exists to catch. `step_size = 1.0e-1` is
the value Step 3.C.3's sweep chose (measurement 8, `dev/reports/phase3C_report.md`): P1's realistic
fixture converges to `chi2_end/chi2_start` of order `1e-7` to `1e-5` across the whole sweep
(`[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]`), essentially indifferent to `step_size` in that range, but
the cut-down multi-flow problem -- the sweep's stand-in for a realistically-sized fit -- needs a
step at least of order `1e-2` to move meaningfully in `steps = 2000` at all (`1e-3`/`3e-3` leave it
at `0.91`/`0.91` of its start), and `1e-1` is the best value measured on it (`0.0084`, against
`0.125` at `3e-2` and `0.0090` at `3e-1`); see that report for the full sweep table.

**The guard**: if `||grad_y O||_2` is zero, sub-normal (`<= floatmin(Float64)`) or non-finite, the
raw gradient is used in its place, which at an exactly-zero gradient is itself a zero step -- the
noise kick still moves the iterate (`conventions.md` 5.14: both branches return the same type).

**`progress`** (Phase 4.K spec D4, [OQ-61](../../dev/open_questions.md#oq-61)): `0`, the default, prints
nothing; `n > 0` makes `solve` print a header, a line at every `n`th step (the start included) and a
final line to its `progress_io`, each flushed (`solve`'s docstring). Refused below 0.

Spec D29, D38; Phase 4.K spec D4.
"""
Base.@kwdef struct NoisyGradientDescent <: AbstractOptimizer
    steps::Int = 2000              # provisional
    step_size::Float64 = 1.0e-1    # alpha, a step length in y -- Step 3.C.3's sweep, D38
    noise::Float64 = 0.1           # provisional: s0
    noise_decay::Float64 = 500.0   # provisional: tau, in steps
    seed::Int = 1                  # provisional
    progress::Int = 0              # Phase 4.K: print a line every `progress` steps, 0 = never
    function NoisyGradientDescent(steps, step_size, noise, noise_decay, seed, progress)
        progress >= 0 || throw(ArgumentError(
            "NoisyGradientDescent: progress must be an integer ≥ 0; got $progress"))
        return new(steps, step_size, noise, noise_decay, seed, progress)
    end
end

"""
    solve(problem::FitProblem, opt::NoisyGradientDescent; start = flatten(problem.pspec),
          y_trace::Union{Nothing,Vector} = nothing, progress_io::IO = stdout) -> FitResult

Runs `opt.steps` steps of the update in `NoisyGradientDescent`'s own docstring, seeded from
`Xoshiro(opt.seed)` so two runs are bitwise identical (OP6). **Returns the best iterate seen,
including the start** (`status = :steps`, `converged = false` always: it is an explorer, not a
converger, and a run that only makes things worse returns the start unchanged, D29). The
trajectory records **every** step, `opt.steps + 1` entries, improving or not. **The kicks are
added to `y`, never to `x`**, so no iterate can leave its bounds and the optimizer needs no
projection (Pitfall 4/13, OP9); with `opt.noise == 0.0` this is plain gradient descent and
decreases `𝒪` at every step for a small enough `opt.step_size` (OP5).

`y_trace`, given a `Vector`, has one entry per step pushed onto it -- `y` **after** that step's
update, so `opt.steps` entries in all -- on the same footing as `LevenbergMarquardt`'s `mu_trace`
and `jacobian`'s `chunk`: a test hook that changes nothing about the default call shape (D29).
It exists because `FitResult.trajectory` deliberately carries only `(iteration, 𝒪, chi2, R)` and
never `y` or `x`, so without it there is no way to see the optimizer's own iterates from outside
-- OP9 reads this to check that every iterate the optimizer actually takes stays inside its
bounds, which replaying the update formula in a test can't do (Pitfall 13).

**The progress lines** (Phase 4.K spec D5 to D7, [OQ-61](../../dev/open_questions.md#oq-61)): with
`opt.progress = n > 0`, a header before the first evaluation (no `g_abstol`: the explorer has no
gradient stop), a line at every step whose index is a multiple of `n` (the start included) and a final
line once the `FitResult` is built, each one `print` and one `flush` to `progress_io`, which is looked
up at the call (`stdout` by default). `gmax` is `maximum(abs, g)` and `nb` the near-bound count, both at
the step's own iterate; `best` the best objective so far; `noise` the `s_k` of the step just taken (`0`
at the start). The trajectory is preallocated, so a line reads its row by index, never `trajectory[end]`.
Nothing is evaluated for them that the fit does not already compute, `_count_near_bound` aside, and no
random number is drawn, so every `FitResult` field but `elapsed` is the same, bitwise, with the lines on
and off.

Spec D29; Phase 4.K spec D5 to D7.
"""
function solve(problem::AbstractFitProblem, opt::NoisyGradientDescent;
    start::AbstractVector=flatten(problem.pspec),
    y_trace::Union{Nothing,Vector}=nothing, progress_io::IO=stdout)
    t0 = time()
    ntheta = problem.pspec.ntheta
    y = Vector{Float64}(start)
    g = zeros(Float64, ntheta)
    xi = zeros(Float64, ntheta)
    rng = Xoshiro(opt.seed)

    trajectory = Vector{_TRAJECTORY_ROW}(undef, opt.steps + 1)
    opt.progress > 0 && _progress_header(progress_io, _optimizer_short_name(opt), problem, opt.steps,
        opt.progress, nothing)

    Ov, chi2v, Rval = _eval_all!(g, problem, y)
    trajectory[1] = (iteration=0.0, objective=Ov, chi2=chi2v, R=Rval)
    best_obj = Ov
    best_y = copy(y)
    opt.progress > 0 && _progress_line(progress_io, _optimizer_short_name(opt), problem, trajectory, 0,
        y, maximum(abs, g), t0; best=best_obj, noise=0.0)

    for k in 1:opt.steps
        randn!(rng, xi)
        sk = opt.noise * exp(-(k - 1) / opt.noise_decay)
        gn = sqrt(sum(abs2, g))
        ghat = (isfinite(gn) && gn > floatmin(Float64)) ? g ./ gn : g
        y = y .- opt.step_size .* ghat .+ sk .* xi
        y_trace === nothing || push!(y_trace, copy(y))

        Ov, chi2v, Rval = _eval_all!(g, problem, y)
        trajectory[k+1] = (iteration=Float64(k), objective=Ov, chi2=chi2v, R=Rval)
        if Ov < best_obj
            best_obj = Ov
            best_y = copy(y)
        end
        opt.progress > 0 && k % opt.progress == 0 && _progress_line(progress_io,
            _optimizer_short_name(opt), problem, trajectory, k, y, maximum(abs, g), t0;
            best=best_obj, noise=sk)
    end

    elapsed = time() - t0
    result = _build_result(problem, best_y, best_obj, trajectory, false, :steps, opt.steps,
        opt.steps + 1, opt.steps + 1, elapsed, "noisy_descent")
    opt.progress > 0 && _progress_final(progress_io, result)
    return result
end

Base.show(io::IO, opt::NoisyGradientDescent) = print(io, "NoisyGradientDescent(steps = ",
    opt.steps, ", step_size = ", opt.step_size, ", noise = ", opt.noise, ")")

# ------------------------------------------------------------------------------------------
# D30: OptimizerSequence, solve, and the four procedures
# ------------------------------------------------------------------------------------------

_optimizer_short_name(::LBFGS) = "lbfgs"
_optimizer_short_name(::LevenbergMarquardt) = "levenberg_marquardt"
_optimizer_short_name(::NoisyGradientDescent) = "noisy_descent"

# Phase 4.K spec D8 (OQ-61 item 5): whether an optimizer prints progress lines, which decides whether
# _staged_fit prints a stage's banner. A sequence's method is below, after its type.
_progress_on(opt::LBFGS) = opt.progress > 0
_progress_on(opt::LevenbergMarquardt) = opt.progress > 0
_progress_on(opt::NoisyGradientDescent) = opt.progress > 0

"""
    OptimizerSequence(members::AbstractOptimizer...)

Runs `members` in order, each warm-started from the previous member's own minimizer -- its
**best** iterate, not necessarily its last one (D29) -- and returns one `FitResult` whose
`trajectory` is the members' own trajectories concatenated, with the iteration counter continuing
across the join, and whose `converged`/`status`/`termination` are the **last** member's (D30, D13).
`iterations` is the length of the concatenated trajectory minus one, and `f_calls`/`g_calls` are
totals across every member: D30 states only that `converged`/`status` follow the last member, so
this is Level 3's reading of the rest, flagged under rule 11 zero.

Spec D30, D13.
"""
struct OptimizerSequence{T<:Tuple} <: AbstractOptimizer
    members::T
end
OptimizerSequence(members::AbstractOptimizer...) = OptimizerSequence(members)

function _optimizer_short_name(opt::OptimizerSequence)
    return join((_optimizer_short_name(m) for m in opt.members), "_")
end

_progress_on(opt::OptimizerSequence) = any(_progress_on, opt.members)

"""
    solve(problem::FitProblem, opt::OptimizerSequence; start = flatten(problem.pspec),
          progress_io::IO = stdout) -> FitResult

Runs `opt.members` in order (D30), each from the previous member's `FitResult.theta` (its best
iterate). OP10 is the row this warm-start discipline is written for: the second member's first
trajectory entry is the first member's **best**, not its last.

**The progress lines** (Phase 4.K spec D8, [OQ-61](../../dev/open_questions.md#oq-61) item 4):
`progress_io` (looked up at the call, `stdout` by default) is passed to every member, and each member
with `progress > 0` prints its own header, lines and final line under its own name, its `it` the
member's own row index, not the combined trajectory's offset iteration. The sequence prints nothing of
its own.

Spec D30; Phase 4.K spec D8.
"""
function solve(problem::AbstractFitProblem, opt::OptimizerSequence;
    start::AbstractVector=flatten(problem.pspec), progress_io::IO=stdout)
    t0 = time()
    combined = _TRAJECTORY_ROW[]
    current_start = Vector{Float64}(start)
    offset = 0.0
    total_f_calls = 0
    total_g_calls = 0
    local result::FitResult
    for member in opt.members
        result = solve(problem, member; start=current_start, progress_io=progress_io)
        for entry in result.trajectory
            push!(combined, (iteration=entry.iteration + offset, objective=entry.objective,
                chi2=entry.chi2, R=entry.R))
        end
        offset = combined[end].iteration + 1.0
        total_f_calls += result.f_calls
        total_g_calls += result.g_calls
        current_start = result.theta
    end
    elapsed = time() - t0
    name = _optimizer_short_name(opt)
    return _build_result(problem, result.theta, result.objective, combined, result.converged,
        result.status, length(combined) - 1, total_f_calls, total_g_calls, elapsed, name;
        termination=result.termination)
end

Base.show(io::IO, opt::OptimizerSequence) = print(io, "OptimizerSequence(",
    join((_optimizer_short_name(m) for m in opt.members), " -> "), ")")

"""
    _PROCEDURE_NAMES

The deck's four `procedure` names (D30, [OQ-20](../../dev/open_questions.md#oq-20)):
`"noisy_descent_lbfgs"` (the paper's routine, default), `"noisy_descent"`, `"lbfgs"`,
`"levenberg_marquardt"`. `_PROCEDURE_BASE_NAMES` is the three that may also appear as a list,
building an `OptimizerSequence` of them in order.
"""
const _PROCEDURE_NAMES = ("noisy_descent_lbfgs", "noisy_descent", "lbfgs", "levenberg_marquardt")
const _PROCEDURE_BASE_NAMES = ("noisy_descent", "lbfgs", "levenberg_marquardt")

"""
    _build_procedure(procedure; noisy_descent = NoisyGradientDescent(), lbfgs = LBFGS(),
                      levenberg_marquardt = LevenbergMarquardt()) -> AbstractOptimizer

Builds the optimizer D30's table names for `procedure`: one of the four names in
`_PROCEDURE_NAMES`, or a list of `_PROCEDURE_BASE_NAMES` combined into an `OptimizerSequence` in
order. The per-member options come from the matching keyword, each defaulting to its `@kwdef`
struct's own defaults. An unknown name is a loud error listing the four (D39). Not given an
explicit signature by D30 (only the deck's own keys and the table are specified); Level 3's
choice, flagged under rule 11 zero -- Step 3.6a's deck layer calls this once it has parsed the
three option tables.

Spec D30.
"""
function _build_procedure(procedure; noisy_descent::NoisyGradientDescent=NoisyGradientDescent(),
    lbfgs::LBFGS=LBFGS(), levenberg_marquardt::LevenbergMarquardt=LevenbergMarquardt())
    if procedure isa AbstractString
        procedure == "noisy_descent_lbfgs" && return OptimizerSequence(noisy_descent, lbfgs)
        procedure == "noisy_descent" && return noisy_descent
        procedure == "lbfgs" && return lbfgs
        procedure == "levenberg_marquardt" && return levenberg_marquardt
        throw(ArgumentError("unknown procedure \"$procedure\"; valid values are $_PROCEDURE_NAMES"))
    end
    members = AbstractOptimizer[]
    for name in procedure
        if name == "noisy_descent"
            push!(members, noisy_descent)
        elseif name == "lbfgs"
            push!(members, lbfgs)
        elseif name == "levenberg_marquardt"
            push!(members, levenberg_marquardt)
        else
            throw(ArgumentError("unknown procedure \"$name\"; valid values are $_PROCEDURE_NAMES"))
        end
    end
    return OptimizerSequence(members...)
end
