# The objective, the combined evaluator Optim will call, and FitResult -- Step 3.4a, chunk 8 of
# Phase 3. Spec D26. Optim itself joins at chunk 9 (Step 3.4b): nothing here needs it.
#
# The one thing this file must get right: objective and _objective_and_gradient! must agree with
# each other, and with the separate chi2/gradient! pieces, bitwise (OB3) -- so both are written
# directly on top of _full_residuals (residuals.jl) and jacobian/jtvp! (jacobian.jl), never as an
# independent recomputation of chi2 + Lambda * R.

"""
    objective(problem::FitProblem, theta::AbstractVector) -> Real

`𝒪 = chi2 + Lambda * R = ‖r_full‖²` (D25, D26; foo2026 Eq. 8): the sum of squares of every
residual row, data and regularizer alike, from `_full_residuals` (`residuals.jl`). `Lambda` is
`problem.regularizer.weight`; it is already folded into each regularizer row's own
`sqrt(Lambda * w_term)` factor (D25), so this one sum equals `chi2(problem, theta) +
problem.regularizer.weight * R` to round-off, without ever adding the two halves separately (OB1,
OB2). `R` itself excludes `Lambda` (D25) and is not returned here; `chi2`, `R` and `Lambda` are
split apart again only when a `FitResult` is built (Step 3.4b).

**`𝒪` is −2 log posterior up to a constant** ([OQ-20](../../dev/open_questions.md#oq-20)).
`sigma` is frozen (D16, [OQ-14](../../dev/open_questions.md#oq-14)), so `chi2` is a true
`-2 * log likelihood` and `Lambda * R` a Gaussian-prior penalty in the same units; `∇_theta 𝒪`
(`_objective_and_gradient!`, below) is exactly `∇_y 𝒪` in the unconstrained `y` an HMC sampler
would work in. **One trap for Step 4.4, recorded here and built nowhere else**
([OQ-24](../../dev/open_questions.md#oq-24)): a sampler that works in `y`-space with a prior
uniform in the bounded `x` must add the logit's log-Jacobian, `sum(log(abs(dx/dy)))` (D3) over
every fitted entry, to the log-density -- a term this function, and every optimizer that calls
it, rightly omits, since a minimizer wants only the mode, never a normalized density.

Spec D26.
"""
function objective(problem::AbstractFitProblem, theta::AbstractVector)
    return sum(abs2, _full_residuals(problem, theta))
end

"""
    _objective_and_gradient!(g::AbstractVector, problem::FitProblem, theta::AbstractVector) -> Real

The one evaluator D27's Optim `fg!(G, y)` wraps (Step 3.4b): builds `r_full` (`_full_residuals`)
and the Jacobian (`jacobian`) exactly once, fills `g` with `grad_theta O = 2 * J_fullᵀ * r_full`
(`jtvp!`, `jacobian.jl`), and returns `𝒪 = sum(abs2, r_full)`. Calling `objective` and
`gradient!` separately, as chunk 6's `gradient!` itself does internally, would rebuild `r_full`
and the model from `theta` a second time each -- this function is why a fit costs half of that
(D2, D26).

`𝒪` and `g` come out bitwise identical to `objective(problem, theta)` and `gradient!(g, problem,
theta)` (OB3): the residual vector and the Jacobian are computed the same way from the same
`theta`, just once instead of twice.

Spec D26.
"""
function _objective_and_gradient!(g::AbstractVector, problem::AbstractFitProblem, theta::AbstractVector)
    r = _full_residuals(problem, theta)
    Jac = jacobian(problem, theta)
    jtvp!(g, problem, Jac, r)
    g .*= 2
    return sum(abs2, r)
end

# ------------------------------------------------------------------------------------------
# ChannelSummary, FitResult
# ------------------------------------------------------------------------------------------

"""
    ChannelSummary

One channel's own diagnostics from a `FitResult` (Step 4.A.3a, D36): `name` and `weight` (`w_c`,
`1.0` for a single-channel fit); `chi2`, the **unweighted** `chi2(channel_problem, theta)`
(Pitfall 26 -- `FitResult.chi2` is the weighted total `sum(w_c * chi2_c)`, what the optimizer
minimized); `npixels`, the channel's own pixel count; `nparams` (`p_c`), the count of `theta`
entries this channel's Jacobian touches: `pspec.ntheta` minus the global entries whose variable
has at least one binding, **every** one of which is an `:effect` binding of a different channel
(a per-bin entry always counts, since a time-varying variable can only reach the shared plasma);
`reduced_chi2 = chi2 / (npixels - nparams)`, `NaN` when `npixels <= nparams`; `reduced_chi2_simple
= chi2 / npixels`, `NaN` when `npixels == 0`.

For a single-channel fit (the 20-argument `FitResult` constructor, D36) `nparams == ntheta`, since
no binding of a one-channel spec belongs to "a different channel".

Spec D36.
"""
struct ChannelSummary
    name::String
    weight::Float64
    chi2::Float64
    npixels::Int
    nparams::Int
    reduced_chi2::Float64
    reduced_chi2_simple::Float64
end

"""
    _channel_nparams(pspec::ParameterSpec, channel::Int) -> Int

`p_c` (D36): `pspec.ntheta` minus the count of **global** entries (`pspec.global_entries`, one per
scalar fitted variable) whose variable has at least one binding and every one of whose bindings is
an `:effect` binding of a channel other than `channel`. A variable with no binding at all is not
excluded (there is nothing to prove it belongs elsewhere); a per-bin entry is never excluded, since
a time-varying variable reaches only the shared plasma, every channel's own. Computed from the
bindings alone, with no Jacobian call.

Spec D36.
"""
function _channel_nparams(pspec::ParameterSpec, channel::Int)
    excluded = 0
    for (k, _) in enumerate(pspec.global_entries)
        i = pspec.global_slots[k]
        bs = [b for b in pspec.bindings if b.slot == i]
        isempty(bs) && continue
        all(b -> b.kind === :effect && b.channel != channel, bs) && (excluded += 1)
    end
    return pspec.ntheta - excluded
end

"""
    FitResult{FT<:Real,E<:Experiment,P<:ParameterSpec}

The outcome of one fit (D26): everything chunks 9 to 13 fill in or read back out.

- `theta`: the minimizer, in the unconstrained `y` an optimizer works in.
- `experiment`: `unflatten(problem.experiment, pspec, theta)`, the fitted model; `experiments[1]`
  (Step 4.A.3a, D36).
- `pspec`: the `ParameterSpec` the fit was run against.
- `values`: **every fitted variable and every dependent**, by name, as its fitted trace in its
  own unit (a scalar variable's entry is a length-1 vector; a user who fits `logn` reads `ne_cm3`
  here too, D26).
- `near_bound`: D13's bound-proximity flag, one `BitVector` per **variable** (never a dependent,
  which has no bound of its own), aligned with `values[name]`.
- `chi2`: the data term -- for a `JointFitProblem` the **weighted** total `sum(w_c * chi2_c)`,
  what the optimizer minimized (Pitfall 26); `sum(abs2, residuals(problem, theta))` for one
  channel.
- `R`: the regularizer's value, **excluding** `Lambda` (D25).
- `objective`: `chi2 + weight * R` (`==  objective(problem, theta)`).
- `weight`: `Lambda`, the global regularizer weight the fit ran at (`0.0` with no regularizer).
- `term_values`: `w_term * ‖op N(u-ref)‖²` per regularizer term, in term order; sums to `R`.
- `npixels`: `problem.npixels` (the sum over channels for a `JointFitProblem`).
- `trajectory`: one entry per recorded iteration, `(iteration, objective, chi2, R)`.
- `converged`, `status`, `iterations`, `f_calls`, `g_calls`: the optimizer's own report.
- `termination`: Optim's own finer status for `LBFGS` (`Symbol(string(Optim.termination_code(res)))`,
  e.g. `:GradientNorm`, `:SmallObjectiveChange`, `:Iterations`), `:none` for the three optimizers
  TRUST writes itself, and the **last member's** for `OptimizerSequence` (Step 3.C.1, D13). Not
  written to the result file at this step (D17).
- `elapsed`: wall time of the fit, in seconds.
- `optimizer`: a short description of what ran, e.g. `"lbfgs"` or `"noisy_descent_lbfgs"`.
- `experiments` (Step 4.A.3a, D36): one fitted `Experiment` per channel, in channel order --
  `(experiment,)` for a single-channel fit.
- `channels` (Step 4.A.3a, D36): one `ChannelSummary` per channel, in channel order -- a single
  `ChannelSummary("main", 1.0, chi2, npixels, pspec.ntheta, ...)` for a single-channel fit.

**The 20-argument positional constructor (every field above except `experiments` and `channels`)
stays**, as an outer method filling `experiments = (experiment,)` and one `"main"` summary (D36):
`test_objective.jl`'s OB5/OB6 and `test_reproduce.jl`'s hand-built `FitResult`s keep compiling.

Spec D26, D13, D36.
"""
struct FitResult{FT<:Real,E<:Experiment,P<:ParameterSpec}
    theta::Vector{FT}
    experiment::E
    pspec::P
    values::Dict{String,Vector{Float64}}
    near_bound::Dict{String,BitVector}
    chi2::FT
    R::FT
    objective::FT
    weight::Float64
    term_values::Vector{Float64}
    npixels::Int
    trajectory::Vector{NamedTuple{(:iteration, :objective, :chi2, :R),NTuple{4,Float64}}}
    converged::Bool
    status::Symbol
    termination::Symbol
    iterations::Int
    f_calls::Int
    g_calls::Int
    elapsed::Float64
    optimizer::String
    experiments::Tuple
    channels::Vector{ChannelSummary}
end

function FitResult(theta::AbstractVector, experiment::Experiment, pspec::ParameterSpec,
    values::Dict{String,Vector{Float64}}, near_bound::Dict{String,BitVector}, chi2, R, objective,
    weight::Real, term_values::Vector{Float64}, npixels::Int, trajectory::Vector, converged::Bool,
    status::Symbol, termination::Symbol, iterations::Int, f_calls::Int, g_calls::Int,
    elapsed::Real, optimizer::AbstractString)
    p_main = pspec.ntheta
    reduced = npixels > p_main ? chi2 / (npixels - p_main) : NaN
    reduced_simple = npixels == 0 ? NaN : chi2 / npixels
    main_summary = ChannelSummary("main", 1.0, Float64(chi2), npixels, p_main, Float64(reduced),
        Float64(reduced_simple))
    return FitResult(theta, experiment, pspec, values, near_bound, chi2, R, objective, weight,
        term_values, npixels, trajectory, converged, status, termination, iterations, f_calls,
        g_calls, elapsed, optimizer, (experiment,), ChannelSummary[main_summary])
end

# --- show (spec D40, D26) ---

function Base.show(io::IO, result::FitResult)
    print(io, "FitResult(", result.optimizer, ", ", result.converged ? "converged" : "not converged",
        ", chi2 = ", result.chi2, ", R = ", result.R, ")")
end

"""
    Base.show(io, ::MIME"text/plain", result::FitResult)

Prints, in order (D26): the optimizer, `converged` and `status`, `iterations`, `chi2` with
`chi2/npixels`, `R`, `Lambda`, `objective`, the elapsed time, and then one line per variable that
has any `near_bound` flag set, naming the variable, the flagged bins and which bound each is
closest to (compared against `pspec.variables`' own bounds in the same unit as `values`). A
scalar variable is named without a bin, `name (scalar): near the lower bound` (Phase 4.N spec D21,
[OQ-65](../../dev/open_questions.md#oq-65) item 3), as `fit(deck)`'s near-bound warning names it. A
result with no flags prints nothing there.
"""
function Base.show(io::IO, ::MIME"text/plain", result::FitResult)
    println(io, "FitResult")
    println(io, "  optimizer: ", result.optimizer)
    println(io, "  converged: ", result.converged, " (status = :", result.status, ")")
    println(io, "  iterations: ", result.iterations)
    @printf(io, "  chi2: %.6g (chi2/npixels = %.6g)\n", result.chi2, result.chi2 / result.npixels)
    @printf(io, "  R: %.6g\n", result.R)
    @printf(io, "  weight (Lambda): %.6g\n", result.weight)
    @printf(io, "  objective: %.6g\n", result.objective)
    @printf(io, "  elapsed: %.6g s\n", result.elapsed)
    if length(result.channels) > 1
        println(io, "  channels: ", length(result.channels))
        for c in result.channels
            @printf(io, "    %s: weight = %.6g, chi2 = %.6g, npixels = %d, chi2/n = %.6g, chi2/(n-p) = %.6g\n",
                c.name, c.weight, c.chi2, c.npixels, c.reduced_chi2_simple, c.reduced_chi2)
        end
    end
    for v in result.pspec.variables
        nb = get(result.near_bound, v.name, nothing)
        (nb === nothing || !any(nb)) && continue
        vals = result.values[v.name]
        bins = findall(nb)
        which = [vals[j] - v.lower <= v.upper - vals[j] ? "lower" : "upper" for j in bins]
        if !v.time_varying
            println(io, "  ", v.name, " (scalar): near the ", only(which), " bound")
            continue
        end
        label = length(bins) == 1 ? "bin" : "bins"
        println(io, "  ", v.name, ": near the ", join(unique(which), "/"), " bound at ", label,
            " ", join(bins, ", "))
    end
end
