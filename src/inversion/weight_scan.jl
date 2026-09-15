# The weight scan: the L-curve as a diagnostic -- Step 3.5, chunk 10 of Phase 3, the second deep
# hold. Spec D31, OQ-19.
#
# Nothing in this file chooses a weight. The user decided 2026-09-12 (OQ-19) that weights are
# user-defined scalars and that the maximum-curvature estimate of Lambda_opt (Hansen and O'Leary
# 1993) and any multidimensional search over per-term weights are Step 4.3's, to be discussed
# with the user first -- Section 12 of the spec. This function reports the (chi2, R) trade-off at
# each of a user-given list of weights, normalized as foo2026 Fig. 1 does, and nothing more.
#
# Scans are not the solver's job (conventions.md 6.8): a run over many weights or noise
# realizations is launched from an external script or job array, never from inside the package
# -- nothing below spawns a process or a thread (WS10).

"""
    WeightScanResult

The outcome of `weight_scan` (D31): one row per requested weight, plus the unregularized fit
each row is normalized against.

- `weights`: the requested Lambda values, sorted ascending.
- `chi2`, `R`: the data term and the (Lambda-free, D25) regularizer value at each weight's own
  fit.
- `chi2_normalized`, `R_normalized`: `chi2 ./ chi2_unreg` and `R ./ R_unreg` -- foo2026 Fig. 1's
  x and y axes.
- `chi2_unreg`, `R_unreg`: the same two quantities at the unregularized fit's own minimizer (D25:
  `R` is well defined and evaluated there even though nothing there penalized it).
- `rss`, `rss_normalized`, `rss_unreg`: the per-parameter residual sum of squares of foo2026
  Eq. 13, by variable name, over that variable's own bins; empty `Dict`s when `weight_scan` was
  called with no `truth`. **A `rss_normalized` entry above one is a real outcome, not a failure**
  -- foo2026's own Table II reports `u_C` at 2.01.
- `results`: every regularized fit's own `FitResult`, in `weights` order.
- `unregularized`: the unregularized fit's own `FitResult`.

**No field here names an optimum** ([OQ-19](../../dev/open_questions.md#oq-19)): choosing a
weight from this table, by maximum curvature or otherwise, is Step 4.3's, not this one's (WS7).

Spec D31.
"""
struct WeightScanResult
    weights::Vector{Float64}
    chi2::Vector{Float64}
    R::Vector{Float64}
    chi2_normalized::Vector{Float64}
    R_normalized::Vector{Float64}
    chi2_unreg::Float64
    R_unreg::Float64
    rss::Dict{String,Vector{Float64}}
    rss_normalized::Dict{String,Vector{Float64}}
    rss_unreg::Dict{String,Float64}
    results::Vector{FitResult}
    unregularized::FitResult
end

"""
    _with_weight(problem::FitProblem, weight::Real) -> FitProblem

`problem`, with its `Regularizer`'s global weight (Lambda) replaced by `weight` and every term
otherwise unchanged -- the one thing `weight_scan` (D31) varies from one fit to the next.
Requires `problem.regularizer !== nothing`. Not part of D31's own signature; Level 3's helper,
flagged under rule 11 zero, the mechanism the whole scan is built from.
"""
function _with_weight(problem::FitProblem, weight::Real)
    reg = problem.regularizer
    newreg = Regularizer(reg.terms, Float64(weight), reg.nrows)
    return FitProblem(problem.experiment, problem.pspec, problem.data, newreg, problem.mask,
        problem.npixels, problem.sigma, problem.pixel_index, problem.scheduler)
end

"""
    weight_scan(problem::FitProblem, weights::AbstractVector{<:Real}, optimizer::AbstractOptimizer;
                start = flatten(problem.pspec), truth = nothing) -> WeightScanResult

The L-curve as a diagnostic (foo2026 Section IV.A, Fig. 1; [OQ-19](../../dev/open_questions.md#oq-19)).
Fits `problem` once with its regularizer turned off (`weight = 0.0`) to get the unregularized
minimizer `y⁰`, `chi2_unreg` and `R_unreg` -- foo2026 Fig. 1's denominator (D25: `R` is well
defined and evaluated at `y⁰` even though nothing there penalized it) -- then fits at each of
`weights`, sorted ascending, each warm-started from the previous fit's own minimizer, the first
from `y⁰` (foo2026 Section IV: "the unregularized fit was run first and then the best-fit
parameters from that fit were used to warm-start a number of regularized fits"). `weights` must
be positive and distinct, else a loud error (D39).

**It reports; it never picks** ([OQ-19](../../dev/open_questions.md#oq-19), the user,
2026-09-12): nothing here estimates a maximum-curvature Lambda_opt (Hansen and O'Leary 1993) or
searches over per-term weights -- both are **Step 4.3**, to be discussed with the user first, and
`WeightScanResult` has no field naming an optimum (WS7).

`truth`, given as a `Dict{String,Vector{Float64}}` of ground-truth traces in each named
variable's own unit, additionally fills `rss`/`rss_normalized`/`rss_unreg` with the per-parameter
residual sum of squares of foo2026 Eq. 13, `RSS_v = sum_j (u_true,j - u_fit,j)^2` over the
variable's own bins, normalized by the unregularized fit's own RSS for that variable. **A ratio
above one is a real outcome, not a failure**: foo2026's own Table II reports `u_C` at 2.01, a
parameter over-regularized at the paper's own chosen Lambda (WS6).

Every `FitResult` is kept (`results`, one per requested weight, and `unregularized`), because
Step 3.7's reproductions plot the fitted traces themselves, not just the scan table's numbers.

**Launch this from a script for a scan over many weights or noise realizations**
(`conventions.md` 6.8, [OQ-21](../../dev/open_questions.md#oq-21)): nothing in this function
spawns a process or a thread (WS10).

Spec D31.
"""
function weight_scan(problem::FitProblem, weights::AbstractVector{<:Real}, optimizer::AbstractOptimizer;
    start::AbstractVector=flatten(problem.pspec),
    truth::Union{Nothing,Dict{String,Vector{Float64}}}=nothing)
    problem.regularizer === nothing && throw(ArgumentError(
        "weight_scan: problem has no regularizer; build one with Regularizer(...) before scanning Lambda"))

    ws = collect(Float64, weights)
    (length(ws) == length(unique(ws)) && all(>(0), ws)) || throw(ArgumentError(
        "weight_scan: weights must be positive and distinct; got $ws"))
    sorted = sort(ws)

    unregularized = solve(_with_weight(problem, 0.0), optimizer; start=start)
    chi2_unreg = unregularized.chi2
    R_unreg = unregularized.R

    results = Vector{FitResult}(undef, length(sorted))
    current_start = unregularized.theta
    for (k, w) in enumerate(sorted)
        results[k] = solve(_with_weight(problem, w), optimizer; start=current_start)
        current_start = results[k].theta
    end

    chi2v = [r.chi2 for r in results]
    Rv = [r.R for r in results]
    chi2_normalized = chi2v ./ chi2_unreg
    R_normalized = Rv ./ R_unreg

    rss = Dict{String,Vector{Float64}}()
    rss_normalized = Dict{String,Vector{Float64}}()
    rss_unreg = Dict{String,Float64}()
    if truth !== nothing
        for (name, tvals) in truth
            rss_unreg[name] = sum(abs2, tvals .- unregularized.values[name])
            rss[name] = [sum(abs2, tvals .- r.values[name]) for r in results]
            rss_normalized[name] = rss[name] ./ rss_unreg[name]
        end
    end

    return WeightScanResult(sorted, chi2v, Rv, chi2_normalized, R_normalized, chi2_unreg, R_unreg,
        rss, rss_normalized, rss_unreg, results, unregularized)
end

# --- show (spec D40) ---

function Base.show(io::IO, scan::WeightScanResult)
    print(io, "WeightScanResult(", length(scan.weights), " weights, Lambda in [",
        first(scan.weights), ", ", last(scan.weights), "])")
end

function Base.show(io::IO, ::MIME"text/plain", scan::WeightScanResult)
    println(io, "WeightScanResult")
    println(io, "  weights: ", length(scan.weights), " (", first(scan.weights), " to ",
        last(scan.weights), ")")
    @printf(io, "  chi2_unreg: %.6g\n", scan.chi2_unreg)
    @printf(io, "  R_unreg: %.6g\n", scan.R_unreg)
    for (w, c, r) in zip(scan.weights, scan.chi2_normalized, scan.R_normalized)
        @printf(io, "  Lambda = %.4g: chi2/chi2_unreg = %.6g, R/R_unreg = %.6g\n", w, c, r)
    end
end
