# The L-curve's corner: the maximum-curvature estimate of the global weight Lambda over a weight
# scan (Hansen and O'Leary 1993) -- Step 4.C.1, Phase 4.C spec D4-D8, OQ-46.
#
# A REPORT beside the scan, never a choice (OQ-19, WS7): nothing in the package calls this
# function -- not solve, not weight_scan, not fit(deck), not a script that then sets a deck's
# weight -- and WeightScanResult gains no field. The caller asks for it after a scan and reads
# the estimate, its bracket and every curvature; the weight is still the user's to choose.

using Printf: @printf, @sprintf

"""
    LCurveCorner

The outcome of `lcurve_corner` (Phase 4.C spec D6): the maximum-curvature estimate of the global
weight Λ over one weight scan, **a report and never a choice** (OQ-19, OQ-46).

- `status`: `:corner` when some interior scan point has positive curvature, else `:no_corner`.
- `weight`: `Λ_{k*}`, the scan weight at the corner; `nothing` when `status === :no_corner`.
- `index`: `k*`, its position in `weights`; `nothing` when there is no corner.
- `lower`, `upper`: the bracketing scan weights `Λ_{k*−1}` and `Λ_{k*+1}`, between which the
  continuous curve's corner is estimated to lie; `nothing` when there is no corner.
- `at_edge`: `k* == 2` or `k* == n − 1`, the first or last interior point, where the corner may
  lie outside the scan's range; `false` when there is no corner.
- `curvature`: the signed Menger curvature `κ_k` at every scan point, `NaN` at `k = 1` and
  `k = n` (no curvature there) and at an interior point one of whose sides has zero length.
- `weights`: the scan's weights, ascending, exactly as given.
- `bracket_converged`: whether the member fits at `k* − 1`, `k*` and `k* + 1` all converged;
  `nothing` when the caller supplied no convergence flags or there is no corner.

**Λ carries units** (ns⁴ for a `D2` term under a trace-scaled normalization, ns² for `D1`,
dimensionless for `I` and for `unit`), so an estimate is a property of one problem and one term
set: comparable across scans of the same deck, never across operators or decks.

Phase 4.C spec D6.
"""
struct LCurveCorner
    status::Symbol
    weight::Union{Nothing,Float64}
    index::Union{Nothing,Int}
    lower::Union{Nothing,Float64}
    upper::Union{Nothing,Float64}
    at_edge::Bool
    curvature::Vector{Float64}
    weights::Vector{Float64}
    bracket_converged::Union{Nothing,Bool}
end

"""
    lcurve_corner(weights, chi2, R; converged = nothing) -> LCurveCorner
    lcurve_corner(scan::WeightScanResult) -> LCurveCorner

The maximum-curvature estimate of the global weight Λ over a weight scan (Hansen and O'Leary
1993; [OQ-46](../../dev/open_questions.md#oq-46)), **a report beside the scan and never a
choice**: nothing in the package calls it, and the weight a deck carries stays the user's to set
([OQ-19](../../dev/open_questions.md#oq-19)).

With the scan's points `P_k = (log chi2_k, log R_k)` (natural logarithms), `k = 1..n`, `weights`
strictly ascending, the curvature at each interior point is the signed Menger curvature of
`P_{k−1}, P_k, P_{k+1}`:

    κ_k = 2 (d₁ × d₂) / (|d₁| |d₂| |d₁ + d₂|),   d₁ = P_k − P_{k−1},  d₂ = P_{k+1} − P_k,
    d₁ × d₂ = d₁ₓ d₂ᵧ − d₁ᵧ d₂ₓ,

positive where the curve turns counterclockwise, which is the L's elbow (the tangent turning
from downward, `R` falling at nearly constant `chi2`, to rightward, `chi2` rising at nearly
constant `R`). The estimate is the interior point of largest positive `κ_k`, a tie going to the
smaller Λ (the less regularized fit); no interior `κ_k > 0` is **no corner**, returned as
`status = :no_corner` with no number. The estimate is a bracket `[lower, upper]`, not a point,
and at the first or last interior point `at_edge` says the corner may lie outside the scan.

The curvature depends only on the points' order, not on the weights' values, and is unchanged
by a common rescaling of `chi2` or `R` (a shift in the log plane) up to round-off, so the raw
`chi2`/`R` and the normalized `chi2_normalized`/`R_normalized` give the same estimate.

`converged`, when given, is one `Bool` per scan point; `bracket_converged` reports whether the
three members around the corner converged (a corner bracketed by a member that did not is
reported with that flag, never dropped). The `WeightScanResult` method passes each member fit's
own `converged`.

Refused loudly (rule 5.9): fewer than three points; vectors of different lengths; weights not
strictly ascending or not positive and finite; any `chi2` or `R` not positive and finite.

**Λ carries units**: see `LCurveCorner`.

Phase 4.C spec D4-D7; `docs/derivations/regularizer.md` Section 7.
"""
function lcurve_corner(weights::AbstractVector{<:Real}, chi2::AbstractVector{<:Real},
                       R::AbstractVector{<:Real}; converged::Union{Nothing,AbstractVector{Bool}}=nothing)
    n = length(weights)
    n >= 3 || throw(ArgumentError(
        "lcurve_corner: a curvature needs at least three scan points; got $n"))
    (length(chi2) == n && length(R) == n) || throw(ArgumentError(
        "lcurve_corner: weights, chi2 and R must have one entry per scan point; got lengths " *
        "$n, $(length(chi2)) and $(length(R))"))
    (converged === nothing || length(converged) == n) || throw(ArgumentError(
        "lcurve_corner: converged must have one entry per scan point; got $(length(converged)) for $n points"))
    w = collect(Float64, weights)
    for k in 1:n
        (isfinite(w[k]) && w[k] > 0) || throw(ArgumentError(
            "lcurve_corner: weights must be positive and finite; weights[$k] = $(w[k])"))
        (k == 1 || w[k] > w[k-1]) || throw(ArgumentError(
            "lcurve_corner: weights must be strictly ascending; weights[$(k-1)] = $(w[k-1]), weights[$k] = $(w[k])"))
        (isfinite(chi2[k]) && chi2[k] > 0) || throw(ArgumentError(
            "lcurve_corner: chi2 must be positive and finite; chi2[$k] = $(chi2[k])"))
        (isfinite(R[k]) && R[k] > 0) || throw(ArgumentError(
            "lcurve_corner: R must be positive and finite; R[$k] = $(R[k])"))
    end

    x = [log(Float64(c)) for c in chi2]
    y = [log(Float64(r)) for r in R]
    kappa = fill(NaN, n)
    for k in 2:(n-1)
        d1x = x[k] - x[k-1]
        d1y = y[k] - y[k-1]
        d2x = x[k+1] - x[k]
        d2y = y[k+1] - y[k]
        cross = d1x * d2y - d1y * d2x
        denom = hypot(d1x, d1y) * hypot(d2x, d2y) * hypot(d1x + d2x, d1y + d2y)
        kappa[k] = denom > 0 ? 2 * cross / denom : NaN
    end

    kstar = 0
    best = 0.0
    for k in 2:(n-1)
        if kappa[k] > best   # strict: a tie keeps the first, the smaller weight; NaN never wins
            best = kappa[k]
            kstar = k
        end
    end

    kstar == 0 && return LCurveCorner(:no_corner, nothing, nothing, nothing, nothing, false,
        kappa, w, nothing)
    bracket = converged === nothing ? nothing : all(converged[(kstar-1):(kstar+1)])
    return LCurveCorner(:corner, w[kstar], kstar, w[kstar-1], w[kstar+1],
        kstar == 2 || kstar == n - 1, kappa, w, bracket)
end

lcurve_corner(scan::WeightScanResult) =
    lcurve_corner(scan.weights, scan.chi2, scan.R; converged=Bool[r.converged for r in scan.results])

function Base.show(io::IO, c::LCurveCorner)
    if c.status === :corner
        print(io, "LCurveCorner(Lambda = ", c.weight, " in [", c.lower, ", ", c.upper, "]",
            c.at_edge ? ", at the scan's edge" : "", ")")
    else
        print(io, "LCurveCorner(no corner over ", length(c.weights), " weights)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", c::LCurveCorner)
    println(io, "LCurveCorner")
    if c.status === :corner
        @printf(io, "  corner: Lambda = %.4g (scan point %d), bracket [%.4g, %.4g]\n",
            c.weight, c.index, c.lower, c.upper)
        c.at_edge && println(io, "  at the scan's edge: the corner may lie outside the scan's range")
        c.bracket_converged === nothing ||
            println(io, "  bracketing fits converged: ", c.bracket_converged)
    else
        println(io, "  no corner: no interior scan point has positive curvature")
    end
    for (k, (w, kap)) in enumerate(zip(c.weights, c.curvature))
        @printf(io, "  Lambda = %.4g: curvature %s%s\n", w, isnan(kap) ? "NaN" : @sprintf("%+.6g", kap),
            c.index == k ? "  <- corner" : "")
    end
    print(io, "  (Lambda is in the regularizer's own units; an estimate is comparable only across scans of the same deck)")
end
