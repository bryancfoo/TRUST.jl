# The AbstractVelocityDistribution interface: the shape contract, generic dfdx by ForwardDiff,
# and the generic cauchy_dfdv by quadrature (the fallback every new shape gets for free).
# See docs/derivations/shapes.md.

using QuadGK: quadgk
import ForwardDiff

"""
    AbstractVelocityDistribution

The abstract type of a dimensionless velocity-distribution *shape* `φ(x)`, along the
scattering direction k̂, in the reduced variable `x = (v − u)/v_th`. A shape carries no
density, flow or temperature — those belong to the `Species` that carries it — and every
concrete shape must satisfy the three normalizations

    ∫ φ(x) dx = 1,    ∫ x φ(x) dx = 0,    ∫ x² φ(x) dx = 1/2,

checked by the shared test helper `check_shape_normalizations` (`test/shape_contract.jl`).

A concrete shape `f::AbstractVelocityDistribution` is callable, `f(x)`, and every other part of
the interface below (`dfdx`, `cauchy_dfdv`, `moment`, `skewness`, `excess_kurtosis`,
`parameters`, `reconstruct`) has a generic fallback that needs only `f(x)`, so a new shape is
usable the moment it can be evaluated. Nothing in this interface may assume a shape is
analytic, symmetric, or has a closed-form derivative.

`conventions.md` 1.5, Section 4; `docs/derivations/shapes.md`.
"""
abstract type AbstractVelocityDistribution end

"""
    dfdx(f::AbstractVelocityDistribution, x) -> Number

The derivative `φ'(x)` of the shape `f`. The generic method differentiates the shape's own call
`f(x)` with ForwardDiff and works for any real `x`; a shape may override it with a closed form.
A complex `x` has no generic method — see the `Complex` method's docstring — because ForwardDiff
is real-only; a shape that needs `dfdx` below the real axis (for `cauchy_dfdv` at `Im ξ < 0`)
must supply its own complex-capable method.

`conventions.md` Section 4.
"""
dfdx(f::AbstractVelocityDistribution, x::Real) = ForwardDiff.derivative(f, x)

"""
    dfdx(f::AbstractVelocityDistribution, x::Complex)

Throws: the generic `dfdx` differentiates with ForwardDiff, which has no complex derivative.
A shape that must be evaluated at complex `x` (to support `cauchy_dfdv` at `Im ξ < 0`) needs its
own `dfdx` method for `Complex` arguments; `Maxwellian` is an example.
"""
dfdx(f::AbstractVelocityDistribution, x::Complex) = throw(ArgumentError(
    "dfdx($(typeof(f)), ::Complex) is not available: the generic derivative uses ForwardDiff, " *
    "which is real-only. A shape needs an explicit complex-capable dfdx method to support " *
    "cauchy_dfdv below the real axis (Im ξ < 0)."))

"""
    _d2fdx2(f::AbstractVelocityDistribution, x) -> Number

The second derivative `φ''(x)`, by ForwardDiff of `dfdx`. Used only for the removable
singularity at `u = 0` in the real-axis Cauchy-integral fallback (`_cauchy_real`).
"""
_d2fdx2(f::AbstractVelocityDistribution, x) = ForwardDiff.derivative(y -> dfdx(f, y), x)

"""
    CAUCHY_WINDOW, CAUCHY_RTOL, CAUCHY_ATOL

Numerical-method constants for the generic Cauchy-integral fallback (`cauchy_dfdv`), fixed by
spec Decision D9 and not to be changed without re-measuring the table in that decision.
`CAUCHY_WINDOW` (`L = 6`) is the half-width, in units of `x`, of the singularity-subtraction
window in `_cauchy_real`; `CAUCHY_RTOL` and `CAUCHY_ATOL` bound every quadrature call in this
file. `CAUCHY_ATOL` exists because `rtol` alone is a 13x cost blow-up (not a wrong answer) near
the principal value's zero crossing.
"""
const CAUCHY_WINDOW = 6.0
const CAUCHY_RTOL = 1e-10
const CAUCHY_ATOL = 1e-12

"""
    _cauchy_real(f::AbstractVelocityDistribution, xi::Real) -> Complex

`cauchy_dfdv` on the real axis: the Plemelj form `𝒞φ(ξ) = P∫ φ'(x)/(x−ξ) dx + iπ φ'(ξ)`,
evaluated by singularity subtraction in the shifted variable `u = x − ξ` so that `ξ` appears
only in the integrand, never in the quadrature limits (spec D9, D13; Pitfall 1). The symmetric
window's `φ'(ξ) P∫_{−L}^{L} du/u` term is exactly zero and is not computed.

The window's segment list carries one extra breakpoint, at `u0 = −ξ` taken at `ξ`'s *primal
value* (`_primal_value`, never `ξ` itself: `conventions.md` 5.12's clause permitting a breakpoint
at a dual's primal value). `u0` is where the m-partial of a super-Gaussian's integrand is least
smooth (`|x|^m ln|x|` at `x = 0`); splitting the window there is what Step 1.SG.1's D8 measured to
improve the m-derivative of `_cauchy_real` from 5.07e-6 to 1.56e-9 relative at `m = 2, ξ = 0.5`,
with no measurable change on the Maxwellian (spec D8; Phase 0's M3–M5 tests still pass unchanged).
"""
function _cauchy_real(f::AbstractVelocityDistribution, xi::Real)
    QT = _basefloat(xi)
    L = QT(CAUCHY_WINDOW)
    z = zero(QT)
    g(u) = iszero(u) ? _d2fdx2(f, xi) : (dfdx(f, xi + u) - dfdx(f, xi)) / u
    u0 = QT(-_primal_value(xi))
    segments = z < u0 < L  ? (-L, z, u0, L) :
               -L < u0 < z ? (-L, u0, z, L) :
                             (-L, z, L)
    pv, _ = quadgk(g, segments...; rtol = CAUCHY_RTOL, atol = CAUCHY_ATOL)
    tail_neg, _ = quadgk(u -> dfdx(f, xi + u) / u, QT(-Inf), -L; rtol = CAUCHY_RTOL, atol = CAUCHY_ATOL)
    tail_pos, _ = quadgk(u -> dfdx(f, xi + u) / u, L, QT(Inf); rtol = CAUCHY_RTOL, atol = CAUCHY_ATOL)
    return pv + tail_neg + tail_pos + im * QT(pi) * dfdx(f, xi)
end

"""
    _cauchy_upper(f::AbstractVelocityDistribution, xi::Number) -> Complex

`cauchy_dfdv` for `Im ξ > 0`: direct quadrature of `∫ φ'(x)/(x−ξ) dx` along the real axis, no
split and no residue, since the pole at `x = ξ` is off the real axis.
"""
function _cauchy_upper(f::AbstractVelocityDistribution, xi::Number)
    QT = _basefloat(real(xi))
    val, _ = quadgk(x -> dfdx(f, x) / (x - xi), QT(-Inf), zero(QT), QT(Inf);
                     rtol = CAUCHY_RTOL, atol = CAUCHY_ATOL)
    return val
end

"""
    _cauchy_lower(f::AbstractVelocityDistribution, xi::Number) -> Complex

`cauchy_dfdv` for `Im ξ < 0`: the real-axis integral plus the residue `2πi φ'(ξ)` picked up by
deforming the Landau contour below the pole (spec D9). Requires `φ'` at the complex argument
`ξ`; a shape without a complex-capable `dfdx` errors there (D18).
"""
function _cauchy_lower(f::AbstractVelocityDistribution, xi::Number)
    QT = _basefloat(real(xi))
    val, _ = quadgk(x -> dfdx(f, x) / (x - xi), QT(-Inf), zero(QT), QT(Inf);
                     rtol = CAUCHY_RTOL, atol = CAUCHY_ATOL)
    return val + 2im * QT(pi) * dfdx(f, xi)
end

"""
    cauchy_dfdv(f::AbstractVelocityDistribution, xi::Number) -> Complex

The Cauchy integral of the shape's derivative,

    𝒞φ(ξ) = ∫ φ'(x) / (x − ξ) dx        (on the Landau contour; conventions.md 1.5),

dispatched on the sign of `Im ξ`: direct quadrature for `Im ξ > 0`, the real-axis integral plus
`2πi φ'(ξ)` for `Im ξ < 0`, and the Plemelj principal-value form for `Im ξ = 0` (a real `xi` is
treated as `Im ξ = 0`). This generic method is a quadrature fallback available to every shape
that can be evaluated; `Maxwellian` overrides it with the exact `Zprime`.

`conventions.md` 1.5; `docs/derivations/shapes.md`; spec D9.
"""
function cauchy_dfdv(f::AbstractVelocityDistribution, xi::Number)
    if imag(xi) > 0
        _cauchy_upper(f, xi)
    elseif imag(xi) < 0
        _cauchy_lower(f, xi)
    else
        _cauchy_real(f, real(xi))
    end
end

"""
    CAUCHY_SERIES_RTOL, CAUCHY_SERIES_MAXTERMS

Numerical-method constants for `_cauchy_series`, the generic large-`|ξ|` Laurent series for
`𝒞φ(ξ)` (spec D6). `CAUCHY_SERIES_RTOL` is the relative tolerance the running sum must reach
before the series is declared converged; `CAUCHY_SERIES_MAXTERMS` is the term cap, reached at
`m ≤ 2` near the low end of the series' converging range (spec D6 measured 37 of 40 terms needed
at `m = 1.5, |ξ| = 8`) and deliberately not lowered.
"""
const CAUCHY_SERIES_RTOL = 1e-12
const CAUCHY_SERIES_MAXTERMS = 40

"""
    _require_high_moments(f::AbstractVelocityDistribution)

Throws unless `f` defines `moment(f, n)` up to `n = CAUCHY_SERIES_MAXTERMS − 1`, the highest
moment `_cauchy_series` needs. Called once, before the term loop (never inside it, spec D6), so
that a shape whose `moment` stops at the shared contract's `n = 0` to `4`
(`conventions.md` Section 4) fails with a message naming the requirement rather than an opaque
`ArgumentError` from deep inside the series.

Spec D6, D18.
"""
function _require_high_moments(f::AbstractVelocityDistribution)
    try
        moment(f, 5)
    catch
        throw(ArgumentError(
            "_cauchy_series: the shape $(typeof(f)) defines moment only for n = 0 to 4, but the " *
            "Laurent series needs moments up to n = $(CAUCHY_SERIES_MAXTERMS - 1). A shape whose " *
            "cauchy_dfdv uses this series must define moment(f, n) in closed form for every " *
            "integer n >= 0 (conventions.md Section 4, OQ-3 item 7)."))
    end
    return nothing
end

"""
    _cauchy_series(f::AbstractVelocityDistribution, xi::Number) -> (value, converged::Bool)

The large-`|ξ|` asymptotic (not convergent) Laurent series for the principal-value part of
`𝒞φ(ξ)`,

    𝒞φ(ξ) ~ Σ_{n≥1} n M_{n−1} / ξ^{n+1} = 1/ξ² + 3/(2ξ⁴) + 5 M₄/ξ⁶ + 7 M₆/ξ⁸ + …,

from `∫ xⁿ φ' dx = −n M_{n−1}` (integration by parts) and the geometric expansion of
`1/(x−ξ)`, written once against `moment(f, n)` so that any shape with closed-form moments beyond
`n = 4` reuses it. The Plemelj term `iπφ'(ξ)` is exact and is added by the caller, not here.

The stopping rule, in this order for each `n = 1, …, CAUCHY_SERIES_MAXTERMS`: (1) an odd
`M_{n−1}` of a symmetric shape is exactly zero, so its term is skipped and not counted; (2) a
non-finite term (the first term is `Inf` at `ξ = 0`, and `Inf ≤ rtol·Inf` is `true`, so this guard
is mandatory — without it the series reports convergence and returns `Inf`, spec Pitfall 3)
returns `(sum, false)` immediately; (3) a term whose magnitude exceeds the previous term's means
the series has turned, and returns `(sum, false)` without adding it; (4) otherwise the term is
added; (5) if `|term| ≤ CAUCHY_SERIES_RTOL · |sum|` the series has converged, `(sum, true)`.
Falling off the loop without converging returns `(sum, false)`.

Spec D6; `docs/derivations/supergaussian.md`.
"""
function _cauchy_series(f::AbstractVelocityDistribution, xi::Number)
    _require_high_moments(f)
    M0 = moment(f, 0)
    total = zero(promote_type(typeof(xi), typeof(M0)))
    prev_mag = oftype(abs(total), Inf)
    converged = false
    for n in 1:CAUCHY_SERIES_MAXTERMS
        Mnm1 = moment(f, n - 1)
        iszero(Mnm1) && continue
        term = n * Mnm1 / xi^(n + 1)
        if !isfinite(term)
            return (total, false)
        end
        mag = abs(term)
        if mag > prev_mag
            return (total, false)
        end
        total += term
        prev_mag = mag
        if mag <= CAUCHY_SERIES_RTOL * abs(total)
            converged = true
            break
        end
    end
    return (total, converged)
end

"""
    moment(f::AbstractVelocityDistribution, n::Integer) -> Number

The `n`-th moment `∫ xⁿ φ(x) dx` of the shape `f`, by quadrature, for `n = 0` to `4`. Throws an
`ArgumentError` for `n` outside that range. The three normalizations of `conventions.md` 1.5 are
`moment(f,0) == 1`, `moment(f,1) == 0`, `moment(f,2) == 1/2`.

`conventions.md` Section 4.
"""
function moment(f::AbstractVelocityDistribution, n::Integer)
    0 <= n <= 4 || throw(ArgumentError("moment is defined for n = 0 to 4; got n = $n"))
    return quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-12, atol = 1e-14)[1]
end

"""
    skewness(f::AbstractVelocityDistribution) -> Number

The shape's skewness, `moment(f,3) / moment(f,2)^(3/2)`, from the general definition in terms of
`moment` (not assuming unit variance: `moment(f,2) == 1/2`, `conventions.md` 1.5).

`conventions.md` Section 4.
"""
skewness(f::AbstractVelocityDistribution) = moment(f, 3) / moment(f, 2)^(3 / 2)

"""
    excess_kurtosis(f::AbstractVelocityDistribution) -> Number

The shape's excess kurtosis, `moment(f,4) / moment(f,2)^2 − 3`, from the general definition in
terms of `moment`.

`conventions.md` Section 4.
"""
excess_kurtosis(f::AbstractVelocityDistribution) = moment(f, 4) / moment(f, 2)^2 - 3

"""
    parameters(f::AbstractVelocityDistribution) -> NamedTuple

The shape's fittable parameters (the exponent of a super-Gaussian, the index of a kappa; empty
for `Maxwellian`, which has none left after the three normalizations of `conventions.md` 1.5).
No generic method is provided; every shape defines its own.

`conventions.md` Section 4; [OQ-3](../../dev/open_questions.md#oq-3) item 3.
"""
function parameters end

"""
    reconstruct(f::AbstractVelocityDistribution, values::NamedTuple) -> AbstractVelocityDistribution

A shape of the same type as `f` with its fittable parameters replaced by `values` (in the form
returned by `parameters(f)`). No generic method is provided; every shape defines its own.

`conventions.md` Section 4; [OQ-3](../../dev/open_questions.md#oq-3) item 3.
"""
function reconstruct end
