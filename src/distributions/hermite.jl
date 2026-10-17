# The HermiteExpansion shape: the Gram-Charlier A series around the Maxwellian,
# phi(x) = (exp(-x^2)/sqrt(pi)) (1 + c3 H3(x) + c4 H4(x) + c5 H5(x) + c6 H6(x)) with the physicists'
# Hermite polynomials; its Cauchy integral and every coefficient derivative closed form through the
# derivatives of Z, no table. Positivity is reported, never enforced. See docs/derivations/hermite.md.
# Phase 5.A Step 5.A.1, spec D3-D9 (OQ-66).

using QuadGK: quadgk

"""
    HermiteExpansion{FT<:Real} <: AbstractVelocityDistribution
    HermiteExpansion(c3, c4, c5, c6)
    HermiteExpansion(; c3 = 0.0, c4 = 0.0, c5 = 0.0, c6 = 0.0)

The Gram–Charlier A series around the Maxwellian, four fittable coefficients:

    φ(x) = (e^{−x²}/√π) (1 + c₃H₃(x) + c₄H₄(x) + c₅H₅(x) + c₆H₆(x)),

the physicists' Hermite polynomials (`H₃ = 8x³ − 12x`, `H₄ = 16x⁴ − 48x² + 12`, `H₅ = 32x⁵ − 160x³ + 120x`,
`H₆ = 64x⁶ − 480x⁴ + 720x² − 120`), orthogonal under `e^{−x²}`. **The three normalizations hold for every
coefficient** (`conventions.md` 1.5): the `n = 0, 1, 2` terms are what they fix, and they are fixed (`c₀ =
1`, `c₁ = c₂ = 0`), so the fitted temperature is the variance temperature exactly. `c₃` is the skew
(skewness `12√2 c₃`), `c₄` the kurtosis correction (excess kurtosis `96 c₄`), `c₅` and `c₆` the next two
corrections; the order stops at 6 ([OQ-66](../../dev/open_questions.md#oq-66) item 1). Named by its
basis, as `SuperGaussian` and `SkewNormal` are; it is the Gram–Charlier A series, **not** Azzalini's
skew-normal (`SkewNormal`, positive everywhere).

**Construction.** Positionally in the order `c3, c4, c5, c6`, or by keyword with every coefficient
defaulting to zero. `FT = float(promote_type(...))` over the four, so a dual coefficient makes the shape,
and through `_shape_eltype` its species, dual. **Every coefficient must be finite**; a `NaN` or `Inf` is
refused naming the coefficient. Nothing else is refused: **positivity is reported, never enforced**
(the user's choice of 2026-10-16, OQ-66 item 4).

**At zero coefficients the shape is the Maxwellian bitwise**: the call and `dfdx` return `Maxwellian`'s
own expressions when all four are zero, and `cauchy_dfdv` skips every term and returns `Zprime(ξ)` itself;
otherwise each zero coefficient's term is skipped (`iszero(c)`), never added as `0.0·H`. The test is `iszero` on the coefficient itself, deliberately not on
`_primal_value` (Phase 5.A spec D6): a dual coefficient at value zero with a nonzero seed must contribute
its partial, and `iszero` of such a dual is `false` (`conventions.md` 5.12).

**Positivity, and the admissible region.** The series goes negative for most coefficients: `c₃` alone is
never positive (a lobe on the far side of the skew), and `c₄ < 0`, the flattened core, never is either.
`_hermite_lobe(f)` reports the polynomial factor's minimum on `|x| ≤ HERMITE_X_MAX`, where it occurs,
and the negative mass; `show` prints them when the factor is negative, and `fit(deck)` warns when a fitted
bin's negative mass exceeds `HERMITE_NEGATIVE_MASS_WARN` (`1e-3`, a tenth of a percent of the
distribution). The region where the series is positive everywhere, with `c₅ = c₆ = 0`, measured on
2026-10-16 (Phase 5.A spec D7; `docs/derivations/hermite.md` Section 5): `c₄` from 0 to exactly 1/24
(where `H₄`'s minimum meets `H₃`'s zero), and `|c₃|` at most 0.0432 at `c₄ = 0.01`, 0.0594 at 0.02, 0.0618
at 0.025 (about the widest, skewness 1.05), 0.0520 at 0.035. **That region is advice for a deck's bounds,
not a boundary**: a fit may pass through negative-lobe territory, and a few tenths of a percent of negative
mass is the expansion's own truncation error, which the user judges.

**What it cannot do**: a beam or any strong deformation (use a second species), and orders beyond 6.

`conventions.md` 1.5, Section 4, Section 11; Phase 5.A spec D3; `docs/derivations/hermite.md`.
"""
struct HermiteExpansion{FT<:Real} <: AbstractVelocityDistribution
    c3::FT
    c4::FT
    c5::FT
    c6::FT
    function HermiteExpansion{FT}(c3::Real, c4::Real, c5::Real, c6::Real) where {FT<:Real}
        _hermite_finite("c3", c3)
        _hermite_finite("c4", c4)
        _hermite_finite("c5", c5)
        _hermite_finite("c6", c6)
        return new{FT}(FT(c3), FT(c4), FT(c5), FT(c6))
    end
end

# Refuses a non-finite coefficient naming it (Phase 5.A spec D3; `isfinite` reads a dual's value only).
_hermite_finite(name::String, c::Real) =
    isfinite(c) || throw(ArgumentError("HermiteExpansion: $name must be finite; got $c"))

HermiteExpansion(c3::Real, c4::Real, c5::Real, c6::Real) =
    HermiteExpansion{float(promote_type(typeof(c3), typeof(c4), typeof(c5), typeof(c6)))}(c3, c4, c5, c6)

HermiteExpansion(; c3::Real = 0.0, c4::Real = 0.0, c5::Real = 0.0, c6::Real = 0.0) =
    HermiteExpansion(c3, c4, c5, c6)

"""
    _hermite_polys(x::Number) -> NTuple{8}

The physicists' Hermite polynomials `(H₀(x), H₁(x), …, H₇(x))`, element `n + 1` holding `Hₙ`, by the
three-term recursion `H_{n+1} = 2x Hₙ − 2n H_{n−1}` from `H₀ = 1`, `H₁ = 2x`. For any `Number` `x`.

Phase 5.A spec D4; `docs/derivations/hermite.md` Section 1.
"""
function _hermite_polys(x::Number)
    h0 = one(x)
    h1 = 2x
    h2 = 2x * h1 - 2 * h0
    h3 = 2x * h2 - 4 * h1
    h4 = 2x * h3 - 6 * h2
    h5 = 2x * h4 - 8 * h3
    h6 = 2x * h5 - 10 * h4
    h7 = 2x * h6 - 12 * h5
    return (h0, h1, h2, h3, h4, h5, h6, h7)
end

# All four coefficients exactly zero, a dual's partials included (Phase 5.A spec D6).
_hermite_iszero(f::HermiteExpansion) = iszero(f.c3) && iszero(f.c4) && iszero(f.c5) && iszero(f.c6)

"""
    (f::HermiteExpansion)(x::Number) -> Number

`φ(x) = (e^{−x²}/√π) (1 + Σₙ cₙHₙ(x))`, for any `Number` `x`: real, complex or dual. At zero
coefficients `Maxwellian()(x)` itself, converted to the shape's promoted type (spec D6).

Phase 5.A spec D4; `docs/derivations/hermite.md` Section 1.
"""
function (f::HermiteExpansion{FT})(x::Number) where {FT}
    if _hermite_iszero(f)
        m = Maxwellian()(x)
        return convert(promote_type(typeof(m), FT), m)
    end
    g = exp(-x^2) / sqrt(_basefloat(real(x))(pi))
    H = _hermite_polys(x)
    s = one(promote_type(typeof(g), FT))
    iszero(f.c3) || (s += f.c3 * H[4])
    iszero(f.c4) || (s += f.c4 * H[5])
    iszero(f.c5) || (s += f.c5 * H[6])
    iszero(f.c6) || (s += f.c6 * H[7])
    return g * s
end

"""
    dfdx(f::HermiteExpansion, x::Real) -> Real
    dfdx(f::HermiteExpansion, x::Complex) -> Complex

`φ'(x) = −(e^{−x²}/√π) (H₁(x) + Σₙ cₙH_{n+1}(x))`, the closed form, from `Hₙ(x) e^{−x²} = (−1)ⁿ dⁿ/dxⁿ
e^{−x²}` (`H₇ = 128x⁷ − 1344x⁵ + 3360x³ − 1680x`). Separate `Real` and `Complex` methods, never one
`Number` method (`conventions.md` 5.13: `maxwellian.jl`'s reason). At zero coefficients
`dfdx(Maxwellian(), x)` itself (spec D6).

Phase 5.A spec D4; `docs/derivations/hermite.md` Section 2.
"""
dfdx(f::HermiteExpansion, x::Real) = _hermite_dfdx(f, x)
dfdx(f::HermiteExpansion, x::Complex) = _hermite_dfdx(f, x)

function _hermite_dfdx(f::HermiteExpansion{FT}, x::Number) where {FT}
    if _hermite_iszero(f)
        m = dfdx(Maxwellian(), x)
        return convert(promote_type(typeof(m), FT), m)
    end
    g = exp(-x^2) / sqrt(_basefloat(real(x))(pi))
    H = _hermite_polys(x)
    s = convert(promote_type(typeof(g), FT), H[2])
    iszero(f.c3) || (s += f.c3 * H[5])
    iszero(f.c4) || (s += f.c4 * H[6])
    iszero(f.c5) || (s += f.c5 * H[7])
    iszero(f.c6) || (s += f.c6 * H[8])
    return -g * s
end

"""
    cauchy_dfdv(f::HermiteExpansion, xi::Number) -> Complex

The Cauchy integral of the shape's derivative (`conventions.md` 1.5), closed form for every `ξ`, real,
complex or dual, with no table, no quadrature and no series:

    𝒞φ(ξ) = Z'(ξ) − c₃ Z⁽⁴⁾(ξ) + c₄ Z⁽⁵⁾(ξ) − c₅ Z⁽⁶⁾(ξ) + c₆ Z⁽⁷⁾(ξ)

(`Z'(ξ) + Σₙ cₙ (−1)ⁿ Z^{(n+1)}(ξ)`), from `∫ (e^{−x²})^{(k)}/(x − ξ) dx = √π Z^{(k)}(ξ)`, the derivatives of
`Z` from `_Zderiv`. **Every coefficient enters by one multiplication**, so `∂𝒞φ/∂cₙ = (−1)ⁿ Z^{(n+1)}(ξ)`
exactly ([OQ-38](../../dev/open_questions.md#oq-38)'s derivative clause met in closed form). At zero
coefficients every term is skipped and the result is `_Zderiv`'s second element, `Zprime(ξ)` itself
(spec D6).

Phase 5.A spec D4, D5; `docs/derivations/hermite.md` Section 3.
"""
function cauchy_dfdv(f::HermiteExpansion{FT}, xi::Number) where {FT}
    d = _Zderiv(xi, Val(7))
    acc = convert(promote_type(eltype(d), FT), d[2])
    iszero(f.c3) || (acc -= f.c3 * d[5])
    iszero(f.c4) || (acc += f.c4 * d[6])
    iszero(f.c5) || (acc -= f.c5 * d[7])
    iszero(f.c6) || (acc += f.c6 * d[8])
    return acc
end

"""
    moment(f::HermiteExpansion, n::Integer) -> Real

The `n`-th moment `∫ xⁿ φ(x) dx`, closed form for every integer `n ≥ 0`:

    Mₙ = Gₙ + Σ_{k=3}^{6} c_k (n!/(n−k)!) G_{n−k}    (the sum over k ≤ n),

with `Gⱼ = (1/√π) ∫ xʲ e^{−x²} dx = (j−1)!!/2^{j/2}` for even `j` and `0` for odd `j`, from `k`
integrations by parts of `∫ xⁿ (−1)^k (e^{−x²})^{(k)} dx`. So `M₀ = 1`, `M₁ = 0`, `M₂ = ½` exactly for
every coefficient, `M₃ = 6c₃` and `M₄ = ¾ + 24c₄`, and the generic `skewness` and `excess_kurtosis` read
`12√2 c₃` and `96 c₄`. Throws for `n < 0`.

Phase 5.A spec D4; `docs/derivations/hermite.md` Section 4.
"""
function moment(f::HermiteExpansion{FT}, n::Integer) where {FT}
    n >= 0 || throw(ArgumentError("moment(HermiteExpansion, n): n must be non-negative; got $n"))
    m = _hermite_gaussian_moment(n) + zero(FT)
    for (k, c) in ((3, f.c3), (4, f.c4), (5, f.c5), (6, f.c6))
        k <= n || break
        m += c * _hermite_falling(n, k) * _hermite_gaussian_moment(n - k)
    end
    return m
end

# (1/√π) ∫ xʲ e^{−x²} dx: (j−1)!!/2^{j/2} for even j ≥ 0, 0.0 for odd j (Phase 5.A spec D4).
function _hermite_gaussian_moment(j::Integer)
    isodd(j) && return 0.0
    g = 1.0
    for i in 1:2:(j-1)
        g *= i / 2
    end
    return g
end

# n!/(n−k)! = n (n−1) ⋯ (n−k+1), as a Float64 (Phase 5.A spec D4).
function _hermite_falling(n::Integer, k::Integer)
    p = 1.0
    for i in (n-k+1):n
        p *= i
    end
    return p
end

"""
    parameters(f::HermiteExpansion) -> NamedTuple

`(c3 = f.c3, c4 = f.c4, c5 = f.c5, c6 = f.c6)`: always all four. A coefficient a deck leaves out is fixed
at zero and bound to nothing.

`conventions.md` Section 4; Phase 5.A spec D3.
"""
parameters(f::HermiteExpansion) = (c3 = f.c3, c4 = f.c4, c5 = f.c5, c6 = f.c6)

"""
    reconstruct(f::HermiteExpansion, values::NamedTuple) -> HermiteExpansion

A fresh `HermiteExpansion(values.c3, values.c4, values.c5, values.c6)`, promoted over the four; `f` itself
is not read. Allocation-free.

`conventions.md` Section 4; Phase 5.A spec D3.
"""
reconstruct(::HermiteExpansion, values::NamedTuple) =
    HermiteExpansion(values.c3, values.c4, values.c5, values.c6)

"""
    HERMITE_X_MAX, HERMITE_NEGATIVE_MASS_WARN

The lobe report's constants ([OQ-66](../../dev/open_questions.md#oq-66) item 4; Phase 5.A spec D7).
`HERMITE_X_MAX = 6.0` is the half-width of the window `_hermite_lobe` reads, in units of `x`: the polynomial
factor is evaluated on the grid `−6, −5.9, …, 6` and the negative mass integrated over `|x| ≤ 6`, where
`e^{−x²}` has fallen to `2.3e−16`. `HERMITE_NEGATIVE_MASS_WARN = 1e-3` is `fit(deck)`'s warning threshold,
a tenth of a percent of the distribution.
"""
const HERMITE_X_MAX = 6.0
const HERMITE_NEGATIVE_MASS_WARN = 1e-3

# The negative-mass quadrature's breakpoints: the integers across the window (Phase 5.A spec D7).
const _HERMITE_BREAKS = Tuple(-HERMITE_X_MAX:1.0:HERMITE_X_MAX)

"""
    _hermite_lobe(f::HermiteExpansion) -> (factor_min, x_min, negative_mass)
    _hermite_lobe(f::AbstractVelocityDistribution) -> nothing

The positivity report (the user's option C of 2026-10-16, [OQ-66](../../dev/open_questions.md#oq-66)
item 4; Phase 5.A spec D7), at the coefficients' **primal values** (`_primal_value`, `conventions.md`
5.12): `factor_min`, the minimum of the polynomial factor `1 + Σₙ cₙHₙ(x)` over the grid `x = −6, −5.9, …,
6` (`HERMITE_X_MAX`, step 0.1), and `x_min`, the first grid point where it occurs; `negative_mass =
∫ max(−φ(x), 0) dx` over `|x| ≤ HERMITE_X_MAX`, by `quadgk` with the integers as breakpoints and
`CAUCHY_RTOL`, `CAUCHY_ATOL`: a magnitude, `≥ 0`, exactly `0.0` where `φ` is nowhere negative. **A
report, never on the fit path**: nothing in `solve`, the spectrum or the Jacobian calls it. `show` prints
it when `factor_min < 0`, and `fit(deck)`'s `_warn_hermite_lobe` reads its `negative_mass` per bin.

Every other shape has no such report, and returns `nothing` (the dispatch `_warn_hermite_lobe` uses in
place of an `isa`, `conventions.md` Section 4).
"""
_hermite_lobe(::AbstractVelocityDistribution) = nothing

function _hermite_lobe(f::HermiteExpansion)
    p = HermiteExpansion(_primal_value(f.c3), _primal_value(f.c4), _primal_value(f.c5), _primal_value(f.c6))
    factor_min = Inf
    x_min = NaN
    for x in range(-HERMITE_X_MAX, HERMITE_X_MAX; step = 0.1)
        H = _hermite_polys(x)
        fac = 1 + p.c3 * H[4] + p.c4 * H[5] + p.c5 * H[6] + p.c6 * H[7]
        if fac < factor_min
            factor_min = fac
            x_min = x
        end
    end
    negative_mass, _ = quadgk(x -> max(-p(x), 0.0), _HERMITE_BREAKS...; rtol = CAUCHY_RTOL, atol = CAUCHY_ATOL)
    return (factor_min = factor_min, x_min = x_min, negative_mass = negative_mass)
end

"""
    Base.show(io::IO, f::HermiteExpansion)

Single line, every coefficient printed, zeros included: `HermiteExpansion(c3 = 0.04, c4 = 0.01, c5 = 0.0,
c6 = 0.0)`. When the polynomial factor is negative somewhere in `|x| ≤ HERMITE_X_MAX`, the lobe report
follows (`_hermite_lobe`, three significant figures): `HermiteExpansion(c3 = 0.04, c4 = 0.0, c5 = 0.0,
c6 = 0.0; negative lobe: factor -65.2 at x = -6.0, negative mass 0.00421)`.

Phase 5.A spec D8.
"""
function Base.show(io::IO, f::HermiteExpansion)
    print(io, "HermiteExpansion(c3 = ", f.c3, ", c4 = ", f.c4, ", c5 = ", f.c5, ", c6 = ", f.c6)
    lobe = _hermite_lobe(f)
    if lobe.factor_min < 0
        print(io, "; negative lobe: factor ", round(lobe.factor_min; sigdigits = 3), " at x = ", lobe.x_min,
              ", negative mass ", round(lobe.negative_mass; sigdigits = 3))
    end
    print(io, ")")
end
