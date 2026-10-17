# The Fried-Conte plasma dispersion function Z and its derivative Zprime, on the Landau
# contour, with ForwardDiff dual methods and ChainRulesCore rules.
# See docs/derivations/plasma_dispersion.md.

using SpecialFunctions: erfcx
import ForwardDiff
import ChainRulesCore

"""
    Z(xi::Number) -> Complex

The Fried-Conte plasma dispersion function,

    Z(ξ) = (1/√π) ∫ e^{−x²} / (x − ξ) dx      (Im ξ > 0, analytically continued),

implemented as `Z(ξ) = i√π · erfcx(−iξ)`, the Faddeeva-function form. `xi` may be real or
complex; the result is always a `Complex`, since `Z` is complex-valued even on the real axis.
On the real axis, `Im Z(x) = +√π e^{−x²} > 0`, the Landau prescription.

`conventions.md` 1.5; Sheffield Section 5.2 and Appendix A.1; `docs/derivations/plasma_dispersion.md`.
"""
Z(xi::Complex) = im * sqrt(oftype(real(xi), π)) * erfcx(-im * xi)
Z(xi::Real) = Z(complex(xi))

"""
    Zprime(xi::Number) -> Complex

The derivative of the plasma dispersion function,

    Z'(ξ) = −2 (1 + ξ Z(ξ)),

computed from this analytic identity, never by numerical differencing.

`conventions.md` 1.5; Sheffield Eqs. 5.2.5-5.2.6; `docs/derivations/plasma_dispersion.md`.
"""
Zprime(xi::Number) = -2 * (one(xi) + xi * Z(xi))

"""
    _Zpp(xi::Number) -> Complex

The second derivative `Z''(ξ) = −2 (Z(ξ) + ξ Z'(ξ))`, needed by the dual and ChainRules methods
of `Zprime` (spec D7, D8).
"""
_Zpp(xi::Number) = -2 * (Z(xi) + xi * Zprime(xi))

"""
    _Zderiv(xi::Number, ::Val{K}) -> NTuple{K+1}

The plasma dispersion function and its first `K` derivatives, `(Z(ξ), Z'(ξ), Z''(ξ), …, Z^{(K)}(ξ))`,
element `k + 1` holding `Z^{(k)}`. The first two elements are `Z(ξ)` and `Zprime(ξ)` themselves, always,
so element 2 is `Zprime(ξ)` bitwise. The others come by one of two routes (Phase 5.A spec D5; the user's
decision of 2026-10-16, Phase 5.A summary U1):

- **the recursion**, for complex `ξ` and for real `ξ` with `|ξ| < ZDERIV_ASYMPTOTIC_XI`:

      Z^{(k+1)}(ξ) = −2 (k Z^{(k−1)}(ξ) + ξ Z^{(k)}(ξ)),    k ≥ 1,

  `Z'(ξ) = −2 (1 + ξ Z(ξ))` differentiated `k` times (the `k`-th derivative of `ξ Z` is `ξ Z^{(k)} +
  k Z^{(k−1)}`, Leibniz); its `k = 1` step is `_Zpp`'s formula, so element 3 is `_Zpp(ξ)` bitwise here.
  Each step cancels two nearly equal terms once `|ξ|` exceeds about one, so the error grows with `|ξ|`
  and `k`: at real `ξ = 6.99`, about `1e-6` relative in `Z^{(7)}` and `3e-10` in `Z^{(4)}`
  (`docs/derivations/hermite.md` Section 3 tabulates it);
- **the asymptotic series**, for real `ξ` with `|ξ| ≥ ZDERIV_ASYMPTOTIC_XI` (`_Zderiv_asymptotic`): the
  real part from the large-`ξ` expansion of `Re Z`, differentiated term by term, the imaginary part
  `√π (e^{−ξ²})^{(k)}` from the recursion run on the imaginary parts alone, which is stable on the real
  axis; at most `1.3e-11` relative in `Z^{(7)}` at `|ξ| = 7` and about `1e-15` from `|ξ| = 8` on.

The branch compares `ξ`'s primal value (`conventions.md` 5.12). Every step of both routes is algebraic
or elementary in `ξ`, `Z(ξ)` and `Zprime(ξ)`, whose dual methods rest on holomorphy, so a dual or
complex-dual `ξ` propagates with no new rule; both return the same type. Allocation-free.
`HermiteExpansion`'s Cauchy integral reads up to `Z^{(7)}` ([OQ-66](../dev/open_questions.md#oq-66) item 2;
`docs/derivations/hermite.md` Section 3).
"""
_Zderiv(xi::Number, ::Val{K}) where {K} = _Zderiv_recursion(xi, Val(K))

function _Zderiv(xi::Real, ::Val{K}) where {K}
    abs(_primal_value(xi)) < ZDERIV_ASYMPTOTIC_XI && return _Zderiv_recursion(xi, Val(K))
    return _Zderiv_asymptotic(xi, Val(K))
end

"""
    ZDERIV_ASYMPTOTIC_XI, ZDERIV_ASYMPTOTIC_TERMS

`_Zderiv`'s switch (Phase 5.A spec D5): on the real axis, at `|ξ| ≥ ZDERIV_ASYMPTOTIC_XI = 7.0` the real
parts of `Z''` to `Z^{(K)}` come from the asymptotic series with `ZDERIV_ASYMPTOTIC_TERMS = 40` terms.
Measured against a BigFloat evaluation of the convergent series (2026-10-16): with 40 terms the series
reads `Z^{(7)}` to `1.3e-11` at `ξ = 7` (its best count there; 30 terms `1.6e-10`, 50 terms `6.6e-11`) and to
`5e-16` at `ξ = 8`, where the recursion reads `1.0e-6` and `4.1e-6`; at `ξ = 6` the series cannot beat the
recursion (`3.1e-7` at best against `1.4e-7`). Not to be changed without re-measuring.
"""
const ZDERIV_ASYMPTOTIC_XI = 7.0
const ZDERIV_ASYMPTOTIC_TERMS = 40

# The recursion route of `_Zderiv` (Phase 5.A spec D4): one `Z` and one `Zprime` call, the rest a tuple
# built by `ntuple` over `Val`.
function _Zderiv_recursion(xi::Number, ::Val{K}) where {K}
    z0 = Z(xi)
    z1 = Zprime(xi)
    return ntuple(Val(K + 1)) do i
        i == 1 && return z0
        a, b = z0, z1
        for k in 1:(i-2)
            a, b = b, -2 * (k * a + xi * b)
        end
        return b
    end
end

"""
    _Zderiv_asymptotic(x::Real, ::Val{K}) -> NTuple{K+1}

`_Zderiv`'s large-`|ξ|` route on the real axis (Phase 5.A spec D5). Elements 1 and 2 are `Z(x)` and
`Zprime(x)`. For `k ≥ 2`, element `k + 1` is `complex(_reZderiv_asymptotic(x, k), I_k)`, where `I_k = Im
Z^{(k)}(x) = √π (e^{−x²})^{(k)}` comes from the recursion `I_{k+1} = −2 (k I_{k−1} + x I_k)` started at
`imag(Z(x))` and `imag(Zprime(x))`: on the real axis the imaginary parts obey the recursion alone, and
there it is stable (they grow with `k` like `Hₖ(x)`).
"""
function _Zderiv_asymptotic(x::Real, ::Val{K}) where {K}
    z0 = Z(x)
    z1 = Zprime(x)
    i0, i1 = imag(z0), imag(z1)
    return ntuple(Val(K + 1)) do i
        i == 1 && return z0
        i == 2 && return z1
        k = i - 1
        a, b = i0, i1
        for j in 1:(k-1)
            a, b = b, -2 * (j * a + x * b)
        end
        return complex(_reZderiv_asymptotic(x, k), b)
    end
end

"""
    _reZderiv_asymptotic(x::Real, k::Integer) -> Real

`Re Z^{(k)}(x)` for real `x`, `|x|` large, from the asymptotic expansion `Re Z(x) ~ −Σ_{n≥0} aₙ x^{−(2n+1)}`,
`aₙ = (2n−1)!!/2ⁿ` (`a₀ = 1`), differentiated `k` times term by term:

    Re Z^{(k)}(x) ~ −(−1)^k x^{−(k+1)} Σ_{n=0}^{N−1} aₙ (2n+1)ₖ x^{−2n},    (p)ₖ = p (p+1) ⋯ (p+k−1),

`N = ZDERIV_ASYMPTOTIC_TERMS`. Every term is positive, so the sum has no cancellation. Valid for
`|x| ≥ ZDERIV_ASYMPTOTIC_XI` only (the series diverges; `_Zderiv` measures where it may be used).
"""
function _reZderiv_asymptotic(x::Real, k::Integer)
    u = inv(x * x)
    a = 1.0                   # a_n, from a_0 = 1
    r = 1.0                   # (2n+1)_k, from (1)_k = k!
    for j in 1:k
        r *= j
    end
    s = zero(u)
    un = one(u)
    for n in 0:(ZDERIV_ASYMPTOTIC_TERMS-1)
        s += (a * r) * un
        a *= (2n + 1) / 2
        r *= (2n + 1 + k) * (2n + 2 + k) / ((2n + 1) * (2n + 2))
        un *= u
    end
    return -(-1)^k * s / x^(k + 1)
end

# --- ForwardDiff dual methods (spec D7) ---
#
# erfcx has no method for Complex{Dual}, so Z fails under AD even for a real Dual argument,
# because Z's body makes its argument complex internally. These methods rest on holomorphy:
# with xi0 the value part and dxi the (possibly complex) infinitesimal,
#   Z(xi0 + dxi)  = Z(xi0)  + Z'(xi0)  dxi
#   Z'(xi0 + dxi) = Z'(xi0) + Z''(xi0) dxi

function Z(xi::ForwardDiff.Dual{Tg,V,N}) where {Tg,V,N}
    x0 = ForwardDiff.value(xi)
    p  = ForwardDiff.partials(xi)
    w  = Z(x0)
    wp = Zprime(x0)
    return Complex(
        ForwardDiff.Dual{Tg}(real(w), real(wp) * p),
        ForwardDiff.Dual{Tg}(imag(w), imag(wp) * p),
    )
end

function Z(xi::Complex{ForwardDiff.Dual{Tg,V,N}}) where {Tg,V,N}
    a  = real(xi)
    b  = imag(xi)
    x0 = complex(ForwardDiff.value(a), ForwardDiff.value(b))
    pa = ForwardDiff.partials(a)
    pb = ForwardDiff.partials(b)
    w  = Z(x0)
    wp = Zprime(x0)
    re  = real(wp) * pa - imag(wp) * pb
    im_ = imag(wp) * pa + real(wp) * pb
    return Complex(
        ForwardDiff.Dual{Tg}(real(w), re),
        ForwardDiff.Dual{Tg}(imag(w), im_),
    )
end

function Zprime(xi::ForwardDiff.Dual{Tg,V,N}) where {Tg,V,N}
    x0 = ForwardDiff.value(xi)
    p  = ForwardDiff.partials(xi)
    w  = Zprime(x0)
    wp = _Zpp(x0)
    return Complex(
        ForwardDiff.Dual{Tg}(real(w), real(wp) * p),
        ForwardDiff.Dual{Tg}(imag(w), imag(wp) * p),
    )
end

function Zprime(xi::Complex{ForwardDiff.Dual{Tg,V,N}}) where {Tg,V,N}
    a  = real(xi)
    b  = imag(xi)
    x0 = complex(ForwardDiff.value(a), ForwardDiff.value(b))
    pa = ForwardDiff.partials(a)
    pb = ForwardDiff.partials(b)
    w  = Zprime(x0)
    wp = _Zpp(x0)
    re  = real(wp) * pa - imag(wp) * pb
    im_ = imag(wp) * pa + real(wp) * pb
    return Complex(
        ForwardDiff.Dual{Tg}(real(w), re),
        ForwardDiff.Dual{Tg}(imag(w), im_),
    )
end

# --- ChainRulesCore rules (spec D8) ---
#
# frule pushes forward with no conjugation; rrule pulls back with the conjugate. Verified
# empirically against ChainRules.rrule(exp, ·) while the spec was written.

function ChainRulesCore.frule((_, Δxi), ::typeof(Z), xi)
    return Z(xi), Zprime(xi) * Δxi
end

function ChainRulesCore.rrule(::typeof(Z), xi)
    z  = Z(xi)
    zp = Zprime(xi)
    Z_pullback(Ω̄) = (ChainRulesCore.NoTangent(), conj(zp) * Ω̄)
    return z, Z_pullback
end

function ChainRulesCore.frule((_, Δxi), ::typeof(Zprime), xi)
    return Zprime(xi), _Zpp(xi) * Δxi
end

function ChainRulesCore.rrule(::typeof(Zprime), xi)
    zp  = Zprime(xi)
    zpp = _Zpp(xi)
    Zprime_pullback(Ω̄) = (ChainRulesCore.NoTangent(), conj(zpp) * Ω̄)
    return zp, Zprime_pullback
end
