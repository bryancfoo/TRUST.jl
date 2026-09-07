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
