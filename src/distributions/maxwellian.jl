# The Maxwellian shape: phi(x) = exp(-x^2)/sqrt(pi), the reference shape for the whole package.
# See docs/derivations/shapes.md.

"""
    Maxwellian <: AbstractVelocityDistribution

The Maxwellian shape, `φ(x) = e^{−x²}/√π`. A non-parametric singleton: the three
normalizations of `conventions.md` 1.5 (`∫φ = 1`, `∫xφ = 0`, `∫x²φ = 1/2`) determine it
completely, so it carries no fields and no fittable parameters — unlike `SuperGaussian` or
`Kappa` (Phase 1 and 4), whose shape parameter survives normalization.

`conventions.md` 1.5; `docs/derivations/shapes.md`; spec D16.
"""
struct Maxwellian <: AbstractVelocityDistribution end

"""
    (f::Maxwellian)(x::Number) -> Number

`φ(x) = e^{−x²}/√π`. Accepts a real or complex `x`; the complex method is what makes the
Maxwellian usable in `cauchy_dfdv` below the real axis (`_cauchy_lower`).

`conventions.md` 1.5; `docs/derivations/shapes.md`.
"""
(::Maxwellian)(x::Number) = exp(-x^2) / sqrt(_basefloat(real(x))(pi))

"""
    dfdx(f::Maxwellian, x::Number) -> Number

`φ'(x) = −2x e^{−x²}/√π`, the closed form (never the generic ForwardDiff fallback). Accepts a
real or complex `x`, for the same reason as the call method above. Written as separate `Real`
and `Complex` methods, rather than one `Number` method, so that each is strictly more specific
than the corresponding generic method of `distribution.jl` (`dfdx(::AbstractVelocityDistribution,
::Real)` and `::Complex`) and dispatch is unambiguous.

`conventions.md` 1.5; `docs/derivations/shapes.md`.
"""
dfdx(::Maxwellian, x::Real) = -2x * exp(-x^2) / sqrt(_basefloat(real(x))(pi))
dfdx(::Maxwellian, x::Complex) = -2x * exp(-x^2) / sqrt(_basefloat(real(x))(pi))

"""
    cauchy_dfdv(f::Maxwellian, xi::Number) -> Complex

`𝒞φ(ξ) = Z'(ξ)` exactly (`conventions.md` 1.5): the Maxwellian's Cauchy integral is the
Fried–Conte plasma dispersion function's derivative, so it is returned directly rather than by
the generic quadrature fallback.

`conventions.md` 1.5; `docs/derivations/shapes.md`.
"""
cauchy_dfdv(::Maxwellian, xi::Number) = Zprime(xi)

"""
    parameters(f::Maxwellian) -> NamedTuple

Empty: the Maxwellian has no fittable parameters (spec D16).

`conventions.md` Section 4.
"""
parameters(::Maxwellian) = NamedTuple()

"""
    reconstruct(f::Maxwellian, values::NamedTuple) -> Maxwellian

Returns `f` unchanged: there are no parameters to replace.

`conventions.md` Section 4.
"""
reconstruct(f::Maxwellian, ::NamedTuple) = f
