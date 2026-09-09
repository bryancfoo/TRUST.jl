# The SuperGaussian shape: the 1-D marginal along k̂ of the isotropic 3-D super-Gaussian
# F(v) ∝ exp(-|v/v_m|^m), NOT exp(-|x|^m) (a different, lighter-tailed shape, Phase 4).
# See docs/derivations/supergaussian.md.

using SpecialFunctions: gamma, digamma
using QuadGK: quadgk
import ForwardDiff

"""
    SuperGaussian{FT<:Real,L} <: AbstractVelocityDistribution

The 1-D marginal along k̂ of the isotropic 3-D super-Gaussian `F(v) ∝ exp(-|v/v_m|^m)`
(PACM Eq. 26; foo2026 Eq. 12), **not** `exp(-|x|^m)`. Field `m` is the exponent; `x0` and `C` are
the shape's own normalization constants `x₀(m)` and `C(m)` (spec D4), cached at construction so
that the per-call quadrature reference does not recompute them at every quadrature node. `slice`
is the table's ξ-coefficients at this shape's own `m` (Step 1.SG.2); at Step 1.SG.1 no table
exists, so every `SuperGaussian` has `L = 0` and `slice = ()`. All four fields share the element
type `FT`, which follows `m`: a dual `m` gives dual `x0`, `C` and `slice`, all of the same tag,
which is how the m-derivative flows through a cached shape.

`conventions.md` Section 4, Section 11; spec D3; `docs/derivations/supergaussian.md`.
"""
struct SuperGaussian{FT<:Real,L} <: AbstractVelocityDistribution
    m::FT
    x0::FT
    C::FT
    slice::NTuple{L,FT}
end

"""
    _sg_x0(m) -> Real

`x₀(m)² = (3/2) Γ(3/m) / Γ(5/m)`, the super-Gaussian's width constant, fixed by the variance
normalization `∫x²φ dx = 1/2` (`conventions.md` 1.5). Generic over `m`'s type: ForwardDiff
differentiates the one-argument `gamma` with no extra rule (spec D4, verified).

PACM Eq. 20; `docs/derivations/supergaussian.md`.
"""
_sg_x0(m) = sqrt(3 * gamma(3 / m) / (2 * gamma(5 / m)))

"""
    _sg_C(m) -> Real

`C(m) = 1 / (2 x₀(m) Γ(3/m))`, the super-Gaussian's normalization constant, fixed by the unit
integral `∫φ dx = 1`.

PACM Eq. 26; `docs/derivations/supergaussian.md`.
"""
_sg_C(m) = 1 / (2 * _sg_x0(m) * gamma(3 / m))

"""
    SuperGaussian(m::Real)
    (f::SuperGaussian)(x::Real) -> Real

The public, one-argument constructor (`conventions.md` Section 11), and the shape's call
overload. **Both are documented here because they share one doc-system key** (spec Pitfall 21):
`SuperGaussian(m::Real)` and `(f::SuperGaussian)(x::Real)` are both methods of the binding
`SuperGaussian` with signature `Tuple{Real}`, so two separate docstrings on them collide and one
is silently dropped, with a `Replacing docs` warning at precompile.

**The constructor.** Throws for `m ≤ 0`; computes `x0 = _sg_x0(m)` and `C = _sg_C(m)`; promotes
`m`, `x0`, `C` **and the table's own element type**, `eltype(SG_TABLE.coeffs)`, to their common
type `FT` (spec D3, corrected 2026-09-08 round 4: without the table's element type, a `Float32 m`
promotes the three scalars to `Float32` while `_sg_slice` returns an `NTuple{L,Float64}`, and the
implicit constructor has no method for the mismatch — verified, `SuperGaussian(3.0f0)` threw a
`MethodError` before this fix); computes `slice = _sg_slice(SG_TABLE, m)` at the promoted `m` (so
`slice`'s element type is `FT` too); returns `SuperGaussian{FT,L}`, `L` the table's slice length,
the same integer for every shape built in a session. Every constructor path recomputes all three
cached fields and the slice from `m` — there is no constructor that takes them directly. The
uncached form the table's own generator needs, `SuperGaussian{FT,0}` with `slice = ()`, is built
only by `_sg_uncached`, never by this constructor (spec D3, D11), and **must not** get this same
fourth-argument fix: it runs before `SG_TABLE` exists, so referring to it there is an undefined
binding that would reintroduce the cycle D11 exists to break. The asymmetry between the two
constructors is deliberate.

Dimensionless: `m` is the exponent of the 3-D form (`conventions.md` notation dictionary), not a
physical quantity. Tested for `m ∈ [1.4999, 8.5]`; below `m = 1` the quadrature fallback's second
derivative at `x = 0` can be `NaN` under ForwardDiff (spec Pitfall 14), so no test asks for it.

**The call.** `φ(x) = C(m) Γ(2/m, |x/x₀(m)|^m)`, the upper incomplete gamma function.
Dimensionless in and out. `x = 0` is handled in closed form, `φ(0) = C(m) Γ(2/m) · one(x)`, rather
than through `_upper_gamma`: `iszero(x)` is `false` for a `ForwardDiff.Dual` whose value is zero, so
the guard is `iszero(ForwardDiff.value(x))`, and without it `|x/x₀|^m` under a dual `m` evaluates
`0 · log 0 = NaN` in the m-partial (spec Pitfall 1). **The `* one(x)` is not decoration** (spec D5,
corrected 2026-09-08 round 4): without it this branch returns a plain `Float64` while the general
branch returns a `Dual`, so the call's inferred return type at a dual `x` is `Union{Float64,Dual}`
and the partials are silently dropped on this branch — harmless here only because φ′(0) = 0 for
this shape, but a shape whose derivative at the guard point is nonzero would return the wrong
answer.

PACM Eq. 26; spec D3, D4, D5; `docs/derivations/supergaussian.md`.
"""
function SuperGaussian(m::Real)
    m > 0 || throw(ArgumentError("SuperGaussian: m must be positive; got $m"))
    x0 = _sg_x0(m)
    C = _sg_C(m)
    FT = promote_type(typeof(m), typeof(x0), typeof(C), eltype(SG_TABLE.coeffs))
    m_p, x0_p, C_p = FT(m), FT(x0), FT(C)
    slice = _sg_slice(SG_TABLE, m_p)
    return SuperGaussian(m_p, x0_p, C_p, slice)
end

"""
    _sg_uncached(m::Real) -> SuperGaussian{FT,0}

The generator's constructor, the *only* other way to build a `SuperGaussian`: computes `m`, `x0`
and `C` exactly as the public constructor does, but sets `slice = ()`, giving a shape of the
**distinct type** `SuperGaussian{FT,0}`. This breaks the cycle the table would otherwise create
(`SuperGaussian(m)` needs `SG_TABLE`, and building `SG_TABLE` needs shapes to evaluate `R`):
`_sg_principal_value` calls `_sg_cauchy_reference` on a shape built this way, and that path reads
only `m`, `x0` and `C` — never `slice` and never `SG_TABLE` (spec D3, D11, Pitfall 2).

`cauchy_dfdv` on a `SuperGaussian{<:Real,0}` throws (spec D18): a shape with a meaningless empty
slice can never be evaluated by accident.

Spec D3, D11; `docs/derivations/supergaussian.md`.
"""
function _sg_uncached(m::Real)
    x0 = _sg_x0(m)
    C = _sg_C(m)
    FT = promote_type(typeof(m), typeof(x0), typeof(C))
    return SuperGaussian{FT,0}(FT(m), FT(x0), FT(C), ())
end

"""
    cauchy_dfdv(f::SuperGaussian{<:Real,0}, xi::Number)

Throws (spec D18): `f` is the uncached form the table's generator builds, with an empty,
meaningless `slice`. It must never be evaluated; construct with `SuperGaussian(m)` instead. More
specific than the generic `cauchy_dfdv(::SuperGaussian, ::Number)` method below, so a
`SuperGaussian{<:Real,0}` always reaches this one first.
"""
cauchy_dfdv(f::SuperGaussian{<:Real,0}, xi::Number) = throw(ArgumentError(
    "cauchy_dfdv(SuperGaussian{FT,0}, xi): this shape carries no table slice. It is the " *
    "uncached form the table's generator builds and must not be used for evaluation; " *
    "construct the shape with SuperGaussian(m)."))

# Documented above, on SuperGaussian(m::Real): the two share one doc-system key (spec Pitfall 21).
function (f::SuperGaussian)(x::Real)
    iszero(ForwardDiff.value(x)) && return f.C * gamma(2 / f.m) * one(x)
    return f.C * _upper_gamma(2 / f.m, abs(x / f.x0)^f.m)
end

"""
    (f::SuperGaussian)(x::Complex)

Throws: `|x|^m` has no single analytic continuation off the real axis for general `m`, so the
shape is defined for real `x` only (spec D18).
"""
(f::SuperGaussian)(x::Complex) = throw(ArgumentError(
    "(f::SuperGaussian)(x::Complex) is not available: |x|^m has no single analytic " *
    "continuation off the real axis for general m, so the shape is defined for real x only."))

"""
    dfdx(f::SuperGaussian, x::Real) -> Real

`φ'(x) = −(m C(m)/x₀(m)²) x exp(−|x/x₀(m)|^m)`, the closed form (never the chain rule through
`Γ`, which is singular in the m-partial at `x = 0` for the same reason as the call method). The
power term `|x/x₀|^m` is replaced by an exact zero of the promoted type whenever
`iszero(ForwardDiff.value(x))`, which reproduces the same value and derivative the unguarded
formula gives at `x = 0` when `m` is plain (needed by `_d2fdx2`'s removable-singularity subtraction
at `ξ = 0`, spec D9/`_cauchy_real`) while avoiding the `0^{dual m}` singularity when `m` is a
dual (spec Pitfall 1). Dimensionless in and out.

PACM Eq. 44; spec D4; `docs/derivations/supergaussian.md`.
"""
function dfdx(f::SuperGaussian, x::Real)
    t = iszero(ForwardDiff.value(x)) ? zero(promote_type(typeof(x), typeof(f.m))) :
        abs(x / f.x0)^f.m
    return -(f.m * f.C / f.x0^2) * x * exp(-t)
end

"""
    dfdx(f::SuperGaussian, x::Complex)

Throws, for the same reason as the call method's `Complex` method (spec D18, rule 5.13: a split
override, never a single `::Number` method).
"""
dfdx(f::SuperGaussian, x::Complex) = throw(ArgumentError(
    "dfdx(SuperGaussian, ::Complex) is not available: |x|^m has no single analytic " *
    "continuation off the real axis for general m, so the shape is defined for real x only."))

"""
    moment(f::SuperGaussian, n::Integer) -> Real

The `n`-th moment `∫ xⁿ φ(x) dx`, in closed form for **every** integer `n ≥ 0` (not just the
shared contract's `n = 0` to `4`): zero for odd `n` (the shape is symmetric), and
`M_n = x₀^n Γ((n+3)/m) / ((n+1) Γ(3/m))` for even `n`. Needed up to `n ≈ 39` by the Laurent series
of `_cauchy_series` (`conventions.md` Section 4, [OQ-3](../../dev/open_questions.md#oq-3) item 7).
Throws for `n < 0` (spec D18).

Spec D4; `docs/derivations/supergaussian.md`.
"""
function moment(f::SuperGaussian, n::Integer)
    n >= 0 || throw(ArgumentError("moment(SuperGaussian, n): n must be non-negative; got $n"))
    isodd(n) && return zero(f.m)
    return f.x0^n * gamma((n + 3) / f.m) / ((n + 1) * gamma(3 / f.m))
end

"""
    _sg_cauchy_reference(f::SuperGaussian, xi::Number) -> Complex

The reference Cauchy integral: direct quadrature (`_cauchy_upper`) for `Im ξ > 0`, the Plemelj
principal-value quadrature (`_cauchy_real`) for real `ξ`, and an error for `Im ξ < 0` (spec D18:
`|x|^m` has no single analytic continuation off the real axis for general `m`, so the residue
continuation `_cauchy_lower` relies on does not apply). Correct, and slow, for every `m > 0` and
every `ξ` in the closed upper half-plane. This is the whole quadrature chain's single provenance
([OQ-31](../../dev/open_questions.md#oq-31) item 1): it never calls the series and never touches
the table, so it is both what the table's generator calls (Step 1.SG.2) and what every
reference-level test compares against.

Spec D7; `docs/derivations/supergaussian.md`.
"""
function _sg_cauchy_reference(f::SuperGaussian, xi::Number)
    if imag(xi) < 0
        throw(ArgumentError(
            "cauchy_dfdv(SuperGaussian, xi) requires Im xi >= 0; got $xi. |x|^m has no single " *
            "analytic continuation off the real axis for general m, so the residue continuation " *
            "of the generic fallback does not apply."))
    elseif imag(xi) > 0
        return _cauchy_upper(f, xi)
    else
        return _cauchy_real(f, real(xi))
    end
end

"""
    cauchy_dfdv(f::SuperGaussian, xi::Number) -> Complex

The Cauchy integral of the shape's derivative (`conventions.md` 1.5). At Step 1.SG.1, before the
table exists: `Im ξ < 0` throws and `Im ξ > 0` uses direct quadrature (both via
`_sg_cauchy_reference`); real `ξ` tries the large-|ξ| Laurent series (`_cauchy_series`) first and
falls back to the reference quadrature where the series does not converge. Step 1.SG.2's table
inserts a third branch, checked before the series, for real `ξ` inside the table's domain — see
`docs/derivations/supergaussian.md`.

`conventions.md` 1.5; spec D3, D6, D7; `docs/derivations/supergaussian.md`.
"""
function cauchy_dfdv(f::SuperGaussian, xi::Number)
    if !iszero(imag(xi))
        return _sg_cauchy_reference(f, xi)
    end
    rxi = real(xi)
    series_val, converged = _cauchy_series(f, rxi)
    if converged
        QT = _basefloat(rxi)
        return series_val + im * QT(pi) * dfdx(f, rxi)
    end
    return _sg_cauchy_reference(f, rxi)
end

"""
    parameters(f::SuperGaussian) -> NamedTuple

`(m = f.m,)`: the exponent alone, the shape's one fittable parameter. The cached `x0`, `C` and
`slice` are derived state, not parameters (spec D3).

`conventions.md` Section 4.
"""
parameters(f::SuperGaussian) = (m = f.m,)

"""
    reconstruct(f::SuperGaussian, values::NamedTuple) -> SuperGaussian

A fresh `SuperGaussian(values.m)`, recomputing every cached field from `m`; `f` itself is not
read. Allocation-free (TB12).

`conventions.md` Section 4.
"""
reconstruct(::SuperGaussian, values::NamedTuple) = SuperGaussian(values.m)

"""
    Base.show(io::IO, f::SuperGaussian)

Single line, `SuperGaussian(m = <value>)`, the exponent only — the cache is derived state, not a
parameter, and is never printed (spec D19).
"""
Base.show(io::IO, f::SuperGaussian) = print(io, "SuperGaussian(m = ", f.m, ")")

"""
    _upper_gamma(a, t) -> Real

The upper incomplete gamma function `Γ(a,t) = ∫_t^∞ s^{a−1} e^{−s} ds`, with a hand-written dual
rule (spec D5, design (a)): `SpecialFunctions.gamma(a,t)` has no method for `ForwardDiff.Dual`
arguments, so the both-dual method combines the value from `gamma(a,t)`, the a-partial from the
quadrature `_dgamma_da`, and the t-partial from the elementary `∂Γ(a,t)/∂t = −t^{a−1}e^{−t}`
(guarded at `t = 0`, where that elementary form is singular for `a < 1`, since `t = 0` there is
never actually reached with a nonzero t-partial — the call method's own `x = 0` guard routes
around this method entirely).

**The a-partial is computed only when `a` actually carries partials** (corrected 2026-09-08,
round 4, from Level 1's review): `(a::Real, t::Dual)` has its own body below, the t-partial alone,
instead of promoting into the both-dual method — promoting called `_dgamma_da`, an adaptive
quadrature, and then multiplied its result by `a`'s all-zero partials. That is the path every
`T_e`, `u` or `n_e` derivative of a fit takes, since `f_lab` calls `φ(x)` with a dual `x` and a
plain `m`: measured, 79.2 ns plain against 7.2 µs dual (a 91× ratio) and 1536 bytes allocated where
the plain call allocates none, violating `conventions.md` 5.2 and
[OQ-9](../../dev/open_questions.md#oq-9) trap 1 (the x-partial is the closed form, never a
quadrature). The both-dual method itself is also guarded, `iszero(ForwardDiff.partials(a))`, for
the promoted cases (two duals of different tags) that still reach it. **The case where `m` carries
partials keeps its quadrature** — that is design (a), correct, and not this round's business; it is
the next bottleneck for fits with `m` free, recorded in OQ-9 trap 1 for Phase 3 to profile.

Five methods on purpose (plain, both-dual-same-tag with its own body, `(Real, Dual)` with its own
body, and the two promoting methods for a plain second argument and two duals of different tags):
do not collapse them (spec D5, verified to dispatch without ambiguity — the `where {D<:Dual}`
methods are diagonal and therefore more specific than the promoting ones, so they terminate the
recursion once the arguments share a type).

PACM's incomplete gamma; spec D5; `docs/derivations/supergaussian.md`.
"""
_upper_gamma(a::Real, t::Real) = gamma(a, t)

function _upper_gamma(a::D, t::D) where {D<:ForwardDiff.Dual}
    a0 = ForwardDiff.value(a)
    t0 = ForwardDiff.value(t)
    v = _upper_gamma(a0, t0)
    da = iszero(ForwardDiff.partials(a)) ? zero(t0) : _dgamma_da(a0, t0)
    dt = iszero(t0) ? zero(t0) : -t0^(a0 - 1) * exp(-t0)
    return D(v, da * ForwardDiff.partials(a) + dt * ForwardDiff.partials(t))
end

function _upper_gamma(a::Real, t::D) where {D<:ForwardDiff.Dual}
    t0 = ForwardDiff.value(t)
    v = _upper_gamma(a, t0)
    dt = iszero(t0) ? zero(t0) : -t0^(a - 1) * exp(-t0)
    return D(v, dt * ForwardDiff.partials(t))
end

_upper_gamma(a::ForwardDiff.Dual, t::Real) = _upper_gamma(promote(a, t)...)
_upper_gamma(a::ForwardDiff.Dual, t::ForwardDiff.Dual) = _upper_gamma(promote(a, t)...)

"""
    _dgamma_da(a, t) -> Real

`∂Γ(a,t)/∂a = ∫_t^∞ s^{a−1} ln(s) e^{−s} ds` for `t > 0`, by `quadgk` in the shifted variable
`s = t + y` so that `t` appears only in the integrand (`conventions.md` 5.12); at `t = 0` the
closed form `Γ(a) ψ(a)` (`ψ` = `digamma`) is returned directly, branched first, since the shifted
integrand carries `y^{a−1} ln y` at the origin there. `rtol = 1e-12`, `atol = 1e-300`
(`conventions.md` 8.8: `atol` is mandatory since the integral can be near zero).

Spec D5; `docs/derivations/supergaussian.md`.
"""
function _dgamma_da(a::Real, t::Real)
    iszero(t) && return gamma(a) * digamma(a)
    QT = _basefloat(a)
    val, _ = quadgk(y -> (t + y)^(a - 1) * log(t + y) * exp(-(t + y)), zero(QT), QT(1), QT(Inf);
                     rtol = 1e-12, atol = 1e-300)
    return val
end
