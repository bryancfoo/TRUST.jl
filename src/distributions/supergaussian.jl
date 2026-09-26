# The SuperGaussian shape: the 1-D marginal along k̂ of the isotropic 3-D super-Gaussian
# F(v) ∝ exp(-|v/v_m|^m), NOT exp(-|x|^m) (a different, lighter-tailed shape, Phase 4).
# See docs/derivations/supergaussian.md.

using SpecialFunctions: gamma, gamma_inc, digamma
using QuadGK: quadgk
import ForwardDiff

"""
    SuperGaussian{FT<:Real,L,LM} <: AbstractVelocityDistribution

The 1-D marginal along k̂ of the isotropic 3-D super-Gaussian `F(v) ∝ exp(-|v/v_m|^m)`
(PACM Eq. 26; foo2026 Eq. 12), **not** `exp(-|x|^m)`. Field `m` is the exponent; `x0` and `C` are
the shape's own normalization constants `x₀(m)` and `C(m)` (spec D4), cached at construction so
that the per-call quadrature reference does not recompute them at every quadrature node. `slice`
is the Cauchy table's ξ-coefficients at this shape's own `m` (Step 1.SG.2); at Step 1.SG.1 no
table exists, so every `SuperGaussian` has `L = 0` and `slice = ()`. `dm_slice` is the m-partial
table's `|x|`-coefficients at this shape's own `m` (Phase 3.C spec D6), present **only when `m`
carries a `ForwardDiff.Dual`**: a plain `m` gives `LM = 0` and `dm_slice = ()`, so that the
per-value-pass cost and the size of every Phase 1.SG/2/3 fixture (all plain- or fixed-`m`,
`conventions.md` 8.11) are unchanged. All five fields share the element type `FT`, which follows
`m`: a dual `m` gives dual `x0`, `C`, `slice` and `dm_slice`, all of the same tag, which is how
the value and both derivatives flow through a cached shape.

`conventions.md` Section 4, Section 11; spec D3; Phase 3.C spec D6;
`docs/derivations/supergaussian.md`.
"""
struct SuperGaussian{FT<:Real,L,LM} <: AbstractVelocityDistribution
    m::FT
    x0::FT
    C::FT
    slice::NTuple{L,FT}
    dm_slice::NTuple{LM,FT}
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

**The constructor.** Throws for `m ≤ 0`; promotes `m` **first**, to `FT =
promote_type(typeof(m), eltype(SG_TABLE.coeffs))`, then computes `x0 = _sg_x0(m_p)` and
`C = _sg_C(m_p)` at the promoted `m_p` (spec D3, corrected 2026-09-09 at Step 2.0: promoting
before computing, rather than computing at `m`'s own type and promoting the three scalars
afterwards, is what `conventions.md` 5.1 asks for — "a `Float32` input promotes to `Float64` at
the constructor" — since computing `x0` and `C` at `Float32` precision and then widening gives a
different shape from `SuperGaussian(3.0)`, bit for bit, not merely to `Float32` precision; HK1
is the test); computes `slice = _sg_slice(SG_TABLE, m_p)` at the promoted `m_p` (so `slice`'s
element type is `FT` too); returns `SuperGaussian{FT,L,LM}`, `L` the Cauchy table's slice length,
the same integer for every shape built in a session. Every constructor path recomputes all cached
fields and slices from `m` — there is no constructor that takes them directly. The uncached
form the tables' own generators need, `SuperGaussian{FT,0,0}` with `slice = ()` and
`dm_slice = ()`, is built only by `_sg_uncached`, never by this constructor (spec D3, D11), and
**must not** get this same promote-first treatment: it runs before `SG_TABLE` exists, so naming
`SG_TABLE` there is an undefined binding that would reintroduce the cycle D11 exists to break. The
asymmetry between the two constructors is deliberate — do not tidy them into agreement.

**Two public constructor methods** (Phase 3.C spec D6): `SuperGaussian(m::Real)`, above, gives
`LM = 0` and `dm_slice = ()`; `SuperGaussian(m::ForwardDiff.Dual)`, below, additionally computes
`dm_slice = _sg_slice(SG_DM_TABLE, m_p)`, the m-partial table's slice at this shape's own `m`.
`ForwardDiff.Dual <: Real`, so the dual method is strictly more specific and Julia selects it
automatically for a dual `m`; **`LM = 0` for a plain `m` is not an optimization for its own sake**
— it is what keeps `reconstruct`'s cost, and the size of every `Species` and `Plasma` built from a
plain- or fixed-`m` shape (every Phase 1.SG/2/3 fixture, `conventions.md` 8.11), exactly what it
was, and it is what makes the fast dual-`m` call method of D7 below unreachable except when a
dm-slice actually exists (D6 item 3, `_sg_call`'s `LM = 0` override).

Dimensionless: `m` is the exponent of the 3-D form (`conventions.md` notation dictionary), not a
physical quantity. Tested for `m ∈ [1.4999, 8.5]`; below `m = 1` the quadrature fallback's second
derivative at `x = 0` can be `NaN` under ForwardDiff (spec Pitfall 14), so no test asks for it.
**A shape's `m` outside the table's domain (`SG_TABLE.m_lo` to `SG_TABLE.m_hi`) is not an error**:
`cauchy_dfdv` on it runs the reference quadrature instead of the table, correctly but about 60 µs
a call against about 50 ns through the table (spec D18, D14); likewise a dual `m` outside
`SG_DM_TABLE`'s domain (the same rectangle) falls back to the quadrature-based m-partial (Phase
3.C spec D7).

**The call.** `φ(x) = C(m) Γ(2/m, |x/x₀(m)|^m)`, the upper incomplete gamma function.
Dimensionless in and out. `x = 0` is handled in closed form, `φ(0) = C(m) Γ(2/m) · one(x)`, rather
than through `_upper_gamma`: `iszero(x)` is `false` for a `ForwardDiff.Dual` whose value is zero, so
the guard is `iszero(ForwardDiff.value(x))`, and without it `|x/x₀|^m` under a dual `m` evaluates
`0 · log 0 = NaN` in the m-partial (spec Pitfall 1). **The `* one(x)` is not decoration** (spec D5,
corrected 2026-09-08 round 4): without it this branch returns a plain `Float64` while the general
branch returns a `Dual`, so the call's inferred return type at a dual `x` is `Union{Float64,Dual}`
and the partials are silently dropped on this branch — harmless here only because φ′(0) = 0 for
this shape, but a shape whose derivative at the guard point is nonzero would return the wrong
answer. This is `_sg_call_generic` since Phase 3.C spec D7 (`(f::SuperGaussian)(x::Real)` itself
now dispatches through `_sg_call`, below, to either this generic body or the fast dual-`m` table
read).

PACM Eq. 26; spec D3, D4, D5; Phase 3.C spec D6, D7; `docs/derivations/supergaussian.md`.
"""
function SuperGaussian(m::Real)
    m > 0 || throw(ArgumentError("SuperGaussian: m must be positive; got $m"))
    FT = promote_type(typeof(m), eltype(SG_TABLE.coeffs))
    m_p = FT(m)
    x0_p = FT(_sg_x0(m_p))
    C_p = FT(_sg_C(m_p))
    slice = _sg_slice(SG_TABLE, m_p)
    return SuperGaussian(m_p, x0_p, C_p, slice, ())
end

"""
    SuperGaussian(m::ForwardDiff.Dual)

The dual-`m` constructor (Phase 3.C spec D6): promotes and computes `m`, `x0`, `C` and `slice`
exactly as `SuperGaussian(m::Real)` does, and additionally computes
`dm_slice = _sg_slice(SG_DM_TABLE, m_p)`, the m-partial table's `|x|`-coefficients at this shape's
own `m`. Selected automatically over `SuperGaussian(m::Real)` for any dual `m` since
`ForwardDiff.Dual <: Real` makes this method the more specific one. `reconstruct` therefore
returns `LM = 0` for a plain `values.m` and the full slice for a dual one with no special case,
matching exactly what a fit needs.

Spec D3, D4; Phase 3.C spec D6; `docs/derivations/supergaussian.md`.
"""
function SuperGaussian(m::ForwardDiff.Dual)
    m > 0 || throw(ArgumentError("SuperGaussian: m must be positive; got $m"))
    FT = promote_type(typeof(m), eltype(SG_TABLE.coeffs))
    m_p = FT(m)
    x0_p = FT(_sg_x0(m_p))
    C_p = FT(_sg_C(m_p))
    slice = _sg_slice(SG_TABLE, m_p)
    dm_slice = _sg_slice(SG_DM_TABLE, m_p)
    return SuperGaussian(m_p, x0_p, C_p, slice, dm_slice)
end

"""
    _sg_uncached(m::Real) -> SuperGaussian{FT,0,0}

The generators' constructor, the *only* other way to build a `SuperGaussian`: computes `m`, `x0`
and `C` exactly as the public constructors do, but sets `slice = ()` and `dm_slice = ()`, giving a
shape of the **distinct type** `SuperGaussian{FT,0,0}`, regardless of whether `m` itself carries
partials. This breaks the cycle each table would otherwise create (`SuperGaussian(m)` needs
`SG_TABLE`/`SG_DM_TABLE`, and building either table needs shapes to evaluate its sampler):
`_sg_principal_value` and `_sg_dphi_dm_reference` both call through a shape built this way, and
that path (`_sg_call_generic`, via `_sg_cauchy_reference` or directly) reads only `m`, `x0` and
`C` — never `slice`, `dm_slice`, `SG_TABLE` nor `SG_DM_TABLE` (spec D3, D11, Pitfall 2; Phase 3.C
spec D8, D9, Pitfall 7). Fixing `LM = 0` here unconditionally, rather than following whether `m`
is a dual, is what makes D7's fast call method unreachable for this type by dispatch: a `Dual` `m`
built this way (as `_sg_dphi_dm_reference`'s `ForwardDiff.derivative` closure does) still takes
the generic call path, since the fast method has no applicable form at `LM = 0`.

`cauchy_dfdv` on a `SuperGaussian{<:Real,0,0}` throws (spec D18): a shape with meaningless empty
slices can never be evaluated by accident.

Spec D3, D11; Phase 3.C spec D8, D9; `docs/derivations/supergaussian.md`.
"""
function _sg_uncached(m::Real)
    x0 = _sg_x0(m)
    C = _sg_C(m)
    FT = promote_type(typeof(m), typeof(x0), typeof(C))
    return SuperGaussian{FT,0,0}(FT(m), FT(x0), FT(C), (), ())
end

"""
    cauchy_dfdv(f::SuperGaussian{<:Real,0,0}, xi::Number)

Throws (spec D18): `f` is the uncached form the tables' generators build, with empty,
meaningless `slice` and `dm_slice`. It must never be evaluated; construct with `SuperGaussian(m)`
instead. More specific than the generic `cauchy_dfdv(::SuperGaussian, ::Number)` method below, so
a `SuperGaussian{<:Real,0,0}` always reaches this one first.
"""
cauchy_dfdv(f::SuperGaussian{<:Real,0,0}, xi::Number) = throw(ArgumentError(
    "cauchy_dfdv(SuperGaussian{FT,0,0}, xi): this shape carries no table slice. It is the " *
    "uncached form the tables' generators build and must not be used for evaluation; " *
    "construct the shape with SuperGaussian(m)."))

# Documented above, on SuperGaussian(m::Real): the two share one doc-system key (spec Pitfall 21).
(f::SuperGaussian)(x::Real) = _sg_call(f, x)

"""
    _sg_call_generic(f::SuperGaussian, x::Real) -> Real

The Phase 1.SG call body, moved here verbatim and otherwise unmodified at Phase 3.C spec D7:
`φ(x) = C(m) Γ(2/m, |x/x₀(m)|^m)`, `x = 0` handled in closed form as documented on
`SuperGaussian(m::Real)` above. Reached by every plain-`m` call, by a dual `m` outside
`SG_DM_TABLE`'s domain or with an identically-zero partial, and by the generator's own
`SuperGaussian{FT,0,0}` (`_sg_call`'s dispatch below).
"""
function _sg_call_generic(f::SuperGaussian, x::Real)
    iszero(ForwardDiff.value(x)) && return f.C * gamma(2 / f.m) * one(x)
    return f.C * _upper_gamma(2 / f.m, abs(x / f.x0)^f.m)
end

"""
    _sg_call(f::SuperGaussian, x::Real) -> Real

The call dispatch of Phase 3.C spec D7. Three methods: the plain fallback below, always
`_sg_call_generic`; the `LM = 0` override just under it, always `_sg_call_generic` too, so that
a shape with no dm-slice — a plain `m`, `_widen_shape`'s zero-partial output, or the generator's
own `SuperGaussian{FT,0,0}` — takes the Phase 1.SG path **by dispatch**, with no runtime check
and no possibility of reading an empty `dm_slice` (D6 item 3, Pitfall 7); and the fast method
below that, for every other `(D<:ForwardDiff.Dual, L, LM)` combination, which is where `LM ≠ 0`
is guaranteed to hold and the m-partial table read is safe.
"""
_sg_call(f::SuperGaussian, x::Real) = _sg_call_generic(f, x)

_sg_call(f::SuperGaussian{D,L,0}, x::Real) where {D<:ForwardDiff.Dual,L} = _sg_call_generic(f, x)

"""
    _sg_nested_dual(::Type{<:ForwardDiff.Dual}) -> Bool

`true` for a **second**-order dual in `m` (a `Dual` whose own value type is itself a `Dual`),
`false` otherwise. A type-level test, costing nothing at runtime. Phase 3.C spec D7's fast call
method assembles a **first-order** rule (`dm` below is a plain number), so a second derivative in
`m` taken through it would be silently zero; a nested dual therefore falls back to
`_sg_call_generic`, whose `_dgamma_da` differentiates correctly to every order, slowly. Nothing in
the package takes a second derivative in `m` today (DM15 is the safety net against a future one
being silently wrong, not a product).
"""
_sg_nested_dual(::Type{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = V <: ForwardDiff.Dual

"""
    _sg_call(f::SuperGaussian{D,L,LM}, x::Real) where {D<:ForwardDiff.Dual,L,LM} -> Real

The fast dual-`m` assembly of Phase 3.C spec D7, reached only when `LM ≠ 0` (the method just
above intercepts every `LM = 0` case). Three fallbacks to `_sg_call_generic`, each deliberate: an
`m` whose partials are identically zero (a shape widened for another leaf's derivative — this
keeps RF1, GA2 and every Phase 2 dual row bitwise unmoved, since the generic path's own
`iszero(ForwardDiff.partials(a))` guard already skips the quadrature there); a **second**-order
dual in `m` (`_sg_nested_dual`, above); and `m` or `|x|` outside `SG_DM_TABLE`'s domain, read from
the table object itself, never duplicated ([OQ-31](../../dev/open_questions.md#oq-31) item 3).

**The fast assembly**, with `m0`, `x0v`, `Cv` and `xv` all plain floats of the primal type: `val`
is `φ` at the primal `(x0v, Cv, m0)`, through `_upper_gamma`'s plain `(Real, Real)` method so that
φ's value has one source (D49, 2026-09-25) — that method is `gamma_inc(a, t)[2] * gamma(a)`, not a
quadrature, so no quadrature runs on this path at all; `dx` is `dφ/dx` in the same closed form
`dfdx` uses, never the chain rule through `Γ` ([OQ-9](../../dev/open_questions.md#oq-9) trap 1: that path is
`∞·0` at `x = 0`); `dm` is `∂φ/∂m` read out of `SG_DM_TABLE`'s slice by one Clenshaw evaluation
in `|x|`. **The last line is the whole trick and is not a flourish**: `x - xv` has value exactly
zero and partials exactly `x`'s, and `f.m - m0` likewise for `m`, so the returned expression has
value exactly `val` and partials exactly `dx·∂x + dm·∂m` — every promotion, tag and nesting
question is answered by ForwardDiff's own arithmetic, never by a hand-built `Dual` constructor
(where a plain `x` with a dual `m`, a dual `x` and a dual `m` of the same tag, and two different
tags are three cases that a hand-assembly can get one of wrong). It allocates nothing: every term
is a scalar.

`x = 0` is branched **first, on the primal value** (`_primal_value`, which recurses through
nested duals, unlike the generic path's own `iszero(ForwardDiff.value(x))` guard —
[OQ-9](../../dev/open_questions.md#oq-9) trap 2), in both `val` and `dx`. `dm` needs no branch:
`x = 0` is inside the table's first panel and the table carries it. `dx` agrees with the generic
path's t-partial analytically (`C·(−t^{a−1}e^{−t})·∂t/∂x = −(Cm/x₀²) x e^{−t}`) but not
necessarily bitwise, which is why DM8 asserts `1e-9`, not equality.

Phase 3.C spec D6, D7; `docs/derivations/supergaussian.md`.
"""
function _sg_call(f::SuperGaussian{D,L,LM}, x::Real) where {D<:ForwardDiff.Dual,L,LM}
    m0 = ForwardDiff.value(f.m)
    ax = abs(_primal_value(x))
    (iszero(ForwardDiff.partials(f.m)) || _sg_nested_dual(D) ||
        !(SG_DM_TABLE.m_lo <= m0 <= SG_DM_TABLE.m_hi) || ax > SG_DM_TABLE.xi_max) &&
        return _sg_call_generic(f, x)

    x0v = ForwardDiff.value(f.x0)
    Cv = ForwardDiff.value(f.C)
    xv = _primal_value(x)
    val = iszero(xv) ? Cv * gamma(2 / m0) : Cv * _upper_gamma(2 / m0, (ax / x0v)^m0)
    dx = iszero(xv) ? zero(val) : -(m0 * Cv / x0v^2) * xv * exp(-(ax / x0v)^m0)
    dm = _sg_table_eval(SG_DM_TABLE, f.dm_slice, ax)
    return val + dx * (x - xv) + dm * (f.m - m0)
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

The Cauchy integral of the shape's derivative (`conventions.md` 1.5), in its final form (spec
D14, Step 1.SG.3): off the real axis, `Im ξ < 0` throws and `Im ξ > 0` uses direct quadrature,
both via `_sg_cauchy_reference` (D18) — the forward model never evaluates off the real axis, so
this branch exists for tests and for Phase 4. For real `ξ`, three branches in order: **(1)**
inside the table's domain — `|ξ| ≤ SG_TABLE.xi_max` and `SG_TABLE.m_lo ≤ m ≤ SG_TABLE.m_hi`, both
read from the table object and never duplicated (`SG_TABLE` is the only place the domain lives) —
`_sg_table_eval(SG_TABLE, f.slice, |ξ|)` plus the exact Plemelj term `iπ φ'(ξ)`, used **even where
the large-|ξ| series would also converge** (the join at `|ξ| = xi_max` is a test, DP7/TB10, not a
blend); **(2)** outside the table's domain, the large-|ξ| Laurent series (`_cauchy_series`) if it
converges, plus the same Plemelj term; **(3)** otherwise, the reference quadrature
`_sg_cauchy_reference`. **`m` outside `[m_lo, m_hi]` is not an error**: the fallback runs,
correctly, about 60 µs a call, against about 50 ns through the table.

Every branch decision is made on a primal value: comparisons on a `ForwardDiff.Dual` already
compare values, so `|ξ| ≤ xi_max` and `m_lo ≤ m ≤ m_hi` need no help; only `_sg_table_eval`'s panel
index is built from a primal (`_primal_value`, inside that function, spec 5.12).

`conventions.md` 1.5; spec D14; `docs/derivations/supergaussian.md`.
"""
function cauchy_dfdv(f::SuperGaussian, xi::Number)
    if !iszero(imag(xi))
        return _sg_cauchy_reference(f, xi)
    end
    rxi = real(xi)
    axi = abs(rxi)
    if axi <= SG_TABLE.xi_max && SG_TABLE.m_lo <= f.m <= SG_TABLE.m_hi
        QT = _basefloat(rxi)
        return _sg_table_eval(SG_TABLE, f.slice, axi) + im * QT(pi) * dfdx(f, rxi)
    end
    series_val, converged = _cauchy_series(f, rxi)
    if converged
        QT = _basefloat(rxi)
        return series_val + im * QT(pi) * dfdx(f, rxi)
    end
    return _sg_cauchy_reference(f, rxi)
end

"""
    _widen_shape(f::SuperGaussian{FT0,L,LM}, ::Type{FT}) where {FT0,L,LM,FT} -> SuperGaussian{FT,L,LM}

Widens by **copying** the cached fields, never re-deriving them: widening is a type promotion,
not a value change, so `FT(x)` on a `Float64` `x` into a `Dual` is exact and the partials it
carries are genuinely zero. Specializes the generic `_widen_shape(f::AbstractVelocityDistribution,
::Type{FT})` (`src/plasma/trace.jl`), which instead calls `reconstruct` and so re-derives `x0(m)`,
`C(m)` and every slice entry in dual arithmetic for partials that are identically zero (measured,
spec D14: 12.8 µs per call, about 11% of an 18.5 ms `ForwardDiff.derivative` pass on a 128 × 32
streak with five `GradientSmearing` nodes). **Copies both slices** (Phase 3.C spec D6: forgetting
`dm_slice` would make exactly this re-derivation happen for partials that are identically zero,
the same mistake in reverse). `L = 0` and `LM = 0` (the slice-free form `_sg_uncached` builds, and
every plain-`m` shape) widen through the same line with no special case: `NTuple{0,FT}(())` is the
empty tuple either way, and preserving `LM = 0` through the widen is exactly what keeps `_sg_call`'s
`LM = 0` override reachable afterward (D6 item 3).

Spec D14 (roadmap Step 3.1a, carried from [OQ-31](../../dev/open_questions.md#oq-31) item 6);
Phase 3.C spec D6.
"""
_widen_shape(f::SuperGaussian{FT0,L,LM}, ::Type{FT}) where {FT0,L,LM,FT} =
    SuperGaussian{FT,L,LM}(FT(f.m), FT(f.x0), FT(f.C),
                           NTuple{L,FT}(FT.(f.slice)), NTuple{LM,FT}(FT.(f.dm_slice)))

"""
    parameters(f::SuperGaussian) -> NamedTuple

`(m = f.m,)`: the exponent alone, the shape's one fittable parameter. The cached `x0`, `C`,
`slice` and `dm_slice` are derived state, not parameters (spec D3).

`conventions.md` Section 4.
"""
parameters(f::SuperGaussian) = (m = f.m,)

"""
    reconstruct(f::SuperGaussian, values::NamedTuple) -> SuperGaussian

A fresh `SuperGaussian(values.m)`, recomputing every cached field from `m`; `f` itself is not
read. Dispatches to `SuperGaussian(m::Real)` or `SuperGaussian(m::ForwardDiff.Dual)` exactly as
`values.m`'s own type demands, so a plain `m` gives `LM = 0` and a dual one the full `dm_slice`
with no special case (Phase 3.C spec D6) — precisely the behaviour a fit needs. Allocation-free
(TB12, DM14).

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
arguments, so the both-dual method combines the value from `gamma_inc`, the a-partial from the
quadrature `_dgamma_da`, and the t-partial from the elementary `∂Γ(a,t)/∂t = −t^{a−1}e^{−t}`
(guarded at `t = 0`, where that elementary form is singular for `a < 1`, since `t = 0` there is
never actually reached with a nonzero t-partial — the call method's own `x = 0` guard routes
around this method entirely).

**The value comes from `gamma_inc(a, t)[2] * gamma(a)`, not `SpecialFunctions.gamma(a, t)`
directly** (D49, 2026-09-25): `gamma(a, t)` (SpecialFunctions 2.9.0) computes `t^a *
expint(1 - a, t)`, and `expint`'s "ν near an integer" branch (`src/expint.jl`) is wrong in a window
of `t` about 0.003 wide for **every `a < 1/2` (`m > 4`)**; `m < 4` is unaffected. The edge is sharp:
7.8e-14 at m = 3.99, 5.5% at m = 4.01, falling to 1.9% at m = 5 and 7.8e-3 at m = 6 (corrected
2026-09-25, chunk 10: the fault does not need `a` near a half-integer, and the "227 of 1301" figure
first quoted here was an undersampling at an x step of 0.005, which missed most windows).
`gamma_inc(a, t)` returns the regularized `(P, Q)` pair
by a different algorithm (DiDonato and Morris) and agrees with a `quadgk` oracle to 1.5e-14 worst
case over `m ∈ [1.5, 8] × x ∈ [0.1, 3]`; SG22 is the regression that checks it, through both call
paths. **A library special function is never its own test oracle** (`conventions.md` 8.2): SG17
originally compared this function's value against `gamma(a, t)` itself and so could not see the
fault.

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
partials keeps its quadrature** — that is design (a), correct, and remains exactly this method's
business. It was the next bottleneck for fits with `m` free (OQ-9 trap 1, profiled at Phase 3's
close: 93% of the EPW Jacobian). **Phase 3.C spec D7 does not touch this function**: it adds a
faster *route around* it instead, `SuperGaussian`'s `_sg_call` dispatching a dual `m` inside
`SG_DM_TABLE`'s domain to a table read instead of reaching `_sg_call_generic` (and hence this
method and `_dgamma_da`'s quadrature) at all. Every call that still arrives here — a plain `m`, an
`m` outside the table's domain, or a nested dual in `m` — is served exactly as before, unchanged
and unsped-up.

Five methods on purpose (plain, both-dual-same-tag with its own body, `(Real, Dual)` with its own
body, and the two promoting methods for a plain second argument and two duals of different tags):
do not collapse them (spec D5, verified to dispatch without ambiguity — the `where {D<:Dual}`
methods are diagonal and therefore more specific than the promoting ones, so they terminate the
recursion once the arguments share a type).

PACM's incomplete gamma; spec D5; `docs/derivations/supergaussian.md`.
"""
_upper_gamma(a::Real, t::Real) = gamma_inc(a, t)[2] * gamma(a)

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

**This quadrature is `SG_DM_TABLE`'s reference and generator, and stays in the package as the
documented fallback outside the table's domain** (Phase 3.C spec D8), exactly as `_cauchy_real`
is for `SG_TABLE`: it is not deleted and it is not sped up. `_sg_dphi_dm_reference`, below, is what
calls it (through `_upper_gamma`'s both-dual method, through `_sg_call_generic`) to build and
certify the table.

Spec D5; Phase 3.C spec D8; `docs/derivations/supergaussian.md`.
"""
function _dgamma_da(a::Real, t::Real)
    iszero(t) && return gamma(a) * digamma(a)
    QT = _basefloat(a)
    val, _ = quadgk(y -> (t + y)^(a - 1) * log(t + y) * exp(-(t + y)), zero(QT), QT(1), QT(Inf);
                     rtol = 1e-12, atol = 1e-300)
    return val
end

"""
    _sg_dphi_dm_reference(m, x) -> Real

`SG_DM_TABLE`'s sampler, generator and reference (Phase 3.C spec D8), exactly as
`_sg_principal_value` is for `SG_TABLE`. `ForwardDiff.derivative` differentiates the *uncached*
call `_sg_uncached(mm)(x)` in `m`: `_sg_uncached(mm)` is `SuperGaussian{FT,0,0}`, so `_sg_call`'s
fast method (defined only away from `LM = 0`) does not apply and the call takes
`_sg_call_generic`, whose `_upper_gamma` both-dual method runs `_dgamma_da`'s quadrature. This has
**one provenance**, exactly `_sg_principal_value`'s: it never touches either table
([OQ-31](../../dev/open_questions.md#oq-31) item 1, Pitfall 2, Phase 3.C Pitfall 7), so
**neither table's build needs the other** — `SG_TABLE` is still bound first in
`supergaussian_table.jl` (Phase 3.C spec D9), but for an unrelated reason:
`SuperGaussian(m::Real)`'s promotion reads `eltype(SG_TABLE.coeffs)`.

Sampler argument order is `sampler(m, x)`, matching `_sg_principal_value(m, xi)`
(Phase 3.C spec D9).

**Checked by hand against two closed forms before any table is built from it** (spec D8, Step
3.C.4 build order item 2): the `m = 2` closed form `Γ(1,t) = e^{−t}`,
`Γ_a(1,t) = e^{−t}·log t + E₁(t)` (`SpecialFunctions.expint`), `x₀(2) = 1`, `C(2) = 1/√π`; and the
`x = 0` closed form `C′(m)Γ(a) − (2C(m)/m²)Γ(a)ψ(a)`. The full assembly, in the code's own symbols
with `a = 2/m`, `t = |x/x₀(m)|^m`, `ψ = digamma` (D8):

    dphi/dm  = C'(m)*Gamma(a,t) + C(m)*[Gamma_a(a,t)*(-2/m^2) + Gamma_t(a,t)*dt/dm]
    Gamma_t  = -t^(a-1)*exp(-t)
    dt/dm    = t*(log|x/x0(m)| - m*x0'(m)/x0(m))
    x0'(m)/x0(m) = (5*psi(5/m) - 3*psi(3/m)) / (2*m^2)
    C'(m)/C(m)   = (9*psi(3/m) - 5*psi(5/m)) / (2*m^2)

— not implemented as a formula in `src/` at all: `ForwardDiff.derivative` produces it from the
call method above, term for term, with no risk of a hand-transcription error. `x0'/x0` and `C'/C`
are given here so that a test oracle (which must not share machinery with this function,
`conventions.md` 8.2) can implement them independently.

Phase 3.C spec D3, D8; `docs/derivations/supergaussian.md`.
"""
_sg_dphi_dm_reference(m, x) = ForwardDiff.derivative(mm -> _sg_uncached(mm)(x), m)
