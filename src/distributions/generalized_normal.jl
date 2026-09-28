# The GeneralizedNormal shape: the plain one-dimensional super-Gaussian C exp(-|x/x0|^m) of other
# fitting codes (Subbotin's generalized normal distribution), NOT OTIS's own SuperGaussian (which
# is the projection of a 3-D super-Gaussian). See docs/derivations/generalized_normal.md.
# Phase 4.B Step 4.B.3, spec D31-D34, D42.

using SpecialFunctions: loggamma

"""
    GeneralizedNormal{FT<:Real,L} <: AbstractVelocityDistribution

The plain one-dimensional super-Gaussian `C exp(−|x/x₀|^m)` of other fitting codes (Subbotin's
generalized normal distribution); **not** OTIS's `SuperGaussian`, which is the one-dimensional
marginal of an isotropic three-dimensional super-Gaussian. `m ∈ [1.5, 2)` is accepted but
unphysical for an isotropic plasma (the 3-D density would diverge at zero speed, `F(v) ∝
v^(m−2)`); it exists so that a fit's lower bound can sit below the Maxwellian (`m = 2`), and is
served by the quadrature fallback, about 100 to 400 µs a call (spec D30 item 3). Field `m` is the
shape parameter; `x0` and `C` are its own normalization constants `x₀(m) = √(Γ(1/m)/(2Γ(3/m)))`
and `C(m) = m/(2x₀Γ(1/m))` (spec D31), cached at construction. `slice` is `GN_TABLE`'s
ξ-coefficients at this shape's own `m`, computed for **every** `m ≥ 3/2` (including `m ∈ [1.5, 2)`
and `m > 8`, where the Chebyshev series is extrapolated and the slice is never read: `cauchy_dfdv`
checks the domain first).

**No second (φ-derivative) table**, exactly `Kappa`'s reason: φ is elementary (`exp`, `abs`, a
power) and its constants go through `loggamma`, whose ForwardDiff rule (`digamma`) is closed form,
so a dual `m` through the call is already quadrature-free and allocation-free
(`conventions.md` Section 4's derivative clause, [OQ-38](../../dev/open_questions.md#oq-38),
[OQ-42](../../dev/open_questions.md#oq-42) item 4). So there is no `LM` type parameter and no
`dm_slice`.

`conventions.md` Section 4, Section 11; spec D31; `docs/derivations/generalized_normal.md`.
"""
struct GeneralizedNormal{FT<:Real,L} <: AbstractVelocityDistribution
    m::FT
    x0::FT                # sqrt(Gamma(1/m) / (2 Gamma(3/m)))
    C::FT                 # m / (2 x0 Gamma(1/m))
    slice::NTuple{L,FT}   # GN_TABLE's xi-coefficients at m
end

"""
    _gn_x0(m) -> Real

`x₀(m)² = Γ(1/m) / (2 Γ(3/m))`, the shape's width constant, fixed by the variance normalization
`∫x²φ dx = 1/2` (`conventions.md` 1.5). `m = 2` gives `x₀ = 1` exactly (Level 2: `_gn_x0(2.0) ==
1.0` bitwise). Generic over `m`'s type: ForwardDiff differentiates `loggamma` through its own rule
(`digamma`), so a dual `m` gives a dual `x0` with no extra rule needed (spec D31).

Section 10.2; `docs/derivations/generalized_normal.md`.
"""
_gn_x0(m) = sqrt(exp(loggamma(1 / m) - loggamma(3 / m)) / 2)

"""
    _gn_C(m, x0) -> Real

`C(m) = m / (2 x₀(m) Γ(1/m))`, the shape's normalization constant, fixed by the unit integral
`∫φ dx = 1`. `m = 2` gives `C = 1/√π`, the Maxwellian.

Section 10.2; `docs/derivations/generalized_normal.md`.
"""
_gn_C(m, x0) = m * exp(-loggamma(1 / m)) / (2 * x0)

"""
    GeneralizedNormal(m::Real)
    (f::GeneralizedNormal)(x::Real) -> Real

The public, one-argument constructor (`conventions.md` Section 11), and the shape's call
overload. **Both are documented here because they share one doc-system key** (spec D31, Pitfall
21, `Kappa`'s reason): `GeneralizedNormal(m::Real)` and `(f::GeneralizedNormal)(x::Real)` are both
methods of the binding `GeneralizedNormal` with signature `Tuple{Real}`, so a second docstring on
the call collides with this one and is silently dropped, with a `Replacing docs` warning at
precompile.

**The shape.** `φ(x) = C(m) exp(−|x/x₀(m)|^m)`, dimensionless in and out. `m` must be finite and
`≥ 3/2` (below that the constructor throws, spec D30 item 3): a loud `ArgumentError` otherwise
(`conventions.md` 5.9).

**The constructor promotes `m` first** (Step 2.0's `HK1` form, spec D31, GN19): `FT =
promote_type(typeof(m), eltype(GN_TABLE.coeffs))`, then `x0 = _gn_x0(m_p)`, `C = _gn_C(m_p, x0)`
and `slice = _cheb_slice(GN_TABLE, m_p)` are all computed **at** the promoted `m_p`, never at `m`'s
own (possibly narrower) type and widened afterward. **One method, not two** (unlike
`SuperGaussian`'s plain/dual pair): every `m ≥ 3/2`, including `m ∈ [1.5, 2)` and `m > 8`, where the
Chebyshev series is extrapolated, gets the same `slice` computation — that slice is simply never
read there, since `cauchy_dfdv` checks the table's domain before using it. For very large `m`
(above about `10⁴`) the extrapolated entries can overflow to `Inf` or `NaN`; harmless for the same
reason.

**`x = 0`** is branched on `_primal_value(x)`, never on `x == 0` (spec D31, Pitfall 27): the call
returns `C · one(x)`, the `* one(x)` not decoration (rule 5.14, a branch on a runtime value returns
one type). **This guard is defensive**: ForwardDiff 1.4.5 already gives `0^m` the right m-partial
(Level 2, Section 10.2), and the guard keeps that from being a dependency (the discrimination
check's item 4 is expected to pass, unlike `SuperGaussian`'s guard).

**An `m` outside `GN_TABLE`'s domain is not an error**: `cauchy_dfdv` on it runs the Laurent series
if it converges, else the reference quadrature, correctly but slower (spec D31, D32); `m ∈ [1.5,
2)` runs the fallback at every `|ξ|` below about 6 to 9, where the series does not converge (about
100 to 420 µs a call, spec D30 item 3, D47), since the table stops at `m = 2` (the `|ξ|^(m−1)` term
at `ξ = 0` has infinite slope below it).

Section 10.2; spec D31; `docs/derivations/generalized_normal.md`.
"""
function GeneralizedNormal(m::Real)
    (isfinite(m) && _primal_value(m) >= 3 / 2) || throw(ArgumentError(
        "GeneralizedNormal: m must be finite and at least 1.5; got $m"))
    FT = promote_type(typeof(m), eltype(GN_TABLE.coeffs))
    mp = FT(m)
    x0 = _gn_x0(mp)
    C = _gn_C(mp, x0)
    slice = _cheb_slice(GN_TABLE, mp)
    return GeneralizedNormal(mp, FT(x0), FT(C), slice)
end

"""
    _gn_uncached(m::Real) -> GeneralizedNormal{FT,0}

The generator's constructor, exactly `_kappa_uncached`'s role: computes `m`, `x0` and `C` as the
public constructor does, but sets `slice = ()`, giving a shape of the **distinct type**
`GeneralizedNormal{FT,0}`. This breaks the cycle `GN_TABLE`'s own build would otherwise create
(`GeneralizedNormal(m)` needs `GN_TABLE`, and building it needs shapes to evaluate its sampler):
`_gn_principal_value` calls through a shape built this way, and that path (`_gn_cauchy_reference`,
via `_cauchy_real`/`_cauchy_upper`) reads only `m`, `x0` and `C` — never `slice` nor `GN_TABLE`
(spec D31).

`cauchy_dfdv` on a `GeneralizedNormal{<:Real,0}` throws (below): a shape with a meaningless empty
slice can never be evaluated by accident.

Spec D31.
"""
function _gn_uncached(m::Real)
    x0 = _gn_x0(m)
    C = _gn_C(m, x0)
    FT = promote_type(typeof(m), typeof(x0), typeof(C))
    return GeneralizedNormal{FT,0}(FT(m), FT(x0), FT(C), ())
end

"""
    cauchy_dfdv(f::GeneralizedNormal{<:Real,0}, xi::Number)

Throws (spec D31): `f` is the uncached form `GN_TABLE`'s generator builds, with an empty,
meaningless `slice`. It must never be evaluated; construct the shape with `GeneralizedNormal(m)`
instead. More specific than the generic `cauchy_dfdv(::GeneralizedNormal, ::Number)` method below,
so a `GeneralizedNormal{<:Real,0}` always reaches this one first. It **must not** name `GN_TABLE`:
it runs while the table is being built.
"""
cauchy_dfdv(f::GeneralizedNormal{<:Real,0}, xi::Number) = throw(ArgumentError(
    "cauchy_dfdv(GeneralizedNormal{FT,0}, xi): this shape carries no table slice. It is the " *
    "uncached form the table's generator builds and must not be used for evaluation; construct " *
    "the shape with GeneralizedNormal(m)."))

# Documented above, on GeneralizedNormal(m::Real): the two share one doc-system key (spec Pitfall 21).
function (f::GeneralizedNormal)(x::Real)
    iszero(_primal_value(x)) && return f.C * one(x)
    return f.C * exp(-abs(x / f.x0)^f.m)
end

"""
    dfdx(f::GeneralizedNormal, x::Real) -> Real

`φ'(x) = −(m/x₀) sign(x) |x/x₀|^{m−1} φ(x)`, the closed form (Section 10.2). Dimensionless in and
out.

**The `x = 0` branch is load-bearing, unlike the call's** (spec D31, Pitfall 27): it returns
`φ″(0) · x`, which is `−2C/x₀²` at `m = 2` (the Maxwellian's `−2/√π`) and `0` for `m > 2` (for
`m < 2` the true `φ″(0)` is `−∞`, and `0` is returned harmlessly, since only a dual `ξ` at exactly
zero with `m < 2` could see it). Without this branch, `sign(x) · t^(m−1)` under a dual `x` at `0`
carries the wrong partial (`0` instead of `−2C` at `m = 2`, GN4). The test `m == 2` is on the
primal value (spec D47): ForwardDiff 1.x reads a dual `m` of exactly 2 as unequal to 2.

Section 10.2; spec D31; `docs/derivations/generalized_normal.md`.
"""
function dfdx(f::GeneralizedNormal, x::Real)
    iszero(_primal_value(x)) && return (_primal_value(f.m) == 2 ? -2 * f.C / f.x0^2 : zero(f.C)) * x
    t = abs(x / f.x0)
    return -(f.m / f.x0) * sign(x) * t^(f.m - 1) * f.C * exp(-t^f.m)
end

"""
    (f::GeneralizedNormal)(x::Complex)
    dfdx(f::GeneralizedNormal, x::Complex)

Throw: `|x|^m` has no single analytic continuation off the real axis, so the shape is defined for
real `x` only (spec D31; rule 5.13, a split override, never a single `::Number` method).
"""
(f::GeneralizedNormal)(x::Complex) = throw(ArgumentError(
    "(f::GeneralizedNormal)(x::Complex) is not available: |x|^m has no single analytic continuation off the real axis, so the shape is defined for real x only (Phase 4.B spec D31)."))
dfdx(f::GeneralizedNormal, x::Complex) = throw(ArgumentError(
    "dfdx(GeneralizedNormal, ::Complex) is not available: |x|^m has no single analytic continuation off the real axis, so the shape is defined for real x only (Phase 4.B spec D31)."))

"""
    moment(f::GeneralizedNormal, n::Integer) -> Real

The `n`-th moment `∫ xⁿ φ(x) dx`, closed form for every integer `n ≥ 0` (Section 10.2), all
finite (unlike `Kappa`'s):

    M_{2j}   = x₀^{2j} Γ((2j+1)/m) / Γ(1/m) = x₀^{2j} exp(loggamma((2j+1)/m) - loggamma(1/m)),
    M_{2j+1} = 0.

Both returns are of type `typeof(f.m)` (rule 5.14, a branch on a runtime value returns one type).
Throws for `n < 0`. Needed to `n ≈ 39` by the Laurent series of `_cauchy_series`
(`conventions.md` Section 4).

Section 10.2; spec D31; `docs/derivations/generalized_normal.md`.
"""
function moment(f::GeneralizedNormal, n::Integer)
    n >= 0 || throw(ArgumentError("moment(GeneralizedNormal, n): n must be non-negative; got $n"))
    isodd(n) && return zero(f.m)
    return f.x0^n * exp(loggamma((n + 1) / f.m) - loggamma(1 / f.m))
end

"""
    _gn_cauchy_reference(f::GeneralizedNormal, xi::Number) -> Complex

The reference Cauchy integral, exactly `_sg_cauchy_reference`'s and `_kappa_cauchy_reference`'s
twin: direct quadrature (`_cauchy_upper`) for `Im ξ > 0`, the Plemelj principal-value quadrature
(`_cauchy_real`) for real `ξ`, and an error for `Im ξ < 0` (spec D31): the residue continuation
`_cauchy_lower` relies on needs a complex-valued `φ'`, which this shape's `dfdx` does not provide
off the real axis (`|x|^m` has no single analytic continuation there). Correct, and slow, for
every `m ≥ 3/2` and every `ξ` in the closed upper half-plane. This is `GN_TABLE`'s own generator
(through `_gn_principal_value`) and every reference-level test's oracle.

Spec D31, D32; `docs/derivations/generalized_normal.md`.
"""
function _gn_cauchy_reference(f::GeneralizedNormal, xi::Number)
    if imag(xi) < 0
        throw(ArgumentError(
            "cauchy_dfdv(GeneralizedNormal, xi) requires Im xi >= 0; got $xi. |x|^m has no single " *
            "analytic continuation off the real axis, so the residue continuation of the generic " *
            "fallback does not apply."))
    elseif imag(xi) > 0
        return _cauchy_upper(f, xi)
    else
        return _cauchy_real(f, real(xi))
    end
end

"""
    cauchy_dfdv(f::GeneralizedNormal, xi::Number) -> Complex

The Cauchy integral of the shape's derivative (`conventions.md` 1.5), `SuperGaussian`'s branch
order (unlike `Kappa`'s, which has no series branch): off the real axis, `_gn_cauchy_reference`
(the forward model never evaluates there; kept for tests and symmetry). For real `ξ`, three
branches in order: **(1)** inside `GN_TABLE`'s domain (`m ∈ [p_lo, p_hi]` and `|ξ| ≤ xi_max`, both
read from the table object) — `_cheb_table_eval(GN_TABLE, f.slice, |ξ|)` plus the exact Plemelj
term `iπ φ'(ξ)`; **(2)** outside it, the large-`|ξ|` Laurent series (`_cauchy_series`) if it
converges (every moment of this shape is finite, unlike `Kappa`'s); **(3)** otherwise, the
reference quadrature `_gn_cauchy_reference`. `m ∈ [1.5, 2)` or `m > 8` outside the series' reach is
not an error: the fallback runs, correctly, about 100 to 420 µs a call below `m = 2` and 25 to 70
µs above `m = 8` (spec D30 item 3, D32).

**R is even in ξ** (`φ` is even, so `φ'` is odd and the principal value `R` is even): the table
holds `ξ ≥ 0` and this function passes `abs(ξ)` to `_cheb_table_eval` (D13's contract for an even
table). **Both domain comparisons are on primal values** (`_primal_value`): ForwardDiff 1.x orders
a dual at a tie by its partials, so a dual `m` exactly at `p_hi = 8` with a positive partial would
otherwise leave the table for the fallback, about 430× slower (Phase 4.B spec D47).

`conventions.md` 1.5; spec D31, D32; `docs/derivations/generalized_normal.md`.
"""
function cauchy_dfdv(f::GeneralizedNormal, xi::Number)
    iszero(imag(xi)) || return _gn_cauchy_reference(f, xi)
    rxi = real(xi)
    axi = abs(rxi)
    QT = _basefloat(rxi)
    if _primal_value(axi) <= GN_TABLE.xi_max && GN_TABLE.p_lo <= _primal_value(f.m) <= GN_TABLE.p_hi
        return _cheb_table_eval(GN_TABLE, f.slice, axi) + im * QT(pi) * dfdx(f, rxi)
    end
    series_val, converged = _cauchy_series(f, rxi)
    converged && return series_val + im * QT(pi) * dfdx(f, rxi)
    return _gn_cauchy_reference(f, rxi)
end

"""
    _widen_shape(f::GeneralizedNormal{FT0,L}, ::Type{FT}) where {FT0,L,FT} -> GeneralizedNormal{FT,L}

Widens by **copying** the cached fields, never re-deriving them, exactly `Kappa`'s reason (Phase
3's A9): the generic `_widen_shape` re-derives `x0(m)`, `C(m)` and every slice entry through
`reconstruct` in dual arithmetic for partials that are identically zero. `L = 0` (the slice-free
form `_gn_uncached` builds) widens through the same line with no special case.

Spec D31 (Phase 3's A9, `src/plasma/trace.jl`'s `_widen_shape` generic); its deletion is an
expected pass (D34 item 8's reason: one constructor serves plain and dual `m`).
"""
_widen_shape(f::GeneralizedNormal{FT0,L}, ::Type{FT}) where {FT0,L,FT} =
    GeneralizedNormal{FT,L}(FT(f.m), FT(f.x0), FT(f.C), NTuple{L,FT}(FT.(f.slice)))

"""
    parameters(f::GeneralizedNormal) -> NamedTuple

`(m = f.m,)`: the shape parameter alone. The cached `x0`, `C` and `slice` are derived state, not
parameters.

`conventions.md` Section 4.
"""
parameters(f::GeneralizedNormal) = (m = f.m,)

"""
    reconstruct(f::GeneralizedNormal, values::NamedTuple) -> GeneralizedNormal

A fresh `GeneralizedNormal(values.m)`, recomputing every cached field from `m`; `f` itself is not
read. Allocation-free (`_cheb_slice` builds an `NTuple` with `Val(L)`).

`conventions.md` Section 4.
"""
reconstruct(::GeneralizedNormal, values::NamedTuple) = GeneralizedNormal(values.m)

"""
    Base.show(io::IO, f::GeneralizedNormal)

Single line, `GeneralizedNormal(m = <value>)` — the cache is derived state, not a parameter, and
is never printed.
"""
Base.show(io::IO, f::GeneralizedNormal) = print(io, "GeneralizedNormal(m = ", f.m, ")")

"""
    GN_PANEL_EDGES, GN_M_LO, GN_M_HI, GN_FIT_TOL, GN_DEGREE_START, GN_DEGREE_CAP_XI,
    GN_DEGREE_CAP_M, GN_BISECT_MAX

`GN_TABLE`'s build constants (spec D32). **The coordinate is `m`**, domain `m ∈ [GN_M_LO,
GN_M_HI] = [2, 8] × |ξ| ≤ 10`: the table stops at `m = 2` because the `|ξ|^{m−1}` term of `R(ξ) −
R(0)` near `ξ = 0` has a coefficient that vanishes at `m = 2` and an infinite slope below it, so
the table converges on `[2, 8]` and not below (Section 10.2). `m ∈ [1.5, 2)` and `m > 8` are valid
shapes and run the series or the fallback (D30 item 3). The small first panels
(`GN_PANEL_EDGES`'s `1/32, 1/16, 1/8, 1/4, 1/2`) resolve exactly that `|ξ|^{m−1}` corner near
`ξ = 0`. `GN_FIT_TOL = 1e-9`, the value target with one order of margin (rule 8.9), **and it is
not lowered**: a table direction that does not converge is a stop (D18's rule, restated for this
table).

Spec D32.
"""
const GN_PANEL_EDGES = (0.0, 1 / 32, 1 / 16, 1 / 8, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 10.0)
const GN_M_LO = 2.0
const GN_M_HI = 8.0
const GN_FIT_TOL = 1e-9
const GN_DEGREE_START = 8
const GN_DEGREE_CAP_XI = 64
const GN_DEGREE_CAP_M = 48
const GN_BISECT_MAX = 4

"""
    _gn_principal_value(m, xi) -> Real

`GN_TABLE`'s sampler, in the general `sampler(p, x)` order `_build_cheb_table` expects: `R(ξ; m)`,
the real part of the reference Cauchy integral, evaluated on the *uncached* form
`_gn_uncached(m)` — no slice, so the table's generator never depends on the table it is building
(spec D31). **At `m = GN_M_LO = 2`** the generator is this shape's own quadrature
(`_gn_uncached(2.0)` is the Maxwellian to rounding), **not `Zprime`** — so that `Zprime` stays an
independent oracle for GN11 at that edge (spec D32, the same reason as `KAPPA_TABLE`'s: an oracle
must not leak into the generator).

Spec D32.
"""
_gn_principal_value(m, xi) = real(_gn_cauchy_reference(_gn_uncached(m), xi))

"""
    GN_TABLE::ChebyshevTable

The module-level table `const` (`conventions.md` 5.6), built once by `_build_cheb_table` at
include time and read-only afterwards, tabulating `R(ξ; m)` on `m ∈ [2, 8] × |ξ| ∈ [0, 10]` from
`_gn_principal_value`. `GeneralizedNormal(m)`'s public constructor computes its `slice` from this
table for every `m ≥ 3/2`.

Spec D32.
"""
const GN_TABLE = _build_cheb_table(_gn_principal_value, GN_PANEL_EDGES, GN_FIT_TOL,
                                   GN_M_LO, GN_M_HI;
                                   degree_start=GN_DEGREE_START, cap_xi=GN_DEGREE_CAP_XI,
                                   cap_p=GN_DEGREE_CAP_M, bisect_max=GN_BISECT_MAX)
