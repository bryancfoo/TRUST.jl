# The SkewNormal shape: Azzalini's skew-normal velocity shape along k̂, (2/omega) phi(z) Phi(alpha*z)
# with z = (x - x_c)/omega, located and scaled so that its mean is 0 and its variance is 1/2; NOT a
# Hermite or Gram-Charlier skew, and positive everywhere. See docs/derivations/skew_normal.md.
# Phase 4.B Step 4.B.4, spec D35-D38, D42, D47.

using SpecialFunctions: erfc

"""
    SkewNormal{FT<:Real,L} <: AbstractVelocityDistribution

Azzalini's skew-normal velocity shape along k̂, one shape parameter `alpha`: `φ(x) = (2/ω) ϕ(z)
Φ(αz)` with `z = (x − x_c)/ω`, located and scaled so that its mean is 0 and its variance is ½; the
Maxwellian at `α = 0`, skewed along **k̂** for `α ≠ 0`; **not** a Hermite or Gram–Charlier skew (it
is positive everywhere; those are not) (Section 10.3, `docs/derivations/skew_normal.md`). `α > 0`
skews the tail toward **positive** `x`, i.e. along `+k̂`. Field `alpha` is the shape parameter;
`omega` and `xc` are its own normalization constants `ω(α) = 1/√(2v_δ(α))` and `x_c(α) = −ω μ_δ(α)`
(spec D35), cached at construction. `slice` is `SN_TABLE`'s ξ-coefficients at this shape's own `α`,
computed for **every** finite `α` (including `|α| > 3`, where the Chebyshev series is extrapolated
and the slice is never read: `cauchy_dfdv` checks the domain first).

**No second (φ-derivative) table**: φ is built from `exp` and `erfc` alone, and ForwardDiff
differentiates both through DiffRules' closed-form rules (`erfc`'s is `−2 exp(−x²)/√π`), so a dual
`α` through the call is already quadrature-free and allocation-free (`conventions.md` Section 4's
derivative clause, [OQ-38](../../dev/open_questions.md#oq-38)). So there is no `LM` type parameter
and no `dalpha_slice`, exactly `Kappa`'s and `GeneralizedNormal`'s reason.

`conventions.md` Section 4, Section 11; spec D35; `docs/derivations/skew_normal.md`.
"""
struct SkewNormal{FT<:Real,L} <: AbstractVelocityDistribution
    alpha::FT
    omega::FT             # 1 / sqrt(2 v_delta): the scale that makes the variance 1/2
    xc::FT                # -omega * mu_delta: the location that makes the mean 0
    slice::NTuple{L,FT}   # SN_TABLE's xi-coefficients at alpha, over SIGNED xi
end

"""
    _std_normal_pdf(z) -> Real

`ϕ(z) = exp(−z²/2)/√(2π)`, the standard normal density, taken at `z`'s own base float's `π`
(`_basefloat`) so a dual `z` stays generic.

Section 10.3; `docs/derivations/skew_normal.md`.
"""
_std_normal_pdf(z) = exp(-z^2 / 2) / sqrt(2 * _basefloat(z)(pi))

"""
    _std_normal_cdf(y) -> Real

`Φ(y) = ½ erfc(−y/√2)`, the standard normal CDF **through `erfc`, a library special function,
never its own oracle** (`conventions.md` 8.2): SK3 checks it against MPFR's `erfc` on a
0.003-wide grid.

Section 10.3; `docs/derivations/skew_normal.md`.
"""
_std_normal_cdf(y) = erfc(-y / sqrt(_basefloat(y)(2))) / 2

"""
    _sn_constants(alpha) -> (omega, xc)

`δ = α/√(1+α²)`, `μ_δ = δ√(2/π)`, `v_δ = 1 − 2δ²/π`, `ω = 1/√(2v_δ)`, `x_c = −ωμ_δ` (Section 10.3):
the location and scale that make the standard skew-normal `Z`'s affine image `X = x_c + ωZ` have
mean 0 and variance ½. **`α = 0` gives `δ = 0`, `ω = 1/√2`, `x_c = 0`** (`SkewNormal(0.0)` is the
Maxwellian to rounding, SK4). Generic over `α`'s type: every operation is elementary, so a dual
`α` gives dual `ω`, `x_c` with no extra rule needed.

Section 10.3; `docs/derivations/skew_normal.md`.
"""
function _sn_constants(alpha)
    QT = _basefloat(alpha)
    delta = alpha / sqrt(1 + alpha^2)
    mu = delta * sqrt(2 / QT(pi))
    v = 1 - 2 * delta^2 / QT(pi)
    omega = 1 / sqrt(2 * v)
    return omega, -omega * mu
end

"""
    SkewNormal(alpha::Real)
    (f::SkewNormal)(x::Real) -> Real

The public, one-argument constructor (`conventions.md` Section 11), and the shape's call overload.
**Both are documented here because they share one doc-system key** (spec D35, Pitfall 21, exactly
`Kappa`'s and `GeneralizedNormal`'s reason): `SkewNormal(alpha::Real)` and
`(f::SkewNormal)(x::Real)` are both methods of the binding `SkewNormal` with signature
`Tuple{Real}`, so a second docstring on the call collides with this one and is silently dropped,
with a `Replacing docs` warning at precompile.

**The shape.** `φ(x) = (2/ω) ϕ(z) Φ(αz)` with `z = (x − x_c)/ω` (Section 10.3), Azzalini's
skew-normal, located and scaled to mean 0 and variance ½; positive everywhere, unlike a Hermite or
Gram–Charlier skew. `α` must be finite: a loud `ArgumentError` otherwise (`conventions.md` 5.9).
**Any finite `α`, no domain limit** (unlike `Kappa`'s `κ > 3/2` or `GeneralizedNormal`'s `m ≥
3/2`): `|α| > 3` runs the fallback (D30 item 4), correct and slow.

**The constructor promotes `alpha` first** (Step 2.0's `HK1` form, spec D35, SK19): `FT =
promote_type(typeof(alpha), eltype(SN_TABLE.coeffs))`, then `omega, xc = _sn_constants(a)` and
`slice = _cheb_slice(SN_TABLE, a)` are all computed **at** the promoted `a`, never at `alpha`'s
own (possibly narrower) type and widened afterward. **One method, not two**: every finite `α`,
including `|α| > 3`, where the Chebyshev series is extrapolated, gets the same `slice`
computation — that slice is simply never read there, since `cauchy_dfdv` checks the table's domain
before using it. For `|α|` above about `10⁴` the extrapolated entries can overflow to `Inf` or
`NaN`; harmless for the same reason (D31's reasoning).

**φ is analytic and positive everywhere: no `x = 0` guard** is needed, unlike
`GeneralizedNormal`'s.

Section 10.3; spec D35; `docs/derivations/skew_normal.md`.
"""
function SkewNormal(alpha::Real)
    isfinite(alpha) || throw(ArgumentError("SkewNormal: alpha must be finite; got $alpha"))
    FT = promote_type(typeof(alpha), eltype(SN_TABLE.coeffs))
    a = FT(alpha)
    omega, xc = _sn_constants(a)
    slice = _cheb_slice(SN_TABLE, a)
    return SkewNormal(a, FT(omega), FT(xc), slice)
end

"""
    _sn_uncached(alpha::Real) -> SkewNormal{FT,0}

The generator's constructor, exactly `_kappa_uncached`'s and `_gn_uncached`'s role: computes
`alpha`, `omega` and `xc` as the public constructor does, but sets `slice = ()`, giving a shape of
the **distinct type** `SkewNormal{FT,0}`. This breaks the cycle `SN_TABLE`'s own build would
otherwise create (`SkewNormal(alpha)` needs `SN_TABLE`, and building it needs shapes to evaluate
its sampler): `_sn_principal_value` calls through a shape built this way, and that path
(`_sn_cauchy_reference`, via `_cauchy_real`/`_cauchy_upper`) reads only `alpha`, `omega` and `xc` —
never `slice` nor `SN_TABLE` (spec D35).

`cauchy_dfdv` on a `SkewNormal{<:Real,0}` throws (below): a shape with a meaningless empty slice
can never be evaluated by accident.

Spec D35.
"""
function _sn_uncached(alpha::Real)
    omega, xc = _sn_constants(alpha)
    FT = promote_type(typeof(alpha), typeof(omega), typeof(xc))
    return SkewNormal{FT,0}(FT(alpha), FT(omega), FT(xc), ())
end

"""
    cauchy_dfdv(f::SkewNormal{<:Real,0}, xi::Number)

Throws (spec D35): `f` is the uncached form `SN_TABLE`'s generator builds, with an empty,
meaningless `slice`. It must never be evaluated; construct the shape with `SkewNormal(alpha)`
instead. More specific than the generic `cauchy_dfdv(::SkewNormal, ::Number)` method below, so a
`SkewNormal{<:Real,0}` always reaches this one first. It **must not** name `SN_TABLE`: it runs
while the table is being built.
"""
cauchy_dfdv(f::SkewNormal{<:Real,0}, xi::Number) = throw(ArgumentError(
    "cauchy_dfdv(SkewNormal{FT,0}, xi): this shape carries no table slice. It is the uncached " *
    "form the table's generator builds and must not be used for evaluation; construct the shape " *
    "with SkewNormal(alpha)."))

# Documented above, on SkewNormal(alpha::Real): the two share one doc-system key (spec D35, Pitfall 21).
function (f::SkewNormal)(x::Real)
    z = (x - f.xc) / f.omega
    return (2 / f.omega) * _std_normal_pdf(z) * _std_normal_cdf(f.alpha * z)
end

"""
    dfdx(f::SkewNormal, x::Real) -> Real

`φ'(x) = (2/ω²) ϕ(z) [α ϕ(αz) − z Φ(αz)]`, the closed form (Section 10.3). Dimensionless in and
out. No `x = 0` guard: every factor is analytic there.

Section 10.3; spec D35; `docs/derivations/skew_normal.md`.
"""
function dfdx(f::SkewNormal, x::Real)
    z = (x - f.xc) / f.omega
    return (2 / f.omega^2) * _std_normal_pdf(z) *
           (f.alpha * _std_normal_pdf(f.alpha * z) - z * _std_normal_cdf(f.alpha * z))
end

"""
    (f::SkewNormal)(x::Complex)
    dfdx(f::SkewNormal, x::Complex)

Throw: the shape is defined for real `x` only (spec D35; rule 5.13, a split override, never a
single `::Number` method).
"""
(f::SkewNormal)(x::Complex) = throw(ArgumentError(
    "(f::SkewNormal)(x::Complex) is not available: the shape is defined for real x only (Phase 4.B spec D35)."))
dfdx(f::SkewNormal, x::Complex) = throw(ArgumentError(
    "dfdx(SkewNormal, ::Complex) is not available: the shape is defined for real x only (Phase 4.B spec D35)."))

"""
    moment(f::SkewNormal, n::Integer) -> Real

The `n`-th moment `∫ xⁿ φ(x) dx`, closed form for every integer `n ≥ 0` (Section 10.3), by a
three-term recursion derived from `φ'(x) = (2α/ω²) ϕ(z)ϕ(αz) − ((x−x_c)/ω²) φ(x)` and `∫xⁿφ' dx =
−nM_{n−1}`:

    M_0 = 1,   M_1 = 0 (exactly, by the mean normalization),
    M_{k+1} = x_c M_k + k ω² M_{k−1} + g h_k        (k ≥ 1),
    h_0 = 1,   h_1 = x_c,   h_{k+1} = x_c h_k + k τ² h_{k−1},
    s = 1/√(1+α²),   τ = ωs,   g = 2αωs/√(2π).

**`M₁` is set to exactly zero**, not computed from the recursion's own formula (which gives it to
rounding, about `1e-16`): a Laurent-series-style consumer must see a true zero (spec D35, SK6);
`SkewNormal` itself has no series branch (D36), so this matters only if a later shape reuses this
recursion's form. All finite, unlike `Kappa`'s. One return, `typeof(f.omega)`. Throws for `n < 0`.

Section 10.3; spec D35; `docs/derivations/skew_normal.md`.
"""
function moment(f::SkewNormal, n::Integer)
    n >= 0 || throw(ArgumentError("moment(SkewNormal, n): n must be non-negative; got $n"))
    a, w, xc = f.alpha, f.omega, f.xc
    s = 1 / sqrt(1 + a^2)
    tau2 = (w * s)^2
    g = 2 * a * w * s / sqrt(2 * _basefloat(a)(pi))
    m_prev, m_cur = zero(w), one(w)      # M_{k-1}, M_k, starting at k = 0
    h_prev, h_cur = zero(w), one(w)      # h_{k-1}, h_k
    for k in 0:n-1
        m_next = k == 0 ? zero(w) : xc * m_cur + k * w^2 * m_prev + g * h_cur
        h_next = xc * h_cur + k * tau2 * h_prev
        m_prev, m_cur = m_cur, m_next
        h_prev, h_cur = h_cur, h_next
    end
    return m_cur
end

"""
    _sn_cauchy_reference(f::SkewNormal, xi::Number) -> Complex

The reference Cauchy integral, exactly `_kappa_cauchy_reference`'s and `_gn_cauchy_reference`'s
twin: direct quadrature (`_cauchy_upper`) for `Im ξ > 0`, the Plemelj principal-value quadrature
(`_cauchy_real`) for real `ξ`, and an error for `Im ξ < 0` (spec D35): the residue continuation
`_cauchy_lower` relies on needs a complex-valued `φ'`, which this shape's `dfdx` does not provide
off the real axis. Correct, and slow, for every finite `α` and every `ξ` in the closed upper
half-plane. This is `SN_TABLE`'s own generator (through `_sn_principal_value`; at `α = 0` an
**independent** oracle, `_sn_uncached(0.0)` being the Maxwellian to rounding, so `Zprime` stays
independent for SK11) and every reference-level test's oracle.

Spec D35, D36; `docs/derivations/skew_normal.md`.
"""
function _sn_cauchy_reference(f::SkewNormal, xi::Number)
    if imag(xi) < 0
        throw(ArgumentError(
            "cauchy_dfdv(SkewNormal, xi) requires Im xi >= 0; got $xi. The residue continuation " *
            "needs phi' evaluated off the real axis, which this shape's dfdx does not provide " *
            "(dfdx(SkewNormal, ::Complex) throws)."))
    elseif imag(xi) > 0
        return _cauchy_upper(f, xi)
    else
        return _cauchy_real(f, real(xi))
    end
end

"""
    cauchy_dfdv(f::SkewNormal, xi::Number) -> Complex

The Cauchy integral of the shape's derivative (`conventions.md` 1.5), two branches, **no series**
(D36 says why: `_cauchy_series` assumes a symmetric shape, and Level 2 measured it reporting
*converged* and 1.5 % wrong on a skewed one): off the real axis, `_sn_cauchy_reference` (the
forward model never evaluates there; kept for tests and symmetry). For real `ξ`: **(1)** inside
`SN_TABLE`'s domain (`α ∈ [p_lo, p_hi] = [−3, 3]` and `ξ ∈ [xi_lo, xi_max] = [−20, 20]`, both read
from the table object) — `_cheb_table_eval(SN_TABLE, f.slice, ξ)` **at the signed `ξ`, never
`abs(ξ)`** (D13's contract for a table with `xi_lo < 0`; passing `abs(ξ)` is the bug SK10 and
SK15 (f) exist to catch, Section 14 item 25) plus the exact Plemelj term `iπ φ'(ξ)`; **(2)**
otherwise, the reference quadrature `_sn_cauchy_reference`, correct and slow (25 to 100 µs a call,
D30 item 4).

**R is not even in ξ for `α ≠ 0`** (Section 10.3: `φ(x;−α) = φ(−x;α)`, so `R(ξ;−α) = R(−ξ;α)`, and
`R(0.4) − R(−0.4) = 0.4785` at `α = 1.2`, Level 2): a table read at `|ξ|` would be wrong, and
symmetric, and the `α = 0` reduction (SK4, SK11) cannot see the mistake.

**Both domain comparisons are on primal values** (`_primal_value`, spec D47, the same fix as
`GeneralizedNormal`'s and `SuperGaussian`'s): built this way from the start, since ForwardDiff 1.x
orders a dual at a tie by its partials and D39's deck puts `α`'s bounds exactly at `±3`.

`conventions.md` 1.5; spec D35, D36; `docs/derivations/skew_normal.md`.
"""
function cauchy_dfdv(f::SkewNormal, xi::Number)
    iszero(imag(xi)) || return _sn_cauchy_reference(f, xi)
    rxi = real(xi)
    if SN_TABLE.xi_lo <= _primal_value(rxi) <= SN_TABLE.xi_max &&
       SN_TABLE.p_lo <= _primal_value(f.alpha) <= SN_TABLE.p_hi
        QT = _basefloat(rxi)
        return _cheb_table_eval(SN_TABLE, f.slice, rxi) + im * QT(pi) * dfdx(f, rxi)
    end
    return _sn_cauchy_reference(f, rxi)
end

"""
    _widen_shape(f::SkewNormal{FT0,L}, ::Type{FT}) where {FT0,L,FT} -> SkewNormal{FT,L}

Widens by **copying** the cached fields, never re-deriving them, exactly `Kappa`'s and
`GeneralizedNormal`'s reason (Phase 3's A9): the generic `_widen_shape` re-derives `ω(α)`, `x_c(α)`
and every slice entry through `reconstruct` in dual arithmetic for partials that are identically
zero. `L = 0` (the slice-free form `_sn_uncached` builds) widens through the same line with no
special case.

**Unlike `Kappa`'s and `GeneralizedNormal`'s, this one is load-bearing, not merely an optimization**
(spec D38's discrimination item 7 predicted an expected pass, following D34 item 8's reasoning;
Level 3 found it fails instead, chunk 7, 2026-09-27): deleting this method and re-deriving through
the generic `_widen_shape` gives `g.xc` a partial of `-0.0` where the specialized copy gives `0.0`
(both `iszero`, so no value is wrong, but `-0.0 === 0.0` is `false`, SK18) — `_sn_constants`'s chain
of multiplications propagates a zero partial's IEEE sign through a negative intermediate value
(`ω′·μ_δ` in `x_c`'s formula, where `μ_δ` shares `α`'s sign), which `GeneralizedNormal`'s and
`Kappa`'s simpler `loggamma`-based constants never hit.

Spec D35 (Phase 3's A9, `src/plasma/trace.jl`'s `_widen_shape` generic); D38's discrimination item
7, corrected (see `dev/reports/phase4B_report.md`, chunk 7).
"""
_widen_shape(f::SkewNormal{FT0,L}, ::Type{FT}) where {FT0,L,FT} =
    SkewNormal{FT,L}(FT(f.alpha), FT(f.omega), FT(f.xc), NTuple{L,FT}(FT.(f.slice)))

"""
    parameters(f::SkewNormal) -> NamedTuple

`(alpha = f.alpha,)`: the shape parameter alone. The cached `omega`, `xc` and `slice` are derived
state, not parameters.

`conventions.md` Section 4.
"""
parameters(f::SkewNormal) = (alpha = f.alpha,)

"""
    reconstruct(f::SkewNormal, values::NamedTuple) -> SkewNormal

A fresh `SkewNormal(values.alpha)`, recomputing every cached field from `alpha`; `f` itself is not
read. Allocation-free (`_cheb_slice` builds an `NTuple` with `Val(L)`).

`conventions.md` Section 4.
"""
reconstruct(::SkewNormal, values::NamedTuple) = SkewNormal(values.alpha)

"""
    Base.show(io::IO, f::SkewNormal)

Single line, `SkewNormal(alpha = <value>)` — the cache is derived state, not a parameter, and is
never printed.
"""
Base.show(io::IO, f::SkewNormal) = print(io, "SkewNormal(alpha = ", f.alpha, ")")

"""
    SN_PANEL_EDGES, SN_ALPHA_MAX, SN_FIT_TOL, SN_DEGREE_START, SN_DEGREE_CAP_XI,
    SN_DEGREE_CAP_ALPHA, SN_BISECT_MAX

`SN_TABLE`'s build constants (spec D36). **The coordinate is `α` itself**, domain `α ∈
[−SN_ALPHA_MAX, SN_ALPHA_MAX] = [−3, 3] × ξ ∈ [−20, 20]`, **signed**: a skewed shape's `R(ξ)` is
not even, so the table is read at the signed `ξ`, never at `|ξ|` (Section 14 item 25). `α ∈ [−3,
3]` covers skewness `|γ₁| ≤ 0.6670` of the shape's limit `0.9953`; `|α| > 3` is served by the
quadrature fallback, correct and slow (D30 item 4). `|ξ| ≤ 20`, `Kappa`'s reason: an electron shape
at low `T_e` in an EPW window reaches `|ξ| ≈ 8`–10.

**The degree cap in `α` is 64, for this table only** (D30 item 4, through chunk 4's per-table
caps): Level 2 measured `48` failing to converge (`5.6e-9`) and `64` converging (`2.2e-11`).
`SN_FIT_TOL = 1e-9`, the value target with one order of margin (rule 8.9), **and it is not
lowered**: a table direction that does not converge is a stop (`Kappa`'s and `GeneralizedNormal`'s
rule, restated for this table).

Spec D36.
"""
const SN_PANEL_EDGES = (-20.0, -10.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 10.0, 20.0)
const SN_ALPHA_MAX = 3.0
const SN_FIT_TOL = 1e-9
const SN_DEGREE_START = 8
const SN_DEGREE_CAP_XI = 64
const SN_DEGREE_CAP_ALPHA = 64
const SN_BISECT_MAX = 4

"""
    _sn_principal_value(alpha, xi) -> Real

`SN_TABLE`'s sampler, in the general `sampler(p, x)` order `_build_cheb_table` expects: `R(ξ; α)`,
the real part of the reference Cauchy integral, evaluated on the *uncached* form
`_sn_uncached(alpha)` — no slice, so the table's generator never depends on the table it is
building (spec D35). **At `α = 0`** the generator is this shape's own quadrature
(`_sn_uncached(0.0)` is the Maxwellian to rounding), **not `Zprime`** — so that `Zprime` stays an
independent oracle for SK11 at that edge (spec D36, the same reason as `KAPPA_TABLE`'s and
`GN_TABLE`'s: an oracle must not leak into the generator).

Spec D36.
"""
_sn_principal_value(alpha, xi) = real(_sn_cauchy_reference(_sn_uncached(alpha), xi))

"""
    SN_TABLE::ChebyshevTable

The module-level table `const` (`conventions.md` 5.6), built once by `_build_cheb_table` at
include time and read-only afterwards, tabulating `R(ξ; α)` on `α ∈ [−3, 3] × ξ ∈ [−20, 20]`
(**signed**) from `_sn_principal_value`. `SkewNormal(alpha)`'s public constructor computes its
`slice` from this table for every finite `α`.

Spec D36.
"""
const SN_TABLE = _build_cheb_table(_sn_principal_value, SN_PANEL_EDGES, SN_FIT_TOL,
                                   -SN_ALPHA_MAX, SN_ALPHA_MAX;
                                   degree_start=SN_DEGREE_START, cap_xi=SN_DEGREE_CAP_XI,
                                   cap_p=SN_DEGREE_CAP_ALPHA, bisect_max=SN_BISECT_MAX)
