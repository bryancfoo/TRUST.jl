# The Kappa shape: the 1-D marginal along k̂ of the isotropic 3-D kappa distribution
# F(v) ∝ (1 + v^2/(kappa*theta^2))^(-(kappa+1)), AT THE SAME kappa, so a fitted kappa is the
# literature's. See docs/derivations/kappa.md. Phase 4.B Step 4.B.2b, spec D16-D20, D24.

using SpecialFunctions: loggamma

"""
    Kappa{FT<:Real,L} <: AbstractVelocityDistribution

The kappa distribution's velocity shape along k̂: the one-dimensional marginal of the isotropic
3-D kappa distribution `F ∝ (1 + v²/(κθ²))^(−(κ+1))` at the **same** κ, so a fitted κ is the
literature's — unlike `SuperGaussian` ([OQ-30](../../dev/open_questions.md#oq-30)), there is no
second shape to choose between (Section 10.1, `docs/derivations/kappa.md`). Field `kappa` is the
shape parameter; `a` and `C` are its own normalization constants `a(κ) = √(κ − 3/2)` and `C(κ)`
(spec D16), cached at construction so the per-call quadrature reference does not recompute them
at every quadrature node. `slice` is `KAPPA_TABLE`'s ξ-coefficients at `s = 1/κ`, computed for
**every** `κ > 3/2` (including `κ ∈ (3/2, 2)`, where `s` lies above the table's domain and the
slice is built but never read: `cauchy_dfdv` checks the domain first).

**No second (φ-derivative) table**, unlike `SuperGaussian`'s `SG_DM_TABLE`: `Kappa`'s φ is
elementary (`exp`, `log1p`) and its constants go through `loggamma`, whose ForwardDiff rule
(`digamma`) is closed form, so a dual κ through the call is already quadrature-free and
allocation-free (`conventions.md` Section 4's derivative clause, [OQ-38](../../dev/open_questions.md#oq-38),
[OQ-42](../../dev/open_questions.md#oq-42) item 2). So there is no `LM` type parameter and no
`dm_slice`.

`conventions.md` Section 4, Section 11; spec D16; `docs/derivations/kappa.md`.
"""
struct Kappa{FT<:Real,L} <: AbstractVelocityDistribution
    kappa::FT
    a::FT                 # sqrt(kappa - 3/2), the width a = sqrt(kappa) * x0
    C::FT                 # Gamma(kappa) / (sqrt(pi) a Gamma(kappa - 1/2))
    slice::NTuple{L,FT}   # KAPPA_TABLE's xi-coefficients at s = 1/kappa
end

"""
    _kappa_C(kappa, a) -> Real

`C(κ) = Γ(κ) / (√π · a · Γ(κ − 1/2)) = exp(loggamma(κ) − loggamma(κ − 1/2)) / (√π · a)`, the
shape's normalization constant, fixed by the unit integral `∫φ = 1` (the variance normalization
`∫x²φ = 1/2` already fixes `a² = κ − 3/2` — see `Kappa`'s constructor). Generic over `κ`'s type:
ForwardDiff differentiates `loggamma` through its own rule (`digamma`), so a dual `κ` gives a
dual `C` with no extra rule needed (spec D16; [OQ-42](../../dev/open_questions.md#oq-42) item 2's
"closed-form κ-partials of the constants"). `π` is taken at `κ`'s own base float (`_basefloat`, as
`Maxwellian` does), so this stays generic under a dual `κ`.

**Precision**: the `loggamma` difference loses about `eps · loggamma(κ)` relative, so `C` is good
to about `1e-15` for `κ ≤ 10`, `1.3e-12` at `κ = 1000`, and `2e-11` at `κ = 10⁴` (KP2). A `κ` above
about `10⁴` is indistinguishable from the Maxwellian at the package's accuracy, and no upper limit
is imposed.

Section 10.1; `docs/derivations/kappa.md`.
"""
_kappa_C(kappa, a) = exp(loggamma(kappa) - loggamma(kappa - 1 / 2)) / (sqrt(_basefloat(kappa)(pi)) * a)

"""
    Kappa(kappa::Real)
    (f::Kappa)(x::Real) -> Real

The public, one-argument constructor (`conventions.md` Section 11), and the shape's call overload.
**Both are documented here because they share one doc-system key** (spec D16, Pitfall 21, exactly
`SuperGaussian`'s reason): `Kappa(kappa::Real)` and `(f::Kappa)(x::Real)` are both methods of the
binding `Kappa` with signature `Tuple{Real}`, so a second docstring on the call collides with this
one and is silently dropped, with a `Replacing docs` warning at precompile.

**The shape.** `φ(x) = C(κ) (1 + x²/a²)^{−κ}`, the 1-D marginal along k̂ of the isotropic 3-D kappa
distribution `F ∝ (1 + v²/(κθ²))^{−(κ+1)}` at the **same** κ (Section 10.1's label identity,
KP7's executable form): unlike `SuperGaussian`, the 1-D exponent `−κ` and the 3-D exponent
`−(κ+1)` both belong to one literature κ, so a fitted κ here is the literature's directly. `κ`
must be finite and `> 3/2` (below that the variance integral diverges): a loud `ArgumentError`
otherwise (`conventions.md` 5.9).

**The constructor promotes `κ` first** (Step 2.0's `HK1` form, spec D16, KP19): `FT =
promote_type(typeof(kappa), eltype(KAPPA_TABLE.coeffs))`, then `a = √(κ_p − 3/2)`, `C = _kappa_C
(κ_p, a)` and `slice = _cheb_slice(KAPPA_TABLE, 1/κ_p)` are all computed **at** the promoted
`κ_p`, never at `kappa`'s own (possibly narrower) type and widened afterward — computing at
`Float32` and promoting the result gives a different shape bit for bit, not merely to `Float32`
precision. **One method, not two** (unlike `SuperGaussian`'s plain/dual pair): every `κ > 3/2`,
including `κ ∈ (3/2, 2)` where `s = 1/κ` lies above the table's `p_hi = 1/2` and the Chebyshev
series is extrapolated, gets the same `slice` computation — that slice is simply never read there,
since `cauchy_dfdv` checks the table's domain before using it.

**The call.** `φ(x) = C · exp(−κ · log1p((x/a)²))`, dimensionless in and out. No `x = 0` guard is
needed, unlike `SuperGaussian`'s: `log1p(0) = 0` exactly, and under a dual κ the term `−κ·0` has
zero partials, never `0 · log 0` (KP5 checks `x = 0` under a dual κ).

**A κ outside `KAPPA_TABLE`'s domain is not an error**: `cauchy_dfdv` on it runs the reference
quadrature instead of the table, correctly but about 120× slower (124× measured at κ = 1.8,
ξ = 1.3, 2026-09-27; spec D16, D18, Phase 4.B Section 12 row 13); `κ ∈ (3/2, 2)`
always takes this route, since the table never serves `s > 1/2`.

Section 10.1; spec D16; `docs/derivations/kappa.md`.
"""
function Kappa(kappa::Real)
    (isfinite(kappa) && kappa > 3 / 2) || throw(ArgumentError(
        "Kappa: kappa must be finite and greater than 3/2; got $kappa"))
    FT = promote_type(typeof(kappa), eltype(KAPPA_TABLE.coeffs))
    k = FT(kappa)
    a = sqrt(k - FT(3) / 2)
    C = _kappa_C(k, a)
    slice = _cheb_slice(KAPPA_TABLE, inv(k))
    return Kappa(k, FT(a), FT(C), slice)
end

"""
    _kappa_uncached(kappa::Real) -> Kappa{FT,0}

The generator's constructor, exactly `_sg_uncached`'s role: computes `kappa`, `a` and `C` as the
public constructor does, but sets `slice = ()`, giving a shape of the **distinct type**
`Kappa{FT,0}`. This breaks the cycle `KAPPA_TABLE`'s own build would otherwise create
(`Kappa(kappa)` needs `KAPPA_TABLE`, and building it needs shapes to evaluate its sampler):
`_kappa_principal_value` calls through a shape built this way, and that path
(`_kappa_cauchy_reference`, via `_cauchy_real`/`_cauchy_upper`) reads only `kappa`, `a` and `C` —
never `slice` nor `KAPPA_TABLE` (spec D16).

`cauchy_dfdv` on a `Kappa{<:Real,0}` throws (below): a shape with a meaningless empty slice can
never be evaluated by accident.

Spec D16.
"""
function _kappa_uncached(kappa::Real)
    a = sqrt(kappa - oftype(float(kappa), 3) / 2)
    C = _kappa_C(kappa, a)
    FT = promote_type(typeof(kappa), typeof(a), typeof(C))
    return Kappa{FT,0}(FT(kappa), FT(a), FT(C), ())
end

"""
    cauchy_dfdv(f::Kappa{<:Real,0}, xi::Number)

Throws (spec D16): `f` is the uncached form `KAPPA_TABLE`'s generator builds, with an empty,
meaningless `slice`. It must never be evaluated; construct the shape with `Kappa(kappa)` instead.
More specific than the generic `cauchy_dfdv(::Kappa, ::Number)` method below, so a `Kappa{<:Real,0}`
always reaches this one first. It **must not** name `KAPPA_TABLE`: it runs while the table is
being built.
"""
cauchy_dfdv(f::Kappa{<:Real,0}, xi::Number) = throw(ArgumentError(
    "cauchy_dfdv(Kappa{FT,0}, xi): this shape carries no table slice. It is the uncached form " *
    "the table's generator builds and must not be used for evaluation; construct the shape " *
    "with Kappa(kappa)."))

# Documented above, on Kappa(kappa::Real): the two share one doc-system key (spec D16, Pitfall 21).
(f::Kappa)(x::Real) = f.C * exp(-f.kappa * log1p((x / f.a)^2))

"""
    dfdx(f::Kappa, x::Real) -> Real

`φ'(x) = −(2κC/a²) · x · (1 + x²/a²)^{−κ−1} = −(2κC/a²) · x · exp(−(κ+1) · log1p(x²/a²))`, the
closed form (Section 10.1). Dimensionless in and out.

Section 10.1; spec D16; `docs/derivations/kappa.md`.
"""
dfdx(f::Kappa, x::Real) = -(2 * f.kappa * f.C / f.a^2) * x * exp(-(f.kappa + 1) * log1p((x / f.a)^2))

"""
    (f::Kappa)(x::Complex)
    dfdx(f::Kappa, x::Complex)

Throw: the shape is defined for real `x` only (spec D16; rule 5.13, a split override, never a
single `::Number` method).
"""
(f::Kappa)(x::Complex) = throw(ArgumentError(
    "(f::Kappa)(x::Complex) is not available: the shape is defined for real x only (Phase 4.B spec D16)."))
dfdx(f::Kappa, x::Complex) = throw(ArgumentError(
    "dfdx(Kappa, ::Complex) is not available: the shape is defined for real x only (Phase 4.B spec D16)."))

"""
    moment(f::Kappa, n::Integer) -> Real

The `n`-th moment `∫ xⁿ φ(x) dx`, closed form for every integer `n ≥ 0` (Section 10.1):

    M_{2j}   = a^{2j} Γ(j + 1/2) Γ(κ − j − 1/2) / (√π Γ(κ − 1/2))     if κ − j − 1/2 > 0,
    M_{2j}   = +∞                                                     otherwise,
    M_{2j+1} = 0.

So `M₀ = 1`, `M₂ = 1/2` (the normalizations), and `M₄ = (3/4)(κ − 3/2)/(κ − 5/2)` for `κ > 5/2`,
`+∞` at and below `κ = 5/2` (**never a quadrature**, `conventions.md` 8.8): a kappa distribution's
moments exist only below order `2κ − 1`, which is why `cauchy_dfdv` below has **no series branch**
(`_cauchy_series` needs moments to about order 40). All three returns are of type `typeof(f.kappa)`
(rule 5.14, a branch on a runtime value returns one type). Throws for `n < 0`.

`skewness` and `excess_kurtosis` use the generic methods of `distribution.jl`, which then give `0`
and `3/(κ − 5/2)` for `κ > 5/2`, and `Inf` for `κ ≤ 5/2` (`M₄` infinite).

Section 10.1; spec D16; `docs/derivations/kappa.md`.
"""
function moment(f::Kappa, n::Integer)
    n >= 0 || throw(ArgumentError("moment(Kappa, n): n must be non-negative; got $n"))
    isodd(n) && return zero(f.kappa)
    j = n ÷ 2
    f.kappa - j - oftype(f.kappa, 1) / 2 > 0 || return convert(typeof(f.kappa), Inf)
    return f.a^n * exp(loggamma(j + oftype(f.kappa, 1) / 2) + loggamma(f.kappa - j - oftype(f.kappa, 1) / 2) -
                       loggamma(f.kappa - oftype(f.kappa, 1) / 2)) / sqrt(_basefloat(f.kappa)(pi))
end

"""
    _kappa_cauchy_reference(f::Kappa, xi::Number) -> Complex

The reference Cauchy integral, exactly `_sg_cauchy_reference`'s twin: direct quadrature
(`_cauchy_upper`) for `Im ξ > 0`, the Plemelj principal-value quadrature (`_cauchy_real`) for real
`ξ`, and an error for `Im ξ < 0` (spec D16): the residue continuation `_cauchy_lower` relies on
needs a complex-valued `φ'`, which this shape's `dfdx` does not provide off the real axis. Correct,
and slow, for every `κ > 3/2` and every `ξ` in the closed upper half-plane. This is `KAPPA_TABLE`'s
own generator (through `_kappa_principal_value`) and every reference-level test's oracle.

Spec D16, D18; `docs/derivations/kappa.md`.
"""
function _kappa_cauchy_reference(f::Kappa, xi::Number)
    if imag(xi) < 0
        throw(ArgumentError(
            "cauchy_dfdv(Kappa, xi) requires Im xi >= 0; got $xi. The residue continuation " *
            "needs phi' evaluated off the real axis, which this shape's dfdx does not provide " *
            "(dfdx(Kappa, ::Complex) throws)."))
    elseif imag(xi) > 0
        return _cauchy_upper(f, xi)
    else
        return _cauchy_real(f, real(xi))
    end
end

"""
    cauchy_dfdv(f::Kappa, xi::Number) -> Complex

The Cauchy integral of the shape's derivative (`conventions.md` 1.5), three branches: off the real
axis, `_kappa_cauchy_reference` (the forward model never evaluates there; kept for tests and
symmetry with `SuperGaussian`). For real `ξ`: **(1)** inside `KAPPA_TABLE`'s domain (`s = 1/κ ∈
[p_lo, p_hi]` and `|ξ| ≤ xi_max`, both read from the table object) — `_cheb_table_eval(KAPPA_TABLE,
f.slice, |ξ|)` plus the exact Plemelj term `iπ φ'(ξ)`; **(2)** otherwise, the reference quadrature
`_kappa_cauchy_reference`. **No series branch** (spec D16, Section 14 item 4 of the phase spec): a
kappa distribution's moments exist only below order `2κ − 1`, so `_cauchy_series`'s Laurent
expansion (which needs moments to about order 40) does not apply; `κ ∈ (3/2, 2)` and `|ξ| > 20`
both always take the fallback, correctly and about 120× slower — documented, not an error.

**R is even in ξ** (`φ` is even, so `φ'` is odd and the principal value `R` is even): the table
holds `ξ ≥ 0` and this function passes `abs(ξ)` to `_cheb_table_eval` (D13's contract for an even
table; passing the signed `ξ` would silently read outside the table for `ξ < 0`, Section 14
item 4 of the chunk-5 prompt).

**Both domain comparisons are on primal values** (`_primal_value`, Phase 4.C spec D3): ForwardDiff
1.x orders a dual at a tie by its partials, so a dual `κ` exactly at 2 (`s = 1/κ` at `p_hi = 1/2`)
whose partial points `s` outward, or a dual `ξ` exactly at `±20` whose partial points `|ξ|`
outward, would otherwise leave the table for the fallback: correct, and two to three orders of
magnitude slower (37 to 43 µs a call against 0.06 to 0.25 µs inside the table, at `ξ = 1.3`).

`conventions.md` 1.5; spec D16; `docs/derivations/kappa.md`.
"""
function cauchy_dfdv(f::Kappa, xi::Number)
    iszero(imag(xi)) || return _kappa_cauchy_reference(f, xi)
    rxi = real(xi)
    axi = abs(rxi)
    s = inv(f.kappa)
    if _primal_value(axi) <= KAPPA_TABLE.xi_max &&
       KAPPA_TABLE.p_lo <= _primal_value(s) <= KAPPA_TABLE.p_hi
        QT = _basefloat(rxi)
        return _cheb_table_eval(KAPPA_TABLE, f.slice, axi) + im * QT(pi) * dfdx(f, rxi)
    end
    return _kappa_cauchy_reference(f, rxi)
end

"""
    _widen_shape(f::Kappa{FT0,L}, ::Type{FT}) where {FT0,L,FT} -> Kappa{FT,L}

Widens by **copying** the cached fields, never re-deriving them, exactly `SuperGaussian`'s reason
(Phase 3's A9): the generic `_widen_shape` re-derives `a(κ)`, `C(κ)` and every slice entry through
`reconstruct` in dual arithmetic for partials that are identically zero. `L = 0` (the slice-free
form `_kappa_uncached` builds) widens through the same line with no special case: `NTuple{0,FT}
(())` is the empty tuple either way.

Spec D16 (Phase 3's A9, `src/plasma/trace.jl`'s `_widen_shape` generic).
"""
_widen_shape(f::Kappa{FT0,L}, ::Type{FT}) where {FT0,L,FT} =
    Kappa{FT,L}(FT(f.kappa), FT(f.a), FT(f.C), NTuple{L,FT}(FT.(f.slice)))

"""
    parameters(f::Kappa) -> NamedTuple

`(kappa = f.kappa,)`: the shape parameter alone. The cached `a`, `C` and `slice` are derived
state, not parameters.

`conventions.md` Section 4.
"""
parameters(f::Kappa) = (kappa = f.kappa,)

"""
    reconstruct(f::Kappa, values::NamedTuple) -> Kappa

A fresh `Kappa(values.kappa)`, recomputing every cached field from `kappa`; `f` itself is not
read. Allocation-free (`_cheb_slice` builds an `NTuple` with `Val(L)`).

`conventions.md` Section 4.
"""
reconstruct(::Kappa, values::NamedTuple) = Kappa(values.kappa)

"""
    Base.show(io::IO, f::Kappa)

Single line, `Kappa(kappa = <value>)` — the cache is derived state, not a parameter, and is never
printed.
"""
Base.show(io::IO, f::Kappa) = print(io, "Kappa(kappa = ", f.kappa, ")")

"""
    KAPPA_PANEL_EDGES, KAPPA_S_HI, KAPPA_FIT_TOL, KAPPA_DEGREE_START, KAPPA_DEGREE_CAP_XI,
    KAPPA_DEGREE_CAP_S, KAPPA_BISECT_MAX

`KAPPA_TABLE`'s build constants (spec D18). **The coordinate is `s = 1/κ`**, domain `s ∈ [0,
KAPPA_S_HI]` (`KAPPA_S_HI = 1/2`, so every `κ ≥ 2` is served) `× |ξ| ≤ 20` (`KAPPA_PANEL_EDGES[end]`):
`s = 0` is the Maxwellian limit `κ → ∞`, the table's own edge, checked against `Re Z'(ξ)` (KP11);
`κ ∈ (3/2, 2)` is `s ∈ (1/2, 2/3)`, above the table and always served by the fallback, since `R`
is singular at `κ = 3/2` (`a² = 0`, `s = 2/3`) and the build needs degree 48 (the cap) already at
`s ≤ 0.6`, leaving no margin. `|ξ| ≤ 20`, not 10: a heavy-tailed shape's `R` is still far from its
asymptote at `ξ = 10`, and an electron shape at low `T_e` in an EPW window reaches `ξ ≈ 8`–10; at
`|ξ| ≤ 40` the build does not converge, since the generator's own accuracy (about `1e-12` absolute
against `R ≈ 1e-3`) reaches `τ` there. `KAPPA_FIT_TOL = 1e-9`, the value target with one order of
margin (rule 8.9), **and it is not lowered**: if a table direction does not converge, or a
derivative certification fails at `τ = 1e-9`, that is a stop, not a floor to chase (spec D18,
unlike `SG_FIT_TOL`'s one permitted step down).

Spec D18.
"""
const KAPPA_PANEL_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0)
const KAPPA_S_HI = 0.5          # s = 1/kappa in [0, KAPPA_S_HI]: every kappa >= 2
const KAPPA_FIT_TOL = 1e-9      # the value target with one order of margin (rule 8.9)
const KAPPA_DEGREE_START = 8
const KAPPA_DEGREE_CAP_XI = 64
const KAPPA_DEGREE_CAP_S = 48
const KAPPA_BISECT_MAX = 4

"""
    _kappa_principal_value(s, xi) -> Real

`KAPPA_TABLE`'s sampler, in the general `sampler(p, x)` order `_build_cheb_table` expects: `R(ξ;
κ = 1/s)`, the real part of the reference Cauchy integral. **At `s = 0`** (the Maxwellian limit)
the generator is `real(_cauchy_real(Maxwellian(), xi))`, the Maxwellian's own **quadrature**, **not
`Zprime`** — so that `Zprime` stays an independent oracle for KP11 at that edge (spec D18, Section
14 item 2: an oracle must not leak into the generator; no test can see the leak, only reading the
code can, which is why the discrimination check's item 6 is expected to pass). Elsewhere, the
reference is evaluated on the *uncached* form `_kappa_uncached(1/s)` — no slice, so the table's
generator never depends on the table it is building (spec D16).

Spec D18.
"""
_kappa_principal_value(s, xi) = iszero(s) ? real(_cauchy_real(Maxwellian(), xi)) :
                                real(_kappa_cauchy_reference(_kappa_uncached(inv(s)), xi))

"""
    KAPPA_TABLE::ChebyshevTable

The module-level table `const` (`conventions.md` 5.6), built once by `_build_cheb_table` at
include time and read-only afterwards, tabulating `R(ξ; κ)` on `s = 1/κ ∈ [0, 1/2] × |ξ| ∈ [0,
20]` from `_kappa_principal_value`. `Kappa(kappa)`'s public constructor computes its `slice` from
this table for every `κ > 3/2`.

Spec D18.
"""
const KAPPA_TABLE = _build_cheb_table(_kappa_principal_value, KAPPA_PANEL_EDGES, KAPPA_FIT_TOL,
                                      0.0, KAPPA_S_HI;
                                      degree_start=KAPPA_DEGREE_START, cap_xi=KAPPA_DEGREE_CAP_XI,
                                      cap_p=KAPPA_DEGREE_CAP_S, bisect_max=KAPPA_BISECT_MAX)
