# The `GeneralizedNormal` shape

Phase 4.B, Step 4.B.3 (spec D31-D34, D42). One file per non-Maxwellian shape (`conventions.md`
9.1), `kappa.md`'s form.

## 1. What the shape is, and what it is not

`GeneralizedNormal` is the plain one-dimensional super-Gaussian

    φ(x) = C exp(−|x/x₀|^m),   m ≥ 3/2,

the "generalized normal distribution" (Subbotin's) that other Thomson-scattering fitting codes
build directly. **It is not `SuperGaussian`**, which is the one-dimensional marginal, along
**k̂**, of an *isotropic three-dimensional* super-Gaussian `F(v) ∝ exp(−|v/v_m|^m)`; the two
shapes coincide only at `m = 2` (both reduce to the Maxwellian) and differ visibly elsewhere (GN7;
`GeneralizedNormal(4.0)(0) − SuperGaussian(4.0)(0) = −0.0543`, chunk 6).

**What 3-D density, if any, `GeneralizedNormal` would be the marginal of.** For any isotropic
3-D density `F(v)`, the 1-D marginal along **k̂** is `φ(u) = 2π ∫₀^∞ r F(√(u²+r²)) dr`;
substituting `s² = u² + r²` gives `φ(u) = 2π ∫_{|u|}^∞ s F(s) ds`, so `φ'(u) = −2π u F(|u|)`.
Solving this for `F` with `GeneralizedNormal`'s own `φ'(x) = −(m/x₀) sign(x) |x/x₀|^{m−1} φ(x)`
gives

    F(v) ∝ v^{m−2} exp(−(v/x₀)^m).

So **`m = 2` is the only exponent with a finite, nonzero `F(0)`**: `m > 2` makes `F` a shell
(`F(0) = 0`, the density of particles vanishing at zero speed) and `m < 2` makes `F` diverge at
`v = 0`. This is why `m ∈ [1.5, 2)` is accepted as a shape (D30 item 3) — it is a valid
one-dimensional projection along **k̂**, the only object OTIS ever evaluates — but is **not** the
marginal of any physically sensible isotropic 3-D plasma; it exists only so that a fit's lower
bound on `m` can sit below the Maxwellian, avoiding the saturation trap of a bound placed exactly
at it (summary A14).

## 2. The constants

The three normalizations of `conventions.md` 1.5 (`∫φ = 1`, `∫xφ = 0`, `∫x²φ = 1/2`) with
`φ(x) = C exp(−|x/x₀|^m)`:

- **Symmetry** makes `∫xφ = 0` automatic.
- **The variance** fixes `x₀`: substituting `t = |x/x₀|^m` turns `∫x²φ dx` into a Gamma integral,
  giving `∫x²φ/∫φ = x₀² Γ(3/m)/Γ(1/m)`. Setting this to `1/2` gives

      x₀(m)² = Γ(1/m) / (2 Γ(3/m))          (`_gn_x0`)

- **The unit integral** then fixes `C`: `∫φ dx = 2 C x₀ Γ(1/m)/m = 1`, so

      C(m) = m / (2 x₀(m) Γ(1/m))            (`_gn_C`)

**`m = 2` gives `x₀ = 1` exactly** (`_gn_x0(2.0) == 1.0` bitwise, chunk 6) **and `C = 1/√π`**, the
Maxwellian (GN4 checks both the value and the Cauchy integral against `Zprime` on this reduction,
worst measured `1.13e-14`, chunk 6).

**The moments**, closed form for every integer `n ≥ 0`, **all finite** (unlike `Kappa`'s, which
diverge above order `2κ − 1`):

    M_{2j}   = x₀^{2j} Γ((2j+1)/m) / Γ(1/m) = x₀^{2j} exp(loggamma((2j+1)/m) − loggamma(1/m)),
    M_{2j+1} = 0.

So `M₀ = 1`, `M₂ = 1/2` (the normalizations), and the **excess kurtosis** is
`Γ(5/m)Γ(1/m)/Γ(3/m)² − 3`: `0` at `m = 2`, negative for `m > 2` (flatter-topped, lighter-tailed
than Gaussian: `−0.8116` at `m = 4`, → `−1.2` as `m → ∞`) and positive for `m < 2`
(`+0.7620` at `m = 1.5`, heavier-tailed).

## 3. The m-partial of φ, and why no second table is needed

With `ψ = digamma`,

    d ln x₀/dm = (3ψ(3/m) − ψ(1/m)) / (2m²)
    ∂φ/∂m      = φ · [ 1/m − d ln x₀/dm + ψ(1/m)/m² − |x/x₀|^m · (ln|x/x₀| − m · d ln x₀/dm) ]
    ∂φ/∂m(0)   = C · [ 1/m − d ln x₀/dm + ψ(1/m)/m² ]

**Every piece — `x₀(m)`, `C(m)`, and the elementary `exp`/`abs`/power in the call itself — is
built on `loggamma`, whose ForwardDiff derivative rule is `digamma`, closed form and
library-provided.** So a dual `m` through `GeneralizedNormal(m)(x)` differentiates the whole shape
correctly and allocation-free with **no hand-written derivative rule at all**, exactly `Kappa`'s
reason (`docs/derivations/kappa.md` Section 3) and unlike `SuperGaussian`'s `∂φ/∂m`, whose
`_upper_gamma` a-partial has no closed form and needs `_dgamma_da`'s quadrature. This is
`conventions.md` Section 4's derivative clause ([OQ-38](../../dev/open_questions.md#oq-38)) met by
construction rather than by a second Chebyshev table: `GeneralizedNormal` carries no
`dm_slice`-analogue field and no `LM` type parameter.

GN16 checks the closed form above against `ForwardDiff.derivative(mm -> _gn_uncached(mm)(x), m)`
on a grid including `x = 0` (worst measured **2.8e-16**, chunk 6, well inside `rtol = 1e-11`), and
again checks `spectral_density`'s m-derivative against a central difference (worst measured
**6.8e-24** against a floor of **9.7e-22**, chunk 6), for `m ∈ {1.7, 2.2, 3.5, 6.0, 9.0}`, the
first running the fallback and the last the series.

**`x = 0`.** The call returns `C · one(x)` at a primal zero; `dfdx` returns `φ″(0) · x`, where
`φ″(0) = −2C/x₀²` at `m = 2` (the Maxwellian's `−2/√π`) and `0` for `m > 2` (for `m < 2` it is
`−∞`, and `0` is returned harmlessly). **`dfdx`'s guard is load-bearing**: without it,
`sign(x) · |x/x₀|^{m−1}` under a dual `x` at `0` carries the wrong partial at `m = 2` (GN4). **The
call's guard is defensive only**: Level 2 measured that ForwardDiff 1.4.5 already gives `0^m`'s
m-partial correctly unguarded (`φ(0)` and `∂φ/∂m(0) = −0.02918` at `m = 3.5`, matching the guarded
call exactly), and chunk 6's discrimination check (item 4) confirms the same: deleting the call's
`x = 0` guard is an **expected pass**, not a failure.

## 4. The Cauchy integral

**The Plemelj form** (`conventions.md` 1.5): `𝒞φ(ξ) = R(ξ) + iπφ'(ξ)` on the real axis, `R(ξ) = P∫
φ'(x)/(x−ξ) dx`; a table holds `R` alone, and the imaginary part is exact and never tabulated.

**Why the table stops at `m = 2`.** Near `ξ = 0`, `R(ξ) − R(0)` carries a leading `|ξ|^{m−1}` term
(from the `|x/x₀|^{m−1}` kink of `φ'` at `x = 0`). Its coefficient **vanishes exactly at `m = 2`**
(where `φ'` is analytic, being the Maxwellian's `−2x e^{−x²}/√π`) and for every `m < 2` the term's
slope at `ξ = 0` is infinite: a Chebyshev fit converges cleanly on `m ∈ [2, 8]` and cannot on `m < 2` — the small first
panels of `GN_TABLE` (`1/32, 1/16, 1/8, 1/4, 1/2`) exist to resolve exactly this corner as `m → 2⁺`.

**The Laurent series** (`_cauchy_series`, every moment finite, unlike `Kappa`'s): Level 2 found it
converges at `|ξ| ∈ {10, 12, 20}` for every `m` from `1.5` to `30`, within `2.6e-13` relative of a
128-bit BigFloat quadrature (D32). GN25 checks the table-series join at `ξ = 10` for
`m ∈ {2.0, 2.5, 3.0, 5.0, 8.0}` (worst measured **4.2e-13**, chunk 6, against `atol = 1e-10`).

## 5. The table

**The coordinate is `m`, domain `m ∈ [2, 8] × |ξ| ≤ 10`.**

- **Why `m ∈ [2, 8]` and not `SuperGaussian`'s own edges**: `SuperGaussian`'s panel edges
  `(0, 0.25, 0.5, 1, 2, 4, 10)` also converge here, but the first panel bisects three times (to
  `1/32`) and the `[1, 2]` panel needs ξ-degree 64: 15 180 reference evaluations and 2.44 s warm
  (Level 2). The edges actually used, `(0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 1.5, 2, 3, 4, 10)`, are
  the cheapest Level 2 measured of three tried.
- **Why `|ξ| ≤ 10`**: `SuperGaussian`'s own choice; the lighter-tailed shapes this table covers
  (`m ≥ 2`) do not need `Kappa`'s `|ξ| ≤ 20`.
- **The generator at `m = GN_M_LO = 2`** is this shape's own quadrature
  (`_gn_uncached(2.0)` is the Maxwellian to rounding), **not `Zprime`**, so that `Zprime` stays an
  independent oracle for GN11 at that edge (D32, the same reason as `KAPPA_TABLE`'s).

**`τ = 1e-9`, the value target with one order of margin (rule 8.9), and it is not lowered**: a
table direction that does not converge is a stop (D18's rule, restated for this table).

**The build, as measured (chunk 6, 2026-09-27, this machine, Julia 1.12.7)**:

| Quantity | Value |
|---|---|
| Panels | 11 |
| ξ-degrees (`nxi`) | 64, 8, 8, 16, 16, 16, 32, 32, 32, 16, 32 |
| `np` (m-degree) | 32 |
| Slice length `L` | 283 |
| Bisections | 0 |
| Every `xi_converged[p]` | `true` |
| `p_converged` | `true` |
| Reference evaluations | 9 339 |
| Warm build time | 1.04 s (a second, steadier run read 1.49 s; the speed probe read the machine within its normal jitter both times) |

identical, panel for panel, to Level 2's build quoted in spec D32 (11 panels,
`64/8/8/16/16/16/32/32/32/16/32`, `np = 32`, `L = 283`, 9 339 evaluations, ≈1.08 s warm).

**Certification, chunk 6's measured worst errors**:

| Row | Against | Worst measured | Target |
|---|---|---|---|
| GN9 | the generator, 60 seeded points `ξ ∈ [0,10]` | 3.75e-11 | `rtol=atol=1e-8` |
| GN9 | the generator, 40 seeded points `ξ ∈ [0, 0.05]` (the `\|ξ\|^{m-1}` corner) | 1.94e-9 | `rtol=atol=1e-8` |
| GN10(a) | a 64-bit BigFloat principal-value quadrature | 3.77e-13 | `atol=1e-9` |
| GN10(b) | the same, on the `m < 2` fallback | 1.99e-10 | `atol=1e-9` |
| GN11 | `Re Zprime` at the Maxwellian edge `m=2` | 1.13e-14 | `atol=1e-9` |
| GN13 | central differences of the generator, `∂R/∂m` | 0.014 % of its own floor | `rtol=1e-6`, floor per 8.4 |
| GN25 | the Laurent series at the table's edge `ξ=10` | 4.17e-13 | `atol=1e-10` |

(Level 2's own build measured `3.2e-11`, `1.9e-9`, `1.3e-12`, `1.1e-14` respectively against GN9,
GN10 and GN11's counterparts; the near-identical figures reflect the same deterministic build, not
a coincidence.)

## 6. Beyond the table

**The fallback** is `_gn_cauchy_reference`, exactly `_kappa_cauchy_reference`'s and
`_sg_cauchy_reference`'s structure: direct quadrature (`_cauchy_upper`) for `Im ξ > 0`, the
Plemelj principal-value quadrature (`_cauchy_real`) for real `ξ`, and an error for `Im ξ < 0`
(`|x|^m` has no single analytic continuation off the real axis for non-integer `m`, so the residue
continuation does not apply). **What reaches it**: whatever the table does not serve and the series does not converge on. For
`m ∈ [1.5, 2)` that is every `|ξ|` below about 6 to 9 (the series first converges at `|ξ| = 8.75`
for `m = 1.5`, `7.5` for `1.7`, `6.5` for `1.9` and `6.25` for `1.99`; Level 2, 2026-09-27, on a
0.25 grid), about **100 to 420 µs a call** (D30 item 3; Level 2 measured `2.0e-10` worst against
BigFloat at `m = 1.5, ξ = 0` exactly, `≤ 5e-12` elsewhere); for `m > 8`, every `|ξ|` below about 3
(`2.75` to `3.0` for `m` from 8.5 to 30), about **25 to 70 µs a call** (Level 2 measured `≤ 1.2e-14`
against BigFloat at `m ∈ {10, 16, 30}`).

**Chunk 6's own throughput measurements** (Section 12 row 19, ns/call, single-threaded, this
machine): the table (`GeneralizedNormal(3.5)` at `ξ = 1.3`) **158 ns**; the fallback below the
table (`GeneralizedNormal(1.7)` at `ξ = 1.3`) **153 µs**; the fallback above the series' reach
near the table's own `ξ` range (`GeneralizedNormal(12.0)` at `ξ = 1.3`) **39.0 µs**; the series
(`GeneralizedNormal(3.5)` at `ξ = 12.0`) **940 ns**. Both fallback figures sit inside Level 2's
D32 ranges (100–420 µs and 25–70 µs respectively).

## 7. The two-channel fit's EPW χ²/n (Step 4.D.2)

On the two-channel shapes problem (`examples/problems.jl`'s `shapes_problem(:generalized_normal)`,
m(t) from 2.4 to 4.9), the staged fit of Step 4.B.5 reads EPW χ²/n **41.2117** against **3.2335** for
`Kappa` on the same problem (Phase 4.B's A22). The hypothesis was the camera's point spread, which
blurs the data's mean with a kernel the forward model does not carry
([OQ-40](../../dev/open_questions.md#oq-40), B1 not ordered), acting on the flat-topped shape's
sharp EPW feature. The test (`examples/diagnostics/generalized_normal_blur.jl`, Phase 4.D spec D18)
re-synthesizes the problem with the spread off (`sigma_px = 0`) and refits it the same way, with
`Kappa` as the control, since switching the spread off changes σ as well as the image.

| | EPW χ²/n | without its worst bin | worst bin: χ²/n, fitted m against the truth | IAW χ²/n |
|---|---|---|---|---|
| `GeneralizedNormal`, spread on | 41.2117 | 9.2203 | 25: 1032.94, 5.4658 against 4.6111 | 3.0569 |
| `GeneralizedNormal`, spread off | 4.8550 | 1.0418 | 29: 123.06, 5.0162 against 3.8875 | 1.0339 |
| `Kappa`, spread on | 3.2335 | | | 2.6375 |
| `Kappa`, spread off | 0.9956 | | | 1.0250 |

**The spread is not the cause of the headline number.** Most of it is one time bin whose fitted m
leaves the truth, with the spread and without it (a different bin each time), while every other
bin's m stays within 0.11 of the truth: the fit converged, `gn_decrease` is 4.746e-7 and 1.994e-7,
far below the 1e-2 (`GN_DECREASE_WARN`) at which `fit(deck)` warns, and no entry is near a bound, so
one bin's m sits in another minimum, and why is not established. **The spread does raise the shape's
EPW χ²/n near its sharp feature**: with the outlying bin left out it reads 9.2203 against `Kappa`'s
3.2335, the excess in the bins around the profile's peak, and 1.0418 with the spread off, as the
hypothesis said, but by about a factor 3 over `Kappa`, not 13. **Nothing was changed on the strength
of it** (OQ-50, option (a)): the table, the series, the fallback and the forward model are as they
were.
