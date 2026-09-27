# The `Kappa` shape

Phase 4.B, Step 4.B.2b (spec D16-D20, D24). One file per non-Maxwellian shape (`conventions.md`
9.1), modelled on `supergaussian.md` and on `context/PACM_Report.pdf` Section 2's normalization
approach for the constants.

## 1. What the shape is, and what it is not

`Kappa` is the one-dimensional marginal, along **k̂**, of the isotropic 3-D kappa distribution

    F(v) ∝ (1 + v²/(κθ²))^{−(κ+1)},

**at the same κ**. Unlike `SuperGaussian` ([OQ-30](../../dev/open_questions.md#oq-30)), where the
1-D form the package builds and the 1-D marginal of the 3-D form are two different shapes with two
different exponents, a kappa distribution's 1-D marginal has exponent `−κ` while the 3-D form has
exponent `−(κ+1)` — one κ apart, not two different shapes to choose between. So a κ this package
fits is directly the literature's, with no reinterpretation.

**The marginal identity, derived.** In the reduced variable `x = (v − u)/v_th` (`conventions.md`
1.5) and with `a² = κ − 3/2` (Section 2), the 3-D form separates in Cartesian components
`(x, y, z)` and integrating out `y, z` in polar coordinates `(y, z) = (r cos θ, r sin θ)` gives

    ∫∫ (1 + (x² + y² + z²)/a²)^{−(κ+1)} dy dz = 2π ∫₀^∞ r (1 + (x² + r²)/a²)^{−(κ+1)} dr.

Substituting `A = 1 + x²/a²` and `u = r²/a²` (so `du = 2r dr/a²`) turns the integral into

    2π ∫₀^∞ r (A + r²/a²)^{−(κ+1)} dr = π a² ∫₀^∞ (A + u)^{−(κ+1)} du = π a² · A^{−κ}/κ,

using `∫₀^∞ (A + u)^{−(κ+1)} du = A^{−κ}/κ` (elementary). So

    ∫∫ (1 + (x² + y² + z²)/a²)^{−(κ+1)} dy dz = (π a² / κ) · (1 + x²/a²)^{−κ},

the 1-D form `(1 + x²/a²)^{−κ}` up to the constant `πa²/κ`, at the **same** κ as the 3-D form.
**KP7 is this identity made executable**: both sides evaluated by an independent `quadgk` on the
left and the closed form on the right, at κ ∈ {2.0, 3.5, 7.0} × x ∈ {0.0, 0.7, 2.5}, `rtol =
1e-12`.

## 2. The constants

The three normalizations of `conventions.md` 1.5 (`∫φ = 1`, `∫xφ = 0`, `∫x²φ = 1/2`) with
`φ(x) = C · (1 + x²/a²)^{−κ}`:

- **Symmetry** makes `∫xφ = 0` automatic.
- **The variance** fixes `a`: `∫x²(1+x²/a²)^{−κ} dx = a³ · B(3/2, κ − 3/2)` (a Beta-function
  substitution `t = x²/(a²+x²)`), and `∫(1+x²/a²)^{−κ} dx = a · B(1/2, κ − 1/2)`, so
  `∫x²φ/∫φ = a² · B(3/2,κ−3/2)/B(1/2,κ−1/2) = a²/(2κ−3)` (using `B(3/2,κ−3/2) =
  (1/2)/(κ−3/2) · B(1/2,κ−3/2)` and `B(1/2,κ−1/2)/B(1/2,κ−3/2) = (κ−3/2)/(κ−1)`, after
  simplification). Setting this to `1/2` gives **`a² = κ − 3/2`**, requiring `κ > 3/2`.
- **The unit integral** then fixes `C`: `∫(1+x²/a²)^{−κ} dx = a√π · Γ(κ−1/2)/Γ(κ)`, so

      C(κ) = Γ(κ) / (√π · a · Γ(κ − 1/2)) = exp(loggamma(κ) − loggamma(κ − 1/2)) / (√π · a)
                                                                                (`_kappa_C`)

  **`C(2) = 2√2/π`** exactly (`Γ(2) = 1`, `Γ(3/2) = √π/2`, `a(2) = 1/√2`) — KP2's exact check.

**Precision.** `_kappa_C` computes the ratio `Γ(κ)/Γ(κ−1/2)` as `exp(loggamma(κ) −
loggamma(κ−1/2))`, which loses about `eps · loggamma(κ)` relative to catastrophic cancellation for
large κ (the two `loggamma` values are close together and each of size `~κ log κ`): `C` is good to
about `1e-15` for `κ ≤ 10`, `1.3e-12` at `κ = 1000` and `2e-11` at `κ = 10⁴` (KP2's BigFloat check
at `κ = 100, 1000`; KP3's independent-quadrature check on a 2 081-point grid `κ ∈ [1.6, 12]` plus
five spot points to `κ = 300`, worst measured **9.9e-14**, chunk 5). No upper limit is imposed: a κ
above about `10⁴` is indistinguishable from the Maxwellian at the package's working accuracy.

**The moments**, closed form for every integer `n ≥ 0` (never a quadrature, `conventions.md` 8.8):

    M_{2j}   = a^{2j} · Γ(j + 1/2) · Γ(κ − j − 1/2) / (√π · Γ(κ − 1/2))     if κ − j − 1/2 > 0,
    M_{2j}   = +∞                                                          otherwise,
    M_{2j+1} = 0,

by the same Beta-function substitution as the variance derivation above, generalized to `∫x^{2j}
φ dx`. So `M₀ = 1`, `M₂ = 1/2` (the normalizations), and **`M₄ = (3/4)(κ − 3/2)/(κ − 5/2)`** for
`κ > 5/2`, diverging at and below `κ = 5/2`: the **excess kurtosis is `3/(κ − 5/2)`** there and
`+∞` for `κ ≤ 5/2`; the skewness is `0` (KP6).

**The κ → ∞ limit.** Using `Γ(κ)/Γ(κ−1/2) = √κ (1 − 3/(8κ) + O(κ⁻²))` and `1/a = κ^{−1/2}(1 +
3/(4κ) + O(κ⁻²))` (both from Stirling's series), `C√π = 1 + 3/(8κ) + O(κ⁻²)`; a second-order
expansion of `(1+x²/a²)^{−κ} = exp(−κ log(1+x²/a²))` around `x²/a² → x²/κ` gives `κ · max_x |φ_κ(x)
− e^{−x²}/√π| → 0.2116` (the coefficient of the next term in the uniform expansion). **KP4's
measurements at `κ = 10², 10³, 10⁴`, chunk 5**: `κ(C√π−1) = 0.3801, 0.37551, 0.37505` (converging
to `3/8 = 0.375`) and `κ · max|Δφ| = 0.2145, 0.21186, 0.21160` (converging to `0.2116`), both
within the spec's stated tolerances (`≤ 1/κ` and `≤ 0.005` respectively).

## 3. The κ-partial of φ, and why no second table is needed

    ∂φ/∂κ = φ · [ ψ(κ) − ψ(κ − 1/2) − 1/(2a²) − log1p(x²/a²) + κ x² / (a² (a² + x²)) ]      (ψ = digamma)

and at `x = 0`, `∂φ/∂κ = C · [ψ(κ) − ψ(κ − 1/2) − 1/(2a²)]`. **Every piece of this — `C(κ)`,
`a(κ)`, and the elementary `exp`/`log1p` in the call itself — is built on `loggamma`, whose
ForwardDiff derivative rule is `digamma`, already closed form and library-provided.** So a dual κ
through `Kappa(kappa)(x)` differentiates the whole shape correctly and allocation-free with **no
hand-written derivative rule at all** (unlike `SuperGaussian`'s `∂φ/∂m`, whose `_upper_gamma`
a-partial has no closed form and needs `_dgamma_da`'s quadrature, `docs/derivations/supergaussian.md`
Section 6). This is `conventions.md` Section 4's derivative clause
([OQ-38](../../dev/open_questions.md#oq-38)) met by construction rather than by a second
Chebyshev table: `Kappa` carries no `dm_slice`-analogue field and no `LM` type parameter.

KP5 checks the closed form above against `ForwardDiff.derivative(k -> Kappa(k)(x), kappa)` on a
grid, including `x = 0` under a dual κ (worst measured **1.8e-16**, chunk 5, well inside `rtol =
1e-11`); KP16 checks it again on a finer grid and additionally checks
`spectral_density`'s κ-derivative against a central difference (worst measured **1.1e-22** against
a floor of **5.3e-22**, chunk 5).

## 4. The Cauchy integral

**The Plemelj form** (`conventions.md` 1.5): `𝒞φ(ξ) = R(ξ) + iπφ'(ξ)` on the real axis, `R(ξ) = P∫
φ'(x)/(x−ξ) dx`; a table holds `R` alone, and the imaginary part `iπφ'(ξ)` is exact and never
tabulated.

**Why the Laurent series of [OQ-9](../../dev/open_questions.md#oq-9) does not apply.** The series
`_cauchy_series` expands `𝒞φ(ξ) ~ Σ n M_{n−1}/ξ^{n+1}` and needs moments to about order 40 to
converge at the table's edge. A kappa distribution's moments diverge above order `2κ − 1`
(Section 2 above): for `κ = 21`, already only `M₀` through `M₄₀` would exist, and for any smaller
κ fewer still — the series is unusable for essentially every κ this package fits. So `Kappa`'s
`cauchy_dfdv` has **no series branch**: outside the table, the quadrature fallback runs directly.

**The integer-κ residue form** (the independent oracle, D17, `conventions.md` 8.2: a library
special function — here `loggamma`, inside `_kappa_C` and inside the shape's own `C` — is never
its own oracle; a closed form derived by a wholly different route, residues, is). For integer `κ =
n ≥ 2`, `φ(x) ∝ (1+x²/a²)^{−n}` is a rational function of `x` with poles of order `n` at `x =
±ia`. Closing the contour `∫φ'(x)/(x−ξ) dx` in the upper half-plane (`Im ξ > 0`) picks up the pole
at `x = ia`:

    𝒞φ(ξ) = 2πi · [φ'(ξ) + D(ξ)],   D(ξ) = d/dξ Res_{x=ia} φ(x)/(x−ξ),

by the residue theorem applied to `φ(x)/(x−ξ)`, whose only singularities inside the contour (for
`Im ξ > 0`, `x = ξ` is *outside* since the ξ appears with `Im ξ>0` while the contour picks up
`x=ia`... to be precise: the *derivative* `𝒞φ` is `∫ φ'/(x−ξ)`, and writing `φ'` itself via its own
value at ξ plus the residue contribution at the pole `x=ia` is what `D(ξ)` captures) — worked out
by Leibniz's rule on `(x+ia)^{−n}(x−ξ)^{−1}` (the residue of an order-`n` pole is the
`(n−1)`-th derivative, divided by `(n−1)!`, of the rest of the function, and `D(ξ)` is one further
`ξ`-derivative of that):

    D(ξ)  = C · a^{2n} · (−1)^{n−1} · Σ_{j=0}^{n−1} p_j · (n − j) · (2ia)^{−n−j} · (ia − ξ)^{j−n−1},
    p_0 = 1,   p_j = p_{j−1} · (n + j − 1) / j     (p_j = (n)_j / j!, always an integer),

continued to real ξ. Since `2πiφ'(ξ)` is purely imaginary, **`R(ξ) = Re(2πi · D(ξ))`** is the
quantity the table serves, and the **Plemelj check** `Im(2πi·D(ξ)) = −πφ'(ξ)` is a self-consistency
identity of the derivation (KP10(d), worst measured **1.8e-15** against `atol = 1e-13`). At `n = 1`
(the Lorentzian, outside the `κ > 3/2` family) the form reduces to `−1/(ξ+ia)` times the
Lorentzian's normalization, the textbook result, which is how the sign of the whole construction
was first checked.

**Verified before use** (KP10 (a), (b), before any table reads it): against `_kappa_cauchy_reference`
(the quadrature) at `n ∈ {2,3,4,5}`, `ξ ∈ {0.3,1.7,6.0}` (worst measured **1.8e-15**), and against a
128-bit BigFloat principal-value quadrature at `n ∈ {2,3,4}` (worst measured **5.9e-38**, chunk 5;
Level 2 measured `≤ 2.4e-38` and `≤ 3.1e-15` respectively at spec-writing time — both figures agree
to the precision either measurement can distinguish).

## 5. The table

**The coordinate is `s = 1/κ`, domain `s ∈ [0, 1/2] × |ξ| ∈ [0, 20]`.**

- **Why `s = 1/κ`, not κ itself**: the Maxwellian limit `κ → ∞` becomes the table's own edge
  `s = 0`, so every `κ ≥ 2` is served with **no upper limit and no beyond-table branch in κ**. A
  table directly in `κ ∈ [2, 10]`, say, would leave every `κ > 10` — a common suprathermal case —
  to the 400×-slower fallback.
- **Why `κ ≥ 2` (`s ≤ 1/2`)**: `R(ξ; κ)` is singular at `κ = 3/2` (`a² = 0`), which is `s = 2/3`.
  At `s ≤ 1/2` the build needs `np = 32`; at `s ≤ 0.6` (`κ ≥ 1.67`) it needs the cap, `48`, with no
  margin left. `κ ∈ (3/2, 2)` is a valid shape and always runs the fallback.
- **Why `|ξ| ≤ 20`, not 10**: a heavy-tailed shape's `R` is still far from its large-`|ξ|`
  asymptote at `ξ = 10` (unlike `SuperGaussian`'s lighter tails, served to `|ξ| = 10`); an electron
  shape at low `T_e` in an EPW window reaches `ξ ≈ 8`–10. At `|ξ| ≤ 40` the build does not
  converge: the generator's own accuracy (about `1e-12` absolute against `R ≈ 1e-3` there) reaches
  the fit target `τ` before the fit does.
- **The generator at `s = 0`** is the Maxwellian's own **quadrature**, `real(_cauchy_real
  (Maxwellian(), xi))`, **not** `Zprime` — so that `Zprime` stays an independent oracle for KP11 at
  that edge, rather than the generator leaking into its own certification (Section 14 item 2 of the
  chunk-5 prompt; no test can see this, only reading the code can, which is what the
  discrimination check's item 6 demonstrates by reverting it and finding no row fails).

**`τ = 1e-9`, the value target with one order of margin (rule 8.9), and it is not lowered**: unlike
`SG_FIT_TOL`, which Phase 1.SG allowed to drop once to a floor of `1e-10`, a `Kappa` table
direction failing at `τ = 1e-9` is a stop, not a step to take (spec D18).

**The build, as measured (chunk 5, 2026-09-27, this machine, Julia 1.12.7)**:

| Quantity | Value |
|---|---|
| Panels | 7 |
| ξ-degrees (`nxi`) | 16, 16, 16, 16, 16, 32, 16 |
| `np` (s-degree) | 32 |
| Slice length `L` | 135 |
| Bisections | 0 |
| Every `xi_converged[p]` | `true` |
| `p_converged` | `true` |
| Reference evaluations | 4 455 |
| Warm build time | 0.089 s |
| `Base.summarysize(KAPPA_TABLE)` | 36 231 bytes |

identical, panel for panel, to Level 2's build quoted in spec D18 (7 panels, `16/16/16/16/16/32/16`,
`np = 32`, `L = 135`, 4 455 evaluations, ≈0.09 s warm).

**Certification, chunk 5's measured worst errors**:

| Row | Against | Worst measured | Target |
|---|---|---|---|
| KP9 | the generator (60 seeded off-node points + every edge × `s ∈ {0, 0.25, 0.5}`) | 7.8e-12 | `rtol=atol=1e-8` |
| KP10(c) | the integer-κ residue oracle (`n ∈ {2,3,4,5,10}`, 401+60 points) | 9.3e-12 | `atol=1e-9` |
| KP11 | `Re Zprime` at the Maxwellian edge `s=0` | 2.4e-11 | `atol=1e-9` |
| KP12 | central differences of the generator, `∂R/∂ξ` | (inside the shared floor) | `rtol=1e-6`, floor per 8.4 |
| KP13 | central differences of the generator, `∂R/∂s` | 0.42 % of its own floor | `rtol=1e-6`, floor per 8.4 |

(Level 2's own build measured `7.5e-12`, `≤9.3e-12` and `2.4e-11` respectively against the same
three rows; the near-identical figures reflect the same deterministic build, not a coincidence.)

## 6. Beyond the table

**The fallback** is `_kappa_cauchy_reference`, exactly `_sg_cauchy_reference`'s structure: direct
quadrature (`_cauchy_upper`) for `Im ξ > 0`, the Plemelj principal-value quadrature (`_cauchy_real`)
for real `ξ`, and an error for `Im ξ < 0` (the residue continuation needs a complex-valued `φ'`,
which `dfdx(::Kappa, ::Complex)` does not provide — `(1+x²/a²)^{−κ}` for non-integer κ has no single
analytic continuation off the real axis, the same reason `SuperGaussian`'s fallback stops there).
**What reaches it**: every `κ ∈ (3/2, 2)` (`s` above the table's domain), and every `|ξ| > 20`
regardless of κ — the second case is not rare for an **ion** species: a `Kappa` on an ion shape
across a typical EPW window has `ξ_i` in the hundreds, and always takes this route. Both are
documented, not errors; Section 12 row 13 measures the fallback's cost against the table's (about
400× on this machine, matching `SuperGaussian`'s own ratio, since both fallbacks are the same
`_cauchy_real`/`_cauchy_upper` quadrature machinery).
