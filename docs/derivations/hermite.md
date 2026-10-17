# The `HermiteExpansion` shape

Phase 5.A, Step 5.A.1 (spec D3 to D12; [OQ-66](../../dev/open_questions.md#oq-66)). One file per
non-Maxwellian shape (`conventions.md` 9.1), `skew_normal.md`'s form.

## 1. The series, and its normalizations by construction

`HermiteExpansion` is the Gram–Charlier A series around the Maxwellian, truncated at order six:

    φ(x) = (e^{−x²}/√π) (1 + c₃H₃(x) + c₄H₄(x) + c₅H₅(x) + c₆H₆(x)),

with the physicists' Hermite polynomials

    H₀ = 1,   H₁ = 2x,   H₂ = 4x² − 2,   H₃ = 8x³ − 12x,   H₄ = 16x⁴ − 48x² + 12,
    H₅ = 32x⁵ − 160x³ + 120x,   H₆ = 64x⁶ − 480x⁴ + 720x² − 120,   H₇ = 128x⁷ − 1344x⁵ + 3360x³ − 1680x,

from `H_{n+1} = 2x Hₙ − 2n H_{n−1}` (`_hermite_polys`), orthogonal under the Maxwellian's weight,
`∫ H_m Hₙ e^{−x²} dx = √π 2ⁿ n! δ_mn`, and the derivatives of that weight: `Hₙ(x) e^{−x²} = (−1)ⁿ dⁿ/dxⁿ
e^{−x²}`.

**The three normalizations of `conventions.md` 1.5 hold for every coefficient.** Write `1 = H₀`, `x = H₁/2`
and `x² = (H₂ + 2H₀)/4`; each moment `∫ xʲ φ dx` for `j ≤ 2` is then a sum of `∫ H_m Hₙ e^{−x²} dx/√π` with
`m ≤ 2 < 3 ≤ n` for every term carrying a coefficient, which orthogonality makes zero, so the moments are the
Maxwellian's: `1`, `0` and `½`. The series' `n = 0, 1, 2` terms are exactly what the three normalizations fix,
and they are fixed (`c₀ = 1`, `c₁ = c₂ = 0`), so the **flow and the temperature on the species are the mean
and the variance temperature exactly**, whatever the coefficients: a fitted `T` is the second moment of the
fitted distribution, not a core temperature. `c₃` is the skew, `c₄` the kurtosis correction, `c₅` and `c₆`
the next two corrections; the order stops at six (OQ-66 item 1).

## 2. The derivative in x

From `Hₙ e^{−x²} = (−1)ⁿ (e^{−x²})^{(n)}`, `φ = (1/√π)[e^{−x²} + Σₙ cₙ (−1)ⁿ (e^{−x²})^{(n)}]`, and one more
derivative gives

    φ'(x) = −(e^{−x²}/√π) (H₁(x) + Σₙ cₙ H_{n+1}(x)),

closed form for real, complex and dual `x` (`dfdx`, separate `Real` and `Complex` methods, `conventions.md`
5.13). At zero coefficients it is the Maxwellian's `−2x e^{−x²}/√π`.

## 3. The Cauchy integral, through the derivatives of Z

For `Im ξ > 0`, integrating by parts `k` times (the boundary terms vanish with `e^{−x²}`),

    ∫ (e^{−x²})^{(k)} / (x − ξ) dx = k! ∫ e^{−x²} / (x − ξ)^{k+1} dx = √π Z^{(k)}(ξ),

the second equality being `Z(ξ) = (1/√π) ∫ e^{−x²}/(x − ξ) dx` differentiated `k` times under the integral.
Both sides are analytic in `ξ`, so the identity holds on the Landau contour everywhere. With
`φ' = (1/√π)[(e^{−x²})' + Σₙ cₙ (−1)ⁿ (e^{−x²})^{(n+1)}]`,

    𝒞φ(ξ) = Z'(ξ) + Σₙ cₙ (−1)ⁿ Z^{(n+1)}(ξ) = Z' − c₃Z⁽⁴⁾ + c₄Z⁽⁵⁾ − c₅Z⁽⁶⁾ + c₆Z⁽⁷⁾,

the Maxwellian's `Z'` (`shapes.md` Section 2) plus one term per coefficient. **Every coefficient enters by
one multiplication**, so `∂𝒞φ/∂cₙ = (−1)ⁿ Z^{(n+1)}(ξ)` exactly, in closed form: OQ-38's derivative clause
(`conventions.md` Section 4) needs no table and no second route.

### 3.1 The recursion

`Z'(ξ) = −2(1 + ξZ(ξ))` differentiated `k` times, with `(ξZ)^{(k)} = ξZ^{(k)} + kZ^{(k−1)}` (Leibniz):

    Z^{(k+1)}(ξ) = −2 (k Z^{(k−1)}(ξ) + ξ Z^{(k)}(ξ)),    k ≥ 1,

whose `k = 1` case is `_Zpp`'s `Z'' = −2(Z + ξZ')`. `_Zderiv(ξ, Val(7))` returns `(Z, Z', …, Z⁽⁷⁾)`, the
first two `Z(ξ)` and `Zprime(ξ)` themselves. Every step is algebraic in those two, whose dual methods rest on
holomorphy (`plasma_dispersion.md`), so a dual `ξ` propagates with no new rule.

### 3.2 Why the recursion alone is not enough

The recursion is a three-term linear recurrence with two independent solutions: `Z^{(k)}` itself, which on
the real axis falls like `k!/ξ^{k+1}`, and `(e^{−ξ²})^{(k)} = (−1)^k Hₖ(ξ) e^{−ξ²}`, which grows like `(2ξ)^k`.
Run forward, the rounding of each step feeds the growing solution, and the real part of `Z^{(k)}` loses
accuracy roughly as `(2ξ²)^k/k!` once `|ξ|` exceeds about one. Measured by Level 2 (2026-10-16) against a
BigFloat evaluation of the Fried–Conte power series, the relative error of the recursion's real `Z^{(k)}`:

| ξ | `Z''` | `Z⁽⁴⁾` | `Z⁽⁷⁾` |
|---|---|---|---|
| 2.9 | 4.1e-15 | 2.8e-14 | 2.9e-14 |
| 4.0 | 1.5e-13 | 7.3e-12 | 1.8e-10 |
| 6.0 | 4.1e-13 | 1.4e-10 | 1.4e-7 |
| 8.0 | 4.5e-13 | 5.5e-10 | 4.1e-6 |
| 10 | 1.8e-12 | 5.5e-9 | 1.7e-4 |
| 20 | 3.2e-10 | 1.7e-5 | 39 (×) |

In `𝒞φ` the higher terms are weighted by small coefficients and small values, so with `(c₃, c₄, c₅, c₆) =
(0.02, 0.01, 0.003, 0.001)` the recursion alone read `4.1e-10` at `ξ = 6`, `3.4e-6` at 20, `3.5e-4` at 50 and
`1.6` (160 %) at 140; with `c₅ = c₆ = 0`, `2.2e-8` at 20 and `2.1e-4` at 140. An electron shape in an EPW
window stays within `|ξ| ≲ 10`, but an ion shape seen at EPW frequencies sits at `ξᵢ ≈ 50` to 150. Off the real
axis the recursion is used everywhere (the forward model never evaluates there): `3.0 + 1.0i` reads `6.9e-11`
in `Z⁽⁷⁾` and `5.0 − 0.5i` reads `4.0e-7`.

### 3.3 The asymptotic branch, for real |ξ| ≥ 7

**The user's decision of 2026-10-16** (Phase 5.A summary U1): on the real axis at `|ξ| ≥
ZDERIV_ASYMPTOTIC_XI = 7`, `_Zderiv` takes the real parts of `Z''` to `Z⁽⁷⁾` from the asymptotic expansion
of `Re Z = −2D` (D Dawson's integral),

    Re Z(x) ~ −Σ_{n≥0} aₙ x^{−(2n+1)},   aₙ = (2n−1)!!/2ⁿ = 1, ½, ¾, 15/8, …,

differentiated `k` times term by term (the expansion of an analytic function in a sector may be):

    Re Z^{(k)}(x) ~ −(−1)^k x^{−(k+1)} Σ_{n=0}^{N−1} aₙ (2n+1)ₖ x^{−2n},   (p)ₖ = p(p+1)⋯(p+k−1),

with `N = ZDERIV_ASYMPTOTIC_TERMS = 40` (`_reZderiv_asymptotic`). Every term is positive, so the sum does not
cancel. On the real axis `Im Z(x) = √π e^{−x²}` exactly, and `Im Z^{(k)}(x) = √π (e^{−x²})^{(k)}` follows the
same recursion on the imaginary parts alone, where the wanted solution is the growing one and the recursion
is stable. `Z` and `Z'` are `Z(ξ)` and `Zprime(ξ)` in both branches. The switch and the term count were
measured together (relative error of `Re Z⁽⁷⁾`):

| ξ | recursion | series, 30 terms | 40 terms | 50 terms |
|---|---|---|---|---|
| 6 | 1.4e-7 | 3.1e-7 | 5.3e-6 | 1.1e-4 |
| 7 | 1.0e-6 | 1.6e-10 | 1.3e-11 | 6.6e-11 |
| 8 | 4.1e-6 | 3.0e-14 | 5.3e-16 | 0 |
| 10 | 1.7e-4 | 2.2e-16 | 2.2e-16 | 2.2e-16 |

At `|ξ| = 7` with 40 terms the series is near its optimal truncation (the terms stop shrinking at about
`n = ξ²`), and from `ξ = 8` on it is at rounding; at `ξ = 6` it cannot beat the recursion. So the largest error
of `_Zderiv` on the real axis is the recursion's just below 7, about `1e-6` relative in `Z⁽⁷⁾` and `3.5e-10` in
`Z⁽⁴⁾`, and in `𝒞φ` it stays below about `1e-9` everywhere. A dual `ξ` passes through the series as through
any polynomial in `1/ξ`: its partial is the next derivative's series term by term (HE14 checks it to
`6e-16`).

### 3.4 Zero coefficients

At zero coefficients the call and `dfdx` return `Maxwellian`'s own expressions (`Maxwellian()(x)`,
`dfdx(Maxwellian(), x)`), and `cauchy_dfdv` skips every term and returns `_Zderiv`'s second element,
`Zprime(ξ)` itself, so the reduction is **bitwise**, not to rounding; otherwise each zero coefficient's term
is skipped rather than added as `0.0·H`. (The call and `dfdx` need the explicit branch: `g · (1 + 0)` can
differ from the Maxwellian's own expression in a signed zero or in the order of a product.) The test for zero is `iszero` on the coefficient
itself, not on its primal value (`conventions.md` 5.12's rule turned round, deliberately): a dual coefficient
at zero with a nonzero seed, as every fit starts its coefficients, must contribute its partial, and `iszero`
of such a dual is `false`.

## 4. The moments

`k` integrations by parts give `∫ xʲ (−1)^k (e^{−x²})^{(k)} dx = (j!/(j−k)!) ∫ x^{j−k} e^{−x²} dx` for `j ≥ k`
(zero for `j < k`), so with `G_m = (1/√π) ∫ x^m e^{−x²} dx = (m−1)!!/2^{m/2}` for even `m` and `0` for odd,

    Mⱼ = ∫ xʲ φ dx = Gⱼ + Σ_{k=3}^{6} c_k (j!/(j−k)!) G_{j−k}     (k ≤ j),

closed form for every `j ≥ 0` (`moment`): `M₀ = 1`, `M₁ = 0` and `M₂ = ½` exactly, `M₃ = 6c₃`,
`M₄ = ¾ + 24c₄`, `M₅ = 30c₃ + 120c₅`, `M₆ = 15/8 + 360c₄ + 720c₆`. The generic `skewness` and
`excess_kurtosis` then read

    skewness = M₃/M₂^{3/2} = 6c₃ · 2√2 = 12√2 c₃,     excess kurtosis = M₄/M₂² − 3 = 4(¾ + 24c₄) − 3 = 96 c₄.

`_cauchy_series` is not used: its stopping rule assumes a symmetric shape (`conventions.md` Section 4).

## 5. Positivity: a report, not a refusal

**The rule is the user's** (2026-10-16, option C of Level 1's five, OQ-66 item 4): the constructor refuses
only a non-finite coefficient; the shape reports its negative lobe, and `fit(deck)` warns. Why: a strict rule
would forbid every flattened core and every skew without kurtosis (below), its admissible set is a curved
region a deck's box bounds cannot describe, Levenberg–Marquardt must be free to pass through negative-lobe
territory from a poor start, and a lobe of a few tenths of a percent is the truncated expansion's own error,
which the user judges.

**The report**, `_hermite_lobe(f) -> (factor_min, x_min, negative_mass)`, at the coefficients' primal values:
the minimum of the factor `1 + Σₙ cₙHₙ(x)` over the grid `x = −6, −5.9, …, 6` (`HERMITE_X_MAX = 6`) and the
first grid point where it occurs, and `negative_mass = ∫ max(−φ, 0) dx` over `|x| ≤ 6` by `quadgk`, with the
integers as breakpoints (a magnitude, exactly `0.0` where `φ` is nowhere negative). `show` prints it when the
factor is negative; `fit(deck)` warns, after the near-bound warning, at every fitted bin whose negative mass
exceeds `HERMITE_NEGATIVE_MASS_WARN = 1e-3`, a tenth of a percent of the distribution, naming the species, the
bins and the largest mass (`_warn_hermite_lobe`). `converged` does not see it (OQ-36). Three cases, measured:

| `(c₃, c₄)` | factor minimum on the grid | negative mass |
|---|---|---|
| `(0.04, 0)` | −65.24 at x = −6 | 4.232e-3 |
| `(0.04, 0.01)` | +0.159 at x = −2.1 (the true minimum 0.156 at −2.053) | 0 |
| `(0, −0.01)` | −189.2 at x = −6 | 3.750e-3 |

**The geometry, with `c₅ = c₆ = 0`** (measured by Level 2, 2026-10-16: the factor's minimum over the whole real
line on a 1e-3 grid refined by golden section, `c₃`'s limit by bisection; Level 1's figures of the same day on
a 0.01 grid reproduce):

- **`c₃` alone is never positive**: the cubic goes negative on the far side of the skew (`c₃ = 1e-4` already
  reaches −0.37 at the window's edge). Its mass is small for small skews: 2.1e-3 at `c₃ = 0.03`, 4.2e-3 at
  0.04.
- **`c₄ < 0`, the flattened core the super-Gaussian exists for, is never positive**: `16c₄x⁴` wins at large
  `|x|`.
- **`c₄` is bounded above by exactly 1/24 = 0.04167**, for every `c₃`: at `x² = 3/2`, `H₄` takes its minimum
  −24 and `H₃` vanishes, so the factor there is `1 − 24c₄` whatever `c₃` is.
- **Inside, `c₄ > 0` buys skew** (the region of Barton and Dennis): the series is positive everywhere for `|c₃|`
  up to

| `c₄` | 0.001 | 0.0025 | 0.005 | 0.0075 | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 | 0.035 | 0.04 | 0.0416 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c₃` at most | 0.0093 | 0.0177 | 0.0282 | 0.0365 | 0.0432 | 0.0532 | 0.0594 | 0.0618 | 0.0599 | 0.0520 | 0.0302 | 0.0066 |
| skewness | 0.158 | 0.301 | 0.479 | 0.619 | 0.733 | 0.903 | 1.008 | 1.049 | 1.017 | 0.882 | 0.512 | 0.111 |

  the widest at `c₄ ≈ 0.0255`, `|c₃| ≤ 0.0618` (skewness 1.049, excess kurtosis 2.45). A skew of 0.68 needs
  `c₄` of at least about 0.009, an excess kurtosis of about 0.85.
- **Where `fit` does not warn** (negative mass at most 1e-3 on `|x| ≤ 6`) is wider: `c₄` from about −0.0056 to
  about 0.0436, and at `c₄ = 0` `|c₃|` up to 0.0230 (skewness 0.39), at 0.01 up to 0.0494, at 0.025 up to
  0.0684.

**As advice for a deck's bounds, not a boundary**: a fitted `c₄` between 0 and 0.04 and `|c₃|` below the table's
value at the bound on `c₄` keeps every shape the box allows positive; a flattened core needs `c₄ < 0`, which
is never positive and from about −0.0056 on draws the warning, and the deck that wants it accepts the lobe.

## 6. What the expansion cannot do

A truncated Gram–Charlier series describes a modest deformation of a Maxwellian. **A beam**, or any
deformation strong enough to need terms beyond `c₆`, is a second species, as before. Orders beyond six are not
built (beyond them the lobes grow faster than the descriptive power on data of Thomson-scattering quality;
OQ-66 item 1). **The squared-Hermite form**, `φ ∝ (Σ aₙHₙ)² e^{−x²}` with its own shift and scale, is
positive by construction and represents a beam; its Cauchy integral is still closed form through `Z`'s
derivatives (a product of Hermite polynomials is a finite Hermite series). It is the recorded follow-up if
this phase's positivity rule bites (OQ-66, "Not here").

## 7. The tests

`test/distributions/test_hermite.jl`, gate `Hermite` (spec D12): the normalizations (HE1); the Maxwellian
bitwise at zero (HE2); `dfdx` against ForwardDiff (HE3); `cauchy_dfdv` against the quadrature fallback above, on
and below the axis (HE4); the moments and the two identities (HE5); the coefficient partials, of `cauchy_dfdv`
against the BigFloat oracle and of the spectral density against differences (HE6); the recursion against
ForwardDiff and the BigFloat series (HE7); the lobe report, `show`, the warning on a fit result built without
fitting and the constructor's refusals (HE8); four dual coefficients allocate nothing (HE9); the API (HE10); the
deck (HE11); the IAW asymmetry pinned (HE13); the asymptotic branch (HE14). `test/slow/test_hermite.jl`, gate
`Hermite (slow)`: the recovery of a skew-normal and of a flattened electron distribution from a noisy EPW
streak, and the warning on the two fits (HE12); `fit(deck)`'s warning (HE15). Those two fit, and each fit of a
new shape type compiles its fit path, about 26 s together, so they are the slow tier's (the user, 2026-10-16).
The cost guard's table (`test_shape_cost.jl`) carries the shape at `(c₃, c₄) = (0.02, 0.01)`.
