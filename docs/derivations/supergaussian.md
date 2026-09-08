# The `SuperGaussian` shape

This document derives what `src/distributions/supergaussian.jl` (and, from Step 1.SG.2,
`src/distributions/supergaussian_table.jl`) implements, in the package's own normalization.

## 1. Not `exp(−|x|^m)`

`SuperGaussian` is **not** the 1-D form `C exp(−|x/x₀|^m)` that its name might suggest — that is a
different, lighter-tailed shape, reserved for Phase 4 under a different name
(`conventions.md` Section 4, roadmap Step 4.2). `SuperGaussian` is instead the **1-D marginal
along k̂ of the isotropic 3-D super-Gaussian**

    F(v) ∝ exp(−|v/v_m|^m)                                          (PACM Section 2; foo2026 Eq. 12)

integrated over the two velocity components transverse to k̂. PACM Section 2.2 carries this
integral out in spherical coordinates and finds the 1-D marginal to be an **incomplete gamma
function** of `|x|^m`, not a plain exponential of it (PACM Eq. 26) — the transverse integration is
what turns `exp(−|x|^m)` into `Γ(2/m, |x|^m)`. `m = 2` collapses the difference (the two forms
coincide for a Gaussian), which is why the reduction to `Maxwellian` at `m = 2` (Section 4 below)
is silent on this distinction, but at `m ≠ 2` the two shapes have different tails, different
Cauchy integrals and different physics, and the plain exponential form is never called
`SuperGaussian` in this package.

## 2. From the 3-D form to the marginal, in our normalization

Write the isotropic 3-D super-Gaussian in spherical velocity coordinates, `v = |**v**|`, and
project onto k̂ so that the transverse two components are integrated out. PACM's Section 2 result
(its Eqs. 13–26), transcribed into the package's reduced variable `x = (v−u)/v_th`,
`v_th = √(2T/m_s)` (`conventions.md` 1.4), is

    φ(x; m) = C(m) Γ(2/m, |x/x₀(m)|^m),

with `Γ(a, t) = ∫_t^∞ s^{a−1} e^{−s} ds` the **upper** incomplete gamma function, and `x₀(m)`,
`C(m)` fixed by the two normalizations that are not automatic (`∫φ dx = 1` and `∫x²φ dx = 1/2`;
the odd first moment `∫xφ dx = 0` holds by symmetry for any `x₀ > 0`):

    x₀(m)² = (3/2) Γ(3/m) / Γ(5/m)                       (variance 1/2;  PACM Eq. 20)
    C(m)   = 1 / (2 x₀(m) Γ(3/m))                        (unit integral; PACM Eq. 26)

`x₀(m)` is dimensionless in `x` and relates to the two sources' own width parameters by
`v₀ (foo2026 Eq. 12) = v_p (PACM Eq. 19) = x₀(m)·v_th`: both papers' widths are the package's
`x₀(m)` scaled by the package's own thermal speed.

## 3. `φ′`, the moments, and `𝒞φ(0)`

Differentiating `Γ(2/m, |x/x₀|^m)` with respect to `x` through the elementary rule
`d/dt Γ(a,t) = −t^{a−1}e^{−t}` and the chain rule `t = |x/x₀|^m` gives the **closed form**

    φ'(x) = −(m C(m)/x₀(m)²) · x · exp(−|x/x₀(m)|^m)                (PACM Eq. 44)

— never computed as the chain rule through `Γ` itself (Section 6 explains why that path is
singular in the `m`-partial at `x = 0`). At `x = 0`, `φ(0) = C(m) Γ(2/m)` and `φ'(0) = 0` for every
`m > 0` (the multiplicative `x` factor forces it, independent of the rest of the formula).

The moments follow from the same substitution `t = |x/x₀|^m` inside `∫xⁿφ dx`:

    M_n = x₀^n Γ((n+3)/m) / ((n+1) Γ(3/m))   for even n ≥ 0,     M_n = 0 for odd n,

in closed form for **every** integer `n ≥ 0`, needed up to `n ≈ 39` by the Laurent series of
Section 5. `M_0 = 1` and `M_2 = 1/2` reproduce the two normalizations that fixed `x₀` and `C`
above, an internal consistency check.

At `ξ = 0` the principal value of `𝒞φ` is an ordinary (non-singular) integral and the Plemelj term
vanishes, giving the closed form

    𝒞φ(0) = −Γ(1/m) / (x₀(m)² Γ(3/m)).

Halved and negated, this is PACM Fig. 5's `Re W_p(0)`: **1.000, 0.806, 0.729, 0.690** at
`m = 2, 3, 4, 5` respectively (test SG7), an independent check on the whole normalization chain
from a source that predates this package.

Excess kurtosis follows algebraically from the closed-form moments, with no quadrature:

    excess kurtosis = (9/5) Γ(3/m) Γ(7/m) / Γ(5/m)² − 3,     skewness = 0.

## 4. The `m = 2` reduction

At `m = 2`: `x₀(2)² = (3/2)Γ(3/2)/Γ(5/2) = (3/2)/(3/2) = 1`, so `x₀(2) = 1`; and
`C(2) = 1/(2·1·Γ(3/2)) = 1/(2·(√π/2)) = 1/√π`. With `x₀ = 1`, `Γ(1, x²) = e^{−x²}` (the upper
incomplete gamma function of index 1 is simply `exp(−t)`), so

    φ(x; 2) = (1/√π) Γ(1, x²) = e^{−x²}/√π = φ_Maxwellian(x),

**exactly**, and likewise `φ'(x;2) = −2x e^{−x²}/√π` matches the Maxwellian's closed form
identically. Every moment, `excess_kurtosis`, and `𝒞φ(0) = −Γ(1/2)/Γ(3/2) = −√π/(√π/2) = −2`
(matching `Zprime(0) = −2`) reduce exactly. This is why `T` in a `SuperGaussian` species means
exactly what `T` means for a Maxwellian one: both are fixed by the same second-moment
normalization (`conventions.md` 1.4, 1.5), and `m = 2` is literally the same distribution, not an
approximation to it.

**`T` is PACM's second-moment temperature** (PACM Eq. 15), which is precisely what the package's
variance normalization `∫x²φ dx = 1/2` already means (Section 1 of `docs/derivations/shapes.md`).
A `SuperGaussian` species' `T` and `m` therefore mean exactly what foo2026's `T_e` and `m_e` mean,
and nothing needs reconciling between the two sources on this point (spec Pitfall 11).

**A caution on PACM Section 3.2.** PACM's own velocity normalization there (its Eqs. 46, 47) is
**not self-consistent with its own Eq. 2** — it normalizes by `√2 v_Te` in one place and by a
differently-scaled width elsewhere. This package does not transcribe from Section 3.2 directly;
instead it transcribes in `x = v/v_th` with `v_th = √(2T/m_s)` throughout, and lets the `m = 2`
reduction to `Zprime` above, and the closed form `𝒞φ(0)`, be the arbiters of correctness
(`conventions.md` 1.11). Nobody should "fix" PACM Section 3.2's normalization to match this
package; the two are related by a fixed rescaling that this package does not need, since it never
transcribes a formula from that section directly.

## 5. The Laurent series at large `|ξ|`

For `|ξ|` large, expand `1/(x−ξ)` geometrically and integrate `∫xⁿφ' dx = −n M_{n−1}`
(by parts, using `φ(±∞) = 0` and `∫φ = 1`) term by term:

    𝒞φ(ξ) = P∫ φ'(x)/(x−ξ) dx ~ Σ_{n≥1} n M_{n−1} / ξ^{n+1}
           = 1/ξ² + 3/(2ξ⁴) + 5 M₄/ξ⁶ + 7 M₆/ξ⁸ + … ,

implemented generically in `distribution.jl` as `_cauchy_series`, against `moment(f, n)`, so that
any future shape with closed-form moments beyond `n = 4` can reuse it without change. **This series
is asymptotic, not convergent**: for fixed `m` it eventually diverges as `n → ∞`, and the correct
use of an asymptotic series is to sum only while the terms are shrinking, stopping at (or just
before) the smallest term. `_cauchy_series`'s five-clause stopping rule (spec D6) implements
exactly that: skip an exactly-zero odd-moment term without counting it; abandon immediately on a
non-finite term (mandatory, since at `ξ = 0` the first term is literally `Inf`, and an unguarded
"has it converged" test of `Inf ≤ rtol·Inf` is `true` — this destroyed `𝒞φ(0)` for every `m` before
the guard was written); abandon, without adding, the moment the terms start growing again; and
otherwise accept convergence once a term is smaller than `CAUCHY_SERIES_RTOL · |sum|`.

The **switch curve**, the smallest `|ξ|` at which the series converges at these constants
(`CAUCHY_SERIES_RTOL = 1e-12`, `CAUCHY_SERIES_MAXTERMS = 40`), moves outward as `m` decreases:

| m | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 5.0 | 8.0 |
|---|---|---|---|---|---|---|---|
| `|ξ|`_switch | 7.85 | 6.08 | 5.22 | 4.72 | 4.39 | 3.86 | 3.47 |

The term cap binds at `m ≤ 2` — at `m = 1.5, |ξ| = 8` the series needs 37 of its 40 allotted terms
— which is why the cap is not lowered. At `|ξ| = 10` the series converges for every `m ≥ 1.5`,
which is why the table's domain (Section 7, added at Step 1.SG.2) ends there: beyond it the series
always takes over.

## 6. The incomplete-gamma derivative rule

`SpecialFunctions.gamma(a, t)`, the two-argument upper incomplete gamma function used in `φ(x)`,
has no `ForwardDiff.Dual` method, so `_upper_gamma` supplies one (spec D5, design (a) of three
considered): combine the value from `gamma(a,t)`, the `a`-partial from a dedicated quadrature
`_dgamma_da(a,t) = ∫_t^∞ s^{a−1}\ln(s)\,e^{−s}\,ds` (the closed form `Γ(a)ψ(a)` at `t = 0`), and
the `t`-partial from the elementary `∂Γ(a,t)/∂t = −t^{a−1}e^{−t}`. The two other designs
considered — a fully generic incomplete gamma implementation, or a dual argument carried inside a
quadrature evaluated on every call — were rejected: the first would have to reproduce
`SpecialFunctions`'s accuracy on the *value* itself for no measured benefit, and the second is a
quadrature the shape cannot afford on every evaluation (`conventions.md` 5.2).

**The `x`-partial of `φ` is never taken through `Γ`.** At `x = 0`, `iszero(x)` and `x == 0` are
both `false` for a `ForwardDiff.Dual` whose value is zero but whose partials are not — the only
guard that works uniformly is `iszero(ForwardDiff.value(x))`. Without it, `abs(x/x₀)^m` evaluates
`0 · log 0 = NaN` in the `m`-partial once `m` is a dual (the incomplete gamma's chain rule to `t`
passes through `log(t)` via `∂Γ(a,t)/∂a`, and `t = 0` there is genuinely singular for `a < 1`).
The call method routes around this with the closed form `φ(0) = C(m)Γ(2/m)` — safe, since `Γ`'s
one-argument form differentiates cleanly — and `dfdx` routes around the same singularity by
forcing the power term `|x/x₀|^m` to an exact zero of the promoted type whenever
`iszero(ForwardDiff.value(x))`, which reproduces exactly what the unguarded formula gives at
`x = 0` when `m` is plain (needed for the second-derivative removable-singularity subtraction
inside `_cauchy_real`, Section 8) while never evaluating `0^{dual\ m}`.

## 7. The table (Step 1.SG.2)

*This section is completed at Step 1.SG.2.*

## 8. `_cauchy_real`'s breakpoint (spec D8)

`_cauchy_real`, Phase 0's generic real-axis Cauchy-integral fallback (`distribution.jl`), splits
its singularity-subtraction window `[−L, L]` (`u = x − ξ`, `L = CAUCHY_WINDOW = 6`) at one extra
point: `u₀ = −ξ` taken at `ξ`'s **primal value** (`_primal_value`, never `ξ` itself —
`conventions.md` 5.12's clause permitting a breakpoint at a dual's primal value, since the split
point is held fixed while differentiating and so changes neither the value nor the derivative of
the quadrature). `u₀` is where `x = 0`, and that is exactly where the `m`-partial of the
integrand is least smooth: it carries a `|x|^m \ln|x|` term whose higher `x`-derivatives blow up
at the origin, and QuadGK's adaptive Gauss–Kronrod grid adapts from the integrand's *value*, not
its partials, so without an explicit breakpoint there the grid is not fine enough where the
`m`-derivative needs it.

**Measured on this machine, 2026-09-08** (the oracle: a Richardson-extrapolated central
difference of the reference's own value, `D = (4f(h) − f(2h))/3` at `h = 1e-4`, the `(h,2h)` pair
— not `(h,10h)` — with the ratio-2 formula, verified against a ratio-10 form to `7.3e-10`):
relative error of the AD `m`-derivative of `_cauchy_real` (i.e. of `_sg_cauchy_reference`) against
this oracle,

| | ∂R/∂m, m = 2 | | | ∂R/∂m, m = 3.5 | | |
|---|---|---|---|---|---|---|
| ξ | 0.5 | 1 | 2 | 0.5 | 1 | 2 |
| before the breakpoint | 5.07e-6 | 1.70e-6 | 3.75e-6 | 7.5e-11 | 2.2e-11 | 1.9e-11 |
| with the breakpoint | **1.56e-9** | **2.96e-10** | **1.78e-11** | 7.4e-11 | 2.2e-11 | 4.9e-11 |

Three to five orders of magnitude better at `m = 2`, where the `|x|^m \ln|x|` term is strongest,
and no change at `m = 3.5`, where the plain form was already at round-off. On the Maxwellian the
change is invisible (`|_cauchy_real − Zprime|` is `8.05e-16` before and `7.91e-16` after over
Phase 0's own M3 grid, the two forms differing by `2.2e-16`), and Phase 0's own tests
(`test/distributions/test_distribution.jl`, `test/distributions/test_maxwellian.jl` M3–M5) pass
unchanged with this breakpoint in place.
