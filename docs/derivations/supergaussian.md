# The `SuperGaussian` shape

> **Front matter (Step 1.SG.4, 2026-09-09).** Windows 11, Julia 1.12.7; SpecialFunctions 2.9.0,
> QuadGK 2.11.3, ForwardDiff 1.4.5, ChainRulesCore 1.26.1. Measured on this machine, latency (one
> isolated call, warmed up) and throughput (best of 5 passes, warmed up, measured inside a
> function per Phase 0 pitfall 15 — a closure over a non-`const` global lies) and `@allocated`
> after warm-up:
>
> | Call | latency | throughput | `@allocated` |
> |---|---|---|---|
> | `susceptibility`, super-Gaussian species (`m = 3`) | 100 ns | 102.7 ns | 0 |
> | `susceptibility`, Maxwellian (spec D23's baseline) | — | 86.9 ns | 0 |
> | `spectrum!`, 512 bins, super-Gaussian electrons + Maxwellian D ions | 123.6 µs | 139.8 µs | 0 |
> | `spectrum!`, 512 bins, Maxwellian (this machine, same run, cross-check) | — | 63.0 µs | 0 |
> | `spectrum!`, 512 bins, Maxwellian (spec D23's baseline) | — | 84.5 µs | 0 |
> | `cauchy_dfdv`, plain and dual-`m` shape | — | — | 0, 0 |
> | `d(spectrum)/dT_e`, 512 bins (`ForwardDiff.derivative`) | — | 312.3 µs | 113 630 bytes |
> | `d(spectrum)/dm`, 512 bins (`ForwardDiff.derivative`) | — | 3979.5 µs | 883 646 bytes |
> | first super-Gaussian `spectrum` call, fresh session (`using` + build + compile + run) | 1.877 s | — | — |
> | first Maxwellian `spectrum` call, fresh session (cross-check) | 0.899 s | — | — |
>
> **`spectrum!` on a super-Gaussian plasma runs at 139.8 µs against the Maxwellian's 63.0 µs on
> this machine (ratio 2.22×) — or 84.5 µs from spec D23 (ratio 1.65×) — comfortably inside the
> phase's 3× goal either way.** `susceptibility` and `cauchy_dfdv` are `@allocated`-zero for both
> a plain shape and a dual-`m` shape (the table lookup and the Plemelj term `dfdx` are both closed
> forms, so neither ever calls the incomplete gamma function). **`spectrum!` on a dual-`m` plasma
> is *not* zero-allocation**, unlike the plain case: `spectral_density` calls `f_lab`, which
> evaluates `φ(x)` directly (not just its Cauchy integral), and at a dual `m` this reaches
> `_upper_gamma`'s both-dual branch, whose `a`-partial is a genuine quadrature (Section 6, D5,
> [OQ-9](../../dev/open_questions.md#oq-9) trap 1) — the identical, already-documented cost the
> `d(spectrum)/dm` row above measures directly, not a new defect. The `d(spectrum)/dT_e` row pays
> a smaller, related cost (a same-tagged zero-partial dual `m`, needed only so `Plasma`'s element
> type stays uniform across species, per `test/scattering/test_spectrum_supergaussian.jl`'s SS2/SS3
> comments) without ever entering that quadrature, since a zero-partial `a` short-circuits it
> (round 4's fix, D5). The first-call latency carries the `{FT, L}` type parameter's one-time
> specialization cost: about **1.0 s** more than the Maxwellian's first call, on top of the
> package's own `using OTIS` time (~0.36 s, not part of either figure above).

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

**The domain.** `R(ξ; m) = P∫ φ'(x; m)/(x−ξ) dx`, the real (principal-value) part of `𝒞φ` on the
real axis, is tabulated on `|ξ| ∈ [0, 10]` (`SG_XI_MAX`) `× m ∈ [1.5, 8]` (`SG_M_LO`, `SG_M_HI`),
evaluated by the reference quadrature of Section 6 at the Chebyshev–Lobatto nodes of every ξ
panel and of the m interval, at Phase 0's `CAUCHY_RTOL = 1e-10`, `CAUCHY_ATOL = 1e-12`, through
`_sg_uncached(m)` so that the build never touches the table it is building
([OQ-31](../../dev/open_questions.md#oq-31) item 1). Its imaginary part is exactly `iπ φ'(ξ)`,
never tabulated; `R` is even in ξ (`φ'` is odd), so only `ξ ≥ 0` is stored and evaluation reads
`|ξ|`.

**The panels and degrees, as built (this machine, round 2, 2026-09-08, at the corrected
`τ = 1e-9`).** Starting from the default panel edges `(0, 0.25, 0.5, 1, 2, 4, 10)`
(`SG_PANEL_EDGES`), **no panel needed to bisect**:

| Panel | `[a, b]` | `n_ξ` |
|---|---|---|
| 1 | [0, 0.25] | 32 |
| 2 | [0.25, 0.5] | 16 |
| 3 | [0.5, 1] | 16 |
| 4 | [1, 2] | 32 |
| 5 | [2, 4] | 64 |
| 6 | [4, 10] | 32 |

with one global `n_m = 32` and slice length `L = Σ(n_ξ+1) = 198`. **Every `xi_converged[p]` is
`true` and `m_converged` is `true`** (`test/distributions/test_supergaussian_table.jl` TB3): no
panel or direction stopped while still failing its trailing-coefficient check, so there is no cap
finding to report this round.

**τ's escalation history** (spec D12, revised 2026-09-08 round 1): the fit started at
`SG_FIT_TOL = 1e-9`, the value target with one order of margin, and **passed on the first try**
— TB7 (`∂R/∂ξ`) measured a worst absolute error of `6.0e-7` against a floor of `3.8e-6`
(headroom ×6.4) and TB8 (`∂R/∂m`) measured `2.1e-8` against a floor of `1.5e-6` (headroom ×69), so
no escalation to the `1e-10` floor was needed. **`SG_FIT_TOL` is pinned at `1e-9`.**

Round 1's build, at the since-corrected `τ = 1e-12`, gave panel edges bisected once (`[0, 0.125]`,
`[0.125, 0.25]`, …), `n_ξ` up to `64` (its own cap) on two panels, `n_m = 48` (its own cap),
`L = 247` — three cap findings, none of them a real accuracy problem: `τ = 1e-12` was fitting the
generator's own quadrature noise (`CAUCHY_RTOL = 1e-10` measures ~1e-11 relative against a
BigFloat oracle), not genuine interpolation error, which is exactly why chasing it bought larger
degrees without moving TB4's or TB7/TB8's actual errors. See the round-2 section of
`dev/reports/phase1SG_report.md` for the full three-way comparison.

**The error map** (`test/distributions/test_supergaussian_table.jl` TB13, 500 seeded points,
`Xoshiro(20260908)`, 400 uniform over the domain plus 100 concentrated in `ξ ∈ [0, 0.5]`,
`m ∈ [1.5, 2]`): worst relative value error per panel, against the target `rtol = atol = 1e-8`:

| Panel | `[a, b]` | worst value error |
|---|---|---|
| 1 | [0, 0.25] | 5.3e-10 |
| 2 | [0.25, 0.5] | 8.0e-14 |
| 3 | [0.5, 1] | 4.4e-14 |
| 4 | [1, 2] | 1.7e-13 |
| 5 | [2, 4] | 2.2e-11 |
| 6 | [4, 10] | 5.1e-11 |

worst near `ξ = 0` at `m ≤ 2` (the 100 concentrated points): `5.3e-10`, inside target by more than
an order of magnitude (worst absolute value error over the whole 500-point set: `1.3e-9`). `∂R/∂ξ`
worst absolute error `6.0e-7` against the floor `3.8e-6` (headroom ×6.4); `∂R/∂m` worst absolute
error `2.1e-8` against the floor `1.5e-6` (headroom ×69). TB11(c)'s coarse sanity bound on
`|R̂′(0)|`, the domain's least-smooth point (`m = 1.5`), measured `3.5e-5` against its `3.8e-4`
threshold (100× TB7's floor, and barred from driving `τ`, D12) — comfortable, and never asked to
be tighter.

**The smoothness caveat.** `R` is not analytic at `ξ = 0`: its leading non-analytic term is
`|ξ|^{m+1}`, so `R′(0) = 0` for every `m > 0` (`m + 1 > 1`) but `R` is only `C^{⌈m+1⌉-1}` there —
for `m = 1.5`, only `C²` (the second derivative exists; the third does not). A Chebyshev series
fitted on a one-sided panel `[0, b]` has no way to know `R` is even, so it does not automatically
reproduce `R′(0) = 0` to the same relative accuracy the fit achieves in its interior; that is
exactly what drove `τ`'s escalation and panel 1's bisection above, and is the risk
[OQ-31](../../dev/open_questions.md#oq-31) item 4 anticipated.

**Build and evaluation cost, this machine (Windows 11, Julia 1.12.7; SpecialFunctions 2.9.0,
QuadGK 2.11.3, ForwardDiff 1.4.5, ChainRulesCore 1.26.1), round 2, 2026-09-08:**

| Quantity | Kind | Value |
|---|---|---|
| Reference evaluations for one build | count | 6 534 |
| Generation + fit time | latency | ≈ 0.19–0.32 s (budget: 10 s) |
| `Base.summarysize(SG_TABLE)` | size | 52 814 bytes |
| Precompile time, before (`be55f68`) → after the table | latency | ≈ 4.3–4.7 s → ≈ 7.3–7.7 s (Δ ≈ +3.0 s; budget: Δ < 1 s, **missed, and by essentially the same margin as round 1's Δ ≈ +3.1 s** despite the table itself being 2.4× smaller — see the caveat below) |
| Load time (warm cache), before → after | latency | 0.75 s → 0.82 s (Δ ≈ +0.07 s; within the < 1 s budget) |
| `_sg_table_eval`, plain `ξ` | throughput | 39.3 ns |
| `_sg_table_eval`, dual `ξ` | throughput | 52.8 ns |
| `_sg_table_eval` on a dual-`m` shape (eval only, slice already built) | throughput | 41.7 ns |
| `SuperGaussian(m)` construction, plain | throughput | **5.53 µs** |
| `SuperGaussian(m)` construction, dual `m`, chunk 1 | throughput | 9.42 µs |
| `SuperGaussian(m)` construction, dual `m`, chunk 4 | throughput | 18.5 µs |
| `SuperGaussian(m)` construction, dual `m`, chunk 8 | throughput | 21.0 µs |
| `SuperGaussian(m)` construction, dual `m`, chunk 12 | throughput | 25.3 µs |

`@allocated` is `0` for `_sg_table_eval` (plain, dual `ξ`, dual-`m` shape), for `reconstruct`, and
for `SuperGaussian(m)` construction at every chunk size measured. **`L = 198` and plain
construction (`5.53 µs`) are now comfortably under the `L = 102`, `< 10 µs` anchor D3's caching
decision was measured at** — round 1's `L = 247`, `15.0 µs` overshoot was entirely a symptom of
fitting quadrature noise at `τ = 1e-12`, not of anything wrong with the caching design itself.

**The precompile-time caveat.** The Δ ≈ +3.0 s precompile increase barely moved between round 1
(table 2.4× larger, 3× more reference evaluations, ~4× the build time) and round 2, which shows
the increase is dominated by *compiling the two new files' code* (the generic Clenshaw
recurrence, the dual-number paths, the adaptive-fit machinery) rather than by the table's own
size or the cost of running `_sg_build_table()` at precompile time (0.2–0.3 s of the 3.0 s).
Shrinking the table further would not be expected to move this number appreciably. See the
hand-off report for the go/no-go judgement this is input to.

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

## 9. Step 1.SG.4 — the forward model, and the accuracies achieved

`spectrum!` on a super-Gaussian plasma runs through exactly the code Phase 1 wrote for the
Maxwellian case (`src/scattering/spectrum.jl`, unchanged in this phase): nothing above the shape
interface is touched anywhere in Phase 1.SG; `susceptibility` calls `cauchy_dfdv` and that is the
whole coupling (spec Section 7, roadmap Step 1.SG.4).

**The `m = 2` reduction, at the level of the whole spectrum
(`test/scattering/test_spectrum_supergaussian.jl` SS1).** With one electron species carrying
`SuperGaussian(2.0)` against the same plasma with `Maxwellian()`, D22's geometry (foo2026 Section
III.A), all 512 bins: through `_sg_cauchy_reference` the two spectra agree to **`1e-12`
relative** — round-off, since φ, φ′ and `𝒞φ(0)` all reduce exactly at `m = 2` (Section 4). Through
the table they agree to **`1e-8` relative**, the table's own accuracy target (Section 7) rather
than round-off, because the table interpolates `R(ξ; 2)` rather than evaluating the `m = 2` closed
form directly. **The two tolerances are the point**: one certifies the physics, the other
certifies the table.

**The regression case (SS4, SS5), foo2026's EPW geometry (D22).** Through
`_sg_cauchy_reference` — the path SS4 exercises — the three peak positions, FWHM values and peak
values at `m ∈ {2, 3, 5}` reproduce D22's nine printed numbers (Section 6 of the spec) to **`1e-9`
relative**, confirming that the shape and the scattering geometry are both right *before* the
table is asked to reproduce anything. Through the table (SS5), the same nine numbers reproduce
to well inside the `1e-6` relative tolerance the test asserts — table-versus-printed deltas of
**4.2e-11 nm** in peak position, **4.8e-11 nm** in FWHM, and **4.1e-4** in peak value (against a
peak value of order `7.7e+07`, about **3e-12 relative**) — essentially an exact reproduction of
D22's nine numbers, not merely one inside the stated tolerance. `FWHM(5) < FWHM(3) < FWHM(2)`
(weaker Landau damping at higher `m` at fixed second-moment `T_e`; PACM Fig. 5, Milder 2019) holds
through both paths, and the peak moves blue by `1.16` nm from `m = 2` to `m = 5`, matching D22's
characterization.

**What this rests on** is Section 7's error map: the table's worst relative value error over its
whole `(ξ, m)` domain is `5.3e-10`, in the least smooth region (near `ξ = 0` at `m ≤ 2`) —
comfortably inside the `1e-8` target the spectrum-level tolerances above are built on, which is
why the table-versus-printed deltas above come out so much smaller than the `1e-6` the test
actually asserts: a table accurate to `1e-8` in the shape's own Cauchy integral propagates to an
even smaller error in a derived quantity like a peak position or a FWHM, which depends on many
bins at once rather than on a single evaluation of `R`.
