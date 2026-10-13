# The spectral density function S(k, ω)

> **Front matter (Step 1.6, 2026-09-08).** Windows 11, Julia 1.12.7; SpecialFunctions 2.9.0,
> QuadGK 2.11.3, ForwardDiff 1.4.5, ChainRulesCore 1.26.1. Measured on this machine, throughput
> (best mean of 5 passes of 2000 calls, warmed up, measured inside a function) and `@allocated`
> after warm-up:
>
> | Call | throughput | `@allocated` |
> |---|---|---|
> | `spectral_density`, one electron + one ion | 138 ns | 0 |
> | `spectral_density`, two electron + two ion species | 254 ns | 0 |
>
> Spec D24's reference numbers are 127 ns and 200 ns; both of mine run 8-27% slower, ordinary
> machine variation, and the allocation counts (the assertion that matters) agree exactly at
> zero for both species counts.

This document derives what `src/scattering/spectral_density.jl` implements: `S(k, ω)` for any
number of electron and ion species, its reduction to Sheffield's one-electron/one-ion form, the
origin of the ion weight, its units, and the two sum rules Step 1.2 tests.

## 1. The dressed-particle picture, and the many-species form

Sheffield's Eq. 3.4.3 (cited at Chapter 5's opening, Eq. 5.1.2) gives, for one Maxwellian-or-not
electron population and one ion population,

    S(k, ω) = (2π/k) |1 − χ_e/ε|² f_e0(ω/k) + (2πZ/k) |χ_e/ε|² f_i0(ω/k).      (Sheffield 5.1.2)

The physical picture behind it (the "dressed test particle" method, Sheffield Chapter 3): every
free-streaming particle is dressed by the polarization cloud the plasma's collective response
builds around it, and the observed density fluctuation spectrum is a sum of the *screened* free
streaming of every species. `1 − χ_e/ε` is the electrons' own screening factor: a bare electron's
ballistic contribution is reduced by the amount the plasma polarizes to shield it. `χ_e/ε` is
how strongly the electrons dress *each ion*: the ions are heavy and nearly stationary on the
electron plasma-wave timescale, so what an ion's motion contributes to the electron-density
fluctuations comes entirely through the electron cloud responding to it, weighted by `χ_e/ε`.
Both factors multiply that species' own bare 1-D velocity spectrum, `f_s0(ω/k)`.

**Multiple ion species** (Sheffield, p. 104, directly below Eq. 5.1.2): each ion species
contributes its own `f_i0,j(ω/k)` term, screened by the *same* `|χ_e/ε|²` (electrons dress every
ion the same way, since it is the electron response alone that screens ion motion on this
timescale), and weighted by its share of the total ion charge density:

    Σ_j (2π/k) (Z_j² N_j/N) |χ_e/ε|² f_i0,j(ω/k),     N = Σ_j N_j Z_j.       (Sheffield, p. 104)

**Multiple electron species**, by the identical argument applied to the electron term: each
electron population contributes its own bare spectrum `f_ej(ω/k)`, screened by the *total*
electron response `1 − χ_e/ε` with `χ_e = Σ_j χ_ej` (it is the sum of every electron species'
susceptibility that builds the shielding cloud, not any one species' alone), weighted by its
share of the total electron density. Sheffield does not need this generalization because his
worked examples carry one electron population, but the argument is the same one he makes for the
ions, applied to the other side of the dielectric function. This is `conventions.md` 1.7
([OQ-6](../../dev/open_questions.md#oq-6), decided 2026-09-06):

    S(k, ω) = (2π/k) [ |1 − χ_e/ε|² Σ_j (n_ej/n_e) f_ej(ω/k)
                     + |χ_e/ε|²    Σ_i (Z_i² n_i/n_e) f_i(ω/k) ],

    ε = 1 + χ_e + Σ_i χ_i,     χ_e = Σ_j χ_ej,

with `f_s(ω/k) = f_lab(sp, ω/k)` exactly (`conventions.md` 1.4, 1.5) — the kernel calls `f_lab`
and never rebuilds `φ_s((v−u_s)/v_th,s)/v_th,s` inline (`conventions.md` 1.9).

## 2. Sheffield's own weight is `n_e`, not the total ion density

**Sheffield's own multi-species term (p. 104) weights ion species `j` by `Z_j² N_j / N` with
`N = Σ_j N_j Z_j`.** By quasineutrality (`conventions.md` 3.6), `Σ_j N_j Z_j` **is** `n_e`: the
sum of each ion species' charge density equals the total electron density for a neutral plasma
(with one electron species; more generally, `Σ_j N_j Z_j = Σ_j' n_ej'` over all electron
species). So Sheffield's `N` **is** the package's `n_e`, and Sheffield's own printed multi-ion
form is `Z_i² n_i / n_e`, exactly `conventions.md` 1.7's weight.

**foo2026 Eq. 2 and PACM Eq. 29 instead read `N` as the total ion density `Σ_j N_j`** (not
`Σ_j N_j Z_j`), which differs from Sheffield's own `N` whenever any ion has `Z ≠ 1`. This is the
manuscript slip [OQ-6](../../dev/open_questions.md#oq-6) records, and this section is the
cleanest available proof that the package's weight is Sheffield's own, not a reinterpretation of
it: Sheffield's `N` is defined, in his own words on the same page, as `Σ_j N_j Z_j`, and that sum
is `n_e` by neutrality.

## 3. Reduction to Sheffield 5.1.2

For one electron species with `fraction = 1`, `n_e1 = n_e` (`_resolve_electron`), so the electron
sum degenerates to its single term, `f_e(ω/k)`, exactly Sheffield's `f_e0(ω/k)`. For one ion
species with `fraction = 1` and charge `Z`, `n_i = n_e/Z` (`_resolve_ion`), so the ion weight is

    Z² n_i / n_e = Z² (n_e/Z) / n_e = Z,

exactly Sheffield's `2πZ/k` prefactor on the ion term. `conventions.md` 1.7's formula reduces to
Eq. 5.1.2 term by term, not merely in total.

## 4. Units

`f_lab` has units of s/m (`src/plasma/parameters.jl`: `f_lab(sp, v) = φ((v−u)/v_th)/v_th`, and
`φ` is dimensionless while `v_th` is m/s, so `1/v_th` is s/m). `2π/k` has units of m (`k` is
m⁻¹). The bracketed sums are dimensionless weighted averages of `f_lab`, so they carry the same
s/m. `S = (2π/k) × [s/m] = s`. `spectral_density` returns seconds.

## 5. The two sum rules (Step 1.2), from Sheffield Eqs. 5.6.1–5.6.6

### 5.1 Sum rule 1: the Salpeter total, `T_e = T_i`

Sheffield Eq. 5.6.1 defines the total cross section `S_T(k) = ∫_{-∞}^{∞} S(k, ω) dω`. His
Salpeter approximation (`α ≫ 1`, weak ion-acoustic damping) gives, from the contour integral Eq.
5.6.2 combined with the susceptibilities (Eq. 5.6.3),

    S_T(k) ≃ 2π/(1+α²) + 2πZα⁴ / [(1+α²)(1+α²+α²ZT_e/T_i)].      (Sheffield 5.6.3)

At `T_e = T_i`, `ZT_e/T_i = Z`, and the two terms combine algebraically (common denominator
`(1+α²)(1+α²+α²Z)`, numerator `2π(1+α²+α²Z) + 2πZα⁴ = 2π(1+α²)(1+α²Z)` after expanding
`(1+α²Z)(1+α²) = 1+α²+α²Z+α⁴Z`, which matches term by term) to

    S_T(k) = 2π (1 + α² Z) / (1 + α² + α² Z),     at T_e = T_i.

This is the closed form SD2 tests, integrated numerically over the finite range of D20 (`±8 k
v_th,e`, breakpoints at `0, ±ω_ia, ±ω_epw`) rather than Sheffield's `±∞`, because the integrand
is `NaN` (not merely small) at `ω = ±∞` (`Z(±Inf) = NaN`; D20, pitfall 5). At `α → 0` both terms
of Eq. 5.6.3 are dominated by the electron term, `S_T → 2π`, the noncollective (free-streaming)
total — the arbiter of sign and normalization throughout this section.

### 5.2 Sum rule 2: the contour identity on the susceptibilities, arbitrary `T_e/T_i`

Sheffield rewrites `S(k, ω)` (Eq. 5.1.2) directly in terms of `χ_e`, `χ_i` (Eq. 5.6.4):

    S(k, ω) = − (2/(α²ω)) |(1+χ_i)/ε|² Im(χ_e)  −  (2T_i/(T_e α² ω)) |χ_e/ε|² Im(χ_i).

This uses the algebraic identity `1 − χ_e/ε = (1+χ_i)/ε`, immediate from `ε = 1+χ_e+χ_i` for one
electron and one ion species: `1 − χ_e/ε = (ε−χ_e)/ε = (1+χ_i)/ε`. This is what turns
Sheffield's electron term `|1−χ_e/ε|²` into the `|(1+χ_i)/ε|²` form his sum-rule section works
with.

Sheffield then notes (Eq. 5.6.5) that the quantity

    − (2/(α²ω)) Im{ (1+χ_i)χ_e / (1+χ_e+χ_i) }
      = − (2/(α²ω)) |(1+χ_i)/ε|² Im(χ_e)  −  (2/(α²ω)) |χ_e/ε|² Im(χ_i)

differs from Eq. 5.6.4 only by the missing `T_i/T_e` factor on the ion term — so it is **not**
`S(k, ω)` itself, but it is a quantity built from `χ_e`, `χ_i` alone (no separate `f_e0`, `f_i0`
evaluation needed) whose integral is exactly computable by contour integration in the upper
`ω`-half-plane (Appendix A.5), using `χ_e, χ_i → 0` as `|ω| → ∞` and `χ_e → α², χ_i → α²ZT_e/T_i`
as `ω → 0`:

    ∫_{-∞}^{∞} [ − (2/(α²ω)) Im{ (1+χ_i)χ_e/(1+χ_e+χ_i) } ] dω
        = 2π [1 + α²(ZT_e/T_i)] / (1 + α² + α²(ZT_e/T_i))            (Sheffield 5.6.6)

**The sign.** Sheffield's Eq. 5.6.6 is printed with the same leading minus as Eq. 5.6.5's
definition. That minus sign belongs to Sheffield's **Appendix A.4 time convention**
(`docs/derivations/plasma_dispersion.md` Section 7), the opposite of the one this package uses
(Chapter 5's, where `Im χ_e > 0` for `ω > 0`). Written in **our** convention, with
`α_i² ≡ α_e² Z T_e/T_i` (`docs/derivations/susceptibility.md` Section 4), the identity is the
same magnitude with the sign that makes the left side manifestly positive (the integrand of a
non-negative spectral density cannot integrate to a negative number):

    ∫_{-∞}^{∞} (2/(α_e² ω)) Im[(1+χ_i) χ_e / ε] dω = 2π (1 + α_i²) / (1 + α_e² + α_i²).

**The arbiter is the `α → 0` limit**: both Eq. 5.6.6's right-hand side and the closed form above
tend to `2π` as `α_e → 0`, the same noncollective total as sum rule 1 above; taking Sheffield's
printed sign literally would make the left side integrate to `−2π` there, which cannot be a
non-negative spectral quantity's integral. This is `conventions.md` 8.4/8.8's kind of check
applied to a whole equation rather than a single test point, and it is why the package's
formula (D21 row 7, D10) carries the opposite sign from what is printed in Sheffield.

**This sum rule (SD3) exercises `susceptibility` and `dielectric` only, never
`spectral_density`** — it is the independent check that the susceptibilities alone, before any
`f_lab` evaluation, already carry Eq. 5.1.2's normalization correctly.

Both sum rules are integrated exactly as `docs/derivations/spectral_density.md` Section 5.1
describes: finite limits `±8 k v_th,e`, breakpoints at `0, ±ω_ia, ±ω_epw` (Sheffield Eqs. 5.3.5,
5.3.9), `rtol = 1e-10`, `atol = 1e-14` (spec D20).

## 6. A real charge and the mixture it stands for (Phase 4.J)

A species' `charge` is a real number since Step 4.J.1 ([OQ-60](../../dev/open_questions.md#oq-60),
decided 2026-10-10). Nothing in this document's formulas changes; this section says what a
non-integer charge means, and why no new weight is needed.

**Every place an ion's charge enters is the product `Z_i f_i`.** With charge fractions
(`conventions.md` 3.6) an ion species of charge `Z_i` and fraction `f_i` has `n_i = f_i n_e / Z_i`
(`_resolve_ion`, `_density`). It enters the spectrum twice, through its weight in the ion sum
(Section 1) and through its susceptibility in `ε`:

    Z_i² n_i / n_e = Z_i f_i,
    α_i² = n_i (Z_i e)² / (ε₀ T_i k²) = Z_i f_i · n_e e² / (ε₀ T_i k²),

and the rest of `χ_i = −(α_i²/2) 𝒞φ_i(ξ_i)` and of `f_lab(sp_i, ω/k) = φ_i(ξ_i)/v_th,i` depends on
`T_i`, `m_i`, `u_i` and the shape alone (`ξ_i = (ω/k − u_i)/v_th,i`, `v_th,i = √(2T_i/m_i)`).

**Charge states of one element are one species.** If species `1..N` share `T`, `m`, `u` and the
shape, their contributions to the ion sum and to `ε` are `(Σ Z_i f_i)` times one common function, so
one species of charge `Z*` and fraction `x` reproduces them exactly — in `dielectric`,
`spectral_density`, the convective gain (which reads only `χ_e` and `ε`), `measure` and every
derivative — if and only if `Z* x = Σ Z_i f_i`:

    x = Σ f_i,        Z* = Σ Z_i f_i / Σ f_i = Σ Z_i² n_i / Σ Z_i n_i = ⟨Z²⟩/⟨Z⟩,

the number-weighted moments over that element's states. **The spectrum constrains `⟨Z²⟩/⟨Z⟩` of each
element and nothing else about its charge-state distribution**, and that is what a real `charge` is.
CZ5 holds the identity for C⁴⁺ + C⁶⁺ against one carbon at `Z*`, Maxwellian and kappa, to
`rtol = 1e-12`.

**Two consequences.** (i) `Z*` is not the number-average charge `Z̄ = ⟨Z⟩ = x / Σ (f_i/Z_i)`:
`Z̄ ≤ Z*`, equal only for one state; for the C⁴⁺/C⁶⁺ pair `Z* = 10 − 24/Z̄` (5.2 at `Z̄ = 5`). One
carbon at `Z̄` is a different spectrum: its weight is `Z̄ x` against the pair's `Z* x` (CZ6: 0.96 of it
at equal fractions). (ii) The number density the package derives for the species, `x n_e / Z*`, is
that of the equivalent single-state ion, below the mixture's true number density `x n_e / Z̄` by
`Z̄/Z*`. Recovering `Z̄` or the true density needs an ionization model, which the package does not have.

**The floor.** For integer states `Z_i ≥ 0`, `Z_i² ≥ Z_i`, so `⟨Z²⟩/⟨Z⟩ ≥ 1` for any mixture with at
least one ion (neutrals drop out of both moments), with equality only for pure `Z = 1`. `ion` refuses
a charge below 1 or not finite; there is no ceiling.

**In the derivative.** A fitted charge is a leaf (`ions.<name>.charge`, dimensionless) and reaches
the Jacobian as any leaf does, through the dual trace and `plasma(trace, j)`, whose density is
re-resolved from the per-bin charge. The one division by a charge goes through `_divide_by_charge`
(Phase 4.J spec D5): a charge that carries no derivative is divided by its value, so the partials are
`p / Z` exactly as they were while the charge was an integer; ForwardDiff's dual-by-dual division
would form them as `p · inv(Z)`, which differs in the last place for a charge that is not a power of
two, and would have moved every Jacobian of a plasma with carbon by round-off.
