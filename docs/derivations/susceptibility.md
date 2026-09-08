# Susceptibility and the dielectric function

This document derives what `src/scattering/susceptibility.jl` implements: the per-species
susceptibility `χ_s(k, ω)` in the package's shape normalization, and the longitudinal dielectric
function `ε(k, ω)`.

## 1. Sheffield Eq. 3.3.9, in Gaussian units

For a collisionless, unmagnetized, one-dimensional-reduced species `s` with equilibrium
distribution `f_s0(v)` (normalized to 1), the linear susceptibility is (Sheffield Eq. 3.3.9,
Gaussian units)

    χ_s(k, ω) = − (ω_ps² / k²) ∫ (∂f_s0/∂v) / (v − ω/k) dv,     ω_ps² = 4π n_s q_s² / m_s,

with the velocity integral taken on the Landau contour (the pole at `v = ω/k` approached from
`Im ω > 0`).

## 2. To SI

Gaussian and SI differ only in how Coulomb's law is written: Gaussian's `4π n q²/m` becomes SI's
`n q²/(ε₀ m)` under `e² → e²/(4πε₀)` (`conventions.md` 1.10). So in SI,

    ω_ps² = n_s q_s² / (ε₀ m_s),     χ_s(k, ω) = − (ω_ps² / k²) ∫ (∂f_s0/∂v) / (v − ω/k) dv.

## 3. To shape form

Write `f_s0(v) = f_lab(sp, v) = φ_s(x)/v_th,s` with `x = (v − u_s)/v_th,s` (`conventions.md`
1.4, 1.5). Then `∂f_s0/∂v = φ_s'(x)/v_th,s²`, and changing the integration variable from `v` to
`x` (`dv = v_th,s dx`, and `v − ω/k = v_th,s(x − ξ_s)` with `ξ_s = (ω/k − u_s)/v_th,s`, the
`xi(sp, k, omega)` of `plasma/parameters.jl`):

    ∫ (∂f_s0/∂v)/(v − ω/k) dv = (1/v_th,s²) ∫ φ_s'(x) v_th,s dx / (v_th,s(x − ξ_s))
                               = (1/v_th,s²) ∫ φ_s'(x)/(x − ξ_s) dx
                               = 𝒞φ_s(ξ_s)/v_th,s²      (conventions.md 1.5's Cauchy integral).

So

    χ_s(k, ω) = − (ω_ps² / (k² v_th,s²)) · 𝒞φ_s(ξ_s).

Now identify the scattering parameter. `conventions.md` 1.6 defines
`α_s² = n_s q_s² / (ε₀ T_s k²) = 1/(k λ_D,s)²` with `λ_D,s² = ε₀ T_s/(n_s q_s²)`. Since
`v_th,s² = 2T_s/m_s` (the factor of two of `thermal_speed`, `conventions.md` 1.4),

    ω_ps² / (k² v_th,s²) = [n_s q_s²/(ε₀ m_s)] / [k² · 2T_s/m_s] = n_s q_s² / (2 ε₀ T_s k²) = α_s²/2.

Therefore

    χ_s(k, ω) = − (α_s²/2) · 𝒞φ_s(ξ_s),     α_s = alpha(sp, k),     ξ_s = xi(sp, k, omega).

This is exactly `conventions.md` 1.6 and the formula `susceptibility` implements. **The minus
sign and the `/2` appear once**, in `susceptibility`, with a comment; `alpha(sp, k)` returns
`α_s`, not `α_s²`, so the square is written once there too (`conventions.md` Phase 0 pitfall
20).

## 4. The Maxwellian check: Sheffield Eq. 5.2.5 and Eq. 5.2.6

For a Maxwellian electron species, `𝒞φ(ξ) = Z'(ξ)` exactly (`conventions.md` 1.5), so

    χ_e = −(α_e²/2) Z'(x_e) = α_e² [Rw(x_e) + i Iw(x_e)],

using `Rw(x) = −Re Z'(x)/2`, `Iw(x) = −Im Z'(x)/2` (`docs/derivations/plasma_dispersion.md`
Section 4). This is Sheffield Eq. 5.2.5 exactly, and `test/scattering/test_susceptibility.jl`
(CH1) checks it against `Rw`, `Iw` computed independently from `dawson`.

For a Maxwellian ion species with density `n_i = n_e/Z` (the quasineutral closure of one
electron and one ion species at `Z_i = Z`), write `α_i²` in terms of the electron's `α_e²`:

    α_i² = n_i q_i² / (ε₀ T_i k²) = (n_e/Z)(Ze)² / (ε₀ T_i k²) = Z e² n_e / (ε₀ T_i k²)
         = α_e² · (T_e/T_i) · Z            (using α_e² = n_e e²/(ε₀ T_e k²)),

so `α_i² = α_e² Z T_e/T_i`, and

    χ_i = −(α_i²/2) Z'(x_i) = α_e² (Z T_e/T_i) [Rw(x_i) + i Iw(x_i)],

Sheffield Eq. 5.2.6 exactly (CH2 checks this).

## 5. The static limit: Debye screening

At `ω = 0`, `ξ_s = −u_s/v_th,s`; for a species at rest (`u_s = 0`), `ξ_s = 0`. Since
`Z'(0) = −2(1 + 0·Z(0)) = −2` exactly (the analytic formula, no cancellation), and for a
Maxwellian `𝒞φ(0) = Z'(0) = −2`,

    χ_s(k, 0) = −(α_s²/2)(−2) = α_s².

This is Debye screening: `ε(k, 0) = 1 + Σ_s α_s² = 1 + Σ_s 1/(kλ_D,s)² = 1 + 1/(kλ_D)²` with
`1/λ_D² = Σ_s 1/λ_D,s²`, the usual combination of Debye lengths in quadrature. CH3 checks
`χ_s(k,0) == alpha(sp,k)^2` to `1e-14` relative (measured 0.0, exact, since both sides reduce to
the same floating-point expression through `Z'(0) = −2`). This limit is stated as `ξ_s → 0`, not
as `k → 0`, since the code never takes `k → 0` (`wavevector` requires `k ≠ 0`); the two are the
same physical limit, but only one is a limit the code can take.

## 6. The Landau sign

On the real axis, `Im Z(x) = +√π e^{−x²} > 0`
(`docs/derivations/plasma_dispersion.md` Section 1), so `Im Z'(x) = −2x·Im Z(x)`, which is
negative for `x > 0`; hence `Im χ_e = −(α²/2)·Im Z'(x_e) > 0` for `x_e > 0`, i.e. for `ω > 0`
(a blue shift) with `u = 0`. This is the package's Landau sign convention, matching Sheffield's
**Chapter 5** (Eqs. 5.2.5–5.2.8, where `Iw(x) > 0` for `x > 0`). **Sheffield's Appendix A.4 uses
the opposite time convention** (`docs/derivations/plasma_dispersion.md` Section 7): a derivation
that starts from Appendix A.4's contour and reads off a sign will get `Im χ_s` conjugated
relative to this package. CH4 is the one-line check: `imag(susceptibility(e, k, ω)) > 0` for
`ω > 0`.

## 7. The dielectric function

`ε(k, ω) = 1 + Σ_s χ_s(k, ω)`, summed over every species, electrons and ions alike (Sheffield
Eq. 3.3.10). `dielectric` also returns `χ_e = Σ_j χ_ej`, the sum over electron species only,
because `spectral_density` (Step 1.2) needs both `1 − χ_e/ε` and `χ_e/ε` and computing `χ_e`
twice, once inside `ε` and once again for its own sake, would be the "one-2π-written-twice"
mistake `conventions.md` 1.9 warns against. `dielectric` returns a two-field `NamedTuple`,
`(epsilon, chi_e)`, both `Complex`, an immutable bitstype: measured 0 bytes allocated (spec D9,
D24).
