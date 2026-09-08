# The scattering geometry and k(λ_s)

This document derives what `src/scattering/geometry.jl` implements: the probed wavevector and
frequency shift from the light-wave dispersion relation, the exact and approximate forms of `k`,
the mean **k̂** convention, and the polarization factor.

## 1. ω and k from momentum and energy conservation

A probe photon of wavevector **k**_i and frequency ω_i scatters into **k**_s, ω_s. Momentum and
energy conservation for the fluctuation probed give (Sheffield Section 5.4.1)

    k = k_s − k_i,     ω = ω_s − ω_i,                                       (Sheffield, p. 113)

with ω positive for a blue shift (λ_s < λ_i), the package's sign convention throughout
(`conventions.md` 3.1). `wavevector` computes `ω` as `ω_s − ω_i` from the two frequencies
computed independently, **not** as `2πc(1/λ_s − 1/λ_i)`: the two expressions differ in the last
bits of a `Float64`, and `ω_s − ω_i` is the definition (spec D6).

## 2. The light-wave dispersion relation

Because the scattering takes place inside the plasma, `k_i` and `k_s` are not the vacuum
wavenumbers `2π/λ`: each electromagnetic wave in the plasma obeys the transverse dispersion
relation `ω² = ω_pe² + c²k²`, so

    k_i = √(ω_i² − ω_pe²)/c,     k_s = √(ω_s² − ω_pe²)/c,                (Sheffield 5.4.1, 5.4.2)

with `ω_pe² = n_e e²/(ε₀ m_e)` from the **total** electron density `plasma.n_e` (every electron
species screens the light wave, not just the one being probed). Both `ω_i` and `ω_s` must exceed
`ω_pe` for the corresponding wave to propagate; below that, `k_i` or `k_s` would be imaginary,
which is the **cutoff** `wavevector` errors on (Section 5 below).

## 3. The exact magnitude, and the constant-k approximation

`k = |**k**_s − **k**_i|` at the scattering angle `θ` between the two, by the law of cosines:

    k = √(k_i² + k_s² − 2 k_i k_s cos θ)                                        (`k_model = :exact`)

This is the default (`conventions.md` 3.3, [OQ-8](../../dev/open_questions.md#oq-8)): `k`
depends on `λ_s` (through `k_s`) and on `n_e` (through both `k_i` and `k_s`), which matters
because the electron-plasma-wave feature separation depends on `k` (Eq. 5.4.6, Step 1.4).

The usual textbook approximation drops the density dependence entirely, using the **vacuum**
wavenumber `k_i = 2π/λ_i` and the small-angle-independent identity `k_ia ≃ 2 k_i sin(θ/2)`
(Sheffield Eq. 5.4.3, valid when `k_i ≃ k_s`):

    k = (4π/λ_i) sin(θ/2) = 2 · (2π/λ_i) · sin(θ/2)                       (`k_model = :approximate`)

independent of `λ_s` and of `n_e` — `wavevector` uses the vacuum `k_i = 2π/λ_i` here, never the
plasma-corrected one (spec D6, pitfall 3).

## 4. The measured deviation: `k/k_approx − 1 → −n_e/(2 n_cr)`

At `λ_s = λ_i` (`ω = 0`), `ω_s = ω_i` so `k_s = k_i` (the *exact*, plasma-corrected `k_i`) for
**any** `n_e`, and the exact-k formula collapses to `k = 2 k_i sin(θ/2)` exactly, with the same
angle factor as the approximation but the plasma-corrected `k_i` in place of the vacuum one. So

    k / k_approx = k_i / k_i,vacuum = √(ω_i² − ω_pe²) / ω_i = √(1 − ω_pe²/ω_i²).

Writing the critical density `n_cr = ε₀ m_e ω_i²/e²` (the density at which `ω_pe = ω_i`, so that
`ω_pe²/ω_i² = n_e/n_cr` exactly, since `ω_pe² ∝ n_e`), a first-order expansion for `n_e ≪ n_cr`
gives

    k/k_approx = √(1 − n_e/n_cr) ≈ 1 − n_e/(2 n_cr),      so     k/k_approx − 1 ≈ −n_e/(2 n_cr),

with the next correction `O((n_e/n_cr)²)`. GE6 asserts this at `λ_i = 526.5` nm, `θ = 90°`,
`n_e` from `1e13` to `1e19` cm⁻³, to `1e-3` relative — loose because the tolerance must
accommodate the *n_cr* ≈ 4.02e27 m⁻³ reference case's `n_e/n_cr ≈ 2.5e-3` at the highest density
in the grid, where the quadratic correction is still two orders below the tolerance (spec D21
row 10).

## 5. The cutoff

`ω ≤ ω_pe` makes `k_i` or `k_s` imaginary: the corresponding light wave cannot propagate (it is
evanescent, reflected at the critical-density surface). `wavevector` checks this for **both**
`ω_i` and `ω_s`, in **both** `k_model`s, before computing either branch's `k` — the cutoff is a
statement about the plasma and the two frequencies, not about which `k` formula is used, and
running the check unconditionally means switching `k_model` can never silently produce a
spectrum from a region where no light propagates (spec D6). The cutoff wavelength quoted in the
error messages is `2πc/ω_pe`, the vacuum wavelength at which `ω = ω_pe`.

## 6. The mean **k̂**

`conventions.md` 3.2: **one direction serves the whole spectrum**. The true direction of
**k** = **k**_s − **k**_i drifts slightly across a feature as `|**k**_s|` varies with `λ_s`; the
package uses the direction at `λ_s = λ_i` (the "mean" **k**) everywhere, and every species flow
`u` is quoted along it. In the scattering plane, with `k̂_i` the probe direction and `ŝ` the
collection direction (angle `θ` from `k̂_i`),

    k̂ = (ŝ − k̂_i) / |ŝ − k̂_i|,

which bisects the *exterior* angle of the probe and collection directions, at angle `(π+θ)/2`
from the probe direction:

```
                                   k_s
                                  ^
                                 /
                                /
                               /  theta
        k_i                   /
        ------------------->-*------------------>   probe direction
                             |\
                             | \
                    (pi+th)/2|  \   k = k_s - k_i, drawn at lambda_s = lambda_i
                             v   v
                                  k_hat  (the mean k-hat: one direction for the
                                          whole spectrum, conventions.md 3.2)
```

`u > 0` Doppler-shifts a species' feature to positive `ω` (a blue shift), consistent with rule
3.1. Nothing in `src/` computes **k̂** as a vector in Phase 1 — it is a documented convention
that fixes the sign of `u`, not a quantity `wavevector` or `spectrum` needs numerically, since
only the magnitude `k` and the scalar projection `ξ_s = (ω/k − u_s)/v_th,s` enter the physics.

## 7. The polarization factor

Sheffield Eq. 2.3.13's prefactor includes `|ŝ × (ŝ × Ê_i0)|²`, the squared component of the
incident electric field transverse to the scattered direction. Writing `ψ` for the angle of
`Ê_i0` from the normal to the scattering plane (the plane containing `k̂_i` and `ŝ`),

    |ŝ × (ŝ × Ê_i0)|² = 1 − sin²θ sin²ψ.

`ψ = 0` (the incident polarization normal to the scattering plane, the usual arrangement) gives
`1`; `ψ = π/2` (the incident polarization in the scattering plane) gives `cos²θ`. This is
`polarization_factor(geometry)`, computed for reference only: `spectrum` sets `G = 1` and never
multiplies it in, since the gain absorbs it (`conventions.md` 1.8, decided 2026-09-07).
