# The scattering geometry and k(λ_s)

> **Front matter (Step 1.6, 2026-09-08; the split of Section 0 added Step 2.3a, 2026-09-11).**
> Windows 11, Julia 1.12.7; SpecialFunctions 2.9.0, QuadGK 2.11.3, ForwardDiff 1.4.5,
> ChainRulesCore 1.26.1. Measured on this machine, throughput (best mean of 5 passes of 2000
> calls, warmed up, measured inside a function) and `@allocated` after warm-up:
>
> | Call | throughput | `@allocated` (1e/1i) | `@allocated` (2e/2i) |
> |---|---|---|---|
> | `wavevector` | 12 ns | 0 | 0 |
>
> Spec D24's reference is 21 ns; mine is faster on this machine, ordinary run-to-run variation.
> Latency (one isolated call) is at this machine's `time_ns` resolution (~100 ns), too fast to
> resolve individually at 12 ns throughput.

This document derives what `src/scattering/geometry.jl` implements: the probed wavevector and
frequency shift from the light-wave dispersion relation, the exact and approximate forms of `k`,
the mean **k̂** convention (now a callable function, `kdirection`, not only a figure), the
polarization factor, and the detector axis from a dispersion.

## 0. The split: `ProbeBeam` and `Collection`

**Decided by the user, 2026-09-10 (spec D40), built at Step 2.3a.** The single geometry struct
Phase 1 built mixed two kinds of quantity: the probe's own properties (`lambda_probe`,
`polarization`, and now its
energetics for the convective gain of Step 2.3b — `a0_sq`, `diameter`) and the collection's
(`theta`, `k_model`, and now the detector's dispersion). The type is split accordingly:

- **`ProbeBeam`**: wavelength, polarization, the normalized vector potential squared `a0_sq`
  (derived from the probe intensity, Section 8 below), and the beam diameter. Nothing here
  depends on where the light is collected.
- **`Collection`**: the scattering angle `theta` (given directly, or as a direction pair
  `k_probe`/`k_scatter` reduced to an angle and dropped, since `conventions.md` 3.2 fixes
  **k̂** as one direction for the whole spectrum and nothing downstream consumes a direction),
  the `k` model, and optionally the detector dispersion (`lambda_center`, `dispersion`,
  `npixels`, `reference_pixel`) that `detector_axis` (Section 9) reads.

Every function that took the old single geometry struct now takes the `(probe, collection)` pair
in that order: `wavevector(probe, collection, lambda_s, plasma)`, `polarization_factor(probe,
collection)`, `kdirection(probe, collection)`. The acceptance test for the whole refactor is that
it changes no number: with the convective gain off (`gain = :none`, the default), `spectrum`,
`ideal_streak` and `measure` are bitwise what they were before the split (RF1).

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
3.1. Only the magnitude `k` and the scalar projection `ξ_s = (ω/k − u_s)/v_th,s` enter the
physics of `wavevector` or `spectrum` numerically — **k̂** as a vector is not needed there.

**Since Step 2.3a, `kdirection(probe, collection)` makes this figure a callable function**
(spec D41): the canonical frame fixes the probe along **+ẑ** and the scattered direction **ŝ**
in the x–z plane at angle `θ` from **+ẑ**, and

    k̂ = (k_s ŝ − k_i ẑ) / |k_s ŝ − k_i ẑ|      evaluated at λ_s = λ_probe (so k_s = k_i,
                                                 the magnitudes cancel, and kdirection needs no
                                                 plasma argument at all),

which is exactly the `(ŝ − k̂_i)/|ŝ − k̂_i|` above, now returned as an `NTuple{3}` so a user can
print the direction their flows are quoted along instead of reading the figure. GS3 checks it is
a unit vector with zero **ŷ** component (it lies in the x–z plane by construction) for
`θ ∈ {30°, 60°, 90°, 154.2°}`.

## 7. The polarization factor

Sheffield Eq. 2.3.13's prefactor includes `|ŝ × (ŝ × Ê_i0)|²`, the squared component of the
incident electric field transverse to the scattered direction. Writing `ψ` for the angle of
`Ê_i0` from the normal to the scattering plane (the plane containing `k̂_i` and `ŝ`),

    |ŝ × (ŝ × Ê_i0)|² = 1 − sin²θ sin²ψ.

`ψ = 0` (the incident polarization normal to the scattering plane, the usual arrangement) gives
`1`; `ψ = π/2` (the incident polarization in the scattering plane) gives `cos²θ`. This is
`polarization_factor(probe, collection)`, `ψ = probe.polarization` and `θ = collection.theta`
since the split of Step 2.3a. `spectrum` sets `G = 1` and never multiplies it into its own
result, since the sensitivity absorbs it (`conventions.md` 1.8, decided 2026-09-07) — but from
Step 2.3b it is **load-bearing** whenever the convective gain (`docs/derivations/gain.md`) is
on, since Turnbull's projected vector potential is `a0_sq * polarization_factor(probe,
collection)` exactly.

## 8. `a0_sq`: the probe's normalized vector potential squared

`ProbeBeam` derives `a0_sq` at construction from the probe intensity, in SI, with
`A0_SQ_COEFFICIENT = e²/(2π² ε₀ m_e² c⁵)` (`conventions.md` 1.3, the package's first derived
constant):

    a₀² = A0_SQ_COEFFICIENT · λ² · I          λ = probe.lambda (m), I in W/m² (linear polarization)

Checked against the standard form quoted in the literature, `a₀ = 0.85 √(I₁₈ λ_µm²)` (`I₁₈` in
units of 10¹⁸ W/cm², `λ_µm` in µm): at `λ = 1 µm`, `I = 10¹⁸ W/cm²` the package's formula gives
`0.731` against the standard's `0.7225` — agreement to the rounding of the quoted `0.85` (about
1 %, not the 3-figure precision the package's own exact formula carries; GS5). `a0_sq` is `0`
when `intensity_wcm2`/`intensity_wm2` is not given (the default), and the convective gain of
Step 2.3b (`docs/derivations/gain.md`) errors loudly if it is asked to run on a probe with no
intensity.

## 9. The detector axis from a dispersion

**Decided by the user, 2026-09-10 (spec D44):** "writing the lambda axis as a center and nm/px
value makes a lot of sense; those would be easy numbers to write into the input deck." A
`Collection` may optionally carry a linear dispersion — `lambda_center`, `dispersion` (signed, m
per pixel), `npixels`, `reference_pixel` (the 1-based pixel at which `lambda_center` sits) — and
`detector_axis(collection)` returns

    λ[p] = lambda_center + dispersion · (p − reference_pixel),      p = 1, …, npixels,

**always increasing**, regardless of the sign of `dispersion`. This matters because a real
detector's dispersion can be negative — OMEGA's IAW axis runs backwards, `dispersion_nm_px =
-0.00444` — and `conventions.md` 3.4 requires the wavelength axis increasing. So when
`dispersion < 0`, the formula above is evaluated and then the resulting array is **reversed**;
`read_streak`/the dewarp step of Step 2.4 reverse the image's wavelength dimension the same way
whenever the dispersion that generated the axis is negative, so that a feature lands at the same
*wavelength* on either sign, never the same *pixel* (AX3: build the axis both ways with
`reference_pixel` chosen so the two are element-for-element identical, and check a synthetic
feature's peak agrees to `1e-12` relative). Getting the sign wrong silently mirrors the whole
spectrum along the wavelength axis — the failure mode this section exists to rule out.

`Experiment` takes the wavelength axis as `lambda_nm`/`lambda_m` as before, or, when that is
omitted and `collection` carries a dispersion, falls back to `detector_axis(collection)`; giving
both and having them disagree is a loud error naming the first bin (spec D44), exactly the
two-sources-of-truth trap `conventions.md` 3.4 and the trace's own time axis (`Experiment`'s
`t_ns`/`t_s` against `trace.axis`) both close the same way.
