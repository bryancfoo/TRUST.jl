# The convective gain

> **Front matter (Step 2.3b, 2026-09-11).** Windows 11, Julia 1.12.7. Written alongside
> `src/scattering/spectrum.jl`'s `_convective_gain` and `src/instrument/effects.jl`'s
> `Sensitivity` response curve.

This document derives what `_convective_gain` implements: Turnbull et al.'s convective SRS/SBS
gain on the scattered light (D. Turnbull, A. L. Milder, R. K. Follett, J. Katz and D. H. Froula,
"Thomson Scattering with Gain", Phys. Rev. Lett. **136**, 135101 (2026); cited as Turnbull with
an equation number), in this package's symbols and normalization. `conventions.md` 1.8 and
[OQ-32](../../dev/open_questions.md#oq-32) record the decision; this document is the algebra.

## 1. What the gain is, and where it lives

Thermal Thomson scattering seeds a spectrum of density fluctuations; as the scattered light
crosses the rest of the probe beam it is further amplified (Stokes, red of the probe) or
depleted (anti-Stokes, blue) by the same convective instabilities — SRS on the electron feature,
SBS on the ion feature — that drive parametric backscatter. The effect is small (`|γL|` well
below one for the geometries this package's test problems use) but shifts inferred `n_e` by
about 15% and `T_e` by about 20% at OMEGA-like probe intensities (Turnbull Table I, Fig. 3), and
is the reported explanation for EPW Stokes/anti-Stokes asymmetries that a thermal fit alone
cannot reproduce.

**The gain is physics, computed from the plasma and the known probe, with no fitted parameter of
its own** — unlike an instrument effect, which has at least one calibration constant. It
therefore lives beside the relativistic prefactor and the wavelength Jacobian in `_spectrum_at`,
never inside `spectral_density` (which stays the pure structure factor S(k,ω) of
`conventions.md` 1.7 — Turnbull's own Eq. 1 defines S exactly that way, and the sum-rule test of
`conventions.md` 8.5 would not survive a gain folded into S) and never in the instrument chain
(spec D39).

## 2. Turnbull's Eqs. 6 to 8, in this package's symbols

Turnbull's convective gain rate is, verbatim (his Eq. 6, his `ω = ω_0 − ω_s`)

    γ_Turnbull = (k² / 4k_s) · Im(F_χ)_Turnbull · |ā₀|²,

and this package's, in its own `ω = ω_s − ω_i` (Section 6 derives the sign):

    γ = −(k² / 4k_s) · Im(F_χ) · |ā₀|²,

with `k` the probed wavenumber, `k_s` the scattered wavenumber (both `wavevector`'s return, spec
D41), `|ā₀|²` the probe's normalized vector potential squared projected onto the scattered
polarization (Section 3), and

    F_χ = χ_e(1 + Σ_i χ_i)/ε = χ_e(ε − χ_e)/ε,

exactly the susceptibility ratio `dielectric`'s return already carries: with `d = (epsilon =
ε, chi_e = χ_e)`, `F_χ = d.chi_e * (d.epsilon - d.chi_e) / d.epsilon` — no second `dielectric`
call and no extra species loop (`ε − χ_e = 1 + Σ_i χ_i` by `dielectric`'s own definition,
`conventions.md` 1.6). The gain acts over the length the scattered light spends inside the probe
beam,

    L = D₀ / sin θ,

with `D₀ = probe.diameter` and `θ = collection.theta` the scattering angle (Turnbull Fig. 1).
The gain factor on the intensity is, exactly (his Eq. 7),

    g = (e^{γL} − 1) / γ,

which this package writes with `L` absorbed and normalized so that `g → 1` (not `0`) as `γL → 0`
— the sensible convention for a *multiplicative* factor on `S`, since `spectrum`'s `G = 1` when
`gain = :none` and the two must agree in that limit:

    g = expm1(γL) / (γL),                      (`:convective`)

**never written as `(exp(γL) − 1)/γL`**: at small `γL` this loses every significant digit to
cancellation, exactly where the answer must be closest to `1`. Below `|γL| < GAIN_SERIES_CUTOFF =
1e-8`, `_convective_gain` uses the series `1 + γL/2` instead; CG3 checks the two branches agree
to `1e-14` at the crossover and that `g → 1` to `1e-15` as `γL → 0`.

Turnbull's **weak-gain** form (his Eq. 8, valid to `O(γL)` beyond linear) is

    g = exp(γL / 2),                            (`:weak`)

kept as a second option for comparison: `expm1(x)/x` and `exp(x/2)` agree only to `O(x²)`, which
at `x = 0.05` is about `2×10⁻⁴` — CG5 checks the two agree to `2×10⁻³` at `|γL| ≤ 0.05`, ten
times that bound, so the row tests the *form* rather than round-off.

## 3. |ā₀|² = a₀² · polarization_factor, exactly

Turnbull projects the probe's normalized vector potential onto the components parallel and
perpendicular to the scattering plane,

    |ā₀|² = |a_{0,p}|² cos²θ + |a_{0,s}|²,

where `p` and `s` are the components of the incident field in and normal to the scattering
plane. Writing `ψ = probe.polarization` for the angle of `Ê_i0` from the normal to the
scattering plane (`conventions.md` notation, rule 1.8), the two components are `|a_{0,s}|² =
a₀² cos²ψ` (the part of the field already normal to the plane) and `|a_{0,p}|² = a₀² sin²ψ` (the
part in the plane), so

    |ā₀|²/a₀² = sin²ψ cos²θ + cos²ψ
              = sin²ψ cos²θ + cos²ψ(sin²θ + cos²θ)
              = sin²ψ cos²θ + cos²ψ sin²θ + cos²ψ cos²θ
              = cos²θ(sin²ψ + cos²ψ) + cos²ψ sin²θ
              = cos²θ + cos²ψ sin²θ
              = 1 − sin²θ + cos²ψ sin²θ
              = 1 − sin²θ(1 − cos²ψ)
              = 1 − sin²θ sin²ψ,

which is exactly `polarization_factor(probe, collection)` (rule 1.8, Sheffield Eq. 2.3.13). So

    |ā₀|² = a0_sq · polarization_factor(probe, collection),

the identity `_convective_gain` uses directly — the function has existed since Phase 1 for
reference only (never multiplied into `spectrum`'s own result, since `Sensitivity` absorbs it)
and is, from this step on, **load-bearing** whenever `gain` is `:convective` or `:weak` (A21;
`conventions.md` 1.8 amended). CG2 checks the identity numerically, `1e-15` relative, against
`sin²ψ cos²θ + cos²ψ` written out independently in the test.

## 4. a₀²: the probe's normalized vector potential squared

The normalized vector potential of a plane electromagnetic wave of intensity `I` and wavelength
`λ` (linear polarization) is, in SI,

    a₀² = e² λ² I / (2π² ε₀ m_e² c⁵),

so `A0_SQ_COEFFICIENT = e²/(2π² ε₀ m_e² c⁵)` (`conventions.md` 1.3, `constants.jl`) is the
package's first derived constant, and `ProbeBeam` computes `a0_sq = A0_SQ_COEFFICIENT * lambda^2
* intensity` at construction (`intensity_wcm2`/`intensity_wm2`; `0` when not given).

**Numerical check** (GS5, `docs/derivations/geometry.md` Section 8): at `λ = 1 µm`, `I = 10¹⁸
W/cm²`, the package's exact formula gives `a₀² = 0.731`. The form usually quoted in the
literature, `a₀ = 0.85 √(I₁₈ λ_µm²)` (`I₁₈` in units of `10¹⁸` W/cm², `λ_µm` in µm), gives `a₀² =
0.85² = 0.7225` at the same point — agreement to the rounding of the quoted `0.85` (about 1%),
not to the package's own three-figure precision, which is exact.

## 5. The probe used for the test problems

**Decided by the user, 2026-09-11**, closing the two numbers [OQ-32](../../dev/open_questions.md#oq-32)
was waiting on: the quoted intensity for the OMEGA 4ω probe is **cycle-averaged**, not peak, and
the beam is **60 J/ns in a 100 µm diameter circle**:

    power      P  = 60 J/ns                            = 6.0000e10 W
    area       A  = π (50 µm)²                          = 7.853982e-9 m²
    intensity  I  = P / A                                = 7.63944e18 W/m²  = 7.63944e14 W/cm²
    a0_sq         = A0_SQ_COEFFICIENT · λ² · I           = 3.8698e-5         (λ = 263.25 nm)
    gain length L = D₀ / sin θ = 100 µm / sin 60°        = 1.15470e-4 m      = 115.47 µm

so `ProbeBeam(lambda_probe_nm = 263.25, intensity_wcm2 = 7.63944e14, diameter_um = 100)` is the
probe CG4 and CG6 use.

**This geometry's gain is much weaker than Turnbull's own near-backscatter case.** His 99 J shot
is 351 nm at 154.2° with 165 µm FWHM: `I = 9.26e14 W/cm²` (matching his Table I's quoted `9.2`,
which is the check that the intensity formula above is right), `a₀² = 8.34e-5`, `L = 379.8 µm`.
Against this package's OMEGA probe, the three factors that enter `γL` are `a₀²` **0.46×**, `L`
**0.30×** and `k²/(4k_s)` **0.37×** at equal `Im(F_χ)`, so this package's `γL` is roughly **5%**
of Turnbull's at the same plasma conditions. His measured EPW SRS gain reaches `±0.6`, which
puts this package's own test problems near `|γL| ≈ 0.03` — a few percent, an estimate rather than
a measurement, since `Im(F_χ)` genuinely differs between the two plasmas. **CG4's "a probe
intensity giving `|γL| ≈ 0.1`" therefore means deliberately raising the intensity above the
user's own number, to measure the sign cleanly** — the test says so in a comment rather than
quietly using a number the user did not give. **The three test problems of D24 keep `gain =
:none`**: foo2026's ground truth has no convective gain, so turning it on would move the measured
feature table EX1-EX12 assert against.

## 6. The sign: amplification on Stokes, and the minus sign this package's γ carries

Turnbull writes `ω = ω_0 − ω_s`, so his Stokes feature (amplified) is his `ω > 0`. This
package's convention is `ω = ω_s − ω_i`, positive for a **blue** shift (`conventions.md` 3.1) —
the opposite sign — so **Stokes (amplified) is this package's `ω < 0`, red of the probe**, and
anti-Stokes (depleted) is `ω > 0`, blue of the probe.

**A literal transcription of Turnbull's Eq. 6 using this package's `Im(F_χ)` gives the wrong
sign — verified against his own Eq. 2, not assumed.** Turnbull's susceptibility (his Eq. 2) is

    χ(k, ω) = ∫ dv  [4πq² n / (Mk²)] · k·∂f/∂v / (ω − k·v),

the same Landau-contour Vlasov susceptibility `conventions.md` 1.5/1.6 already fixes (the
contour, or equivalently `ω → ω + i0⁺`, passes below the pole at `v = ω/k`). Writing `1/(x+i0⁺)
= P(1/x) − iπδ(x)` and integrating the delta function, for real `k > 0` and real `ω`,

    Im χ(k, ω) = −π (4πq²n/(Mk²)) f'(ω/k).

For any **even** `f(v)` (a Maxwellian, a super-Gaussian, any isotropic shape at zero drift),
`f'` is **odd**, so `Im χ(k, ω)` is an **odd function of `ω`**: `Im χ(k, −ω) = −Im χ(k, ω)`. The
same holds for `F_χ = χ_e(ε−χ_e)/ε`, built entirely from susceptibilities of this form, at fixed
`k`. This is a property of the formula itself, not of which author's `ω` is plugged in — so
Turnbull's `Im(F_χ)` and this package's, evaluated **at the same physical point** (the same
scattered wavelength, hence the same `|ω|`, but opposite-signed `ω` labels), are related by

    Im(F_χ)_Turnbull(ω_Turnbull) = Im(F_χ)_ours(−ω_Turnbull) = −Im(F_χ)_ours(ω_ours),

since `ω_ours = −ω_Turnbull` at that point. At Stokes (`ω_Turnbull > 0`), Turnbull's stated
physics requires `γ_Turnbull > 0` (amplification), hence `Im(F_χ)_Turnbull > 0` there (his
`k²/4k_s` and `|ā₀|²` are manifestly positive). By the identity above, `Im(F_χ)_ours` at the
*same* physical (Stokes, red) point is then **negative** — confirmed numerically for the D22
plasma (`Im(F_χ) ≈ +0.13` a fraction of a nm blue of the probe, `≈ −0.03` to `−0.14` a few nm to
several nm red of it, the expected odd pattern; the large excursion within a fraction of an nm
of the probe is the ion-acoustic resonance, where `F_χ` itself is large). A **naive** `γ =
+(k²/4k_s)·Im(F_χ)_ours·|ā₀|²` therefore comes out **negative** at Stokes/red — depletion — the
opposite of Turnbull's own result.

**The fix is the explicit minus sign in Section 2's boxed formula for this package's `γ`.** With
it, `γ` is positive at Stokes/red (amplification) and negative at anti-Stokes/blue (depletion),
matching Turnbull's stated physics exactly. **This is not a change to the Landau contour or to
`cauchy_dfdv`/`susceptibility`**, both of which stay exactly as `conventions.md` 1.5/1.6 fix them
and both of which are independently tested (`Im χ_e(ω>0) > 0`, Step 1.1) — it is the sign that
appears when Turnbull's `ω`-labeled formula is re-expressed in this package's `ω`-labeling,
derived above from his own Eq. 2. **Deviation from spec D43's literal formula, approved by the
user, 2026-09-11**: D43 gives `γ` without this sign; `conventions.md` 3.1's `ω` convention is
the source of truth Sheffield and this package share, and Turnbull's formula is adjusted to it,
not the reverse. **CG4 is the test**, asserting `g > 1` at every wavelength red of the probe and
`g < 1` at every wavelength blue of it, and it now passes with the sign above.

## 7. The uniform-plasma caveat

Turnbull's Eq. 7 integrates the gain over the scattering volume assuming a **uniform** plasma
along the beam. He states the limitation himself: gain "complicates the analysis of situations
in which gradients cannot be neglected. The Stokes feature, which sees positive gain,
preferentially samples the back of the Thomson-scattering volume." So `gain = :convective`
together with `GradientSmearing` is **not** strictly self-consistent — the two effects describe
the same scattering volume with different weightings, the gain implicitly favouring the far side
of the volume along the beam while `GradientSmearing`'s Gauss–Hermite quadrature weights the
whole volume symmetrically. This package allows the combination and documents the
inconsistency; it does not attempt to correct it. **Steady state is also assumed**: SRS grows in
under a picosecond and SBS in a few picoseconds, both far shorter than a streak bin (tens of
picoseconds), and absolute instability (as opposed to the convective gain modelled here) is
assumed absent.
