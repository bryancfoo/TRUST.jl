# The scattered power spectrum P(λ_s)

> **Front matter (Step 1.6, 2026-09-08).** Windows 11, Julia 1.12.7; SpecialFunctions 2.9.0,
> QuadGK 2.11.3, ForwardDiff 1.4.5, ChainRulesCore 1.26.1. Measured on this machine, latency
> (one isolated call, warmed up) and throughput (best mean of 5 passes of 2000 calls for
> `_spectrum_at`, 500 calls for `spectrum!`, both warmed up, measured inside a function per
> Phase 0 pitfall 15 / spec pitfall 14) and `@allocated` after warm-up:
>
> | Call | latency | throughput | `@allocated` |
> |---|---|---|---|
> | `_spectrum_at`, Maxwellian | ~100 ns (timer resolution) | 157 ns | 0 |
> | `spectrum!`, 512 bins, Maxwellian | 84.1 µs | 84.5 µs | 0 |
>
> Spec D24's reference numbers are 144 ns and 71 µs; both of mine run 9-19% slower, ordinary
> machine variation — the allocation counts, the assertion that actually matters, are zero in
> both cases, confirmed for a one-electron/one-ion **and** a two-electron/two-ion plasma
> (`test/scattering/test_susceptibility.jl` CH8, `test/scattering/test_spectral_density.jl` SD9,
> `test/scattering/test_geometry.jl` GE12, `test/scattering/test_spectrum.jl` SP11, spec PF2).
> A `Plasma{ForwardDiff.Dual}` and a `ProbeBeam`/`Collection{ForwardDiff.Dual}` pair go through the exact
> same `spectrum!` code as the `Float64` build (no method specializes on element type beyond
> `FT<:Real`), which is what the SP9, SP10 gradient tests and PF3 (the dual plasma's value part
> matching the `Float64` spectrum to `1e-14` relative) exercise. No cache was added and
> `src/distributions/` was not touched (Phase 1 adds no quadrature to `src/`, spec Section 7).

This document derives what `src/scattering/spectrum.jl` implements: the prefactor, the exact
wavelength Jacobian, and the three benchmarks against Sheffield's noncollective and collective
limiting cases.

## 1. Sheffield Eq. 2.3.13 and the first-order factor Eq. 2.3.17

The scattered power in solid angle `dΩ` and frequency range `dω_s` about `ω_s` is (Sheffield
Eq. 2.3.13, p. 41)

    P_s(R, ω_s) dΩ dω_s = (P_i r_0² / 2πA) dΩ dω_s |ŝ × (ŝ × Ê_i0)|² N S(k, ω),

derived to zeroth order in `v/c` (the electron treated as stationary except for the Doppler
shift). Keeping terms to **first order in v/c** (Sheffield Section 2.3.3) adds one multiplicative
factor, `(1 + 2ω/ω_i)` (Eq. 2.3.17, p. 42):

    P_s(R, ω_s) dΩ dω_s = (P_i r_0² / 2πA) dΩ dω_s (1 + 2ω/ω_i) |ŝ × (ŝ × Ê_i0)|² N S(k, ω).

Sheffield attributes this term to two effects that both scale the same way to first order: the
relativistic "headlight" aberration (radiation preferentially directed along the emitter's
instantaneous velocity) and the **v** × **B**_i force from the incident wave acting on the
already-moving electron. Both are `O(v/c)`, adequate for `T_e ≲ 5 keV` (Sheffield notes this
bound explicitly); the fully relativistic treatment (Sheffield Section 3.5) is Phase 4.

`conventions.md` 1.8 packages everything except `S(k, ω)` and the `(1+2ω/ω_i)` factor into a
single gain `G`, since only the *shape* of `P(λ_s)` matters for fitting:

    P(λ_s) = G · (1 + 2ω/ω_i) · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²,     G = 1 in `spectrum`.

`_prefactor(omega, omega_probe, first_order)` is `1 + 2ω/ω_i` when `first_order` is true (the
default, `prefactor = :first_order`) and `one(omega)` when false (`prefactor = :none`, which
recovers the zeroth-order Eq. 2.3.13 and is what the `±ω` symmetry test SP6 needs, since the
first-order term is itself odd in `ω` and breaks that symmetry by design).
`polarization_factor(probe, collection)` (`|ŝ × (ŝ × Ê_i0)|²`) is **not** multiplied in: it is
absorbed into `G` (`Sensitivity` since Step 2.3a), computed only for reference in `spectrum`
itself (`conventions.md` 1.8, geometry.md Section 7) — though since Step 2.3b it is load-bearing
through the convective gain, `docs/derivations/gain.md`.

## 2. The exact Jacobian, and why it belongs only in `spectrum`

`S(k, ω)` is a spectral density **per unit frequency** (`docs/derivations/spectral_density.md`
Section 4: seconds). The package's wavelength axis is `λ_s`, so converting `P` per unit
frequency to `P` per unit wavelength needs the Jacobian `|dω_s/dλ_s|`. Since `ω_s = 2πc/λ_s`
exactly (no approximation, unlike the constant-`k` treatment of Section 4.5),

    dω_s/dλ_s = −2πc/λ_s²,     |dω_s/dλ_s| = 2πc/λ_s² ≡ `_wavelength_jacobian(lambda_s)`.

This is applied **once**, inside `_spectrum_at`, and is its own function precisely so that a
test can divide it back out (`conventions.md` 5.7) — every benchmark below that compares against
a Sheffield formula written per unit *frequency* (Eq. 4.5.1, Eq. 5.4.10) does exactly that.
Nowhere else in the package computes a Jacobian: `wavevector` returns `ω` directly from the two
frequencies, never differentiated.

## 3. Eq. 4.5.3, and the origin of its −3.5

Sheffield's noncollective wavelength spectrum (Eq. 4.5.3, p. 80), for a Maxwellian plasma at
`α ≪ 1` (no collective shielding) and to first order in `Δλ/λ_i` and in `v/c`:

    P(λ_s) dλ_s ∝ [1 − 3.5(Δλ/λ_i) + c²Δλ³/(4a²λ_i³sin²(θ/2))] ·
                  exp[−c²Δλ²/(4a²λ_i²sin²(θ/2))] dλ_s,     Δλ = λ_s − λ_i, a = v_th,e.

The `−3.5` is not one effect but the sum of three, each contributing a term linear in `Δλ/λ_i`
to the leading bracket, from three independent places the wavelength dependence enters when
Eq. 4.5.1's frequency-domain form is rewritten in `Δλ`:

1. **The prefactor**, `1 + 2ω/ω_i`. Since `ω/ω_i = λ_i/λ_s − 1 ≈ −Δλ/λ_i` to first order, this
   contributes `1 − 2(Δλ/λ_i)`: **−2**.
2. **The amplitude's `1/k`.** The noncollective spectral density's normalization is
   `1/(π^{1/2} k a)`; at low density Sheffield expands (Eq. 4.5.2, p. 80)
   `1/k ≈ [c/(2^{1/2}(1−cosθ)^{1/2}ω_i)]·(1 + Δλ/(2λ_i))`, contributing `+(1/2)(Δλ/λ_i)`:
   **+1/2**.
3. **The Jacobian.** `dω_s ≈ (2πc/λ_i²)(1 − 2Δλ/λ_i) dλ_s` (expanding `1/λ_s² = 1/(λ_i+Δλ)²`
   to first order), contributing another `−2(Δλ/λ_i)`: **−2**.

Sum: `−2 + 1/2 − 2 = −3.5`, exactly Eq. 4.5.3's coefficient. `spectrum`'s own Jacobian is the
*exact* `2πc/λ_s²` (Section 2), not this linearization — the linearization is only Sheffield's,
used to get a closed form to compare against, and the comparison (SP4) is at conditions where
`α² < 1e-4` and `(Δλ/λ_i)² < 1e-4` (spec D21 row 12), where dropping second order costs `3.66e-4`
relative, matching the test's tolerance.

## 4. Eq. 4.5.4, the centre shift, and the book's own arithmetic

Differentiating Eq. 4.5.3 with respect to `Δλ` and setting the result to zero (Sheffield does
this in closed form, p. 80) gives the shift of the spectrum's centroid toward the blue:

    Δλ_m/λ_i = −7 (a/c)² sin²(θ/2)                                          (Eq. 4.5.4)

with the numerical shortcut `a²/c² ≈ 4×10⁻⁶ T_e(eV)`, giving the rounded

    Δλ_m/λ_i ≈ −2.8×10⁻⁵ T_e(eV) sin²(θ/2).                                 (Eq. 4.5.5)

Sheffield evaluates both for `θ = 83°`, `T_e = 117` eV, `λ_i = 6943` Å (694.3 nm) and quotes
`Δλ_m = −10.5` Å. **This is not what either printed equation gives with the package's CODATA
constants**: Eq. 4.5.4 gives **−9.77 Å** and the rounded Eq. 4.5.5 gives **−9.99 Å** at these
conditions — both close to each other (as expected, since 4.5.5 is 4.5.4 with `a²/c²` rounded to
one significant figure) but neither close to the book's `−10.5` Å. SP5 compares against
**Eq. 4.5.4 evaluated with our constants** (−9.77 Å), not the book's printed number, to `5%`
(measured `0.125%`, spec D21 row 13) — loose because the peak of a first-order-truncated
spectrum moves at second order in `Δλ/λ_i`, so Eq. 4.5.4 itself is already an approximation to
`spectrum`'s exact peak.

## 5. Eqs. 5.4.6 and 5.4.10: both approximate, and the 5.3.5-versus-5.4.10 factor

**Eq. 5.4.6** (the electron-plasma-wave feature separation, from substituting Eqs. 5.4.1–5.4.2
into the Bohm–Gross dispersion relation Eq. 5.4.5 and specializing to `θ = 90°`):

    Δλ_epw/λ_i ≈ 2 [n_e/n_cr + 3a²/c²]^{1/2} (1 + (3/2) n_e/n_cr),     a = v_th,e,

an approximation valid for `ω_i ≫ ω_pe` (Sheffield states the assumption explicitly). SP7 tests
it to `6%` (measured `3.93%`, spec D21 row 15) — loose because it keeps only the leading density
and thermal corrections to the exact dispersion relation `wavevector` uses.

**Eq. 5.4.10** (the blue/red peak-power ratio, for weak Landau damping, `α ≳ 2`, where the exact
`k`'s blue/red asymmetry, Eq. 5.4.9's "term A", is small): substituting the Bohm–Gross frequency
`ω_epw` for the peak location in the nonrelativistic prefactor ratio
`(1+2ω_epw/ω_i)/(1−2ω_epw/ω_i)`,

    P_blue/P_red ≈ 1 + 4 (ω_pe² + a²k²)^{1/2} / ω_i,     a = v_th,e, k at λ_s = λ_i.

**This printed form uses `a²k² = (2κT_e/m_e)k²`, not the Bohm–Gross `3κT_e k²/m_e` of Eq. 5.3.5**
— Sheffield's own p. 118 text writes `(2κT_e/m_e)k²` inside the square root, a factor of `2/3`
smaller than the exact plasma-wave dispersion's thermal term, not a transcription slip (it is
consistent with `a²` being literally `v_th,e²` as defined throughout the chapter). The package
uses the **printed** form (`a²k²`, i.e. `thermal_speed(electron)^2 * k^2`), and at the test
conditions of SP8 (spec D21 row 16) the difference between `2` and `3` inside the root changes
`P_blue/P_red` by far less than the test's `10%` tolerance — both approximations are already
subsumed by the accuracy Eq. 5.4.10 itself provides (measured `5.10%` at `α = 2.07`, spec D21
row 16). **This agreement holds only near `α ≈ 2`**: term A of Eq. 5.4.9 (the difference in
magnitude between the blue- and red-shifted resonant wavevectors under the *exact* `k`, absent
from Eq. 5.4.10's derivation, which implicitly assumes it is 1) grows with `α`, reaching a point
(`α ≈ 3.5`–`5` in the measured grid) where it drives the blue peak *below* the red one — the
opposite of what Eq. 5.4.10 predicts. `spectrum!` uses the exact `k` unconditionally
(`conventions.md` 3.3), so this growing disagreement at large `α` is expected and is not a defect
(pitfall 9).

**The ratio being compared is of powers per unit *frequency***, `P(λ)/`
`_wavelength_jacobian(λ)`, not of `P(λ)` peaks directly: the two scattered wavelengths (blue and
red features) have different `λ_s`, so their Jacobians differ by `(λ_red/λ_blue)² ≈ 1.37` at the
SP7/SP8 conditions, and comparing raw `P(λ)` peaks instead multiplies that factor in and fails
by 37% for a reason that has nothing to do with the physics (spec D21 row 16, pitfall 9).

**What happens to `P(λ_s)` after this point — the instrument chain, from `measure` — is
`docs/derivations/instrument.md` (Phase 2, Step 2.2).**
