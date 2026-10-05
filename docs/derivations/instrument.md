# The instrument chain

> **Front matter (Step 2.2, 2026-09-09/10; Section 6 added Step 2.3b, 2026-09-11).** Windows 11,
> Julia 1.12.7. This document's parts 1 to 4 are written at Step 2.2, alongside
> `src/instrument/effects.jl`, `src/instrument/background.jl` and
> `src/instrument/experiment.jl`. Part 5 (the noise model) is Step 2.5's; Part 6
> (`RationalBackground`) is Step 2.3b's; Part 7 (`PixelIntegration` and the resonance-width check)
> is Phase 4.F's Step 4.F.2 (2026-09-30); Part 8 (`TabulatedKSmearing`) is Phase 4.G's Step 4.G.1
> (2026-10-03).

## 1. The chain

`measure(experiment)` returns the measured streak from the ideal one in two groups, in a fixed
order ([OQ-13](../../dev/open_questions.md#oq-13)):

- **Group 1, inside the per-bin kernel**: effects that change what physics is evaluated rather
  than acting on the streak — `KSmearing`, `GradientSmearing`, `WavelengthCalibration`. Any
  number of these may be given; they run in the user's own relative order. `PixelIntegration`
  (Section 7) is one more, at most one per chain, and always runs last of them, whatever the
  order given. `TabulatedKSmearing` (Section 8) is another of the first kind, any number of them.
- **Group 2, after the per-bin loop, in this fixed order**: `SpectralIRF`, `TemporalIRF`,
  `Sensitivity`, then any number of backgrounds.

In one line, with `P_kernel` the ideal streak the per-bin kernel produces (group 1 already
folded in):

```
M = B + g · (K_λ · P_kernel · K_tᵀ)
```

with `K_λ` the spectral convolution matrix, `K_t` the temporal one, `g` the scalar sensitivity, and `B`
the sum of the background terms (each additive; `ConstantBackground` and `PolynomialBackground`
add a term independent of `P_kernel`, `ImageBackground` adds `scale · image`,
`RationalBackground` (Section 6) adds a rational function of the same scaled wavelength
`PolynomialBackground` uses). `Experiment`'s
constructor **stably sorts** the user's effects into this order (`_chain_order`), so the same
set of effects given in any order produces a bitwise identical streak (XP6).

## 2. The convolution operators

Both IRFs share one builder, `_convolution_matrix(kernel, axis)` (spec D13):

- **`GaussianKernel`.** `K[l, l'] = exp(-(a[l]-a[l'])²/(2σ²))`, `σ = FWHM/(2√(2 ln 2))`,
  computed once in `_sigma_from_fwhm` and nowhere else as a literal (Pitfall 5). `σ` is in the
  axis's own SI unit (m on `lambda`, s on `t`).
- **`TabulatedKernel`.** `K[l'+offset_k, l'] = weight_k` for each of its `N` taps, on a
  **uniform** axis (checked to `1e-10` relative spacing).
- **Column normalization, both kinds.** Every column of `K` is divided by its own sum, so the
  source bin `l'` distributes *all* of its power over whichever detector bins exist — no zero
  padding, no wraparound ([OQ-16](../../dev/open_questions.md#oq-16)). This conserves the
  streak's total exactly, `sum(K*P) == sum(P)` (EF7), at every column including the edges. It
  does **not** leave a flat field flat at the edges (EF6): a truncated kernel column, once
  renormalized, has a row-sum profile that only reaches one in the interior. That is a
  deliberate trade — power conservation over flat-field preservation — not an oversight.
- **Orientation.** Sheffield Eq. 7.2.1 writes the transmitted power at output `p` as an integral
  of the input over the source wavelength, `I_t(p) = ∫dλ I_i(λ) K(p,λ)`: the **output** index is
  the row, the **source** index is the column. In matrix form the spectral IRF is therefore a
  **left** multiplication, `M = K_λ·P`, and the temporal one a **right** multiplication by the
  transpose, `M = P·K_tᵀ` (`apply!`, spec D16) — a streak is wavelength-first (`N_λ x N_t`), so
  the temporal convolution acts along its second dimension.
- **The temporal IRF acts after the per-bin loop, never inside the kernel.** It couples
  *measured* bins, and putting it inside `_bin_kernel!` would couple the *ideal* ones instead,
  breaking the per-bin block structure Step 3.2's Jacobian depends on
  ([OQ-16](../../dev/open_questions.md#oq-16), Pitfall 2). Nothing in `_bin_kernel!` may see it.
- **Element type.** `promote_type` of the kernel's own type and the axis's — never `Float64`
  (Pitfall 3): a dual `σ` (from a fitted IRF width) gives a dual matrix, which is how that
  width's Jacobian column is obtained (XP12).
- **The axis itself, since Step 4.A.2 (2026-09-23).** `K_t` is built on `expt.t`, and since
  [OQ-34](../../dev/open_questions.md#oq-34) that axis is the user's own **fitted** axis: one or
  more `AveragingRegion`s, each a start, an end and a bin count, onto which `average_regions`
  (`src/data/average.jl`, beside `dewarp`) averages the detector's raw pixels, σ and mask in data
  preparation, before `Experiment` is ever built. A streak's 1024 detector time bins therefore
  become the tens of fitted bins `K_t` actually operates on. Nothing here changes because of it:
  `TemporalIRF` never sees a source pixel that was not already averaged onto the fitted axis, and
  the square, columns-sum-to-one matrix this section describes is unchanged in form and in code —
  [OQ-16](../../dev/open_questions.md#oq-16)'s "one plasma snapshot per detector time bin" survives
  in generalized form, the "detector time bin" now being the averaged one.

## 3. The in-kernel effects

All three are weighted sums over samples of the plasma, the probe and/or the collection, folded
by `_accumulate!` (spec D16):

```
P(λ) = Σ_q w_q · spectrum(p_q, probe_q, collection_q, λ)
```

- **`KSmearing`.** The collection aperture is modelled as a uniform spread of angles of
  half-width `half_angle`: `θ_q = θ + half_angle·ξ_q` on Gauss–Legendre nodes `ξ_q` on
  `[-1, 1]`. Changes `θ` and therefore `k` (Sheffield's dispersion relation), not `ω`.
- **`GradientSmearing`.** A normal distribution of one plasma leaf (a species' `T` or `u`, or
  `n_e`) inside the scattering volume: `x_q = x + width·√2·ξ_q` (additive) on Gauss–Hermite
  nodes `ξ_q`, or, for `n_e` only, `x_q = x·exp(width·√2·ξ_q)` (**multiplicative**, so the
  smeared density stays strictly positive at any width and any node — EF11 is the reason this
  form was chosen over an additive one).
- **`WavelengthCalibration`** (replaces the Phase 2/3 probe-wavelength-shift effect at Step
  4.A.1a, the user, 2026-09-21, [OQ-24](../../dev/open_questions.md#oq-24)). A scalar offset and
  a dimensionless dispersion scale perturb **the axis the kernel evaluates the spectrum at**, one
  pixel at a time:

  ```
  λ_eval,l = λ_l + offset + dispersion_scale · (λ_l − λ_ref),   λ_ref = (first(λ) + last(λ)) / 2
  ```

  with `λ` the evaluation axis the effect receives and `λ_ref` its midpoint, **always** — not a
  field, a keyword or a deck key. `λ_ref` is a pure reparameterization of `(offset,
  dispersion_scale)` (a shift of it is exactly absorbed by `offset`), so the only thing any choice
  of it changes is the conditioning of the two parameters against each other, and the window's own
  midpoint is the choice that decorrelates them (the two basis functions of the map, `1` and
  `λ − λ_ref`, are orthogonal over the window exactly there). A single sample of weight one, lazily
  evaluated (`_CalibratedAxis`, never materialized, so it costs no `N_λ`-element allocation).
  **The operators never move**: `SpectralIRF`, `TemporalIRF`, the `Sensitivity` response curve and
  every background's abscissa are all built from `expt.lambda` — the unperturbed detector axis —
  at `Experiment` construction, and stay there regardless of this effect (WC8); a spectrometer's
  response belongs on the detector's pixel axis, not on a perturbed evaluation point. The probe's
  own wavelength is never touched.

**Both quadrature rules use an odd node count** (`1, 3, 5, 7`) so the nominal (unsmeared) value
is always one of the nodes, and both normalize their weights to sum to one at construction, so
`nodes = 1` is the identity to round-off for either rule.

## 4. Linearity

`islinear(effect)` answers exactly the question Step 3.2's Jacobian assembly will ask: *may
this effect be pushed through as a fixed linear operator?*

| Effect | `islinear` | Why |
|---|---|---|
| `SpectralIRF`, `TemporalIRF`, `Sensitivity` | `true` | each is a fixed matrix or scalar multiplication of the streak |
| `ConstantBackground`, `PolynomialBackground`, `ImageBackground` | `false` | **affine**, not linear: `effect(aP+bQ) - a·effect(P) - b·effect(Q) = (1-a-b)·B`, tested directly (EF15, BG4) |
| `KSmearing`, `TabulatedKSmearing`, `GradientSmearing`, `WavelengthCalibration`, `PixelIntegration` | `false` | these act on the **plasma** or the **evaluation axis**, not the streak — `effect(aP+bQ) = a·effect(P)+b·effect(Q)` is not even a meaningful statement about them, since they change what is evaluated rather than post-processing a value that already exists |

Every effect with `islinear(e) == true` satisfies the linear identity to round-off (EF14); every
background satisfies the affine identity instead (EF15, BG4); the three kernel effects have
`stage(e) == :kernel` and `islinear(e) == false` by definition.

## 5. The noise model

Swadling et al. (2022, `context/swadling2022.pdf`) Eqs. 22-26, 37, 40, in this package's symbols
(spec D21-D23; [OQ-14](../../dev/open_questions.md#oq-14)).

**The variance identity, and why `synthesize` uses the `G²` form.** Swadling's Eqs. 33-35 give
the CCD signal's total variance as the sum of the photon-counting (shot) term and the gain
(excess noise factor) term:

    σ²_total = S̄·G·F²  (Eq. 37, with n = 1 photoelectron avalanches per read),

decomposing into a Poisson term in photoelectrons, `G²·N_e` (`N_e = S̄/G`, so this is `G·S̄`),
plus a gain-process term. Differentiating the gain relation `N_C = G·N_e` while `N_e` is fixed
and `G` fluctuates gives `σ²_{N_C,gain} = (dN_C/dG)²·σ_G² = N_e²·G³(F²-1)/S̄ = S̄·G·(F²-1)` (using
`σ_G²/G² = (F²-1)/N_e`, the definition of the noise factor), which is `N_e·G²·(F²-1)`. Adding
the Poisson term `G²·N_e` gives `G²·N_e·F²  =  S̄·G·F²`, exactly Eq. 37. **This is the `G²` form
of step 4** (`N_C = G·N_e + G·√((F²-1)·N_e)·randn()`), and it is the one that reproduces Eq. 37;
Swadling's own Eq. 24 as printed in the PDF, `σ_{N_C} = √((F²-1)(N_e·G))` (a single power of
`G`), does not. `docs/derivations/instrument.md`'s job is to say this once so nobody "corrects"
step 4 back to the printed form; NZ2 is the test that would catch a `G` instead of `G²`.

**Two departures from Swadling's own simulation, both deliberate.** The counts are **not**
rounded to integers — a continuous image is what χ² and forward-mode AD need, and quantization
is not modelled. The point spread (`_point_spread`, step 5) uses the same column-normalized
`_convolution_matrix` the IRFs use (spec D13), which **conserves counts exactly**
(`sum(_point_spread(img)) == sum(img)`, NZ5); Swadling's own kernel is normalized over the full
window and loses flux at the edges instead. The two differ only in the outermost `~8σ_px`
pixels.

**σ is the realized per-pixel σ, from the expected signal** (Phase 4.B, OQ-40 option A, spec D8).
The point spread (step 5) is linear, so a pixel's variance after it is its pre-spread variance
propagated through the squared weights: with `V = S̄·G·F²` the pre-spread variance of independent
pixels (Eq. 37) and `K_λ`, `K_t` the column-normalized Gaussian matrices of `_point_spread`,

    σ² = (K_λ ∘ K_λ) · V · (K_t ∘ K_t)ᵀ + σ_RN²,

`∘` the elementwise square and the read noise added after the spread, as step 6 adds it. At
`sigma_px = 0` this is Eq. 40's σ bitwise. It is written against the **expected** signal,
`measure(expt)` (or an externally supplied expectation): passing a *realised*, noisy streak gives
that realisation's own σ, not the χ² weight. It is **diagonal-approximate**: the spread also
correlates neighbouring pixels (Eq. 38), which is not modelled, exactly as the dewarp's σ is
diagonal-approximate. **Consequence for averaging**: `average_regions` combines σ as for
independent slices, so an average of n ≫ σ_px slices has its variance understated by up to
1/Σ_b w_b² = 2√π σ_px (≈ 3.54 at `sigma_px = 1`; measured 3.36 at 32 slices per bin). Averaged
fits of synthetic data therefore read χ²/n ≈ 3, not 1; the user decided on 2026-09-26 that this
is reported and fixed later.

**The mean's blur is not modelled** (the user, 2026-09-26: "not a big deal in the long run"). The
data's mean is `E[image] = K_λ S̄ K_tᵀ`, and the forward model has no point spread, so a fit sees a
mean blurred by one pixel in both directions. Its size against σ on the three example problems
and the two-channel problem, measured once (Phase 4.B, chunk 3's report), is: `epw_ablation`
(512 × 256) max 3.049 at pixel (511, 2), RMS 0.138; `iaw_multiflow` (512 × 256) max 1.165 at
pixel (277, 256), RMS 0.052; `iaw_shock` (512 × 256) max 4.831 at pixel (271, 72), RMS 0.162;
the two-channel problem's `epw` channel (256 × 1024) max 2.812 at pixel (256, 1), RMS 0.158; its
`iaw` channel (256 × 960) max 1.008 at pixel (120, 959), RMS 0.046. A deck whose spectral and
temporal IRFs span many pixels hides it; a small problem with no
spectral IRF shows it (Phase 4.A: χ²/n 7.2 and a +200 eV `T_D` bias on 24 wavelengths). The
fast fixtures therefore use `GaussianNoise`.

**Its effect on a fit, measured once** (Phase 4.D spec D18): on the two-channel shapes problem the
blur raises `GeneralizedNormal`'s EPW χ²/n in the bins around the profile's peak: 9.2203 with the
fit's one outlying bin left out, against 3.2335 for `Kappa` over all its bins. With the spread
switched off at synthesis the same two read 1.0418 and 0.9956 (`generalized_normal.md` Section 7).

**Reproducibility.** `synthesize` draws every pixel in fixed **column-major** order (wavelength
fastest, then time) on a single thread, so one `rng` seed gives the same image regardless of
`Threads.nthreads()` (NZ7). Threading `synthesize` would change the random stream, not the wall
time that matters for a one-off synthetic-data generation.

**Sheffield Section 7.3.3** (book p. 193): a streak camera's own temporal dispersion is of order
100 ps, the sanity check behind the `fwhm_ns` used in the three test problems (Step 2.5, spec
D24): `0.05` to `0.08` ns, comfortably in that range.

## 6. `RationalBackground`: a fourth background family

**Decided by the user, 2026-09-11 (spec D46), closing A26.** `rat11` (linear/linear) and `rat21`
(quadratic/linear), on the same scaled variable `s = (λ-λ₀)/Δλ` `PolynomialBackground` uses:

    B[l] = (Σ_{k=0}^{P-1} numerator[k+1]·s_l^k) / (1 + Σ_{k=1}^{Q} denominator[k]·s_l^k)

**The denominator's constant term is fixed at `1`, not its leading coefficient** — unlike
MATLAB's `rat11`/`rat21`, which are monic in the numerator's leading term instead:
`(p1·s+p2)/(s+q1) ≡ (a0+a1·s)/(1+b1·s)` with `a0 = p2/q1`, `a1 = p1/q1`, `b1 = 1/q1`, valid
whenever `q1 ≠ 0` (RB5 checks this conversion, `1e-14` relative — it is an algebraic identity).
Fixing `b_0 = 1` does two things: it nests `PolynomialBackground` exactly (`denominator = ()`
gives `1 + (empty sum) = 1` in the denominator, so `B` reduces to `PolynomialBackground`'s own
formula bitwise — RB2, the row that makes the rational family a strict generalization of the
polynomial one rather than a parallel family), and it is well conditioned at `s = 0`, the middle
of the detector window, which is where the fitted feature usually sits.

**The pole.** A rational background has a failure mode the other three do not: the denominator
can vanish inside the window. `_prepare` (the first point that knows the axis) evaluates `1 +
Σ b_k s^k` over the whole detector window and throws if it changes sign or if its minimum
modulus falls below `1e-3` (RB4) — deliberately loose: the guard is not trying to locate the pole
exactly, it is refusing a background that is already numerically ill-conditioned over the data.
A fit that walks `denominator` into a pole would trip this guard on every rebuild; Step 3.1 will
need bounds or a transform on `denominator` to fit one safely — noted here, not built.

Like the other three backgrounds, `stage = :detector`, `_chain_order = 5`, `islinear = false`
(RB6 checks the affine identity, as BG4 does for the others); `parameters(e) = (numerator =
e.numerator, denominator = e.denominator)`, both tuple-valued, which is why `_effect_eltype`
(spec D16) flattens a tuple-valued parameter generically rather than special-casing
`PolynomialBackground`'s `coefficients` by name (spec D46).

## 7. `PixelIntegration` and the resonance-width check

**Decided by the user, 2026-09-30** ([OQ-53](../../dev/open_questions.md#oq-53); Phase 4.F spec
D10 to D16, Step 4.F.2). **Why.** The per-bin kernel evaluates the spectrum at each pixel's centre
wavelength, and `SpectralIRF` then convolves those samples; a detector integrates the blurred
spectrum over each pixel instead. The two agree while the spectrum varies slowly across a pixel.
An electron-plasma-wave resonance can be far narrower than one: its width follows the Landau
damping at the resonance's `ξ`, which falls steeply with `m` for a flat-topped shape and, for any
shape, exponentially with `ξ`, that is with `α`. Phase 4.E found a `GeneralizedNormal` resonance
0.198 of a pixel wide (bin 25 of the two-channel shapes fit, `m` 5.4658), whose true peak is 16.7
times the largest value any pixel centre sees, and an objective along `m` with spurious minima, one
of which the fit sat in (`generalized_normal.md` Section 8). The synthetic data are made by the
same sampled model, so a synthetic fit cannot see the error as a misfit; real data can.

**The sampled integral.** `PixelIntegration(subsamples = N)` replaces the pixel-centre value by the
mean of `N` equally spaced samples across the pixel:

    S̄_l = (1/N) Σ_{i=1}^{N} S(λ_l + (i − (N+1)/2) · Δ_l / N),    Δ_l = (λ_{l+1} − λ_{l−1}) / 2,

one-sided at the two ends of the axis (`Δ_1 = λ_2 − λ_1`, `Δ_n = λ_n − λ_{n−1}`). On a uniform axis
`Δ_l` is the pitch and the points are the midpoints of `N` equal sub-pixels: the midpoint rule for
the pixel's mean (a box), equal weights. On a non-uniform axis the pixel is taken symmetric about
its centre with the centred spacing as its width, exact for a uniform axis and first-order
otherwise. `SpectralIRF` then convolves `S̄`, so the order of operations is the detector's: blur,
then integrate, with the integral applied to the unblurred spectrum, which the linearity of the
convolution allows.

**Where it sits.** It is an in-kernel effect (`stage = :kernel`, `_chain_order = 1`), at most one per
chain, with no fittable parameter, and `Experiment` always sorts it **after** every other in-kernel
effect (`_kernel_rank`), so that every node of `KSmearing` and `GradientSmearing` is sub-sampled and
the axis it sub-samples is `WavelengthCalibration`'s perturbed one, with that axis's own spacing
`(1 + dispersion_scale) · Δ`. The kernel folds it with its own method of `_accumulate!`
(`src/instrument/measure.jl`), the same samples, weights and order as `samples(::PixelIntegration)`
(PI3 checks them bitwise against each other): through `samples`' lazy axes, the per-bin dual pass
allocated 14 400 bytes a bin with every in-kernel parameter bound (Level 2, 2026-09-30), and through
the method it allocates none. **`N = 1` is today's model exactly**: the method returns the
pixel-centre fold itself, and a deck that does not name the key builds no `PixelIntegration` at all.

**How finely.** Measured on Phase 4.E's bin 25 (one bin, the two-channel EPW window of 256 pixels,
`SpectralIRF` 0.5 nm and `Sensitivity`; the largest difference from `N = 64` over the largest value
of the `N = 64` column; PI5):

| `N` | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| error | 0.692 | 0.103 | 0.198 | 0.0118 | 7.7e-4 |

**Not monotonic**: at `N = 4` the sub-sample spacing, 0.25 of a pixel, is still wider than the
0.198-pixel line, which is aliased again. Once the spacing is half the line's width or less the
error falls as the midpoint rule's does, so the count the check recommends (below) is the smallest
power of two at or above `2 / w`, `w` the width in pixels: 16 here. The cost is about `N` times the
forward model and the Jacobian (Phase 4.F spec D18).

**The resonance-width check** (`src/instrument/resonance.jl`, `fit(deck)`). After its other warnings,
`fit(deck)` evaluates, at every bin's fitted plasma and for every channel, the width of each EPW
resonance in the channel's window:

- **Where.** `Re ε(k(λ_l), ω(λ_l))` on the pixel axis; a resonance lies between two pixels where it
  changes sign **and** `|ω| > ω_pe` at both. The Bohm–Gross branch (`ω² ≈ ω_pe² + 3k²⟨v²⟩`) is always
  above `ω_pe`, which drops the strongly damped crossing nearer the probe and every ion-acoustic
  crossing (Phase 4.E's bin 25 has two sign changes, at pixels 145 and 187; the second lies below
  `ω_pe`).
- **How wide.** The spectrum on a grid of 64 points per pixel over 8 pixels either side; its maximum;
  the half-maximum crossing on each side, interpolated linearly between grid points; the full width
  between them in pixels of the channel's axis. Against a grid sixteen times finer it agrees to
  0.03 % (RW1). A side that never falls to half the maximum inside the grid gives an infinite width.
  A line narrower than the grid's step reads at about one step, 0.016 pixel: the width is then an
  upper bound, and the count recommended for it a lower bound.
- **When it warns.** A width below `RESONANCE_WIDTH_WARN = 1.0` pixel, in a channel whose count is
  below the recommended one; one warning naming each channel's bins, widths and the key to refit
  with, `pixel_integration = { subsamples = N }`; silent otherwise. It changes nothing.
- **What it ignores.** The in-kernel effects: every `KSmearing` or `GradientSmearing` node's line is
  as narrow as the unsmeared one and is sampled the same way, and the calibration's offset moves a
  line, not its width.
- **What one pixel leaves.** A line just over a pixel wide is silent, yet the pixel-centre model
  misses it after the spectral IRF by 8 % of the peak at 1.05 pixels and by about 1 % from 1.5 pixels
  on (Phase 4.F spec D14): 19σ and 2 to 3σ at the shapes problem's peak signal-to-noise of 221. The
  threshold is the user's decision, "about one pixel", and is recorded with those numbers beside it.

**Measured by hand** (Step 4.F.3, `examples/diagnostics/pixel_integration_refit.jl`). On the
two-channel `GeneralizedNormal` fit the check flags one bin and asks for 16; refitted at 16 that bin's m
is 4.5309 against the truth's 4.6111 and the warning is silent (`generalized_normal.md` Section 8). On
Phase 4.D's cut-down EPW reproduction (`SuperGaussian`, 128 × 32 pixels and slices, its fit's m on the
upper bound 5.5 in 17 of 32 bins) the truth's lines are 4.6 pixels wide or more, but the fit's are not:
the check flags 15 of the 17 bins, 0.038 to 0.270 of a pixel, and asks for 64. Refitted at 64 from the
deck's start, m is still on its bound in 17 bins (EPW χ²/n 65.9212, against 76.5608 from the same start
at one point a pixel), and ten of those bins' lines now read 0.016 to 0.024 of a pixel: **the count did
not cure that fit's saturation**; the fit moved to lines narrower than the new sampling resolves. A
width at the grid's step, 1/64 of a pixel, is an upper bound on the line's width, and the count asked
for a lower bound.

## 8. `TabulatedKSmearing`: an aperture given as rays

**Decided 2026-10-03** ([OQ-56](../../dev/open_questions.md#oq-56); Phase 4.G spec D3 to D8, Step
4.G.1). **Why.** `KSmearing` (Section 3) models the collection aperture as a uniform spread of
angles on Gauss–Legendre nodes. An optics model gives a list of rays with weights instead. Shot
116349's collection, in the user's model, is ten rays in the scattering plane at 53.64 to 66.12 deg
from the probe, 1.38684 deg apart, with bell-shaped weights: weighted mean 60.01 deg, standard
deviation 2.56 deg, and |k| over the rays 0.902 to 1.091 of its weighted mean (3.9 % rms). A uniform
spread can match the variance (half-angle √3 × 2.56 = 4.44 deg) but not the shape.

**The average.** With `δ_q` the offsets (rad) from the `Collection`'s own `θ`, strictly increasing,
and `w_q` the weights, non-negative and summing to one,

    P(λ) = Σ_q w_q · spectrum(p, probe, collection(θ + δ_q), λ),

`KSmearing`'s fold exactly, on the user's nodes. Each node changes `θ`, and with it the magnitude
`k(λ_s)` (`conventions.md` 3.3); the direction **k̂** stays the `Collection`'s for every node
(`conventions.md` 3.2), so a species' flow is its projection on that one direction and is not
re-projected on each ray's own. The error is 1 − cos of a ray's angle from the mean **k̂**: for shot
116349's rays 3.24 deg at worst, 0.16 % of the flow. The offsets are relative so that the
`Collection` stays the one source of the geometry (the prefactor, `kdirection` and the polarization
factor read its `θ`), and a single node at offset zero is the unsmeared model exactly.

**The weights** are divided by their sum, **unless that sum is already one to within `N` ulps**
(`|Σ w − 1| ≤ N·eps`), in which case they are kept exactly as given. `KSmearing`'s own five weights,
normalized at its construction, sum to `1 + 2.2e-16`, and dividing them again would move them by an
ulp; kept, an effect built from a `KSmearing`'s `offsets` and `weights` gives that `KSmearing`'s streak
bitwise (TK3). The user's ten weights sum to `1 − 6.7e-16` and are kept as given too.

**Where it sits.** An in-kernel effect (`stage = :kernel`, `_chain_order = 1`) with no fittable
parameter (`parameters` empty, `reconstruct` the identity), any number per chain (`_chain_class =
:none`). Beside a `KSmearing`, or a second `TabulatedKSmearing`, the fold runs over every pair of
nodes, `θ + δ_k + δ_t`, as two `KSmearing`s compose. Its samples rebuild only the `Collection`,
through its inner constructor, as `KSmearing`'s do: allocation-free in the value pass and in the dual
pass, the latter measured with the ten rays in test_jacobian.jl's fully bound KB chain, with and
without a `PixelIntegration` after them (TK5, TK9). The cost is proportional to the node count.

**Measured** on the two-channel example's probe and windows with the `Collection` at 60.01 deg (four
bins of its truth, a Maxwellian electron; `SpectralIRF` 0.5 nm on the EPW window, 0.08 nm on the IAW;
the largest difference over the largest value of the reference streak, TK6):

| window | ten rays against the uniform 4.44 deg on 5 nodes | ten rays against no smearing |
|---|---|---|
| EPW, 220 to 260 nm | 9.4e-4 | 3.1e-2 |
| IAW, 262 to 264.5 nm | 5.9e-5 | 4.3e-3 |

The shape of the spread changes the spectrum by about 3 % of what the spread itself does: the
equal-variance uniform spread captures most of the smearing, and the rays capture the rest.
**The cost** is about the node count: on the EPW window cut to 128 pixels, 32 bins, the ten rays' forward
model and Jacobian take 2.18 and 1.96 times the five-node `KSmearing`'s (warmed medians, 8.06 against
3.70 ms and 31.8 against 16.2 ms; Phase 4.G spec D10).

**Not modelled.** A flow direction per ray (above); a fittable scale on the offsets; absolute angles;
a ray list on the `Collection`; reading the rays from a file (the deck quotes them).
