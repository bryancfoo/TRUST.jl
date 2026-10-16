# OTIS.jl

OTIS, the **O**ptical **T**homson **I**nversion **S**uite, is a Julia package for forward
modelling and inverting optical Thomson scattering (OTS) spectra from plasmas, including
streaked (time-resolved) data inverted by regularizing across the whole streak at once. Its
central design choice is that the velocity distribution of each species is an **input, not an
assumption**: every species carries a dimensionless shape, and the scattering theory is written
once against the interface that shape provides, so non-Maxwellian distributions (super-Gaussian,
kappa, and others) are first-class rather than fitted away.

## Status

Phase 0 (CODATA constants and boundary units, the plasma dispersion function `Z` with derivative
rules, the velocity-shape interface with the Maxwellian, and `Species`, `Plasma` and the derived
plasma quantities) and Phase 1 (the static Maxwellian forward model: susceptibilities and the
dielectric function, the spectral density `S(k, ω)`, the scattering geometry with the exact
`k(λ_s)`, and the scattered power spectrum `P(λ_s)`) are complete.
Phase 1.SG is complete too: the `SuperGaussian` shape — the one-dimensional marginal of an
isotropic three-dimensional super-Gaussian, not `exp(-|x|^m)` — with closed-form moments, a
derivative rule for the incomplete gamma function, and its Cauchy integral served from a
Chebyshev table in (ξ, m) built when the package precompiles, so that a super-Gaussian spectrum
costs about what a Maxwellian one does and is differentiable in the exponent `m`. Its own
derivative in `m` is served the same way, from a second table (Phase 3.C) rather than the
adaptive quadrature that once dominated the cost of fitting it — a shape-agnostic cost guard in
the fast suite checks every registered shape's own parameter derivative against the same bar.

Phase 2 is complete: time and the instrument. A `PlasmaTrace` carries a template plasma, a named
time axis and a table of per-bin values for the leaves declared time-varying, and materializes
the plasma at any bin without allocating; an `Experiment` adds the detector axes and a chain of
instrument effects — aperture and gradient smearing and a probe-wavelength offset inside the
per-bin kernel, then spectral and temporal instrument functions, gain and backgrounds after it —
and `measure` returns the streak on the detector axes, evaluated across all available threads
with per-task caches and bitwise identical to the serial result. Measured streaks are read and
written as JLD2 files of plain arrays in a documented layout, and the streak-camera noise model
of Swadling et al. (2022) supplies both the χ² weights and the synthetic-data generator. The
three test problems of the reference paper — an EPW ablation problem with a time-varying
super-Gaussian exponent, and two ion-acoustic problems, one with interpenetrating flows and one
with a shock — are in `examples/`, and plotting methods load with Makie.
**[`examples/figures/`](examples/figures/) shows what the package produces**: the three test
problems' streaks and parameter traces, what each instrument effect does to one spectrum, and
the noise model against its own σ.

Phase 3 is complete: the inversion. Any numeric leaf of the model — a plasma parameter at every
time bin, a shape exponent, an instrument width, a background level — is fitted by declaring a
**bounded variable** and writing its name where the deck expects a number; quantities that span
decades, sums that must close, orderings and functional forms in time are written as
**dependent expressions** in ordinary Julia syntax, parsed and interpreted rather than
evaluated, and a **named time interval** lets a parameter take different values either side of
a shock. Every fitted variable passes through one logit onto its bounds, so the optimizer works
in unconstrained, order-one coordinates and no iterate can leave its range. The Jacobian is
assembled per time bin by forward-mode automatic differentiation and never formed densely — on
a 512 × 256 streak that is a few megabytes of blocks instead of the eight hundred a dense matrix
would be — and L-BFGS, Levenberg–Marquardt and a noisy gradient-descent explorer all consume the
same two matrix-free products. A **Tikhonov regularizer** composable along the time axis
penalizes the second time derivative of whichever parameters are named, in the axis's own unit
so that a rebinned dataset fits the same way, with per-interval terms that leave a
discontinuity unpenalized; **a scan over the weight reports the trade-off and chooses nothing**.
`fit("deck.toml")` runs the whole of it and writes a result file from which the fit can be
reproduced.

The three foo2026 reproductions (Step 3.7) are tested cut down to a 128 × 32 grid
(`test/slow/test_reproduce.jl`, run by hand); full resolution (512 × 256) is `examples/`.
**Phase 3.C (convergence, and the shape-parameter derivative) is complete, and all three
full-resolution reproductions now converge.** Phase 3 closed with every realistically-sized fit
ending at its iteration cap rather than at a real minimum — not because the physics or the
derivatives were wrong, but because the convergence criterion was an absolute threshold on the
gradient that meant a different thing on every problem's scale, a false convergence could be
reported when the logit's own scaling made the gradient look small near a bound, and L-BFGS's
starting step was badly scaled on the worst-conditioned problem. A convergence criterion stated
per pixel, a saturation guard that reports a stalled fit as `:saturated` rather than as a false
convergence, and a diagonal Gauss–Newton preconditioner (built from the same Jacobian blocks
every fit already assembles) together close that gap: at 512 × 256, the IAW shock, EPW ablation
and IAW multi-flow problems each fit unregularized and across their full seven-point weight
scans (plus the paper's own chosen weight where it is not already one of the seven), every fit
converging by the per-pixel criterion, end to end in roughly 18, 38 and 94 minutes respectively
at eight threads. The paper's own two-phase `noisy_descent_lbfgs` procedure, tried
beside plain `lbfgs` on the multi-flow problem, reaches a substantially better minimum in less
wall time than `lbfgs` alone — the shipped per-deck default stays `lbfgs` pending a wider
comparison, but the difference is large enough to be worth a future look. The cut-down
(128 × 32) reproductions in `test/slow/test_reproduce.jl` converge under the same criterion —
their scan's monotonicity check now holds cleanly — but at that resolution two of their
regularization-quality checks still do not: the EPW and multi-flow problems' RSS against ground
truth is not reliably improved by the paper's own chosen weight on a grid this coarse, unlike at
full resolution, where both improve substantially. The shock problem's own `u_pre` total
variation, over a nine-bin pre-shock interval, is likewise not smoothed by regularization at
128 × 32; at 512 × 256 the same variable's total variation drops by an order of magnitude under
regularization, consistent with a short-interval noise-floor effect rather than a solver defect.
Step 3.8 adds `WarpMap` and `dewarp`, the OMEGA streak's geometric correction, built and tested
against a synthetic warp; no real OMEGA streak was available this chunk, so a first fit of one
has not been run.

**Phase 4.A (multi-channel fits and uncertainty) is complete.** A fit can be driven by a
multi-channel deck (`channels`/`[channel.<name>]`), joining several streaks that share one plasma
into one `JointFitProblem`, each channel's own data averaged first onto a shared, user-chosen set
of time bins (`[[fit.regions]]`, `average_regions`) rather than the model resampled onto the
detector's own axis. A per-channel `WavelengthCalibration` effect — a scalar wavelength offset
and an optional dispersion scale, acting on the axis the kernel evaluates the spectrum at, never
on the probe or the IRFs — replaces the old `ProbeOffset`, since the dominant experimental
uncertainty in practice is the measured wavelength axis, not the probe beam.
`examples/two_channel.jl` and `examples/fit_two_channel.jl` synthesize a two-camera EPW/IAW
streak pair from one plasma — an EPW camera that sees density, temperature and the
super-Gaussian exponent but is blind to the ion temperature, and an IAW camera the other way
round — and fit it jointly and single-channel; `examples/figures/two_channel_fit.png` is the
committed figure showing each channel pinning what the other barely sees.

Three uncertainty reports sit beside a fit. A **calibration scan** repeats the regularized fit
with a channel's calibration offset fixed at a stated set of shifts, warm-started, and reports
every fitted variable's movement per bin — the systematic uncertainty, propagated by hand and
left for the user to read, not marginalized. A **Laplace covariance** at the minimizer, from the
Gauss–Newton Hessian assembled without ever forming a dense Jacobian, is reported in both a
posterior form (the credible width under the fit's own smoothness prior) and a sampling form
(comparable to a bootstrap over noise, since it uses only the data rows' own Fisher information);
every report first checks it is actually at a minimum and refuses, naming the shortfall, if it is
not. A **per-slice scatter** is the statistical uncertainty: every camera is averaged onto a
shared grid of windows nested finely inside each fitted bin, each fine window fitted on its own
as an unregularized joint problem with the instrument held fixed, and the detrended spread of
those single-slice estimates across a bin — never divided by the count, which would answer a
different question — is reported beside the regularized value, together with every window's own
convergence status. All three write to, and read back from, the result file; `examples/
uncertainty_two_channel.jl` runs the full-resolution two-channel problem through every one of
them, by hand. The roadmap is kept outside this tree.

**Phase 4.B (shapes) is complete.** Three more velocity shapes join the Maxwellian and the projected
super-Gaussian, each with its own Cauchy integral served from a Chebyshev table built when the
package precompiles, so each costs about what a Maxwellian does and is differentiable in its own
parameter: `Kappa`, the one-dimensional marginal of the isotropic kappa distribution, whose fitted
κ is the literature's; `GeneralizedNormal`, the plain `exp(-|x|^m)` of other fitting codes (not the
projected `SuperGaussian`); and `SkewNormal`, Azzalini's skew-normal, skewed along the scattering
vector. Outside each table a documented quadrature runs, correct and slower. Each table is checked
against an independent method — `Kappa`'s against a closed form by residues at integer κ, the
other two against a high-precision quadrature — and at its Maxwellian edge against `Z′` itself.
`examples/fit_shapes.jl` fits each shape, time-varying, from a deck on the two-channel synthetic
problem with the three uncertainty reports beside it (median relative error of the fitted shape
parameter against truth: 0.54% for κ, 0.65% for m, and a median absolute error of 0.012 for α);
`examples/figures/shapes.png` shows the shapes and what each does to the ion-acoustic
and electron-plasma features. The synthetic noise model's σ is now the realized per-pixel σ, the
point spread included. The table machinery itself is now general (`ChebyshevTable`), with
per-table degree caps and a signed ξ domain.

**Phase 4.C (the L-curve corner, and the shock fitted piecewise) is complete.** `lcurve_corner`
estimates the corner of the L-curve over a weight scan's result, the point of maximum curvature of
log R against log χ² (Hansen and O'Leary 1993), and reports it with the scan weights that bracket
it and the curvature at every scan point; it is a report beside the scan, never a choice: nothing
in the package calls it, and the weight a deck carries stays the user's to set. The shock problem
is also fitted with its time regularization split into two intervals around a 0.2 ns window at
the shock left unregularized (`examples/iaw_shock_piecewise.toml`), so that the jump is fitted from
the data alone. At cut-down (128 × 32) its residual against the truth across the jump, averaged
over `f_D`, `T_D` and `T_C`, is 0.70 of the whole-axis fit's at the same weight (the slow tier's
FR9), most of the gain in `T_C`. It is not fitted at full resolution: there each unregularized bin
is a third of the instrument's 0.05 ns temporal blur, so the data alone do not determine the gap's
bins one by one. A regularizer term whose interval reaches outside a selected variable's own
interval is now refused, naming both, where it used to make the regularizer's value `NaN` silently.

**Phase 4.D (consolidation) is complete.** `fit(deck)` now warns whenever a fitted entry ends within
`BOUND_PROXIMITY_TOL` of one of its bounds, naming the variable and its bins, and the result file
lists those entries (`fit/near_bound_entries`); the fit and its `converged` are unchanged. One
coordinate stuck on a bound used to pass without a word: the slow tier's cut-down reproductions carry
19 such entries on the EPW problem and 30 on the multi-flow one. A sweep of `src/` for comparisons
on automatic-differentiation duals fixed 26 sites where a dual exactly at a boundary was admitted or
refused differently from its plain value. And `GeneralizedNormal`'s EPW χ²/n of about 41 on the
two-channel example is mostly one time bin whose fitted exponent leaves the truth, with or without
the camera's point spread; the point spread accounts for a smaller excess near the profile's peak.
Nothing was changed on the strength of it.

**Phase 4.E (the `GeneralizedNormal` excursion) is complete.** The one time bin whose fitted exponent
left the truth in Phase 4.D's two-channel fit (m 5.4658 against 4.6111) was diagnosed by
`examples/diagnostics/generalized_normal_excursion.jl`. The derivatives are correct and the fit is a
genuine local minimum; a refit from the truth finds a basin whose EPW χ²/n is 9.6179 rather than
41.2117. The minimum comes from the forward model evaluating the spectrum at pixel-centre wavelengths
before the spectral instrument function: at large m the flat-topped shape leaves the
electron-plasma-wave resonance almost undamped and a fifth of a pixel wide, so the sampled model can
hide its peak between pixels. A real spectrometer blurs the spectrum before it reaches the pixels, so
this is an approximation of the model, not of the instrument; it matters only for a resonance narrower
than a wavelength pixel. Nothing was changed.

**Phase 4.F (the staged fit as a deck option, and the wavelength sampling) is complete.** A two-camera
deck can now say `procedure = "staged"`: `fit(deck)` then fits each camera alone, starts the joint fit
from those fits and polishes it, the procedure that makes a two-camera fit converge, and writes each
camera's own fit into the result file (`examples/two_channel_staged.toml`). A new instrument key,
`pixel_integration = { subsamples = N }`, makes the forward model average the spectrum over `N` points
inside each wavelength pixel before the spectral instrument function; without it, or with `N = 1`, the
model is exactly what it was. After each fit, `fit(deck)` warns when a fitted electron-plasma-wave
resonance is narrower than a pixel, naming the count to refit with. Refitted at the count it asks for
(16), Phase 4.E's outlying `GeneralizedNormal` bin returns to the truth (m 4.5309 against 4.6111) and
the EPW χ²/n falls from 41.2117 to 8.7121. Phase 4.D's cut-down EPW reproduction, whose exponent sits on
its upper bound in 17 of 32 bins, is not cured that way: at the count asked for there (64) the same
number of bins stay on the bound. No default changed.

**Phase 4.G (the tabulated k-smearing) is complete.** A new instrument key, `tabulated_k_smearing = {
offsets_deg = [...], weights = [...] }`, averages the spectrum over an explicit list of collection angles
and their weights, the rays an optics model gives, where `k_smearing` assumes a uniform spread; the angles
are offsets from the geometry's `theta_deg`, and nothing in it is fitted. Given a `k_smearing`'s own five
angles and weights it reproduces that effect's spectrum exactly. On the first real shot's ten rays the
spectrum differs from a uniform spread of the same width by at most 0.09 % of the peak on the two-channel
example's electron-plasma-wave window, against 3.1 % for the smearing itself, at about twice the cost of
five angles. No default changed.

**Phase 4.H (the convective gain in a fit) is complete.** The convective gain of Turnbull et al., which
`spectrum` has computed since Phase 2, now reaches the fit: a geometry key, `gain = "convective"` (or
`"weak"`), given beside the probe's `intensity_wcm2` and `diameter_um`, makes the forward model, its
Jacobian and the resonance-width check all carry it, each collection ray with its own gain; a deck that
asks for it without the probe's intensity or diameter is refused when it is read. Without the key the
model is exactly what it was. And an instrument table can no longer give `k_smearing` and
`tabulated_k_smearing` together, which smeared the spectrum twice.

**Phase 4.I (a per-bin amplitude, a start from a file, held series and the sequential stage) is
complete.** A camera's sensitivity may now take one fitted value per time bin: write `sensitivity = {
scale = "a_epw" }` with `a_epw` declared `time_varying = true`, the one instrument key a time-varying
variable may be bound to, which normalizes each time slice on its own (and gives up the absolute
intensity's information about the density); a `temporal_irf` beside it is refused. `[fit.start] = {
file = "...", variables = [...] }` starts the named variables from a series in a file, and `[fit.hold]`
of the same form holds them there, unfitted; the file is a result file or one written by
`write_start(path; axis_ns, values, units)`, its series interpolated linearly onto the fitted bins and
never extrapolated. A staged deck may give `sequence = ["epw", "iaw"]` and `hold = { iaw = ["logn",
"T_e", "m_e"] }` in `[optimizer.staged]`: the stages then run in that order, a later stage holding what
an earlier one fitted and starting the rest from it. `fit(deck)` now warns when a stage or the final fit
stops at its iteration cap. On a two-camera synthetic problem with the signal ramped six-fold across
the window, one scale per camera biases the fitted density by 17 % (median) and the per-bin amplitude
recovers it to 1.7 %, the un-ramped fit's own accuracy. Without the new keys every fit is exactly what
it was.

**Phase 4.K (progress lines during a fit) is complete.** A long fit whose output goes to a file can now
show how it is going while it runs: `progress = 10` in `[optimizer.levenberg_marquardt]`,
`[optimizer.lbfgs]` or `[optimizer.noisy_descent]` prints a line every ten iterations with the iteration,
χ² and χ² per pixel, the change in χ² over the last 20 iterations and its rate, the largest gradient entry
beside the optimizer's own stopping threshold, the number of fitted values at a bound and the time per
iteration, each line written out at once, then a final line with the fit's status; a staged deck names
each stage before it runs. The lines go to standard output (or to any stream passed to `solve` as
`progress_io`) and change nothing about the fit: every result is the same, bitwise, with them on or off.
Without the key nothing is printed.

**Phase 4.J (the continuous ion charge) is complete.** An ion's `charge` may now be any real number of
at least 1, and a fitted variable: write `charge = 4.6` in a `[plasma.ions.<name>]` table, or `charge =
"ZC"` with `ZC` declared in `[fit.variables]`, time-varying and regularized like any other variable; its
lower bound must be at least 1. A non-integer charge stands for a mixture of charge states of one
element at one temperature and flow and is that mixture's ⟨Z²⟩/⟨Z⟩, which the spectrum reproduces
exactly; it is not the number-average charge, and the density the package derives for the species is
the equivalent single-state ion's. Trace files are now written as `otis-trace-2`, the charge a real
number, and an older trace file is refused. For every integer charge the model and every fit are
exactly what they were.

**Phase 4.M (the readable fit output) is complete.** Every deck fit now also writes `readable.h5`
beside `result.jld2`, a file to read directly from Python: the time and wavelength axes in ns and nm;
per camera the data, σ and model the fit was made against, every pixel, with the mask of fitted pixels
and the χ² of each time bin; every fitted variable and dependent as one array in its deck unit, with
its unit and a flag per bin saying whether it was fitted, held, a dependent or left on a bound; and the
fit's summary. In h5py the arrays read one row per time bin. `write_readable(path, result, problem;
deck, laplace, scatter)` writes the same file from a script, adding the Laplace σ and, after
`slice_scatter`, every fine window's fitted values with that window's own data and model. `readable =
false` in `[output]` switches the file off; `result.jld2` and every fit are the same either way.

**Phase 4.L (a stall test for Levenberg–Marquardt) is complete.** A `levenberg_marquardt` fit that has
reached a plateau can now end there instead of running on to its iteration cap: `lm_rtol = 1e-6` and
`lm_rtol_window = 50` in `[optimizer.levenberg_marquardt]` end the fit, reported converged with status
`lm_rtol`, once the objective has moved by less than one part in a million over the last 50 accepted
steps, the change the progress lines' six printed figures show. Up to the step where it stops the fit is
exactly the one without the keys, and `fit`'s warnings, among them the check of whether the stopped point
is a minimum, run as before; the result file's `fit/status` reads `"lm_rtol"`. Without the keys every fit
is exactly what it was. A deck fit with `readable = false` now also removes a `readable.h5` an earlier run
left in its output directory.

**Phase 4.N (the per-slice scatter on a real deck, and housekeeping): its first step is complete.**
`slice_scatter(deck, result)` now runs on a deck with `[fit.intervals]` and `[fit.hold]`, as real decks
have them: a variable with an interval is fitted in the windows of the bins its interval holds and fixed
at its `init` in the others, and a held variable takes its series at each window's averaged time, or the
bin's held value where the window lies beyond the file's time range. Beside each bin's detrended spread
the scatter now reports the median and the 16th and 84th percentiles of the bin's window values (`p16`,
`median`, `p84`, numpy's default definition), and the value each window held a held variable at
(`held`); `write_readable(…; scatter = sc)` writes them as `scatter/p16/<name>`, `scatter/median/<name>`,
`scatter/p84/<name>` and `scatter/held/<name>`, and the readable file is now `otis-readable-2`. On a deck
without intervals or held series the scatter is exactly what it was, and `result.jld2` is unchanged.

## Installing and testing

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Set `OTIS_TESTS` to a comma-separated, case-insensitive list of substrings to run only the
`@safetestset`s whose name matches one of them (unset or empty runs everything). The slow tier
(sweeps and anything needing CairoMakie) is not part of `Pkg.test()`; run it by hand with
`julia --project=test/slow test/slow/runslow.jl`, which uses its own environment and includes
`test/slow/test_reproduce.jl`, the cut-down (128 × 32) reproduction of the three foo2026 test
problems.

## Running

```
julia --project -t auto
```

launches Julia with all available threads. Call `pin_blas_threads()` before a large evaluation:
OTIS threads over time bins itself, and BLAS on many threads oversubscribes the machine against
it. **Thread a real fit**: the Jacobian's cost is the per-bin dual pass, which `foreach_bin!`
parallelizes, measured at 3.4× from `-t 8` (`docs/real_data.md`); start one with

```
julia --project=. -t 8 -e 'using OTIS; fit("shot12345.toml")'
```

not a bare `julia --project=.`.
