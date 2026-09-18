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
costs about what a Maxwellian one does and is differentiable in the exponent `m`.

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
**The full-resolution EPW reproduction is deferred to Phase 3.C** — fitting the super-Gaussian
exponent spends about 93% of the Jacobian in an adaptive quadrature, which makes it too slow
for this phase's decks as they stand; it is not done, and it did not fail, it was not attempted
at full resolution. A full-resolution run of
the two Maxwellian problems was attempted and found impractical in one sitting: even with plain
`lbfgs` in place of the paper's own two-phase procedure, a single fit did not converge inside
1000 iterations, so a seven-point scan is many times a fifteen-minute budget
(`dev/reports/phase3_report.md`, Chunk 14). Step 3.8 adds `WarpMap` and `dewarp`, the OMEGA
streak's geometric correction, built and tested against a synthetic warp; no real OMEGA streak
was available this chunk, so a first fit of one has not been run. Phase 3.C, convergence and
the shape-parameter derivative, then Phase 4, the extensions, are next. The roadmap is kept
outside this tree.

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
