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
the noise model against its own σ. Phase 3, the inversion, is next. The roadmap is kept outside
this tree.

## Installing and testing

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Running

```
julia --project -t auto
```

launches Julia with all available threads. Call `pin_blas_threads()` before a large evaluation:
OTIS threads over time bins itself, and BLAS on many threads oversubscribes the machine against
it.
