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
costs about what a Maxwellian one does and is differentiable in the exponent `m`. Phase 2, time
and the instrument model, is next. The roadmap is kept outside this tree.

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
