# OTIS.jl

OTIS, the **O**ptical **T**homson **I**nversion **S**uite, is a Julia package for forward
modelling and inverting optical Thomson scattering (OTS) spectra from plasmas, including
streaked (time-resolved) data inverted by regularizing across the whole streak at once. Its
central design choice is that the velocity distribution of each species is an **input, not an
assumption**: every species carries a dimensionless shape, and the scattering theory is written
once against the interface that shape provides, so non-Maxwellian distributions (super-Gaussian,
kappa, and others) are first-class rather than fitted away.

## Status

Phase 0 is complete: CODATA constants and boundary units, the plasma dispersion function
`Z` with derivative rules, the velocity-shape interface with the Maxwellian, and `Species`,
`Plasma` and the derived plasma quantities. Phase 1, the static forward model, is next. The
roadmap is kept outside this tree.

## Installing and testing

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Running

```
julia --project -t auto
```

launches Julia with all available threads. BLAS will be pinned to a single thread by the
fitting driver once it exists, to avoid oversubscription when the package's own parallelism is
in use.
