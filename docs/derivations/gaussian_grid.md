# `GaussianGrid`: a distribution on a fixed grid in lab velocity

This document derives what `src/distributions/gaussian_grid.jl` and `src/distributions/grid_projection.jl`
implement, and the lab-frame susceptibility of `src/scattering/susceptibility.jl` (Phase G.1,
[OQ-68](../../dev/open_questions.md#oq-68); `conventions.md` 1.12). Units are SI throughout: velocities in
m/s, `f` in s/m, `𝒞f` in s²/m², temperatures in J. `Z′(ζ) = −2(1 + ζZ(ζ))` is the derivative of the plasma
dispersion function (`plasma_dispersion.md`). Every number quoted was measured through the package on
2026-11-12 (Linux x86-64, Julia 1.13.1, ForwardDiff 1.4.5, SpecialFunctions 2.9.0, QuadGK 2.11.3).

## 1. The comb

A `GaussianGrid` is the reduced distribution of one species along **k̂**, written as `N_v` Gaussians of one
width `σ` at fixed, evenly spaced nodes:

    f(v) = (1/m₀) Σₙ wₙ G(v; vₙ, σ),     G(v; c, s) = e^{−((v − c)/s)²} / (s√π),     m₀ = Σₙ wₙ,
    vₙ = u₀ + (n − (N_v + 1)/2) dv,      σ = width · dv,      wₙ = w⁰ₙ e^{gₙ}.

`G(·; c, s)` is a Maxwellian of thermal speed `s` centred on `c`, with unit integral, so `∫ f dv = 1` for
any weights; with `wₙ > 0` the distribution is positive everywhere. Its derivative is

    f′(v) = (1/m₀) Σₙ wₙ (−2 (v − vₙ)/σ²) G(v; vₙ, σ) = −(2/(m₀ σ² √π)) Σₙ wₙ zₙ e^{−zₙ²},    zₙ = (v − vₙ)/σ.

**The parameters are the `gₙ`**, log-ratios of the weights to a default `w⁰`: `g = 0` is the default, every
finite `g` gives positive weights, and positivity needs no constraint in a fit. **One direction is flat**:
adding a constant `c` to every `gₙ` multiplies every weight by `e^c` and `m₀` by the same factor, so `f`,
`f′`, `𝒞f` and every moment are unchanged (GG8 tests it to round-off). A fit fixes that direction by holding
one `g` (the user's decision of 2026-11-12, built in Phase G.4); the type keeps all `N_v` parameters.

**The grid is fixed**: `u₀`, `dv`, `σ` and `w⁰` are fields, never parameters, the same at every time bin
([OQ-68](../../dev/open_questions.md#oq-68) item 2, the user's). There is no fitted flow or temperature;
both are moments (Section 3).

## 2. The Cauchy integral

For one Gaussian, substitute `v = c + s y`, `dv = s dy`, `v − v_φ = s(y − ζ)` with `ζ = (v_φ − c)/s`:

    ∫ G′(v; c, s)/(v − v_φ) dv = ∫ (1/s²)(−2y e^{−y²}/√π) (s dy)/(s (y − ζ)) = (1/s²) 𝒞_Maxwellian(ζ) = Z′(ζ)/s²,

the Maxwellian's Cauchy integral (`shapes.md` Section 2) divided by `s²`, on the same Landau contour. Summing
over the nodes,

    𝒞f(v_φ) ≡ ∫ f′(v)/(v − v_φ) dv = (1/(m₀ σ²)) Σₙ wₙ Z′((v_φ − vₙ)/σ).

This is `cauchy_dfdv(f, v_φ)`: `Zprime` and nothing else, no table, no quadrature. The comb is entire, so
`𝒞f` is defined for every complex `v_φ` in both half-planes with no branch, and `Zprime`'s dual methods
(`plasma_dispersion.jl`) carry every derivative. **On the real axis** `Im Z′(x) = −2√π x e^{−x²}`, so

    Im 𝒞f(v) = (1/(m₀σ²)) Σₙ wₙ (−2√π zₙ e^{−zₙ²}) = π f′(v)

identically (GG5: 7.8e-16 of the largest `π|f′|`). **A partial in one weight** is one term: since
`∂wₖ/∂gₖ = wₖ` and `∂m₀/∂gₖ = wₖ`,

    ∂𝒞f/∂gₖ = (wₖ/m₀) (Z′((v_φ − vₖ)/σ)/σ² − 𝒞f(v_φ)),

which ForwardDiff reaches through multiplication alone (GG7, to 1e-12); with the symmetric option a
parameter is shared by two mirrored nodes and its partial is the sum of their two terms.

## 3. The moments, the flow and the temperature

Each Gaussian's moments about the origin are those of `c + sY`, `Y` the unit Gaussian `e^{−y²}/√π`, whose
even moments are `E[Y^j] = (j − 1)!!/2^{j/2}` and odd moments zero:

    ∫ vⁿ G(v; c, s) dv = Σ_{j even ≤ n} C(n, j) c^{n−j} s^j (j − 1)!!/2^{j/2},
    moment(f, n) = ∫ vⁿ f dv = (1/m₀) Σₖ wₖ Σ_{j even ≤ n} C(n, j) vₖ^{n−j} σ^j (j − 1)!!/2^{j/2},

closed form for every `n ≥ 0`. The mean velocity and the variance follow, the variance in two passes for
accuracy:

    ū = (Σₙ wₙ vₙ)/m₀,      V = (Σₙ wₙ [(vₙ − ū)² + σ²/2])/m₀,

each Gaussian adding its own variance `σ²/2`. **A species carrying the distribution has `u = ū` and
`T = m V`** (`conventions.md` 1.12): the mean velocity and the variance temperature along **k̂**, exactly
what `u` and `T` mean for a shape (`shapes.md` Section 3). They are computed wherever the species is built
or its distribution replaced (`electrons`, `ion`, `plasma(trace, j)`), never given, and never leaves.

## 4. The susceptibility in lab velocity, and its equivalence with rule 1.6

Sheffield Eq. 3.3.9 in SI (`susceptibility.md` Section 2) is

    χ_s(k, ω) = −(ω_ps²/k²) ∫ (∂f_s0/∂v)/(v − ω/k) dv,    ω_ps² = n_s q_s²/(ε₀ m_s),

with `f_s0` the unit-normalized reduced distribution along **k̂**. For a lab-frame distribution the integral
is `𝒞f(ω/k)` by definition, so

    χ_s(k, ω) = −(ω_ps²/k²) · 𝒞f(ω/k),

which is `susceptibility`'s method for the family. **It is rule 1.6 with the thermal speed divided out**:
for `f(v) = φ((v − u)/v_th)/v_th`, `susceptibility.md` Section 3 shows `𝒞f(ω/k) = 𝒞φ(ξ)/v_th²` with
`ξ = (ω/k − u)/v_th`, and `ω_ps²/(k² v_th²) = α_s²/2`, so the two forms are the same number. GC3 checks it
exactly, with no sampling error: a test-local lab-frame distribution that is a Maxwellian in closed form,
carried by a species, gives the shape route's `χ` and `S(k, ω)` to 1e-13. `ω_ps²` is written without a
square root (`_plasma_frequency_sq`, linear in the density), so that its partial at a density of exactly
zero is finite (`conventions.md` 5.15; GC9). In `S(k, ω)` (`conventions.md` 1.7) the species' term is its
weight times `f(ω/k)`, with no `1/v_th`: `f_lab(sp, v) = f(v)`. No temperature and no flow appear in either.

## 5. The Maxwellian default, and the symmetric option

**The temperature form** takes the user's inputs: `u₀`, `T_ref`, the node count and the spacing as a
fraction of the reference thermal speed `v_ref = √(2 T_ref/m)` of the species' mass, `dv = spacing · v_ref`.
Its default is the Maxwellian of `T_ref` and `u₀`, written on the grid in closed form:

    w⁰ₙ = dv · G(vₙ; u₀, s₀),     s₀ = √(v_ref² − σ²).

Why `s₀`: the comb is a sampled convolution, `Σₙ dv h(vₙ) G(v; vₙ, σ) ≈ ∫ h(y) G(v; y, σ) dy`, and the
convolution of two Gaussians of widths `s₀` and `σ` is a Gaussian of width `√(s₀² + σ²) = v_ref`: the
default is narrowed by exactly the width the comb's own Gaussians add back. So `σ < v_ref`, that is
`spacing · width < 1`, is required. The weights are written from the offset `(n − (N_v + 1)/2) dv`, so they
are exactly symmetric about `u₀`. At `g = 0` the moments are `u₀` and `T_ref` to the grid's extent (GG2:
`T` within 1.2e-12 of `T_ref` at the defaults, the Gaussian beyond ±5 `v_ref` being what is missing).

**The explicit form** takes `dv` itself and the default weights from the caller, with no temperature.

**The symmetric option** (off by default) ties `wₙ = w_{N_v+1−n}`: the parameters are `g1` to `g⌈N_v/2⌉`
and node `n` reads parameter `min(n, N_v + 1 − n)`. With symmetric default weights, `f` is even about `u₀`,
and it is bitwise the full form given the mirrored `g` (GG10).

## 6. How well the default grid holds a Maxwellian, and why

**The error floor is the comb's own ripple.** By Poisson summation the sampled convolution of Section 5 is the
continuous one plus aliases at wavenumbers `2πm/dv`; the first has relative amplitude about

    2 exp(−(π s_p/dv)²),     s_p = s₀ σ / v_ref,

`s_p` being the width of the product `h(y) G(v; y, σ)`. It depends on `σ/dv = width` and hardly on the
spacing: the defaults give `s_p/dv = 1.145` and a predicted 4.8e-6 in `f`; measured 4.83e-6. In `f′` and `𝒞f`
the ripple is multiplied by its wavenumber, about 26 times here.

| `g = 0` against the Maxwellian of `T_ref`, `u₀`, on `x = (v − u₀)/v_ref ∈ [−6, 6]` | in `f` | in `f′` | in `𝒞f` (of its peak, 2) |
|---|---|---|---|
| defaults: 41 nodes, spacing 0.25, width 1.2 (GG2) | 4.83e-6 | 1.28e-4 | **9.80e-5** |
| width 1.5 (GG2) | 1.03e-8 | 2.58e-7 | **1.97e-7** |
| 97 nodes, spacing 0.125, width 1.2 (the projection, Section 7) | 1.85e-6 | | 8.06e-5 |

So **halving the spacing at the same width ratio leaves the floor near 1e-4**, and widening the Gaussians
removes it. The plan's M1 measured the same floor and attributed it to its scratch fit; it is the basis's.
The width cannot simply be raised, because positive weights of a wider Gaussian cannot make a steep
edge (the plan's M1: at width 1.5 the flat-topped shapes worsen to 8.6e-3 and 2.5e-2).

**The spectrum** inherits it: a gridded Maxwellian's `spectrum` against the package's Maxwellian spectrum,
5e19 cm⁻³, electrons at 700 eV, deuterons at 250 eV, probe 263.25 nm, θ = 60°, 512 wavelengths, no
instrument function:

| Case | max \|ΔP\| / peak |
|---|---|
| gridded electrons, EPW window 220 to 260 nm (GC4; the plan's M4 read 5.7e-5 at 49 nodes with a 0.5 nm instrument function) | 6.14e-5 |
| gridded deuterons at `T_ref` = 250 eV, IAW window 262 to 264.5 nm (GC5) | 6.23e-5 |
| one electron grid at `T_ref` = 700 eV holding a Maxwellian at 500, 700, 1000 eV (GC6; M8) | 7.61e-5, 6.14e-5, 5.55e-5 |

At 1000 eV on the 700 eV grid the derived `T` reads 1.23e-8 low: the hotter Maxwellian reaches past the
grid's ±5 `v_ref`.

**Off the real axis.** At `g = 0` the comb's `𝒞f` follows `Z′` better above the axis and departs from it fast
below: each Gaussian continued to `v − iy` grows like `e^{(y/σ)²}`, the Maxwellian's only like `e^{(y/v_ref)²}`,
so the ripple is amplified. Measured (default grid, max `|Δ𝒞f|`/2 over `Re ξ ∈ [−5, 5]`, `ξ = (v − u₀)/v_ref`):

| `Im ξ` | +0.5 | +0.2 | +0.1 | +0.05 | −0.05 | −0.1 | −0.2 | −0.5 |
|---|---|---|---|---|---|---|---|---|
| error | 1.3e-9 | 1.0e-6 | 1.0e-5 | 3.1e-5 | 3.1e-4 | 9.8e-4 | 1.0e-2 | 12 |

The comb's own `𝒞f` is exact everywhere (GG4 checks it against quadrature in both half-planes); what fails
below the axis is the comb as **an approximation of a Maxwellian**. A search for a damped root of `ε` in the
lower half-plane should not trust a grid there beyond about `Im ξ ≈ −0.05`. Nothing in the package does that
search.

**Far outside the grid** the comb's `𝒞f` is the Maxwellian's own: at `v = u₀ ± 150 v_ref` (an ion distribution
at electron phase velocities), `𝒞f v_ref²` against `Z′(±150)` reads 2.4e-11 relative (GG9), the round-off of
`Z′` at large argument, since both tend to `1/ζ²` with the same variance.

## 7. The projection onto a grid

`project_to_grid(source, target)` writes a distribution on `target`'s grid (Phase G.1 spec D14). The source is
a species of either family (a shape through `f_lab` and `𝒞φ(ξ)/v_th²`) or any lab-frame distribution, a
`GaussianGrid` on another grid among them.

**The fit.** The unnormalized comb `p(v) = Σₙ w⁰ₙ e^{g_{π(n)}} G(v; vₙ, σ)` is fitted to the source's `f`
by least squares on the collocation set `v₁ − 5σ` to `v_{N_v} + 5σ` in steps of `dv/4`, residuals scaled by
`dv`, in the logarithms of the weights, so the weights stay positive. The start is the source sampled at the
nodes, `w⁰ₙ e^{gₙ} = f(vₙ) dv`, floored at 1e-300. Levenberg–Marquardt: `(JᵀJ + λD) δ = −Jᵀr`, `D` the
diagonal of `JᵀJ` floored at 1e-12 of its largest entry, solved by Cholesky; `λ` from 1e-3, divided by 3
on an accepted step and multiplied by 5 on a rejected one; every `g` held in [−200, 200]. It stops,
converged, at the first accepted step that lowers the sum of squares by less than 1e-12 of itself, or when
no step lowers it at `λ` above 1e12; after 500 iterations, not converged. Since `p` is unnormalized and
the source has unit integral, the scale is fixed by the data and nothing is flat. Deterministic: the same
inputs give the same `g` bitwise, whatever `g` the target carried. At the stop the residual is orthogonal to
every column of the Jacobian to about 1e-8 (GP7, which writes the problem out again and checks it).

**The errors it reports** are measured on the evaluation set `v₁` to `v_{N_v}` in steps of `dv/8`: the
largest difference in `f` and in `𝒞f`, each divided by the source's largest value there.

**Each registered shape** at the default spacing and width, on a grid of `T_ref = T` and 49 nodes out to ±6
thermal speeds (GP1), against the plan's M1 (which standardized its comb to the three normalizations):

| Source | error in `𝒞f` | error in `f` | iterations | M1 |
|---|---|---|---|---|
| `Maxwellian` | 9.80e-5 | 4.83e-6 | 14 | 9.6e-5 |
| `SuperGaussian(3)` | 9.44e-5 | 4.11e-6 | 19 | 8.9e-5 |
| `SuperGaussian(5)` | 2.57e-3 | 2.21e-4 | 21 | 2.9e-3 |
| `Kappa(3)` | 6.39e-4 | 1.30e-4 | 7 | 1.1e-2 |
| `SkewNormal(2)` | 1.65e-4 | 9.66e-6 | 16 | 2.0e-4 |
| `GeneralizedNormal(4)` | 3.16e-3 | 4.38e-4 | 13 | 3.3e-3 |
| `HermiteExpansion(0.02, 0.01, 0, 0)` | 1.44e-4 | 8.69e-6 | 15 | not measured |

The smooth shapes sit on Section 6's floor; the flat-topped ones (`SuperGaussian(5)`, `GeneralizedNormal(4)`)
are limited by the width, since a steep edge needs narrower Gaussians: at spacing 0.125 (97 nodes) both
read 8.5e-5 and 7.4e-5. A projection takes 5 to 10 ms once compiled.

**`Kappa(3)` is limited by the grid's extent, and it shows in the moments**, not in `𝒞f` inside the grid. Its
power-law tail beyond the last node carries variance a finite grid cannot hold: the derived `T` of the
projection reads 1.1 % low on ±6 thermal speeds (`T/T_source` 0.98877), 0.27 % low on ±10 (0.99728) and
0.11 % on ±15 (0.99887), while the error in `𝒞f` on the grid falls only from 6.4e-4 to 5.6e-4 (GP2). M1's
1.1e-2 is the standardization's: forcing the variance to the source's moved the whole comb.

**A projection onto a grid twice as fine and back** (GP4: `SkewNormal(2)` onto the default grid, then onto 81
nodes at spacing 0.125, then back): each step reports 1.0e-4 in `𝒞f`, the finer grid's own floor, yet the
round trip's `𝒞f` lies within 5.3e-8 of the original's: back on its own grid the coarse comb is recovered, the
ripple of one step undone by the next.

## 8. What a fixed grid cannot hold

Three limits, each a limit of a fixed grid of positive Gaussians (the plan's M7), with their numbers here:

- **Narrower than `σ`**: a comb is never sharper than its own Gaussians. M7's narrow beam, 5 % of the
  distribution in a Gaussian of width 0.1 `v_th` at 2.5 `v_th` beside a Maxwellian core, projects onto the
  default grid with an error of **0.87 in `𝒞f`** and 0.29 in `f` (GP3), and onto a grid twice as fine still
  0.49 and 0.12: not representable. A cold beam is a species of its own.
- **Beyond the last node**: a tail past the grid is cut off, and its variance with it (`Kappa(3)`, Section 7).
  Size the grid for the hottest state a streak reaches.
- **A negative lobe**: positive weights make a positive `f`. `HermiteExpansion` with a lobe is projected onto
  its positive part.

And, by Section 6, the comb is a Maxwellian's approximation to about 1e-4 of the peak at the default width,
not bitwise: tests compare with a stated tolerance.

## 9. The cost

One `spectral_density` call, latency in a loop of 20 000, warmed, one thread, the plasma of Section 6 at one
`(k, ω)` (Phase G.1 spec D17; the plan's M6 read 2.6 to 2.8 µs at 49 nodes, 10.2 µs and 26.9 µs under duals
of 8 and 52):

| Electrons | value | dual, 8 partials | dual, 44 partials |
|---|---|---|---|
| `Maxwellian` (dual in `T`) | 0.18 µs | 0.69 µs | 1.4 µs |
| `SuperGaussian(3)` (dual in `m`) | 0.37 µs | 1.1 µs | 8.7 to 13.8 µs, **17 872 B** |
| `GaussianGrid`, 41 nodes (dual in every `g`) | 2.1 to 2.2 µs | 3.6 µs | 6.6 µs |

Every `GaussianGrid` call allocates nothing; so does `plasma(trace, j)` with a gridded species whose `g` vary,
as a value and under duals of 8 and 44 partials (GC8). The 44-partial `SuperGaussian` row is recorded for
the phases that will take one wide dual sweep over a gridded species beside a fitted super-Gaussian; nothing
here uses it.

## 10. The tests

`test/distributions/test_gaussian_grid.jl` (gate `GaussianGrid`, GG1 to GG13): the distribution's closed
forms against quadrature, ForwardDiff and the Maxwellian; `test/plasma/test_species_lab_frame.jl` (gate
`lab-frame family`, GC1 to GC12): the species, the spectra, the trace and the refusals;
`test/distributions/test_grid_projection.jl` (gate `grid projection`, GP1 to GP7): the projection. Every
loose tolerance is a sampling error of the grid, pinned at the measurement above; every tight one is an
identity. `dev/phaseG1_instructions.md` D10 lists each row with its tolerance and its reason.
