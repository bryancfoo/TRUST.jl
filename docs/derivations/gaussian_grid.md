# `GaussianGrid`: a distribution on a fixed grid in lab velocity

This document derives what `src/distributions/gaussian_grid.jl` implements, and the lab-frame
susceptibility of `src/scattering/susceptibility.jl` (Phase G.1,
[OQ-68](../../dev/open_questions.md#oq-68); `conventions.md` 1.12). Units are SI throughout: velocities in
m/s, `f` in s/m, `𝒞f` in s²/m², temperatures in J. `Z′(ζ) = −2(1 + ζZ(ζ))` is the derivative of the plasma
dispersion function (`plasma_dispersion.md`). *(Step G.1.3 adds Sections 7 and 8.)* Every number quoted was measured through the package on
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
`lab-frame family`, GC1 to GC12): the species, the spectra, the trace and the refusals. Every loose tolerance
is a sampling error of the grid, pinned at the measurement above; every tight one is an identity.
`dev/phaseG1_instructions.md` D10 lists each row with its tolerance and its reason.
