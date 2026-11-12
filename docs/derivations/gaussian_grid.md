# `GaussianGrid`: a distribution on a fixed grid in lab velocity

This document derives what `src/distributions/gaussian_grid.jl` implements (Phase G.1,
[OQ-68](../../dev/open_questions.md#oq-68); `conventions.md` 1.12). Units are SI throughout: velocities in
m/s, `f` in s/m, `𝒞f` in s²/m², temperatures in J. `Z′(ζ) = −2(1 + ζZ(ζ))` is the derivative of the plasma
dispersion function (`plasma_dispersion.md`). *(Step G.1.2 adds Sections 4 and 9, Step G.1.3 Sections 7 and 8.)* Every number quoted was measured through the package on
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
what `u` and `T` mean for a shape (`shapes.md` Section 3).

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

## 10. The tests

`test/distributions/test_gaussian_grid.jl` (gate `GaussianGrid`, GG1 to GG13): the distribution's closed
forms against quadrature, ForwardDiff and the Maxwellian. Every loose tolerance is a sampling error of the
grid, pinned at the measurement above; every tight one is an identity. `dev/phaseG1_instructions.md` D10 lists
each row with its tolerance and its reason.
