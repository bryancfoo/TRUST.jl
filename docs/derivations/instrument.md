# The instrument chain

> **Front matter (Step 2.2, 2026-09-09/10).** Windows 11, Julia 1.12.7. This document's parts 1
> to 4 are written at Step 2.2, alongside `src/instrument/effects.jl`,
> `src/instrument/background.jl` and `src/instrument/experiment.jl`. Part 5 (the noise model)
> is Step 2.5's.

## 1. The chain

`measure(experiment)` returns the measured streak from the ideal one in two groups, in a fixed
order ([OQ-13](../../dev/open_questions.md#oq-13)):

- **Group 1, inside the per-bin kernel**: effects that change what physics is evaluated rather
  than acting on the streak — `KSmearing`, `GradientSmearing`, `ProbeOffset`. Any number of
  these may be given; they run in the user's own relative order.
- **Group 2, after the per-bin loop, in this fixed order**: `SpectralIRF`, `TemporalIRF`,
  `Gain`, then any number of backgrounds.

In one line, with `P_kernel` the ideal streak the per-bin kernel produces (group 1 already
folded in):

```
M = B + g · (K_λ · P_kernel · K_tᵀ)
```

with `K_λ` the spectral convolution matrix, `K_t` the temporal one, `g` the scalar gain, and `B`
the sum of the background terms (each additive; `ConstantBackground` and `PolynomialBackground`
add a term independent of `P_kernel`, `ImageBackground` adds `scale · image`). `Experiment`'s
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

## 3. The in-kernel effects

All three are weighted sums over samples of the plasma and/or the geometry, folded by
`_accumulate!` (spec D16):

```
P(λ) = Σ_q w_q · spectrum(p_q, g_q, λ)
```

- **`KSmearing`.** The collection aperture is modelled as a uniform spread of angles of
  half-width `half_angle`: `θ_q = θ + half_angle·ξ_q` on Gauss–Legendre nodes `ξ_q` on
  `[-1, 1]`. Changes `θ` and therefore `k` (Sheffield's dispersion relation), not `ω`.
- **`GradientSmearing`.** A normal distribution of one plasma leaf (a species' `T` or `u`, or
  `n_e`) inside the scattering volume: `x_q = x + width·√2·ξ_q` (additive) on Gauss–Hermite
  nodes `ξ_q`, or, for `n_e` only, `x_q = x·exp(width·√2·ξ_q)` (**multiplicative**, so the
  smeared density stays strictly positive at any width and any node — EF11 is the reason this
  form was chosen over an additive one).
- **`ProbeOffset`.** `lambda_probe -> lambda_probe + offset`, a single sample of weight one. It
  enters `wavevector` and `_omega_probe` through `geometry.lambda_probe` alone; the detector
  axis `expt.lambda` is never touched (EF12, Pitfall 11) — the probe moves, not the detector.

**Both quadrature rules use an odd node count** (`1, 3, 5, 7`) so the nominal (unsmeared) value
is always one of the nodes, and both normalize their weights to sum to one at construction, so
`nodes = 1` is the identity to round-off for either rule.

## 4. Linearity

`islinear(effect)` answers exactly the question Step 3.2's Jacobian assembly will ask: *may
this effect be pushed through as a fixed linear operator?*

| Effect | `islinear` | Why |
|---|---|---|
| `SpectralIRF`, `TemporalIRF`, `Gain` | `true` | each is a fixed matrix or scalar multiplication of the streak |
| `ConstantBackground`, `PolynomialBackground`, `ImageBackground` | `false` | **affine**, not linear: `effect(aP+bQ) - a·effect(P) - b·effect(Q) = (1-a-b)·B`, tested directly (EF15, BG4) |
| `KSmearing`, `GradientSmearing`, `ProbeOffset` | `false` | these act on the **plasma**, not the streak — `effect(aP+bQ) = a·effect(P)+b·effect(Q)` is not even a meaningful statement about them, since they change what is evaluated rather than post-processing a value that already exists |

Every effect with `islinear(e) == true` satisfies the linear identity to round-off (EF14); every
background satisfies the affine identity instead (EF15, BG4); the three kernel effects have
`stage(e) == :kernel` and `islinear(e) == false` by definition.
