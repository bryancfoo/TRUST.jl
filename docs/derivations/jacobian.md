# The block-structured Jacobian

> **Front matter (Step 3.2b, 2026-09-13).** Windows 11, Julia 1.12.7; ForwardDiff 1.4.5. Written
> alongside `src/inversion/jacobian.jl`.

This document derives the assembled Jacobian `src/inversion/jacobian.jl` builds: the chain it
differentiates, why the ideal streak's Jacobian is block-diagonal in the time bins while the full
model's is block-*banded*, the per-bin dual pass that produces the per-bin blocks and the kernel
globals, the closed forms of the chain globals, and the two matrix-free products `jvp!`/`jtvp!`
with their transposes derived rather than asserted. Spec D17-D21.

## 1. The chain

`measure.jl`'s `apply!` implements, in canonical order (kernel effects, `SpectralIRF`,
`TemporalIRF`, `Sensitivity`, backgrounds):

```
    M = B + g .* (K_λ P K_tᵀ)
```

where `P` is the ideal (pre-instrument) streak, `N_λ × N_t`; `K_λ` is the spectral convolution
matrix (identity when no `SpectralIRF` is present); `K_t` is the temporal convolution matrix
(identity when no `TemporalIRF` is present); `g = scale .* response` is `Sensitivity`'s
wavelength-dependent response (all ones when no `Sensitivity` is present, `scale` alone when it
carries no calibration curve); and `B` is the sum of every background's contribution. `P` itself
is built bin by bin, `P[:, j] = _bin_kernel!(⋯, j)`, folding the in-kernel effects
(`KSmearing`, `GradientSmearing`, `WavelengthCalibration`) into the physics evaluated at each bin
— never touching the post-kernel chain. `KSmearing` and `GradientSmearing` each carry one
fittable kernel global (`half_angle`, `width`); `WavelengthCalibration` carries two,
`offset` and `dispersion_scale` (Step 4.A.1b). Because an effect may now have more than one bound
parameter, each kernel-stage effect carries a `Tuple` of its own `Binding`s, not a single optional
one (D11(b), `_collect_kernel_bindings`) — empty for an unbound effect, one entry per bound
parameter.

The residual is `r_k = (M[l,j] - D[l,j]) / σ[l,j]` at the `k`-th mask-true pixel, column-major
order (`residuals.jl`, D16). Since `D` and `σ` do not depend on θ,

```
    ∂r_k/∂θ = (1/σ[l,j]) · ∂M[l,j]/∂θ
```

so everything below derives `∂M/∂θ`, and the `1/σ` divide is the last step of both `jvp!` and
`jtvp!`.

## 2. Per-bin entries: the ideal streak is block-diagonal, the full model is not

For a trace variable's entry at bin `j` (one column of θ, in the sense of D11's index map),
`P[:, j']` for `j' ≠ j` does not depend on it at all — `_bin_kernel!` at bin `j'` reads only
`plasma(trace, j')`, which this entry's write (`trace.values[row, j]`) never touches. So

```
    ∂P[:, j'] / ∂θ_(v,j) = 0   for j' ≠ j
```

exactly, by construction — not an approximation, and JT5 checks it as an equality, not a
tolerance. This is what makes `blocks[j]`, an `N_λ × p_j` matrix, the whole of
`∂P/∂θ_(v,j)` for every `v` active at bin `j`.

Differentiating the chain at fixed everything else,

```
    ∂M[:,n] / ∂θ_(v,j) = g .* ( K_λ · (∂P[:,j]/∂θ_(v,j)) ) · K_t[n, j]
```

`K_t[n, j]` is in general **not** zero for `n ≠ j` — a temporal IRF spreads bin `j`'s ideal
column over every measured column `n` its kernel reaches. The *ideal* streak's Jacobian is
block-diagonal (Section 2's equation); the *measured* streak's is not, because the last stage of
the chain (`K_t`) is a convolution, not the identity. Concretely: for a `TemporalIRF` whose kernel
has support `[-w, w]` bins (a `GaussianKernel` truncated in practice by its column
renormalization, or a `TabulatedKernel`'s literal `offsets`), `K_t[n, j] ≠ 0` only for
`|n - j| ≤ w`, so `∂M[:,n]/∂θ_(v,j)` is nonzero only for `n` within `w` bins of `j`.
This is why `JᵀJ` (equivalently, `J_full^T J_full` restricted to two time-bins' worth of
columns) is **block-banded of half-bandwidth `w`, not block-diagonal**: entry `(θ_(v,j),
θ_(v',j'))` of `JᵀJ` involves `Σ_n (∂M[:,n]/∂θ_(v,j))·(∂M[:,n]/∂θ_(v',j'))`, a sum over the
columns `n` both `j` and `j'` reach, which is nonempty whenever `|j - j'| ≤ 2w`. JT6 exercises
this on P2, whose `TemporalIRF` (`fwhm_ns = 0.4` against a `0.33` ns bin spacing) is deliberately
wide enough that neighbouring bins couple by more than round-off.

A **global** entry (a scalar fitted variable bound to a plasma leaf, or to an in-kernel effect's
parameter) affects `P` at *every* bin, not one: `∂P[:,j]/∂θ_v ≠ 0` for every `j`, and its
contribution to `∂M` is the same convolution applied to the whole `N_λ × N_t` slice
`∂P/∂θ_v`, not to a single column. This is `kernel_globals[:,:,k]`, D17's third field: one full
`N_λ × N_t` matrix per kernel global, not a sum over bins (the sum over bins appears only in
`jtvp!`, contracting over pixels).

## 3. The per-bin dual pass: seeding and the split

`∂P[:,j]/∂θ_(v,j)` for every `v` active at bin `j`, and `∂P/∂θ_v` for every kernel global `v`,
come from **one forward-mode dual pass of the per-bin kernel per bin per sweep** (D18). The seed
vector at bin `j` has length `n_j = p_j + G` (`p_j` the bin's own entries, `G` the kernel-global
count): the bin's own θ entries first, then every kernel global, in a fixed order held constant
across bins. Because the seed is placed on θ itself (the *unconstrained*, logit-transformed
variable, D3) — not on the bound value `x` — the dual pass differentiates straight through
`_from_unconstrained`, so `dx/dy` (D3's `_dx_dy`) is obtained **for free** as part of the same
arithmetic that computes `dP/dθ`, with no separate multiplication anywhere in the per-bin path.

The chunk width `C = min(max_j p_j + G, JACOBIAN_CHUNK)` is one constant for the whole assembly
(never per bin), so the dual type `Dual{Tag,FT,C}` is a single concrete type the kernel
specializes on once. When `n_j > C`, a bin needs `⌈n_j / C⌉` sweeps; within sweep `s` (`0`-based),
only the seed positions in `(sC, (s+1)C]` carry a one-hot partial — every other entry, including
ones seeded in an *earlier* sweep of the *same* bin, carries an all-zero-partial dual, rebuilt
fresh each sweep rather than mutated (`ForwardDiff.Dual` is immutable, and a stale nonzero partial
surviving into the wrong sweep is exactly Pitfall 6's failure mode).

After the sweep, each entry of the dual output column `P[:,j]` carries `C` partials, indexed by
seed position within the sweep. The split (not an accumulation, a slice): partial at position `k`
(`k ≤ p_j`) becomes `blocks[j][:, k]`; partial at position `p_j + k` (kernel global `k`) becomes
`kernel_globals[:, j, k]`. `blocks[j]` therefore stays `N_λ × p_j` whatever `G` is — the global
columns live in their own array, indexed by bin, never enlarging a block.

Chain globals are never seeded in this pass (Section 4 gives their closed forms instead): an
`IRF` width or a background level does not enter `P` at all, so seeding it here would spend a
partial on an identically-zero derivative and grow `C` for nothing.

A variable may be bound to both a plasma leaf/in-kernel parameter (making it a kernel global) and
a chain-stage parameter (making it also a chain global) at once — D11, JT8. Nothing in the
assembler special-cases this: the variable's theta index simply appears in both
`kernel_global_idx` and `chain_global_idx`, and `jvp!`/`jtvp!` (Section 5) accumulate both
lists independently into the same output, so the column is their sum by construction.

## 4. The chain globals' closed forms

A chain global changes `M` at fixed `P` (it is a parameter of a chain-stage effect: an IRF's
`sigma`, `Sensitivity.scale`, or a background coefficient). Writing `Q = K_λ P K_tᵀ` (the
pre-`Sensitivity` streak) and `s` the scaled wavelength `(λ - λ₀)/Δλ`,
`λ₀ = (λ_first+λ_last)/2`, `Δλ = (λ_last-λ_first)/2` (the same scaling
`PolynomialBackground`/`RationalBackground` use):

| Parameter | `∂M/∂c` | Derivation |
|---|---|---|
| `Sensitivity.scale` | `curve .* Q` | `M ⊃ (scale·curve).*Q`; differentiate the product at fixed `curve`, `Q` |
| `ConstantBackground.level` | `1` (every entry) | `M ⊃ level`, added uniformly |
| `PolynomialBackground.coefficients[m]` | `s^(m-1)` (every column) | `B[l] = Σ_m c_m s_l^{m-1}`; one term differentiated |
| `RationalBackground.numerator[m]` | `s^(m-1) / denom` | `B = N(s)/D(s)`, `∂B/∂N_m = s^{m-1}/D(s)` |
| `RationalBackground.denominator[m]` | `-(N(s)·s^m) / D(s)²` | quotient rule, `∂D/∂d_m = s^m` |
| `ImageBackground.scale` | `image` | `M ⊃ scale · image` |
| `SpectralIRF.sigma` (`GaussianKernel`) | `curve_g .* ((∂K_λ/∂σ) P K_tᵀ)` | differentiate `K_λ` alone, `curve_g = g` |
| `TemporalIRF.sigma` (`GaussianKernel`) | `curve_g .* (K_λ P (∂K_t/∂σ)ᵀ)` | differentiate `K_t` alone |

**`∂K/∂σ` is obtained by forward-mode AD through `_convolution_matrix` itself**
(`ForwardDiff.partials.(_convolution_matrix(GaussianKernel(Dual(σ,1)), axis), 1)`), never
hand-differentiated: `_convolution_matrix` is generic in its element type (a dual `σ` gives a
dual matrix, XP12), and the renormalization step (`K[:,l'] ./= sum(K[:,l'])`) is nonlinear in
`σ`, so a closed form would have to re-derive the quotient rule through that step by hand for no
benefit — one dual matrix build costs about the same as one plain one.

**`Sensitivity.scale`'s row uses `curve` (the response curve alone, `1` with no curve), not
`g = scale·curve`** — since `∂(scale·curve·Q)/∂scale = curve·Q` at fixed `curve`, dividing the
`scale` factor back out. Every *other* row above holds `scale` fixed and uses the full `g`
(`response` in the code), since `scale` is not what is being differentiated there.

Each closed form above is w.r.t. the effect's own (SI, "native") parameter value `c`; the
column stored in `chain_globals` is `∂M/∂c · dc/dθ`, with `dc/dθ = _dx_dy(c, lower, upper)` (D3)
supplying the logit factor that the per-bin dual pass gets automatically but a closed form does
not. `_build_chain_globals` supports a chain-stage binding only on a directly fitted variable
(not a dependent) — no test problem of this phase binds a dependent to a chain-stage parameter
(D46), and doing so would need the chain rule through the dependent's own expression tree rather
than one scalar `dc/dθ`; this is refused with a loud error rather than silently computed wrong.
**A frozen variable has no column** (Phase 4.I spec D11): it has no θ entry, and through `c43e7d4`
one bound to a chain-stage parameter made `_build_chain_globals` read `theta[0]`; it is skipped.

## 5. `jvp!` and `jtvp!`: the transposes, derived

`jvp!` computes `out = J v` directly from the chain of Section 1-2: build the ideal streak's
image of `v`,

```
    Q_v[:,j] = blocks[j] · v[bin_entries[j]] + Σ_k v[kernel_global_idx[k]] · kernel_globals[:,j,k]
```

then push it through the forward chain exactly as `M` itself is built,
`W = g .* (K_λ Q_v K_tᵀ)`, add the chain globals' direct contributions
(`W += Σ_c v[chain_global_idx[c]] · chain_globals[c]`, since each already **is** `∂M/∂θ_c`), and
divide by `σ` at the mask-true pixels.

`jtvp!` computes `out = Jᵀ w` and must be `jvp!`'s exact adjoint (JT2), which forces its own
matrix orientations rather than letting them be chosen freely. Starting from
`w`, undo the `1/σ` divide into a dense `Z` (`Z[l,j] = w_k/σ[l,j]` at mask-true pixels, `0`
elsewhere) — the adjoint of "gather `npixels` entries and divide by `σ`" is "scatter and divide
by `σ`", since a diagonal (elementwise) operator is its own transpose. The chain-global
contribution is immediate: since `⟨v_c · chain_globals[c], Z⟩ = v_c · ⟨chain_globals[c], Z⟩`,
`jtvp!`'s chain-global row is `dot(chain_globals[c], Z)`, the adjoint of "multiply by a scalar and
add" being "take the inner product".

For the convolution half, write the forward map as three linear operators in sequence,
`Z ↦ Y = Z .* g` (`g` diagonal, so this is a diagonal operator), `Y ↦ Kλᵀ` is **not** applied here
— forward is `W = g .* (Kλ Q Ktᵀ)`, i.e. reading right to left on `Q`: first left-multiply by
`Kλ`, then right-multiply by `Ktᵀ`, then scale by `g`. Adjoints compose in **reverse order**, and
each factor's adjoint is its own transpose:

```
    forward:  Q  --(left-mult Kλ)-->  Kλ Q  --(right-mult Ktᵀ)-->  Kλ Q Ktᵀ  --(scale g)-->  W
    adjoint:  U  <--(left-mult Kλᵀ)--  Kλᵀ Y  <--(right-mult Kt)--  Y  <--(scale g, self-adjoint)--  Z
```

So, reading the adjoint forward from `Z`: **first** scale by `g` (self-adjoint, `Y = Z .* g`);
**then** undo the *last* forward step first — the forward step that right-multiplied by `Ktᵀ`
is undone by right-multiplying by `(Ktᵀ)ᵀ = Kt`; **then** undo the forward step that
left-multiplied by `Kλ`, by left-multiplying by `Kλᵀ`:

```
    U = Kλᵀ · Y · Kt
```

**Both matrices change which side they multiply from, relative to the forward direction**
(forward: `Kλ` on the left un-transposed, `Ktᵀ` on the right; adjoint: `Kλᵀ` on the left,
`Kt` un-transposed on the right). Getting the order of composition backwards, or transposing the
wrong factor, produces an operator that is *plausible* — it has the right shape and passes any
test built on a **symmetric** kernel, since a symmetric `K` satisfies `K = Kᵀ` and the bug is
invisible — and fails only where the kernel is asymmetric. This is exactly why P3's `TemporalIRF`
is a deliberately asymmetric `TabulatedKernel((-1,0,2),(0.2,0.5,0.3))` (JT3): a swapped transpose
passes JT2 on a symmetric-kernel problem and fails JT3.

`out`'s per-bin block and kernel-global rows are then read off `U` the same way `jvp!`'s were
built: `out[bin_entries[j]] += blocksⱼᵀ U[:,j]` (the adjoint of a matrix-vector product is the
transpose matrix-vector product) and `out[kernel_global_idx[k]] += dot(kernel_globals[:,:,k], U)`
(the adjoint of "broadcast-multiply by a scalar and sum" is the inner product, as for the chain
globals).

## 6. Sizes

For the full-size EPW problem (512 × 256) with three trace variables and two scalar globals
(D17's own worked example): `blocks` is 256 matrices of `512 × 3` doubles ≈ 3.1 MB;
`kernel_globals` is `512 × 256 × 2` doubles ≈ 2.1 MB; `K_λ` and `K_t` (shared, not part of
`FitJacobian`'s per-bin storage) are 2.10 MB and 0.52 MB respectively (Phase 2's own measurement).
A dense `J` would be `131072 × 770` doubles ≈ 807 MB. The 512 × 256 EPW problem's actual measured
sizes are in the chunk 6 hand-off report (`dev/reports/phase3_report.md`), alongside `D43`'s other
measurements.

## 7. The joint products (Phase 4.A, Step 4.A.3a, D34)

A `JointFitProblem` holds one `FitProblem` per channel `c = 1 … C` sharing one `ParameterSpec`
and, at most, one `Regularizer`. Each channel's own residual rows are `r_c = (M_c − D_c)/σ_c`
over its own `pixel_index`, exactly as a single-channel `FitProblem` computes them (D15, D16) --
**a channel's `FitJacobian` is built by the unmodified single-channel `jacobian` function**, with
only the channel filter of Section 3's box (below) telling it which of `pspec.bindings` are its
own. Writing `w_c` for the channel's weight (`1.0` unless the deck or the caller says otherwise,
OQ-35):

```
r_full = [ √w_1 r_1 ; … ; √w_C r_C ; √(Λ w_t) (op (u − ref))_r / N_r … ]
𝒪      = Σ_c w_c χ²_c + Λ R
J·v    = [ √w_1 J_1 v ; … ; √w_C J_C v ; J_R v ]
Jᵀ·w   = Σ_c √w_c J_cᵀ w_c + J_Rᵀ w_R
```

`J_R`/`J_Rᵀ` are the regularizer's own closed-form rows (Section 4), computed once by
`_regularizer_closed_form(jp.regularizer, jp.pspec, θ)` -- **the same call a single-channel
problem's own `jacobian` makes**, so the regularizer's contribution to every product is bit-for-
bit the code already proven in Sections 1-5, run once instead of once per channel.

**Five rules keep this both correct and bitwise-compatible with the single-channel code**
(D34; `src/inversion/residuals.jl`, `src/inversion/jacobian.jl`):

1. A channel with `n_c = 0` (fully masked) is skipped in every product. This is an optimization,
   never a correctness requirement: every one of that channel's contributions is `+0.0` regardless
   (`(M−D)/σ` is `+0.0` when `M == D`, and nothing here forms a signed zero any other way), so
   including a masked channel's zero contribution would not change a single bit -- skipping it
   only saves the wasted `measure`/IRF work.
2. `jvp!`/`residuals!` write each live channel directly into its own disjoint row range
   (`jp.row_ranges[c]`), so there is no "first vs. later channel" question there. `jtvp!` and
   `_gauss_newton_diagonal` accumulate into **one shared** output, so the **first contributing
   channel writes directly and later ones accumulate through a temporary** -- again an
   optimization (one fewer `fill!`), not a correctness requirement, since `0.0 + x === x` for
   every finite, non-negative-zero `x` that arises here.
3. `√w_c` is applied only when `w_c ≠ 1.0` (`1.0 * x === x` for every `x`), so a channel with the
   default weight contributes rows and columns that are **bit-identical** to the single-channel
   `FitProblem`'s own.
4. The regularizer's part of every product is computed by exactly the single-channel code
   (above), never re-derived, so a change of order or reassociation cannot creep in between the
   two.
5. Where two channels both contribute to one sum (`jtvp!`, `_gauss_newton_diagonal`, and the
   pixel count `Σ n_c`), they are visited in `jp.channels`'s own order, so a result depends only
   on that order, never on anything incidental to how the caller built the tuple.

**Why the masked-channel test (D35) is bitwise.** With `w_c = 1` and every channel but one (say
channel 1) fully masked: channel 1's own contribution to every product above is produced by
calling channel 1's `FitProblem` methods with the *exact* arguments (`experiment`, `pspec`, `data`,
`theta`) a plain single-channel `FitProblem` over the same channel would use, and rule 3 means no
`√w` multiplication ever touches those values; every masked channel contributes nothing (rule 1);
and the regularizer's rows are the same closed-form call either way (rule 4). So `θ*`, `χ²` and
every product the optimizers touch are bit-for-bit the single-channel answer -- Pitfall 22 and
Pitfall 28 exist because *any* deviation from this (a regularizer re-derived per channel, a
`√1.0` actually multiplied, a masked channel's zero written through a different code path) reads
as "close enough" in every other test and only shows up as a flipped bit here.

**The per-channel diagnostics (D36).** `ChannelSummary.chi2` is the channel's own *unweighted*
`chi2(channel, θ)` (never the weighted row scaling above), and `FitResult.chi2`/the optimizer's
own `𝒪` are the *weighted* total `Σ w_c χ²_c` -- the two are deliberately different numbers
(Pitfall 26). `nparams` (`p_c`) is read off `pspec.bindings` alone: `pspec.ntheta` minus the
global (scalar) entries whose variable's bindings are *every one* an `:effect` binding of a
different channel; a per-bin entry always counts, since a time-varying variable can only reach
the one shared plasma every channel sees.
