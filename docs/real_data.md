# Fitting a real OMEGA streak

> **Front matter (Step 3.8, 2026-09-16).** Windows 11, Julia 1.12.7; JLD2 0.6.6. Written
> alongside `src/data/warp.jl`. **No real OMEGA streak or `*_dewarp_data.h5` file was available
> this chunk** (`context/warp correction/` holds only the two raw HDF4 comb-grid calibration
> files and the collaborator's own Jupyter notebooks, never a produced `*_dewarp_data.h5`, and
> there is no Python on this machine to run the notebooks). Sections 1-2 below are demonstrated
> against a synthetic warp and streak, exactly as `test/data/test_warp.jl`'s DW1-DW7 do, and are
> tried and tested. **Section 3 onward describes the real-data workflow but is untried** — no
> fit of a real streak has been run. This is Step 3.8's own carried-forward outcome
> (`dev/reports/phase3_report.md`, Chunk 14), not a defect in this document.

## 1. What OTIS builds, and what it does not

A raw OMEGA streak image has a geometric distortion the collaborator's own comb-grid calibration
corrects: a rectangular lattice of dots imaged through the instrument lands displaced, and the
collaborator's pipeline finds the dot centroids, hand-prunes and sorts them into a lattice, and
bilinearly interpolates a **backward** map — for each pixel of a rectified destination grid,
where in the raw image to sample
([OQ-15](../dev/open_questions.md#oq-15), decided 2026-09-10, unchanged at 2026-09-11 except for
its timing, A27).

**OTIS never builds this map.** The dot-finding and lattice-fitting pipeline is hand-tuned per
file and belongs to whoever owns the camera calibration; the raw comb-grid files are HDF4, which
nothing maintained in Julia reads. What OTIS consumes is the collaborator's own output, and what
it provides is two things: `WarpMap`, holding that backward map as plain arrays, and
`dewarp(streak, warpmap) -> StreakData`, which applies it. Both are built and tested
(`src/data/warp.jl`, `test/data/test_warp.jl`, DW1-DW7) without any real data at all — a
synthetic warp with a known, invertible form is the oracle (DW2).

## 2. The synthetic walkthrough (tried and tested)

```julia
using OTIS

# A raw (still-warped) streak, however it was measured or synthesized.
sd = StreakData(lambda_m = lambda, t_s = t, image = raw_image, sigma = raw_sigma)

# The backward map: for destination pixel (i, j), (source_row[i,j], source_col[i,j]) is the
# fractional 1-based (row, column) = (wavelength index, time index) in sd.image to sample.
wm = WarpMap(source_row = source_row, source_col = source_col,
             lambda_m = destination_lambda, t_s = destination_t)

dewarped = dewarp(sd, wm)   # a StreakData on wm's own (destination_lambda, destination_t) axes
```

`dewarp` bilinearly resamples `sd.image` (and `sd.sigma`, `sd.mask` when given) at each
destination pixel's source point; `sigma` propagates in quadrature through the same four
bilinear weights, and the result is declared **diagonal-approximate** — bilinear resampling
correlates the noise of neighbouring pixels over about one pixel, and that off-diagonal
covariance is real and not modelled (`dewarp`'s own docstring carries this, not only
`meta["sigma_note"]`). A destination pixel is masked **out** (`mask = false`) unless all four of
its source pixels are inside `sd.image` **and** (when `sd.mask` is given) mask-true there — a
source point outside `sd.image` is excluded, never clamped back in. `sd.image` must be the same
size as `wm.source_row`: the warp map's indices are defined on that raster.

`test/data/test_warp.jl`'s DW1 (an identity warp reproduces the input bitwise), DW2 (a smooth,
small-amplitude synthetic warp inverts to `1e-3` relative, the bound bilinear interpolation sets)
and DW3 (a half-pixel shift gives the closed-form `σ_out = σ/2`) exercise exactly this path.

## 3. Reading the collaborator's own file — untried

The collaborator's notebook (`context/warp correction/create_deformation_map.ipynb`) emits
`*_dewarp_data.h5`, an HDF5 file (via `h5py`, not JLD2) holding two datasets: `deformation_map`,
an `N_λ × N_t × 2` array with `deformation_map[:,:,1]` the fractional source row and `[:,:,2]`
the fractional source column, and `time_indices`. `read_warpmap(path)` reads a file in this
layout with JLD2:

```julia
wm = read_warpmap("shot12345_dewarp_data.h5")
```

**Two things found reading the notebooks while building this function, neither exercised by any
test here, both worth knowing before pointing `read_warpmap` at a real file:**

1. **JLD2 opening a file `h5py` wrote is not guaranteed**
   ([OQ-15](../dev/open_questions.md#oq-15)). `read_warpmap` is tested only against a file OTIS
   itself wrote in this same layout (DW6); whether it opens a real `*_dewarp_data.h5` is a
   **reported**, not tested, outcome, and this chunk has nothing to report it against.
2. **`read_warpmap` also requires a `lambda_nm` dataset, which the collaborator's own file does
   not have**, and treats `time_indices` as an already-physical time axis in ns, which the
   collaborator's own `time_indices` is not: `create_deformation_map.ipynb` builds it as the
   **pixel column positions of the detected comb-grid calibration dots**
   (`time_indices[i] = mean(sorted_dots[:, i, 1])`), and `dewarp_streaks.ipynb` converts that to
   a real calibration with `ns_per_index = mean(0.506 ./ diff(time_indices))` (0.506 ns being
   the comb grid's own known dot spacing) before building a time axis from it. Neither that
   conversion nor an equivalent wavelength calibration is implemented in `read_warpmap` — doing
   so is a dot-spacing fit, arguably the camera-calibration owner's job just as the dot-finding
   itself is ([OQ-15](../dev/open_questions.md#oq-15)) — so **a real collaborator file is
   expected to fail here today**, naming the missing `lambda_nm` dataset, which is the honest
   "reported outcome" [OQ-15](../dev/open_questions.md#oq-15) anticipates rather than a defect.
   Flagged as a [propagate] item for Level 2/Level 1: a future `read_warpmap` revision may need
   the `ns_per_index` fit above, and an equivalent wavelength calibration, to open a real file
   correctly.

## 4. What the collaborator's own preprocessing notebook does that `dewarp` deliberately does not

Two things in `dewarp_streaks.ipynb` must not cross into a real fit:

- **A 20-pixel Gaussian smooth along time** (≈0.29 ns at this instrument's ~70 px/ns), applied
  in the notebook for display. Doing this to real data before fitting would double-count against
  `TemporalIRF`, which already models the instrument's own temporal response, and would
  correlate the noise well beyond the one-pixel bilinear correlation `dewarp` already declares.
  **Do not apply it.**
- **Division of the raw image by a frame the notebook calls a dark frame** (the notebook itself
  is inconsistent about this — commented out in one cell, used in four). A dark frame is
  **additive**, not multiplicative: if a dark frame is available and believed real, **subtract**
  it from the raw image as preprocessing, before `dewarp` is ever called. If the level is worth
  fitting rather than subtracting, `ImageBackground(image = dark, scale = 1.0)` is the model-side
  home for it in the effect chain, not a division baked into the data.

## 5. Masks, background, and the convective gain — untried

- **Saturated or dead regions** go through `StreakData.mask` (already built, Phase 2): `false`
  where a pixel should not be fitted. `dewarp` already carries a source mask through
  conservatively (Section 2); an additional mask for saturation or dead channels, found on the
  dewarped image, is combined with it (`.&`) before the streak is written for a deck to read.
- **A background model** — `ConstantBackground`, `PolynomialBackground`, `ImageBackground`,
  `RationalBackground` (`rat11`/`rat21`) — goes in through the effect chain, exactly as the three
  foo2026 reproductions use `ConstantBackground` (Step 3.7). Nothing new is needed for real data.
- **The convective gain.** With [OQ-32](../dev/open_questions.md#oq-32)'s probe numbers now
  known (cycle-averaged; 60 J/ns in a 100 µm circle), a real fit **may** want `gain = :convective`
  rather than `:none` — a few-percent effect at this probe energy. **Whether it does is the real
  fit's own finding, not something to assume**; the three foo2026 reproductions keep `gain =
  :none` throughout Phase 3 ([OQ-32](../dev/open_questions.md#oq-32)), and a worked example with
  the convective gain on is Phase 4's (`dev/phase3_instructions.md` Section 12).

## 6. Running a real fit — untried, but thread it

A deck for a real streak looks exactly like the three `examples/*.toml` decks (`dev/roadmap.md`
Step 3.6): `[data].file` points at the dewarped, masked, background-modelled streak; the rest is
the same fitted-variable, dependent, instrument, regularizer and optimizer machinery Phase 3
already builds and tests.

**Thread it.** The Jacobian's cost is the per-bin dual pass, and `foreach_bin!` parallelizes it
across `Threads.nthreads()`; Level 2 measured **3.4×** from `-t 8` on this workload (the
per-bin Jacobian, 39.8 ms → 11.8 ms single call, `dev/phase3_overall_summary.md`). A real fit,
run from the command line, should be started as

```
julia --project=. -t 8 -e 'using OTIS; fit("shot12345.toml")'
```

not a bare `julia --project=.` — the README and this document both say so for exactly this
reason.
