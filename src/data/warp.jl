# WarpMap and dewarp: the OMEGA streak-image geometric correction, built once per shot from the
# collaborator's own backward map. conventions.md Section 4; OQ-15; spec D36, Step 3.8.

using Printf: @printf

"""
    WarpMap

The collaborator's own **backward** map from a destination pixel to where in the raw (still
warped) source image to sample: for destination pixel `(i, j)`, `(source_row[i,j],
source_col[i,j])` is the fractional 1-based source-image location -- row the wavelength index,
column the time index, matching `StreakData`'s own `image` layout -- to sample **bilinearly**.
This is already `deformation_map[i, j] = (source_row, source_col)` in the collaborator's own
pipeline ([OQ-15](../../dev/open_questions.md#oq-15)), so nothing here inverts it.

Fields: `source_row`, `source_col` (`N_λ × N_t` matrices of fractional source indices, the same
size as the raw streak `dewarp` will sample from); `lambda` (m, the destination wavelength axis,
length `size(source_row, 1)`); `t` (s, the destination time axis, length `size(source_row, 2)`).

Built with `WarpMap(; source_row, source_col, lambda_nm | lambda_m, t_ns | t_s)` (rule 1.2's
boundary-unit keywords).

Spec D36.
"""
struct WarpMap
    source_row::Matrix{Float64}
    source_col::Matrix{Float64}
    lambda::Vector{Float64}
    t::Vector{Float64}
end

function WarpMap(; source_row, source_col, kwargs...)
    lambda_raw = _si_axis(:lambda, kwargs)
    t_raw = _si_axis(:t, kwargs)
    _reject_extra_kwargs(kwargs, (:lambda_nm, :lambda_m, :t_ns, :t_s))

    size(source_row) == size(source_col) || throw(ArgumentError(
        "WarpMap: source_row is $(size(source_row)) but source_col is $(size(source_col))"))
    nl, nt = size(source_row)
    length(lambda_raw) == nl || throw(ArgumentError(
        "WarpMap: lambda has length $(length(lambda_raw)) but source_row has $nl rows"))
    length(t_raw) == nt || throw(ArgumentError(
        "WarpMap: t has length $(length(t_raw)) but source_row has $nt columns"))

    return WarpMap(Matrix{Float64}(source_row), Matrix{Float64}(source_col),
        Vector{Float64}(lambda_raw), Vector{Float64}(t_raw))
end

"""
    Base.size(wm::WarpMap) -> (N_λ, N_t)

The destination grid's size, `size(wm.source_row)`.
"""
Base.size(wm::WarpMap) = size(wm.source_row)

function Base.show(io::IO, wm::WarpMap)
    nl, nt = size(wm)
    print(io, "WarpMap($nl x $nt)")
end

function Base.show(io::IO, ::MIME"text/plain", wm::WarpMap)
    nl, nt = size(wm)
    println(io, "WarpMap")
    @printf(io, "  destination: %d x %d (lambda x t)\n", nl, nt)
    @printf(io, "  lambda: %e to %e nm\n", lambda_from_si(wm.lambda[1], :lambda_nm),
        lambda_from_si(wm.lambda[end], :lambda_nm))
    @printf(io, "  t:      %e to %e ns\n", t_from_si(wm.t[1], :t_ns), t_from_si(wm.t[end], :t_ns))
end

"""
    _bilinear(row::Float64, col::Float64, nrows::Int, ncols::Int)
        -> (i0::Int, i1::Int, j0::Int, j1::Int, fi::Float64, fj::Float64, inbounds::Bool)

The four source-pixel indices and the two fractional weights bilinear interpolation needs to
sample the fractional 1-based source coordinate `(row, col)` from an `nrows x ncols` source
image: `i0 = floor(row)`, `fi = row - i0` (the row weight); `i1 = i0` when `fi == 0` (so an exact
integer coordinate, including the last row, needs no pixel beyond `i0` and cannot go out of
bounds looking for one -- `dewarp`'s identity-warp case, DW1) and `i1 = i0 + 1` otherwise; `col`
likewise. `inbounds` is `true` only when every index actually needed (`i0`, and `i1` when
`fi > 0`; `j0`, and `j1` when `fj > 0`) lies in `1:nrows` / `1:ncols` -- a source point outside
the image is reported as such, **never clamped** back into it (D36).

The weights this function's caller builds from `fi`/`fj` are `w00 = (1-fi)(1-fj)`,
`w01 = (1-fi)fj`, `w10 = fi(1-fj)`, `w11 = fi*fj` for corners `(i0,j0)`, `(i0,j1)`, `(i1,j0)`,
`(i1,j1)`; they sum to `1` by construction and are exactly `(1,0,0,0)` when `fi = fj = 0`.
"""
function _bilinear(row::Float64, col::Float64, nrows::Int, ncols::Int)
    i0 = floor(Int, row)
    j0 = floor(Int, col)
    fi = row - i0
    fj = col - j0
    i1 = fi > 0 ? i0 + 1 : i0
    j1 = fj > 0 ? j0 + 1 : j0
    inbounds = 1 <= i0 && i1 <= nrows && 1 <= j0 && j1 <= ncols
    return i0, i1, j0, j1, fi, fj, inbounds
end

"""
    dewarp(sd::StreakData, wm::WarpMap) -> StreakData

Resamples `sd` onto `wm`'s destination grid by bilinear interpolation of the collaborator's
**backward** map ([OQ-15](../../dev/open_questions.md#oq-15)): for destination pixel `(i, j)`,
samples `sd.image` (and `sd.sigma`, `sd.mask` when given) at the four source pixels around
`(wm.source_row[i,j], wm.source_col[i,j])` (`_bilinear`).

**`sigma` propagates in quadrature through the same four bilinear weights**,
`σ_out² = Σ_k w_k² σ_k²`, the variance of a weighted sum of independent pixels. **The result is
diagonal-approximate**: bilinear resampling correlates the noise of neighbouring output pixels
over about one pixel, and those off-diagonal covariance terms are real and are **not modelled**
here ([OQ-14](../../dev/open_questions.md#oq-14), [OQ-15](../../dev/open_questions.md#oq-15)) --
a caveat every dewarped streak's χ² inherits, since Phase 3's χ² treats σ as diagonal (D25).
`dewarp` records this in `meta["sigma_note"]` as well as here.

**The mask is conservative**: a destination pixel is used (`mask = true`) only when all four
source pixels its sample needs are inside `sd.image` **and**, when `sd.mask` is given, all four
are mask-true there. A destination pixel whose source point falls (even partly) outside
`sd.image` is excluded, **never clamped** to the nearest valid pixel.

**Two things from the collaborator's own preprocessing notebook are deliberately not done here**
([OQ-15](../../dev/open_questions.md#oq-15)): the notebook's 20-pixel Gaussian smooth along time
(≈0.29 ns), applied for display, which would double-count against `TemporalIRF` and correlate
the noise well beyond the one-pixel bilinear correlation already declared above; and the
notebook's division of the raw image by a suspected dark frame, which is **additive** and wants
**subtracting** -- as preprocessing before `dewarp` is ever called, not a multiplicative
correction folded in here.

`sd` must be `size(wm)` (`sd.image`'s size must equal `wm.source_row`'s: the warp map's backward
indices are defined on the same raster `sd.image` is), else a loud error naming both sizes (D39).
`dewarp` takes its axes from `wm`, not from `sd`: the returned `StreakData`'s `lambda`/`t` are
`wm.lambda`/`wm.t`. `sd.meta` is carried over, with `meta["dewarped"] = 1.0` and
`meta["sigma_note"]` set (superseding either key if already present, as `write_streak` does for
its own reserved keys, `data/io.jl`).

Spec D36.
"""
function dewarp(sd::StreakData, wm::WarpMap)
    size(sd) == size(wm) || throw(ArgumentError(
        "dewarp: the warp map is $(size(wm)) but the streak is $(size(sd))"))

    nl_src, nt_src = size(sd)
    nl, nt = size(wm)
    has_sigma = sd.sigma !== nothing
    has_mask = sd.mask !== nothing

    image = Matrix{Float64}(undef, nl, nt)
    sigma = has_sigma ? Matrix{Float64}(undef, nl, nt) : nothing
    mask = Matrix{Bool}(undef, nl, nt)

    for j in 1:nt, i in 1:nl
        row = wm.source_row[i, j]
        col = wm.source_col[i, j]
        i0, i1, j0, j1, fi, fj, inb = _bilinear(row, col, nl_src, nt_src)
        if !inb
            image[i, j] = 0.0
            has_sigma && (sigma[i, j] = 1.0)
            mask[i, j] = false
            continue
        end
        w00 = (1 - fi) * (1 - fj)
        w01 = (1 - fi) * fj
        w10 = fi * (1 - fj)
        w11 = fi * fj
        image[i, j] = w00 * sd.image[i0, j0] + w01 * sd.image[i0, j1] +
                      w10 * sd.image[i1, j0] + w11 * sd.image[i1, j1]
        if has_sigma
            sigma[i, j] = sqrt(w00^2 * sd.sigma[i0, j0]^2 + w01^2 * sd.sigma[i0, j1]^2 +
                                w10^2 * sd.sigma[i1, j0]^2 + w11^2 * sd.sigma[i1, j1]^2)
        end
        mask[i, j] = !has_mask || (sd.mask[i0, j0] && sd.mask[i0, j1] &&
                                    sd.mask[i1, j0] && sd.mask[i1, j1])
    end

    meta = copy(sd.meta)
    meta["dewarped"] = 1.0
    meta["sigma_note"] = "diagonal-approximate: bilinear resampling correlates the noise over about one pixel"

    return StreakData(; lambda_m=wm.lambda, t_s=wm.t, image=image, sigma=sigma, mask=mask, meta=meta)
end

"""
    read_warpmap(path::AbstractString) -> WarpMap

Reads a warp-correction file with JLD2: `deformation_map`, an `N_λ × N_t × 2` array with
`deformation_map[:,:,1]` the fractional source row and `deformation_map[:,:,2]` the fractional
source column (the collaborator's own backward map, named exactly this in
[OQ-15](../../dev/open_questions.md#oq-15)'s `*_dewarp_data.h5`); `time_indices`, the destination
time axis in ns, one entry per column; and `lambda_nm`, the destination wavelength axis in nm.

**JLD2's ability to open a file the collaborator's own Python pipeline wrote (plain HDF5 via
h5py, not JLD2) is not guaranteed** ([OQ-15](../../dev/open_questions.md#oq-15)):
`read_warpmap` is therefore tested only against a file OTIS itself wrote in this same layout
(DW6), and whether it opens a real `*_dewarp_data.h5` is a **reported**, not tested, outcome of
Step 3.8. If it does not, this loader stays as OTIS's own facility loader regardless (D36) --
the raw comb-grid HDF4 files behind it are out of scope forever
([OQ-15](../../dev/open_questions.md#oq-15)).

**`lambda_nm` is not one of the collaborator's own two datasets, and `time_indices` is not
already a physical time axis there either** (found reading `context/warp correction/`'s own
notebooks while building this function, Step 3.8, 2026-09-16): the collaborator's
`create_deformation_map.ipynb` builds `time_indices` as the **pixel column positions of the
detected comb-grid calibration dots** (`time_indices[i] = mean(sorted_dots[:, i, 1])`), not one
entry per destination time bin, and its `dewarp_streaks.ipynb` converts that to a physical
calibration with `ns_per_index = mean(0.506 ./ diff(time_indices))` (0.506 ns being the comb
grid's own known dot spacing, [OQ-15](../../dev/open_questions.md#oq-15)) before building a real
time axis from it; neither that conversion nor an equivalent wavelength calibration is
implemented here (out of this step's scope: the dot-finding and calibration fit belongs to
whoever owns the camera calibration, [OQ-15](../../dev/open_questions.md#oq-15)). This function
therefore reads `time_indices` **directly** as `t_ns`, correct only for a file already carrying
a physical axis under that name -- true of a file OTIS wrote for its own DW6 round trip, not of
the collaborator's own file, which also has no `lambda_nm` dataset at all. Reported to Level 2
as a [propagate] candidate: a future `read_warpmap` revision may need the dot-spacing fit above
to open a real file's `time_indices` correctly.

Spec D36.
"""
function read_warpmap(path::AbstractString)
    jldopen(path, "r") do f
        deformation_map = _require_dataset(f, "deformation_map", path)
        time_indices = _require_dataset(f, "time_indices", path)
        lambda_nm = _require_dataset(f, "lambda_nm", path)

        size(deformation_map, 3) == 2 || throw(ArgumentError(
            "read_warpmap: \"$path\" deformation_map's third dimension is " *
            "$(size(deformation_map, 3)); expected 2 (source row, source column)"))
        source_row = Matrix{Float64}(deformation_map[:, :, 1])
        source_col = Matrix{Float64}(deformation_map[:, :, 2])

        return WarpMap(; source_row=source_row, source_col=source_col,
            lambda_nm=Vector{Float64}(lambda_nm), t_ns=Vector{Float64}(time_indices))
    end
end
