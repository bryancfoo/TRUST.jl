# average_regions: the first function in the package that averages, rebins or resamples a
# StreakData along either axis. dev/phase4A_instructions.md Section 7, D13, D14, D16 (amended
# 2026-09-23), OQ-34. Sits beside warp.jl: dewarp is a bilinear 2-D warp correction on a fixed
# grid, this is a 1-D average onto a user-chosen coarser one.

"""
    AveragingRegion

One averaging region of the fitted time axis ([OQ-34](../../dev/open_questions.md#oq-34)): `lo`
and `hi` in SI (s), `nbins` the number of output bins the window `[lo, hi]` is divided into
evenly, `interval` the name of a `[fit.intervals]` entry this region also declares (`""` when it
names none).

Built with `AveragingRegion(; nbins, interval = "", t_range_ns | t_range_s)`, mirroring
`FitInterval`'s own boundary keyword exactly. Both bounds must be finite, `hi > lo`, and
`nbins >= 1`, each a loud error otherwise.

Spec D13.
"""
struct AveragingRegion
    lo::Float64        # s
    hi::Float64        # s
    nbins::Int
    interval::String   # "" when the region names no interval
end

const _AVERAGING_REGION_RANGE_KEYS = (t_range_ns=:t_ns, t_range_s=:t_s)

function AveragingRegion(; nbins::Integer, interval::AbstractString="", kwargs...)
    present = Tuple(k for k in keys(_AVERAGING_REGION_RANGE_KEYS) if haskey(kwargs, k))
    length(present) == 1 || throw(ArgumentError(
        "AveragingRegion: expected exactly one of $(keys(_AVERAGING_REGION_RANGE_KEYS)); got $present"))
    key = present[1]
    rng = kwargs[key]
    length(rng) == 2 || throw(ArgumentError(
        "AveragingRegion: \"$key\" must give exactly [lo, hi]; got $(collect(rng))"))
    factor = Float64(getfield(getfield(ACCEPTED_UNITS, :t), _AVERAGING_REGION_RANGE_KEYS[key]))
    lo, hi = Float64(rng[1]) * factor, Float64(rng[2]) * factor
    (isfinite(lo) && isfinite(hi)) || throw(ArgumentError(
        "AveragingRegion: \"$key\" must be finite; got $(collect(rng))"))
    hi > lo || throw(ArgumentError(
        "AveragingRegion: hi must be greater than lo; got lo=$lo, hi=$hi (SI)"))
    nbins >= 1 || throw(ArgumentError("AveragingRegion: nbins must be >= 1; got $nbins"))
    return AveragingRegion(lo, hi, Int(nbins), String(interval))
end

function Base.show(io::IO, r::AveragingRegion)
    print(io, "AveragingRegion(", t_from_si(r.lo, :t_ns), " to ", t_from_si(r.hi, :t_ns),
        " ns, nbins=", r.nbins)
    isempty(r.interval) || print(io, ", interval=\"", r.interval, "\"")
    print(io, ")")
end

"""
    average_regions(sd::StreakData, regions::AbstractVector{AveragingRegion};
                    min_usable_fraction::Real = 0.5,
                    t_shift_ns | t_shift_s = 0) -> StreakData

Averages `sd`'s image, σ and mask onto the time axis `regions` describes, one output bin per
window ([OQ-34](../../dev/open_questions.md#oq-34); spec D13, D14). `sd.lambda` is copied through
unchanged -- only the time axis is averaged, and the returned streak's `lambda` is `sd.lambda`.

`regions` must be non-empty, sorted and non-overlapping (`regions[k].lo >= regions[k-1].hi` for
every `k > 1`), a loud error otherwise naming both regions -- this is what keeps the output axis
strictly increasing, which `StreakData` requires. `min_usable_fraction` (`f_min`, default `0.5`)
must be in `[0, 1]`. `t_shift_ns | t_shift_s` is the channel's fixed timing offset `Δt`, resolved
through `_si_value(:t_shift, kwargs, 0.0)` so that giving neither keyword means zero and giving
both is the unit table's own loud error.

**The formulas (D14).** With `t_j' = sd.t[j] + Δt` the shifted source pixel centres, a region
`[t_a, t_b]` with `n` bins and `h = (t_b - t_a)/n` has windows `W_k = [t_a+(k-1)h, t_a+kh)` for
`k = 1 … n-1` and `W_n = [t_a+(n-1)h, t_b]` for the last, **closed at the top** -- unless the next
region starts exactly at `t_b`, in which case `W_n` is half-open too and a pixel at the shared
endpoint belongs to the next region's first window instead (amended 2026-09-23, so that two
adjacent regions do not both claim the shared pixel). A source pixel in no window is dropped. For
`W_k` with `|W_k| = 0` (no source pixel at all, a region finer than the pixel grid), a loud error
names the region and the window index.

The axis is one number per window over **all** of `W_k`'s pixels, masked or not,
`t̄_k = mean(t_j' for j in W_k)` -- never row-dependent, since a mask-dependent axis would be a
different streak per row and cannot be a `StreakData` (Pitfall 4). Per row `l`, with
`U_k,l = {j ∈ W_k : mask[l,j]}` (every pixel of `W_k` when `sd.mask === nothing`),
`D̄_k,l = mean(D_l,j for j in U_k,l)` and `σ̄_k,l = sqrt(Σ σ_l,j²) / |U_k,l|` -- dividing by the
count, not its square root, since the frozen σ of the averaged bin is the averaged bin's own
([OQ-14](../../dev/open_questions.md#oq-14)) and non-overlapping windows correlate no noise across
output bins. `mask_k,l = |U_k,l| >= f_min * |W_k|`, **except** a window with no usable pixel in a
row (`|U_k,l| = 0`), which gets `D̄ = 0`, `σ̄ = 1`, `mask = false` -- `dewarp`'s own unusable-pixel
convention (`src/data/warp.jl`), taking precedence over the `f_min` rule so that `f_min = 0` does
not produce an unmasked 0/0 bin.

Summation is a plain increasing-`j` `for` loop, never `sum` over a view (which reassociates), so
the result is reproducible bit for bit across runs and thread counts.

`average_regions` always writes a mask (the only place the usable-fraction rule is recorded), and
writes `sigma` only when `sd.sigma !== nothing`. It writes nine `meta` keys under a `regions_`
prefix (D16, amended 2026-09-23 from `regions/`, since `write_streak` stores each meta key as the
JLD2 path `meta/<key>`, where `/` is a group separator): `regions_source_t` (the source axis in s,
before the shift), `regions_t_shift`, `regions_min_usable_fraction`, `regions_starts`,
`regions_ends`, `regions_nbins`, `regions_intervals`, `regions_window_counts` (`|W_k|`, one per
output bin) and `regions_usable_counts` (`|U_k,l|`, `N_λ × N_t_out`). A source `meta` already
carrying a key beginning `regions_` is a loud error, since these nine describe an operation that
must not be silently reapplied.

Spec D13, D14, D16.
"""
function average_regions(sd::StreakData, regions::AbstractVector{AveragingRegion};
    min_usable_fraction::Real=0.5, kwargs...)
    dt = _si_value(:t_shift, kwargs, 0.0)
    _reject_extra_kwargs(kwargs, (:t_shift_ns, :t_shift_s))

    fmin = Float64(min_usable_fraction)
    (0.0 <= fmin <= 1.0) || throw(ArgumentError(
        "average_regions: min_usable_fraction must be in [0, 1]; got $fmin"))

    isempty(regions) && throw(ArgumentError("average_regions: regions must be non-empty"))
    for k in 2:length(regions)
        regions[k].lo >= regions[k-1].hi || throw(ArgumentError(
            "average_regions: regions must be sorted and non-overlapping; region $k " *
            "(lo=$(regions[k].lo) s) starts before region $(k-1) ends (hi=$(regions[k-1].hi) s)"))
    end

    for k in keys(sd.meta)
        startswith(k, "regions_") && throw(ArgumentError(
            "average_regions: sd.meta already has a key \"$k\" beginning \"regions_\"; " *
            "average_regions would silently reapply over it"))
    end

    tshift = sd.t .+ dt
    nl = length(sd.lambda)
    n_out = sum(r.nbins for r in regions)

    has_sigma = sd.sigma !== nothing
    has_mask = sd.mask !== nothing

    t_out = Vector{Float64}(undef, n_out)
    image_out = Matrix{Float64}(undef, nl, n_out)
    sigma_out = has_sigma ? Matrix{Float64}(undef, nl, n_out) : nothing
    mask_out = Matrix{Bool}(undef, nl, n_out)
    window_counts = Vector{Int}(undef, n_out)
    usable_counts = Matrix{Int}(undef, nl, n_out)

    nregions = length(regions)
    starts = Vector{Float64}(undef, nregions)
    ends = Vector{Float64}(undef, nregions)
    nbins_v = Vector{Int}(undef, nregions)
    intervals_v = Vector{String}(undef, nregions)

    out_k = 0
    for (ridx, region) in enumerate(regions)
        starts[ridx] = region.lo
        ends[ridx] = region.hi
        nbins_v[ridx] = region.nbins
        intervals_v[ridx] = region.interval

        lo, hi, n = region.lo, region.hi, region.nbins
        h = (hi - lo) / n
        next_starts_here = ridx < nregions && regions[ridx+1].lo == hi

        for k in 1:n
            out_k += 1
            is_last = k == n
            wlo = lo + (k - 1) * h
            whi = is_last ? hi : lo + k * h
            closed_top = is_last && !next_starts_here

            first_j = searchsortedfirst(tshift, wlo)
            last_j = closed_top ? searchsortedlast(tshift, whi) : (searchsortedfirst(tshift, whi) - 1)
            nw = max(last_j - first_j + 1, 0)
            window_counts[out_k] = nw
            nw == 0 && throw(ArgumentError(
                "average_regions: region $ridx window $k of $n has no source pixel in " *
                "[$wlo, $whi$(closed_top ? "]" : ")") (s); a region finer than the pixel grid"))

            tsum = 0.0
            for j in first_j:last_j
                tsum += tshift[j]
            end
            t_out[out_k] = tsum / nw

            for l in 1:nl
                nu = 0
                dsum = 0.0
                ssum = 0.0
                for j in first_j:last_j
                    used = !has_mask || sd.mask[l, j]
                    used || continue
                    nu += 1
                    dsum += sd.image[l, j]
                    has_sigma && (ssum += sd.sigma[l, j]^2)
                end
                usable_counts[l, out_k] = nu
                if nu == 0
                    image_out[l, out_k] = 0.0
                    has_sigma && (sigma_out[l, out_k] = 1.0)
                    mask_out[l, out_k] = false
                else
                    image_out[l, out_k] = dsum / nu
                    has_sigma && (sigma_out[l, out_k] = sqrt(ssum) / nu)
                    mask_out[l, out_k] = nu >= fmin * nw
                end
            end
        end
    end

    meta = copy(sd.meta)
    meta["regions_source_t"] = Vector{Float64}(sd.t)
    meta["regions_t_shift"] = Float64(dt)
    meta["regions_min_usable_fraction"] = fmin
    meta["regions_starts"] = starts
    meta["regions_ends"] = ends
    meta["regions_nbins"] = nbins_v
    meta["regions_intervals"] = intervals_v
    meta["regions_window_counts"] = window_counts
    meta["regions_usable_counts"] = usable_counts

    return StreakData(; lambda_m=sd.lambda, t_s=t_out, image=image_out, sigma=sigma_out,
        mask=mask_out, meta=meta)
end
