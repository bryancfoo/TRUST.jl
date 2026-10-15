# slice_scatter: the per-slice fine-grid joint scatter -- Step 4.A.4b, chunk 10 of Phase 4.A.
# Spec D51, dev/phase4A_instructions.md Section 10.2. Needs Deck and build_problem's three D50
# stages (_read_channels, _average_channels, _assemble_problem), hence included after
# config/deck.jl.

"""
    slice_scatter(deck::Deck, result::FitResult; optimizer::AbstractOptimizer = LevenbergMarquardt(),
                 tol::Real = 1e-6) -> ScatterResult

The per-slice statistical uncertainty report (D51, [OQ-24](../../dev/open_questions.md#oq-24)
item 3): both cameras averaged, not interpolated, onto a shared fine grid of nested windows inside
each coarse `[[fit.regions]]` bin; each fine window fitted **jointly** (every camera; every free
time-varying plasma variable free, every variable that is not time-varying held at `result`'s
value, Pitfall 42) on its own, unregularized (`Λ = 0`); per coarse bin, the detrended spread of
the window estimates (`_detrended_spread`), **never divided by `sqrt(m)`** (Pitfall 40). With
`Λ = 0`, the instrument fixed and those variables held, this decouples exactly into one
independent one-bin joint fit per fine window -- the full-resolution unregularized fit of OQ-24
item 3 with the same variables held, fitted window by window instead of column by column, **and
without a `TemporalIRF`**, which a one-bin window cannot carry (Pitfall 41; corrected 2026-09-26,
Phase 4.B spec D4: this said "every plasma variable free" and claimed the equivalence
unconditionally).

`result` must be `deck`'s own regularized fit (θ\\*), normally polished; every variable that is
**not time-varying** is held at `result`'s own value in every window (Pitfall 42) -- only free
**time-varying** variables (`vary && time_varying`) get a scatter.

**Refuses** (`ArgumentError`, the quoted substring):
- no `[[fit.regions]]`: `"slice_scatter: the deck has no [[fit.regions]]"` (Pitfall 43);
- `[fit.intervals]` non-empty, or any region naming an interval:
  `"slice_scatter: intervals are not supported"`;
- a `[fit.hold]` table (Phase 4.I spec D12): `"slice_scatter: held series ([fit.hold]) are not
  supported"`. A `[fit.start]` table is ignored: every window starts from `result`'s own values;
- `result` was not fitted from `deck` (its θ's length does not match what `deck`'s own variables
  and regions imply, or `result.pspec.axis` does not match the coarse axis this deck's own
  channels average to): `"slice_scatter: result was not fitted from this deck"`.

**The fine grid** (Pitfall 39). `Delta_max`, the largest slice spacing of any camera on its own
recorded (post-crop, pre-averaging) axis; per region with bin width `w = (hi - lo) / nbins`,

    m = floor(Int, (w / Delta_max) * (1 + 1e-9))

fine windows per coarse bin -- the `(1 + 1e-9)` guard, since a uniform recorded axis's largest
spacing can land a whisker above its true value (`1.0000000000000007e-10` s, not `1e-10`) and an
unguarded `floor` would give one window short. The fine region is the same `[lo, hi]` with
`nbins * m` bins, so fine windows nest exactly inside coarse bins. Every camera is averaged onto
it with `_average_channels`; because a fine window is at least `Delta_max` wide, every window
holds at least one slice of every camera (`average_regions` refuses an empty window loudly
anyway).

**What the one-bin fits do not model** (pitfall 41): a temporal IRF becomes the identity on a
one-bin axis, so the decoupling into independent windows is exact only without one -- the same
approximation OQ-24 item 3's "each detector slice fitted on its own" makes, stated and accepted
for this phase, not corrected.

**Each window's data and model** (Phase 4.M spec D13, [OQ-63](../../dev/open_questions.md#oq-63)
item 5): the returned `ScatterResult` also keeps, per channel, column `j` of `data`, `sigma` and
`mask` the window problem's own (its one-bin averaged image, σ and mask) and of `model`
`measure(pol.result.experiments[c])`, the model at the window's polished solution, one evaluation
per window; and `chi2[j]`, that solution's χ². Nothing else in the loop changes, so every value the
scatter reported before is bitwise what it was.

Spec D51.
"""
function slice_scatter(deck::Deck, result::FitResult;
    optimizer::AbstractOptimizer=LevenbergMarquardt(), tol::Real=1e-6)
    s = deck.sections

    s["fit_regions"] === nothing && throw(ArgumentError(
        "slice_scatter: the deck has no [[fit.regions]]"))
    isempty(s["fit_intervals"]) || throw(ArgumentError(
        "slice_scatter: intervals are not supported"))
    for region in s["fit_regions"]
        isempty(region["interval"]) || throw(ArgumentError(
            "slice_scatter: intervals are not supported"))
    end
    get(s, "fit_hold", nothing) === nothing || throw(ArgumentError(
        "slice_scatter: held series ([fit.hold]) are not supported"))

    chan_list = _channel_list(s)
    nchannels = length(chan_list)
    regions = _build_averaging_regions(s["fit_regions"])

    raw, probes, collections = _read_channels(s, chan_list, true)
    coarse = _average_channels(raw, chan_list, regions)

    # "result was not fitted from this deck": length(result.theta) against this deck's own
    # implied ntheta (from its variables and regions, without assembling the full plasma and
    # instrument chain, which the axis check below makes unnecessary), and the coarse axis.
    nt = length(coarse[1].t)
    vars = _build_variables(s["fit_variables"], Dict{String,Int}())
    _, _, _, _, _, _, ntheta = _build_index_map(vars, FitInterval[], nt)
    (length(result.theta) != ntheta || result.pspec.axis != coarse[1].t) && throw(ArgumentError(
        "slice_scatter: result was not fitted from this deck"))

    # D51 step 3: the fine grid, nested inside the coarse regions (Pitfall 39).
    Delta_max = maximum(maximum(diff(r.t)) for r in raw)

    m = Vector{Int}(undef, length(regions))
    fine = Vector{AveragingRegion}(undef, length(regions))
    for (ridx, r) in enumerate(regions)
        w = (r.hi - r.lo) / r.nbins
        m_r = floor(Int, (w / Delta_max) * (1 + 1e-9))
        m_r < 1 && throw(ArgumentError(
            "slice_scatter: region $ridx's bins are narrower than the largest slice spacing"))
        m[ridx] = m_r
        fine[ridx] = AveragingRegion(r.lo, r.hi, r.nbins * m_r, "")
    end

    fine_data = _average_channels(raw, chan_list, fine)
    t_fine = fine_data[1].t
    n_fine = length(t_fine)

    counts = Matrix{Int}(undef, n_fine, nchannels)
    for c in 1:nchannels
        counts[:, c] .= fine_data[c].meta["regions_window_counts"]
    end

    coarse_bin = Vector{Int}(undef, n_fine)
    j = 0
    coarse_offset = 0
    for (ridx, r) in enumerate(regions)
        m_r = m[ridx]
        for i in 1:(r.nbins*m_r)
            j += 1
            coarse_bin[j] = coarse_offset + cld(i, m_r)
        end
        coarse_offset += r.nbins
    end
    ncoarse = coarse_offset

    # D51 step 4: the window sections -- Lambda = 0, every non-time-varying variable held at
    # theta* (Pitfall 42). A time-varying variable already fixed (vary = false) stays as the
    # deck has it, untouched.
    sw = deepcopy(deck.sections)
    sw["regularizer"] = nothing
    # Phase 4.I spec D12: each window starts from result's own values (below), so a [fit.start] table
    # would only be read and discarded, and refused where a window reaches past its file's times.
    sw["fit_start"] = nothing
    free = String[]
    for v in result.pspec.variables
        if v.vary && v.time_varying
            push!(free, v.name)
        elseif !v.time_varying
            sw["fit_variables"][v.name]["vary"] = false
            sw["fit_variables"][v.name]["init"] = result.values[v.name][1]
        end
    end

    status = Vector{Symbol}(undef, n_fine)
    decrease = Vector{Float64}(undef, n_fine)
    values = Dict{String,Vector{Float64}}(name => Vector{Float64}(undef, n_fine) for name in free)
    near_bound = Dict{String,BitVector}(name => falses(n_fine) for name in free)
    # Phase 4.M spec D13 (OQ-63 item 5): each window's data, sigma, mask and model, one column per
    # window, and its chi2, kept for write_readable.
    win_data = [Matrix{Float64}(undef, length(fine_data[c].lambda), n_fine) for c in 1:nchannels]
    win_sigma = [Matrix{Float64}(undef, length(fine_data[c].lambda), n_fine) for c in 1:nchannels]
    win_mask = [Matrix{Bool}(undef, length(fine_data[c].lambda), n_fine) for c in 1:nchannels]
    win_model = [Matrix{Float64}(undef, length(fine_data[c].lambda), n_fine) for c in 1:nchannels]
    win_chi2 = Vector{Float64}(undef, n_fine)

    # D51 step 5: each fine window, its own one-bin joint fit, LM then polish.
    for jj in 1:n_fine
        k = coarse_bin[jj]
        onebin = Vector{StreakData}(undef, nchannels)
        for c in 1:nchannels
            d = fine_data[c]
            onebin[c] = StreakData(; lambda_m=d.lambda, t_s=[t_fine[jj]], image=d.image[:, jj:jj],
                sigma=d.sigma === nothing ? nothing : d.sigma[:, jj:jj],
                mask=d.mask === nothing ? nothing : d.mask[:, jj:jj])
        end
        problem_j, _, _ = _assemble_problem(sw, chan_list, onebin, probes, collections, AveragingRegion[])
        start = flatten(problem_j.pspec, [name => [result.values[name][k]] for name in free])
        fit_j = solve(problem_j, optimizer; start=start)
        pol = polish(problem_j, fit_j; tol=tol)
        status[jj] = fit_j.status
        decrease[jj] = pol.decreases[end]
        for name in free
            values[name][jj] = pol.result.values[name][1]
            near_bound[name][jj] = pol.result.near_bound[name][1]
        end
        for c in 1:nchannels
            ch = _channel_problem(problem_j, c)
            win_data[c][:, jj] = ch.data.image
            win_sigma[c][:, jj] = ch.sigma
            win_mask[c][:, jj] = ch.mask
            win_model[c][:, jj] = measure(pol.result.experiments[c])
        end
        win_chi2[jj] = pol.result.chi2
    end

    # D51 step 6: the detrended spread, per coarse bin and free variable, over that bin's own
    # fine windows -- never divided by sqrt(m).
    s_stat = Dict{String,Vector{Float64}}(name => Vector{Float64}(undef, ncoarse) for name in free)
    n_near_bound = Dict{String,Vector{Int}}(name => Vector{Int}(undef, ncoarse) for name in free)
    for k in 1:ncoarse
        J = findall(==(k), coarse_bin)
        for name in free
            s_stat[name][k] = _detrended_spread(t_fine[J], values[name][J])
            n_near_bound[name][k] = count(near_bound[name][J])
        end
    end

    return ScatterResult(copy(result.theta), m, t_fine, coarse_bin, counts, status, decrease,
        values, near_bound, s_stat, n_near_bound, win_data, win_sigma, win_mask, win_model, win_chi2)
end
