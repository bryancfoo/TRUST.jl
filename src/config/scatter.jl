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
- `result` was not fitted from `deck` (its θ's length does not match what `deck`'s own variables,
  intervals, held series and regions imply, or `result.pspec.axis` does not match the coarse axis
  this deck's own channels average to): `"slice_scatter: result was not fitted from this deck"`.

A `[fit.start]` table is ignored: every window starts from `result`'s own values.

**Intervals and held series** (Phase 4.N spec D3 and D4, [OQ-64](../../dev/open_questions.md#oq-64)
items 1 and 2). `[fit.intervals]` (and a region naming an interval): a window outside a variable's
interval, by its coarse bin, fixes that variable at its `init`. A window takes its coarse bin's
membership, not its own time's: a variable is free in every window of a bin its interval holds,
started at `result`'s value there, and frozen at its deck `init` in every window of any other bin,
the value the regularized fit's trace carried there; its `values` are `NaN` and its `near_bound`
`false` in those windows, and its `s_stat`, `p16`, `median`, `p84` and `n_near_bound` `NaN`, `NaN`,
`NaN`, `NaN` and `0` in those bins. `[fit.hold]`: the held series evaluated at each window's averaged
time, the coarse bin's value beyond the file's range. The file is read once; a time-varying variable
held at a series of more than one value takes `_interpolate_series` at `t_fine[j]` when the window's
time lies within the file's time range and `result`'s value at the window's coarse bin (the value the
coarse fit held that bin at) otherwise; a one-value series, or a scalar variable, its one value. A
held variable with an interval is held only in the windows of its bins, frozen at `init` elsewhere.
`held` records each window's value.

**The quantiles** (Phase 4.N spec D6, OQ-64 item 3): per free variable and coarse bin, `p16`,
`median` and `p84`, the 16th, 50th and 84th percentiles of the bin's own windows' values by
Hyndman and Fan's type 7 (`Statistics.quantile`'s and numpy's), over every window whatever its
`status` (`_quantile_sorted`).

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

On a deck without intervals or held series every field the scatter returned before Phase 4.N is
bitwise what it was: the window fits are the same fits, and the quantiles are computed from the same
values.

Spec D51; Phase 4.N spec D3 to D6.
"""
function slice_scatter(deck::Deck, result::FitResult;
    optimizer::AbstractOptimizer=LevenbergMarquardt(), tol::Real=1e-6)
    s = deck.sections

    s["fit_regions"] === nothing && throw(ArgumentError(
        "slice_scatter: the deck has no [[fit.regions]]"))

    chan_list = _channel_list(s)
    nchannels = length(chan_list)
    regions = _build_averaging_regions(s["fit_regions"])

    raw, probes, collections = _read_channels(s, chan_list, true)
    coarse = _average_channels(raw, chan_list, regions)

    # "result was not fitted from this deck": length(result.theta) against this deck's own
    # implied ntheta (from its variables, intervals, held series and regions, without assembling the
    # full plasma and instrument chain, which the axis check below makes unnecessary), and the coarse
    # axis. Phase 4.N spec D3: the intervals are built as _assemble_problem builds them, the deck's
    # [fit.intervals] merged with the ones its regions declare, against the coarse axis; a variable
    # [fit.hold] names is not fitted, as _assemble_problem makes it.
    nt = length(coarse[1].t)
    synthesized = Dict{String,Any}()
    for region in regions
        isempty(region.interval) && continue
        synthesized[region.interval] = Dict{String,Any}("t_range_s" => Any[region.lo, region.hi])
    end
    ivals, interval_index = _build_intervals(merge(s["fit_intervals"], synthesized), coarse[1].t)
    hold = get(s, "fit_hold", nothing)
    held_names = hold === nothing ? String[] : String[nm for nm in hold["variables"]]
    vars = _build_variables(s["fit_variables"], interval_index)
    for (i, v) in enumerate(vars)
        v.name in held_names || continue
        vars[i] = FittedVariable(v.name; bounds=(v.lower, v.upper), init=v.init,
            time_varying=v.time_varying, vary=false, interval=v.interval)
    end
    _, _, _, _, _, _, ntheta = _build_index_map(vars, ivals, nt)
    (length(result.theta) != ntheta || result.pspec.axis != coarse[1].t) && throw(ArgumentError(
        "slice_scatter: result was not fitted from this deck"))

    # Phase 4.N spec D3 (OQ-64 item 1): whether each coarse bin lies in a variable's interval, every
    # bin for a variable without one or a scalar. A window takes its coarse bin's membership.
    member = Dict{String,BitVector}(v.name => (v.interval == 0 || !v.time_varying ? trues(nt) :
        BitVector([k in ivals[v.interval].bins for k in 1:nt])) for v in vars)

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

    # Phase 4.N spec D4 (OQ-64 item 2): [fit.hold]'s file, read once, and each window's held value: a
    # series of more than one value at a time-varying variable interpolated at the window's averaged
    # time inside the file's time range and the coarse bin's held value beyond it, otherwise the one
    # value; NaN in a window whose coarse bin lies outside the variable's interval (frozen at init).
    held = Dict{String,Vector{Float64}}()
    if hold !== nothing
        hfile = _read_series_file(hold["file"], "hold")
        time_varying = Dict{String,Bool}(v.name => v.time_varying for v in vars)
        for name in held_names
            haskey(hfile.values, name) || throw(ArgumentError(
                "build_problem: [fit.hold] names \"$name\", which file \"$(hold["file"])\" does not hold; " *
                "it holds $(sort(collect(keys(hfile.values))))"))
            series = hfile.values[name]
            row = fill(NaN, n_fine)
            for jj in 1:n_fine
                k = coarse_bin[jj]
                member[name][k] || continue
                t = t_fine[jj]
                row[jj] = !(time_varying[name] && length(series) > 1) ? series[1] :
                          first(hfile.axis) <= t <= last(hfile.axis) ?
                          _interpolate_series(hfile.axis, series, t) : result.values[name][k]
            end
            held[name] = row
        end
    end

    # D51 step 4: the window sections -- Lambda = 0, every non-time-varying variable held at
    # theta* (Pitfall 42). A time-varying variable already fixed (vary = false) stays as the
    # deck has it, untouched.
    sw = deepcopy(deck.sections)
    sw["regularizer"] = nothing
    # Phase 4.I spec D12: each window starts from result's own values (below), so a [fit.start] table
    # would only be read and discarded, and refused where a window reaches past its file's times.
    sw["fit_start"] = nothing
    # Phase 4.N spec D3, D4: no interval and no hold table in a window; every variable lives on the
    # one-bin axis, and the held values reach _assemble_problem through its held keyword.
    sw["fit_intervals"] = Dict{String,Any}()
    sw["fit_hold"] = nothing
    for (_, entry) in sw["fit_variables"]
        entry["interval"] = 0
    end
    free = String[]
    toggled = String[]
    for v in result.pspec.variables
        if v.vary && v.time_varying
            push!(free, v.name)
        elseif !v.time_varying
            sw["fit_variables"][v.name]["vary"] = false
            sw["fit_variables"][v.name]["init"] = result.values[v.name][1]
        end
        # fitted or held, with an interval: free (or held) in its bins' windows, frozen elsewhere
        (v.time_varying && !all(member[v.name]) && (v.vary || v.held !== nothing)) && push!(toggled, v.name)
    end
    # the names free in each coarse bin's windows, in free's order
    free_in = [String[name for name in free if member[name][k]] for k in 1:ncoarse]

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
        for name in toggled
            sw["fit_variables"][name]["vary"] = member[name][k]
        end
        held_j = hold === nothing ? nothing :
                 Dict{String,Vector{Float64}}(name => [held[name][jj]] for name in held_names if member[name][k])
        problem_j, _, _ = _assemble_problem(sw, chan_list, onebin, probes, collections, AveragingRegion[];
            held=held_j)
        start = flatten(problem_j.pspec, [name => [result.values[name][k]] for name in free_in[k]])
        fit_j = solve(problem_j, optimizer; start=start)
        pol = polish(problem_j, fit_j; tol=tol)
        status[jj] = fit_j.status
        decrease[jj] = pol.decreases[end]
        for name in free
            member[name][k] || (values[name][jj] = NaN; continue)
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
    # fine windows -- never divided by sqrt(m). Phase 4.N spec D6: beside it the 16th, 50th and 84th
    # percentiles of the same windows' values, one sort for the three; NaN, NaN and 0 for a bin
    # outside the variable's interval.
    s_stat = Dict{String,Vector{Float64}}(name => Vector{Float64}(undef, ncoarse) for name in free)
    n_near_bound = Dict{String,Vector{Int}}(name => Vector{Int}(undef, ncoarse) for name in free)
    p16 = Dict{String,Vector{Float64}}(name => Vector{Float64}(undef, ncoarse) for name in free)
    p50 = Dict{String,Vector{Float64}}(name => Vector{Float64}(undef, ncoarse) for name in free)
    p84 = Dict{String,Vector{Float64}}(name => Vector{Float64}(undef, ncoarse) for name in free)
    for k in 1:ncoarse
        J = findall(==(k), coarse_bin)
        for name in free
            if !member[name][k]
                s_stat[name][k], n_near_bound[name][k] = NaN, 0
                p16[name][k], p50[name][k], p84[name][k] = NaN, NaN, NaN
                continue
            end
            s_stat[name][k] = _detrended_spread(t_fine[J], values[name][J])
            n_near_bound[name][k] = count(near_bound[name][J])
            x = sort(values[name][J])
            p16[name][k] = _quantile_sorted(x, 0.16)
            p50[name][k] = _quantile_sorted(x, 0.5)
            p84[name][k] = _quantile_sorted(x, 0.84)
        end
    end

    return ScatterResult(copy(result.theta), m, t_fine, coarse_bin, counts, status, decrease,
        values, near_bound, s_stat, n_near_bound, win_data, win_sigma, win_mask, win_model, win_chi2,
        held, p16, p50, p84)
end
