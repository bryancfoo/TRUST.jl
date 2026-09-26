# write_result/read_result (JLD2, plain arrays and strings, exactly the streak and trace
# layouts' own rule): the otis-result-1 layout, bumped to otis-result-2 at Step 3.C.2 (D35), which
# also added scan/termination at Step 3.C.3 (carried item 2, inside the same otis-result-2 version
# -- D35 forbids a *second* layout bump, not a dataset added within one before the phase closes).
# Step 3.6a, chunk 11 of Phase 3; spec D34, extending the file family of
# docs/derivations/data_layout.md (Section 3), which now describes otis-result-2 in full.

using Dates: Dates, DateTime, UTC, now
using JLD2: jldopen

"""
    RESULT_LAYOUT_VERSION

The `layout_version` string written to every result JLD2 file and checked on read.

**Bumped to `"otis-result-2"` at Step 3.C.2 (D35, D17)**, one bump, carrying `fit/termination`,
`fit/near_bound_fraction` and (when a scan is written) `scan/status` together, and **later joined,
within the same `"otis-result-2"`, by `scan/termination`** (Step 3.C.3 carried item 2, D17-vs-D35):
`read_result` rejects a version mismatch before it reads anything, so one bump covering every new
dataset introduced before the phase closes is all Phase 3.C needs -- D35 forbids a *second* bump,
not a dataset added inside the one it already made.

**Bumped to `"otis-result-3"` at Step 4.A.3b (D40, chunk 7)**, one bump, carrying a `channels/`
group: `channels/names` and, per channel, `weight`, `chi2`, `npixels`, `nparams`, `reduced_chi2`,
`reduced_chi2_simple`, `lambda`, `model`, `effects/*`, and, only when `write_result` is given the
problem, `residual` and `mask`. Every `otis-result-3` file has `channels/`, since every `FitResult`
carries at least one `ChannelSummary` (D36); a shorthand fit writes `channels/main`.

Spec D34, D45, D35, D40.
"""
const RESULT_LAYOUT_VERSION = "otis-result-3"

"""
    _channel_problem(problem::FitProblem, c::Int) -> FitProblem
    _channel_problem(problem::JointFitProblem, c::Int) -> FitProblem

Channel `c`'s own `FitProblem` (D40 amendment 7): `problem` itself for a `FitProblem` (`c == 1`
always, since a single-channel problem has one channel), `problem.channels[c]` for a
`JointFitProblem`.
"""
_channel_problem(problem::FitProblem, ::Int) = problem
_channel_problem(problem::JointFitProblem, c::Int) = problem.channels[c]

"""
    _near_bound_fraction(result::FitResult) -> Float64

The fraction of `result.theta`'s coordinates flagged `near_bound` (D31, D35): the count of `true`
entries in `result.near_bound[v.name]`, summed over every **fitted** variable (`v.vary`, so a
frozen variable's own near-a-bound `init` never counts -- it has no entry in `theta` at all),
divided by `result.pspec.ntheta`. This is deliberately **not** a fresh
`_count_near_bound(result.pspec, result.theta)` call: it is built from `result.near_bound`, which
`_build_result` already computed from the same `_from_unconstrained(theta[idx], ...)` values at
the same coordinates, so the two are the same quantity computed the same way (D31's own test in
`solve` computes it directly from `theta`; this reads it back off the result already carrying it).
A time-varying fitted variable's row carries one entry per bin of the whole axis, `NaN`-filled
(hence `_near_bound` false) outside its own interval by `_value_table`/`_build_result` -- those
entries are never in `theta` either, and `_near_bound(NaN, a, b)` is `false` (`min` propagates the
`NaN`, and `NaN < x` is `false`), so summing every variable's `count(true, ...)` and dividing by
`ntheta` (never by the summed row lengths) gives exactly `_count_near_bound(pspec, theta) /
pspec.ntheta` without needing to re-walk `theta` at all. Not part of D35's own signature; Level 3's
helper, flagged under rule 11 zero.

Spec D35, D31.
"""
function _near_bound_fraction(result::FitResult)
    pspec = result.pspec
    nb = sum((count(result.near_bound[v.name]) for v in pspec.variables if v.vary); init=0)
    return nb / pspec.ntheta
end

"""
    _iso8601_utc_now() -> String

The current UTC time as an ISO-8601 string with an explicit `"Z"` (UTC) suffix, millisecond
precision (`meta/created`, OQ-1 A6). `Dates.DateTime`'s own `string` has no zone marker, hence
the appended `"Z"` rather than relying on a `DateFormat` literal for it.
"""
_iso8601_utc_now() = Dates.format(now(UTC), Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"

"""
    _variable_unit_keys(bindings) -> Dict{String,String}

One `unit_key` per bound name (spec D9: one variable, one conversion factor, so every binding of
a given name already carries the same `unit_key`), read off `ParameterSpec.bindings`. A name with
no `:leaf` binding (frozen, dependent-only, or bound only to an effect parameter) is absent, and
`write_result` reports `""` for it (D34: `""` if none).
"""
function _variable_unit_keys(bindings::Vector{Binding})
    units = Dict{String,String}()
    for b in bindings
        b.kind === :leaf || continue
        haskey(units, b.name) || (units[b.name] = b.unit_key)
    end
    return units
end

"""
    _variable_bins(v::FittedVariable, intervals::Vector{FitInterval}, nt::Int) -> (Int, Int)

The first and last bin of `v`'s own interval (`(1, nt)` for `interval == 0`, the whole axis),
`variables/<name>/bins` (D34).
"""
function _variable_bins(v::FittedVariable, intervals::Vector{FitInterval}, nt::Int)
    v.interval == 0 && return 1, nt
    bins = intervals[v.interval].bins
    return first(bins), last(bins)
end

"""
    write_result(path, result::FitResult; problem::Union{Nothing,AbstractFitProblem} = nothing,
                 deck = "", source = "fit",
                 scan::Union{Nothing,WeightScanResult} = nothing,
                 laplace::Union{Nothing,LaplaceResult} = nothing,
                 calibration_scan::Union{Nothing,CalibrationScanResult} = nothing,
                 scatter::Union{Nothing,ScatterResult} = nothing,
                 gn_decrease::Union{Nothing,Real} = nothing) -> path

Writes `result` (and, if given, `scan`, `laplace` and `calibration_scan`) to `path` as a JLD2 file
of plain numeric arrays and strings (`docs/derivations/data_layout.md` Section 3, spec D34, D40,
D47): `layout_version`, a `meta` group, the effective `deck` text, the axis, one group per fitted
variable and per dependent, one group per leaf of the fitted trace, the `fit/*` scalars and
vectors (including, since `otis-result-2`, D35, `fit/termination` and `fit/near_bound_fraction`;
`fit/chi2` and `fit/npixels` are the **weighted total** and the **sum over channels** for a
`JointFitProblem` result, D40), and, only when given `gn_decrease` (which `fit(deck)` always
passes), `fit/gn_decrease`, a `Float64` that may be `NaN` (Phase 4.B spec D5), the `trajectory/*`
arrays, `channels/names` and, per channel
(D40), the six summary scalars, `lambda`, `model` and `effects/*`, and, only when `problem !==
nothing`, `residual` and `mask`; and, only when `scan !== nothing`, the `scan/*` arrays (including
`scan/status` and `scan/termination`, one string per requested weight each) and per-variable
`scan/rss*` groups, including `scan/rss_unreg/<variable>`.

**`laplace` (D47), only when given**: `laplace/decrease`, and, per variable,
`laplace/variables/<name>/sigma_post`, `.../sigma_samp` (`Vector{Float64}`, `NaN` kept) and
`.../near_bound` (`UInt8`) -- **not** the `p x p` covariances (13 MB at `p = 1280`). Must satisfy
`laplace.theta == result.theta` exactly, else a loud error (`write_result` has no other way to
know `laplace` was computed at the θ `result` reports).

**`calibration_scan` (D47), only when given**: `calibration_scan/shifts_nm`, `channel`,
`base_offset_nm`, `chi2`, `objective`, `decrease` and `status` (strings), and, per variable,
`calibration_scan/variables/<name>/delta` (`Matrix`) and `.../s_sys`.

**`scatter` (D52), only when given**: `scatter/m`, `t_fine`, `coarse_bin`, `counts`, `status`
(strings) and `decrease`, and, per free variable, `scatter/variables/<name>/values`, `near_bound`
(`UInt8`), `s_stat` and `n_near_bound`. Must satisfy `scatter.theta == result.theta` exactly, else
a loud error (`"scatter was computed at a different θ than result.theta"`).

All three groups are written inside `otis-result-3` (D17 forbids a *second* bump within the phase,
not a dataset added inside the one bump already made); **RE1's exact key set is unchanged**, since
none is written without its keyword.

When `problem` is given, `problem.pspec === result.pspec` is required (a loud error otherwise,
naming what to pass instead): `write_result` has no other way to know the problem it is given is
the one `result` was actually fitted from, and the residual needs the problem's own `pixel_index`,
`data` and `sigma`. Without a problem, `residual` and `mask` are omitted and `read_result` returns
`nothing` for them -- the three `examples/fit_*.jl`, `test_result.jl` and `test_reproduce.jl` all
called `write_result` this way before chunk 7 and stay legal (D40 item 9).

Spec D34, D35, D40, D47.
"""
function write_result(path::AbstractString, result::FitResult;
    problem::Union{Nothing,AbstractFitProblem}=nothing, deck::AbstractString="",
    source="fit", scan::Union{Nothing,WeightScanResult}=nothing,
    laplace::Union{Nothing,LaplaceResult}=nothing,
    calibration_scan::Union{Nothing,CalibrationScanResult}=nothing,
    scatter::Union{Nothing,ScatterResult}=nothing,
    gn_decrease::Union{Nothing,Real}=nothing)
    pspec = result.pspec
    nt = nbins(pspec.trace)
    unit_keys = _variable_unit_keys(pspec.bindings)

    if problem !== nothing
        problem.pspec === result.pspec || throw(ArgumentError(
            "write_result: problem.pspec is not === result.pspec; pass the problem the result " *
            "was fitted from"))
    end

    if laplace !== nothing
        laplace.theta == result.theta || throw(ArgumentError(
            "write_result: laplace was computed at a different θ than result.theta"))
    end

    if scatter !== nothing
        scatter.theta == result.theta || throw(ArgumentError(
            "write_result: scatter was computed at a different θ than result.theta"))
    end

    jldopen(path, "w") do f
        f["layout_version"] = RESULT_LAYOUT_VERSION
        f["meta/source"] = String(source)
        f["meta/created"] = _iso8601_utc_now()
        f["meta/created_unix"] = Float64(time())
        f["meta/otis_version"] = string(pkgversion(OTIS))
        f["deck"] = String(deck)

        f["axis_name"] = String(pspec.trace.axis_name)
        f["axis"] = Vector{Float64}(pspec.axis)
        f["axis_unit"] = pspec.trace.axis_name === :t ? "s" : "SI"
        f["axis_boundary_unit"] = pspec.axis_unit

        for v in pspec.variables
            prefix = "variables/$(v.name)"
            f["$prefix/values"] = Vector{Float64}(result.values[v.name])
            f["$prefix/unit"] = get(unit_keys, v.name, "")
            f["$prefix/bounds"] = Float64[v.lower, v.upper]
            first_bin, last_bin = _variable_bins(v, pspec.intervals, nt)
            f["$prefix/bins"] = Int64[first_bin, last_bin]
            f["$prefix/time_varying"] = UInt8(v.time_varying)
            f["$prefix/vary"] = UInt8(v.vary)
            f["$prefix/near_bound"] = UInt8.(result.near_bound[v.name])
        end

        for dep in pspec.dependents
            prefix = "dependents/$(dep.name)"
            f["$prefix/values"] = Vector{Float64}(result.values[dep.name])
            f["$prefix/expression"] = dep.text
        end

        fitted_trace = result.experiment.trace
        for (i, p) in enumerate(fitted_trace.paths)
            f["leaves/$(string(p))/values"] = Vector{Float64}(@view fitted_trace.values[i, :])
        end

        f["fit/theta"] = Vector{Float64}(result.theta)
        f["fit/chi2"] = Float64(result.chi2)
        f["fit/R"] = Float64(result.R)
        f["fit/objective"] = Float64(result.objective)
        f["fit/weight"] = Float64(result.weight)
        f["fit/term_values"] = Vector{Float64}(result.term_values)
        f["fit/npixels"] = Int64(result.npixels)
        f["fit/iterations"] = Int64(result.iterations)
        f["fit/f_calls"] = Int64(result.f_calls)
        f["fit/g_calls"] = Int64(result.g_calls)
        f["fit/converged"] = UInt8(result.converged)
        f["fit/status"] = string(result.status)
        f["fit/termination"] = string(result.termination)
        f["fit/near_bound_fraction"] = _near_bound_fraction(result)
        gn_decrease === nothing || (f["fit/gn_decrease"] = Float64(gn_decrease))
        f["fit/optimizer"] = result.optimizer
        f["fit/elapsed"] = Float64(result.elapsed)

        f["trajectory/iteration"] = Int64[Int64(round(row.iteration)) for row in result.trajectory]
        f["trajectory/objective"] = Float64[row.objective for row in result.trajectory]
        f["trajectory/chi2"] = Float64[row.chi2 for row in result.trajectory]
        f["trajectory/R"] = Float64[row.R for row in result.trajectory]

        f["channels/names"] = String[c.name for c in result.channels]
        for (c, summ) in enumerate(result.channels)
            prefix = "channels/$(summ.name)"
            f["$prefix/weight"] = Float64(summ.weight)
            f["$prefix/chi2"] = Float64(summ.chi2)
            f["$prefix/npixels"] = Int64(summ.npixels)
            f["$prefix/nparams"] = Int64(summ.nparams)
            f["$prefix/reduced_chi2"] = Float64(summ.reduced_chi2)
            f["$prefix/reduced_chi2_simple"] = Float64(summ.reduced_chi2_simple)

            expt = result.experiments[c]
            f["$prefix/lambda"] = Vector{Float64}(expt.lambda)
            model = Matrix{Float64}(measure(expt))
            f["$prefix/model"] = model

            for (k, e) in enumerate(expt.effects)
                eprefix = "$prefix/effects/$k"
                f["$eprefix/type"] = string(nameof(typeof(e)))
                for (field, v) in pairs(parameters(e))
                    f["$eprefix/$field"] = v isa Union{Tuple,AbstractVector} ?
                        collect(Float64, v) : Float64(v)
                end
            end

            if problem !== nothing
                ch = _channel_problem(problem, c)
                residual = fill(NaN, size(ch.data.image))
                D = ch.data.image
                sig = ch.sigma
                for i in ch.pixel_index
                    residual[i] = (model[i] - D[i]) / sig[i]
                end
                f["$prefix/residual"] = residual
                f["$prefix/mask"] = UInt8.(ch.mask)
            end
        end

        if scan !== nothing
            f["scan/status"] = String[string(r.status) for r in scan.results]
            f["scan/termination"] = String[string(r.termination) for r in scan.results]
            f["scan/weights"] = Vector{Float64}(scan.weights)
            f["scan/chi2"] = Vector{Float64}(scan.chi2)
            f["scan/R"] = Vector{Float64}(scan.R)
            f["scan/chi2_normalized"] = Vector{Float64}(scan.chi2_normalized)
            f["scan/R_normalized"] = Vector{Float64}(scan.R_normalized)
            f["scan/chi2_unreg"] = Float64(scan.chi2_unreg)
            f["scan/R_unreg"] = Float64(scan.R_unreg)
            for (name, v) in scan.rss
                f["scan/rss/$name"] = Vector{Float64}(v)
            end
            for (name, v) in scan.rss_normalized
                f["scan/rss_normalized/$name"] = Vector{Float64}(v)
            end
            for (name, v) in scan.rss_unreg
                f["scan/rss_unreg/$name"] = Float64(v)
            end
        end

        if laplace !== nothing
            f["laplace/decrease"] = Float64(laplace.decrease)
            for (name, sp) in laplace.sigma_post
                prefix = "laplace/variables/$name"
                f["$prefix/sigma_post"] = Vector{Float64}(sp)
                f["$prefix/sigma_samp"] = Vector{Float64}(laplace.sigma_samp[name])
                f["$prefix/near_bound"] = UInt8.(laplace.near_bound[name])
            end
        end

        if calibration_scan !== nothing
            f["calibration_scan/shifts_nm"] = Vector{Float64}(calibration_scan.shifts_nm)
            f["calibration_scan/channel"] = Int64(calibration_scan.channel)
            f["calibration_scan/base_offset_nm"] = Float64(calibration_scan.base_offset_nm)
            f["calibration_scan/chi2"] = Vector{Float64}(calibration_scan.chi2)
            f["calibration_scan/objective"] = Vector{Float64}(calibration_scan.objective)
            f["calibration_scan/decrease"] = Vector{Float64}(calibration_scan.decrease)
            f["calibration_scan/status"] = String[string(s) for s in calibration_scan.status]
            for (name, m) in calibration_scan.delta
                prefix = "calibration_scan/variables/$name"
                f["$prefix/delta"] = Matrix{Float64}(m)
                f["$prefix/s_sys"] = Vector{Float64}(calibration_scan.s_sys[name])
            end
        end

        if scatter !== nothing
            f["scatter/m"] = Int64.(scatter.m)
            f["scatter/t_fine"] = Vector{Float64}(scatter.t_fine)
            f["scatter/coarse_bin"] = Int64.(scatter.coarse_bin)
            f["scatter/counts"] = Int64.(scatter.counts)
            f["scatter/status"] = String[string(st) for st in scatter.status]
            f["scatter/decrease"] = Vector{Float64}(scatter.decrease)
            for (name, v) in scatter.values
                prefix = "scatter/variables/$name"
                f["$prefix/values"] = Vector{Float64}(v)
                f["$prefix/near_bound"] = UInt8.(scatter.near_bound[name])
                f["$prefix/s_stat"] = Vector{Float64}(scatter.s_stat[name])
                f["$prefix/n_near_bound"] = Int64.(scatter.n_near_bound[name])
            end
        end
    end
    return path
end

function _read_group(f, prefix::AbstractString)
    haskey(f, prefix) || return String[]
    return collect(String, keys(f[prefix]))
end

"""
    read_result(path::AbstractString) -> NamedTuple

Reads a result written by `write_result`, returning plain arrays and strings, like `read_trace`'s
own `:arrays` mode: `(; layout_version, meta, deck, axis_name, axis, axis_unit,
axis_boundary_unit, variables, dependents, leaves, fit, trajectory, channels, scan, laplace,
calibration_scan, scatter)`. `variables` and `dependents` are `Dict{String,<:NamedTuple}` keyed by name;
`leaves` is `Dict{String,Vector{Float64}}` keyed by SI leaf path; `fit` and `trajectory` are
`NamedTuple`s of the file's own `fit/*` and `trajectory/*` datasets; `channels` (D40) is a
`Vector{<:NamedTuple}`, one per channel in channel order, each `(; name, weight, chi2, npixels,
nparams, reduced_chi2, reduced_chi2_simple, lambda, model, effects, residual, mask)` -- `effects`
a `Vector` of `(; type::String, parameters::Dict{String,Any})` in index order, and `residual`/
`mask` `nothing` when the file carries none (`write_result` was not given a problem); `scan` is
`nothing` when the file carries no `scan/*` group, else a `NamedTuple` (`rss`/`rss_normalized`
themselves `Dict{String,Vector{Float64}}` and `rss_unreg` `Dict{String,Float64}`, empty when the
file carries no ground truth). `fit.gn_decrease` is `nothing` when the file carries none (Phase
4.B spec D5).

**`laplace` (D47)**: `nothing` when the file carries no `laplace/*` group, else `(; decrease,
sigma_post, sigma_samp, near_bound)`, `sigma_post`/`sigma_samp`/`near_bound` each
`Dict{String,Vector}` keyed by variable name.

**`calibration_scan` (D47)**: `nothing` when the file carries no `calibration_scan/*` group, else
`(; shifts_nm, channel, base_offset_nm, chi2, objective, decrease, status, delta, s_sys)`,
`delta`/`s_sys` each `Dict{String,<:AbstractArray}` keyed by variable name.

**`scatter` (D52)**: `nothing` when the file carries no `scatter/*` group, else `(; m, t_fine,
coarse_bin, counts, status, decrease, values, near_bound, s_stat, n_near_bound)`, `values`,
`near_bound`, `s_stat` and `n_near_bound` each `Dict{String,<:AbstractArray}` keyed by variable
name, in the same forms as `calibration_scan`'s.

Throws, naming the version found and expected, if `layout_version` does not match
`RESULT_LAYOUT_VERSION`.

Spec D34, D40, D47.
"""
function read_result(path::AbstractString)
    jldopen(path, "r") do f
        version = _require_dataset(f, "layout_version", path)
        version == RESULT_LAYOUT_VERSION || throw(ArgumentError(
            "read_result: \"$path\" has layout_version \"$version\"; expected \"$RESULT_LAYOUT_VERSION\""))

        meta = _read_meta(f)
        deck = _require_dataset(f, "deck", path)
        axis_name = Symbol(_require_dataset(f, "axis_name", path))
        axis = _require_dataset(f, "axis", path)
        axis_unit = _require_dataset(f, "axis_unit", path)
        axis_boundary_unit = _require_dataset(f, "axis_boundary_unit", path)

        variables = Dict{String,Any}()
        for nm in _read_group(f, "variables")
            prefix = "variables/$nm"
            variables[nm] = (
                values=f["$prefix/values"],
                unit=f["$prefix/unit"],
                bounds=f["$prefix/bounds"],
                bins=f["$prefix/bins"],
                time_varying=f["$prefix/time_varying"],
                vary=f["$prefix/vary"],
                near_bound=f["$prefix/near_bound"],
            )
        end

        dependents = Dict{String,Any}()
        for nm in _read_group(f, "dependents")
            prefix = "dependents/$nm"
            dependents[nm] = (values=f["$prefix/values"], expression=f["$prefix/expression"])
        end

        leaves = Dict{String,Vector{Float64}}()
        for nm in _read_group(f, "leaves")
            leaves[nm] = f["leaves/$nm/values"]
        end

        fit = (
            theta=f["fit/theta"], chi2=f["fit/chi2"], R=f["fit/R"], objective=f["fit/objective"],
            weight=f["fit/weight"], term_values=f["fit/term_values"], npixels=f["fit/npixels"],
            iterations=f["fit/iterations"], f_calls=f["fit/f_calls"], g_calls=f["fit/g_calls"],
            converged=f["fit/converged"], status=f["fit/status"], termination=f["fit/termination"],
            near_bound_fraction=f["fit/near_bound_fraction"], optimizer=f["fit/optimizer"],
            elapsed=f["fit/elapsed"],
            gn_decrease=haskey(f, "fit/gn_decrease") ? f["fit/gn_decrease"] : nothing,
        )

        trajectory = (
            iteration=f["trajectory/iteration"], objective=f["trajectory/objective"],
            chi2=f["trajectory/chi2"], R=f["trajectory/R"],
        )

        channels = NamedTuple[]
        for nm in _require_dataset(f, "channels/names", path)
            prefix = "channels/$nm"
            effect_keys = _read_group(f, "$prefix/effects")
            effects = NamedTuple[]
            for k in sort(parse.(Int, effect_keys))
                eprefix = "$prefix/effects/$k"
                pnames = [n for n in _read_group(f, eprefix) if n != "type"]
                params = Dict{String,Any}(n => f["$eprefix/$n"] for n in pnames)
                push!(effects, (; type=f["$eprefix/type"], parameters=params))
            end
            has_residual = haskey(f, "$prefix/residual")
            push!(channels, (
                name=nm, weight=f["$prefix/weight"], chi2=f["$prefix/chi2"],
                npixels=f["$prefix/npixels"], nparams=f["$prefix/nparams"],
                reduced_chi2=f["$prefix/reduced_chi2"],
                reduced_chi2_simple=f["$prefix/reduced_chi2_simple"],
                lambda=f["$prefix/lambda"], model=f["$prefix/model"], effects=effects,
                residual=has_residual ? f["$prefix/residual"] : nothing,
                mask=has_residual ? f["$prefix/mask"] : nothing,
            ))
        end

        scan = nothing
        if haskey(f, "scan")
            rss = Dict{String,Vector{Float64}}(nm => f["scan/rss/$nm"] for nm in _read_group(f, "scan/rss"))
            rss_normalized = Dict{String,Vector{Float64}}(
                nm => f["scan/rss_normalized/$nm"] for nm in _read_group(f, "scan/rss_normalized"))
            rss_unreg = Dict{String,Float64}(
                nm => f["scan/rss_unreg/$nm"] for nm in _read_group(f, "scan/rss_unreg"))
            scan = (
                weights=f["scan/weights"], chi2=f["scan/chi2"], R=f["scan/R"],
                chi2_normalized=f["scan/chi2_normalized"], R_normalized=f["scan/R_normalized"],
                chi2_unreg=f["scan/chi2_unreg"], R_unreg=f["scan/R_unreg"],
                status=f["scan/status"], termination=f["scan/termination"],
                rss=rss, rss_normalized=rss_normalized, rss_unreg=rss_unreg,
            )
        end

        laplace = nothing
        if haskey(f, "laplace")
            sigma_post = Dict{String,Vector{Float64}}()
            sigma_samp = Dict{String,Vector{Float64}}()
            near_bound = Dict{String,Vector{UInt8}}()
            for nm in _read_group(f, "laplace/variables")
                prefix = "laplace/variables/$nm"
                sigma_post[nm] = f["$prefix/sigma_post"]
                sigma_samp[nm] = f["$prefix/sigma_samp"]
                near_bound[nm] = f["$prefix/near_bound"]
            end
            laplace = (decrease=f["laplace/decrease"], sigma_post=sigma_post, sigma_samp=sigma_samp,
                near_bound=near_bound)
        end

        calibration_scan = nothing
        if haskey(f, "calibration_scan")
            delta = Dict{String,Matrix{Float64}}()
            s_sys = Dict{String,Vector{Float64}}()
            for nm in _read_group(f, "calibration_scan/variables")
                prefix = "calibration_scan/variables/$nm"
                delta[nm] = f["$prefix/delta"]
                s_sys[nm] = f["$prefix/s_sys"]
            end
            calibration_scan = (
                shifts_nm=f["calibration_scan/shifts_nm"], channel=f["calibration_scan/channel"],
                base_offset_nm=f["calibration_scan/base_offset_nm"], chi2=f["calibration_scan/chi2"],
                objective=f["calibration_scan/objective"], decrease=f["calibration_scan/decrease"],
                status=f["calibration_scan/status"], delta=delta, s_sys=s_sys,
            )
        end

        scatter = nothing
        if haskey(f, "scatter")
            values = Dict{String,Vector{Float64}}()
            near_bound = Dict{String,Vector{UInt8}}()
            s_stat = Dict{String,Vector{Float64}}()
            n_near_bound = Dict{String,Vector{Int64}}()
            for nm in _read_group(f, "scatter/variables")
                prefix = "scatter/variables/$nm"
                values[nm] = f["$prefix/values"]
                near_bound[nm] = f["$prefix/near_bound"]
                s_stat[nm] = f["$prefix/s_stat"]
                n_near_bound[nm] = f["$prefix/n_near_bound"]
            end
            scatter = (
                m=f["scatter/m"], t_fine=f["scatter/t_fine"], coarse_bin=f["scatter/coarse_bin"],
                counts=f["scatter/counts"], status=f["scatter/status"], decrease=f["scatter/decrease"],
                values=values, near_bound=near_bound, s_stat=s_stat, n_near_bound=n_near_bound,
            )
        end

        return (; layout_version=version, meta, deck, axis_name, axis, axis_unit,
            axis_boundary_unit, variables, dependents, leaves, fit, trajectory, channels, scan,
            laplace, calibration_scan, scatter)
    end
end
