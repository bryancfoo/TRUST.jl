# write_result/read_result (JLD2, plain arrays and strings, exactly the streak and trace
# layouts' own rule): the otis-result-1 layout. Step 3.6a, chunk 11 of Phase 3; spec D34,
# extending the file family of docs/derivations/data_layout.md (Section 3).

using Dates: Dates, DateTime, UTC, now
using JLD2: jldopen

"""
    RESULT_LAYOUT_VERSION

The `layout_version` string written to every result JLD2 file and checked on read.

Spec D34, D45.
"""
const RESULT_LAYOUT_VERSION = "otis-result-1"

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
    write_result(path, result::FitResult; deck = "", source = "fit",
                 scan::Union{Nothing,WeightScanResult} = nothing) -> path

Writes `result` (and, if given, `scan`) to `path` as a JLD2 file of plain numeric arrays and
strings (`docs/derivations/data_layout.md` Section 3, spec D34): `layout_version`, a `meta`
group, the effective `deck` text, the axis, one group per fitted variable and per dependent, one
group per leaf of the fitted trace, the `fit/*` scalars and vectors, the `trajectory/*` arrays,
and, only when `scan !== nothing`, the `scan/*` arrays and per-variable `scan/rss*` groups,
including `scan/rss_unreg/<variable>`.

Spec D34.
"""
function write_result(path::AbstractString, result::FitResult; deck::AbstractString="",
    source="fit", scan::Union{Nothing,WeightScanResult}=nothing)
    pspec = result.pspec
    nt = nbins(pspec.trace)
    unit_keys = _variable_unit_keys(pspec.bindings)

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
        f["fit/optimizer"] = result.optimizer
        f["fit/elapsed"] = Float64(result.elapsed)

        f["trajectory/iteration"] = Int64[Int64(round(row.iteration)) for row in result.trajectory]
        f["trajectory/objective"] = Float64[row.objective for row in result.trajectory]
        f["trajectory/chi2"] = Float64[row.chi2 for row in result.trajectory]
        f["trajectory/R"] = Float64[row.R for row in result.trajectory]

        if scan !== nothing
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
axis_boundary_unit, variables, dependents, leaves, fit, trajectory, scan)`. `variables` and
`dependents` are `Dict{String,<:NamedTuple}` keyed by name; `leaves` is `Dict{String,Vector{Float64}}`
keyed by SI leaf path; `fit` and `trajectory` are `NamedTuple`s of the file's own `fit/*` and
`trajectory/*` datasets; `scan` is `nothing` when the file carries no `scan/*` group, else a
`NamedTuple` (`rss`/`rss_normalized` themselves `Dict{String,Vector{Float64}}` and `rss_unreg`
`Dict{String,Float64}`, empty when the file carries no ground truth). Throws, naming the version
found and expected, if `layout_version` does not match `RESULT_LAYOUT_VERSION`.

Spec D34.
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
            converged=f["fit/converged"], status=f["fit/status"], optimizer=f["fit/optimizer"],
            elapsed=f["fit/elapsed"],
        )

        trajectory = (
            iteration=f["trajectory/iteration"], objective=f["trajectory/objective"],
            chi2=f["trajectory/chi2"], R=f["trajectory/R"],
        )

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
                rss=rss, rss_normalized=rss_normalized, rss_unreg=rss_unreg,
            )
        end

        return (; layout_version=version, meta, deck, axis_name, axis, axis_unit,
            axis_boundary_unit, variables, dependents, leaves, fit, trajectory, scan)
    end
end
