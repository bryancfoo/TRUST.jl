# write_start/read_start: the otis-start-1 layout, a time axis and one series per variable name
# (Phase 4.I, Step 4.I.1; OQ-59 item 2; docs/derivations/data_layout.md Section 5). The file a deck's
# [fit.start] and [fit.hold] tables may name beside an otis-result-3 result: written by the package's
# own write_start, so that converting a fit made elsewhere is a script calling the package and the
# package reads no foreign format. Plain arrays and strings only, as every OTIS layout.

using JLD2: jldopen

"""
    START_LAYOUT_VERSION

The `layout_version` string written to every start file and checked on read: `"otis-start-1"`
(Phase 4.I spec D5).
"""
const START_LAYOUT_VERSION = "otis-start-1"

const _START_AXIS_KEYS = (axis_s=1.0, axis_ns=1e-9)

"""
    write_start(path::AbstractString; axis_s | axis_ns, values, units = Dict{String,String}(),
                source::AbstractString = "") -> path

Writes the `otis-start-1` layout (`docs/derivations/data_layout.md` Section 5; Phase 4.I spec D5):
`layout_version`, `meta/source`, `axis` (`Float64`, s, strictly increasing), and, per name of
`values`, `variables/<name>/values` (`Float64`, `length(axis)` entries, or one for a value that
holds at every time) and, when `units` names it, `variables/<name>/unit` (`String`, the boundary
keyword the values are in, e.g. `"T_eV"`; a deck refuses a file whose unit differs from the one its
variable is bound through).

`axis_s` or `axis_ns`, exactly one, is the time axis in s or ns. `values` is an iterable of `name =>
vector` pairs (a `Dict` or a `Vector` of `Pair`s); each name a valid variable name, each vector of
finite numbers. Every refusal is an `ArgumentError` naming what failed. An empty `source` or unit is
written as `"-"` (Phase 4.N spec D20: h5py cannot read an empty scalar string), and `read_start` returns
it as `""`.
"""
function write_start(path::AbstractString; values, units=Dict{String,String}(), source::AbstractString="",
    kwargs...)
    _reject_extra_kwargs(kwargs, keys(_START_AXIS_KEYS))
    present = Tuple(k for k in keys(_START_AXIS_KEYS) if haskey(kwargs, k))
    length(present) == 1 || throw(ArgumentError(
        "write_start: give exactly one of axis_s or axis_ns; got $(present)"))
    axis = Float64.(collect(kwargs[present[1]])) .* _START_AXIS_KEYS[present[1]]
    _check_start_axis(axis, "write_start")

    pairs_list = [String(k) => Float64.(collect(v)) for (k, v) in values]
    isempty(pairs_list) && throw(ArgumentError("write_start: values names no variable"))
    seen = Set{String}()
    for (nm, v) in pairs_list
        occursin(_VARIABLE_NAME_RE, nm) || throw(ArgumentError(
            "write_start: \"$nm\" is not a valid variable name; expected $(_VARIABLE_NAME_RE.pattern)"))
        nm in seen && throw(ArgumentError("write_start: \"$nm\" is given twice"))
        push!(seen, nm)
        (length(v) == length(axis) || length(v) == 1) || throw(ArgumentError(
            "write_start: \"$nm\" has $(length(v)) values; the axis has $(length(axis)) (or give one value)"))
        all(isfinite, v) || throw(ArgumentError("write_start: \"$nm\" has a value that is not finite"))
    end
    unit_pairs = [String(k) => String(u) for (k, u) in units]
    for (nm, _) in unit_pairs
        nm in seen || throw(ArgumentError("write_start: units names \"$nm\", which values does not"))
    end

    jldopen(path, "w") do f
        f["layout_version"] = START_LAYOUT_VERSION
        f["meta/source"] = _empty_marker(source)
        f["axis"] = axis
        for (nm, v) in pairs_list
            f["variables/$nm/values"] = v
        end
        for (nm, u) in unit_pairs
            f["variables/$nm/unit"] = _empty_marker(u)
        end
    end
    return path
end

function _check_start_axis(axis::AbstractVector, caller::AbstractString)
    isempty(axis) && throw(ArgumentError("$caller: the axis is empty"))
    all(isfinite, axis) || throw(ArgumentError("$caller: the axis has a value that is not finite"))
    for k in 2:length(axis)
        axis[k] > axis[k-1] || throw(ArgumentError("$caller: the axis must be strictly increasing"))
    end
    return nothing
end

"""
    read_start(path::AbstractString) -> (; layout_version, source, axis, values, units)

Reads a file written by `write_start` (or by any writer of the same plain datasets, Section 5's h5py
snippet): `axis` the time axis in s, `values` a `Dict{String,Vector{Float64}}` and `units` a
`Dict{String,String}` holding only the names the file gives a unit. Throws, naming the path, when
`layout_version` is not `"otis-start-1"`, when a required dataset is missing, when the axis is not
strictly increasing, or when a series' length is neither the axis's nor one. `source` and a unit read
`""` where the file holds `"-"`, the marker `write_start` writes for an empty string (Phase 4.N spec D20).
"""
function read_start(path::AbstractString)
    jldopen(path, "r") do f
        haskey(f, "layout_version") || throw(ArgumentError(
            "read_start: \"$path\" is missing the required dataset \"layout_version\""))
        version = f["layout_version"]
        version == START_LAYOUT_VERSION || throw(ArgumentError(
            "read_start: \"$path\" has layout_version \"$version\"; expected \"$START_LAYOUT_VERSION\""))
        for name in ("axis", "variables")
            haskey(f, name) || throw(ArgumentError(
                "read_start: \"$path\" is missing the required dataset \"$name\""))
        end
        axis = Vector{Float64}(f["axis"])
        _check_start_axis(axis, "read_start: \"$path\"")
        values = Dict{String,Vector{Float64}}()
        units = Dict{String,String}()
        for nm in keys(f["variables"])
            v = Vector{Float64}(f["variables/$nm/values"])
            (length(v) == length(axis) || length(v) == 1) || throw(ArgumentError(
                "read_start: \"$path\": \"$nm\" has $(length(v)) values; the axis has $(length(axis))"))
            values[nm] = v
            haskey(f, "variables/$nm/unit") && (units[nm] = _from_empty_marker(String(f["variables/$nm/unit"])))
        end
        isempty(values) && throw(ArgumentError("read_start: \"$path\" holds no variable"))
        source = haskey(f, "meta/source") ? _from_empty_marker(String(f["meta/source"])) : ""
        return (; layout_version=version, source, axis, values, units)
    end
end

"""
    _interpolate_series(axis::AbstractVector, values::AbstractVector, t::Real) -> Float64

`values` interpolated linearly in time at `t` (Phase 4.I spec D6): `values[k]` itself, bitwise, when
`t == axis[k]`; otherwise `y0 + (t - x0)/(x1 - x0) * (y1 - y0)` on the bracketing pair that
`searchsortedlast` finds, the form `_interpolate_response` uses. The caller checks `first(axis) <= t
<= last(axis)` (a series is never extrapolated) and handles a one-value series.
"""
function _interpolate_series(axis::AbstractVector, values::AbstractVector, t::Real)
    k = searchsortedlast(axis, t)
    axis[k] == t && return Float64(values[k])
    x0, x1 = axis[k], axis[k+1]
    y0, y1 = values[k], values[k+1]
    return Float64(y0 + (t - x0) / (x1 - x0) * (y1 - y0))
end
