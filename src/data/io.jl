# write_streak/read_streak (JLD2, plain arrays and strings) and write_trace/read_trace.
# docs/derivations/data_layout.md; spec D19, D20.

using JLD2: jldopen

"""
    STREAK_LAYOUT_VERSION

The `layout_version` string written to every streak JLD2 file and checked on read.

Spec D19, D45.
"""
const STREAK_LAYOUT_VERSION = "otis-streak-1"

"""
    TRACE_LAYOUT_VERSION

The `layout_version` string written to every trace JLD2 file and checked on read.

Spec D20, D45.
"""
const TRACE_LAYOUT_VERSION = "otis-trace-1"


# Keys written explicitly by _write_meta! and _write_probe_collection_meta!: an sd.meta entry
# under one of these names (e.g. one synthesize() already populated) is superseded, not
# duplicated -- write_streak's explicit source/probe/collection arguments win.
const _RESERVED_META_KEYS = ("source", "created_unix", "otis_version",
    "lambda_probe_nm", "theta_deg", "polarization_deg", "k_model")

function _write_meta!(f, sd::StreakData, source::AbstractString)
    f["meta/source"] = String(source)
    f["meta/created_unix"] = Float64(time())
    f["meta/otis_version"] = string(pkgversion(OTIS))
    for (k, v) in sd.meta
        k in _RESERVED_META_KEYS && continue
        f["meta/$k"] = v isa AbstractArray ? collect(v) : v
    end
    return nothing
end

function _write_probe_collection_meta!(f, probe, collection)
    f["meta/lambda_probe_nm"] = lambda_from_si(probe.lambda, :lambda_nm)
    f["meta/theta_deg"] = theta_from_si(collection.theta, :theta_deg)
    f["meta/polarization_deg"] = theta_from_si(probe.polarization, :theta_deg)
    f["meta/k_model"] = string(collection.k_model)
    return nothing
end

"""
    write_streak(path::AbstractString, sd::StreakData; source = "",
                 probe::Union{Nothing,ProbeBeam} = nothing,
                 collection::Union{Nothing,Collection} = nothing) -> path

Writes `sd` to `path` as a JLD2 file of plain numeric arrays and strings (`docs/derivations/data_layout.md`):
`layout_version`, `lambda_nm`, `t_ns`, `image`, optionally `sigma` and `mask` (`UInt8`, `1` = use
the pixel), and a `meta` group (`source`, `created_unix`, `otis_version`, the probe/collection
geometry keys when given, and every entry of `sd.meta`). Arrays are written in Julia's
column-major order, **wavelength first**, so an h5py reader sees the transpose `(N_t, N_λ)`.

`sd`'s element type must be a plain float (`Float64`, `Float32`, ...; converted to `Float64` in
the file) — a **dual** streak is a computation, not data, and throws naming its element type
(IO10).

Spec D19.
"""
function write_streak(path::AbstractString, sd::StreakData; source="",
    probe::Union{Nothing,ProbeBeam}=nothing, collection::Union{Nothing,Collection}=nothing)
    eltype(sd.lambda) <: AbstractFloat || throw(ArgumentError(
        "write_streak: streak element type must be a plain float (not a dual); got $(eltype(sd.lambda))"))
    jldopen(path, "w") do f
        f["layout_version"] = STREAK_LAYOUT_VERSION
        f["lambda_nm"] = lambda_from_si.(sd.lambda, :lambda_nm)
        f["t_ns"] = t_from_si.(sd.t, :t_ns)
        f["image"] = Matrix{Float64}(sd.image)
        sd.sigma !== nothing && (f["sigma"] = Matrix{Float64}(sd.sigma))
        sd.mask !== nothing && (f["mask"] = UInt8.(sd.mask))
        _write_meta!(f, sd, source)
        (probe !== nothing && collection !== nothing) && _write_probe_collection_meta!(f, probe, collection)
    end
    return path
end

function _require_dataset(f, name::AbstractString, path::AbstractString)
    haskey(f, name) || throw(ArgumentError(
        "read_streak: \"$path\" is missing the required dataset \"$name\""))
    return f[name]
end

function _read_meta(f)
    meta = Dict{String,Any}()
    haskey(f, "meta") || return meta
    for k in keys(f["meta"])
        meta[k] = f["meta/$k"]
    end
    return meta
end

"""
    read_streak(path::AbstractString) -> StreakData

Reads a streak written by `write_streak`. Throws, naming the dataset and the path, if a required
dataset is missing; throws, naming the version found and expected, if `layout_version` does not
match `STREAK_LAYOUT_VERSION`. Converts `lambda_nm`/`t_ns` to SI; converts `mask` back to `Bool`;
every `meta/*` entry becomes an entry of the returned `StreakData.meta` under its own key.

Spec D19.
"""
function read_streak(path::AbstractString)
    jldopen(path, "r") do f
        version = _require_dataset(f, "layout_version", path)
        version == STREAK_LAYOUT_VERSION || throw(ArgumentError(
            "read_streak: \"$path\" has layout_version \"$version\"; expected \"$STREAK_LAYOUT_VERSION\""))
        lambda_nm = _require_dataset(f, "lambda_nm", path)
        t_ns = _require_dataset(f, "t_ns", path)
        image = _require_dataset(f, "image", path)
        lambda = lambda_to_si.(lambda_nm, :lambda_nm)
        t = t_to_si.(t_ns, :t_ns)
        size(image) == (length(lambda), length(t)) || throw(ArgumentError(
            "read_streak: \"$path\" image is $(size(image)) but the axes give " *
            "$(length(lambda)) x $(length(t))"))
        sigma = haskey(f, "sigma") ? Matrix{Float64}(f["sigma"]) : nothing
        mask = haskey(f, "mask") ? Matrix{Bool}(f["mask"] .!= 0) : nothing
        meta = _read_meta(f)
        return StreakData(; lambda_m=lambda, t_s=t, image=Matrix{Float64}(image), sigma=sigma,
            mask=mask, meta=meta)
    end
end

_shape_name(f::AbstractVelocityDistribution) = string(nameof(typeof(f)))

function _write_species_group!(f, prefix::AbstractString, sp)
    f["$prefix/mass_kg"] = Float64(sp.mass)
    f["$prefix/charge"] = Int64(sp.charge)
    f["$prefix/u_ms"] = Float64(sp.u)
    f["$prefix/T_J"] = Float64(sp.T)
    f["$prefix/fraction"] = Float64(sp.fraction)
    f["$prefix/shape"] = _shape_name(sp.f)
    p = parameters(sp.f)
    for k in keys(p)
        f["$prefix/shape_p/$k"] = Float64(p[k])
    end
    return nothing
end

function _write_species_set!(f, prefix::AbstractString, set)
    f["$prefix/names"] = collect(String.(set.names))
    for (i, nm) in enumerate(set.names)
        _write_species_group!(f, "$prefix/$nm", set[i])
    end
    return nothing
end

"""
    write_trace(path::AbstractString, trace::PlasmaTrace; source = "") -> path

Writes the trace's tabulated values (`docs/derivations/data_layout.md`): `layout_version`,
`axis_name`, `axis` (SI), `axis_unit`, `paths` (the SI leaf paths, row order), `values` (`N_leaf
x N_t`, SI), and a `template` group describing `trace.template` (density, and, per species,
mass, charge, flow, temperature, fraction, shape name and shape parameters). Does **not** write
enough to reconstruct a `PlasmaTrace` (that needs the component registry of Step 3.6);
`read_trace` returns plain arrays.

Spec D20.
"""
function write_trace(path::AbstractString, trace::PlasmaTrace; source="")
    jldopen(path, "w") do f
        f["layout_version"] = TRACE_LAYOUT_VERSION
        f["axis_name"] = String(trace.axis_name)
        f["axis"] = Vector{Float64}(trace.axis)
        f["axis_unit"] = trace.axis_name === :t ? "s" : "SI"
        f["paths"] = String.(string.(trace.paths))
        f["values"] = Matrix{Float64}(trace.values)
        f["template/n_e_m3"] = Float64(trace.template.n_e)
        _write_species_set!(f, "template/electrons", trace.template.electrons)
        _write_species_set!(f, "template/ions", trace.template.ions)
        f["meta/source"] = String(source)
        f["meta/created_unix"] = Float64(time())
        f["meta/otis_version"] = string(pkgversion(OTIS))
    end
    return path
end

function _read_species_group(f, prefix::AbstractString)
    d = Dict{String,Any}()
    d["mass_kg"] = f["$prefix/mass_kg"]
    d["charge"] = f["$prefix/charge"]
    d["u_ms"] = f["$prefix/u_ms"]
    d["T_J"] = f["$prefix/T_J"]
    d["fraction"] = f["$prefix/fraction"]
    d["shape"] = f["$prefix/shape"]
    shape_p = Dict{String,Float64}()
    if haskey(f, "$prefix/shape_p")
        for k in keys(f["$prefix/shape_p"])
            shape_p[k] = f["$prefix/shape_p/$k"]
        end
    end
    d["shape_p"] = shape_p
    return d
end

function _read_species_set(f, prefix::AbstractString)
    names = f["$prefix/names"]
    d = Dict{String,Any}()
    for nm in names
        d[nm] = _read_species_group(f, "$prefix/$nm")
    end
    return Dict{String,Any}("names" => names, "species" => d)
end

"""
    read_trace(path::AbstractString) -> NamedTuple

Reads a trace written by `write_trace`, returning `(; layout_version, axis_name, axis,
axis_unit, paths, values, template)` with `template` a `Dict{String,Any}` of the group's
contents. **Does not reconstruct a `PlasmaTrace`** — the round trip (IO6) is on these arrays,
bit for bit, not on a reconstructed object; that step is Step 3.6, once the component registry
exists.

Spec D20.
"""
function read_trace(path::AbstractString)
    jldopen(path, "r") do f
        layout_version = _require_dataset(f, "layout_version", path)
        layout_version == TRACE_LAYOUT_VERSION || throw(ArgumentError(
            "read_trace: \"$path\" has layout_version \"$layout_version\"; expected \"$TRACE_LAYOUT_VERSION\""))
        axis_name = _require_dataset(f, "axis_name", path)
        axis = _require_dataset(f, "axis", path)
        axis_unit = _require_dataset(f, "axis_unit", path)
        paths = _require_dataset(f, "paths", path)
        values = _require_dataset(f, "values", path)
        template = Dict{String,Any}(
            "n_e_m3" => f["template/n_e_m3"],
            "electrons" => _read_species_set(f, "template/electrons"),
            "ions" => _read_species_set(f, "template/ions"),
        )
        return (; layout_version, axis_name=Symbol(axis_name), axis, axis_unit, paths, values, template)
    end
end
