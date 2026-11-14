# write_streak/read_streak (JLD2, plain arrays and strings) and write_trace/read_trace.
# docs/derivations/data_layout.md; spec D19, D20.

using JLD2: jldopen

"""
    STREAK_LAYOUT_VERSION

The `layout_version` string written to every streak JLD2 file and checked on read.

Spec D19, D45.
"""
const STREAK_LAYOUT_VERSION = "trust-streak-1"

"""
    TRACE_LAYOUT_VERSION

The `layout_version` string written to every trace JLD2 file and checked on read.
**`"trust-trace-3"` since Step G.2.2** (Phase G.2 spec D11, [OQ-69](../../dev/open_questions.md#oq-69)
item 4): a species carrying a lab-frame distribution stores its grid and default weights and no `u` or
`T`. `"trust-trace-2"` (Step 4.J.1, Phase 4.J spec D9: a species' `charge` written `Float64`, where
`trust-trace-1` wrote `Int64`) is still read, since a file with no gridded species has the same content
in both (`TRACE_LAYOUTS_READ`); `read_trace` refuses any other version, naming the ones it reads.

Spec D20, D45; Phase 4.J spec D9; Phase G.2 spec D11.
"""
const TRACE_LAYOUT_VERSION = "trust-trace-3"

"""
    TRACE_LAYOUTS_READ

The trace layouts `read_trace` accepts, oldest first: `("trust-trace-2", "trust-trace-3")` (Phase G.2
spec D11, [OQ-69](../../dev/open_questions.md#oq-69) item 4). A file's `layout_version` is compared after
`_canonical_version`, so `"otis-trace-2"` reads as `"trust-trace-2"`.
"""
const TRACE_LAYOUTS_READ = ("trust-trace-2", TRACE_LAYOUT_VERSION)

"""
    _canonical_version(v)

Canonicalizes a `layout_version` or `deck_version` string read from a file: a string that begins
with `"otis-"`, the package's name before 2026-10-18, is returned with that prefix replaced by
`"trust-"`; any other value, including `nothing`, is returned unchanged. The number is never
touched, so an old spelling of a *different* layout number is refused exactly as its new spelling
is. Writers never emit the old prefix. (Level 1 rename decision D3, 2026-10-18.)
"""
_canonical_version(v::AbstractString) = startswith(v, "otis-") ? "trust-" * SubString(v, 6) : String(v)
_canonical_version(v) = v

# Keys written explicitly by _write_meta! and _write_probe_collection_meta!: an sd.meta entry
# under one of these names (e.g. one synthesize() already populated) is superseded, not
# duplicated -- write_streak's explicit source/probe/collection arguments win.
const _RESERVED_META_KEYS = ("source", "created_unix", "trust_version",
    "lambda_probe_nm", "theta_deg", "polarization_deg", "k_model")

"""
    _empty_marker(s::AbstractString) -> String

`s`, or `"-"` when `s` is empty (Phase 4.N spec D20, `conventions.md` Section 4's rule): no writer of the
package writes an empty scalar string dataset, whose zero-size string type h5py cannot read
(`H5Tget_size` fails), while an array holding `""` reads. Every reader turns the marker back into `""`
(`_from_empty_marker`), so a caller sees `""` as before and a file written before Phase 4.N reads the same.
"""
_empty_marker(s::AbstractString) = isempty(s) ? "-" : String(s)

"""
    _from_empty_marker(v) -> v

The readers' half of `_empty_marker` (Phase 4.N spec D20): `""` where a scalar string dataset holds the
marker `"-"`, every other value as it is. A string deliberately written as `"-"` therefore reads back as
`""` (the user's choice of 2026-10-16, Phase 4.N summary U2).
"""
_from_empty_marker(v) = (v isa AbstractString && v == "-") ? "" : v

function _write_meta!(f, sd::StreakData, source::AbstractString)
    f["meta/source"] = _empty_marker(source)
    f["meta/created_unix"] = Float64(time())
    f["meta/trust_version"] = string(pkgversion(TRUST))
    for (k, v) in sd.meta
        k in _RESERVED_META_KEYS && continue
        f["meta/$k"] = v isa AbstractArray ? collect(v) : v isa AbstractString ? _empty_marker(v) : v
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
the pixel), and a `meta` group (`source`, `created_unix`, `trust_version`, the probe/collection
geometry keys when given, and every entry of `sd.meta`). Arrays are written in Julia's
column-major order, **wavelength first**, so an h5py reader sees the transpose `(N_t, N_λ)`. An empty
`source`, or an empty string value in `sd.meta`, is written as `"-"` (Phase 4.N spec D20: h5py cannot read
an empty scalar string), and `read_streak` returns it as `""`.

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
        meta[k] = _from_empty_marker(f["meta/$k"])
    end
    return meta
end

"""
    read_streak(path::AbstractString) -> StreakData

Reads a streak written by `write_streak`. Throws, naming the dataset and the path, if a required
dataset is missing; throws, naming the version found and expected, if `layout_version` does not
match `STREAK_LAYOUT_VERSION`. Converts `lambda_nm`/`t_ns` to SI; converts `mask` back to `Bool`;
every `meta/*` entry becomes an entry of the returned `StreakData.meta` under its own key, a scalar string
`"-"` as `""` (`write_streak` writes an empty string as `"-"`, Phase 4.N spec D20). A `Bool` (or an array
of them) reads back as a `Bool`.

Spec D19; Phase 4.N spec D20.
"""
function read_streak(path::AbstractString)
    jldopen(path, "r") do f
        version = _canonical_version(_require_dataset(f, "layout_version", path))
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

# The registry name a species group stores: the distribution's type name, of either family (Phase G.2 spec D11).
_shape_name(f::Union{AbstractVelocityDistribution, AbstractLabDistribution}) = string(nameof(typeof(f)))

# A shape's species stores its u and T, inputs; a lab-frame distribution's species does not, since they are
# computed from the distribution (Phase G.2 spec D11, OQ-69 item 4: nothing derived is stored).
function _write_u_T!(f, prefix::AbstractString, ::AbstractVelocityDistribution, sp)
    f["$prefix/u_ms"] = Float64(sp.u)
    f["$prefix/T_J"] = Float64(sp.T)
    return nothing
end
_write_u_T!(f, prefix::AbstractString, ::AbstractLabDistribution, sp) = nothing

# A shape has no grid; a GaussianGrid writes its own (Phase G.2 spec D11): the centre, spacing and width in
# m/s, the node count, the symmetry flag (UInt8, 1 = symmetric), the default weights, and the node
# velocities in m/s, redundant with the first three, for a reader without the package.
_write_grid!(f, prefix::AbstractString, ::AbstractVelocityDistribution) = nothing
function _write_grid!(f, prefix::AbstractString, g::GaussianGrid{FT, Nv, NP}) where {FT, Nv, NP}
    f["$prefix/grid/u0_ms"] = g.u0
    f["$prefix/grid/dv_ms"] = g.dv
    f["$prefix/grid/sigma_ms"] = g.sigma
    f["$prefix/grid/nodes"] = Int64(Nv)
    f["$prefix/grid/symmetric"] = UInt8(NP < Nv)
    f["$prefix/grid/default_weights"] = collect(g.w0)
    f["$prefix/grid/node_velocities_ms"] = [_grid_node(g, n) for n in 1:Nv]
    return nothing
end

function _write_species_group!(f, prefix::AbstractString, sp)
    f["$prefix/mass_kg"] = Float64(sp.mass)
    f["$prefix/charge"] = Float64(sp.charge)
    _write_u_T!(f, prefix, sp.f, sp)
    f["$prefix/fraction"] = Float64(sp.fraction)
    f["$prefix/shape"] = _shape_name(sp.f)
    p = parameters(sp.f)
    for k in keys(p)
        f["$prefix/shape_p/$k"] = Float64(p[k])
    end
    _write_grid!(f, prefix, sp.f)
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
mass, charge, flow, temperature, fraction, shape name and shape parameters). `read_trace(path; as =
:trace)` rebuilds the trace through `SHAPE_REGISTRY`. An empty `source` is written as `"-"` (Phase 4.N
spec D20). **The layout is `trust-trace-3`** (Phase G.2 spec D11): a species carrying a lab-frame
distribution stores its registry name under `shape`, its template log-weights under `shape_p`, and its
grid under `grid/` (`u0_ms`, `dv_ms`, `sigma_ms`, `nodes`, `symmetric`, `default_weights`,
`node_velocities_ms`), and **no `u_ms` and no `T_J`**, which are computed from the distribution; its
per-bin log-weights are rows of `values`, as every leaf is (`docs/derivations/data_layout.md` Section 2).

Spec D20; Phase 4.N spec D20; Phase G.2 spec D11.
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
        f["meta/source"] = _empty_marker(source)
        f["meta/created_unix"] = Float64(time())
        f["meta/trust_version"] = string(pkgversion(TRUST))
    end
    return path
end

function _read_species_group(f, prefix::AbstractString)
    d = Dict{String,Any}()
    d["mass_kg"] = f["$prefix/mass_kg"]
    d["charge"] = f["$prefix/charge"]
    # Phase G.2 spec D11: a gridded species stores no u_ms or T_J, and a grid/ group instead
    haskey(f, "$prefix/u_ms") && (d["u_ms"] = f["$prefix/u_ms"])
    haskey(f, "$prefix/T_J") && (d["T_J"] = f["$prefix/T_J"])
    d["fraction"] = f["$prefix/fraction"]
    d["shape"] = f["$prefix/shape"]
    shape_p = Dict{String,Float64}()
    if haskey(f, "$prefix/shape_p")
        for k in keys(f["$prefix/shape_p"])
            shape_p[k] = f["$prefix/shape_p/$k"]
        end
    end
    d["shape_p"] = shape_p
    if haskey(f, "$prefix/grid")
        grid = Dict{String,Any}()
        for k in keys(f["$prefix/grid"])
            grid[k] = f["$prefix/grid/$k"]
        end
        d["grid"] = grid
    end
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
    read_trace(path::AbstractString; as::Symbol = :arrays) -> NamedTuple or PlasmaTrace

Reads a trace written by `write_trace`, in either layout of `TRACE_LAYOUTS_READ` (`trust-trace-3`, and
`trust-trace-2`, whose files hold no gridded species and read exactly as before; Phase G.2 spec D11).
`as = :arrays` (the default, Phase 2's behaviour, unchanged) returns `(; layout_version, axis_name, axis,
axis_unit, paths, values, template)` with `template` a `Dict{String,Any}` of the group's contents, every
entry as stored (a gridded species' dictionary has no `"u_ms"` or `"T_J"` and has a `"grid"` dictionary);
the round trip (IO6) is on these arrays, bit for bit, not on a reconstructed object.

`as = :trace` (Step 3.6, spec D35) rebuilds a `PlasmaTrace`: each species' distribution is looked up by
name in `SHAPE_REGISTRY` and built from its `shape_p` (and, for a `GaussianGrid`, its `grid`, the grid,
the default weights and every log-weight to the bit), the species themselves through `electrons`/
`ion`, the template through `Plasma`'s own keyword constructor, and the leaf table through the
package-internal `_finish_trace` directly (not the public `PlasmaTrace(template; t_ns|t_s, ...)`
constructor, whose axis keyword is hard-coded to the `:t` unit row regardless of `axis_name` —
`axis` here is already SI, so no such keyword is needed). An unknown `as` is a loud error naming
both accepted values.

Spec D20, D35; Phase G.2 spec D11.
"""
function read_trace(path::AbstractString; as::Symbol=:arrays)
    as === :arrays || as === :trace || throw(ArgumentError(
        "read_trace: as must be :arrays or :trace; got :$as"))
    arrays = jldopen(path, "r") do f
        layout_version = _canonical_version(_require_dataset(f, "layout_version", path))
        layout_version in TRACE_LAYOUTS_READ || throw(ArgumentError(
            "read_trace: \"$path\" has layout_version \"$layout_version\"; expected " *
            join(("\"$v\"" for v in TRACE_LAYOUTS_READ), " or ")))
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
    as === :arrays && return arrays
    return _plasma_trace_from_arrays(arrays)
end

"""
    _build_shape(ctor, shape_p::AbstractDict) -> AbstractVelocityDistribution

Builds the shape `ctor` (a `SHAPE_REGISTRY` value) names from its stored parameters, by dispatch
on `ctor` — never an `if name == ...` chain (`conventions.md` 7.2). Add a method here when
`SHAPE_REGISTRY` gains an entry whose shape takes parameters. A lab-frame distribution is built by
`_build_distribution` from the whole species dictionary, since it stores more than its parameters.
"""
_build_shape(::Type{Maxwellian}, ::AbstractDict) = Maxwellian()
_build_shape(::Type{SuperGaussian}, shape_p::AbstractDict) = SuperGaussian(Float64(shape_p["m"]))
_build_shape(::Type{Kappa}, shape_p::AbstractDict) = Kappa(Float64(shape_p["kappa"]))
_build_shape(::Type{GeneralizedNormal}, shape_p::AbstractDict) = GeneralizedNormal(Float64(shape_p["m"]))
_build_shape(::Type{SkewNormal}, shape_p::AbstractDict) = SkewNormal(Float64(shape_p["alpha"]))
_build_shape(::Type{HermiteExpansion}, shape_p::AbstractDict) =
    HermiteExpansion(Float64(shape_p["c3"]), Float64(shape_p["c4"]), Float64(shape_p["c5"]), Float64(shape_p["c6"]))

"""
    _build_distribution(ctor, d::AbstractDict) -> distribution

The distribution `ctor` (a `SHAPE_REGISTRY` value) names, from one species dictionary of
`_read_species_group` (Phase G.2 spec D12): a shape from `d["shape_p"]` through `_build_shape`; a
`GaussianGrid` from `d["grid"]` and `d["shape_p"]`, its centre, spacing, width, default weights and
log-weights exactly as stored (through the inner constructor, so nothing is recomputed). A grid whose
`nodes` is not the length of its `default_weights` is refused.
"""
_build_distribution(ctor::Type{<:AbstractVelocityDistribution}, d::AbstractDict) = _build_shape(ctor, d["shape_p"])

function _build_distribution(::Type{GaussianGrid}, d::AbstractDict)
    grid = d["grid"]
    w0 = Vector{Float64}(grid["default_weights"])
    Int(grid["nodes"]) == length(w0) || throw(ArgumentError(
        "read_trace: a GaussianGrid's grid/nodes is $(grid["nodes"]) but its grid/default_weights has $(length(w0)) entries"))
    g0 = _gaussian_grid(Float64(grid["u0_ms"]), Float64(grid["dv_ms"]), Float64(grid["sigma_ms"]), Tuple(w0),
                        grid["symmetric"] != 0)
    return _grid_from_stored(g0, d["shape_p"])
end

# The stored log-weights g1, g2, … onto a grid (spec D12), behind a function barrier on the node count.
function _grid_from_stored(g0::GaussianGrid{Float64, Nv, NP}, shape_p::AbstractDict) where {Nv, NP}
    length(shape_p) == NP || throw(ArgumentError(
        "read_trace: a GaussianGrid of $Nv nodes has $NP parameters; the file stores $(length(shape_p))"))
    names = _grid_names(Val(NP))
    return reconstruct(g0, NamedTuple{names}(ntuple(k -> Float64(shape_p[String(names[k])]), Val(NP))))
end

"""
    _shape_from_dict(d::AbstractDict) -> distribution

`d`'s `"shape"` name looked up in `SHAPE_REGISTRY` (a loud error naming the shape and the valid
names if absent), then built from `d` through `_build_distribution`.
"""
function _shape_from_dict(d::AbstractDict)
    name = d["shape"]
    haskey(SHAPE_REGISTRY, name) || throw(ArgumentError(
        "read_trace: unknown shape \"$name\"; valid shapes are $(sort(collect(keys(SHAPE_REGISTRY))))"))
    return _build_distribution(SHAPE_REGISTRY[name], d)
end

# The keywords a species builder takes beside its distribution (Phase G.2 spec D12): a shape's T and u as
# stored; nothing for a lab-frame distribution, whose T and u its builder computes.
_species_u_T_from_dict(::AbstractVelocityDistribution, d::AbstractDict) = (T_eV = d["T_J"] / EV, u_ms = d["u_ms"])
_species_u_T_from_dict(::AbstractLabDistribution, d::AbstractDict) = NamedTuple()

"""
    _species_spec_from_dict(name, d::AbstractDict, kind::Symbol) -> SpeciesSpec

One electron or ion `SpeciesSpec` (`kind === :electrons` or `:ions`) from `read_trace`'s
`_read_species_group` dictionary `d`, through `electrons`/`ion`'s own keyword constructors —
`T_eV` from `d["T_J"] / EV` (the one place this file converts a unit), `u_ms`, `fraction` and the
shape passed through directly (already SI). Electron mass is not read from `d["mass_kg"]`:
`electrons` always uses `ELECTRON_MASS`, matching how the mass was written in the first place. A
species carrying a lab-frame distribution is built from its distribution and fraction alone, its `T`
and `u` computed by the builder (`_species_u_T_from_dict`; Phase G.2 spec D12).
"""
function _species_spec_from_dict(name::AbstractString, d::AbstractDict, kind::Symbol)
    f = _shape_from_dict(d)
    uT = _species_u_T_from_dict(f, d)
    if kind === :electrons
        return electrons(; name=name, uT..., fraction=d["fraction"], f=f)
    else
        return ion(; name=name, mass_kg=d["mass_kg"], charge=d["charge"], uT..., fraction=d["fraction"], f=f)
    end
end

"""
    _plasma_template_from_dict(template::Dict{String,Any}) -> Plasma

The template `Plasma` `read_trace(; as = :trace)` rebuilds, from `read_trace(; as = :arrays)`'s
own `template` dictionary, through `Plasma`'s keyword constructor (D35).
"""
function _plasma_template_from_dict(template::Dict{String,Any})
    electrons_group = template["electrons"]
    ions_group = template["ions"]
    electron_specs = [_species_spec_from_dict(nm, electrons_group["species"][nm], :electrons)
                      for nm in electrons_group["names"]]
    ion_specs = [_species_spec_from_dict(nm, ions_group["species"][nm], :ions)
                 for nm in ions_group["names"]]
    return Plasma(; n_e_m3=template["n_e_m3"], electrons=electron_specs, ions=ion_specs)
end

"""
    _plasma_trace_from_arrays(arrays::NamedTuple) -> PlasmaTrace

Rebuilds the `PlasmaTrace` `read_trace(; as = :trace)` returns from the plain arrays
`read_trace(; as = :arrays)` reads: the template through `_plasma_template_from_dict`, each
leaf's `LeafRef` through `_resolve_leaf` against it, and the object itself through the
package-internal `_finish_trace` (spec D35's docstring explains why not the public
`PlasmaTrace(template; ...)` constructor).
"""
function _plasma_trace_from_arrays(arrays::NamedTuple)
    template = _plasma_template_from_dict(arrays.template)
    FT = typeof(template.n_e)
    paths = LeafPath.(arrays.paths)
    refs = [_resolve_leaf(template, p) for p in paths]
    axis_si = Vector{FT}(arrays.axis)
    values_mat = Matrix{FT}(arrays.values)
    return _finish_trace(template, arrays.axis_name, axis_si, paths, refs, values_mat)
end
