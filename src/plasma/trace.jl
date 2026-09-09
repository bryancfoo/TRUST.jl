# PlasmaTrace: a Plasma template, a named time axis, and a table of per-bin values for the
# leaves declared time-varying. plasma(trace, j) materializes the snapshot at bin j, allocating
# nothing (spec D7). conventions.md Section 4 (OQ-12, OQ-16); spec D4-D8.

"""
    LeafPath

A leaf's SI path, `n_e`, `electrons.<name>.T`, `electrons.<name>.f.<p>`, and so on, stored as
its dot-separated segments. `LeafPath(s::AbstractString)` splits `s` on `.`; `LeafPath(segs...)`
takes the segments directly. Exported: Phase 3 addresses leaves through it. Used at construction
and in reporting, never inside the per-bin kernel.

Spec D4, D5.
"""
struct LeafPath
    segments::Vector{String}
    function LeafPath(segments::Vector{String})
        (isempty(segments) || any(isempty, segments)) && throw(ArgumentError(
            "\"$(join(segments, "."))\" is not a valid leaf path"))
        return new(segments)
    end
end

LeafPath(s::AbstractString) = LeafPath(String.(split(s, '.')))
LeafPath(segs::AbstractString...) = LeafPath(String[String(s) for s in segs])

Base.string(p::LeafPath) = join(p.segments, ".")
Base.show(io::IO, p::LeafPath) = print(io, string(p))
Base.:(==)(a::LeafPath, b::LeafPath) = a.segments == b.segments
Base.hash(p::LeafPath, h::UInt) = hash(p.segments, hash(:LeafPath, h))

"""
    LeafRef

The resolved form of a `LeafPath` against one `Plasma`: which set (`:n_e`, `:electrons` or
`:ions`), which position in that `SpeciesSet` (`0` when `set === :n_e`), which field (`:T`,
`:u`, `:fraction`, or a shape-parameter key), and whether that field is a shape parameter
(`shape`). `isbits`, unexported, built once at construction and never in the per-bin kernel.

Spec D5.
"""
struct LeafRef
    set::Symbol
    index::Int
    field::Symbol
    shape::Bool
end

"""
    _resolve_leaf(p::Plasma, path::LeafPath) -> LeafRef

Validate `path` against `p` and resolve it to a `LeafRef`. Throws (spec D28) for a first
segment other than `"n_e"`, `"electrons"` or `"ions"`; a species name not in the named set; a
field other than `T`, `u`, `fraction` or `f.<parameter>`; `mass` or `charge` (not fittable
leaves); a shape parameter not a key of `parameters(sp.f)`; or the wrong number of segments.

Spec D5, D28.
"""
function _resolve_leaf(p::Plasma, path::LeafPath)
    segs = path.segments
    root = segs[1]
    if root == "n_e"
        length(segs) == 1 || throw(ArgumentError(
            "\"$(string(path))\" is not a leaf; expected T, u, fraction or f.<parameter>"))
        return LeafRef(:n_e, 0, :n_e, false)
    elseif root == "electrons" || root == "ions"
        set = root == "electrons" ? (:electrons) : (:ions)
        length(segs) >= 3 || throw(ArgumentError(
            "\"$(string(path))\" is not a leaf; expected T, u, fraction or f.<parameter>"))
        name = segs[2]
        species_set = set === :electrons ? p.electrons : p.ions
        idx = findfirst(==(name), species_set.names)
        idx === nothing && throw(ArgumentError(
            "\"$(string(path))\" has no species named \"$name\""))
        rest = segs[3:end]
        if length(rest) == 1
            field = rest[1]
            field in ("mass", "charge") && throw(ArgumentError(
                "\"$(string(path))\": mass and charge are not fittable leaves"))
            field in ("T", "u", "fraction") || throw(ArgumentError(
                "\"$(string(path))\" is not a leaf; expected T, u, fraction or f.<parameter>"))
            return LeafRef(set, idx, Symbol(field), false)
        elseif length(rest) == 2 && rest[1] == "f"
            pname = rest[2]
            sp = species_set[idx]
            haskey(parameters(sp.f), Symbol(pname)) || throw(ArgumentError(
                "\"$pname\" is not a parameter of $(typeof(sp.f))"))
            return LeafRef(set, idx, Symbol(pname), true)
        else
            throw(ArgumentError(
                "\"$(string(path))\" is not a leaf; expected T, u, fraction or f.<parameter>"))
        end
    else
        throw(ArgumentError(
            "\"$(string(path))\" must begin with \"n_e\", \"electrons\" or \"ions\""))
    end
end

"""
    _leaf_si_path(key::AbstractString) -> (si_path::String, factor::Float64)

Convert a boundary leaf key (the SI path with its last segment optionally replaced by a unit
keyword from `ACCEPTED_UNITS`) into the SI path and the multiplicative factor to SI. If the last
segment is found in exactly one `ACCEPTED_UNITS` row, the SI segment is that row's own name
(`:T_eV` -> `"T"`) and the factor is that keyword's; if found in no row, the segment is already
SI or dimensionless and the factor is `1.0`; if found in more than one row, that is an error (it
cannot happen with the rows that exist, and the check exists so a future collision is loud).

Spec D4, D28.
"""
function _leaf_si_path(key::AbstractString)
    segs = String.(split(key, '.'))
    (isempty(key) || any(isempty, segs)) && throw(ArgumentError("\"$key\" is not a valid leaf path"))
    last_sym = Symbol(segs[end])
    matches = Symbol[]
    for (rowname, row) in pairs(ACCEPTED_UNITS)
        haskey(row, last_sym) && push!(matches, rowname)
    end
    if isempty(matches)
        return join(segs, "."), 1.0
    elseif length(matches) == 1
        row = getfield(ACCEPTED_UNITS, matches[1])
        factor = getfield(row, last_sym)
        si_segs = copy(segs)
        si_segs[end] = String(matches[1])
        return join(si_segs, "."), Float64(factor)
    else
        throw(ArgumentError("\"$key\" is ambiguous across unit rows: $matches"))
    end
end

"""
    SpeciesRows{NP}

The inverted index for one species: the row of `PlasmaTrace.values` holding its flow, its
temperature, its fraction (`0` when that leaf is constant), and one row per entry of
`parameters(sp.f)`, in that `NamedTuple`'s key order (`0` for an entry that is constant). `NP`
is `length(parameters(sp.f))`.

Spec D6.
"""
struct SpeciesRows{NP}
    u::Int
    T::Int
    fraction::Int
    shape::NTuple{NP,Int}
end

"""
    PlasmaTrace{FT<:Real, P<:Plasma{FT}, A<:AbstractVector{FT}, M<:AbstractMatrix{FT}, E<:Tuple, I<:Tuple}

A time-dependent plasma: a template `Plasma` whose species, shapes, masses and charges are
fixed; a named axis (`axis_name`, SI values in `axis`, strictly increasing); the SI leaf paths
tabulated (`paths`, row order) and their values (`values`, `N_leaf x N_t`, SI); and the row plan
`plasma(trace, j)` walks (`n_e_row`, `electron_rows`, `ion_rows`). `paths`/`values` and the row
fields are two views of the same information built together at construction and never diverging.

Built with three keyword constructors: from per-bin vectors or functions of the axis
(`leaves = key => value` pairs, boundary keys, spec D4/D6); from a vector of snapshot `Plasma`s
(`leaves` an iterable of SI path strings to tabulate, everything else required constant); or
with no `leaves` at all, the constant trace, `values` a `0 x N_t` matrix.

`fraction` rows are **not** checked to sum to one per bin (unlike `Plasma`'s own keyword
constructor, which does): a fit may pass through a slightly non-neutral state, and
`neutrality_residual` already reports it. This asymmetry is deliberate.

`conventions.md` Section 4 (OQ-12, OQ-16); spec D6.
"""
struct PlasmaTrace{FT<:Real,P<:Plasma{FT},A<:AbstractVector{FT},M<:AbstractMatrix{FT},E<:Tuple,I<:Tuple}
    template::P
    axis_name::Symbol
    axis::A
    paths::Vector{LeafPath}
    values::M
    n_e_row::Int
    electron_rows::E
    ion_rows::I
end

Base.eltype(::Type{<:PlasmaTrace{FT}}) where {FT} = FT

"""
    nbins(trace::PlasmaTrace) -> Int

The number of bins on `trace`'s axis.
"""
nbins(trace::PlasmaTrace) = length(trace.axis)
Base.length(trace::PlasmaTrace) = nbins(trace)

function _check_axis(axis::AbstractVector)
    isempty(axis) && throw(ArgumentError("axis must have at least one bin"))
    for j in 2:length(axis)
        axis[j] > axis[j-1] || throw(ArgumentError(
            "axis must be strictly increasing; axis[$(j-1)] = $(axis[j-1]), axis[$j] = $(axis[j])"))
    end
    return nothing
end

function _axis_kwarg(kwargs)
    row = getfield(ACCEPTED_UNITS, :t)
    present = Tuple(k for k in keys(row) if haskey(kwargs, k))
    if length(present) == 0
        throw(ArgumentError("no unit keyword given for t; expected exactly one of $(keys(row))"))
    elseif length(present) > 1
        throw(ArgumentError("more than one unit keyword given for t: $present; expected exactly one"))
    end
    key = present[1]
    return kwargs[key], Float64(getfield(row, key))
end

function _to_si(raw, factor, key, j)
    local v
    try
        v = raw * factor
    catch
        throw(ArgumentError(
            "\"$key\"[$j] = $raw does not convert to the template's element type"))
    end
    v isa Real || throw(ArgumentError(
        "\"$key\"[$j] = $v does not convert to the template's element type"))
    return v
end

function _to_ft(::Type{FT}, si_value, key, j) where {FT}
    try
        return FT(si_value)
    catch
        throw(ArgumentError(
            "\"$key\"[$j] = $si_value does not convert to the template's element type $FT"))
    end
end

"""
    _species_rows(sp::Species, set::Symbol, idx::Int, refs::Vector{LeafRef}) -> SpeciesRows

Scan `refs` for the rows belonging to species `idx` of `set`, in `sp`'s own shape-parameter key
order. Construction-time only (not allocation-free, and does not need to be).
"""
function _species_rows(sp::Species, set::Symbol, idx::Int, refs::Vector{LeafRef})
    pnames = keys(parameters(sp.f))
    NP = length(pnames)
    u_row = 0
    T_row = 0
    frac_row = 0
    shape_rows = zeros(Int, NP)
    for i in eachindex(refs)
        r = refs[i]
        (r.set === set && r.index == idx) || continue
        if r.shape
            k = findfirst(==(r.field), pnames)
            k !== nothing && (shape_rows[k] = i)
        elseif r.field === :T
            T_row = i
        elseif r.field === :u
            u_row = i
        elseif r.field === :fraction
            frac_row = i
        end
    end
    return SpeciesRows{NP}(u_row, T_row, frac_row, Tuple(shape_rows))
end

function _finish_trace(template::Plasma{FT}, axis_name::Symbol, axis_si::Vector{FT},
                        paths::Vector{LeafPath}, refs::Vector{LeafRef},
                        values_mat::Matrix{FT}) where {FT}
    n_e_row = 0
    for i in eachindex(refs)
        refs[i].set === :n_e && (n_e_row = i)
    end
    electron_rows = ntuple(s -> _species_rows(template.electrons[s], :electrons, s, refs),
                            length(template.electrons))
    ion_rows = ntuple(s -> _species_rows(template.ions[s], :ions, s, refs), length(template.ions))
    return PlasmaTrace{FT,typeof(template),typeof(axis_si),typeof(values_mat),
                        typeof(electron_rows),typeof(ion_rows)}(
        template, axis_name, axis_si, paths, values_mat, n_e_row, electron_rows, ion_rows)
end

"""
    PlasmaTrace(template::Plasma; axis_name::Symbol = :t, t_ns | t_s, leaves = ())

Build a trace from per-bin vectors and/or functions of the axis. `leaves` is an iterable of
`key => value` pairs; `key` is a boundary leaf key (spec D4); `value` is either an
`AbstractVector` of length `N_t` (converted elementwise, `values[row,j] = FT(v[j] * factor)`) or
a callable `g`, evaluated as `g(axis_boundary[j]) * factor` where `axis_boundary` is the axis
**in the unit of the keyword that supplied it** (nanoseconds for `t_ns`, seconds for `t_s`) —
everything at this boundary is in the unit its own keyword names. Duplicate keys resolving to
the same SI path are an error; the row order is the order `leaves` is given in.

Spec D6, D28.
"""
function _widen_shape(f::AbstractVelocityDistribution, ::Type{FT}) where {FT}
    p = parameters(f)
    isempty(p) && return f
    return reconstruct(f, NamedTuple{keys(p)}(FT.(values(p))))
end

function _widen_species(sp::Species, ::Type{FT}) where {FT}
    newf = _widen_shape(sp.f, FT)
    return Species{FT,typeof(newf)}(sp.name, FT(sp.mass), sp.charge, FT(sp.u), FT(sp.T), newf,
        FT(sp.fraction), FT(sp.n))
end

"""
    _widen_plasma(p::Plasma, ::Type{FT}) -> Plasma{FT,...}

`p` converted to element type `FT`, or `p` itself, unchanged (`===`), when it already has that
type. Used by the vectors/functions `PlasmaTrace` constructor so that a template built at
`Float64` can still carry a dual-valued table: the constructor determines the *promoted* `FT`
from the template and every leaf value given, exactly as `Plasma`'s own keyword constructor
promotes `FT` over its species and scalar fields, and widens the template to match before the
`PlasmaTrace` is built — `PlasmaTrace{FT, P<:Plasma{FT}, ...}` requires the stored template to
share the trace's own element type.
"""
function _widen_plasma(p::Plasma, ::Type{FT}) where {FT}
    typeof(p.n_e) === FT && return p
    new_electrons = map(sp -> _widen_species(sp, FT), p.electrons.species)
    new_ions = map(sp -> _widen_species(sp, FT), p.ions.species)
    electron_set = SpeciesSet(p.electrons.names, new_electrons)
    ion_set = SpeciesSet(p.ions.names, new_ions)
    return Plasma{FT,typeof(electron_set),typeof(ion_set)}(FT(p.n_e), electron_set, ion_set)
end

function PlasmaTrace(template::Plasma; axis_name::Symbol=:t, leaves=(), kwargs...)
    axis_raw, axis_factor = _axis_kwarg(kwargs)
    nt = length(axis_raw)

    pairs_list = collect(leaves)
    n_leaf = length(pairs_list)
    paths = Vector{LeafPath}(undef, n_leaf)
    refs = Vector{LeafRef}(undef, n_leaf)
    rows = Vector{Any}(undef, n_leaf)
    keys_given = Vector{Any}(undef, n_leaf)
    seen = Set{String}()

    FT = typeof(template.n_e)

    for (i, kv) in enumerate(pairs_list)
        key, val = kv
        si_path, factor = _leaf_si_path(String(key))
        si_path in seen && throw(ArgumentError("\"$si_path\" is given twice"))
        push!(seen, si_path)
        path = LeafPath(si_path)
        ref = _resolve_leaf(template, path)
        paths[i] = path
        refs[i] = ref
        keys_given[i] = key
        if val isa AbstractVector
            length(val) == nt || throw(ArgumentError(
                "\"$key\" has length $(length(val)) but the axis has $nt bins"))
            row = [_to_si(val[j], factor, key, j) for j in 1:nt]
        else
            row = [_to_si(val(axis_raw[j]), factor, key, j) for j in 1:nt]
        end
        rows[i] = row
        FT = promote_type(FT, eltype(row))
    end
    FT = float(FT)

    axis_si = FT[FT(axis_raw[j] * axis_factor) for j in 1:nt]
    _check_axis(axis_si)

    values_mat = Matrix{FT}(undef, n_leaf, nt)
    for i in 1:n_leaf, j in 1:nt
        values_mat[i, j] = _to_ft(FT, rows[i][j], keys_given[i], j)
    end

    template_ft = _widen_plasma(template, FT)
    return _finish_trace(template_ft, axis_name, axis_si, paths, refs, values_mat)
end

function _enumerate_leaves(template::Plasma)
    out = Tuple{LeafRef,LeafPath}[]
    push!(out, (LeafRef(:n_e, 0, :n_e, false), LeafPath(["n_e"])))
    for (setsym, set) in ((:electrons, template.electrons), (:ions, template.ions))
        for idx in 1:length(set)
            sp = set[idx]
            nm = set.names[idx]
            push!(out, (LeafRef(setsym, idx, :T, false), LeafPath([String(setsym), nm, "T"])))
            push!(out, (LeafRef(setsym, idx, :u, false), LeafPath([String(setsym), nm, "u"])))
            push!(out, (LeafRef(setsym, idx, :fraction, false),
                        LeafPath([String(setsym), nm, "fraction"])))
            for pname in keys(parameters(sp.f))
                push!(out, (LeafRef(setsym, idx, pname, true),
                            LeafPath([String(setsym), nm, "f", String(pname)])))
            end
        end
    end
    return out
end

function _check_same_species(template::Plasma, p::Plasma, j::Integer)
    (template.electrons.names == p.electrons.names && template.ions.names == p.ions.names) ||
        throw(ArgumentError("snapshot at bin $j differs from the template's species (names)"))
    for idx in 1:length(template.electrons)
        t_sp, p_sp = template.electrons[idx], p.electrons[idx]
        (typeof(t_sp.f) === typeof(p_sp.f) && t_sp.mass == p_sp.mass && t_sp.charge == p_sp.charge) ||
            throw(ArgumentError(
                "snapshot at bin $j differs from the template's species \"$(t_sp.name)\""))
    end
    for idx in 1:length(template.ions)
        t_sp, p_sp = template.ions[idx], p.ions[idx]
        (typeof(t_sp.f) === typeof(p_sp.f) && t_sp.mass == p_sp.mass && t_sp.charge == p_sp.charge) ||
            throw(ArgumentError(
                "snapshot at bin $j differs from the template's species \"$(t_sp.name)\""))
    end
    return nothing
end

"""
    PlasmaTrace(snapshots::AbstractVector{<:Plasma}; axis_name::Symbol = :t, t_ns | t_s, leaves = ())

Build a trace from a vector of snapshot `Plasma`s, one per bin. `leaves` is an iterable of SI
path strings (not boundary keys) naming the leaves to tabulate; the template is `snapshots[1]`.
Every snapshot must share the template's species names, shape types, masses and charges. A leaf
not named in `leaves` that differs from the template at any bin is an error naming the leaf and
the first bin where it differs (compared with `!=` on the SI value, so a bitwise-constant leaf
passes); this check runs once, at construction.

Spec D6, D28.
"""
function PlasmaTrace(snapshots::AbstractVector{<:Plasma}; axis_name::Symbol=:t, leaves=(), kwargs...)
    axis_raw, axis_factor = _axis_kwarg(kwargs)
    nt = length(snapshots)
    length(axis_raw) == nt || throw(ArgumentError(
        "axis has $(length(axis_raw)) bins but $nt snapshots were given"))
    template = snapshots[1]
    FT = typeof(template.n_e)
    axis_si = FT[FT(axis_raw[j] * axis_factor) for j in 1:nt]
    _check_axis(axis_si)

    for j in 2:nt
        _check_same_species(template, snapshots[j], j)
    end

    wanted_paths = LeafPath[]
    for s in leaves
        path = LeafPath(String(s))
        _resolve_leaf(template, path)
        push!(wanted_paths, path)
    end
    wanted_strs = Set(string.(wanted_paths))

    paths = LeafPath[]
    refs = LeafRef[]
    rows = Vector{FT}[]
    for (ref, path) in _enumerate_leaves(template)
        sp_str = string(path)
        if sp_str in wanted_strs
            vals = FT[FT(_leaf_value(snapshots[j], ref)) for j in 1:nt]
            push!(paths, path)
            push!(refs, ref)
            push!(rows, vals)
        else
            v0 = _leaf_value(template, ref)
            for j in 2:nt
                vj = _leaf_value(snapshots[j], ref)
                vj != v0 && throw(ArgumentError(
                    "\"$sp_str\" varies at bin $j but is not in the leaves list"))
            end
        end
    end

    n_leaf = length(paths)
    values_mat = Matrix{FT}(undef, n_leaf, nt)
    for i in 1:n_leaf, j in 1:nt
        values_mat[i, j] = rows[i][j]
    end

    return _finish_trace(template, axis_name, axis_si, paths, refs, values_mat)
end

"""
    _leaf_row(trace::PlasmaTrace, path::LeafPath) -> Int

The row of `trace.values` holding `path`, or `0` if `path` is constant on this trace.
"""
function _leaf_row(trace::PlasmaTrace, path::LeafPath)
    i = findfirst(==(path), trace.paths)
    return i === nothing ? 0 : i
end

"""
    _leaf_value(p::Plasma, ref::LeafRef) -> Real

The SI value of the leaf `ref` names, read from `p`.
"""
function _leaf_value(p::Plasma, ref::LeafRef)
    ref.set === :n_e && return p.n_e
    sp = ref.set === :electrons ? p.electrons[ref.index] : p.ions[ref.index]
    ref.shape && return parameters(sp.f)[ref.field]
    ref.field === :T && return sp.T
    ref.field === :u && return sp.u
    return sp.fraction
end

_density(fraction, n_e, kind::Symbol, charge) =
    kind === :electrons ? fraction * n_e : fraction * n_e / charge

function _redensify(sp::Species{FT,F}, n_e::FT, kind::Symbol) where {FT,F}
    n = _density(sp.fraction, n_e, kind, sp.charge)
    return Species{FT,F}(sp.name, sp.mass, sp.charge, sp.u, sp.T, sp.f, sp.fraction, n)
end

function _apply_field(sp::Species{FT,F}, ref::LeafRef, v::FT) where {FT,F}
    if ref.shape
        pr = parameters(sp.f)
        newp = merge(pr, NamedTuple{(ref.field,)}((v,)))
        newf = reconstruct(sp.f, newp)
        return Species{FT,F}(sp.name, sp.mass, sp.charge, sp.u, sp.T, newf, sp.fraction, sp.n)
    elseif ref.field === :T
        return Species{FT,F}(sp.name, sp.mass, sp.charge, sp.u, v, sp.f, sp.fraction, sp.n)
    elseif ref.field === :u
        return Species{FT,F}(sp.name, sp.mass, sp.charge, v, sp.T, sp.f, sp.fraction, sp.n)
    else
        return Species{FT,F}(sp.name, sp.mass, sp.charge, sp.u, sp.T, sp.f, v, sp.n)
    end
end

function _maybe_update_species(sp::Species{FT,F}, i::Int, ref::LeafRef, kind::Symbol,
                                v::FT, n_e::FT) where {FT,F}
    if ref.set === kind && ref.index == i
        return _redensify(_apply_field(sp, ref, v), n_e, kind)
    end
    return sp
end

"""
    _with_leaf(p::Plasma, ref::LeafRef, value) -> Plasma

A `Plasma` equal to `p` but for the one leaf `ref` names, replaced by `value` (SI), with every
species' density re-resolved from `n_e` and its (possibly just-changed) fraction, exactly as
`plasma(trace, j)` does. `value`'s type need not match `p`'s own element type — a dual `value`
(from a fitted `GradientSmearing` width, say) widens `p` first (`_widen_plasma`, the same
promotion `PlasmaTrace`'s constructor does), so the widening is a compile-time-resolved no-op
(and hence allocation-free) whenever `value`'s type already matches `p`'s. Allocation-free in
that common case by the same argument as `plasma(trace, j)`: `ntuple` over each `SpeciesSet`'s
tuple, one position of which conditionally rewrites its species while every other position
passes its species through unchanged — never a shape-type change, so every position's static
return type is fixed regardless of which branch runs. What `GradientSmearing` calls (spec D10);
EF9 asserts the zero for the plain (non-widening) pass.

Spec D8.
"""
function _with_leaf(p::Plasma, ref::LeafRef, value)
    FT = promote_type(typeof(p.n_e), typeof(value))
    p2 = _widen_plasma(p, FT)
    v = FT(value)
    if ref.set === :n_e
        new_electrons = map(sp -> _redensify(sp, v, :electrons), p2.electrons.species)
        new_ions = map(sp -> _redensify(sp, v, :ions), p2.ions.species)
        electron_set = SpeciesSet(p2.electrons.names, new_electrons)
        ion_set = SpeciesSet(p2.ions.names, new_ions)
        return Plasma{FT,typeof(electron_set),typeof(ion_set)}(v, electron_set, ion_set)
    end

    kind = ref.set
    if kind === :electrons
        new_species = ntuple(i -> _maybe_update_species(p2.electrons.species[i], i, ref, kind, v, p2.n_e),
                              Val(length(p2.electrons)))
        electron_set = SpeciesSet(p2.electrons.names, new_species)
        return Plasma{FT,typeof(electron_set),typeof(p2.ions)}(p2.n_e, electron_set, p2.ions)
    else
        new_species = ntuple(i -> _maybe_update_species(p2.ions.species[i], i, ref, kind, v, p2.n_e),
                              Val(length(p2.ions)))
        ion_set = SpeciesSet(p2.ions.names, new_species)
        return Plasma{FT,typeof(p2.electrons),typeof(ion_set)}(p2.n_e, p2.electrons, ion_set)
    end
end

_snapshot_shape(f, ::Tuple{}, values, j) = f
function _snapshot_shape(f, rows::NTuple{NP,Int}, values, j) where {NP}
    all(iszero, rows) && return f
    p = parameters(f)
    vals = ntuple(i -> rows[i] == 0 ? p[i] : values[rows[i], j], Val(NP))
    return reconstruct(f, NamedTuple{keys(p)}(vals))
end

function _snapshot_species(sp::Species{FT}, r::SpeciesRows, values::AbstractMatrix{FT},
                            j::Integer, n_e::FT, kind::Symbol) where {FT}
    u = r.u == 0 ? sp.u : values[r.u, j]
    T = r.T == 0 ? sp.T : values[r.T, j]
    fraction = r.fraction == 0 ? sp.fraction : values[r.fraction, j]
    f = _snapshot_shape(sp.f, r.shape, values, j)
    n = _density(fraction, n_e, kind, sp.charge)
    return Species{FT,typeof(f)}(sp.name, sp.mass, sp.charge, u, T, f, fraction, n)
end

"""
    plasma(trace::PlasmaTrace, j::Integer) -> Plasma

The plasma snapshot at bin `j`: every leaf `trace` tabulates read from `trace.values[:, j]`,
every other field copied from `trace.template`. **Every species' density is re-resolved from
`n_e` and its fraction**, exactly as `Plasma`'s own keyword constructor does (`n = fraction *
n_e` for an electron species, `fraction * n_e / charge` for an ion) — a snapshot that copied the
template's `n` would be wrong the moment `n_e` or a fraction varies. Built with `ntuple`/`map`
over the template's tuples and the inner constructors of `Species`, `SpeciesSet` and `Plasma`
directly (the template was validated once, at `Plasma`'s own construction), so it allocates
nothing, checked by `@allocated` in TR14.

Spec D7, Pitfall 6.
"""
function plasma(trace::PlasmaTrace{FT}, j::Integer) where {FT}
    (1 <= j <= nbins(trace)) || throw(ArgumentError(
        "bin $j is out of range: trace has $(nbins(trace)) bins"))
    template = trace.template
    values = trace.values

    n_e = trace.n_e_row == 0 ? template.n_e : values[trace.n_e_row, j]

    electron_species = ntuple(Val(length(template.electrons))) do i
        _snapshot_species(template.electrons.species[i], trace.electron_rows[i], values, j, n_e, :electrons)
    end
    ion_species = ntuple(Val(length(template.ions))) do i
        _snapshot_species(template.ions.species[i], trace.ion_rows[i], values, j, n_e, :ions)
    end

    electron_set = SpeciesSet(template.electrons.names, electron_species)
    ion_set = SpeciesSet(template.ions.names, ion_species)
    return Plasma{FT,typeof(electron_set),typeof(ion_set)}(n_e, electron_set, ion_set)
end

# --- show (spec D29) ---

function Base.show(io::IO, trace::PlasmaTrace)
    print(io, "PlasmaTrace($(length(trace.paths)) leaves, $(nbins(trace)) bins of :$(trace.axis_name))")
end

function _leaf_display_unit(path::LeafPath)
    sym = Symbol(path.segments[end])
    if hasproperty(ACCEPTED_UNITS, sym)
        row = getfield(ACCEPTED_UNITS, sym)
        key = first(keys(row))
        return String(key), Float64(getfield(row, key))
    end
    return "", 1.0
end

function Base.show(io::IO, ::MIME"text/plain", trace::PlasmaTrace)
    println(io, "PlasmaTrace")
    println(io, "  axis (:$(trace.axis_name)): $(t_from_si(trace.axis[1], :t_ns)) to " *
                "$(t_from_si(trace.axis[end], :t_ns)) ns, $(nbins(trace)) bins")
    for (i, path) in enumerate(trace.paths)
        row = @view trace.values[i, :]
        lo, hi = extrema(row)
        unit_key, factor = _leaf_display_unit(path)
        suffix = isempty(unit_key) ? "" : " $unit_key"
        println(io, "  $(string(path)): $(lo / factor) to $(hi / factor)$suffix")
    end
    print(io, "  template: ")
    show(io, trace.template)
    println(io)
end
