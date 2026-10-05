# The TOML deck: read_deck, build_problem, effective_deck, fit -- Step 3.6b, chunk 12 of Phase 3.
# Spec D33. This is how the package is driven day to day: `fit("deck.toml")`.
#
# The one idea this file turns on (conventions.md 7.6, D9): a variable is plugged into any
# numeric key of any section by writing its name as a string where a number is expected. Which
# keys are numeric is read from the option structs' own field types (D32, D9's corrected
# <:AbstractFloat rule) -- never from a hand-written list of key names -- but the reference
# mechanism itself is scoped to the two sections whose values become ParameterSpec bindings,
# [plasma] and [instrument]: nowhere else does a deck value ever reach a Binding, so nowhere
# else can a string sensibly mean "read this fitted value here". [optimizer], [regularizer],
# [fit.variables] and [fit.intervals] therefore fill every option-struct field the same way but
# never accept a reference (see the box above _fill_option_fields).
#
# read_deck does essentially all of the validation: every key checked against the component that
# owns it, every component name checked against its registry, every variable reference checked
# against the declared variable/dependent names -- D39's table attributes every read_deck error
# to this function, not to build_problem. It returns a Deck whose `sections` field is the parsed
# TOML tree with every numeric-and-referencing string already replaced by a `_VarRef` marker, and
# every option-struct field already filled with its own default when the deck omitted it. This is
# what makes effective_deck (every default filled, 7.5) almost free: it is mostly a
# re-serialization of what read_deck already computed, plus the three things D33 says the input
# deck does not say (a variable's resolved binding, a dependent's text verbatim -- already
# present -- and the axis unit).
#
# build_problem walks the same, already-validated tree and does the actual construction: the
# template Plasma, the instrument chain, the Experiment, the ParameterSpec (which resolves every
# binding for real, D9), the Regularizer and the optimizer.

import TOML
using LinearAlgebra: PosDefException
using Printf: @sprintf

const DECK_VERSION = "otis-deck-2"

# ------------------------------------------------------------------------------------------
# _VarRef: a numeric-key string that names a declared variable or dependent
# ------------------------------------------------------------------------------------------

struct _VarRef
    name::String
end
Base.:(==)(a::_VarRef, b::_VarRef) = a.name == b.name
Base.isequal(a::_VarRef, b::_VarRef) = isequal(a.name, b.name)
Base.hash(v::_VarRef, h::UInt) = hash(v.name, hash(_VarRef, h))

_to_toml_value(v::_VarRef) = v.name
_to_toml_value(d::AbstractDict) = Dict{String,Any}(String(k) => _to_toml_value(v) for (k, v) in d)
_to_toml_value(v::AbstractVector) = Any[_to_toml_value(x) for x in v]
_to_toml_value(v) = v

# ------------------------------------------------------------------------------------------
# Deck
# ------------------------------------------------------------------------------------------

"""
    Deck

A parsed, validated, still-inert description of a fit (spec D33): `deck_version`, and `sections`,
the deck's own nested table with every unknown key and unknown component name already refused
(7.3), every numeric key's string value already checked to name a declared variable or dependent
and wrapped in an internal `_VarRef` marker (D9, 7.6), and every option-struct field already
filled with its own default when the input omitted it (7.5). Nothing here is a Julia physics
object yet -- that is `build_problem`'s job.

Not given an explicit struct by D33, which describes only the five functions; this shape (the
validated TOML tree plus the version string, rather than a bespoke field per section) is Level
3's choice under rule 11 zero, flagged in the chunk 12 report: it is what makes `effective_deck`
close to a re-serialization of what `read_deck` already computed, and what makes DK1's
field-for-field round trip a plain (`isequal`-based, see `==` below) `Dict` comparison.
"""
struct Deck
    deck_version::String
    sections::Dict{String,Any}
end

"""
    Base.:(==)(a::Deck, b::Deck)

Field-for-field structural equality (DK1), via `isequal` rather than `==`: `LBFGS`'s own default
`time_limit = NaN` (D27) is a legitimate defaulted value that round-trips bit for bit, but
IEEE 754 makes `NaN == NaN` false, which would make `==` (and the plain `Dict` `==` it would
otherwise inherit) fail a correct round trip. `isequal` treats `NaN` as equal to itself, which is
the "are these two structures identical" question DK1 actually asks.
"""
Base.:(==)(a::Deck, b::Deck) = a.deck_version == b.deck_version && isequal(a.sections, b.sections)

function Base.show(io::IO, deck::Deck)
    print(io, "Deck(", deck.deck_version, ", ", length(deck.sections), " sections)")
end

# ------------------------------------------------------------------------------------------
# Generic key/value validation, shared by every section
# ------------------------------------------------------------------------------------------

function _check_known_keys(given::AbstractDict, valid, section::AbstractString)
    validset = Set(String.(valid))
    for k in keys(given)
        k in validset || throw(ArgumentError(
            "read_deck: unknown key \"$k\" in section \"$section\"; valid keys are " *
            "$(sort(collect(validset)))"))
    end
    return nothing
end

function _check_registry_name(registry::AbstractDict, name::AbstractString, kind::AbstractString)
    haskey(registry, name) || throw(ArgumentError(
        "read_deck: unknown $kind \"$name\"; valid names are $(sort(collect(keys(registry))))"))
    return registry[name]
end

"""
    _fill_numeric_scalar(v, known_names, section, key, refs_allowed, refs_reason = "") -> Float64 or _VarRef

One numeric deck value (D9, 7.6): a `Real` becomes `Float64`; a `String` is a variable reference
when `refs_allowed`, checked against `known_names` (D39's two read_deck fragments), and is
otherwise a loud error naming the key -- the generic "takes a string, not a variable reference"
message, unless `refs_reason` is non-empty (D12, chunk 4), in which case that text is thrown
instead (`gradient_smearing.width`'s "its conversion factor is the smeared leaf's own" reason;
every other caller leaves `refs_reason` empty, so no other refusal's text moves). Shared by every
numeric key this file resolves, whether it belongs to an option struct (`_fill_option_fields`) or,
like a plasma species' `T_eV`, does not.
"""
function _fill_numeric_scalar(v, known_names::AbstractSet{String}, section::AbstractString,
                               key::AbstractString, refs_allowed::Bool,
                               refs_reason::AbstractString="")
    if v isa AbstractString
        if !refs_allowed
            throw(ArgumentError(isempty(refs_reason) ?
                "read_deck: key \"$key\" in section \"$section\" takes a string, not a variable reference" :
                refs_reason))
        end
        v in known_names || throw(ArgumentError(
            "read_deck: key \"$key\" in section \"$section\" refers to \"$v\", which is not a " *
            "declared variable or dependent"))
        return _VarRef(v)
    elseif v isa Real
        return Float64(v)
    else
        throw(ArgumentError(
            "read_deck: key \"$key\" in section \"$section\" must be a number" *
            (refs_allowed ? " or a variable reference naming a fitted variable or dependent" : "") *
            "; got $(repr(v))"))
    end
end

"""
    _fill_option_field(v, FT::Type, known_names, section, key, refs_allowed, refs_reason = "")

One field of an option struct, dispatched on its declared type (D9's corrected rule, D32): an
`<:AbstractFloat` field (or a vector of one) goes through `_fill_numeric_scalar`, element by
element for a vector; a `<:Bool` field (D41) is checked and returned as-is, *before* the
`<:Integer` branch below -- `Bool <: Integer` in Julia, so a branch written after the integer test
would never run; an `<:Integer` field is never a reference (`nodes`, `offsets`: D9's box);
`Symbol`, `String` fields, and every other declared type, are literal and untouched -- this is
what makes DK5 hold: a string in a non-numeric key stays a string even when a variable of that
name exists, because the field's own declared type, not the value, decides.
"""
function _fill_option_field(v, ::Type{FT}, known_names::AbstractSet{String},
                             section::AbstractString, key::AbstractString,
                             refs_allowed::Bool, refs_reason::AbstractString="") where {FT}
    if FT <: AbstractFloat
        return _fill_numeric_scalar(v, known_names, section, key, refs_allowed, refs_reason)
    elseif FT <: Bool
        v isa Bool || throw(ArgumentError(
            "read_deck: key \"$key\" in section \"$section\" must be true or false; got $(repr(v))"))
        return v
    elseif FT <: Integer
        v isa Integer || throw(ArgumentError(
            "read_deck: key \"$key\" in section \"$section\" must be an integer; got $(repr(v))"))
        return Int(v)
    elseif FT <: Union{AbstractVector,Tuple}
        v isa AbstractVector || throw(ArgumentError(
            "read_deck: key \"$key\" in section \"$section\" must be a list"))
        et = eltype(FT)
        if et <: AbstractFloat
            return Any[_fill_numeric_scalar(x, known_names, section, key, refs_allowed, refs_reason) for x in v]
        elseif et <: Integer
            return Int[Int(x) for x in v]
        else
            return collect(v)
        end
    else
        return v
    end
end

"""
    _fill_option_fields(::Type{T}, given::AbstractDict, known_names, section, refs_allowed, refs_reason = "") -> Dict{String,Any}

Every field of the option struct `T` (D32), present: `given`'s own value where `given` has the
key, `T`'s own default (read off a defaults-only instance, `T()`, since every option struct in
`options.jl` gives every field a default) otherwise. An unknown key in `given` is a loud error
naming the section, the key and the valid names (7.3, DK7). This is the one place the classifier
runs off `fieldtype` rather than a hand-written list (D32's own instruction), and it is reused for
every option-struct-backed section: shapes, the six single instrument effects, the four
backgrounds, and the three optimizers.
"""
function _fill_option_fields(::Type{T}, given::AbstractDict, known_names::AbstractSet{String},
                              section::AbstractString, refs_allowed::Bool,
                              refs_reason::AbstractString="") where {T}
    _check_known_keys(given, String.(fieldnames(T)), section)
    default_inst = T()
    out = Dict{String,Any}()
    for f in fieldnames(T)
        fname = String(f)
        FT = fieldtype(T, f)
        out[fname] = haskey(given, fname) ?
            _fill_option_field(given[fname], FT, known_names, section, fname, refs_allowed, refs_reason) :
            getfield(default_inst, f)
    end
    return out
end

# > **Effect-parameter bindings read their conversion factor from `_INSTRUMENT_BINDING`**
# > (D9, chunk 4, 2026-09-23; replaces the box Level 3 wrote at Phase 3 chunk 12, quoted at the
# > end of this box). `_INSTRUMENT_BINDING` is a
# > `Dict{String,Dict{String,Tuple{Symbol,Float64,String}}}`, `regname => (deck key =>
# > (parameters(effect) field, factor, unit_key))`, defined beside `_INSTRUMENT_OPTIONS` below.
# > `_build_effect_and_pending!` looks the deck key up in this table instead of assuming the deck
# > key already equals the `parameters(effect)` field name, so every unit-bearing key converts
# > correctly at `_resolve_bindings` (`src/inversion/parameters.jl`, D10) instead of silently
# > taking `factor = 1.0`: `WavelengthCalibrationOptions.lambda_offset_nm` (`parameters` key
# > `offset`, factor `1e-9`), `KSmearingOptions.half_angle_deg` (`parameters` key `half_angle`,
# > factor `pi/180`), and `SpectralIRFOptions.fwhm_nm`/`TemporalIRFOptions.fwhm_ns` (`parameters`
# > key `sigma`, factor `1e-9 * _SIGMA_PER_FWHM` = `1e-9/(2*sqrt(2*log(2)))`, D9's
# > `_SIGMA_PER_FWHM`). Both the value path (`_rebuild_effect`, `src/inversion/parameters.jl`) and
# > the Jacobian path (`_build_chain_globals`, `src/inversion/jacobian.jl`, D11a) multiply by
# > `Binding.factor`, so the two agree.
# >
# > **`GradientSmearingOptions.width` is the one key still refused**, with a message naming why
# > (`_fill_option_fields`'s `refs_reason` argument, D12): its conversion factor is the smeared
# > leaf's own (`_leaf_si_path(e.leaf)[2]`), which is not a constant of the key and so cannot live
# > in this static table. `_INSTRUMENT_REF_OK`, the old four-component safe subset, is gone;
# > `haskey(_INSTRUMENT_BINDING, regname)` is the whole test now.
# >
# > **What this box said through chunk 3** (Phase 3 chunk 12, quoted so the correction is on the
# > record): effect-parameter bindings were scoped to the four keys whose deck name already
# > equalled the `parameters(effect)` field name (`Sensitivity.scale`, `ConstantBackground.level`,
# > `PolynomialBackground.coefficients`, `RationalBackground.numerator`/`denominator`), with the
# > other five refused outright. It also said, of `SpectralIRFOptions.fwhm_nm` and
# > `TemporalIRFOptions.fwhm_ns`, that "no factor fixes it, since `parameters`' key `sigma` is not
# > even the same quantity as `fwhm`". **That was wrong**: `_sigma_from_fwhm(fwhm) = fwhm /
# > (2*sqrt(2*log(2)))` is a purely multiplicative map, so a single constant factor is exactly
# > right, and this chunk's `_INSTRUMENT_BINDING` row for both keys proves it.

# ------------------------------------------------------------------------------------------
# [data]
# ------------------------------------------------------------------------------------------

# The regions (D15) moved to a shared [[fit.regions]] at chunk 7 (D40 amendment 1): a [data]
# table's own "regions" key is now refused, naming the new home. The four cross-section loud
# errors of D15 (both regions and t_range_ns; a shift/fraction key without regions; a region's
# interval clashing with [fit.intervals]; two or more channels without regions) all move to
# read_deck itself (D40 amendment 2, item 10), since they need to see every channel's data table
# and the shared [[fit.regions]] together. _validate_data_section keeps only the checks that are
# purely local to one [data]/[channel.<n>.data] table.
const _DATA_KEYS = ("file", "lambda_range_nm", "t_range_ns", "t_shift_ns", "t_shift_s",
    "min_usable_fraction")
const _REGION_KEYS = ("t_range_ns", "t_range_s", "nbins", "interval")
const _REGION_RANGE_KEYS = ("t_range_ns", "t_range_s")

"""
    _validate_data_section(given::AbstractDict) -> Dict{String,Any}

Validates one `[data]` or `[channel.<n>.data]` table (D15, moved to `[[fit.regions]]` at chunk 7,
D40 amendment 1). A `"regions"` key is refused before the generic known-key check runs, naming its
new home (`[[fit.regions]]`); the two-loud-errors-in-one-table check that remains local to this
table is `t_shift_ns` and `t_shift_s` together (the unit table's own error, surfaced early here as
well as later through `_si_value`). The cross-section loud errors -- both `regions` and
`t_range_ns`; a shift/fraction key given without `[[fit.regions]]`; a region's interval clashing
with `[fit.intervals]`; two or more channels without `[[fit.regions]]` -- and the
`min_usable_fraction` default are `read_deck`'s own job now (D40 amendment 2, items 10-11), since
none of them can be decided from one table alone.

Spec D15, D40.
"""
function _validate_data_section(given::AbstractDict)
    haskey(given, "regions") && throw(ArgumentError(
        "read_deck: \"regions\" moved from [data] to [[fit.regions]] on 2026-09-23; write each " *
        "region as a [[fit.regions]] table"))
    _check_known_keys(given, _DATA_KEYS, "data")
    haskey(given, "file") || throw(ArgumentError("read_deck: section \"data\" needs a \"file\" key"))
    given["file"] isa AbstractString || throw(ArgumentError(
        "read_deck: key \"file\" in section \"data\" must be a string"))
    for k in ("lambda_range_nm", "t_range_ns")
        haskey(given, k) || continue
        v = given[k]
        (v isa AbstractVector && length(v) == 2 && all(x -> x isa Real, v)) || throw(ArgumentError(
            "read_deck: key \"$k\" in section \"data\" must be a two-element [lo, hi] of numbers"))
    end

    (haskey(given, "t_shift_ns") && haskey(given, "t_shift_s")) && throw(ArgumentError(
        "read_deck: section \"data\" gives both \"t_shift_ns\" and \"t_shift_s\"; give exactly one"))

    return Dict{String,Any}(given)
end

"""
    _validate_regions(given) -> Union{Nothing,Vector{Dict{String,Any}}}

Validates `[[fit.regions]]` (D40 amendment 1, moved here from `_validate_data_section`'s old
`"regions"` handling): `given` is `nothing` when the deck has no `[[fit.regions]]`, in which case
this returns `nothing`; otherwise an array of tables, each carrying exactly the unit key given
(`t_range_ns`|`t_range_s`, a two-element `Any[Float64, Float64]`), `nbins::Int`, and `interval`
filled with `""` when absent -- the same shape `[[data.regions]]` stored through chunk 6.

Spec D15, D40.
"""
function _validate_regions(given)
    given === nothing && return nothing
    given isa AbstractVector || throw(ArgumentError(
        "read_deck: \"fit.regions\" must be an array of tables"))
    regions_out = Vector{Dict{String,Any}}(undef, length(given))
    for (i, entry) in enumerate(given)
        entry isa AbstractDict || throw(ArgumentError(
            "read_deck: \"fit.regions[$i]\" must be a table"))
        _check_known_keys(entry, _REGION_KEYS, "fit.regions[$i]")
        present = [k for k in _REGION_RANGE_KEYS if haskey(entry, k)]
        length(present) == 1 || throw(ArgumentError(
            "read_deck: \"fit.regions[$i]\" needs exactly one of $_REGION_RANGE_KEYS"))
        key = present[1]
        rng = entry[key]
        (rng isa AbstractVector && length(rng) == 2 && all(x -> x isa Real, rng)) ||
            throw(ArgumentError(
                "read_deck: \"fit.regions[$i].$key\" must be a two-element [lo, hi]"))
        haskey(entry, "nbins") || throw(ArgumentError(
            "read_deck: \"fit.regions[$i]\" needs \"nbins\""))
        entry["nbins"] isa Integer || throw(ArgumentError(
            "read_deck: \"fit.regions[$i].nbins\" must be an integer"))
        interval = get(entry, "interval", "")
        interval isa AbstractString || throw(ArgumentError(
            "read_deck: \"fit.regions[$i].interval\" must be a string"))
        regions_out[i] = Dict{String,Any}(
            key => Any[Float64(rng[1]), Float64(rng[2])],
            "nbins" => Int(entry["nbins"]),
            "interval" => String(interval))
    end
    return regions_out
end

function _crop_streak(sd::StreakData, lambda_range_nm, t_range_ns)
    l_idx = if lambda_range_nm === nothing
        collect(eachindex(sd.lambda))
    else
        lo, hi = lambda_to_si(Float64(lambda_range_nm[1]), :lambda_nm),
                 lambda_to_si(Float64(lambda_range_nm[2]), :lambda_nm)
        findall(l -> lo <= l <= hi, sd.lambda)
    end
    t_idx = if t_range_ns === nothing
        collect(eachindex(sd.t))
    else
        lo, hi = t_to_si(Float64(t_range_ns[1]), :t_ns), t_to_si(Float64(t_range_ns[2]), :t_ns)
        findall(t -> lo <= t <= hi, sd.t)
    end
    isempty(l_idx) && throw(ArgumentError("read_deck: \"data.lambda_range_nm\" selects no bin"))
    isempty(t_idx) && throw(ArgumentError("read_deck: \"data.t_range_ns\" selects no bin"))
    (l_idx == eachindex(sd.lambda) && t_idx == eachindex(sd.t)) && return sd
    return StreakData(; lambda_m=sd.lambda[l_idx], t_s=sd.t[t_idx], image=sd.image[l_idx, t_idx],
        sigma=sd.sigma === nothing ? nothing : sd.sigma[l_idx, t_idx],
        mask=sd.mask === nothing ? nothing : sd.mask[l_idx, t_idx], meta=sd.meta)
end

"""
    _build_averaging_regions(regions_raw::AbstractVector{Dict{String,Any}}) -> Vector{AveragingRegion}

One `AveragingRegion` per entry of `_validate_data_section`'s own `"regions"` (each already
carrying exactly one of `t_range_ns`/`t_range_s`, `nbins`, `interval`), in deck order.

Spec D15.
"""
function _build_averaging_regions(regions_raw::AbstractVector)
    regions = AveragingRegion[]
    for r in regions_raw
        key = only(k for k in _REGION_RANGE_KEYS if haskey(r, k))
        push!(regions, AveragingRegion(; nbins=r["nbins"], interval=r["interval"],
            NamedTuple{(Symbol(key),)}((r[key],))...))
    end
    return regions
end

"""
    _averaging_kwargs(data_section::AbstractDict) -> Dict{Symbol,Any}

The keyword arguments `average_regions` needs from `[data]`: `min_usable_fraction` (always
present -- `_validate_data_section` filled its default) and whichever of `t_shift_ns`/`t_shift_s`
the deck gave, if either (neither filled in means zero, D15 amendment 2).
"""
function _averaging_kwargs(data_section::AbstractDict)
    kwargs = Dict{Symbol,Any}(:min_usable_fraction => data_section["min_usable_fraction"])
    haskey(data_section, "t_shift_ns") && (kwargs[:t_shift_ns] = data_section["t_shift_ns"])
    haskey(data_section, "t_shift_s") && (kwargs[:t_shift_s] = data_section["t_shift_s"])
    return kwargs
end

# ------------------------------------------------------------------------------------------
# [geometry] -- fixed in Phase 3 (D33): no key here accepts a variable reference
# ------------------------------------------------------------------------------------------

const _PROBEBEAM_KEYS = ("lambda_probe_nm", "lambda_probe_m", "polarization_deg", "polarization_rad",
    "intensity_wcm2", "intensity_wm2", "diameter_um", "diameter_m")
const _COLLECTION_KEYS = ("theta_deg", "theta_rad", "k_model", "lambda_center_nm", "lambda_center_m",
    "dispersion_nm_px", "dispersion_m_px", "npixels", "reference_pixel")

function _validate_geometry_section(given::AbstractDict)
    _check_known_keys(given, (_PROBEBEAM_KEYS..., _COLLECTION_KEYS...), "geometry")
    for (k, v) in given
        (k != "k_model" && v isa AbstractString) && throw(ArgumentError(
            "read_deck: key \"$k\" in section \"geometry\" cannot be a variable reference; the " *
            "probe and the collection are fixed. Fit the wavelength axis with " *
            "instrument.wavelength_calibration instead."))
    end
    return Dict{String,Any}(given)
end

function _build_geometry(section::AbstractDict)
    probe_kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in section if k in _PROBEBEAM_KEYS)
    probe = ProbeBeam(; probe_kwargs...)
    coll_given = Dict{String,Any}(k => v for (k, v) in section if k in _COLLECTION_KEYS)
    k_model = haskey(coll_given, "k_model") ? Symbol(pop!(coll_given, "k_model")) : :exact
    coll_kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in coll_given)
    collection = Collection(; k_model=k_model, coll_kwargs...)
    return probe, collection
end

# ------------------------------------------------------------------------------------------
# [fit], [fit.intervals], [fit.variables], [fit.dependent]
# ------------------------------------------------------------------------------------------

const _FIT_SCALAR_KEYS = ("sigma", "axis_unit")
const _FIT_INTERVAL_KEYS = ("t_range_ns", "t_range_s")
const _FIT_VARIABLE_KEYS = ("bounds", "init", "time_varying", "vary", "interval", "bindings")

function _validate_fit_scalars(given::AbstractDict)
    _check_known_keys(given, (_FIT_SCALAR_KEYS..., "intervals", "variables", "dependent", "regions"), "fit")
    sigma = get(given, "sigma", "supplied")
    sigma == "supplied" || throw(ArgumentError(
        "read_deck: \"fit.sigma\" must be \"supplied\"; sigma is frozen and supplied with the " *
        "data (make one with sigma(model, streak) before writing the data file)"))
    axis_unit = get(given, "axis_unit", "ns")
    axis_unit in ("ns", "s") || throw(ArgumentError(
        "read_deck: \"fit.axis_unit\" must be \"ns\" or \"s\"; got \"$axis_unit\""))
    return Dict{String,Any}("sigma" => sigma, "axis_unit" => axis_unit)
end

function _validate_intervals_section(given::AbstractDict)
    out = Dict{String,Any}()
    for (nm, entry) in given
        entry isa AbstractDict || throw(ArgumentError(
            "read_deck: \"fit.intervals.$nm\" must be a table"))
        _check_known_keys(entry, _FIT_INTERVAL_KEYS, "fit.intervals.$nm")
        present = [k for k in _FIT_INTERVAL_KEYS if haskey(entry, k)]
        length(present) == 1 || throw(ArgumentError(
            "read_deck: \"fit.intervals.$nm\" needs exactly one of $_FIT_INTERVAL_KEYS"))
        key = present[1]
        rng = entry[key]
        (rng isa AbstractVector && length(rng) == 2) || throw(ArgumentError(
            "read_deck: \"fit.intervals.$nm.$key\" must be a two-element [lo, hi]"))
        out[nm] = Dict{String,Any}(key => Any[Float64(rng[1]), Float64(rng[2])])
    end
    return out
end

function _validate_variable_entry(nm::AbstractString, entry::AbstractDict, interval_names::AbstractSet{String})
    entry isa AbstractDict || throw(ArgumentError("read_deck: \"fit.variables.$nm\" must be a table"))
    _check_known_keys(entry, _FIT_VARIABLE_KEYS, "fit.variables.$nm")
    haskey(entry, "bounds") || throw(ArgumentError("read_deck: \"fit.variables.$nm\" needs \"bounds\""))
    haskey(entry, "init") || throw(ArgumentError("read_deck: \"fit.variables.$nm\" needs \"init\""))
    bounds = entry["bounds"]
    (bounds isa AbstractVector && length(bounds) == 2) || throw(ArgumentError(
        "read_deck: \"fit.variables.$nm.bounds\" must be a two-element [lower, upper]"))
    out = Dict{String,Any}(
        "bounds" => Any[Float64(bounds[1]), Float64(bounds[2])],
        "init" => Float64(entry["init"]),
        "time_varying" => Bool(get(entry, "time_varying", false)),
        "vary" => Bool(get(entry, "vary", true)),
    )
    interval = get(entry, "interval", 0)
    if interval isa AbstractString
        interval in interval_names || throw(ArgumentError(
            "read_deck: \"fit.variables.$nm.interval\" names \"$interval\", which is not a " *
            "declared interval; valid names are $(sort(collect(interval_names)))"))
        out["interval"] = interval
    else
        out["interval"] = Int(interval)
    end
    return out
end

function _validate_dependent_section(given::AbstractDict)
    out = Dict{String,Any}()
    for (nm, text) in given
        text isa AbstractString || throw(ArgumentError(
            "read_deck: \"fit.dependent.$nm\" must be a string"))
        out[nm] = String(text)
    end
    return out
end

# ------------------------------------------------------------------------------------------
# [plasma], [plasma.electrons.*], [plasma.ions.*]
# ------------------------------------------------------------------------------------------

const _PLASMA_TOP_KEYS = ("n_e_cm3", "n_e_m3", "electrons", "ions")
const _ELECTRON_SPECIES_KEYS = ("T_eV", "T_keV", "u_kms", "u_ms", "fraction", "shape")
const _ION_SPECIES_KEYS = (_ELECTRON_SPECIES_KEYS..., "mass_amu", "mass_kg", "charge")
const _SHAPE_OPTIONS = Dict{String,Any}("Maxwellian" => MaxwellianOptions, "SuperGaussian" => SuperGaussianOptions,
                                         "Kappa" => KappaOptions, "GeneralizedNormal" => GeneralizedNormalOptions,
                                         "SkewNormal" => SkewNormalOptions)

function _validate_shape_entry(given::AbstractDict, known_names::AbstractSet{String}, section::AbstractString)
    given isa AbstractDict || throw(ArgumentError("read_deck: \"$section\" must be a table"))
    haskey(given, "type") || throw(ArgumentError("read_deck: \"$section\" needs a \"type\" key"))
    type = given["type"]
    type isa AbstractString || throw(ArgumentError("read_deck: \"$section.type\" must be a string"))
    OptT = _check_registry_name(_SHAPE_OPTIONS, type, "shape")
    rest = Dict(k => v for (k, v) in given if k != "type")
    filled = _fill_option_fields(OptT, rest, known_names, section, true)
    return Dict{String,Any}("type" => type, filled...)
end

function _validate_species_entry(sptable::AbstractDict, valid_keys, section::AbstractString,
                                  known_names::AbstractSet{String}, is_ion::Bool)
    sptable isa AbstractDict || throw(ArgumentError("read_deck: \"$section\" must be a table"))
    _check_known_keys(sptable, valid_keys, section)
    out = Dict{String,Any}()
    for k in ("T_eV", "T_keV", "u_kms", "u_ms", "fraction")
        haskey(sptable, k) || continue
        out[k] = _fill_numeric_scalar(sptable[k], known_names, section, k, true)
    end
    if is_ion
        for k in ("mass_amu", "mass_kg")
            haskey(sptable, k) || continue
            out[k] = _fill_numeric_scalar(sptable[k], known_names, section, k, false)
        end
        if haskey(sptable, "charge")
            sptable["charge"] isa Integer || throw(ArgumentError(
                "read_deck: key \"charge\" in section \"$section\" must be an integer"))
            out["charge"] = Int(sptable["charge"])
        end
    end
    if haskey(sptable, "shape")
        out["shape"] = _validate_shape_entry(sptable["shape"], known_names, "$section.shape")
    end
    return out
end

function _validate_plasma_section(given::AbstractDict, known_names::AbstractSet{String})
    _check_known_keys(given, _PLASMA_TOP_KEYS, "plasma")
    out = Dict{String,Any}()
    ne_keys = [k for k in ("n_e_cm3", "n_e_m3") if haskey(given, k)]
    length(ne_keys) == 1 || throw(ArgumentError(
        "read_deck: section \"plasma\" needs exactly one of \"n_e_cm3\"/\"n_e_m3\""))
    nk = ne_keys[1]
    out[nk] = _fill_numeric_scalar(given[nk], known_names, "plasma", nk, true)

    electrons_given = get(given, "electrons", Dict{String,Any}())
    isempty(electrons_given) && throw(ArgumentError(
        "read_deck: section \"plasma\" needs at least one \"[plasma.electrons.<name>]\" table"))
    out["electrons"] = Dict{String,Any}(
        nm => _validate_species_entry(sp, _ELECTRON_SPECIES_KEYS, "plasma.electrons.$nm",
            known_names, false)
        for (nm, sp) in electrons_given)

    ions_given = get(given, "ions", Dict{String,Any}())
    out["ions"] = Dict{String,Any}(
        nm => _validate_species_entry(sp, _ION_SPECIES_KEYS, "plasma.ions.$nm", known_names, true)
        for (nm, sp) in ions_given)

    return out
end

# ------------------------------------------------------------------------------------------
# [instrument]
# ------------------------------------------------------------------------------------------

const _INSTRUMENT_SINGLE_REGISTRY_NAME = Dict{String,String}(
    "spectral_irf" => "SpectralIRF", "temporal_irf" => "TemporalIRF", "k_smearing" => "KSmearing",
    "gradient_smearing" => "GradientSmearing", "wavelength_calibration" => "WavelengthCalibration",
    "sensitivity" => "Sensitivity", "pixel_integration" => "PixelIntegration",
    "tabulated_k_smearing" => "TabulatedKSmearing")
const _INSTRUMENT_TOP_KEYS = (keys(_INSTRUMENT_SINGLE_REGISTRY_NAME)..., "background")

const _INSTRUMENT_OPTIONS = Dict{String,Any}(
    "SpectralIRF" => SpectralIRFOptions, "TemporalIRF" => TemporalIRFOptions,
    "KSmearing" => KSmearingOptions, "GradientSmearing" => GradientSmearingOptions,
    "WavelengthCalibration" => WavelengthCalibrationOptions, "Sensitivity" => SensitivityOptions,
    "Constant" => ConstantBackgroundOptions, "Polynomial" => PolynomialBackgroundOptions,
    "Rational" => RationalBackgroundOptions, "PixelIntegration" => PixelIntegrationOptions,
    "TabulatedKSmearing" => TabulatedKSmearingOptions)

"""
    _SIGMA_PER_FWHM

`sigma = fwhm * _SIGMA_PER_FWHM = fwhm / (2*sqrt(2*log(2)))` (`_sigma_from_fwhm`,
`src/instrument/effects.jl`): the multiplicative factor the deck's `fwhm_nm`/`fwhm_ns` keys
convert through to reach `SpectralIRF`/`TemporalIRF`'s `parameters(effect)` key `sigma` (D9).
"""
const _SIGMA_PER_FWHM = 1 / (2 * sqrt(2 * log(2)))

"""
    _INSTRUMENT_BINDING

`regname => (deck key => (parameters(effect) field, factor, unit_key))` (D9, closing Phase 3's
B5, the box above `_fill_option_fields`): the one table every effect-parameter binding's
conversion factor is read from, whether the binding comes from a deck (`_build_effect_and_pending!`)
or is hand-built with the five-element target form (`_resolve_bindings`, D10). Only keys that are
fields of the component's own option struct appear, since `_fill_option_fields` already rejects
any other key against `fieldnames(T)` -- which is why there is no `lambda_offset_m`,
`half_angle_rad` or `fwhm_m` row: those are constructor keywords, not deck keys.
`haskey(_INSTRUMENT_BINDING, regname)` is what `refs_allowed` now means for an instrument
component (replacing `_INSTRUMENT_REF_OK`); `GradientSmearing` is the one component absent, D12.
"""
const _INSTRUMENT_BINDING = Dict{String,Dict{String,Tuple{Symbol,Float64,String}}}(
    "WavelengthCalibration" => Dict(
        "lambda_offset_nm" => (:offset, 1e-9, "lambda_offset_nm"),
        "dispersion_scale" => (:dispersion_scale, 1.0, ""),
    ),
    "KSmearing" => Dict(
        "half_angle_deg" => (:half_angle, pi / 180, "half_angle_deg"),
    ),
    "SpectralIRF" => Dict(
        "fwhm_nm" => (:sigma, 1e-9 * _SIGMA_PER_FWHM, "fwhm_nm"),
    ),
    "TemporalIRF" => Dict(
        "fwhm_ns" => (:sigma, 1e-9 * _SIGMA_PER_FWHM, "fwhm_ns"),
    ),
    "Sensitivity" => Dict(
        "scale" => (:scale, 1.0, ""),
    ),
    "Constant" => Dict(
        "level" => (:level, 1.0, ""),
    ),
    "Polynomial" => Dict(
        "coefficients" => (:coefficients, 1.0, ""),
    ),
    "Rational" => Dict(
        "numerator" => (:numerator, 1.0, ""),
        "denominator" => (:denominator, 1.0, ""),
    ),
)

# GradientSmearing.width's conversion factor is the smeared leaf's own (_leaf_si_path(e.leaf)[2]),
# not a constant of the key, so it cannot live in the table above; this is its refusal's reason
# (D12), threaded through _fill_option_fields as refs_reason so the generic "takes a string, not a
# variable reference" message does not move for every other refused key (there are none left).
const _GRADIENT_SMEARING_WIDTH_REFS_REASON =
    "read_deck: key \"width\" in section \"instrument.gradient_smearing\" cannot be a variable " *
    "reference: its conversion factor is the smeared leaf's own and is not known to the binding " *
    "layer (conventions.md 7.6)."

# TabulatedKSmearing has no fittable parameter (Phase 4.G spec D7), so a string in either of its two
# lists is refused with this reason rather than the generic one, which would call the key a string.
const _TABULATED_K_SMEARING_REFS_REASON =
    "read_deck: section \"instrument.tabulated_k_smearing\" takes numbers only: offsets_deg and " *
    "weights cannot hold a variable reference, since a TabulatedKSmearing has no fittable parameter"

const _INSTRUMENT_REFS_REASON = Dict{String,String}(
    "GradientSmearing" => _GRADIENT_SMEARING_WIDTH_REFS_REASON,
    "TabulatedKSmearing" => _TABULATED_K_SMEARING_REFS_REASON)

const _BACKGROUND_TYPES = ("Constant", "Polynomial", "Rational", "Image")

function _validate_instrument_section(given::AbstractDict, known_names::AbstractSet{String})
    _check_known_keys(given, _INSTRUMENT_TOP_KEYS, "instrument")
    out = Dict{String,Any}()
    for (key, regname) in _INSTRUMENT_SINGLE_REGISTRY_NAME
        haskey(given, key) || continue
        entry = given[key]
        entry isa AbstractDict || throw(ArgumentError("read_deck: \"instrument.$key\" must be a table"))
        OptT = _INSTRUMENT_OPTIONS[regname]
        refs_ok = haskey(_INSTRUMENT_BINDING, regname)
        refs_reason = get(_INSTRUMENT_REFS_REASON, regname, "")
        out[key] = _fill_option_fields(OptT, entry, known_names, "instrument.$key", refs_ok, refs_reason)
    end
    if haskey(given, "background")
        bg = given["background"]
        bg isa AbstractVector || throw(ArgumentError("read_deck: \"instrument.background\" must be a list of tables"))
        out["background"] = Any[_validate_background_entry(b, known_names, i) for (i, b) in enumerate(bg)]
    end
    return out
end

function _validate_background_entry(given, known_names::AbstractSet{String}, i::Integer)
    given isa AbstractDict || throw(ArgumentError("read_deck: \"instrument.background[$i]\" must be a table"))
    haskey(given, "type") || throw(ArgumentError("read_deck: \"instrument.background[$i]\" needs a \"type\" key"))
    type = given["type"]
    type isa AbstractString || throw(ArgumentError("read_deck: \"instrument.background[$i].type\" must be a string"))
    type in _BACKGROUND_TYPES || throw(ArgumentError(
        "read_deck: unknown background type \"$type\"; valid names are $(sort(collect(_BACKGROUND_TYPES)))"))
    type == "Image" && throw(ArgumentError(
        "read_deck: an \"Image\" background needs image data, which is not a deck key; " *
        "build it in a script deck (conventions.md 7.4)"))
    OptT = _INSTRUMENT_OPTIONS[type]
    rest = Dict(k => v for (k, v) in given if k != "type")
    filled = _fill_option_fields(OptT, rest, known_names, "instrument.background[$i]", true)
    return Dict{String,Any}("type" => type, filled...)
end

# ------------------------------------------------------------------------------------------
# [regularizer]
# ------------------------------------------------------------------------------------------

const _REGULARIZER_TOP_KEYS = ("weight", "terms")
const _REGULARIZER_TERM_KEYS = ("select", "operator", "normalization", "scale", "weight",
    "interval", "reference")

function _validate_regularizer_section(given::AbstractDict)
    _check_known_keys(given, _REGULARIZER_TOP_KEYS, "regularizer")
    haskey(given, "terms") || throw(ArgumentError("read_deck: section \"regularizer\" needs \"terms\""))
    terms_raw = given["terms"]
    terms_raw isa AbstractVector || throw(ArgumentError("read_deck: \"regularizer.terms\" must be a list"))
    terms = Any[]
    for (i, t) in enumerate(terms_raw)
        t isa AbstractDict || throw(ArgumentError("read_deck: \"regularizer.terms[$i]\" must be a table"))
        _check_known_keys(t, _REGULARIZER_TERM_KEYS, "regularizer.terms[$i]")
        haskey(t, "select") || throw(ArgumentError("read_deck: \"regularizer.terms[$i]\" needs \"select\""))
        haskey(t, "operator") || throw(ArgumentError("read_deck: \"regularizer.terms[$i]\" needs \"operator\""))
        haskey(t, "normalization") || throw(ArgumentError(
            "read_deck: \"regularizer.terms[$i]\" needs \"normalization\""))
        _check_registry_name(OPERATOR_REGISTRY, t["operator"], "operator")
        _check_registry_name(NORMALIZATION_REGISTRY, t["normalization"], "normalization")
        push!(terms, Dict{String,Any}(t))
    end
    out = Dict{String,Any}("weight" => Float64(get(given, "weight", 1.0)), "terms" => terms)
    return out
end

# ------------------------------------------------------------------------------------------
# [optimizer]
# ------------------------------------------------------------------------------------------

const _OPTIMIZER_TOP_KEYS = ("procedure", "lbfgs", "levenberg_marquardt", "noisy_descent", "staged")
# _PROCEDURE_NAMES and _PROCEDURE_BASE_NAMES are optimizers.jl's own (Step 4.A.0, D22 item 1):
# this file used to redefine both, with a different order, and the redefinition silently won
# by include order, so every message below printed the wrong list.

"""
    STAGED_PROCEDURE

`"staged"`: the deck's one `procedure` value that is not an optimizer (Phase 4.F spec D3,
[OQ-54](../../dev/open_questions.md#oq-54)). `fit(deck)` fits each channel alone, then every
channel jointly from those fits, then `polish`es (`_staged_fit`). Not in `_PROCEDURE_NAMES`, which
names optimizers and is `optimizers.jl`'s own (OP20 pins it); this file alone accepts it.
"""
const STAGED_PROCEDURE = "staged"

function _check_procedure_name(nm; staged_ok::Bool=true)
    nm isa AbstractString && (nm in _PROCEDURE_NAMES || (staged_ok && nm == STAGED_PROCEDURE)) ||
        throw(ArgumentError(
        "read_deck: unknown procedure \"$nm\"; valid values are $_PROCEDURE_NAMES" *
        (staged_ok ? ", \"$STAGED_PROCEDURE\"," : ",") * " or a list of $_PROCEDURE_BASE_NAMES"))
    return String(nm)
end

"""
    _validate_procedure(raw, where; staged_ok) -> String or Vector{Any}

One `procedure` value (D30): a name of `_PROCEDURE_NAMES` (or `"staged"` when `staged_ok`), or a
list of `_PROCEDURE_BASE_NAMES`. `where` names the key in the message when it is not
`optimizer.procedure` (Phase 4.F spec D3: the staged table's two procedures, where `"staged"` is
refused by name).
"""
function _validate_procedure(raw, where::AbstractString; staged_ok::Bool)
    if raw isa AbstractVector
        for nm in raw
            nm isa AbstractString && nm in _PROCEDURE_BASE_NAMES || throw(ArgumentError(
                "read_deck: unknown procedure \"$nm\"; valid values are $_PROCEDURE_NAMES, or a " *
                "list of $_PROCEDURE_BASE_NAMES"))
        end
        return Any[String(nm) for nm in raw]
    end
    (!staged_ok && raw == STAGED_PROCEDURE) && throw(ArgumentError(
        "read_deck: \"$where\" cannot be \"$STAGED_PROCEDURE\": a stage is one fit, so it names an " *
        "optimizer, $_PROCEDURE_NAMES, or a list of $_PROCEDURE_BASE_NAMES"))
    return _check_procedure_name(raw; staged_ok=staged_ok)
end

function _validate_optimizer_section(given::AbstractDict)
    _check_known_keys(given, _OPTIMIZER_TOP_KEYS, "optimizer")
    procedure = _validate_procedure(get(given, "procedure", "noisy_descent_lbfgs"),
        "optimizer.procedure"; staged_ok=true)
    out = Dict{String,Any}("procedure" => procedure)
    empty_names = Set{String}()
    for (key, ctor) in OPTIMIZER_REGISTRY
        haskey(given, key) || continue
        entry = given[key]
        entry isa AbstractDict || throw(ArgumentError("read_deck: \"optimizer.$key\" must be a table"))
        out[key] = _fill_option_fields(ctor, entry, empty_names, "optimizer.$key", false)
    end
    if procedure == STAGED_PROCEDURE
        entry = get(given, "staged", Dict{String,Any}())
        entry isa AbstractDict || throw(ArgumentError("read_deck: \"optimizer.staged\" must be a table"))
        st = _fill_option_fields(StagedProcedureOptions, entry, empty_names, "optimizer.staged", false)
        for key in ("stage_procedure", "joint_procedure")
            st[key] = _validate_procedure(st[key], "optimizer.staged.$key"; staged_ok=false)
        end
        tol = st["polish_tol"]
        (isfinite(tol) && tol > 0) || throw(ArgumentError(
            "read_deck: \"optimizer.staged.polish_tol\" must be a finite number > 0; got $tol"))
        st["start_from"] isa AbstractDict || throw(ArgumentError(
            "read_deck: \"optimizer.staged.start_from\" must be a table of variable = \"channel\""))
        st["start_from"] = Dict{String,Any}(String(k) => v for (k, v) in st["start_from"])
        out["staged"] = st
    elseif haskey(given, "staged")
        throw(ArgumentError(
            "read_deck: [optimizer.staged] is read only with procedure = \"$STAGED_PROCEDURE\"; this " *
            "deck's procedure is $(repr(procedure)), so the table would be silently ignored"))
    end
    return out
end

"""
    _check_staged_deck(sections, channel_names) -> nothing

`procedure = "staged"`'s two cross-section checks (Phase 4.F spec D3), run by `read_deck` once
the channels and the variables are known: the deck is in the channel form with two or more
channels; and every `start_from` entry names a fitted (`vary`) variable and a channel of
`channels`.
"""
function _check_staged_deck(sections::AbstractDict, channel_names::Union{Nothing,Vector{String}})
    sections["optimizer"]["procedure"] == STAGED_PROCEDURE || return nothing
    channel_names === nothing && throw(ArgumentError(
        "read_deck: procedure \"$STAGED_PROCEDURE\" fits each channel alone and then every channel " *
        "jointly, so it needs a channel-form deck with two or more channels; this deck is in the " *
        "shorthand form (bare [data], [geometry], [instrument])"))
    length(channel_names) >= 2 || throw(ArgumentError(
        "read_deck: procedure \"$STAGED_PROCEDURE\" fits each channel alone and then every channel " *
        "jointly, so it needs a channel-form deck with two or more channels; this deck has one, " *
        "\"$(only(channel_names))\""))
    for (name, chan) in sections["optimizer"]["staged"]["start_from"]
        entry = get(sections["fit_variables"], name, nothing)
        (entry !== nothing && entry["vary"]) || throw(ArgumentError(
            "read_deck: \"optimizer.staged.start_from\" names \"$name\", which is not a fitted " *
            "variable of [fit.variables] (declared, with vary = true)"))
        (chan isa AbstractString && chan in channel_names) || throw(ArgumentError(
            "read_deck: \"optimizer.staged.start_from.$name\" must name a channel of \"channels\", " *
            "$channel_names; got $(repr(chan))"))
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# [output]
# ------------------------------------------------------------------------------------------

const _OUTPUT_KEYS = ("dir",)

function _validate_output_section(given::AbstractDict)
    _check_known_keys(given, _OUTPUT_KEYS, "output")
    haskey(given, "dir") || throw(ArgumentError("read_deck: section \"output\" needs \"dir\""))
    given["dir"] isa AbstractString || throw(ArgumentError("read_deck: \"output.dir\" must be a string"))
    return Dict{String,Any}("dir" => String(given["dir"]))
end

# ------------------------------------------------------------------------------------------
# read_deck
# ------------------------------------------------------------------------------------------

# The top-level known-key check D40 item 1 adds (Pitfall 9): a [channel.epw] table in a deck
# read_deck did not know about would otherwise be silently ignored and the fit would run
# single-channel. "channel" is the channel-form's table-of-tables; "channels" its order list.
const _TOP_LEVEL_KEYS = ("deck_version", "channels", "channel", "data", "geometry", "instrument",
    "fit", "plasma", "regularizer", "optimizer", "output")

const _CHANNEL_TABLE_KEYS = ("weight", "data", "geometry", "instrument")

const _CHANNEL_NAME_RE = r"^[a-z][a-z0-9_]*$"

function _check_top_level_keys(raw::AbstractDict)
    validset = Set(_TOP_LEVEL_KEYS)
    for k in keys(raw)
        k in validset || throw(ArgumentError(
            "read_deck: unknown top-level key \"$k\"; valid keys are $(sort(collect(validset)))"))
    end
    return nothing
end

"""
    _channel_list(sections::AbstractDict) -> Vector{Dict{String,Any}}

Both deck forms as one shape (D40 amendment 3): the channel form's own `sections["channels"]`
(one entry per `[channel.<name>]`, in `channels` order, each `{"name", "weight", "data",
"geometry", "instrument"}`) when present, or, for the shorthand, one entry named `"main"` with
weight `1.0` whose `"data"`/`"geometry"`/`"instrument"` are `sections["data"]`/`["geometry"]`/
`["instrument"]` **by reference** -- `build_problem` must never write into any of them.
"""
function _channel_list(sections::AbstractDict)
    haskey(sections, "channels") && return sections["channels"]
    return Dict{String,Any}[Dict{String,Any}("name" => "main", "weight" => 1.0,
        "data" => sections["data"], "geometry" => sections["geometry"],
        "instrument" => sections["instrument"])]
end

"""
    _in_channel(f, nm::AbstractString)

Runs `f()` and, on an `ArgumentError`, re-throws it prefixed `"read_deck: in [channel.<nm>]: "`
(D40 amendment 2, item 7), so a two-or-more-channel deck's error names which channel is at fault
while every existing substring the wrapped validator produces still matches (the wrapped message's
own leading `"read_deck: "` is not stripped -- D40 writes the prefix onto the message whole).
"""
function _in_channel(f, nm::AbstractString)
    try
        return f()
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("read_deck: in [channel.$nm]: " * e.msg))
    end
end

"""
    read_deck(path::AbstractString) -> Deck

Reads and validates the TOML deck at `path` (spec D33, D40): checks `deck_version` (a missing or
unknown value is a loud error naming the version found and the one supported, DK16); checks every
top-level key (D40 item 1, Pitfall 9); checks that the deck is in exactly one of the shorthand or
channel forms (D40 item 2); checks every key of every section against the component that owns it
(7.3, DK7); checks every numeric key's string value against the declared
`[fit.variables]`/`[fit.dependent]` names (7.6, DK4-DK6); fills every option-struct field with its
own default where the deck omits it (7.5, DK11); checks the four region cross-section rules and
fills `min_usable_fraction`'s default (D40 items 10-11). Returns a `Deck`, still inert -- build the
actual problem with `build_problem`.

**The shorthand** has a bare `[data]`, `[geometry]` and optional `[instrument]`, and no `channels`
or `channel`; its three sections are stored exactly as before chunk 7, `sections["data"]`,
`["geometry"]`, `["instrument"]`. **The channel form** has `channels` (the channel order, a
non-empty array of distinct names each matching `^[a-z][a-z0-9_]*\$`) and `channel` (a table of
`[channel.<name>]` tables, one per name in `channels`, each holding an optional `weight` and
required `data`/`geometry` and optional `instrument`); it is stored as `sections["channels"]`, a
`Vector{Dict{String,Any}}` in `channels` order (`_channel_list`). Mixing the two forms, or giving
one half of the channel-form pair without the other, is a loud error. The regions live in a shared
`sections["fit_regions"]` (`nothing`, or a `Vector` from `_validate_regions`) regardless of form.

Spec D33, D40.
"""
function read_deck(path::AbstractString)
    raw = TOML.parsefile(path)
    version = get(raw, "deck_version", nothing)
    version == DECK_VERSION || throw(ArgumentError(
        "read_deck: \"$path\" has deck_version " *
        (version === nothing ? "<missing>" : "\"$version\"") *
        "; this version of OTIS reads \"$DECK_VERSION\""))

    _check_top_level_keys(raw)

    has_channels_key = haskey(raw, "channels")
    has_channel_key = haskey(raw, "channel")
    bare_keys = sort([k for k in ("data", "geometry", "instrument") if haskey(raw, k)])

    if (has_channels_key || has_channel_key) && !isempty(bare_keys)
        throw(ArgumentError(
            "read_deck: this deck mixes the channel form (\"channels\" and [channel.<name>]) " *
            "with the bare section(s) $bare_keys; in the channel form, data, geometry and " *
            "instrument go inside each [channel.<name>]"))
    end

    is_channel_form = has_channels_key || has_channel_key
    if is_channel_form && !(has_channels_key && has_channel_key)
        which = has_channels_key ? "\"channels\"" : "[channel.<name>]"
        throw(ArgumentError(
            "read_deck: the channel form needs both a \"channels\" list and [channel.<name>] " *
            "tables; this deck has only $which"))
    end

    sections = Dict{String,Any}()
    channel_entries = Dict{String,Any}[]
    channel_table = Dict{String,Any}()

    if is_channel_form
        channels_raw = raw["channels"]
        (channels_raw isa AbstractVector && !isempty(channels_raw) &&
            all(x -> x isa AbstractString, channels_raw)) || throw(ArgumentError(
            "read_deck: \"channels\" must be a non-empty array of channel names"))
        names = String.(channels_raw)
        for nm in names
            occursin(_CHANNEL_NAME_RE, nm) || throw(ArgumentError(
                "read_deck: channel name \"$nm\" must match ^[a-z][a-z0-9_]*\$"))
        end
        seen = Set{String}()
        for nm in names
            nm in seen && throw(ArgumentError(
                "read_deck: channel name \"$nm\" appears more than once in \"channels\""))
            push!(seen, nm)
        end

        raw["channel"] isa AbstractDict || throw(ArgumentError("read_deck: \"channel\" must be a table"))
        channel_table = raw["channel"]
        table_names = Set(String.(keys(channel_table)))
        name_set = Set(names)
        missing_tables = sort(collect(setdiff(name_set, table_names)))
        extra_tables = sort(collect(setdiff(table_names, name_set)))
        (isempty(missing_tables) && isempty(extra_tables)) || throw(ArgumentError(
            "read_deck: \"channels\" and the [channel.<name>] tables disagree: no table for " *
            "$missing_tables; tables not listed in \"channels\": $extra_tables"))

        for nm in names
            chan_raw = channel_table[nm]
            chan_raw isa AbstractDict || throw(ArgumentError("read_deck: \"channel.$nm\" must be a table"))
            _check_known_keys(chan_raw, _CHANNEL_TABLE_KEYS, "channel.$nm")
            haskey(chan_raw, "data") || throw(ArgumentError(
                "read_deck: \"channel.$nm\" is missing section \"data\""))
            haskey(chan_raw, "geometry") || throw(ArgumentError(
                "read_deck: \"channel.$nm\" is missing section \"geometry\""))
            weight = if haskey(chan_raw, "weight")
                w = chan_raw["weight"]
                w isa AbstractString && throw(ArgumentError(
                    "read_deck: \"channel.$nm.weight\" must be a number, not a variable " *
                    "reference; a channel's weight is fixed (OQ-35)"))
                (w isa Real && isfinite(w) && w > 0) || throw(ArgumentError(
                    "read_deck: \"channel.$nm.weight\" must be a finite number > 0; got $(repr(w))"))
                Float64(w)
            else
                1.0
            end
            data_section = _in_channel(nm) do
                _validate_data_section(chan_raw["data"])
            end
            geometry_section = _in_channel(nm) do
                _validate_geometry_section(chan_raw["geometry"])
            end
            push!(channel_entries, Dict{String,Any}("name" => nm, "weight" => weight,
                "data" => data_section, "geometry" => geometry_section))
        end

        if length(names) == 1 && channel_entries[1]["weight"] != 1.0
            throw(ArgumentError(
                "read_deck: a deck with one channel cannot set a weight " *
                "($(channel_entries[1]["weight"])): a weight only scales a channel against the " *
                "others, and alone it would silently rescale the regularizer"))
        end
    else
        haskey(raw, "data") || throw(ArgumentError("read_deck: \"$path\" is missing section \"data\""))
        sections["data"] = _validate_data_section(raw["data"])
        haskey(raw, "geometry") || throw(ArgumentError("read_deck: \"$path\" is missing section \"geometry\""))
        sections["geometry"] = _validate_geometry_section(raw["geometry"])
    end

    fit_raw = get(raw, "fit", Dict{String,Any}())
    interval_names = Set{String}(keys(get(fit_raw, "intervals", Dict{String,Any}())))
    variables_raw = get(fit_raw, "variables", Dict{String,Any}())
    dependent_raw = get(fit_raw, "dependent", Dict{String,Any}())
    known_names = Set{String}(vcat(collect(keys(variables_raw)), collect(keys(dependent_raw))))

    sections["fit_intervals"] = _validate_intervals_section(get(fit_raw, "intervals", Dict{String,Any}()))
    sections["fit_variables"] = Dict{String,Any}(
        nm => _validate_variable_entry(nm, entry, interval_names) for (nm, entry) in variables_raw)
    sections["fit_dependent"] = _validate_dependent_section(dependent_raw)
    sections["fit"] = _validate_fit_scalars(fit_raw)
    sections["fit_regions"] = _validate_regions(get(fit_raw, "regions", nothing))

    haskey(raw, "plasma") || throw(ArgumentError("read_deck: \"$path\" is missing section \"plasma\""))
    sections["plasma"] = _validate_plasma_section(raw["plasma"], known_names)

    if is_channel_form
        for entry in channel_entries
            nm = entry["name"]
            instrument_raw = get(channel_table[nm], "instrument", Dict{String,Any}())
            entry["instrument"] = _in_channel(nm) do
                _validate_instrument_section(instrument_raw, known_names)
            end
        end
        sections["channels"] = channel_entries
    else
        sections["instrument"] = _validate_instrument_section(get(raw, "instrument", Dict{String,Any}()), known_names)
    end

    sections["regularizer"] = haskey(raw, "regularizer") ?
        _validate_regularizer_section(raw["regularizer"]) : nothing

    sections["optimizer"] = _validate_optimizer_section(get(raw, "optimizer", Dict{String,Any}()))
    _check_staged_deck(sections, is_channel_form ? String[e["name"] for e in channel_entries] : nothing)

    sections["output"] = haskey(raw, "output") ? _validate_output_section(raw["output"]) : nothing

    # D40 items 10-11: the four cross-section loud errors, then min_usable_fraction's default,
    # over every channel's own data table (the shorthand counts as one channel, "data").
    regions_given = sections["fit_regions"] !== nothing
    data_by_where = is_channel_form ?
        [("channel.$(e["name"]).data", e["data"]) for e in channel_entries] :
        [("data", sections["data"])]

    for (where, data) in data_by_where
        (haskey(data, "t_range_ns") && regions_given) && throw(ArgumentError(
            "read_deck: section \"$where\" gives \"t_range_ns\", and the deck gives " *
            "[[fit.regions]]; regions supersede t_range_ns, and a deck whose stated time window " *
            "would be silently ignored is refused instead"))
        for k in ("t_shift_ns", "t_shift_s", "min_usable_fraction")
            (haskey(data, k) && !regions_given) && throw(ArgumentError(
                "read_deck: section \"$where\" gives \"$k\" without [[fit.regions]]; without " *
                "regions nothing is averaged and the value would be silently ignored"))
        end
    end

    if regions_given
        for region in sections["fit_regions"]
            interval = region["interval"]
            isempty(interval) && continue
            haskey(sections["fit_intervals"], interval) && throw(ArgumentError(
                "read_deck: [[fit.regions]] names interval \"$interval\", which [fit.intervals] " *
                "also declares"))
        end
    end

    nchannels_declared = is_channel_form ? length(channel_entries) : 1
    (nchannels_declared >= 2 && !regions_given) && throw(ArgumentError(
        "read_deck: a deck with $nchannels_declared channels needs [[fit.regions]]: two cameras " *
        "never share a time axis, so both are averaged onto one (OQ-34)"))

    if regions_given
        for (where, data) in data_by_where
            if haskey(data, "min_usable_fraction")
                v = data["min_usable_fraction"]
                v isa Real || throw(ArgumentError(
                    "read_deck: \"$where.min_usable_fraction\" must be a number"))
                data["min_usable_fraction"] = Float64(v)
            else
                data["min_usable_fraction"] = 0.5
            end
        end
    end

    return Deck(DECK_VERSION, sections)
end

# ------------------------------------------------------------------------------------------
# build_problem: resolving a _VarRef into a placeholder value and a binding
# ------------------------------------------------------------------------------------------

"""
    _resolve_and_collect!(refs, v, vals, names, fieldsym, element = 0)

Walk one already-classified deck value (a literal, a `_VarRef`, or a vector of either):
substitutes every `_VarRef` by the variable's or dependent's own placeholder value from `vals`
(D9: the raw value in the unit the deck declared it in, exactly what `electrons`/`ion`/the effect
constructors already expect from that same key), and pushes `(name, field, element)` onto `refs`
for every reference found (`element = 0` for a scalar field, the 1-based position for a vector
field). Used identically for a plasma leaf and an instrument effect parameter; the two differ only
in how the caller turns `(field, element)` into a `Binding` target.
"""
function _resolve_and_collect!(refs::Vector{Tuple{String,Symbol,Int}}, v, vals::Dict{String,Float64},
                                fieldsym::Symbol, element::Int=0)
    if v isa _VarRef
        push!(refs, (v.name, fieldsym, element))
        return vals[v.name]
    elseif v isa AbstractVector
        return Any[_resolve_and_collect!(refs, x, vals, fieldsym, i) for (i, x) in enumerate(v)]
    else
        return v
    end
end

"""
    _placeholder_values(vars::Vector{FittedVariable}, dep_pairs, t0::Float64) -> Dict{String,Float64}

Every declared variable's own `init`, and every declared dependent's value evaluated once, at
`t0`, from those same inits (D8's mechanism, reused directly rather than reimplemented): the
number the deck layer substitutes for a `_VarRef` while building the template `Plasma` and the
instrument chain. Never stored, never read again after `build_problem` returns -- the trace this
produces is only a placeholder that `ParameterSpec`'s own construction overwrites at the first
`unflatten` (D9's own reasoning for why a newly-added row's filler value does not matter).
"""
function _placeholder_values(vars::Vector{FittedVariable}, dep_pairs, t0::Float64)
    nvar = length(vars)
    dep_tuple, names = _topological_dependents(vars, dep_pairs, nvar)
    vals = Vector{Float64}(undef, nvar + length(dep_tuple))
    for (i, v) in enumerate(vars)
        vals[i] = v.init
    end
    for (k, dep) in enumerate(dep_tuple)
        vals[nvar+k] = _eval(dep.node, vals, t0)
    end
    return Dict{String,Float64}(nm => vals[slot] for (nm, slot) in names)
end

# ------------------------------------------------------------------------------------------
# build_problem: [plasma]
# ------------------------------------------------------------------------------------------

function _species_kwargs_and_bindings!(bindings::Vector{Pair{String,Any}}, sp::AbstractDict,
                                        vals::Dict{String,Float64}, prefix::String)
    kwargs = Dict{Symbol,Any}()
    for k in ("T_eV", "T_keV", "u_kms", "u_ms", "fraction", "mass_amu", "mass_kg", "charge")
        haskey(sp, k) || continue
        refs = Tuple{String,Symbol,Int}[]
        val = _resolve_and_collect!(refs, sp[k], vals, Symbol(k))
        kwargs[Symbol(k)] = val
        for (nm, _fieldsym, _elem) in refs
            push!(bindings, nm => "$prefix.$k")
        end
    end
    if haskey(sp, "shape")
        kwargs[:f] = _build_shape_from_deck(sp["shape"], bindings, vals, "$prefix.f")
    end
    return kwargs
end

function _build_shape_from_deck(shape::AbstractDict, bindings::Vector{Pair{String,Any}},
                                 vals::Dict{String,Float64}, prefix::String)
    type = shape["type"]
    OptT = _SHAPE_OPTIONS[type]
    resolved = Dict{String,Any}()
    for (k, v) in shape
        k == "type" && continue
        refs = Tuple{String,Symbol,Int}[]
        resolved[k] = _resolve_and_collect!(refs, v, vals, Symbol(k))
        for (nm, _fieldsym, _elem) in refs
            push!(bindings, nm => "$prefix.$k")
        end
    end
    return _construct_shape(OptT, resolved)
end

_construct_shape(::Type{MaxwellianOptions}, ::AbstractDict) = Maxwellian()
_construct_shape(::Type{SuperGaussianOptions}, resolved::AbstractDict) = SuperGaussian(resolved["m"])
_construct_shape(::Type{KappaOptions}, resolved::AbstractDict) = Kappa(resolved["kappa"])
_construct_shape(::Type{GeneralizedNormalOptions}, resolved::AbstractDict) = GeneralizedNormal(resolved["m"])
_construct_shape(::Type{SkewNormalOptions}, resolved::AbstractDict) = SkewNormal(resolved["alpha"])

"""
    _build_template(plasma_section, vals) -> (Plasma, Vector{Pair{String,Any}})

The template `Plasma` (spec D33) `[plasma]` and its species tables describe, with every `_VarRef`
substituted by its placeholder value (`vals`), plus every plasma-leaf binding this discovered
along the way, as `name => boundary_leaf_key` pairs ready for `ParameterSpec`'s own `bindings`
keyword (D9).
"""
function _build_template(plasma_section::AbstractDict, vals::Dict{String,Float64})
    bindings = Pair{String,Any}[]
    ne_key = only(k for k in ("n_e_cm3", "n_e_m3") if haskey(plasma_section, k))
    refs = Tuple{String,Symbol,Int}[]
    n_e_val = _resolve_and_collect!(refs, plasma_section[ne_key], vals, Symbol(ne_key))
    for (nm, _fieldsym, _elem) in refs
        push!(bindings, nm => ne_key)
    end

    electron_specs = SpeciesSpec[]
    for (nm, sp) in plasma_section["electrons"]
        kwargs = _species_kwargs_and_bindings!(bindings, sp, vals, "electrons.$nm")
        push!(electron_specs, electrons(; name=nm, kwargs...))
    end

    ion_specs = SpeciesSpec[]
    for (nm, sp) in plasma_section["ions"]
        kwargs = _species_kwargs_and_bindings!(bindings, sp, vals, "ions.$nm")
        push!(ion_specs, ion(; name=nm, kwargs...))
    end

    template = Plasma(; Dict(Symbol(ne_key) => n_e_val)..., electrons=electron_specs, ions=ion_specs)
    return template, bindings
end

# ------------------------------------------------------------------------------------------
# build_problem: [instrument]
# ------------------------------------------------------------------------------------------

_construct_effect(ctor, kwargs::AbstractDict, ::Plasma) = ctor(; kwargs...)
_construct_effect(::Type{GradientSmearing}, kwargs::AbstractDict, template::Plasma) =
    GradientSmearing(template; kwargs...)

const _EFFECT_CTOR = Dict{String,Any}(
    "SpectralIRF" => SpectralIRF, "TemporalIRF" => TemporalIRF, "KSmearing" => KSmearing,
    "GradientSmearing" => GradientSmearing, "WavelengthCalibration" => WavelengthCalibration,
    "Sensitivity" => Sensitivity, "Constant" => ConstantBackground,
    "Polynomial" => PolynomialBackground, "Rational" => RationalBackground,
    "PixelIntegration" => PixelIntegration, "TabulatedKSmearing" => TabulatedKSmearing)

"""
    _build_effect_and_pending!(pending, regname, filled, vals, template) -> AbstractInstrumentEffect

One instrument effect (spec D33): `filled`'s own `_VarRef`s substituted by their placeholder
values and pushed onto `pending` as `(name, effect_object, field, element, factor, unit_key)`
six-tuples (D9, D10, chunk 4) -- `field`, `factor` and `unit_key` come from
`_INSTRUMENT_BINDING[regname][deck_key]`, not from the deck key itself (that table is what closes
Phase 3's B5; see the box above `_fill_option_fields`), `effect_object` this very effect, resolved
to a binding target only once `build_problem` knows the effect's final index in `expt.effects`
(the `findfirst(e -> e === effect, expt0.effects)` loop there). `GradientSmearing`'s `leaf` carries
a `"plasma."` prefix in the deck (`conventions.md` 7.6); it is stripped here before
`GradientSmearing`'s own constructor parses it, which is the one place a raw SI/boundary leaf key
is legal without that prefix. `GradientSmearing` itself never reaches the table lookup below: its
one numeric key, `width`, is refused a variable reference before `read_deck` gets here (D12).
"""
_deck_field_value!(::Any, key::String, v, refs::Vector{Tuple{String,Symbol,Int}},
                    vals::Dict{String,Float64}, fieldsym::Symbol) =
    _resolve_and_collect!(refs, v, vals, fieldsym)

"""
    _deck_field_value!(::Type{GradientSmearing}, key, v, refs, vals, fieldsym)

`GradientSmearing`'s one exception (`conventions.md` 7.6): its `leaf` key carries a `"plasma."`
prefix in the deck, stripped here before `GradientSmearing`'s own constructor parses it (the one
place a raw SI/boundary leaf key is legal without that prefix). Dispatched on the effect's own
constructor type, exactly as `_construct_effect`'s `template` special case is -- never a
hand-written `if` on the component's name (`conventions.md` 7.2, DK10).
"""
function _deck_field_value!(::Type{GradientSmearing}, key::String, v, refs::Vector{Tuple{String,Symbol,Int}},
                             vals::Dict{String,Float64}, fieldsym::Symbol)
    key == "leaf" && v isa AbstractString && return startswith(v, "plasma.") ? v[8:end] : v
    return _resolve_and_collect!(refs, v, vals, fieldsym)
end

function _build_effect_and_pending!(pending::Vector{Tuple{String,Any,Symbol,Int,Float64,String}},
                                     regname::String, filled::AbstractDict, vals::Dict{String,Float64},
                                     template::Plasma)
    ctor = _EFFECT_CTOR[regname]
    resolved = Dict{Symbol,Any}()
    local_refs = Tuple{String,String,Int}[]   # (variable name, deck key, element)
    for (k, v) in filled
        fieldsym = Symbol(k)
        refs = Tuple{String,Symbol,Int}[]
        resolved[fieldsym] = _deck_field_value!(ctor, k, v, refs, vals, fieldsym)
        for (nm, fs, elem) in refs
            push!(local_refs, (nm, String(fs), elem))
        end
    end
    effect = _construct_effect(ctor, resolved, template)
    ks = keys(parameters(effect))
    binding_table = get(_INSTRUMENT_BINDING, regname, nothing)
    for (nm, deck_key, elem) in local_refs
        field, factor, unit_key = binding_table[deck_key]
        field in ks || throw(ArgumentError(
            "\"$nm\" is bound to \"$regname.$field\", which has no matching reconstruct-facing " *
            "parameter; valid parameters are $(collect(ks))"))
        push!(pending, (nm, effect, field, elem, factor, unit_key))
    end
    return effect
end

# ------------------------------------------------------------------------------------------
# build_problem: [fit.intervals], [fit.variables]
# ------------------------------------------------------------------------------------------

function _build_intervals(intervals_section::AbstractDict, axis::AbstractVector)
    names = sort(collect(keys(intervals_section)))
    ivals = FitInterval[]
    for nm in names
        entry = intervals_section[nm]
        key = only(k for k in _FIT_INTERVAL_KEYS if haskey(entry, k))
        push!(ivals, FitInterval(nm, axis; NamedTuple{(Symbol(key),)}((entry[key],))...))
    end
    return ivals, Dict{String,Int}(nm => i for (i, nm) in enumerate(names))
end

function _build_variables(variables_section::AbstractDict, interval_index::Dict{String,Int})
    names = sort(collect(keys(variables_section)))
    vars = FittedVariable[]
    for nm in names
        entry = variables_section[nm]
        interval = entry["interval"]
        idx = interval isa AbstractString ? interval_index[interval] : Int(interval)
        push!(vars, FittedVariable(nm; bounds=Tuple(entry["bounds"]), init=entry["init"],
            time_varying=entry["time_varying"], vary=entry["vary"], interval=idx))
    end
    return vars
end

# ------------------------------------------------------------------------------------------
# build_problem: [regularizer], [optimizer]
# ------------------------------------------------------------------------------------------

function _build_regularizer(section, pspec::ParameterSpec)
    section === nothing && return nothing
    terms = RegularizerTerm[]
    for t in section["terms"]
        kwargs = Dict{Symbol,Any}(:operator => OPERATOR_REGISTRY[t["operator"]],
            :normalization => NORMALIZATION_REGISTRY[t["normalization"]])
        haskey(t, "scale") && (kwargs[:scale] = Float64(t["scale"]))
        haskey(t, "weight") && (kwargs[:weight] = Float64(t["weight"]))
        haskey(t, "interval") && (kwargs[:interval] = t["interval"])
        haskey(t, "reference") && (kwargs[:reference] = t["reference"])
        push!(terms, RegularizerTerm(pspec; select=t["select"], kwargs...))
    end
    return Regularizer(terms; weight=section["weight"])
end

function _build_one_optimizer(key::String, section::AbstractDict)
    ctor = OPTIMIZER_REGISTRY[key]
    filled = get(section, key, nothing)
    filled === nothing && return ctor()
    kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in filled)
    return ctor(; kwargs...)
end

function _build_optimizer(section::AbstractDict)
    procedure = section["procedure"]
    # Phase 4.F spec D3: a staged deck's optimizer is its joint stage's, built from the same
    # option tables; the stages themselves are _staged_fit's.
    procedure == STAGED_PROCEDURE && return _build_optimizer(
        merge(section, Dict{String,Any}("procedure" => section["staged"]["joint_procedure"])))
    keys_ = procedure == "noisy_descent_lbfgs" ? ["noisy_descent", "lbfgs"] :
            procedure isa AbstractVector ? procedure : [procedure]
    members = [_build_one_optimizer(k, section) for k in keys_]
    return length(members) == 1 ? members[1] : OptimizerSequence(members...)
end

# ------------------------------------------------------------------------------------------
# build_problem
# ------------------------------------------------------------------------------------------

"""
    _read_channels(s::AbstractDict, chan_list::AbstractVector, regions_given::Bool)
        -> (raw::Vector{StreakData}, probes::Vector{Any}, collections::Vector{Any})

`build_problem`'s first stage (D50, 2026-09-25), split out **unchanged** so the scatter (D51) can
call it directly on the deck's own channels without re-reading through `average_regions`: for each
channel of `chan_list`, in order, `read_streak`s its data file and crops it
(`_crop_streak`, the lambda crop always, the time crop from `[data].t_range_ns`/
`[channel.<n>.data].t_range_ns` **only when `!regions_given`** -- with regions the crop would
silently discard what `average_regions` should see), and builds its probe and collection
(`_build_geometry`). Not averaged yet -- that is `_average_channels`.

Spec D33, D40, D50.
"""
function _read_channels(s::AbstractDict, chan_list::AbstractVector, regions_given::Bool)
    nchannels = length(chan_list)
    raw = Vector{StreakData}(undef, nchannels)
    chan_probe = Vector{Any}(undef, nchannels)
    chan_collection = Vector{Any}(undef, nchannels)
    for c in 1:nchannels
        chan = chan_list[c]
        data0 = read_streak(chan["data"]["file"])
        data1 = _crop_streak(data0, get(chan["data"], "lambda_range_nm", nothing),
            regions_given ? nothing : get(chan["data"], "t_range_ns", nothing))
        raw[c] = data1
        chan_probe[c], chan_collection[c] = _build_geometry(chan["geometry"])
    end
    return raw, chan_probe, chan_collection
end

"""
    _average_channels(raw::AbstractVector{StreakData}, chan_list::AbstractVector,
                      regions::AbstractVector{AveragingRegion}) -> Vector{StreakData}

`build_problem`'s second stage (D50, 2026-09-25), split out **unchanged**: `raw[c]` averaged onto
`regions` with `_averaging_kwargs(chan_list[c]["data"])` when `regions` is non-empty, else `raw[c]`
itself (Route B, OQ-34; D13-D16, D39). Then D39's axis -- one channel's own `t̄` (Pitfall 27,
bitwise as before chunk 7), or, with two or more, the pooled mean of every channel's window counts,
a plain loop in channel order so the summation order is fixed and reproducible -- and every
channel's `StreakData` rebuilt on it.

Spec D33, D40, D50, D39.
"""
function _average_channels(raw::AbstractVector{StreakData}, chan_list::AbstractVector,
    regions::AbstractVector{AveragingRegion})
    nchannels = length(chan_list)
    regions_given = !isempty(regions)

    chan_data = Vector{StreakData}(undef, nchannels)
    for c in 1:nchannels
        chan_data[c] = regions_given ?
            average_regions(raw[c], regions; _averaging_kwargs(chan_list[c]["data"])...) : raw[c]
    end

    axis = if nchannels == 1
        chan_data[1].t
    else
        n_out = length(chan_data[1].t)
        a = Vector{Float64}(undef, n_out)
        for k in 1:n_out
            num = 0.0
            den = 0
            for c in 1:nchannels
                wc = chan_data[c].meta["regions_window_counts"]
                num += wc[k] * chan_data[c].t[k]
                den += wc[k]
            end
            a[k] = num / den
        end
        a
    end

    for c in 1:nchannels
        d = chan_data[c]
        chan_data[c] = StreakData(; lambda_m=d.lambda, t_s=axis, image=d.image, sigma=d.sigma,
            mask=d.mask, meta=d.meta)
    end
    return chan_data
end

"""
    _assemble_problem(s::AbstractDict, chan_list::AbstractVector, chan_data::AbstractVector{StreakData},
                      chan_probe::AbstractVector, chan_collection::AbstractVector,
                      regions::AbstractVector{AveragingRegion})
        -> (AbstractFitProblem, AbstractOptimizer, Vector{Float64})

`build_problem`'s third stage (D50, 2026-09-25), split out **unchanged**: from the "no usable
pixel" check to the `return`, moved verbatim. `regions` feeds only the interval synthesis and the
region-interval bins check (D15 amendment 1 and 4) -- `_read_channels`/`_average_channels` have
already done everything the averaging itself needs.

Spec D33, D40, D50, D15.
"""
function _assemble_problem(s::AbstractDict, chan_list::AbstractVector,
    chan_data::AbstractVector{StreakData}, chan_probe::AbstractVector, chan_collection::AbstractVector,
    regions::AbstractVector{AveragingRegion})
    is_channel_form = haskey(s, "channels")
    nchannels = length(chan_list)

    for c in 1:nchannels
        d = chan_data[c]
        (d.mask !== nothing && !any(d.mask)) && throw(ArgumentError(
            "build_problem: channel \"$(chan_list[c]["name"])\" has no usable pixel (its mask " *
            "excludes every pixel)"))
    end

    axis = chan_data[1].t

    axis_unit = Symbol(s["fit"]["axis_unit"])
    factor = axis_unit === :ns ? 1e-9 : 1.0

    # D15 amendment 1, D40 (correction): the interval-conflict check now runs in read_deck, over
    # the shared [[fit.regions]]; build_problem keeps only the local synthesis, into a LOCAL copy
    # of fit_intervals, never into deck.sections (effective_deck re-serializes deck.sections as
    # they stand).
    fit_intervals = s["fit_intervals"]
    if !isempty(regions)
        synthesized = Dict{String,Any}()
        for region in regions
            isempty(region.interval) && continue
            synthesized[region.interval] = Dict{String,Any}("t_range_s" => Any[region.lo, region.hi])
        end
        fit_intervals = merge(fit_intervals, synthesized)
    end
    ivals, interval_index = _build_intervals(fit_intervals, axis)

    # D15 amendment 4: a region-declared interval's bins must equal the region's own output-bin
    # range (contiguous, in region order on the averaged axis), or the deck is refused.
    if !isempty(regions)
        offset = 0
        for region in regions
            region_bins = (offset+1):(offset+region.nbins)
            offset += region.nbins
            isempty(region.interval) && continue
            iv = ivals[interval_index[region.interval]]
            iv.bins == region_bins || throw(ArgumentError(
                "read_deck: region interval \"$(region.interval)\" resolved to bins " *
                "$(iv.bins), but the region's own output bins are $region_bins"))
        end
    end

    vars = _build_variables(s["fit_variables"], interval_index)
    dep_pairs = collect(Pair{String,String}, s["fit_dependent"])

    vals = _placeholder_values(vars, dep_pairs, axis[1] / factor)

    template, bindings_plasma = _build_template(s["plasma"], vals)

    pending_by_channel = Vector{Vector{Tuple{String,Any,Symbol,Int,Float64,String}}}(undef, nchannels)
    effects_by_channel = Vector{Tuple}(undef, nchannels)
    for c in 1:nchannels
        chan = chan_list[c]
        pending = Tuple{String,Any,Symbol,Int,Float64,String}[]
        effects = AbstractInstrumentEffect[]
        for key in ("k_smearing", "tabulated_k_smearing", "gradient_smearing", "wavelength_calibration",
                    "pixel_integration", "spectral_irf", "temporal_irf", "sensitivity")
            haskey(chan["instrument"], key) || continue
            regname = _INSTRUMENT_SINGLE_REGISTRY_NAME[key]
            push!(effects, _build_effect_and_pending!(pending, regname, chan["instrument"][key], vals, template))
        end
        for bg in get(chan["instrument"], "background", Any[])
            bg_fields = Dict(k => v for (k, v) in bg if k != "type")
            push!(effects, _build_effect_and_pending!(pending, bg["type"], bg_fields, vals, template))
        end
        pending_by_channel[c] = pending
        effects_by_channel[c] = Tuple(effects)
    end

    trace0 = PlasmaTrace(template; axis_name=:t, t_s=axis, leaves=())
    expt0s = [Experiment(; trace=trace0, probe=chan_probe[c], collection=chan_collection[c],
        lambda_m=chan_data[c].lambda, effects=effects_by_channel[c]) for c in 1:nchannels]

    bindings_instrument = Pair{String,Any}[]
    for c in 1:nchannels
        for (nm, effect, field, elem, fac, unit_key) in pending_by_channel[c]
            idx = findfirst(e -> e === effect, expt0s[c].effects)
            target = is_channel_form ?
                (channel=c, effect=idx, field=field, element=elem, factor=fac, unit_key=unit_key) :
                (idx, field, elem, fac, unit_key)
            push!(bindings_instrument, nm => target)
        end
    end

    pspec = ParameterSpec(Tuple(expt0s); variables=vars, dependents=dep_pairs, intervals=ivals,
        bindings=vcat(bindings_plasma, bindings_instrument), axis_unit=axis_unit)

    working = [Experiment(; trace=pspec.trace, probe=expt0s[c].probe, collection=expt0s[c].collection,
        lambda_m=expt0s[c].lambda, effects=expt0s[c].effects, prefactor=expt0s[c].prefactor)
        for c in 1:nchannels]

    regularizer = _build_regularizer(s["regularizer"], pspec)

    problem = if nchannels == 1
        FitProblem(working[1], pspec, chan_data[1]; regularizer=regularizer, scheduler=:auto,
            channel=1, name=chan_list[1]["name"])
    else
        weights = Float64[c["weight"] for c in chan_list]
        fitprobs = Tuple(FitProblem(working[c], pspec, chan_data[c]; scheduler=:auto, channel=c,
            name=chan_list[c]["name"]) for c in 1:nchannels)
        JointFitProblem(fitprobs; weights=weights, regularizer=regularizer)
    end

    optimizer = _build_optimizer(s["optimizer"])
    start = flatten(pspec)
    return problem, optimizer, start
end

"""
    build_problem(deck::Deck) -> (AbstractFitProblem, AbstractOptimizer, Vector{Float64})

Builds the actual Julia objects a validated `Deck` describes (spec D33, D40): for each channel of
`_channel_list(deck.sections)`, in order, reads and crops its data (`[data]`/`[channel.<n>.data]`),
averages it onto `[[fit.regions]]` when given, and builds its probe and collection
(`[geometry]`/`[channel.<n>.geometry]`, fixed). With one channel the trace axis is that channel's
own averaged `t̄` (D14), bitwise unchanged from before chunk 7; with two or more it is the pooled
mean of every channel's window counts (D39), and every channel's data is rebuilt on it. Builds the
named intervals and fitted variables (`[fit.intervals]`, `[fit.variables]`, shared), the template
plasma (`[plasma]`, shared) and each channel's own instrument chain
(`[instrument]`/`[channel.<n>.instrument]`, every `_VarRef` substituted by its placeholder value
and recorded as a binding -- the shorthand's five-element target, the channel form's named-tuple
target carrying `channel`), one shared `ParameterSpec` over every channel's `Experiment` (which
resolves every binding for real, D9, and rebuilds the trace with the union of leaves), the shared
regularizer (`[regularizer]`) and the optimizer (`[optimizer]`). Returns a plain `FitProblem` for
one channel or a `JointFitProblem` for two or more, the optimizer, and the start vector
`flatten(pspec)`.

**Split into three stages at chunk 10 (D50, 2026-09-25)**: `_read_channels`, `_average_channels`
and `_assemble_problem`, moved verbatim -- so that `slice_scatter` (D51) can call the same three
stages on data it has averaged itself, one fine window at a time, without re-reading the streak
files. Not one expression changed; the split is bitwise, checked in scratch against `4311330`.

Spec D33, D40, D50.
"""
function build_problem(deck::Deck)
    s = deck.sections
    chan_list = _channel_list(s)
    regions = s["fit_regions"] === nothing ? AveragingRegion[] : _build_averaging_regions(s["fit_regions"])
    raw, probes, collections = _read_channels(s, chan_list, !isempty(regions))
    chan_data = _average_channels(raw, chan_list, regions)
    return _assemble_problem(s, chan_list, chan_data, probes, collections, regions)
end

# ------------------------------------------------------------------------------------------
# effective_deck
# ------------------------------------------------------------------------------------------

"""
    _binding_description(b::Binding, channel_names::Union{Nothing,Vector{String}} = nothing)

One binding's text under `[fit.variables.<name>].bindings` (D9, D10, D40 amendment 5).
`channel_names === nothing` in the shorthand: the text is unchanged from before chunk 7. In the
channel form, an `:effect` binding's text gains the prefix `channel.<channel_names[b.channel]>.`
(`channel.iaw.instrument.effect[3].offset (unit_key=lambda_offset_nm)`); a `:leaf` binding's text
never changes, since every channel shares one plasma.
"""
function _binding_description(b::Binding, channel_names::Union{Nothing,Vector{String}}=nothing)
    b.kind === :leaf && return string(b.path) * (isempty(b.unit_key) ? "" : " (unit_key=$(b.unit_key))")
    suffix = b.element == 0 ? "" : "[$(b.element)]"
    prefix = channel_names === nothing ? "" : "channel.$(channel_names[b.channel])."
    return "$(prefix)instrument.effect[$(b.effect)].$(b.field)$suffix" *
           (isempty(b.unit_key) ? "" : " (unit_key=$(b.unit_key))")
end

"""
    effective_deck(deck::Deck, problem::AbstractFitProblem, optimizer::AbstractOptimizer) -> String

The effective configuration (`conventions.md` 7.5, spec D33, D40 amendment 5): `deck`'s own
validated sections (every default already filled by `read_deck`), re-serialized as TOML, with
`deck_version` written first and three things the input deck does not say added: every variable's
resolved binding(s) (under its own `[fit.variables.<name>]` table, `bindings`, one string per
binding, D9's leaf-path-and-unit-keyword or effect-parameter form, channel-prefixed in the channel
form), every dependent's expression text (already present, verbatim, under `[fit.dependent]`), and
the axis unit (`[fit].axis_unit`, already filled in by `read_deck`'s own default). The shorthand
writes exactly what it always has, plus `[[fit.regions]]` when given; the channel form writes
`channels`, `channel` (one table per name, weight always explicit) and no bare `data`, `geometry`
or `instrument`. A fit is reproducible from this string alone (7.5): re-reading it with `read_deck`
succeeds and yields the same `Deck` (DK1, CD6).

Spec D33, D40.
"""
function effective_deck(deck::Deck, problem::AbstractFitProblem, ::AbstractOptimizer)
    s = deck.sections
    out = Dict{String,Any}()
    out["deck_version"] = deck.deck_version

    is_channel_form = haskey(s, "channels")
    channel_names = nothing
    if is_channel_form
        names = String[c["name"] for c in s["channels"]]
        channel_names = names
        out["channels"] = names
        out["channel"] = Dict{String,Any}(
            c["name"] => Dict{String,Any}("weight" => c["weight"], "data" => c["data"],
                "geometry" => c["geometry"], "instrument" => c["instrument"])
            for c in s["channels"])
    else
        out["data"] = s["data"]
        out["geometry"] = s["geometry"]
        out["instrument"] = s["instrument"]
    end
    out["plasma"] = s["plasma"]
    s["regularizer"] === nothing || (out["regularizer"] = s["regularizer"])
    out["optimizer"] = s["optimizer"]
    s["output"] === nothing || (out["output"] = s["output"])

    bindings_by_name = Dict{String,Vector{String}}()
    for b in problem.pspec.bindings
        push!(get!(bindings_by_name, b.name, String[]), _binding_description(b, channel_names))
    end
    variables = Dict{String,Any}()
    for (nm, entry) in s["fit_variables"]
        v = Dict{String,Any}(entry)
        v["bindings"] = haskey(bindings_by_name, nm) ? Any[bindings_by_name[nm]...] : String[]
        variables[nm] = v
    end
    fit_out = Dict{String,Any}(
        "sigma" => s["fit"]["sigma"], "axis_unit" => s["fit"]["axis_unit"],
        "intervals" => s["fit_intervals"], "variables" => variables,
        "dependent" => s["fit_dependent"])
    s["fit_regions"] === nothing || (fit_out["regions"] = s["fit_regions"])
    out["fit"] = fit_out

    toml_value = _to_toml_value(out)
    buf = IOBuffer()
    TOML.print(buf, toml_value; sorted=true)
    return String(take!(buf))
end

# ------------------------------------------------------------------------------------------
# The staged procedure (Phase 4.F, Step 4.F.1; OQ-54, the user's decision of 2026-09-30)
# ------------------------------------------------------------------------------------------

"""
    _variable_channels(pspec::ParameterSpec, nchannels::Int) -> Vector{Vector{Int}}

For each variable of `pspec.variables`, in order, the channels whose model it reaches, ascending
(Phase 4.F spec D5): every channel through a `:leaf` binding (the plasma is shared), channel `c`
through an `:effect` binding of channel `c`, each counted for the variable itself and for every
variable a bound dependent reads, directly or through earlier dependents. Empty for a variable
bound to nothing.
"""
function _variable_channels(pspec::ParameterSpec, nchannels::Int)
    nvar = length(pspec.variables)
    dep_vars = Vector{Set{Int}}(undef, length(pspec.dependents))
    for (k, dep) in enumerate(pspec.dependents)
        refs = Set{Int}()
        _referenced_slots!(refs, dep.node)
        acc = Set{Int}()
        for r in refs
            r <= nvar ? push!(acc, r) : union!(acc, dep_vars[r-nvar])
        end
        dep_vars[k] = acc
    end
    reach = [Set{Int}() for _ in 1:nvar]
    for b in pspec.bindings
        vars = b.slot <= nvar ? Set([b.slot]) : dep_vars[b.slot-nvar]
        chans = b.kind === :leaf ? (1:nchannels) : (b.channel:b.channel)
        for i in vars
            union!(reach[i], chans)
        end
    end
    return [sort!(collect(r)) for r in reach]
end

"""
    _stage_deck(deck::Deck, c::Int, procedure, frozen::AbstractVector{<:AbstractString}) -> Deck

Channel `c`'s single-channel deck, derived from the parsed channel deck (Phase 4.F spec D4), as
`examples/problems.jl`'s `single_channel_deck_text` derives it from text: `channels` is that one
channel, its weight `1.0` (a weight only scales a channel against others); `[optimizer]` is the
channel deck's own option tables with `procedure = procedure` and no `staged` table; every name in
`frozen` is `vary = false`, at its `init`. Everything else (the variables, dependents, intervals,
regions, plasma, regularizer, output) is the channel deck's own, deep-copied.
"""
function _stage_deck(deck::Deck, c::Int, procedure, frozen::AbstractVector{<:AbstractString})
    s = deepcopy(deck.sections)
    entry = s["channels"][c]
    entry["weight"] = 1.0
    s["channels"] = Dict{String,Any}[entry]
    opt = Dict{String,Any}(k => v for (k, v) in s["optimizer"] if k != "staged")
    opt["procedure"] = procedure
    s["optimizer"] = opt
    for name in frozen
        s["fit_variables"][name]["vary"] = false
    end
    return Deck(deck.deck_version, s)
end

"""
    _staged_fit(deck::Deck; stage_procedure, joint_optimizer, polish_tol, start_from) -> NamedTuple

The staged procedure (Phase 4.F spec D4 to D6, [OQ-54](../../dev/open_questions.md#oq-54)), on a
channel deck with two or more channels. Each keyword defaults to the deck's own
`[optimizer.staged]` (`joint_optimizer` built from `joint_procedure` and the deck's option tables
by `_build_optimizer`), which is how `fit(deck)` and `examples/fit_shapes.jl` call it; a deck whose
procedure is not `"staged"` is fitted the same way when all four are given
(`examples/problems.jl`'s `staged_joint_fit`). The procedure:

1. `build_problem(deck)`, the joint problem; each fitted variable's reach (`_variable_channels`);
   **every fitted variable's source stage, before anything is fitted**: the channel `start_from`
   names, or else the first channel, in `channels` order, that it reaches. A variable that reaches
   no channel, or a `start_from` channel it does not reach, is a loud error naming it.
2. For each channel in order, its stage deck (`_stage_deck`, every variable that does not reach
   that channel frozen at its `init`), `build_problem`, `solve` with `stage_procedure`'s optimizer
   from the deck's own start.
3. The joint start `flatten(pspec, name => values)` over the fitted variables, each from its
   source stage's `FitResult.values` (a scalar variable's one value), by bin index: the stages
   share the deck's regions and so the bin count.
4. `solve(problem, joint_optimizer; start)`, then `polish(problem, result; tol = polish_tol)`.

Returns `(; deck, problem, optimizer = joint_optimizer, start, result, decreases, stages,
sources)`: `result` and `decreases` `polish`'s; `stages` one `(; name, deck, problem, result,
frozen)` per channel; `sources` one `name => channel` per fitted variable, in `pspec.variables`
order. Pins nothing: `fit(deck)` pins BLAS before calling it (`conventions.md` 6.6).
"""
function _staged_fit(deck::Deck;
    stage_procedure=deck.sections["optimizer"]["staged"]["stage_procedure"],
    joint_optimizer::AbstractOptimizer=_build_optimizer(deck.sections["optimizer"]),
    polish_tol::Real=deck.sections["optimizer"]["staged"]["polish_tol"],
    start_from::AbstractDict=deck.sections["optimizer"]["staged"]["start_from"])
    chan_list = _channel_list(deck.sections)
    nch = length(chan_list)
    names = String[c["name"] for c in chan_list]
    (haskey(deck.sections, "channels") && nch >= 2) || throw(ArgumentError(
        "fit: the staged procedure needs a channel-form deck with two or more channels"))

    problem, _, _ = build_problem(deck)
    pspec = problem.pspec
    reach = _variable_channels(pspec, nch)

    sources = Pair{String,String}[]
    source_index = Dict{String,Int}()
    frozen = [String[] for _ in 1:nch]
    for (i, v) in enumerate(pspec.variables)
        v.vary || continue
        chans = reach[i]
        isempty(chans) && throw(ArgumentError(
            "fit: procedure \"staged\": the fitted variable \"$(v.name)\" reaches no channel's " *
            "model (it is bound to nothing, directly or through a dependent), so no stage fits it"))
        for c in 1:nch
            c in chans || push!(frozen[c], v.name)
        end
        src = first(chans)
        if haskey(start_from, v.name)
            src = findfirst(==(start_from[v.name]), names)
            src in chans || throw(ArgumentError(
                "fit: [optimizer.staged] start_from takes \"$(v.name)\" from channel " *
                "\"$(names[src])\", whose stage freezes it: it reaches only $(names[chans])"))
        end
        push!(sources, v.name => names[src])
        source_index[v.name] = src
    end

    stages = NamedTuple[]
    for c in 1:nch
        sdeck = _stage_deck(deck, c, stage_procedure, frozen[c])
        p_c, opt_c, start_c = build_problem(sdeck)
        r_c = solve(p_c, opt_c; start=start_c)
        push!(stages, (name=names[c], deck=sdeck, problem=p_c, result=r_c, frozen=frozen[c]))
    end

    values = Pair{String,Any}[]
    for v in pspec.variables
        v.vary || continue
        row = stages[source_index[v.name]].result.values[v.name]
        push!(values, v.name => (v.time_varying ? row : row[1]))
    end
    start = flatten(pspec, values)
    joint = solve(problem, joint_optimizer; start=start)
    pol = polish(problem, joint; tol=polish_tol)

    return (; deck, problem, optimizer=joint_optimizer, start, result=pol.result,
        decreases=pol.decreases, stages, sources)
end

# ------------------------------------------------------------------------------------------
# The resonance-width check (Phase 4.F, Step 4.F.2; OQ-53, the user's decision of 2026-09-30)
# ------------------------------------------------------------------------------------------

"""
    _resonance_widths(result::FitResult) -> Vector{Vector{Float64}}

One vector per channel of `result.experiments`, one entry per bin of the fitted axis: the
narrowest EPW resonance's full width at half maximum in that channel's wavelength pixels at the
bin's fitted plasma (`_epw_resonance_widths`), `Inf` where the window holds no resonance or the
bin's plasma has a non-finite leaf (a variable outside its interval). Phase 4.F spec D13.
"""
function _resonance_widths(result::FitResult)
    out = Vector{Vector{Float64}}()
    for expt in result.experiments
        first_order = _prefactor_flag(expt.prefactor)
        tr = expt.trace
        w = fill(Inf, nbins(tr))
        for j in 1:nbins(tr)
            all(isfinite, view(tr.values, :, j)) || continue
            ws = _epw_resonance_widths(plasma(tr, j), expt.probe, expt.collection, expt.lambda,
                first_order)
            isempty(ws) || (w[j] = minimum(ws))
        end
        push!(out, w)
    end
    return out
end

"""
    _narrow_resonance_entries(result, widths) -> Vector{@NamedTuple{channel::String, bin::Int, fwhm_px::Float64, subsamples::Int}}

Every bin of every channel whose resonance is narrower than `RESONANCE_WIDTH_WARN` pixels and
whose channel's sub-sample count (`_subsamples` over its effects, 1 without a `PixelIntegration`)
is below `_recommended_subsamples` of that width; `subsamples` is the channel's count today.
Channels in order, then bins ascending. Phase 4.F spec D15.
"""
function _narrow_resonance_entries(result::FitResult, widths)
    entries = @NamedTuple{channel::String, bin::Int, fwhm_px::Float64, subsamples::Int}[]
    for (c, expt) in enumerate(result.experiments)
        n = maximum(_subsamples, expt.effects; init=1)
        for (j, w) in enumerate(widths[c])
            (w < RESONANCE_WIDTH_WARN && n < _recommended_subsamples(w)) || continue
            push!(entries, (channel=result.channels[c].name, bin=j, fwhm_px=w, subsamples=n))
        end
    end
    return entries
end

"""
    _narrow_resonance_message(result::FitResult, entries) -> String

The text of `_warn_narrow_resonance`'s warning (Phase 4.F spec D15): a first line with the count
and the reason, then one line per channel with an entry, its bins and widths in pixels to two
decimals, and the key to refit with, `pixel_integration = { subsamples = N }`, `N`
`_recommended_subsamples` of the channel's narrowest width, in that channel's instrument table
(`[instrument]` for a shorthand deck's one channel, `"main"`).
"""
function _narrow_resonance_message(result::FitResult, entries)
    n = length(entries)
    head = n == 1 ? "fit: 1 bin has" : "fit: $n bins have"
    lines = String[head * " an EPW resonance narrower than RESONANCE_WIDTH_WARN = " *
                   "$RESONANCE_WIDTH_WARN wavelength pixel, which the model's sampling of each pixel " *
                   "aliases (the result is as fitted; refit with the key named for each channel):"]
    for chan in unique(e.channel for e in entries)
        mine = [e for e in entries if e.channel == chan]
        bins = (length(mine) == 1 ? "bin " : "bins ") * join((e.bin for e in mine), ", ")
        fwhm = join((@sprintf("%.2f", e.fwhm_px) for e in mine), ", ")
        rec = _recommended_subsamples(minimum(e.fwhm_px for e in mine))
        table = (length(result.channels) == 1 && chan == "main") ? "[instrument]" :
                "[channel.$chan.instrument]"
        push!(lines, "  $chan: $bins (FWHM $fwhm px; $(mine[1].subsamples) per pixel now); " *
                     "$table pixel_integration = { subsamples = $rec }")
    end
    return join(lines, "\n")
end

"""
    _warn_narrow_resonance(result::FitResult) -> Vector{@NamedTuple{channel::String, bin::Int, fwhm_px::Float64, subsamples::Int}}

[OQ-53](../../dev/open_questions.md#oq-53) (a), the user's decision of 2026-09-30 (Phase 4.F spec
D13 to D16): one `@warn` when a fitted bin's EPW resonance is narrower than
`RESONANCE_WIDTH_WARN` pixels and its channel samples each pixel at fewer points than the width
asks for (`_narrow_resonance_entries`), naming the channels, bins and the count to refit with;
silent otherwise; returns the entries either way. A report: the fit, its `converged`, `status`,
`theta` and every dataset `write_result` writes are exactly as they would be without it.
"""
function _warn_narrow_resonance(result::FitResult)
    entries = _narrow_resonance_entries(result, _resonance_widths(result))
    isempty(entries) && return entries
    @warn _narrow_resonance_message(result, entries) threshold = RESONANCE_WIDTH_WARN status = result.status converged = result.converged
    return entries
end

# ------------------------------------------------------------------------------------------
# fit
# ------------------------------------------------------------------------------------------

"""
    _fit_gn_decrease(problem::AbstractFitProblem, result::FitResult) -> Float64

`gn_decrease(problem, result.theta)`, or `NaN` with a warning when `JᵀJ` over the free entries is
not positive definite (`PosDefException`, e.g. a fitted variable no residual depends on). Any other
error propagates. The fit itself succeeded, so a failed check must not lose its result (Phase 4.B
spec D5).
"""
function _fit_gn_decrease(problem::AbstractFitProblem, result::FitResult)
    try
        return gn_decrease(problem, result.theta)
    catch err
        err isa PosDefException || rethrow()
        @warn "fit: gn_decrease could not be computed: J'J over the free entries is not positive definite" status = result.status converged = result.converged
        return NaN
    end
end

"""
    _warn_fit_minimum(result::FitResult, decrease::Real, fraction::Real) -> nothing

OQ-41 option 1's two warnings, whatever `result.status` is (Phase 4.B spec D5). `NaN > x` is
`false`, so a `NaN` decrease, already warned by `_fit_gn_decrease`, does not warn again here.
"""
function _warn_fit_minimum(result::FitResult, decrease::Real, fraction::Real)
    if decrease > GN_DECREASE_WARN
        @warn "fit: gn_decrease exceeds GN_DECREASE_WARN; the returned theta may not be a minimum (polish(problem, result) takes Gauss-Newton steps to one)" gn_decrease = decrease threshold = GN_DECREASE_WARN status = result.status converged = result.converged
    end
    if fraction >= SATURATION_FRACTION
        @warn "fit: the near-bound fraction reaches SATURATION_FRACTION; the returned theta is pinned at its bounds" near_bound_fraction = fraction threshold = SATURATION_FRACTION status = result.status converged = result.converged
    end
    return nothing
end

"""
    _near_bound_message(result::FitResult, entries) -> String

The text of `_warn_near_bound`'s warning (Phase 4.D spec D5): a first line with the count, then one
line per fitted variable with an entry, `"  <name>: bin 1 (lower bound 2.0)"`, `"bins 3, 4 (upper
bound 300.0)"`, or `"scalar (upper bound 150.0)"` for a scalar variable, the lower bound's entries
before the upper's, variables in `pspec.variables` order.
"""
function _near_bound_message(result::FitResult, entries)
    n = length(entries)
    head = n == 1 ? "fit: 1 fitted entry is" : "fit: $n fitted entries are"
    lines = String[head * " within BOUND_PROXIMITY_TOL of a bound, where dx/dy -> 0 and an entry " *
                   "may have stalled short of a minimum (the result is as fitted; `converged` does not see this):"]
    for v in result.pspec.variables
        mine = [e for e in entries if e.variable == v.name]
        isempty(mine) && continue
        parts = String[]
        for (side, bnd) in ((:lower, v.lower), (:upper, v.upper))
            bins = [e.bin for e in mine if e.bound === side]
            isempty(bins) && continue
            place = bins == [0] ? "scalar" : (length(bins) == 1 ? "bin " : "bins ") * join(bins, ", ")
            push!(parts, "$place ($side bound $bnd)")
        end
        push!(lines, "  $(v.name): " * join(parts, "; "))
    end
    return join(lines, "\n")
end

"""
    _warn_near_bound(result::FitResult) -> Vector{@NamedTuple{variable::String, bin::Int, bound::Symbol}}

[OQ-49](../../dev/open_questions.md#oq-49) option (a), the user's decision of 2026-09-29 (Phase 4.D
spec D5): one `@warn` when **any** fitted entry of `result` is within `BOUND_PROXIMITY_TOL` of a
bound, naming each variable and its bins (`_near_bound_message`), silent when there is none;
returns `_near_bound_entries(result)` either way. It changes nothing: the fit, its `converged` and
its `status` (OQ-36) and the result `fit(deck)` writes are exactly as they would be without it.
`SATURATION_FRACTION`'s warning (`_warn_fit_minimum`) still fires beside it at its own threshold;
this one sees the single saturated coordinate that one cannot (Phase 4.B's A14). Called by
`fit(deck)` after `solve`, and by the slow tier's FR11 on fits `fit(deck)` does not make (TS2's
staged fit goes through `fit(deck)` since Phase 4.F, spec D9).
"""
function _warn_near_bound(result::FitResult)
    entries = _near_bound_entries(result)
    isempty(entries) && return entries
    @warn _near_bound_message(result, entries) near_bound_fraction = _near_bound_fraction(result) status = result.status converged = result.converged
    return entries
end

"""
    fit(path::AbstractString) -> FitResult
    fit(deck::Deck) -> FitResult

The driver (spec D33, D40, `conventions.md` 6.6): reads the deck (if given a path), builds the
problem and the optimizer (`build_problem`; a plain `FitProblem` for one channel, a
`JointFitProblem` for two or more), pins BLAS to one thread (`pin_blas_threads()`, called exactly
once -- `fit` is the driver `conventions.md` 6.6 has been waiting for since Phase 2), solves,
writes the result and the effective deck into `[output].dir` (`<dir>/result.jld2`,
`<dir>/effective_deck.toml`, creating the directory if needed), and returns the `FitResult`.
`write_result` is always given the problem, so `fit`'s own result files always carry a residual and
a mask per channel.

**The minimum check** (Phase 4.B spec D5, [OQ-41](../../dev/open_questions.md#oq-41) option 1):
after `solve`, `fit` computes `gn_decrease(problem, result.theta)` and the near-bound fraction,
warns when the first exceeds `GN_DECREASE_WARN` or the second reaches `SATURATION_FRACTION`,
**whatever the stop**, and writes `fit/gn_decrease` into the result file. `converged` means what
OQ-36 says and is not changed: a converged fit may still warn. The check costs one
`normal_matrices` call, which scales with `ntheta` times the pixel count: about 1 s at `ntheta =
128` on the two-channel problem, about a minute at `ntheta = 768` on `examples/epw_ablation.toml`'s
full-resolution problem (Phase 4.B spec D28, measured 2026-09-26).

**The saturation warning** (Phase 4.D spec D5, [OQ-49](../../dev/open_questions.md#oq-49) option
(a)): after the minimum check, `fit` calls `_warn_near_bound(result)`, one warning naming every
fitted entry within `BOUND_PROXIMITY_TOL` of a bound (a variable and its bins, or `scalar`), silent
when there is none, and `write_result` lists the same entries in `fit/near_bound_entries/*`. Neither
changes the fit: its `theta`, `converged` and `status` are `solve`'s.

**The staged procedure** (Phase 4.F spec D3 to D7, [OQ-54](../../dev/open_questions.md#oq-54)):
with `procedure = "staged"`, `fit` pins BLAS and runs `_staged_fit` in place of `build_problem`
and `solve`: each channel fitted alone, the joint fit started from those fits, then `polish`. Every
check, warning and file this docstring describes is the final fit's, and `write_result` also
writes the stages' results (`stages/*`).

**The resonance-width check** (Phase 4.F spec D13 to D16, [OQ-53](../../dev/open_questions.md#oq-53)):
after the saturation warning, `fit` calls `_warn_narrow_resonance(result)`, one warning when a
fitted bin's EPW resonance is narrower than `RESONANCE_WIDTH_WARN` wavelength pixels and its
channel does not yet sample each pixel finely enough, naming the bins and the
`pixel_integration` count to refit with; silent otherwise. It changes nothing.

Spec D33, D40.
"""
function fit(deck::Deck)
    deck.sections["output"] === nothing && throw(ArgumentError(
        "fit: the deck has no \"[output]\" section (needs \"dir\")"))
    staged = nothing
    if deck.sections["optimizer"]["procedure"] == STAGED_PROCEDURE
        pin_blas_threads()
        staged = _staged_fit(deck)
        problem, optimizer, result = staged.problem, staged.optimizer, staged.result
    else
        problem, optimizer, start = build_problem(deck)
        pin_blas_threads()
        result = solve(problem, optimizer; start=start)
    end
    decrease = _fit_gn_decrease(problem, result)
    _warn_fit_minimum(result, decrease, _near_bound_fraction(result))
    _warn_near_bound(result)
    _warn_narrow_resonance(result)
    dir = deck.sections["output"]["dir"]
    mkpath(dir)
    eff = effective_deck(deck, problem, optimizer)
    write_result(joinpath(dir, "result.jld2"), result; problem=problem, deck=eff, source="fit",
        gn_decrease=decrease, stages=staged)
    open(joinpath(dir, "effective_deck.toml"), "w") do io
        write(io, eff)
    end
    return result
end

fit(path::AbstractString) = fit(read_deck(path))
