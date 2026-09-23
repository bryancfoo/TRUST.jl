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

# The regions (D15) supersede t_range_ns, are averaged onto in data preparation (D13, D14) and
# may each name a [fit.intervals] entry -- amended 2026-09-23 before chunk 5 with four loud
# errors in place of a silent behaviour: see _validate_data_section's docstring.
const _DATA_KEYS = ("file", "lambda_range_nm", "t_range_ns", "regions", "t_shift_ns", "t_shift_s",
    "min_usable_fraction")
const _REGION_KEYS = ("t_range_ns", "t_range_s", "nbins", "interval")
const _REGION_RANGE_KEYS = ("t_range_ns", "t_range_s")

"""
    _validate_data_section(given::AbstractDict) -> Dict{String,Any}

Validates `[data]` (D15, amended 2026-09-23). `regions` is stored as a `Vector{Dict{String,Any}}`,
one per `[[data.regions]]` table, each carrying exactly the unit key the deck gave (a two-element
`Any[Float64, Float64]`), `nbins::Int`, and `interval` filled with `""` when absent;
`min_usable_fraction` is filled with `0.5` when `regions` is given and the key is absent; neither
`t_shift` key is ever filled in (absent means zero to `average_regions`, and filling one next to
the user's other unit keyword would make the effective deck re-read with both given).

Four loud errors, each in place of a silent behaviour:

1. Both `regions` and `t_range_ns` -- the roadmap's "regions supersede t_range_ns" would silently
   ignore whichever is not honoured; `lambda_range_nm` is unaffected. **[propagate]** to
   `roadmap.md` Step 4.A.2 (this document decides it, Level 3 does not choose it).
2. `t_shift_ns` and `t_shift_s` together -- the unit table's own error, surfaced early here as
   well as later through `_si_value`.
3. A region table naming a key other than `t_range_ns`/`t_range_s`/`nbins`/`interval`, or naming
   both/neither of the two range keys.
4. `t_shift_ns`, `t_shift_s` or `min_usable_fraction` given without `regions` -- without regions
   nothing is averaged, so the value would be silently ignored.

(The fifth loud error of D15 -- a region naming an interval `[fit.intervals]` also declares -- is
cross-section and is checked in `build_problem` instead.)

Spec D15.
"""
function _validate_data_section(given::AbstractDict)
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

    out = Dict{String,Any}(given)

    if haskey(given, "regions")
        haskey(given, "t_range_ns") && throw(ArgumentError(
            "read_deck: section \"data\" gives both \"regions\" and \"t_range_ns\"; regions " *
            "supersede t_range_ns, and a deck whose stated time window would be silently " *
            "ignored is refused instead"))

        regions_raw = given["regions"]
        regions_raw isa AbstractVector || throw(ArgumentError(
            "read_deck: \"data.regions\" must be an array of tables"))
        regions_out = Vector{Dict{String,Any}}(undef, length(regions_raw))
        for (i, entry) in enumerate(regions_raw)
            entry isa AbstractDict || throw(ArgumentError(
                "read_deck: \"data.regions[$i]\" must be a table"))
            _check_known_keys(entry, _REGION_KEYS, "data.regions[$i]")
            present = [k for k in _REGION_RANGE_KEYS if haskey(entry, k)]
            length(present) == 1 || throw(ArgumentError(
                "read_deck: \"data.regions[$i]\" needs exactly one of $_REGION_RANGE_KEYS"))
            key = present[1]
            rng = entry[key]
            (rng isa AbstractVector && length(rng) == 2 && all(x -> x isa Real, rng)) ||
                throw(ArgumentError(
                    "read_deck: \"data.regions[$i].$key\" must be a two-element [lo, hi]"))
            haskey(entry, "nbins") || throw(ArgumentError(
                "read_deck: \"data.regions[$i]\" needs \"nbins\""))
            entry["nbins"] isa Integer || throw(ArgumentError(
                "read_deck: \"data.regions[$i].nbins\" must be an integer"))
            interval = get(entry, "interval", "")
            interval isa AbstractString || throw(ArgumentError(
                "read_deck: \"data.regions[$i].interval\" must be a string"))
            regions_out[i] = Dict{String,Any}(
                key => Any[Float64(rng[1]), Float64(rng[2])],
                "nbins" => Int(entry["nbins"]),
                "interval" => String(interval))
        end
        out["regions"] = regions_out
        out["min_usable_fraction"] = Float64(get(given, "min_usable_fraction", 0.5))
    else
        for k in ("t_shift_ns", "t_shift_s", "min_usable_fraction")
            haskey(given, k) && throw(ArgumentError(
                "read_deck: section \"data\" gives \"$k\" without \"regions\"; without " *
                "regions nothing is averaged and the value would be silently ignored"))
        end
    end

    return out
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
    _check_known_keys(given, (_FIT_SCALAR_KEYS..., "intervals", "variables", "dependent"), "fit")
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
const _SHAPE_OPTIONS = Dict{String,Any}("Maxwellian" => MaxwellianOptions, "SuperGaussian" => SuperGaussianOptions)

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
    "sensitivity" => "Sensitivity")
const _INSTRUMENT_TOP_KEYS = (keys(_INSTRUMENT_SINGLE_REGISTRY_NAME)..., "background")

const _INSTRUMENT_OPTIONS = Dict{String,Any}(
    "SpectralIRF" => SpectralIRFOptions, "TemporalIRF" => TemporalIRFOptions,
    "KSmearing" => KSmearingOptions, "GradientSmearing" => GradientSmearingOptions,
    "WavelengthCalibration" => WavelengthCalibrationOptions, "Sensitivity" => SensitivityOptions,
    "Constant" => ConstantBackgroundOptions, "Polynomial" => PolynomialBackgroundOptions,
    "Rational" => RationalBackgroundOptions)

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
        refs_reason = regname == "GradientSmearing" ? _GRADIENT_SMEARING_WIDTH_REFS_REASON : ""
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

const _OPTIMIZER_TOP_KEYS = ("procedure", "lbfgs", "levenberg_marquardt", "noisy_descent")
# _PROCEDURE_NAMES and _PROCEDURE_BASE_NAMES are optimizers.jl's own (Step 4.A.0, D22 item 1):
# this file used to redefine both, with a different order, and the redefinition silently won
# by include order, so every message below printed the wrong list.

function _check_procedure_name(nm)
    nm isa AbstractString && nm in _PROCEDURE_NAMES || throw(ArgumentError(
        "read_deck: unknown procedure \"$nm\"; valid values are $_PROCEDURE_NAMES, or a list of " *
        "$_PROCEDURE_BASE_NAMES"))
    return String(nm)
end

function _validate_optimizer_section(given::AbstractDict)
    _check_known_keys(given, _OPTIMIZER_TOP_KEYS, "optimizer")
    procedure_raw = get(given, "procedure", "noisy_descent_lbfgs")
    procedure = if procedure_raw isa AbstractVector
        for nm in procedure_raw
            nm isa AbstractString && nm in _PROCEDURE_BASE_NAMES || throw(ArgumentError(
                "read_deck: unknown procedure \"$nm\"; valid values are $_PROCEDURE_NAMES, or a " *
                "list of $_PROCEDURE_BASE_NAMES"))
        end
        Any[String(nm) for nm in procedure_raw]
    else
        _check_procedure_name(procedure_raw)
    end
    out = Dict{String,Any}("procedure" => procedure)
    empty_names = Set{String}()
    for (key, ctor) in OPTIMIZER_REGISTRY
        haskey(given, key) || continue
        entry = given[key]
        entry isa AbstractDict || throw(ArgumentError("read_deck: \"optimizer.$key\" must be a table"))
        out[key] = _fill_option_fields(ctor, entry, empty_names, "optimizer.$key", false)
    end
    return out
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

"""
    read_deck(path::AbstractString) -> Deck

Reads and validates the TOML deck at `path` (spec D33): checks `deck_version` (a missing or
unknown value is a loud error naming the version found and the one supported, DK16); checks every
key of every section against the component that owns it (7.3, DK7); checks every numeric key's
string value against the declared `[fit.variables]`/`[fit.dependent]` names (7.6, DK4-DK6); fills
every option-struct field with its own default where the deck omits it (7.5, DK11). Returns a
`Deck`, still inert -- build the actual `FitProblem` with `build_problem`.

Sections are processed in the order D33 lists them: `[data]`, `[geometry]`, `[fit.intervals]`,
`[fit.variables]`, `[fit.dependent]`, `[plasma]` (and its species tables), `[instrument]`,
`[fit]`, `[regularizer]`, `[optimizer]`, `[output]`.

Spec D33.
"""
function read_deck(path::AbstractString)
    raw = TOML.parsefile(path)
    version = get(raw, "deck_version", nothing)
    version == DECK_VERSION || throw(ArgumentError(
        "read_deck: \"$path\" has deck_version " *
        (version === nothing ? "<missing>" : "\"$version\"") *
        "; this version of OTIS reads \"$DECK_VERSION\""))

    sections = Dict{String,Any}()

    haskey(raw, "data") || throw(ArgumentError("read_deck: \"$path\" is missing section \"data\""))
    sections["data"] = _validate_data_section(raw["data"])

    haskey(raw, "geometry") || throw(ArgumentError("read_deck: \"$path\" is missing section \"geometry\""))
    sections["geometry"] = _validate_geometry_section(raw["geometry"])

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

    haskey(raw, "plasma") || throw(ArgumentError("read_deck: \"$path\" is missing section \"plasma\""))
    sections["plasma"] = _validate_plasma_section(raw["plasma"], known_names)

    sections["instrument"] = _validate_instrument_section(get(raw, "instrument", Dict{String,Any}()), known_names)

    sections["regularizer"] = haskey(raw, "regularizer") ?
        _validate_regularizer_section(raw["regularizer"]) : nothing

    sections["optimizer"] = _validate_optimizer_section(get(raw, "optimizer", Dict{String,Any}()))

    sections["output"] = haskey(raw, "output") ? _validate_output_section(raw["output"]) : nothing

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
    "Polynomial" => PolynomialBackground, "Rational" => RationalBackground)

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
    keys_ = procedure == "noisy_descent_lbfgs" ? ["noisy_descent", "lbfgs"] :
            procedure isa AbstractVector ? procedure : [procedure]
    members = [_build_one_optimizer(k, section) for k in keys_]
    return length(members) == 1 ? members[1] : OptimizerSequence(members...)
end

# ------------------------------------------------------------------------------------------
# build_problem
# ------------------------------------------------------------------------------------------

"""
    build_problem(deck::Deck) -> (FitProblem, AbstractOptimizer, Vector{Float64})

Builds the actual Julia objects a validated `Deck` describes (spec D33): reads and crops the data
(`[data]`), builds the probe and collection (`[geometry]`, fixed), the named intervals and fitted
variables (`[fit.intervals]`, `[fit.variables]`), the template plasma and the instrument chain
(`[plasma]`, `[instrument]`, every `_VarRef` substituted by its placeholder value and recorded as a
binding), the `Experiment`, the `ParameterSpec` (which resolves every binding for real, D9, and
rebuilds the trace with the union of leaves), the regularizer (`[regularizer]`) and the optimizer
(`[optimizer]`). Returns the `FitProblem`, the optimizer, and the start vector `flatten(pspec)`.

Spec D33.
"""
function build_problem(deck::Deck)
    s = deck.sections
    data0 = read_streak(s["data"]["file"])
    regions_given = haskey(s["data"], "regions")
    data1 = _crop_streak(data0, get(s["data"], "lambda_range_nm", nothing),
        regions_given ? nothing : get(s["data"], "t_range_ns", nothing))

    # Route B's master axis, the data averaged onto it rather than the model mapped to the
    # detector's bins (OQ-34, decided 2026-09-21; D13-D16). Only the time axis is averaged, and
    # this is the only place average_regions is called from the deck.
    regions = regions_given ? _build_averaging_regions(s["data"]["regions"]) : AveragingRegion[]
    data = regions_given ? average_regions(data1, regions; _averaging_kwargs(s["data"])...) : data1

    probe, collection = _build_geometry(s["geometry"])

    axis = data.t
    axis_unit = Symbol(s["fit"]["axis_unit"])
    factor = axis_unit === :ns ? 1e-9 : 1.0

    # D15 amendment 1: a region naming an interval [fit.intervals] also declares is a loud error;
    # otherwise the region's own interval is synthesized into a LOCAL copy of fit_intervals, never
    # into deck.sections (effective_deck re-serializes deck.sections as they stand, D15 amendment).
    fit_intervals = s["fit_intervals"]
    if !isempty(regions)
        synthesized = Dict{String,Any}()
        for region in regions
            isempty(region.interval) && continue
            haskey(fit_intervals, region.interval) && throw(ArgumentError(
                "read_deck: \"data.regions\" names interval \"$(region.interval)\", which " *
                "\"fit.intervals\" also declares"))
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

    pending = Tuple{String,Any,Symbol,Int,Float64,String}[]
    effects = AbstractInstrumentEffect[]
    for key in ("k_smearing", "gradient_smearing", "wavelength_calibration", "spectral_irf", "temporal_irf",
                "sensitivity")
        haskey(s["instrument"], key) || continue
        regname = _INSTRUMENT_SINGLE_REGISTRY_NAME[key]
        push!(effects, _build_effect_and_pending!(pending, regname, s["instrument"][key], vals, template))
    end
    for bg in get(s["instrument"], "background", Any[])
        bg_fields = Dict(k => v for (k, v) in bg if k != "type")
        push!(effects, _build_effect_and_pending!(pending, bg["type"], bg_fields, vals, template))
    end

    trace0 = PlasmaTrace(template; axis_name=:t, t_s=axis, leaves=())
    expt0 = Experiment(; trace=trace0, probe=probe, collection=collection, lambda_m=data.lambda,
        effects=Tuple(effects))

    bindings_instrument = Pair{String,Any}[]
    for (nm, effect, field, elem, factor, unit_key) in pending
        idx = findfirst(e -> e === effect, expt0.effects)
        target = (idx, field, elem, factor, unit_key)
        push!(bindings_instrument, nm => target)
    end

    pspec = ParameterSpec(expt0; variables=vars, dependents=dep_pairs, intervals=ivals,
        bindings=vcat(bindings_plasma, bindings_instrument), axis_unit=axis_unit)

    working_expt = Experiment(; trace=pspec.trace, probe=expt0.probe, collection=expt0.collection,
        lambda_m=expt0.lambda, effects=expt0.effects, prefactor=expt0.prefactor)

    regularizer = _build_regularizer(s["regularizer"], pspec)
    problem = FitProblem(working_expt, pspec, data; regularizer=regularizer, scheduler=:auto)

    optimizer = _build_optimizer(s["optimizer"])
    start = flatten(pspec)
    return problem, optimizer, start
end

# ------------------------------------------------------------------------------------------
# effective_deck
# ------------------------------------------------------------------------------------------

function _binding_description(b::Binding)
    b.kind === :leaf && return string(b.path) * (isempty(b.unit_key) ? "" : " (unit_key=$(b.unit_key))")
    suffix = b.element == 0 ? "" : "[$(b.element)]"
    return "instrument.effect[$(b.effect)].$(b.field)$suffix" *
           (isempty(b.unit_key) ? "" : " (unit_key=$(b.unit_key))")
end

"""
    effective_deck(deck::Deck, problem::FitProblem, optimizer::AbstractOptimizer) -> String

The effective configuration (`conventions.md` 7.5, spec D33): `deck`'s own validated sections
(every default already filled by `read_deck`), re-serialized as TOML, with `deck_version` written
first and three things the input deck does not say added: every variable's resolved binding(s)
(under its own `[fit.variables.<name>]` table, `bindings`, one string per binding, D9's
leaf-path-and-unit-keyword or effect-parameter form), every dependent's expression text (already
present, verbatim, under `[fit.dependent]`), and the axis unit (`[fit].axis_unit`, already filled
in by `read_deck`'s own default). A fit is reproducible from this string alone (7.5): re-reading it
with `read_deck` succeeds and yields the same `Deck` (DK1).

Spec D33.
"""
function effective_deck(deck::Deck, problem::FitProblem, ::AbstractOptimizer)
    s = deck.sections
    out = Dict{String,Any}()
    out["deck_version"] = deck.deck_version
    out["data"] = s["data"]
    out["geometry"] = s["geometry"]
    out["plasma"] = s["plasma"]
    out["instrument"] = s["instrument"]
    s["regularizer"] === nothing || (out["regularizer"] = s["regularizer"])
    out["optimizer"] = s["optimizer"]
    s["output"] === nothing || (out["output"] = s["output"])

    bindings_by_name = Dict{String,Vector{String}}()
    for b in problem.pspec.bindings
        push!(get!(bindings_by_name, b.name, String[]), _binding_description(b))
    end
    variables = Dict{String,Any}()
    for (nm, entry) in s["fit_variables"]
        v = Dict{String,Any}(entry)
        v["bindings"] = haskey(bindings_by_name, nm) ? Any[bindings_by_name[nm]...] : String[]
        variables[nm] = v
    end
    out["fit"] = Dict{String,Any}(
        "sigma" => s["fit"]["sigma"], "axis_unit" => s["fit"]["axis_unit"],
        "intervals" => s["fit_intervals"], "variables" => variables,
        "dependent" => s["fit_dependent"])

    toml_value = _to_toml_value(out)
    buf = IOBuffer()
    TOML.print(buf, toml_value; sorted=true)
    return String(take!(buf))
end

# ------------------------------------------------------------------------------------------
# fit
# ------------------------------------------------------------------------------------------

"""
    fit(path::AbstractString) -> FitResult
    fit(deck::Deck) -> FitResult

The driver (spec D33, `conventions.md` 6.6): reads the deck (if given a path), builds the
`FitProblem` and the optimizer (`build_problem`), pins BLAS to one thread (`pin_blas_threads()`,
called exactly once -- `fit` is the driver `conventions.md` 6.6 has been waiting for since Phase
2), solves, writes the result and the effective deck into `[output].dir`
(`<dir>/result.jld2`, `<dir>/effective_deck.toml`, creating the directory if needed), and returns
the `FitResult`.

Spec D33.
"""
function fit(deck::Deck)
    deck.sections["output"] === nothing && throw(ArgumentError(
        "fit: the deck has no \"[output]\" section (needs \"dir\")"))
    problem, optimizer, start = build_problem(deck)
    pin_blas_threads()
    result = solve(problem, optimizer; start=start)
    dir = deck.sections["output"]["dir"]
    mkpath(dir)
    eff = effective_deck(deck, problem, optimizer)
    write_result(joinpath(dir, "result.jld2"), result; deck=eff, source="fit")
    open(joinpath(dir, "effective_deck.toml"), "w") do io
        write(io, eff)
    end
    return result
end

fit(path::AbstractString) = fit(read_deck(path))
