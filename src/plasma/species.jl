# SpeciesSpec (input form), Species (resolved form, produced only by Plasma), SpeciesSet
# (named, indexable container), the electrons/ion builders, spec, show, and name validation.

using Printf: @printf

"""
    SpeciesSpec{FT<:Real, F<:Union{AbstractVelocityDistribution, AbstractLabDistribution}}

The input form of one particle population: its name, mass, charge number, flow, temperature,
distribution and fraction, with **no density** (density is resolved only by `Plasma`, which turns a
`SpeciesSpec` into a `Species`). Built by `electrons` and `ion`, never directly.

**The distribution `f` is of either family** (Phase G.1, [OQ-68](../../dev/open_questions.md#oq-68)
item 5; `conventions.md` 1.12): a dimensionless shape (`AbstractVelocityDistribution`), for which
`u` and `T` are inputs, or a lab-frame distribution (`AbstractLabDistribution`, a `GaussianGrid`),
for which `u` and `T` are **results**, the distribution's mean velocity and `mass` times its
variance, set by the builder and never given.

**`charge` is `FT`, a real number** (Phase 4.J, [OQ-60](../../dev/open_questions.md#oq-60); an `Int`
until then): `-1` for an electron, an ion's charge number otherwise, at least 1. An integer is one
charge state; a non-integer stands for a mixture of charge states of one element at one temperature
and flow, and is the mixture's ⟨Z²⟩/⟨Z⟩, which reproduces it exactly (`conventions.md` 1.7).

`conventions.md` Section 4; spec D11; Phase 4.J spec D7.
"""
struct SpeciesSpec{FT<:Real, F<:Union{AbstractVelocityDistribution, AbstractLabDistribution}}
    name::String
    mass::FT
    charge::FT
    u::FT
    T::FT
    f::F
    fraction::FT
end

"""
    Species{FT<:Real, F<:Union{AbstractVelocityDistribution, AbstractLabDistribution}}

The resolved form of one particle population: every `SpeciesSpec` field plus the number
density `n` (m⁻³). Its distribution `f` is of either family (see `SpeciesSpec`); for a lab-frame
distribution `u` and `T` are the distribution's moments, recomputed wherever the distribution is
replaced (`plasma(trace, j)`), so that `T` is the variance temperature along **k̂** and `u` the mean
velocity on every species (`conventions.md` 1.12). Produced **only** by the `Plasma` constructor, which is what turns a
fraction of `n_e` into a density; there is no outer constructor and no lowercase builder for
`Species`, and constructing one directly is not supported (Julia cannot hide the default inner
constructor of an exported type; this is the closest faithful implementation of "not
constructible by users," spec D11).

**`charge` is `FT`** (Phase 4.J): see `SpeciesSpec`. An ion's `n` is `fraction · n_e / charge`; for a
species that stands for a mixture of charge states that is the equivalent single-state ion's density,
below the mixture's own number density by ⟨Z⟩/(⟨Z²⟩/⟨Z⟩) ([OQ-60](../../dev/open_questions.md#oq-60)).

`conventions.md` Section 4; spec D11; Phase 4.J spec D7.
"""
struct Species{FT<:Real, F<:Union{AbstractVelocityDistribution, AbstractLabDistribution}}
    name::String
    mass::FT
    charge::FT
    u::FT
    T::FT
    f::F
    fraction::FT
    n::FT
end

"""
    spec(sp::Species) -> SpeciesSpec

The input form of `sp`: every field except the density `n`.

`conventions.md` Section 4; spec D11, D28.
"""
spec(sp::Species) = SpeciesSpec(sp.name, sp.mass, sp.charge, sp.u, sp.T, sp.f, sp.fraction)

"""
    _shape_eltype(f) -> Type

`promote_type` over the types of `values(parameters(f))`, or `Bool` (the promotion identity)
for a shape with no parameters (`Maxwellian`). Folded into `electrons` and `ion`'s own element
type so that a shape with dual parameters makes its species (and hence its plasma) dual with
zero partials on the scalar fields, without `_spectrum_eltype` needing any knowledge of shapes
([OQ-9](../../dev/open_questions.md#oq-9) trap 6, decided by Level 1 2026-09-09; spec Step 2.0
item 2). The shape `f` itself is stored as given and is never converted here: it already carries
whatever element type its own parameters do. A lab-frame distribution is served by the same method,
and `GaussianGrid` by its own (`gaussian_grid.jl`), whose parameters are one `NTuple` (Phase G.1 spec D8).
"""
function _shape_eltype(f::Union{AbstractVelocityDistribution, AbstractLabDistribution})
    p = values(parameters(f))
    isempty(p) && return Bool
    return promote_type(typeof.(p)...)
end

"""
    _distribution_u_T(f, mass, u, T) -> (u, T)

A species' flow and temperature given its distribution `f` (Phase G.1 spec D8): `(u, T)` unchanged
for a shape, whose `u` and `T` are inputs; `(mean_velocity(f), mass * velocity_variance(f))` for a
lab-frame distribution, whose `u` and `T` are results (`conventions.md` 1.12). Called where a species
is built from a new distribution: `electrons`, `ion` and `_snapshot_species` (`plasma(trace, j)`).
Allocation-free.
"""
_distribution_u_T(::AbstractVelocityDistribution, mass, u, T) = (u, T)
_distribution_u_T(f::AbstractLabDistribution, mass, u, T) = (mean_velocity(f), mass * velocity_variance(f))

"""
    _derives_u_T(f) -> Bool

`true` for a lab-frame distribution, whose species' `u` and `T` are computed from it and are not
leaves; `false` for a shape (Phase G.1 spec D8, D9).
"""
_derives_u_T(::AbstractVelocityDistribution) = false
_derives_u_T(::AbstractLabDistribution) = true

# The keyword set that makes a species' T and u; refused beside a lab-frame distribution (spec D9).
const _SPECIES_U_T_KEYWORDS = (:T_eV, :T_keV, :u_kms, :u_ms)

"""
    _species_distribution(name, f, grid, mass) -> distribution

The distribution a builder stores (Phase G.1 spec D9): `Maxwellian()` when neither `f` nor `grid`
is given; `f` itself when it is a distribution of either family; and for `grid`, a `NamedTuple` of
`GaussianGrid`'s temperature-form keywords, `GaussianGrid(; mass_kg = mass, grid...)` with the
species' own mass, so that the mass is never typed twice. Refused, naming the species: `f` and
`grid` both given; an `f` that is not a distribution; a `grid` that is not a `NamedTuple`, that
names a mass, or that has no `T_ref_eV` or `T_ref_keV`; a mass that is not positive.
"""
_species_distribution(name, ::Nothing, ::Nothing, mass) = Maxwellian()
_species_distribution(name, f::Union{AbstractVelocityDistribution, AbstractLabDistribution}, ::Nothing, mass) = f
function _species_distribution(name, ::Nothing, grid::NamedTuple, mass)
    (haskey(grid, :mass_amu) || haskey(grid, :mass_kg)) && throw(ArgumentError(
        "species \"$name\": the grid takes the species' own mass; mass_amu or mass_kg is not a grid keyword here"))
    (haskey(grid, :T_ref_eV) || haskey(grid, :T_ref_keV)) || throw(ArgumentError(
        "species \"$name\": grid is GaussianGrid's temperature form and needs T_ref_eV or T_ref_keV; " *
        "for an explicit spacing pass f = GaussianGrid(dv_kms = ..., default_weights = ...)"))
    _primal_value(mass) > 0 || throw(ArgumentError("mass must be positive; got $mass kg for species \"$name\""))
    return GaussianGrid(; mass_kg = Float64(_primal_value(mass)), grid...)
end
function _species_distribution(name, f, grid, mass)
    (f !== nothing && grid !== nothing) && throw(ArgumentError(
        "species \"$name\": give f or grid, not both"))
    f === nothing || throw(ArgumentError(
        "species \"$name\": f must be a distribution (an AbstractVelocityDistribution or an " *
        "AbstractLabDistribution); got $(typeof(f))"))
    throw(ArgumentError(
        "species \"$name\": grid must be a NamedTuple of GaussianGrid's temperature-form keywords " *
        "(T_ref_eV or T_ref_keV, u0_kms or u0_ms, nodes, spacing, width, symmetric); got $(typeof(grid))"))
end

"""
    _species_u_T(dist, name, mass, kwargs, consumed) -> (u, T)

A builder's flow and temperature (Phase G.1 spec D9). For a shape, from the unit keywords exactly as
before Phase G.1 (`T` required, `u` defaulting to zero, then every keyword outside `consumed`
refused). For a lab-frame distribution, a `T_…` or `u_…` keyword is refused, naming the species and
saying that the two are computed from its distribution; every other keyword outside `consumed`
(less the four temperature and flow keywords) is refused; and `(u, T)` are the moments,
`_distribution_u_T`. `consumed` ends in `:grid` since Phase G.2 (spec D9, the G.1 summary's A6), so
both refusals list it.
"""
function _species_u_T(::AbstractVelocityDistribution, name, mass, kwargs, consumed)
    T = _si_value(:T, kwargs)
    u = _si_value(:u, kwargs, 0.0)
    _reject_extra_kwargs(kwargs, consumed)
    return (u, T)
end

function _species_u_T(dist::AbstractLabDistribution, name, mass, kwargs, consumed)
    given = Tuple(k for k in _SPECIES_U_T_KEYWORDS if haskey(kwargs, k))
    isempty(given) || throw(ArgumentError(
        "species \"$name\" carries a lab-frame distribution ($(nameof(typeof(dist)))): its T and u are " *
        "computed from the distribution and cannot be given; got $given"))
    _reject_extra_kwargs(kwargs, Tuple(k for k in consumed if !(k in _SPECIES_U_T_KEYWORDS)))
    return _distribution_u_T(dist, mass, nothing, nothing)
end

"""
    _divide_by_charge(x, charge)

`x / charge`, the one division by a species' charge number in the package (`_resolve_ion`,
`_density`: an ion's density from its charge fraction). For two duals of one tag whose `charge`
carries no partial derivative it divides by the charge's value instead, so that the partials are
`p / Z`, bitwise what they were while `charge` was an `Int`; ForwardDiff's dual-by-dual division
forms them as `p * inv(Z) + 0`, which differs in the last place whenever `Z` is not a power of two,
and would move every Jacobian of a plasma with carbon by round-off (Phase 4.J spec D5, the user's
decision of 2026-10-10). A charge with partials (a fitted charge) takes the dual division, which is
the derivative. The value is `x / charge` either way, and both branches return `typeof(x)`.
"""
_divide_by_charge(x, charge) = x / charge
_divide_by_charge(x::ForwardDiff.Dual{T}, charge::ForwardDiff.Dual{T}) where {T} =
    iszero(ForwardDiff.partials(charge)) ? x / ForwardDiff.value(charge) : x / charge

"""
    _validate_species_name(name::AbstractString)

Throws an `ArgumentError` unless `name` is non-empty and matches `r"^[A-Za-z0-9_-]+\$"`, TOML's
bare-key character set (spec D23): a species name is also a parameter-path and deck-table
segment, so it must be usable unquoted in both. `"names"` and `"species"` are reserved (they
are `SpeciesSet`'s field names).
"""
function _validate_species_name(name::AbstractString)
    occursin(r"^[A-Za-z0-9_-]+$", name) || throw(ArgumentError(
        "invalid species name \"$name\"; names must be non-empty and match [A-Za-z0-9_-]+ " *
        "(TOML's bare-key set) so they are valid as parameter-path and deck-table segments"))
    name in ("names", "species") && throw(ArgumentError(
        "invalid species name \"$name\"; \"names\" and \"species\" are reserved, being the " *
        "field names of SpeciesSet (conventions.md 11.17)"))
    return nothing
end

"""
    electrons(; name = "e", T_eV | T_keV, u_kms | u_ms = 0, f = Maxwellian(), fraction = 1) -> SpeciesSpec
    electrons(; name = "e", f::AbstractLabDistribution, fraction = 1) -> SpeciesSpec
    electrons(; name = "e", grid = (T_ref_eV = …, …), fraction = 1) -> SpeciesSpec

Build the input form of an electron species. `mass` is `ELECTRON_MASS` and `charge` is `-1`;
neither is a keyword. With a shape `f` (the default `Maxwellian()`), `T` is required, in eV or keV,
and `u` defaults to zero, in km/s or m/s. The element type `FT` is promoted over the scalar fields
**and** over `values(parameters(f))` (`_shape_eltype`), so a shape with dual parameters makes the
species (and hence the plasma) dual with zero partials on the scalars
([OQ-9](../../dev/open_questions.md#oq-9) trap 6, decided 2026-09-09; Phase 2 spec Step 2.0 item 2).

**A lab-frame distribution** (Phase G.1, [OQ-68](../../dev/open_questions.md#oq-68) item 5): `f` a
`GaussianGrid`, or `grid`, a `NamedTuple` of `GaussianGrid`'s temperature-form keywords (`T_ref_eV`
or `T_ref_keV`, `u0_kms` or `u0_ms`, `nodes`, `spacing`, `width`, `symmetric`) from which the builder
makes the grid with the electron mass. **`T` and `u` are then computed, never given**: the species'
`u` is `mean_velocity(f)` and its `T` is `ELECTRON_MASS * velocity_variance(f)`, and a `T_…` or `u_…`
keyword is refused, naming the species (`conventions.md` 1.12). `f` and `grid` together are refused.

`conventions.md` 1.2, 1.12, Section 4; spec D11, D23; Phase 2 spec Step 2.0; Phase G.1 spec D9.
"""
function electrons(; name = "e", f = nothing, grid = nothing, fraction = 1, kwargs...)
    _validate_species_name(name)
    mass = ELECTRON_MASS
    dist = _species_distribution(name, f, grid, mass)
    u, T = _species_u_T(dist, name, mass, kwargs, (:name, :T_eV, :T_keV, :u_kms, :u_ms, :f, :fraction, :grid))
    _primal_value(T) > 0 || throw(ArgumentError("temperature must be positive; got $T J for species \"$name\""))

    charge = -1
    FT = float(promote_type(typeof(mass), typeof(u), typeof(T), typeof(fraction), _shape_eltype(dist)))
    return SpeciesSpec{FT,typeof(dist)}(String(name), FT(mass), FT(charge), FT(u), FT(T), dist, FT(fraction))
end

"""
    ion(; name = "i", mass_amu | mass_kg, charge, T_eV | T_keV, u_kms | u_ms = 0, f = Maxwellian(), fraction = 1) -> SpeciesSpec
    ion(; name = "i", mass_amu | mass_kg, charge, f::AbstractLabDistribution, fraction = 1) -> SpeciesSpec
    ion(; name = "i", mass_amu | mass_kg, charge, grid = (T_ref_eV = …, …), fraction = 1) -> SpeciesSpec

Build the input form of an ion species. `mass`, `charge` and, with a shape, `T` are required
(`charge` has no default: a charge number must always be stated); `u` defaults to zero. `FT` is
promoted over the scalar fields, `charge` among them, and over `values(parameters(f))`, as
`electrons`' docstring explains, so a dual charge makes the species dual. **A lab-frame
distribution**, `f` or `grid` (the grid made with this ion's own mass), makes `u` and `T` computed
and a `T_…` or `u_…` keyword refused, exactly as for `electrons` (Phase G.1 spec D9).

**`charge` is any finite real ≥ 1** (Phase 4.J, [OQ-60](../../dev/open_questions.md#oq-60); an
integer until then): an integer for one charge state, and, for a species that stands for a mixture
of charge states of one element at one temperature and flow, the mixture's ⟨Z²⟩/⟨Z⟩ = Σᵢ Zᵢ fᵢ / Σᵢ fᵢ,
which reproduces the mixture exactly wherever the charge enters (`conventions.md` 1.7, 3.6). The floor
is 1, inclusive, since Zᵢ² ≥ Zᵢ for integer states makes ⟨Z²⟩/⟨Z⟩ ≥ 1 for any mixture; there is no
ceiling. A charge below 1 or not finite is refused, comparing a dual charge's primal value. Negative
ions are out of scope.

`conventions.md` 1.2, 1.7, 1.12, 3.6, Section 4; spec D11, D23; Phase 2 spec Step 2.0; Phase 4.J spec D7;
Phase G.1 spec D9.
"""
function ion(; name = "i", charge::Real, f = nothing, grid = nothing, fraction = 1, kwargs...)
    _validate_species_name(name)
    mass = _si_value(:mass, kwargs)
    dist = _species_distribution(name, f, grid, mass)
    u, T = _species_u_T(dist, name, mass, kwargs,
                        (:name, :mass_amu, :mass_kg, :charge, :T_eV, :T_keV, :u_kms, :u_ms, :f, :fraction, :grid))
    _primal_value(mass) > 0 || throw(ArgumentError("mass must be positive; got $mass kg for species \"$name\""))
    _primal_value(T) > 0 || throw(ArgumentError("temperature must be positive; got $T J for species \"$name\""))
    (isfinite(charge) && _primal_value(charge) >= 1) || throw(ArgumentError("ion \"$name\" has charge $charge; an ion's charge number must be a finite real ≥ 1: for a mixture of charge states it is ⟨Z²⟩/⟨Z⟩, never below 1"))

    FT = float(promote_type(typeof(mass), typeof(u), typeof(T), typeof(fraction), typeof(charge), _shape_eltype(dist)))
    return SpeciesSpec{FT,typeof(dist)}(String(name), FT(mass), FT(charge), FT(u), FT(T), dist, FT(fraction))
end

"""
    SpeciesSet{N, S<:Tuple}

A named, indexable, iterable container of `N` species (electrons or ions), used by `Plasma`.
The species tuple is concrete, so a set with mixed shape types is still one concrete type.
Indexable by position (`set[1]`) or by name (`set["D"]`, `set.D`); name lookup is for
interactive use and may be type-unstable — the kernel iterates the tuple.

`conventions.md` Section 4; spec D12.
"""
struct SpeciesSet{N, S<:Tuple}
    names::NTuple{N, String}
    species::S
end

Base.length(s::SpeciesSet) = length(s.species)
Base.isempty(s::SpeciesSet) = length(s) == 0
Base.keys(s::SpeciesSet) = s.names
Base.getindex(s::SpeciesSet, i::Integer) = s.species[i]

function Base.getindex(s::SpeciesSet, nm::AbstractString)
    i = findfirst(==(nm), s.names)
    i === nothing && throw(KeyError(nm))
    return s.species[i]
end

Base.iterate(s::SpeciesSet, st...) = iterate(s.species, st...)
Base.eltype(::Type{<:SpeciesSet{N,S}}) where {N,S} = eltype(S)
Base.firstindex(s::SpeciesSet) = 1
Base.lastindex(s::SpeciesSet) = length(s)

function Base.getproperty(s::SpeciesSet, p::Symbol)
    p === :names && return getfield(s, :names)
    p === :species && return getfield(s, :species)
    return s[String(p)]
end

Base.propertynames(s::SpeciesSet) = (:names, :species, Symbol.(s.names)...)

# --- show methods (spec D22) ---

Base.show(io::IO, sp::SpeciesSpec) = print(io, "SpeciesSpec(\"$(sp.name)\")")
Base.show(io::IO, sp::Species) = print(io, "Species(\"$(sp.name)\")")
Base.show(io::IO, s::SpeciesSet) = print(io, "SpeciesSet(" * join(("\"$n\"" for n in s.names), ", ") * ")")

# The note `show` prints after a computed T and u (Phase G.1 spec D11): nothing for a shape.
_u_T_note(::AbstractVelocityDistribution) = ""
_u_T_note(::AbstractLabDistribution) = " (computed from f)"

function Base.show(io::IO, ::MIME"text/plain", sp::Species)
    println(io, "Species \"$(sp.name)\"")
    @printf(io, "  mass     = %f amu\n", mass_from_si(sp.mass, :mass_amu))
    println(io, "  charge   = $(sp.charge)")
    @printf(io, "  n        = %e cm^-3\n", n_from_si(sp.n, :n_cm3))
    @printf(io, "  T        = %e eV%s\n", T_from_si(sp.T, :T_eV), _u_T_note(sp.f))
    @printf(io, "  u        = %e km/s%s\n", u_from_si(sp.u, :u_kms), _u_T_note(sp.f))
    println(io, "  fraction = $(sp.fraction)")
    println(io, "  f        = $(sp.f)")
end
