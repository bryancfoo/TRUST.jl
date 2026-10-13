# SpeciesSpec (input form), Species (resolved form, produced only by Plasma), SpeciesSet
# (named, indexable container), the electrons/ion builders, spec, show, and name validation.

using Printf: @printf

"""
    SpeciesSpec{FT<:Real, F<:AbstractVelocityDistribution}

The input form of one particle population: its name, mass, charge number, flow, temperature,
shape and fraction, with **no density** (density is resolved only by `Plasma`, which turns a
`SpeciesSpec` into a `Species`). Built by `electrons` and `ion`, never directly.

**`charge` is `FT`, a real number** (Phase 4.J, [OQ-60](../../dev/open_questions.md#oq-60); an `Int`
until then): `-1` for an electron, an ion's charge number otherwise, at least 1. An integer is one
charge state; a non-integer stands for a mixture of charge states of one element at one temperature
and flow, and is the mixture's ⟨Z²⟩/⟨Z⟩, which reproduces it exactly (`conventions.md` 1.7).

`conventions.md` Section 4; spec D11; Phase 4.J spec D7.
"""
struct SpeciesSpec{FT<:Real, F<:AbstractVelocityDistribution}
    name::String
    mass::FT
    charge::FT
    u::FT
    T::FT
    f::F
    fraction::FT
end

"""
    Species{FT<:Real, F<:AbstractVelocityDistribution}

The resolved form of one particle population: every `SpeciesSpec` field plus the number
density `n` (m⁻³). Produced **only** by the `Plasma` constructor, which is what turns a
fraction of `n_e` into a density; there is no outer constructor and no lowercase builder for
`Species`, and constructing one directly is not supported (Julia cannot hide the default inner
constructor of an exported type; this is the closest faithful implementation of "not
constructible by users," spec D11).

**`charge` is `FT`** (Phase 4.J): see `SpeciesSpec`. An ion's `n` is `fraction · n_e / charge`; for a
species that stands for a mixture of charge states that is the equivalent single-state ion's density,
below the mixture's own number density by ⟨Z⟩/(⟨Z²⟩/⟨Z⟩) ([OQ-60](../../dev/open_questions.md#oq-60)).

`conventions.md` Section 4; spec D11; Phase 4.J spec D7.
"""
struct Species{FT<:Real, F<:AbstractVelocityDistribution}
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
    _shape_eltype(f::AbstractVelocityDistribution) -> Type

`promote_type` over the types of `values(parameters(f))`, or `Bool` (the promotion identity)
for a shape with no parameters (`Maxwellian`). Folded into `electrons` and `ion`'s own element
type so that a shape with dual parameters makes its species (and hence its plasma) dual with
zero partials on the scalar fields, without `_spectrum_eltype` needing any knowledge of shapes
([OQ-9](../../dev/open_questions.md#oq-9) trap 6, decided by Level 1 2026-09-09; spec Step 2.0
item 2). The shape `f` itself is stored as given and is never converted here: it already carries
whatever element type its own parameters do.
"""
function _shape_eltype(f::AbstractVelocityDistribution)
    p = values(parameters(f))
    isempty(p) && return Bool
    return promote_type(typeof.(p)...)
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

Build the input form of an electron species. `mass` is `ELECTRON_MASS` and `charge` is `-1`;
neither is a keyword. `T` is required, in eV or keV; `u` defaults to zero, in km/s or m/s.
The element type `FT` is promoted over the scalar fields **and** over `values(parameters(f))`
(`_shape_eltype`), so a shape with dual parameters makes the species (and hence the plasma)
dual with zero partials on the scalars ([OQ-9](../../dev/open_questions.md#oq-9) trap 6,
decided 2026-09-09; Phase 2 spec Step 2.0 item 2).

`conventions.md` 1.2, Section 4; spec D11, D23; Phase 2 spec Step 2.0.
"""
function electrons(; name = "e", f::F = Maxwellian(), fraction = 1, kwargs...) where {F<:AbstractVelocityDistribution}
    _validate_species_name(name)
    T = _si_value(:T, kwargs)
    u = _si_value(:u, kwargs, 0.0)
    _reject_extra_kwargs(kwargs, (:name, :T_eV, :T_keV, :u_kms, :u_ms, :f, :fraction))
    _primal_value(T) > 0 || throw(ArgumentError("temperature must be positive; got $T J for species \"$name\""))

    mass = ELECTRON_MASS
    charge = -1
    FT = float(promote_type(typeof(mass), typeof(u), typeof(T), typeof(fraction), _shape_eltype(f)))
    return SpeciesSpec{FT,F}(String(name), FT(mass), FT(charge), FT(u), FT(T), f, FT(fraction))
end

"""
    ion(; name = "i", mass_amu | mass_kg, charge, T_eV | T_keV, u_kms | u_ms = 0, f = Maxwellian(), fraction = 1) -> SpeciesSpec

Build the input form of an ion species. `mass`, `charge` and `T` are required (`charge` has no
default: a charge number must always be stated); `u` defaults to zero. `FT` is promoted over the
scalar fields, `charge` among them, and over `values(parameters(f))`, as `electrons`' docstring
explains, so a dual charge makes the species dual.

**`charge` is any finite real ≥ 1** (Phase 4.J, [OQ-60](../../dev/open_questions.md#oq-60); an
integer until then): an integer for one charge state, and, for a species that stands for a mixture
of charge states of one element at one temperature and flow, the mixture's ⟨Z²⟩/⟨Z⟩ = Σᵢ Zᵢ fᵢ / Σᵢ fᵢ,
which reproduces the mixture exactly wherever the charge enters (`conventions.md` 1.7, 3.6). The floor
is 1, inclusive, since Zᵢ² ≥ Zᵢ for integer states makes ⟨Z²⟩/⟨Z⟩ ≥ 1 for any mixture; there is no
ceiling. A charge below 1 or not finite is refused, comparing a dual charge's primal value. Negative
ions are out of scope.

`conventions.md` 1.2, 1.7, 3.6, Section 4; spec D11, D23; Phase 2 spec Step 2.0; Phase 4.J spec D7.
"""
function ion(; name = "i", charge::Real, f::F = Maxwellian(), fraction = 1, kwargs...) where {F<:AbstractVelocityDistribution}
    _validate_species_name(name)
    mass = _si_value(:mass, kwargs)
    T = _si_value(:T, kwargs)
    u = _si_value(:u, kwargs, 0.0)
    _reject_extra_kwargs(kwargs, (:name, :mass_amu, :mass_kg, :charge, :T_eV, :T_keV, :u_kms, :u_ms, :f, :fraction))
    _primal_value(mass) > 0 || throw(ArgumentError("mass must be positive; got $mass kg for species \"$name\""))
    _primal_value(T) > 0 || throw(ArgumentError("temperature must be positive; got $T J for species \"$name\""))
    (isfinite(charge) && _primal_value(charge) >= 1) || throw(ArgumentError("ion \"$name\" has charge $charge; an ion's charge number must be a finite real ≥ 1: for a mixture of charge states it is ⟨Z²⟩/⟨Z⟩, never below 1"))

    FT = float(promote_type(typeof(mass), typeof(u), typeof(T), typeof(fraction), typeof(charge), _shape_eltype(f)))
    return SpeciesSpec{FT,F}(String(name), FT(mass), FT(charge), FT(u), FT(T), f, FT(fraction))
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

function Base.show(io::IO, ::MIME"text/plain", sp::Species)
    println(io, "Species \"$(sp.name)\"")
    @printf(io, "  mass     = %f amu\n", mass_from_si(sp.mass, :mass_amu))
    println(io, "  charge   = $(sp.charge)")
    @printf(io, "  n        = %e cm^-3\n", n_from_si(sp.n, :n_cm3))
    @printf(io, "  T        = %e eV\n", T_from_si(sp.T, :T_eV))
    @printf(io, "  u        = %e km/s\n", u_from_si(sp.u, :u_kms))
    println(io, "  fraction = $(sp.fraction)")
    println(io, "  f        = $(sp.f)")
end
