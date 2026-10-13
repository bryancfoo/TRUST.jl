# Plasma: n_e plus a named electron SpeciesSet and a named ion SpeciesSet, quasineutral by
# construction from the fractions. conventions.md 1.7 (OQ-4), Section 4; spec D19, D20.

using Printf: @printf

"""
    Plasma{FT<:Real, E<:SpeciesSet, I<:SpeciesSet}

A plasma at one instant: the total electron density `n_e` (the ground truth; never recomputed
from the ions) plus a named `SpeciesSet` of electron species and a named `SpeciesSet` of ion
species, quasineutral by construction.

Built with `Plasma(; n_e_cm3 | n_e_m3, electrons, ions = ())`, `electrons` and `ions` given as
vectors or tuples of `SpeciesSpec`. Electron species densities are number fractions of `n_e`;
ion species densities are charge fractions of `n_e` divided by the ion's charge number
(`conventions.md` 1.7, [OQ-4](../../dev/open_questions.md#oq-4)). `ions` may be empty;
`electrons` may not. Every species is converted to one shared element type `FT`.

`conventions.md` Section 4; spec D19.
"""
struct Plasma{FT<:Real, E<:SpeciesSet, I<:SpeciesSet}
    n_e::FT
    electrons::E
    ions::I
end

_spec_eltype(::SpeciesSpec{FT}) where {FT} = FT

function _resolve_electron(sp::SpeciesSpec, n_e::FT) where {FT}
    n = FT(sp.fraction) * n_e
    F = typeof(sp.f)
    return Species{FT,F}(sp.name, FT(sp.mass), FT(sp.charge), FT(sp.u), FT(sp.T), sp.f, FT(sp.fraction), n)
end

# Phase 4.J spec D5: the division by the charge goes through _divide_by_charge, which keeps the dual
# arithmetic of a charge without partials bitwise what it was while the charge was an Int.
function _resolve_ion(sp::SpeciesSpec, n_e::FT) where {FT}
    n = _divide_by_charge(FT(sp.fraction) * n_e, sp.charge)
    F = typeof(sp.f)
    return Species{FT,F}(sp.name, FT(sp.mass), FT(sp.charge), FT(sp.u), FT(sp.T), sp.f, FT(sp.fraction), n)
end

function _validate_species_set(specs::Tuple, kind::String)
    isempty(specs) && return nothing

    names = map(sp -> sp.name, specs)
    for i in eachindex(names), j in (i + 1):length(names)
        if names[i] == names[j]
            throw(ArgumentError(
                "duplicate species name \"$(names[i])\" in $(kind)s; names must be unique within a set"))
        end
    end

    for sp in specs
        (0 <= _primal_value(sp.fraction) <= 1) || throw(ArgumentError(
            "fraction $(sp.fraction) for species \"$(sp.name)\" is outside [0, 1]"))
    end

    total = sum(sp.fraction for sp in specs)
    abs(total - 1) <= 1e-8 || throw(ArgumentError(
        "$(kind) fractions sum to $(total), not 1, within 1e-8"))

    return nothing
end

function Plasma(; electrons, ions = (), kwargs...)
    n_e_raw = _si_value(:n_e, kwargs)
    _reject_extra_kwargs(kwargs, (:n_e_cm3, :n_e_m3))
    _primal_value(n_e_raw) > 0 || throw(ArgumentError("n_e must be positive; got $n_e_raw m^-3"))

    electron_specs = Tuple(electrons)
    ion_specs = Tuple(ions)
    isempty(electron_specs) && throw(ArgumentError("a Plasma needs at least one electron species"))

    _validate_species_set(electron_specs, "electron")
    _validate_species_set(ion_specs, "ion")

    FT = float(promote_type(typeof(n_e_raw), map(_spec_eltype, electron_specs)..., map(_spec_eltype, ion_specs)...))
    n_e = FT(n_e_raw)

    electron_species = map(sp -> _resolve_electron(sp, n_e), electron_specs)
    ion_species = map(sp -> _resolve_ion(sp, n_e), ion_specs)

    electron_set = SpeciesSet(map(sp -> sp.name, electron_specs), electron_species)
    ion_set = SpeciesSet(map(sp -> sp.name, ion_specs), ion_species)

    return Plasma{FT, typeof(electron_set), typeof(ion_set)}(n_e, electron_set, ion_set)
end

"""
    neutrality_residual(p::Plasma) -> FT

The signed quasineutrality residual, `(Σ_i charge_i·n_i − Σ_j n_ej) / n_e`, zero to round-off
for any `Plasma` built from fractions (the closure is exact by construction). Signed, not an
absolute value, so a future broken closure is diagnosable from its sign.

`conventions.md` Section 4; spec D20.
"""
function neutrality_residual(p::Plasma)
    total_ion_charge = sum(sp.charge * sp.n for sp in p.ions; init = zero(p.n_e))
    total_electron_n = sum(sp.n for sp in p.electrons; init = zero(p.n_e))
    return (total_ion_charge - total_electron_n) / p.n_e
end

"""
    mean_charge_state(p::Plasma) -> FT

The number-weighted mean ion charge state, `Z̄ = Σ_i n_i Z_i / Σ_i n_i` (equivalently
`n_e / Σ_i n_i`, by the neutrality closure). Throws for a plasma with no ion species.

Each `Z_i` is a species' own `charge`, a real number since Phase 4.J. A species that stands for a
mixture of charge states carries the equivalent single-state ion (charge ⟨Z²⟩/⟨Z⟩, density
`fraction · n_e / charge`), so for a plasma holding one this averages the equivalent ions, not the
mixture's own states; the mixture's number-average charge is not a package output
([OQ-60](../../dev/open_questions.md#oq-60)).

`conventions.md` Section 4; spec D20.
"""
function mean_charge_state(p::Plasma)
    isempty(p.ions) && throw(ArgumentError("mean_charge_state is undefined for a plasma with no ion species"))
    return sum(sp.n * sp.charge for sp in p.ions) / sum(sp.n for sp in p.ions)
end

"""
    mean_ion_mass(p::Plasma) -> FT

The number-weighted mean ion mass, `m̄ = Σ_i n_i m_i / Σ_i n_i`. Throws for a plasma with no
ion species.

`conventions.md` Section 4; spec D20.
"""
function mean_ion_mass(p::Plasma)
    isempty(p.ions) && throw(ArgumentError("mean_ion_mass is undefined for a plasma with no ion species"))
    return sum(sp.n * sp.mass for sp in p.ions) / sum(sp.n for sp in p.ions)
end

"""
    zeff(p::Plasma) -> FT

The "other" mean charge, `Z_eff = ⟨Z²⟩/⟨Z⟩ = Σ_i n_i Z_i² / n_e`, the sum of the ion weights
that appear in `spectral_density` (`conventions.md` 1.7, Section 4). Unlike
`mean_charge_state` and `mean_ion_mass`, **`zeff` is 0, not an error, for a plasma with no ion
species** — it is a plain sum with `init = zero(p.n_e)`; that contrast is by design.

A real `charge` (Phase 4.J) changes nothing here: each species contributes `charge · fraction`, so a
species at ⟨Z²⟩/⟨Z⟩ contributes exactly what the mixture it stands for would
([OQ-60](../../dev/open_questions.md#oq-60)).

`conventions.md` Section 4; spec D10.
"""
function zeff(p::Plasma)
    return sum(sp -> sp.charge^2 * sp.n, p.ions; init = zero(p.n_e)) / p.n_e
end

# --- show methods (spec D22) ---

Base.show(io::IO, p::Plasma) = print(io, "Plasma($(length(p.electrons)) electron species, $(length(p.ions)) ion species)")

function Base.show(io::IO, ::MIME"text/plain", p::Plasma)
    println(io, "Plasma")
    @printf(io, "  n_e = %e cm^-3\n", n_from_si(p.n_e, :n_cm3))
    println(io, "  electrons ($(length(p.electrons))): " * join(("\"$n\"" for n in p.electrons.names), ", "))
    println(io, "  ions      ($(length(p.ions))): " * join(("\"$n\"" for n in p.ions.names), ", "))
end
