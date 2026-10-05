# CODATA 2022 constants (SI); the accepted boundary-unit table and its resolver;
# `_basefloat`, the primal float type behind a possibly-dual `Real`.

import ForwardDiff

"""
    ELEMENTARY_CHARGE

The elementary charge, in C. CODATA 2022 (exact by SI definition).
"""
const ELEMENTARY_CHARGE = 1.602176634e-19

"""
    SPEED_OF_LIGHT

The speed of light in vacuum, in m/s. CODATA 2022 (exact by SI definition).
"""
const SPEED_OF_LIGHT = 2.99792458e8

"""
    ELECTRON_MASS

The electron mass, in kg. CODATA 2022.
"""
const ELECTRON_MASS = 9.1093837139e-31

"""
    VACUUM_PERMITTIVITY

The vacuum electric permittivity ε₀, in F/m. CODATA 2022.
"""
const VACUUM_PERMITTIVITY = 8.8541878188e-12

"""
    ATOMIC_MASS_UNIT

The unified atomic mass unit, in kg. CODATA 2022.
"""
const ATOMIC_MASS_UNIT = 1.66053906892e-27

"""
    EV

One electronvolt, in J. Numerically equal to `ELEMENTARY_CHARGE`: that equality is the SI
definition of the electronvolt, not a coincidence, and the two names are kept distinct because
they play different roles (a charge and an energy conversion factor).
"""
const EV = 1.602176634e-19

"""
    ACCEPTED_UNITS

The accepted boundary-unit keywords, one row per physical quantity, mapping each unit keyword
to its multiplicative conversion factor to SI. The first keyword in each row is canonical: it is
the one `show` methods and result files use.

`conventions.md` 1.2 ([OQ-5](../dev/open_questions.md#oq-5)); the `n_e` row is a Phase 0
addition (spec D4) carrying the same factors as `n`, needed because `Plasma`'s density field is
named `n_e` (`conventions.md` 11.1) while OQ-5's original table only names `n`. The
`lambda_probe` and `polarization` rows are a Phase 1 addition (Step 1.3, spec D4), carrying the
same factors as `lambda` and `theta` respectively, needed because `ProbeBeam`'s (the single
combined geometry type of Phase 1, split into `ProbeBeam`/`Collection` at Step 2.3a, spec D40)
constructor keywords are `lambda_probe_nm`/`lambda_probe_m` and
`polarization_deg`/`polarization_rad`, not `lambda_nm` or `theta_deg`. The `fwhm_lambda`,
`fwhm_t` and `half_angle` rows are a Phase 2 addition (Step 2.2, spec D3): their factors
duplicate `lambda`, `t` and `theta` respectively, and the rows exist only so that `_si_value`
accepts the keyword names `fwhm_nm`/`fwhm_ns`/`half_angle_deg`/`half_angle_rad` and
`_reject_extra_kwargs` catches a typo in one of them loudly; no `<quantity>_to_si`/`_from_si`
pair is added for these three, since the existing `lambda_to_si` and `theta_to_si` already do
the arithmetic. Unlike `lambda_probe` and `n_e`, whose keyword names carry their row's own name,
`fwhm_nm` and `fwhm_ns` do not carry `fwhm_lambda`/`fwhm_t`'s row name, because they are what
instrument papers quote (the user, 2026-09-09). A row for the Phase 2 probe-wavelength-shift
effect's own factor, added at Step 2.2, was **retired at Step 4.A.1a** (2026-09-22) and replaced,
in its place in the tuple, by `lambda_offset` (`lambda_offset_nm`/`lambda_offset_m`,
`WavelengthCalibration`'s own factor; the user, 2026-09-21,
[OQ-24](../dev/open_questions.md#oq-24)): the two effects are nearly degenerate, and the
probe-wavelength-shift effect is removed rather than kept alongside the new one. The `t_shift` row
is a Step 4.A.2 addition (2026-09-23, spec D13): `average_regions`' own channel timing offset,
in the style of `fwhm_lambda`/`fwhm_t` -- its factors duplicate `t`'s, and the row exists only so
that `_si_value` accepts `t_shift_ns`/`t_shift_s` and `_reject_extra_kwargs` catches a typo in
one of them. The `offsets` row is a Step 4.G.1 addition (2026-10-03, Phase 4.G spec D4):
`TabulatedKSmearing`'s list of angles relative to the collection's θ, in the style of `half_angle`
-- its factors duplicate `theta`'s, and the row exists so that `_si_axis` accepts
`offsets_deg`/`offsets_rad` for a vector, refusing neither or both.
"""
const ACCEPTED_UNITS = (
    T      = (T_eV      = EV,               T_keV     = 1000 * EV),
    n      = (n_cm3     = 1e6,              n_m3      = 1.0),
    n_e    = (n_e_cm3   = 1e6,              n_e_m3    = 1.0),
    u      = (u_kms     = 1e3,              u_ms      = 1.0),
    mass   = (mass_amu  = ATOMIC_MASS_UNIT, mass_kg   = 1.0),
    lambda = (lambda_nm = 1e-9,             lambda_m  = 1.0),
    t      = (t_ns      = 1e-9,             t_s       = 1.0),
    theta  = (theta_deg = π / 180,          theta_rad = 1.0),
    lambda_probe = (lambda_probe_nm = 1e-9,    lambda_probe_m   = 1.0),
    polarization = (polarization_deg = π / 180, polarization_rad = 1.0),
    fwhm_lambda  = (fwhm_nm = 1e-9,          fwhm_m   = 1.0),
    fwhm_t       = (fwhm_ns = 1e-9,          fwhm_s   = 1.0),
    half_angle   = (half_angle_deg = π/180,  half_angle_rad = 1.0),
    lambda_offset = (lambda_offset_nm = 1e-9,  lambda_offset_m = 1.0),
    t_shift        = (t_shift_ns = 1e-9,         t_shift_s = 1.0),
    intensity      = (intensity_wcm2 = 1e4,      intensity_wm2 = 1.0),
    diameter       = (diameter_um = 1e-6,        diameter_m = 1.0),
    dispersion     = (dispersion_nm_px = 1e-9,   dispersion_m_px = 1.0),
    lambda_center  = (lambda_center_nm = 1e-9,   lambda_center_m = 1.0),
    offsets        = (offsets_deg = π / 180,     offsets_rad = 1.0),
)

"""
    A0_SQ_COEFFICIENT

The coefficient of the probe's normalized vector potential squared,
`A0_SQ_COEFFICIENT = e²/(2π² ε₀ m_e² c⁵)`, so that `a0_sq = A0_SQ_COEFFICIENT * lambda^2 *
intensity` is one multiplication (`lambda` in m, `intensity` in W/m², linear polarization). The
package's first derived constant (`conventions.md` 1.3), checked against the standard
`a₀ = 0.85 √(I₁₈ λ_µm²)`: `0.731` against `0.7225` at `λ = 1 µm`, `I = 10¹⁸ W/cm²`, the
difference being the rounding of the quoted `0.85` (spec D40, GS5).

`docs/derivations/gain.md`; `conventions.md` 1.3.
"""
const A0_SQ_COEFFICIENT = ELEMENTARY_CHARGE^2 /
    (2 * π^2 * VACUUM_PERMITTIVITY * ELECTRON_MASS^2 * SPEED_OF_LIGHT^5)

"""
    T_to_si(x, key::Symbol = :T_eV) -> Float64

Convert a temperature `x`, given in the unit named by `key` (`:T_eV` or `:T_keV`), to joules.

`conventions.md` 1.2, 11.4.
"""
T_to_si(x, key::Symbol = :T_eV) = x * _unit_factor(:T, key, :T_to_si)

"""
    T_from_si(x, key::Symbol = :T_eV) -> Float64

Convert a temperature `x`, in joules, to the unit named by `key` (`:T_eV` or `:T_keV`).

`conventions.md` 1.2, 11.4.
"""
T_from_si(x, key::Symbol = :T_eV) = x / _unit_factor(:T, key, :T_from_si)

"""
    n_to_si(x, key::Symbol = :n_cm3) -> Float64

Convert a number density `x`, given in the unit named by `key` (`:n_cm3` or `:n_m3`), to m⁻³.

`conventions.md` 1.2, 11.4.
"""
n_to_si(x, key::Symbol = :n_cm3) = x * _unit_factor(:n, key, :n_to_si)

"""
    n_from_si(x, key::Symbol = :n_cm3) -> Float64

Convert a number density `x`, in m⁻³, to the unit named by `key` (`:n_cm3` or `:n_m3`).

`conventions.md` 1.2, 11.4.
"""
n_from_si(x, key::Symbol = :n_cm3) = x / _unit_factor(:n, key, :n_from_si)

"""
    u_to_si(x, key::Symbol = :u_kms) -> Float64

Convert a velocity `x`, given in the unit named by `key` (`:u_kms` or `:u_ms`), to m/s.

`conventions.md` 1.2, 11.4.
"""
u_to_si(x, key::Symbol = :u_kms) = x * _unit_factor(:u, key, :u_to_si)

"""
    u_from_si(x, key::Symbol = :u_kms) -> Float64

Convert a velocity `x`, in m/s, to the unit named by `key` (`:u_kms` or `:u_ms`).

`conventions.md` 1.2, 11.4.
"""
u_from_si(x, key::Symbol = :u_kms) = x / _unit_factor(:u, key, :u_from_si)

"""
    mass_to_si(x, key::Symbol = :mass_amu) -> Float64

Convert a mass `x`, given in the unit named by `key` (`:mass_amu` or `:mass_kg`), to kg.

`conventions.md` 1.2, 11.4.
"""
mass_to_si(x, key::Symbol = :mass_amu) = x * _unit_factor(:mass, key, :mass_to_si)

"""
    mass_from_si(x, key::Symbol = :mass_amu) -> Float64

Convert a mass `x`, in kg, to the unit named by `key` (`:mass_amu` or `:mass_kg`).

`conventions.md` 1.2, 11.4.
"""
mass_from_si(x, key::Symbol = :mass_amu) = x / _unit_factor(:mass, key, :mass_from_si)

"""
    lambda_to_si(x, key::Symbol = :lambda_nm) -> Float64

Convert a wavelength `x`, given in the unit named by `key` (`:lambda_nm` or `:lambda_m`), to m.

`conventions.md` 1.2, 11.4.
"""
lambda_to_si(x, key::Symbol = :lambda_nm) = x * _unit_factor(:lambda, key, :lambda_to_si)

"""
    lambda_from_si(x, key::Symbol = :lambda_nm) -> Float64

Convert a wavelength `x`, in m, to the unit named by `key` (`:lambda_nm` or `:lambda_m`).

`conventions.md` 1.2, 11.4.
"""
lambda_from_si(x, key::Symbol = :lambda_nm) = x / _unit_factor(:lambda, key, :lambda_from_si)

"""
    t_to_si(x, key::Symbol = :t_ns) -> Float64

Convert a time `x`, given in the unit named by `key` (`:t_ns` or `:t_s`), to s.

`conventions.md` 1.2, 11.4.
"""
t_to_si(x, key::Symbol = :t_ns) = x * _unit_factor(:t, key, :t_to_si)

"""
    t_from_si(x, key::Symbol = :t_ns) -> Float64

Convert a time `x`, in s, to the unit named by `key` (`:t_ns` or `:t_s`).

`conventions.md` 1.2, 11.4.
"""
t_from_si(x, key::Symbol = :t_ns) = x / _unit_factor(:t, key, :t_from_si)

"""
    theta_to_si(x, key::Symbol = :theta_deg) -> Float64

Convert an angle `x`, given in the unit named by `key` (`:theta_deg` or `:theta_rad`), to rad.

`conventions.md` 1.2, 11.4.
"""
theta_to_si(x, key::Symbol = :theta_deg) = x * _unit_factor(:theta, key, :theta_to_si)

"""
    theta_from_si(x, key::Symbol = :theta_deg) -> Float64

Convert an angle `x`, in rad, to the unit named by `key` (`:theta_deg` or `:theta_rad`).

`conventions.md` 1.2, 11.4.
"""
theta_from_si(x, key::Symbol = :theta_deg) = x / _unit_factor(:theta, key, :theta_from_si)

"""
    _unit_factor(quantity::Symbol, key::Symbol, caller::Symbol) -> Float64

Look up the SI conversion factor for `key` in the `ACCEPTED_UNITS` row named `quantity`,
throwing an `ArgumentError` naming `caller` (the conversion function the user actually called,
e.g. `:n_from_si`) if `key` is not an accepted unit for `quantity`.
"""
function _unit_factor(quantity::Symbol, key::Symbol, caller::Symbol)
    row = getfield(ACCEPTED_UNITS, quantity)
    haskey(row, key) || throw(ArgumentError(
        "$(caller): :$key is not an accepted unit key for $quantity; " *
        "expected one of $(keys(row))"))
    return getfield(row, key)
end

"""
    _si_value(quantity::Symbol, kwargs) -> value in SI
    _si_value(quantity::Symbol, kwargs, default) -> value in SI

Resolve the value of `quantity` from a constructor's captured keyword arguments `kwargs`,
using the accepted unit keywords of `ACCEPTED_UNITS[quantity]`. Exactly one of that row's
keywords must appear in `kwargs`, unless `default` (already in SI) is given, in which case
none appearing returns `default`. Throws an `ArgumentError` if none appear and no default is
given, or if more than one appears.

`conventions.md` 1.2 (spec D4).
"""
function _si_value(quantity::Symbol, kwargs)
    row = getfield(ACCEPTED_UNITS, quantity)
    present = filter(k -> haskey(kwargs, k), keys(row))
    if length(present) == 1
        key = present[1]
        return kwargs[key] * getfield(row, key)
    elseif length(present) == 0
        throw(ArgumentError(
            "no unit keyword given for $quantity; expected exactly one of $(keys(row))"))
    else
        throw(ArgumentError(
            "more than one unit keyword given for $quantity: $present; expected exactly one"))
    end
end

function _si_value(quantity::Symbol, kwargs, default)
    row = getfield(ACCEPTED_UNITS, quantity)
    present = filter(k -> haskey(kwargs, k), keys(row))
    if length(present) == 1
        key = present[1]
        return kwargs[key] * getfield(row, key)
    elseif length(present) == 0
        return default
    else
        throw(ArgumentError(
            "more than one unit keyword given for $quantity: $present; expected exactly one"))
    end
end

"""
    _reject_extra_kwargs(kwargs, consumed::Tuple{Vararg{Symbol}})

Throw an `ArgumentError` naming every key of `kwargs` not in `consumed`. Without this check, a
mistyped unit keyword (`T_ev` for `T_eV`) would be silently ignored by `_si_value` rather than
reported.
"""
function _reject_extra_kwargs(kwargs, consumed::Tuple{Vararg{Symbol}})
    extra = Tuple(k for k in keys(kwargs) if !(k in consumed))
    isempty(extra) || throw(ArgumentError(
        "unrecognised keyword(s) $extra; accepted keywords are $consumed"))
    return nothing
end

"""
    _basefloat(x) -> Type{<:AbstractFloat}

The primal (non-dual) float type behind `x`, used to construct quadrature integration limits
(`conventions.md` 5.12) and float literals such as `sqrt(FT(π))` so that a dual argument does
not drag a dead partial through a constant. Returns `Float64` for an `Integer` and unwraps a
`ForwardDiff.Dual`, including a nested one, down to its underlying float type; any other type
raises an `ArgumentError` naming it.
"""
_basefloat(::Type{FT}) where {FT<:AbstractFloat} = FT
_basefloat(::Type{I}) where {I<:Integer} = Float64
_basefloat(::Type{D}) where {D<:ForwardDiff.Dual} = _basefloat(ForwardDiff.valtype(D))
_basefloat(x) = _basefloat(typeof(x))
_basefloat(::Type{T}) where {T} = throw(ArgumentError(
    "_basefloat: no primal float type for $T; for a complex x call _basefloat(real(x)) " *
    "(conventions.md 5.12)"))

"""
    _primal_value(x) -> Real

The primal (non-dual) *value* behind `x`, the value analogue of `_basefloat` (which returns the
primal float *type*). Recurses through nested `ForwardDiff.Dual`s, so that a
`Dual{T1,Dual{T2,...}}` unwraps to a plain float. Used to build a quadrature breakpoint from a
dual argument without making the breakpoint itself dual (`conventions.md` 5.12's clause
permitting a breakpoint at a dual's primal value).

`conventions.md` 5.12; spec D8.
"""
_primal_value(x::Real) = x
_primal_value(d::ForwardDiff.Dual) = _primal_value(ForwardDiff.value(d))
