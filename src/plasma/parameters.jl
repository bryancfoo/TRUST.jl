# Per-species derived quantities: thermal speed, plasma frequency, Debye length, the
# dimensionless scattering parameter alpha and phase velocity xi, and f_lab. All SI in, SI or
# dimensionless out. conventions.md 1.4, 1.6.

"""
    thermal_speed(sp) -> FT

The thermal speed `v_th = √(2T/m)`, **with the factor of two** (`conventions.md` 1.4): the one
place this formula is written in the package. `sp` is a `Species` or a `SpeciesSpec`. `sp.T` is
in J, `sp.mass` in kg; returns m/s.

`conventions.md` 1.4; `docs/derivations/shapes.md`.
"""
thermal_speed(sp::Union{Species,SpeciesSpec}) = sqrt(2 * sp.T / sp.mass)

"""
    plasma_frequency(sp::Species) -> FT

The species plasma frequency `ω_p = √(n q² / (ε₀ m))`, with `q = charge·e`, `charge` the species'
charge number, a real number since Phase 4.J (`-1` for an electron;
[OQ-60](../../dev/open_questions.md#oq-60)). `sp.n` is in m⁻³, `sp.mass` in kg; returns rad/s.

`conventions.md` 1.6.
"""
plasma_frequency(sp::Species) = sqrt(_plasma_frequency_sq(sp))

"""
    _plasma_frequency_sq(sp::Species) -> FT

`ω_p² = n q² / (ε₀ m)`, the square `plasma_frequency` takes the root of: linear in the density, so
that its partial at a density of exactly zero is finite (`conventions.md` 5.15). The lab-frame
`susceptibility` uses it, never `plasma_frequency(sp)^2` (Phase G.1 spec D12).

`conventions.md` 1.12, 5.15.
"""
_plasma_frequency_sq(sp::Species) = sp.n * (sp.charge * ELEMENTARY_CHARGE)^2 / (VACUUM_PERMITTIVITY * sp.mass)

"""
    debye_length(sp::Species) -> FT

The species Debye length `λ_D = √(ε₀ T / (n q²))`, with `q = charge·e`, `charge` a real number since
Phase 4.J. `sp.T` is in J, `sp.n` in m⁻³; returns m.

`conventions.md` 1.6.
"""
debye_length(sp::Species) = sqrt(VACUUM_PERMITTIVITY * sp.T / (sp.n * (sp.charge * ELEMENTARY_CHARGE)^2))

"""
    alpha(sp::Species, k) -> FT

The species scattering parameter `α = 1/(k λ_D)` (**not** `α²`; `conventions.md` 1.6 defines
`α² = 1/(kλ_D)²`, and `susceptibility` at Step 1.1 squares it). `k` is in m⁻¹; returns a
dimensionless value.

`conventions.md` 1.6.
"""
alpha(sp::Species, k) = 1 / (k * debye_length(sp))

"""
    xi(sp, k, omega) -> FT

The dimensionless phase velocity in the species frame, `ξ = (ω/k − u) / v_th`. `sp` is a
`Species` or a `SpeciesSpec`. `k` is in m⁻¹, `omega` in rad/s; returns a dimensionless value.

`conventions.md` 1.6.
"""
xi(sp::Union{Species,SpeciesSpec}, k, omega) = (omega / k - sp.u) / thermal_speed(sp)

"""
    f_lab(sp, v) -> FT

The dimensional, lab-frame 1-D distribution along k̂, `f_lab(v) = φ((v−u)/v_th) / v_th` — the
only place the shift and the `1/v_th` scaling are applied (`conventions.md` 1.5). `sp` is a
`Species` or a `SpeciesSpec`. `v` is in m/s; returns s/m.

`conventions.md` 1.4, 1.5; `docs/derivations/shapes.md`.
"""
f_lab(sp::Union{Species,SpeciesSpec}, v) = sp.f((v - sp.u) / thermal_speed(sp)) / thermal_speed(sp)

"""
    f_lab(sp, v) -> FT    (a species carrying a lab-frame distribution)

`f(v)` itself: a lab-frame distribution is already the dimensional distribution in lab velocity, so
there is no shift and no `1/v_th` (`conventions.md` 1.12). `v` in m/s; returns s/m.

`conventions.md` 1.12; Phase G.1 spec D12.
"""
f_lab(sp::Union{Species{<:Real,<:AbstractLabDistribution},SpeciesSpec{<:Real,<:AbstractLabDistribution}}, v) =
    sp.f(v)
