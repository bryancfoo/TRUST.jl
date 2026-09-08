# The scattered power spectrum P(lambda_s): the prefactor, the exact wavelength Jacobian, and
# the per-bin kernel. conventions.md 1.8, 3.4; docs/derivations/spectrum.md; spec D11.

"""
    _omega_probe(geometry::ScatteringGeometry) -> FT

The probe frequency `ω_i = 2πc/λ_i`, rad/s.

Sheffield Section 5.4.1; `conventions.md` 1.8; `docs/derivations/spectrum.md`; spec D11.
"""
_omega_probe(geometry::ScatteringGeometry) = 2π * SPEED_OF_LIGHT / geometry.lambda_probe

"""
    _wavelength_jacobian(lambda_s) -> FT

The exact Jacobian `|dω_s/dλ_s| = 2πc/λ_s²`, in rad/(s·m), applied once to convert the spectral
density (per unit frequency) to `spectrum`'s per-unit-wavelength axis.

`conventions.md` 3.4, 1.8; `docs/derivations/spectrum.md` Section 2; spec D11.
"""
_wavelength_jacobian(lambda_s) = 2π * SPEED_OF_LIGHT / lambda_s^2

"""
    _prefactor(omega, omega_probe, first_order::Bool) -> Real

The first-order-in-v/c relativistic factor `1 + 2ω/ω_i` (Sheffield Eq. 2.3.17) when
`first_order` is true, or `one(omega)` (Eq. 2.3.13's zeroth-order form) when false.

Sheffield Eq. 2.3.17; `conventions.md` 1.8; `docs/derivations/spectrum.md` Section 1; spec D11.
"""
_prefactor(omega, omega_probe, first_order::Bool) = first_order ? (1 + 2 * omega / omega_probe) : one(omega)

"""
    _prefactor_flag(prefactor::Symbol) -> Bool

Validates and converts the `prefactor` keyword: `:first_order` -> `true`, `:none` -> `false`,
anything else throws.

`conventions.md` 1.8; spec D11, D15.
"""
function _prefactor_flag(prefactor::Symbol)
    prefactor === :first_order && return true
    prefactor === :none && return false
    throw(ArgumentError("spectrum: prefactor must be :first_order or :none; got :$prefactor"))
end

"""
    _spectrum_eltype(plasma::Plasma{FT1}, geometry::ScatteringGeometry{FT2}, lambda) where {FT1,FT2}

The element type of `spectrum`'s output: the promotion of the plasma's, the geometry's and the
wavelength axis's element types.

Spec D11.
"""
_spectrum_eltype(::Plasma{FT1}, ::ScatteringGeometry{FT2}, lambda) where {FT1,FT2} =
    promote_type(FT1, FT2, eltype(lambda))

"""
    _spectrum_at(plasma::Plasma, geometry::ScatteringGeometry, lambda_s, first_order::Bool = true) -> Real

The scattered power at one scattered wavelength `lambda_s` (m):
`(1 ± prefactor) · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²`. The relativistic-prefactor flag is positional
with a default so that `spectrum!` can hoist the `Symbol`-to-`Bool` conversion out of its loop
while a test can still call this with one argument.

`conventions.md` 1.8; `docs/derivations/spectrum.md`; spec D11.
"""
function _spectrum_at(plasma::Plasma, geometry::ScatteringGeometry, lambda_s, first_order::Bool = true)
    k, omega = wavevector(geometry, lambda_s, plasma)
    S = spectral_density(plasma, k, omega)
    omega_probe = _omega_probe(geometry)
    return _prefactor(omega, omega_probe, first_order) * S * _wavelength_jacobian(lambda_s)
end

"""
    spectrum!(P::AbstractVector, plasma::Plasma, geometry::ScatteringGeometry, lambda::AbstractVector;
              prefactor::Symbol = :first_order) -> P

The in-place form of `spectrum`: fills `P` with the scattered power spectrum at every wavelength
in `lambda` (m, increasing). `prefactor` is `:first_order` (Sheffield Eq. 2.3.17, default) or
`:none` (Eq. 2.3.13). `G = 1`; no `r_0`. This is the per-bin kernel Phase 2 calls and the one the
allocation tests run.

Sheffield Eqs. 2.3.13, 2.3.17; `conventions.md` 1.8, 3.4; `docs/derivations/spectrum.md`;
spec D11.
"""
function spectrum!(P::AbstractVector, plasma::Plasma, geometry::ScatteringGeometry,
                    lambda::AbstractVector; prefactor::Symbol = :first_order)
    length(P) == length(lambda) || throw(ArgumentError(
        "spectrum!: P has length $(length(P)) but lambda has length $(length(lambda))"))
    first_order = _prefactor_flag(prefactor)
    @inbounds for i in eachindex(lambda)
        P[i] = _spectrum_at(plasma, geometry, lambda[i], first_order)
    end
    return P
end

"""
    spectrum(plasma::Plasma, geometry::ScatteringGeometry, lambda::AbstractVector;
             prefactor::Symbol = :first_order) -> Vector

    P(λ_s) = (1 + 2ω/ω_i) · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²,     G = 1,     units m⁻¹

The scattered power spectrum on the wavelength axis `lambda` (m, increasing). Allocates its
output with `similar(lambda, ...)` and calls `spectrum!`.

Sheffield Eqs. 2.3.13, 2.3.17; `conventions.md` 1.8, 3.4; `docs/derivations/spectrum.md`;
spec D11.
"""
function spectrum(plasma::Plasma, geometry::ScatteringGeometry, lambda::AbstractVector;
                   prefactor::Symbol = :first_order)
    P = similar(lambda, _spectrum_eltype(plasma, geometry, lambda))
    return spectrum!(P, plasma, geometry, lambda; prefactor = prefactor)
end
