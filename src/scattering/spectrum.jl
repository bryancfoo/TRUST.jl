# The scattered power spectrum P(lambda_s): the prefactor, the exact wavelength Jacobian, and
# the per-bin kernel. conventions.md 1.8, 3.4; docs/derivations/spectrum.md; spec D11, D39.

"""
    _omega_probe(probe::ProbeBeam) -> FT

The probe frequency `ω_i = 2πc/λ_i`, rad/s.

Sheffield Section 5.4.1; `conventions.md` 1.8; `docs/derivations/spectrum.md`; spec D41.
"""
_omega_probe(probe::ProbeBeam) = 2π * SPEED_OF_LIGHT / probe.lambda

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
    _spectrum_eltype(plasma::Plasma{FT1}, probe::ProbeBeam{FT2}, collection::Collection{FT3}, lambda) where {FT1,FT2,FT3}

The element type of `spectrum`'s output: the promotion of the plasma's, the probe's, the
collection's and the wavelength axis's element types.

Spec D11, D40.
"""
_spectrum_eltype(::Plasma{FT1}, ::ProbeBeam{FT2}, ::Collection{FT3}, lambda) where {FT1,FT2,FT3} =
    promote_type(FT1, FT2, FT3, eltype(lambda))

"""
    _spectrum_at(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda_s, first_order::Bool = true) -> Real

The scattered power at one scattered wavelength `lambda_s` (m):
`(1 ± prefactor) · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²`. `dielectric` is evaluated **once** here and
threaded into the four-argument `spectral_density`, so that Step 2.3b's convective gain can reuse
`(ε, χ_e)` without a second `dielectric` call (spec D39; RF2 counts the calls). The
relativistic-prefactor flag is positional with a default so that `spectrum!` can hoist the
`Symbol`-to-`Bool` conversion out of its loop while a test can still call this with one argument.

`conventions.md` 1.8; `docs/derivations/spectrum.md`; spec D11, D39.
"""
function _spectrum_at(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda_s,
                       first_order::Bool = true)
    k, omega, k_s = wavevector(probe, collection, lambda_s, plasma)
    d = dielectric(plasma, k, omega)
    S = spectral_density(plasma, k, omega, d)
    omega_probe = _omega_probe(probe)
    return _prefactor(omega, omega_probe, first_order) * S * _wavelength_jacobian(lambda_s)
end

"""
    spectrum!(P::AbstractVector, plasma::Plasma, probe::ProbeBeam, collection::Collection,
              lambda::AbstractVector; prefactor::Symbol = :first_order) -> P

The in-place form of `spectrum`: fills `P` with the scattered power spectrum at every wavelength
in `lambda` (m, increasing). `prefactor` is `:first_order` (Sheffield Eq. 2.3.17, default) or
`:none` (Eq. 2.3.13). `G = 1`; no `r_0`. This is the per-bin kernel Phase 2 calls and the one the
allocation tests run.

Sheffield Eqs. 2.3.13, 2.3.17; `conventions.md` 1.8, 3.4; `docs/derivations/spectrum.md`;
spec D11, D40.
"""
function spectrum!(P::AbstractVector, plasma::Plasma, probe::ProbeBeam, collection::Collection,
                    lambda::AbstractVector; prefactor::Symbol = :first_order)
    length(P) == length(lambda) || throw(ArgumentError(
        "spectrum!: P has length $(length(P)) but lambda has length $(length(lambda))"))
    first_order = _prefactor_flag(prefactor)
    @inbounds for i in eachindex(lambda)
        P[i] = _spectrum_at(plasma, probe, collection, lambda[i], first_order)
    end
    return P
end

"""
    spectrum(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector;
             prefactor::Symbol = :first_order) -> Vector

    P(λ_s) = (1 + 2ω/ω_i) · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²,     G = 1,     units m⁻¹

The scattered power spectrum on the wavelength axis `lambda` (m, increasing). Allocates its
output with `similar(lambda, ...)` and calls `spectrum!`.

Sheffield Eqs. 2.3.13, 2.3.17; `conventions.md` 1.8, 3.4; `docs/derivations/spectrum.md`;
spec D11, D40.
"""
function spectrum(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector;
                   prefactor::Symbol = :first_order)
    P = similar(lambda, _spectrum_eltype(plasma, probe, collection, lambda))
    return spectrum!(P, plasma, probe, collection, lambda; prefactor = prefactor)
end
