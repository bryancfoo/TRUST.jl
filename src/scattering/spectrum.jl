# The scattered power spectrum P(lambda_s): the prefactor, the exact wavelength Jacobian, the
# convective gain, and the per-bin kernel. conventions.md 1.8, 3.4; docs/derivations/spectrum.md,
# docs/derivations/gain.md; spec D11, D39, D43.

"""
    GAIN_SERIES_CUTOFF

Below `|γL| < GAIN_SERIES_CUTOFF`, `_convective_gain` uses the series `1 + γL/2` instead of
`expm1(γL)/γL`: at this scale the two agree to machine precision, but the small-γL limit is
exactly where the gain must reduce to no gain, and `(exp(γL)-1)/γL` (never written; `expm1` is
used instead) would lose every significant digit there.

Spec D43, D45.
"""
const GAIN_SERIES_CUTOFF = 1e-8

"""
    SIN_THETA_MIN

The convective gain's length `L = probe.diameter / sin(collection.theta)` diverges as `θ → 0`;
`_convective_gain` throws when `abs(sin(collection.theta)) < SIN_THETA_MIN`.

Spec D43, D45.
"""
const SIN_THETA_MIN = 1e-6

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
    _convective_gain(gain::Symbol, d, k, k_s, probe::ProbeBeam, collection::Collection) -> Real

Turnbull's convective SRS/SBS gain factor `g` multiplying `S` in forming `P` (Turnbull Eqs. 6-8;
`docs/derivations/gain.md`). `gain` is `:none` (returns `one`, no extra work), `:convective`
(Eq. 7) or `:weak` (Eq. 8, the small-gain approximation); anything else throws. `d` is
`dielectric`'s return, already computed once by the caller and reused here — no second
`dielectric` call.

    F_χ = χ_e (ε − χ_e) / ε
    γ   = −k² / (4 k_s) · Im(F_χ) · a0_sq · polarization_factor(probe, collection)
    L   = probe.diameter / sin(collection.theta)
    g   = expm1(γL) / (γL)     (:convective; the series `1 + γL/2` below `GAIN_SERIES_CUTOFF`)
    g   = exp(γL / 2)          (:weak)

**The minus sign is not in Turnbull's own Eq. 6** — it translates his gain rate, written in his
`ω = ω_0 − ω_s` (Stokes at his `ω > 0`), into this package's `ω = ω_s − ω_i` (Stokes at our
`ω < 0`, red of the probe, `conventions.md` 3.1). Turnbull's own susceptibility (his Eq. 2) is
the same Landau-contour form `conventions.md` 1.5/1.6 already fixes, and `Im(χ)` is an odd
function of `ω` for any even `f(v)`, so `Im(F_χ)` evaluated at our `ω` is minus `Im(F_χ)`
evaluated at his `ω` for the same physical point — without the sign, Stokes would come out
depleted and anti-Stokes amplified, the opposite of Turnbull's own stated result.
`docs/derivations/gain.md` Section 6 derives this from his Eq. 2; CG4 is the test (the
load-bearing row of Step 2.3b: `g > 1` red of the probe, `g < 1` blue of it).

`gain = :convective`/`:weak` with `probe.a0_sq == 0` or `probe.diameter == 0` throws naming which
is missing; `abs(sin(collection.theta)) < SIN_THETA_MIN` throws naming `theta` (`L` would
diverge at `θ = 0`).

Turnbull Eqs. 6-8 (translated to this package's `ω`, see above); `conventions.md` 1.8;
`docs/derivations/gain.md`; spec D43, deviation approved by the user 2026-09-11 (the literal
`γ` of D43 omitted this sign; `conventions.md` 3.1's `ω` convention is the source of truth).
"""
function _convective_gain(gain::Symbol, d, k, k_s, probe::ProbeBeam, collection::Collection)
    gain === :none && return one(real(d.chi_e))
    gain in (:convective, :weak) || throw(ArgumentError(
        "spectrum: gain must be :none, :convective or :weak; got :$gain"))

    iszero(_primal_value(probe.a0_sq)) && throw(ArgumentError(
        "_convective_gain: gain = :$gain needs the probe's intensity, but a0_sq == 0 " *
        "(build the ProbeBeam with intensity_wcm2 or intensity_wm2)"))
    iszero(_primal_value(probe.diameter)) && throw(ArgumentError(
        "_convective_gain: gain = :$gain needs the probe's beam diameter, but diameter == 0 " *
        "(build the ProbeBeam with diameter_um or diameter_m)"))
    s = sin(collection.theta)
    abs(s) >= SIN_THETA_MIN || throw(ArgumentError(
        "_convective_gain: sin(theta) is below SIN_THETA_MIN at theta = $(collection.theta) rad; " *
        "the gain length diverges as theta -> 0"))

    F_chi = d.chi_e * (d.epsilon - d.chi_e) / d.epsilon
    # The minus sign translates Turnbull's gain rate, written in HIS omega = omega_0 - omega_s
    # (Stokes = his omega > 0), into our omega = omega_s - omega_i (Stokes = our omega < 0, red
    # of the probe, conventions.md 3.1). Turnbull's own chi (his Eq. 2) is the same Landau-
    # contour susceptibility conventions.md 1.5/1.6 already fixes, and Im(chi) is an odd function
    # of omega for any even f(v), so Im(F_chi) evaluated at our omega equals minus Im(F_chi)
    # evaluated at Turnbull's omega for the same physical point -- without this sign, Stokes
    # would come out depleted and anti-Stokes amplified, the opposite of Turnbull's own result.
    # docs/derivations/gain.md Section 6 derives this from his Eq. 2 directly; CG4 is the test.
    gamma = -k^2 / (4 * k_s) * imag(F_chi) * probe.a0_sq * polarization_factor(probe, collection)
    L = probe.diameter / s
    gl = gamma * L

    if gain === :convective
        return abs(gl) < GAIN_SERIES_CUTOFF ? 1 + gl / 2 : expm1(gl) / gl
    else
        return exp(gl / 2)
    end
end

"""
    _spectrum_at(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda_s,
                 first_order::Bool = true, gain::Symbol = :none) -> Real

The scattered power at one scattered wavelength `lambda_s` (m):
`(1 ± prefactor) · g · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²`. `dielectric` is evaluated **once** here and
threaded into the four-argument `spectral_density` and into `_convective_gain`, so that the
convective gain reuses `(ε, χ_e)` without a second `dielectric` call (spec D39; RF2 counts the
calls). `first_order` and `gain` are positional with defaults so that `spectrum!` can hoist their
`Symbol`-to-typed conversion out of its loop while a test can still call this with fewer
arguments.

`conventions.md` 1.8; `docs/derivations/spectrum.md`, `docs/derivations/gain.md`; spec D11, D39,
D43.
"""
function _spectrum_at(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda_s,
                       first_order::Bool = true, gain::Symbol = :none)
    k, omega, k_s = wavevector(probe, collection, lambda_s, plasma)
    d = dielectric(plasma, k, omega)
    S = spectral_density(plasma, k, omega, d)
    g = _convective_gain(gain, d, k, k_s, probe, collection)
    omega_probe = _omega_probe(probe)
    return _prefactor(omega, omega_probe, first_order) * g * S * _wavelength_jacobian(lambda_s)
end

"""
    spectrum!(P::AbstractVector, plasma::Plasma, probe::ProbeBeam, collection::Collection,
              lambda::AbstractVector; prefactor::Symbol = :first_order, gain::Symbol = :none) -> P

The in-place form of `spectrum`: fills `P` with the scattered power spectrum at every wavelength
in `lambda` (m, increasing). `prefactor` is `:first_order` (Sheffield Eq. 2.3.17, default) or
`:none` (Eq. 2.3.13). `gain` is `:none` (default, bitwise the pre-2026-09-10 result),
`:convective` or `:weak` (Turnbull's convective SRS/SBS gain, `docs/derivations/gain.md`). `G =
1`; no `r_0`. This is the per-bin kernel Phase 2 calls and the one the allocation tests run.

Sheffield Eqs. 2.3.13, 2.3.17; Turnbull Eqs. 6-8; `conventions.md` 1.8, 3.4;
`docs/derivations/spectrum.md`, `docs/derivations/gain.md`; spec D11, D40, D43.
"""
function spectrum!(P::AbstractVector, plasma::Plasma, probe::ProbeBeam, collection::Collection,
                    lambda::AbstractVector; prefactor::Symbol = :first_order, gain::Symbol = :none)
    length(P) == length(lambda) || throw(ArgumentError(
        "spectrum!: P has length $(length(P)) but lambda has length $(length(lambda))"))
    first_order = _prefactor_flag(prefactor)
    @inbounds for i in eachindex(lambda)
        P[i] = _spectrum_at(plasma, probe, collection, lambda[i], first_order, gain)
    end
    return P
end

"""
    spectrum(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector;
             prefactor::Symbol = :first_order, gain::Symbol = :none) -> Vector

    P(λ_s) = G · (1 + 2ω/ω_i) · g(λ_s) · S(k(λ_s), ω(λ_s)) · 2πc/λ_s²,     G = 1,     units m⁻¹

The scattered power spectrum on the wavelength axis `lambda` (m, increasing), `g` the convective
gain of `gain` (`:none` the default, under which `g ≡ 1` and the result is bitwise what it was
before 2026-09-10). Allocates its output with `similar(lambda, ...)` and calls `spectrum!`.

Sheffield Eqs. 2.3.13, 2.3.17; Turnbull Eqs. 6-8; `conventions.md` 1.8, 3.4;
`docs/derivations/spectrum.md`, `docs/derivations/gain.md`; spec D11, D40, D43.
"""
function spectrum(plasma::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector;
                   prefactor::Symbol = :first_order, gain::Symbol = :none)
    P = similar(lambda, _spectrum_eltype(plasma, probe, collection, lambda))
    return spectrum!(P, plasma, probe, collection, lambda; prefactor = prefactor, gain = gain)
end
