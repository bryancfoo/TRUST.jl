# The scattering geometry, k(lambda_s) from the light-wave dispersion relation, the mean k-hat
# convention, and the polarization factor. conventions.md 1.2, 3.2, 3.3, 1.8;
# docs/derivations/geometry.md; spec D4-D7, D15, D16.

using Printf: @sprintf, @printf

"""
    ScatteringGeometry{FT<:Real}

The probe wavelength, scattering angle and incident polarization angle of one scattering
geometry, plus the choice of `k` formula, `k_model`. All fields are SI (`lambda_probe`, `theta`,
`polarization` in m, rad, rad); `k_model` is `:exact` (the light-wave dispersion relation,
default) or `:approximate` (the constant `k = 2k_i sin(θ/2)`, never the default).

Built with `ScatteringGeometry(; lambda_probe_nm | lambda_probe_m, theta_deg | theta_rad,
polarization_deg | polarization_rad = 0, k_model::Symbol = :exact)`.

`conventions.md` 1.2, 3.3, Section 4; `docs/derivations/geometry.md`; spec D5.
"""
struct ScatteringGeometry{FT<:Real}
    lambda_probe::FT     # m
    theta::FT            # rad, the angle between k_i and k_s
    polarization::FT     # rad, the angle of E_i0 from the normal to the scattering plane
    k_model::Symbol      # :exact (default) or :approximate
end

function ScatteringGeometry(; k_model::Symbol = :exact, kwargs...)
    lambda_probe_raw = _si_value(:lambda_probe, kwargs)
    theta_raw = _si_value(:theta, kwargs)
    polarization_raw = _si_value(:polarization, kwargs, 0.0)
    _reject_extra_kwargs(kwargs, (:lambda_probe_nm, :lambda_probe_m, :theta_deg, :theta_rad,
                                   :polarization_deg, :polarization_rad, :k_model))

    k_model in (:exact, :approximate) || throw(ArgumentError(
        "ScatteringGeometry: k_model must be :exact or :approximate; got :$k_model"))
    lambda_probe_raw > 0 || throw(ArgumentError(
        "ScatteringGeometry: lambda_probe must be positive; got $lambda_probe_raw m"))
    0 <= theta_raw <= π || throw(ArgumentError(
        "ScatteringGeometry: theta must be between 0 and pi radians; got $theta_raw rad"))

    FT = float(promote_type(typeof(lambda_probe_raw), typeof(theta_raw), typeof(polarization_raw)))
    return ScatteringGeometry{FT}(FT(lambda_probe_raw), FT(theta_raw), FT(polarization_raw), k_model)
end

"""
    wavevector(geometry::ScatteringGeometry, lambda_s, plasma::Plasma) -> (k, omega)

The probed wavevector magnitude and frequency shift for one scattered wavelength `lambda_s`
(m), from the light-wave dispersion relation:

    ω_i   = 2πc/λ_i,             λ_i = geometry.lambda_probe
    ω_s   = 2πc/λ_s,             ω = ω_s − ω_i               (ω > 0 is a blue shift)
    ω_pe² = n_e e²/(ε₀ m_e),     n_e = plasma.n_e             (the TOTAL electron density)
    k_i   = √(ω_i² − ω_pe²)/c,   k_s = √(ω_s² − ω_pe²)/c
    k     = √(k_i² + k_s² − 2 k_i k_s cos θ)                  (`k_model = :exact`, default)
    k     = (4π/λ_i) sin(θ/2)                                 (`k_model = :approximate`, the
                                                                **vacuum** k_i = 2π/λ_i)

`k` is in m⁻¹, `omega` in rad/s. Throws if `λ_i` or `λ_s` is at or beyond the plasma cutoff
(`ω ≤ ω_pe`), or if the resulting `k` is zero (only at `θ = 0` with `λ_s = λ_i`).

Sheffield Eqs. 5.4.1-5.4.3; `conventions.md` 3.3 ([OQ-8](../../dev/open_questions.md#oq-8));
`docs/derivations/geometry.md`; spec D6.
"""
function wavevector(geometry::ScatteringGeometry, lambda_s, plasma::Plasma)
    lambda_i = geometry.lambda_probe
    omega_probe = 2π * SPEED_OF_LIGHT / lambda_i
    omega_s = 2π * SPEED_OF_LIGHT / lambda_s
    omega = omega_s - omega_probe

    omega_pe_sq = plasma.n_e * ELEMENTARY_CHARGE^2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS)
    omega_pe = sqrt(omega_pe_sq)
    cutoff = 2π * SPEED_OF_LIGHT / omega_pe

    if !(omega_probe > omega_pe)
        throw(ArgumentError("wavevector: the probe wavelength $(@sprintf("%e", lambda_i)) m is at " *
            "or beyond the plasma cutoff $(@sprintf("%e", cutoff)) m (omega_probe <= omega_pe) " *
            "for n_e = $(@sprintf("%e", plasma.n_e)) m^-3"))
    end
    if !(omega_s > omega_pe)
        throw(ArgumentError("wavevector: the scattered wavelength $(@sprintf("%e", lambda_s)) m is at " *
            "or beyond the plasma cutoff $(@sprintf("%e", cutoff)) m (omega_s <= omega_pe) " *
            "for n_e = $(@sprintf("%e", plasma.n_e)) m^-3"))
    end

    k_i = sqrt(omega_probe^2 - omega_pe_sq) / SPEED_OF_LIGHT
    k_s = sqrt(omega_s^2 - omega_pe_sq) / SPEED_OF_LIGHT

    if geometry.k_model === :exact
        k = sqrt(k_i^2 + k_s^2 - 2 * k_i * k_s * cos(geometry.theta))
        if iszero(k)
            throw(ArgumentError("wavevector: k = 0 at theta = $(geometry.theta) rad with lambda_s " *
                "equal to the probe wavelength; this geometry probes no wavevector"))
        end
    else
        s = sin(geometry.theta / 2)
        if iszero(s)
            throw(ArgumentError("wavevector: k = 0 at theta = $(geometry.theta) rad with lambda_s " *
                "equal to the probe wavelength; this geometry probes no wavevector"))
        end
        k = 4π * s / lambda_i
    end

    return k, omega
end

"""
    polarization_factor(geometry::ScatteringGeometry) -> FT

The polarization factor `|ŝ × (ŝ × Ê_i0)|² = 1 − sin²θ sin²ψ`, with `ψ = geometry.polarization`
the angle of the incident polarization from the normal to the scattering plane. `ψ = 0` (the
usual arrangement) gives 1; `ψ = π/2` gives `cos²θ`. Computed for reference only: it is never
multiplied into `spectrum`, since the gain `G` absorbs it (`conventions.md` 1.8).

Sheffield Eq. 2.3.13; `conventions.md` 1.8; `docs/derivations/geometry.md`; spec D7.
"""
function polarization_factor(geometry::ScatteringGeometry)
    return 1 - sin(geometry.theta)^2 * sin(geometry.polarization)^2
end

# --- show methods (spec D16) ---

function Base.show(io::IO, g::ScatteringGeometry)
    lam = round(lambda_from_si(g.lambda_probe, :lambda_nm); sigdigits = 12)
    th = round(theta_from_si(g.theta, :theta_deg); sigdigits = 12)
    print(io, "ScatteringGeometry($lam nm, $th deg, :$(g.k_model))")
end

function Base.show(io::IO, ::MIME"text/plain", g::ScatteringGeometry)
    println(io, "ScatteringGeometry")
    @printf(io, "  lambda_probe = %e nm\n", lambda_from_si(g.lambda_probe, :lambda_nm))
    @printf(io, "  theta        = %e deg\n", theta_from_si(g.theta, :theta_deg))
    @printf(io, "  polarization = %e deg\n", theta_from_si(g.polarization, :theta_deg))
    println(io, "  k_model      = :$(g.k_model)")
end
