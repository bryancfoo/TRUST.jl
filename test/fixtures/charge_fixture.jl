# The continuous charge's fixture (Phase 4.J spec D11), shared by test/plasma/test_charge.jl (CZ1 to
# CZ13, fast) and its slow namesake under test/slow/ (CZ14 to CZ17). A deuterium-carbon plasma in the
# spirit of Phase E's late carbon (shot 116349, 4.4 to 5.6 ns: x_C 0.6 to 0.9, T 1.5 to 2.8 keV), cut
# down: its carbon either as two charge states sharing T, u, mass and shape (C4 and C6, Phase E's
# workaround), or as one species of real charge -- at Z* = (4 f4 + 6 f6)/(f4 + f6) = <Z^2>/<Z>, which
# reproduces the pair exactly (OQ-60), or at the number average Zbar = (f4 + f6)/(f4/4 + f6/6), which
# does not. Step 4.J.1's part: the plasmas, the (k, omega) sweep, the two-window measurement with the
# gain on, and a one-bin deck. Step 4.J.2 appends the derivative's problem and the recovery.

using Random: Xoshiro

const _CZ_NE_CM3 = 5.0e19
const _CZ_TE_EV = 1500.0
const _CZ_TD_EV = 1500.0
const _CZ_TC_EV = 2000.0
const _CZ_UC_KMS = 100.0
# Shot 116349's probe with its energetics (Phase E's U22 and U24; psi = 26 deg, the user's at Hold 2).
const _CZ_PROBE = ProbeBeam(lambda_probe_nm=263.25, intensity_wcm2=5.55e14, diameter_um=100.0,
    polarization_deg=26.0)
const _CZ_COLL = Collection(theta_deg=60.0, k_model=:exact)
const _CZ_IAW = collect(range(262.6, 263.9; length=48)) .* 1e-9
const _CZ_EPW = collect(range(215.0, 255.0; length=64)) .* 1e-9

# `4 * f4`, never `4f4`: Julia reads `6f6` as the Float32 literal 6.0e6 (spec D27, 2026-10-11)
_cz_zstar(f4, f6) = (4 * f4 + 6 * f6) / (f4 + f6)
_cz_zbar(f4, f6) = (f4 + f6) / (f4 / 4 + f6 / 6)

# The carbon's shape, the same for every carbon species of one plasma.
_cz_shape(shape::Symbol) = shape === :kappa ? Kappa(4.0) : Maxwellian()

# The carbon as C4 (charge 4, fraction f4) and C6 (charge 6, fraction f6) sharing T, u, mass and shape.
function _cz_pair(f4, f6; shape::Symbol=:maxwellian)
    return Plasma(n_e_cm3=_CZ_NE_CM3, electrons=(electrons(T_eV=_CZ_TE_EV),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=_CZ_TD_EV, fraction=1 - (f4 + f6)),
            ion(name="C4", mass_amu=12.011, charge=4, T_eV=_CZ_TC_EV, u_kms=_CZ_UC_KMS, f=_cz_shape(shape),
                fraction=f4),
            ion(name="C6", mass_amu=12.011, charge=6, T_eV=_CZ_TC_EV, u_kms=_CZ_UC_KMS, f=_cz_shape(shape),
                fraction=f6)))
end

# The carbon as one species "C" of charge Z and fraction x (the deuterium at 1 - x, the same Float64 as
# _cz_pair's 1 - (f4 + f6) when x = f4 + f6).
function _cz_single(Z, x; shape::Symbol=:maxwellian)
    return Plasma(n_e_cm3=_CZ_NE_CM3, electrons=(electrons(T_eV=_CZ_TE_EV),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=_CZ_TD_EV, fraction=1 - x),
            ion(name="C", mass_amu=12.011, charge=Z, T_eV=_CZ_TC_EV, u_kms=_CZ_UC_KMS, f=_cz_shape(shape),
                fraction=x)))
end

# The (k, omega) sweep (spec D11): k0, the probe's k at 60 deg in the constant-k approximation, and
# 0.8 k0 and 1.25 k0; at each, 61 frequencies omega = k x v_C for x from -6 to 6 (v_C the carbon's
# thermal speed: both acoustic peaks and the ions' wings), and one point on the electron plasma wave
# (Bohm-Gross, omega^2 = omega_pe^2 + 1.5 k^2 v_th,e^2 in the package's v_th = sqrt(2T/m)).
function _cz_sweep()
    k0 = 2 * (2π / 263.25e-9) * sin(π / 6)
    vC = sqrt(2 * _CZ_TC_EV * EV / (12.011 * ATOMIC_MASS_UNIT))
    e = _cz_single(5.0, 0.7).electrons[1]
    pts = Tuple{Float64,Float64}[]
    for k in (0.8k0, k0, 1.25k0)
        for x in range(-6.0, 6.0; length=61)
            push!(pts, (k, k * x * vC))
        end
        push!(pts, (k, sqrt(plasma_frequency(e)^2 + 1.5 * k^2 * thermal_speed(e)^2)))
    end
    return pts
end

# measure on a two-bin constant trace of `p` with the gain on: the IAW window (a 0.03 nm spectral IRF)
# or the EPW window (0.5 nm).
function _cz_measure(p::Plasma, window::Symbol)
    lam, fwhm = window === :epw ? (_CZ_EPW, 0.5) : (_CZ_IAW, 0.03)
    expt = Experiment(; trace=PlasmaTrace(p; t_ns=[0.0, 1.0]), probe=_CZ_PROBE, collection=_CZ_COLL,
        lambda_m=lam, effects=(SpectralIRF(fwhm_nm=fwhm), Sensitivity(scale=1.0)), gain=:convective)
    return measure(expt)
end

# The one-bin deck's streak (CZ2, CZ8, CZ13): _cz_single(5.0, 0.7)'s IAW window at t = 4.5 ns.
function _cz_streak(dir::AbstractString)
    expt = Experiment(; trace=PlasmaTrace(_cz_single(5.0, 0.7); t_ns=[4.5]),
        probe=ProbeBeam(lambda_probe_nm=263.25), collection=_CZ_COLL, lambda_m=_CZ_IAW,
        effects=(SpectralIRF(fwhm_nm=0.03), Sensitivity(scale=1.0e-6)))
    image = measure(expt)
    write_streak(joinpath(dir, "cz_iaw.jld2"), StreakData(lambda_m=_CZ_IAW, t_s=[4.5e-9], image=image,
        sigma=sqrt.(abs.(image)) .+ 1.0); source="charge fixture")
    return nothing
end

# The one-bin deck on that streak: `charge` is the text written after "charge = " in [plasma.ions.C]
# ("6", "4.6", "\"ZC\"", ...); `variables` is added to [fit.variables]; `tables` is appended at the end.
function _cz_deck_text(dir::AbstractString; charge::AbstractString="6", variables::AbstractString="",
    tables::AbstractString="")
    file = replace(joinpath(dir, "cz_iaw.jld2"), "\\" => "/")
    out = replace(joinpath(dir, "out"), "\\" => "/")
    return """
    deck_version = "otis-deck-2"
    [data]
    file = "$file"
    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    [instrument]
    spectral_irf = { fwhm_nm = 0.03 }
    sensitivity = { scale = 1.0e-6 }
    [fit.variables]
    T_i = { bounds = [300.0, 6000.0], init = 2000.0 }
    $variables
    [plasma]
    n_e_cm3 = 5.0e19
    [plasma.electrons.e]
    T_eV = 1500.0
    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 0.3
    T_eV = "T_i"
    [plasma.ions.C]
    mass_amu = 12.011
    charge = $charge
    fraction = 0.7
    T_eV = "T_i"
    u_kms = 100.0
    [optimizer]
    procedure = "levenberg_marquardt"
    [output]
    dir = "$out"
    $tables
    """
end

# Writes `text` to `dir/name` and returns the path.
_cz_write(dir::AbstractString, name::AbstractString, text::AbstractString) =
    (path = joinpath(dir, name); write(path, text); path)
