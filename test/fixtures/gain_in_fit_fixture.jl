# The convective gain in the fit path's fixture (Phase 4.H spec D10), shared by
# test/instrument/test_gain_in_fit.jl and its slow namesake under test/slow/. It is written against
# test/fixtures/tabulated_k_smearing_fixture.jl, which the including file includes first: the
# Collection at 60.01 deg, the two-channel example's EPW and IAW windows, its four-bin trace and the
# ten rays (_TK_*, _tk_*). Added here: shot 116349's probe with its energetics (_GF_PROBE); a
# cut-down problem with the gain on, a kernel-stage effect bound and n_e, T_e and a charge fraction
# fitted (_gf_*); and a one-bin deck whose geometry carries the gain (_gf_deck_*).

# Shot 116349's probe (the user, 2026-10-05; Phase E's U22 and U24): 263.25 nm, 5.55e14 W/cm^2 in a
# 100 um beam, and psi = 26 deg, the user's choice at Phase E's Hold 2 (the factor 1 - sin^2(psi)
# sin^2(theta) is 0.8558 at 60.01 deg). Any psi other than zero would do: it makes each ray's
# polarization factor its own.
const _GF_PROBE = ProbeBeam(lambda_probe_nm=263.25, intensity_wcm2=5.55e14, diameter_um=100.0,
    polarization_deg=26.0)

# One camera of the ten-ray fixture with the gain probe: no detector effect unless `chain`, so that
# measure is the ideal streak and a column is spectrum's output (GF2 to GF4).
function _gf_expt(window::Symbol, gain::Symbol, extra...; chain::Bool=false)
    lam, fwhm = window === :epw ? (_TK_EPW, 0.5) : (_TK_IAW, 0.08)
    effects = chain ? (extra..., SpectralIRF(fwhm_nm=fwhm), Sensitivity(scale=2.0e-4)) : (extra...,)
    return Experiment(; trace=_tk_trace(), probe=_GF_PROBE, collection=_TK_COLL, lambda_m=lam,
        effects=effects, gain=gain)
end

# The Jacobian's problem (GF5, GF6): four bins of a deuterium-carbon plasma on 16 IAW wavelengths;
# n_e and T_e fitted per bin, the deuterium charge fraction fitted with carbon's its complement (a
# dependent), and the KSmearing half-angle a kernel-stage global, so the dual pass carries a dual
# plasma and a dual collection into the gain.
_gf_template() = Plasma(n_e_cm3=3e19, electrons=(electrons(T_eV=400.0),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, fraction=0.7),
          ion(name="C", mass_amu=12.011, charge=6, T_eV=200.0, u_kms=100.0, fraction=0.3)))

function _gf_problem(gain::Symbol)
    trace = PlasmaTrace(_gf_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    expt = Experiment(; trace=trace, probe=_GF_PROBE, collection=_TK_COLL,
        lambda_m=collect(range(262.9, 263.6; length=16)) .* 1e-9,
        effects=(KSmearing(half_angle_deg=2.0, nodes=3), SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6)),
        gain=gain)
    ks = findfirst(e -> e isa KSmearing, expt.effects)
    pspec = ParameterSpec(expt;
        variables=[FittedVariable("ne"; bounds=(1e19, 1e20), init=3e19, time_varying=true),
            FittedVariable("Te"; bounds=(100.0, 1000.0), init=400.0, time_varying=true),
            FittedVariable("fD"; bounds=(0.2, 0.95), init=0.7),
            FittedVariable("half"; bounds=(0.001, 0.2), init=2 * pi / 180)],
        dependents=["fC" => "1 - fD"],
        bindings=["ne" => "n_e_cm3", "Te" => "electrons.e.T_eV", "fD" => "ions.D.fraction",
            "fC" => "ions.C.fraction", "half" => (ks, :half_angle)])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, gain=expt.gain)
    theta0 = flatten(pspec)
    image = measure(unflatten(working, pspec, theta0))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image,
        sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    return FitProblem(working, pspec, data), theta0 .+ 0.02 .* (1:length(theta0))
end

# The deck (GF7, GF8): one bin of the EPW camera, shorthand or one-channel channel form, its geometry
# the fixture's with `geometry` added (the probe's energetics and the gain key, or nothing); its streak.
const _GF_ENERGETICS = "intensity_wcm2 = 5.55e14\ndiameter_um = 100.0\npolarization_deg = 26.0"

function _gf_deck_text(dir::AbstractString; form::Symbol=:shorthand, geometry::AbstractString="")
    file = replace(joinpath(dir, "s.jld2"), "\\" => "/")
    head = form === :shorthand ? """
        [data]
        file = "$file"
        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.01
        $geometry
        [instrument]
        spectral_irf = { fwhm_nm = 0.5 }
        sensitivity = { scale = 2.0e-4 }
        """ : """
        channels = ["epw"]
        [channel.epw.data]
        file = "$file"
        [channel.epw.geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.01
        $geometry
        [channel.epw.instrument]
        spectral_irf = { fwhm_nm = 0.5 }
        sensitivity = { scale = 2.0e-4 }
        """
    return """
    deck_version = "trust-deck-2"
    $head
    [fit.variables]
    logn = { bounds = [18.5, 20.5], init = 19.7 }
    T_e = { bounds = [100.0, 2000.0], init = 600.0 }
    [fit.dependent]
    ne_cm3 = "10^logn"
    [plasma]
    n_e_cm3 = "ne_cm3"
    [plasma.electrons.e]
    T_eV = "T_e"
    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    T_eV = 400.0
    [optimizer]
    procedure = "levenberg_marquardt"
    [output]
    dir = "$(replace(joinpath(dir, "out"), "\\" => "/"))"
    """
end

function _gf_streak(dir::AbstractString)
    p = Plasma(n_e_cm3=10.0^19.7, electrons=(electrons(T_eV=600.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=400.0),))
    image = measure(Experiment(; trace=PlasmaTrace(p; t_ns=[4.5]), probe=_GF_PROBE, collection=_TK_COLL,
        lambda_m=_TK_EPW, effects=(SpectralIRF(fwhm_nm=0.5), Sensitivity(scale=2.0e-4)), gain=:convective))
    write_streak(joinpath(dir, "s.jld2"), StreakData(lambda_m=_TK_EPW, t_s=[4.5e-9], image=image,
        sigma=sqrt.(image) .+ 1.0); source="test")
    return nothing
end
