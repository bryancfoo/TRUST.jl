# The wavelength sampling's fixtures (Phase 4.F spec D17), shared by the fast files
# test/instrument/test_pixel_integration.jl and test/config/test_deck_resonance.jl and their slow
# counterparts under test/slow/: Phase 4.E's bin 25 at its reported fit (GeneralizedNormal m 5.4658,
# an EPW resonance 0.198 pixels wide) on the two-channel EPW window, as an experiment (_pi_gn_*) and
# as a one-bin deck (_rw_*); and test_jacobian.jl's KB chain, every kernel-stage parameter bindable,
# with a PixelIntegration (_pi_kb_*).

const _PI_PROBE = ProbeBeam(lambda_probe_nm=263.25)
const _PI_COLL = Collection(theta_deg=60.0, k_model=:exact)
const _PI_LAMBDA = collect(range(220e-9, 260e-9; length=256))   # the two-channel EPW window

# Phase 4.E's bin 25 at the reported fit: an EPW resonance 0.198 pixels wide (spec D17).
_pi_gn_plasma() = Plasma(n_e_cm3=10.0^19.793557,
    electrons=(electrons(T_eV=467.6205, f=GeneralizedNormal(5.4658)),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=497.4972),))

function _pi_gn_expt(extra...)
    trace = PlasmaTrace(_pi_gn_plasma(); t_ns=[4.6])
    return Experiment(; trace=trace, probe=_PI_PROBE, collection=_PI_COLL, lambda_m=_PI_LAMBDA,
        effects=(SpectralIRF(fwhm_nm=0.5), Sensitivity(scale=2.0e-4), extra...))
end

# test_jacobian.jl's KB fixture (every kernel-stage parameter bindable), with PixelIntegration.
_pi_kb_template() = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
    ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])

# n = 0 builds the same chain with no PixelIntegration at all.
function _pi_kb_expt(n::Integer)
    trace = PlasmaTrace(_pi_kb_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    pi_effect = n == 0 ? () : (PixelIntegration(subsamples=n),)
    effects = (pi_effect..., KSmearing(half_angle_deg=2.0, nodes=3),
        GradientSmearing(_pi_kb_template(); leaf="electrons.e.T_eV", width=30.0, nodes=3),
        WavelengthCalibration(lambda_offset_nm=0.02, dispersion_scale=0.001),
        SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6))
    return Experiment(trace=trace, probe=_PI_PROBE, collection=_PI_COLL,
        lambda_m=collect(range(262.9, 263.6; length=16)) .* 1e-9, effects=effects)
end

function _pi_kb_problem(n::Integer)
    expt = _pi_kb_expt(n)
    idx(T) = findfirst(e -> e isa T, expt.effects)
    pspec = ParameterSpec(expt;
        variables=[FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true),
            FittedVariable("half"; bounds=(0.001, 0.2), init=2 * pi / 180),
            FittedVariable("gw"; bounds=(1.0 * EV, 100.0 * EV), init=30.0 * EV),
            FittedVariable("dlam"; bounds=(-0.2e-9, 0.2e-9), init=0.02e-9),
            FittedVariable("eps"; bounds=(-0.01, 0.01), init=0.001)],
        bindings=["T_D" => "ions.D.T_eV", "half" => (idx(KSmearing), :half_angle),
            "gw" => (idx(GradientSmearing), :width), "dlam" => (idx(WavelengthCalibration), :offset),
            "eps" => (idx(WavelengthCalibration), :dispersion_scale)])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects)
    theta0 = flatten(pspec)
    image = measure(unflatten(working, pspec, theta0))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image,
        sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    return FitProblem(working, pspec, data), theta0 .+ 0.02 .* (1:length(theta0))
end

# The same bin as a one-bin deck, shorthand or one-channel channel form, with `line` added to its
# instrument table; its streak; and the warning's text on it at one point a pixel.
const _RW_PROBE = _PI_PROBE
const _RW_COLL = _PI_COLL
const _RW_EPW = _PI_LAMBDA

_rw_gn(m, Te, logn, TD) = Plasma(n_e_cm3=10.0^logn,
    electrons=(electrons(T_eV=Te, f=GeneralizedNormal(m)),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=TD),))

function _rw_deck_text(dir::AbstractString; form::Symbol=:shorthand, line::AbstractString="")
    head = form === :shorthand ? """
        [data]
        file = "$(replace(joinpath(dir, "s.jld2"), "\\" => "/"))"
        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0
        [instrument]
        spectral_irf = { fwhm_nm = 0.5 }
        sensitivity = { scale = 2.0e-4 }
        $line
        """ : """
        channels = ["epw"]
        [channel.epw.data]
        file = "$(replace(joinpath(dir, "s.jld2"), "\\" => "/"))"
        [channel.epw.geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0
        [channel.epw.instrument]
        spectral_irf = { fwhm_nm = 0.5 }
        sensitivity = { scale = 2.0e-4 }
        $line
        """
    return """
    deck_version = "otis-deck-2"
    $head
    [fit.variables]
    logn = { bounds = [18.5, 20.5], init = 19.793557 }
    T_e = { bounds = [100.0, 2000.0], init = 467.6205 }
    m_e = { bounds = [1.8, 8.0], init = 5.4658 }
    [fit.dependent]
    ne_cm3 = "10^logn"
    [plasma]
    n_e_cm3 = "ne_cm3"
    [plasma.electrons.e]
    T_eV = "T_e"
    shape = { type = "GeneralizedNormal", m = "m_e" }
    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    T_eV = 497.4972
    [optimizer]
    procedure = "levenberg_marquardt"
    [output]
    dir = "$(replace(joinpath(dir, "out"), "\\" => "/"))"
    """
end

function _rw_streak(dir::AbstractString)
    trace = PlasmaTrace(_rw_gn(5.4658, 467.6205, 19.793557, 497.4972); t_ns=[4.6])
    image = measure(Experiment(; trace=trace, probe=_RW_PROBE, collection=_RW_COLL, lambda_m=_RW_EPW,
        effects=(SpectralIRF(fwhm_nm=0.5), Sensitivity(scale=2.0e-4))))
    write_streak(joinpath(dir, "s.jld2"), StreakData(lambda_m=_RW_EPW, t_s=[4.6e-9], image=image,
        sigma=sqrt.(image) .+ 1.0); source="test")
    return nothing
end

const _RW_SHORT_MSG = "fit: 1 bin has an EPW resonance narrower than RESONANCE_WIDTH_WARN = 1.0 " *
    "wavelength pixel, which the model's sampling of each pixel aliases (the result is as fitted; " *
    "refit with the key named for each channel):\n" *
    "  main: bin 1 (FWHM 0.20 px; 1 per pixel now); [instrument] pixel_integration = { subsamples = 16 }"
