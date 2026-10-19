# The tabulated k-smearing's fixture (Phase 4.G spec D9), shared by
# test/instrument/test_tabulated_k_smearing.jl and its slow namesake under test/slow/: shot 116349's
# ten collection rays as the user gave them on 2026-10-03 (experiment/README.md, "Geometry (D25 item
# 1)"), held here once; the two-channel example's probe and EPW and IAW windows, with the Collection
# at the rays' weighted mean angle, 60.01 deg; a four-bin trace of that example's truth with a
# Maxwellian electron (_tk_*); a one-bin deck (_tk_deck_*); and test_jacobian.jl's KB chain with the
# rays added (_tk_kb_*).

# The user's numbers (2026-10-03): each ray's angle to the probe (deg) and its weight.
const _TK_THETA_DEG = (53.63756, 55.02440, 56.41124, 57.79807, 59.18491, 60.57175, 61.95859, 63.34542,
    64.73226, 66.11910)
const _TK_WEIGHTS = (0.00702671050853565, 0.03914238097383, 0.091797666771767, 0.15030854466015,
    0.189541011666141, 0.195351560740507, 0.164271879645061, 0.106526733030044, 0.047475338948696,
    0.00855817305526778)
# The offsets the deck quotes: theta_q - 60.01 deg, the decimal differences.
const _TK_OFFSETS_DEG = (-6.37244, -4.98560, -3.59876, -2.21193, -0.82509, 0.56175, 1.94859, 3.33542,
    4.72226, 6.10910)

_tk_rays() = TabulatedKSmearing(offsets_deg=collect(_TK_OFFSETS_DEG), weights=collect(_TK_WEIGHTS))
# The equal-variance comparison: KSmearing's uniform spread, sqrt(3) x 2.56 deg, on five nodes.
_tk_uniform() = KSmearing(half_angle_deg=4.44, nodes=5)

const _TK_PROBE = ProbeBeam(lambda_probe_nm=263.25)
const _TK_COLL = Collection(theta_deg=60.01, k_model=:exact)
const _TK_EPW = collect(range(220e-9, 260e-9; length=256))     # the two-channel EPW window
const _TK_IAW = collect(range(262.0e-9, 264.5e-9; length=256))  # and its IAW window

_tk_template() = Plasma(n_e_cm3=1.5e19, electrons=(electrons(T_eV=350.0),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))

# two_channel_problem()'s n_e, T_e and T_D at four times; the electron Maxwellian.
function _tk_trace()
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    TD_of_t(t) = 150 + 350 * (1 - exp(-(t - 2.0) / 1.0))
    return PlasmaTrace(_tk_template(); t_ns=[2.5, 3.5, 4.5, 5.5],
        leaves=("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t, "ions.D.T_eV" => TD_of_t))
end

# One camera (:epw, SpectralIRF 0.5 nm; :iaw, 0.08 nm), `extra` in-kernel effects first.
function _tk_expt(window::Symbol, extra...)
    lam, fwhm = window === :epw ? (_TK_EPW, 0.5) : (_TK_IAW, 0.08)
    return Experiment(; trace=_tk_trace(), probe=_TK_PROBE, collection=_TK_COLL, lambda_m=lam,
        effects=(extra..., SpectralIRF(fwhm_nm=fwhm), Sensitivity(scale=2.0e-4)))
end

# The deck: one bin of the EPW camera, shorthand or one-channel channel form, `line` added to its
# instrument table; its streak, made with the rays.
const _TK_DECK_LINE = "tabulated_k_smearing = { offsets_deg = [$(join(_TK_OFFSETS_DEG, ", "))], " *
    "weights = [$(join(_TK_WEIGHTS, ", "))] }"

function _tk_deck_text(dir::AbstractString; form::Symbol=:shorthand, line::AbstractString=_TK_DECK_LINE)
    file = replace(joinpath(dir, "s.jld2"), "\\" => "/")
    head = form === :shorthand ? """
        [data]
        file = "$file"
        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.01
        [instrument]
        spectral_irf = { fwhm_nm = 0.5 }
        sensitivity = { scale = 2.0e-4 }
        $line
        """ : """
        channels = ["epw"]
        [channel.epw.data]
        file = "$file"
        [channel.epw.geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.01
        [channel.epw.instrument]
        spectral_irf = { fwhm_nm = 0.5 }
        sensitivity = { scale = 2.0e-4 }
        $line
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

function _tk_streak(dir::AbstractString)
    p = Plasma(n_e_cm3=10.0^19.7, electrons=(electrons(T_eV=600.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=400.0),))
    image = measure(Experiment(; trace=PlasmaTrace(p; t_ns=[4.5]), probe=_TK_PROBE, collection=_TK_COLL,
        lambda_m=_TK_EPW, effects=(_tk_rays(), SpectralIRF(fwhm_nm=0.5), Sensitivity(scale=2.0e-4))))
    write_streak(joinpath(dir, "s.jld2"), StreakData(lambda_m=_TK_EPW, t_s=[4.5e-9], image=image,
        sigma=sqrt.(image) .+ 1.0); source="test")
    return nothing
end

# test_jacobian.jl's KB fixture (every kernel-stage parameter bindable) with the rays after its
# KSmearing, and a PixelIntegration of `n` sub-samples last (none for n = 0).
_tk_kb_template() = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
    ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])

function _tk_kb_problem(n::Integer)
    trace = PlasmaTrace(_tk_kb_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    pe = n == 0 ? () : (PixelIntegration(subsamples=n),)
    expt = Experiment(trace=trace, probe=_TK_PROBE, collection=_TK_COLL,
        lambda_m=collect(range(262.9, 263.6; length=16)) .* 1e-9,
        effects=(KSmearing(half_angle_deg=2.0, nodes=3), _tk_rays(),
            GradientSmearing(_tk_kb_template(); leaf="electrons.e.T_eV", width=30.0, nodes=3),
            WavelengthCalibration(lambda_offset_nm=0.02, dispersion_scale=0.001), pe...,
            SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6)))
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
