# Phase 5.A's shared fixtures (spec D12): the recovery's single-channel EPW problem and HE11's deck, used by
# test/distributions/test_hermite.jl (HE8, HE11) and test/slow/test_hermite.jl (HE12, HE15). The including
# file loads Test, OTIS and Random.

const _HE_LAMBDA_NM = collect(range(238.0, 292.0; length = 48))
const _HE_MASK = reshape([abs(l - 263.25) >= 5.0 for l in _HE_LAMBDA_NM], :, 1)

# The warning's first line, quoted by HE8 and HE15 (spec D10).
const _HE_WARN_HEAD = "fit: in the final fit, a HermiteExpansion shape is negative over more than " *
    "HERMITE_NEGATIVE_MASS_WARN = 0.001 of the distribution at the bins below, where the expansion has left " *
    "its region of validity (the result is as fitted; `converged` does not see this):"

function _he_epw_expt(f)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = f)],
        ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 200.0)])
    return Experiment(trace = PlasmaTrace(template; t_ns = [0.0]), probe = ProbeBeam(lambda_probe_nm = 263.25),
        collection = Collection(theta_deg = 60.0, k_model = :exact), lambda_m = _HE_LAMBDA_NM .* 1e-9,
        effects = (Sensitivity(scale = 1.0e-6), ConstantBackground(level = 25.0)))
end

# The recovery's problem (OQ-66 (l)): a single-channel EPW streak from electrons of shape `truth_f`, one bin,
# 48 pixels from 238 to 292 nm with the 10 nm around the probe masked, GaussianNoise(a = 0.0625, b = 0.0025)
# from Xoshiro(20261017), the model's electrons a HermiteExpansion with c3, c4 and T_e free. Returns the
# problem, its ParameterSpec and the start (0, 0, 300).
function _he_recovery_problem(truth_f)
    truth = _he_epw_expt(truth_f)
    s = synthesize(truth, GaussianNoise(a = 0.0625, b = 0.0025); rng = Random.Xoshiro(20261017))
    data = StreakData(lambda_m = s.lambda, t_s = s.t, image = s.image, sigma = s.sigma, mask = _HE_MASK)
    model = _he_epw_expt(HermiteExpansion())
    pspec = ParameterSpec(model; variables = [
            FittedVariable("c3_e"; bounds = (-0.06, 0.06), init = 0.0),
            FittedVariable("c4_e"; bounds = (-0.02, 0.04), init = 0.0),
            FittedVariable("T_e"; bounds = (100.0, 1000.0), init = 300.0)],
        bindings = ["c3_e" => "electrons.e.f.c3", "c4_e" => "electrons.e.f.c4", "T_e" => "electrons.e.T_eV"])
    working = Experiment(; trace = pspec.trace, probe = model.probe, collection = model.collection,
        lambda_m = model.lambda, effects = model.effects, prefactor = model.prefactor)
    return FitProblem(working, pspec, data), pspec, flatten(pspec, ["c3_e" => 0.0, "c4_e" => 0.0, "T_e" => 300.0])
end

# HE11's deck (spec D9): the electrons' shape { type = "HermiteExpansion", c3 = "c3_e", c4 = "c4_e", c5 = 0.002 }
# with c3_e time-varying and c4_e scalar, on an IAW streak of two bins.
function _he11_deck(datafile, outdir)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    c3_e = { bounds = [-0.05, 0.05], init = 0.0, time_varying = true }
    c4_e = { bounds = [-0.01, 0.04], init = 0.0 }
    T_D = { bounds = [50.0, 800.0], init = 200.0 }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0
    shape = { type = "HermiteExpansion", c3 = "c3_e", c4 = "c4_e", c5 = 0.002 }

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = 40.0

    [instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "levenberg_marquardt"

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

# HE11's data: two bins of the IAW window from electrons with c3 = 0.04 and the deck's own c5 = 0.002.
function _he11_write_streak(path)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = HermiteExpansion(c3 = 0.04, c5 = 0.002))],
        ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0, u_kms = 40.0)])
    expt = Experiment(trace = PlasmaTrace(template; t_ns = [0.0, 1.0]), probe = ProbeBeam(lambda_probe_nm = 263.25),
        collection = Collection(theta_deg = 60.0, k_model = :exact),
        lambda_m = collect(range(262.0, 264.5; length = 24)) .* 1e-9,
        effects = (Sensitivity(scale = 1.0e-6), ConstantBackground(level = 25.0)))
    image = measure(expt)
    write_streak(path, StreakData(lambda_m = expt.lambda, t_s = expt.t, image = image, sigma = ones(size(image)));
        source = "test")
end
