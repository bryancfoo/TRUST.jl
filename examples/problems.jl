# The three test problems of foo2026 Section III, as functions of grid resolution. Spec D24.
# Included by `examples/*.jl` and by `test/instrument/test_examples.jl` (both by explicit
# relative path, so the profiles have one definition).

using OTIS

"""
    epw_ablation_problem(; nlambda = 512, nt = 256)

foo2026 Section III.A (Figs. 3 top, 4 top): a super-Gaussian electron species with a
time-varying exponent `m(t)`, ablating deuterium ions at fixed temperature.

Spec D24.
"""
function epw_ablation_problem(; nlambda=512, nt=256)
    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    t_ns = collect(range(2.0, 5.6; length=nt))
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    m_of_t(t) = 2.0 + 2.9 * exp(-((t - 4.4) / 1.15)^2)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t, "electrons.e.f.m" => m_of_t))
    lambda = collect(range(220e-9, 260e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
               ConstantBackground(level=25.0))
    return (; name="epw_ablation", trace, probe, collection, lambda, effects,
        noise=StreakCameraNoise(), seed=20260901)
end

"""
    iaw_multiflow_problem(; nlambda = 512, nt = 256)

foo2026 Section III.B (Figs. 3 bottom, 4 bottom): interpenetrating deuteron and carbon flows,
Maxwellian, with a fixed electron background.

Spec D24.
"""
function iaw_multiflow_problem(; nlambda=512, nt=256)
    template = Plasma(n_e_cm3=3e19, electrons=(electrons(T_eV=400.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=200, fraction=0.95),
              ion(name="C", mass_amu=12.011, charge=6, T_eV=200, fraction=0.05)))
    t_ns = collect(range(0.0, 3.0; length=nt))
    n_e_of_t(t) = 3e19 * (10 / 3)^(t / 3.0)
    fD_of_t(t) = 0.95 - 0.45 * (t / 3.0)^2
    fC_of_t(t) = 1 - fD_of_t(t)
    uC_of_t(t) = 500 * exp(-t / 0.90)
    TD_of_t(t) = 200 + 400 * t
    TC_of_t(t) = 200 + 800 * (t / 3.0)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "ions.D.fraction" => fD_of_t, "ions.C.fraction" => fC_of_t,
                "ions.C.u_kms" => uC_of_t, "ions.D.T_eV" => TD_of_t, "ions.C.T_eV" => TC_of_t))
    lambda = collect(range(262.0e-9, 264.5e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6))
    return (; name="iaw_multiflow", trace, probe, collection, lambda, effects,
        noise=StreakCameraNoise(), seed=20260902)
end

"""
    iaw_shock_problem(; nlambda = 512, nt = 256)

foo2026 Section III.C (Fig. 5): a shock front at `t_s = 1.15` ns, everything discontinuous
there, deuteron and carbon ions sharing one flow velocity.

Spec D24.
"""
function iaw_shock_problem(; nlambda=512, nt=256)
    template = Plasma(n_e_cm3=6e19, electrons=(electrons(T_eV=500.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=60, fraction=0.78),
              ion(name="C", mass_amu=12.011, charge=6, T_eV=100, fraction=0.22)))
    t_ns = collect(range(0.0, 4.0; length=nt))
    ts = 1.15
    TD_of_t(t) = t < ts ? 60.0 : 60 + 380 * exp(-(t - ts) / 1.60)
    TC_of_t(t) = t < ts ? 100.0 : 100 + 1450 * exp(-(t - ts) / 0.85)
    fD_of_t(t) = t < ts ? 0.78 : 0.63
    fC_of_t(t) = 1 - fD_of_t(t)
    u_of_t(t) = t < ts ? 250.0 : 40.0
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("ions.D.T_eV" => TD_of_t, "ions.C.T_eV" => TC_of_t,
                "ions.D.fraction" => fD_of_t, "ions.C.fraction" => fC_of_t,
                "ions.D.u_kms" => u_of_t, "ions.C.u_kms" => u_of_t))
    lambda = collect(range(262.0e-9, 264.5e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6))
    return (; name="iaw_shock", trace, probe, collection, lambda, effects,
        noise=StreakCameraNoise(), seed=20260903)
end

"""
    build_experiment(problem) -> Experiment

Builds the `Experiment` for one of the three problem `NamedTuple`s above.
"""
function build_experiment(problem)
    return Experiment(trace=problem.trace, probe=problem.probe, collection=problem.collection,
        lambda_m=problem.lambda, effects=problem.effects)
end
