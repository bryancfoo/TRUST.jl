# The three test problems of foo2026 Section III, as functions of grid resolution. Spec D24.
# Included by `examples/*.jl` and by `test/instrument/test_examples.jl` (both by explicit
# relative path, so the profiles have one definition).
#
# two_channel_problem, synthesize_channel and single_channel_deck_text (spec D41, Step 4.A.3c,
# chunk 8): the synthetic two-channel problem, added beside the three above.

using OTIS
using Random: Xoshiro
import TOML

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

"""
    two_channel_problem(; nlambda_epw = 256, nlambda_iaw = 256, nt_epw = 1024, nt_iaw = 960, nbins = 32)

The synthetic two-channel problem of Phase 4.A Step 4.A.3c (spec D41): the EPW ablation plasma
plus a time-varying deuterium temperature, seen by an EPW and an IAW camera through one probe
and one collection. Each camera has its own window, instrument chain, noise seed and time axis;
the IAW camera's recorded axis runs 0.05 ns behind true time.
"""
function two_channel_problem(; nlambda_epw=256, nlambda_iaw=256, nt_epw=1024, nt_iaw=960, nbins=32)
    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.0)),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    m_of_t(t) = 2.0 + 2.9 * exp(-((t - 4.4) / 1.15)^2)
    TD_of_t(t) = 150 + 350 * (1 - exp(-(t - 2.0) / 1.0))
    leaves = ("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t,
              "electrons.e.f.m" => m_of_t, "ions.D.T_eV" => TD_of_t)
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    epw = (; t_ns_recorded=collect(range(2.0, 5.8; length=nt_epw)), t_shift_ns=0.0,
        lambda=collect(range(220e-9, 260e-9; length=nlambda_epw)),
        effects=(SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
                 ConstantBackground(level=25.0)),
        noise=StreakCameraNoise(), seed=20260924)
    iaw = (; t_ns_recorded=collect(range(1.95, 5.75; length=nt_iaw)), t_shift_ns=0.05,
        lambda=collect(range(262.0e-9, 264.5e-9; length=nlambda_iaw)),
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6)),
        noise=StreakCameraNoise(), seed=20260925)
    regions = [AveragingRegion(; nbins=nbins, t_range_ns=[2.2, 5.6])]
    return (; name="two_channel", template, leaves, probe, collection, epw, iaw, regions)
end

"""
    synthesize_channel(problem, name::Symbol) -> StreakData

One camera's synthetic streak, as that camera records it: the trace is built on the camera's
TRUE axis (recorded + shift), synthesized with its own seed (so its temporal IRF acts on its own
axis), and the result is relabelled with the RECORDED axis.
"""
function synthesize_channel(problem, name::Symbol)
    c = getproperty(problem, name)
    trace = PlasmaTrace(problem.template; t_ns=c.t_ns_recorded .+ c.t_shift_ns, leaves=problem.leaves)
    expt = Experiment(; trace=trace, probe=problem.probe, collection=problem.collection,
        lambda_m=c.lambda, effects=c.effects)
    sd = synthesize(expt, c.noise; rng=Xoshiro(c.seed))
    return StreakData(; lambda_m=sd.lambda, t_s=c.t_ns_recorded .* 1e-9, image=sd.image,
        sigma=sd.sigma, mask=sd.mask)
end

"""
    single_channel_deck_text(text::AbstractString, name::AbstractString) -> String

The single-channel deck derived, in code, from a channel-form deck's own text (spec D41, Section
9.3): parses `text`, restricts `channels` to `[name]`, drops every other `[channel.<other>]`
table, points `[output].dir` at `runs/two_channel/<name>`, and re-serializes it with
`TOML.print`. Every variable, the regions and the regularizer are left exactly as the input deck
gave them -- only the channel selection and the output directory change, so the single-channel
fit is free to move every variable the joint fit is.
"""
function single_channel_deck_text(text::AbstractString, name::AbstractString)
    raw = TOML.parse(text)
    raw["channels"] = Any[name]
    for k in collect(keys(raw["channel"]))
        k == name || delete!(raw["channel"], k)
    end
    raw["output"]["dir"] = "runs/two_channel/$name"
    buf = IOBuffer()
    TOML.print(buf, raw; sorted=true)
    return String(take!(buf))
end
