using Test
using OTIS
using OTIS: jvp!, jtvp!, jacobian, _gauss_newton_diagonal, _gauss_newton_floor, PRECOND_FLOOR_REL

# CV1, CV2 against dev/phase3C_instructions.md Section 4's Test table (spec D13, D16), and CV3
# against Section 6's Test table (spec D37), gated "optimizer diagnostics". CV1/CV2 use P1,
# test_optimizers.jl's own one-bin noise-free fixture (D46), rebuilt here rather than imported
# across the @safetestset boundary -- the same discipline test_jacobian.jl's P2/P3 already follow.
# CV3 uses P3 (test_jacobian.jl's own every-effect-on oracle fixture), rebuilt here for the same
# reason -- Section 6's own instruction (Pitfall 7): CV2's own P1 has no IRFs, no globals and no
# regularizer, so a floor test on it would exercise only one branch of _gauss_newton_diagonal.

const _CV_T_TRUTH = 250.0
const _CV_U_TRUTH = 40.0

function _cv_p1_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _cv_p1_expt()
    trace = PlasmaTrace(_cv_p1_template(); t_ns=[0.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=24)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
end

function _cv_p1_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0)
    v_u = FittedVariable("u_D"; bounds=(-200.0, 200.0), init=0.0)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

function _cv_p1_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

# P1's FitProblem, noise-free data generated at (T_D, u_D) = (_CV_T_TRUTH, _CV_U_TRUTH).
function _cv_p1_problem()
    expt = _cv_p1_expt()
    pspec = _cv_p1_pspec(expt)
    working = _cv_p1_working(expt, pspec)
    theta_truth = flatten(pspec, ["T_D" => _CV_T_TRUTH, "u_D" => _CV_U_TRUTH])
    truth = unflatten(working, pspec, theta_truth)
    image = measure(truth)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=ones(size(image)))
    return FitProblem(working, pspec, data)
end

# ------------------------------------------------------------------------------------------
# CV1: the g_trace hook records what it claims (P1)
# ------------------------------------------------------------------------------------------

@testset "CV1: the hook records what it claims (P1)" begin
    problem = _cv_p1_problem()
    start = flatten(problem.pspec, ["T_D" => _CV_T_TRUTH * 1.25, "u_D" => _CV_U_TRUTH * 0.75])

    g_trace = []
    result = solve(problem, LBFGS(); start=start, g_trace=g_trace)

    @test length(g_trace) == length(result.trajectory)
    @test all(g_trace[k].iteration == Int(result.trajectory[k].iteration) for k in eachindex(g_trace))

    # Recover the recorded iterate by re-running to the same iteration cap: LBFGS is deterministic
    # from a fixed start, so a run capped at the last recorded iteration reproduces it exactly.
    last_iter = Int(result.trajectory[end].iteration)
    result2 = solve(problem, LBFGS(; iterations=last_iter); start=start)
    g = zeros(problem.pspec.ntheta)
    gradient!(g, problem, result2.theta)
    @test isapprox(g_trace[end].gnorm_inf, maximum(abs, g); rtol=1e-12)
end

# ------------------------------------------------------------------------------------------
# CV2: _gauss_newton_diagonal is the diagonal of J_full^T J_full (P1)
# ------------------------------------------------------------------------------------------

@testset "CV2: _gauss_newton_diagonal against jvp! on unit vectors (P1)" begin
    problem = _cv_p1_problem()
    theta = flatten(problem.pspec, ["T_D" => _CV_T_TRUTH * 1.1, "u_D" => _CV_U_TRUTH * 0.8])
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta

    oracle = zeros(ntheta)
    out = zeros(problem.npixels)
    for k in 1:ntheta
        v = zeros(ntheta)
        v[k] = 1.0
        jvp!(out, problem, Jac, v)
        oracle[k] = sum(abs2, out)
    end
    @test maximum(oracle) > 0   # D21: a vacuous oracle must never pass

    d = _gauss_newton_diagonal(problem, Jac)
    @test length(d) == ntheta
    floor_d = 1e-12 * maximum(d)
    for k in 1:ntheta
        @test isapprox(d[k], oracle[k]; rtol=1e-12, atol=floor_d)
    end
end

# ------------------------------------------------------------------------------------------
# CV3: _gauss_newton_floor, on P3 (every effect on -- Section 6's own correction of CV2's gap)
# ------------------------------------------------------------------------------------------

# P3 (D46): four bins, eight wavelengths, every effect on -- test_jacobian.jl's own oracle
# fixture, rebuilt here rather than imported across the @safetestset boundary.
function _cv_p3_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0, f=SuperGaussian(2.5))],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=0.7),
              ion(name="C", mass_amu=12.0, charge=6, T_eV=200.0, u_kms=0.0, fraction=0.3)])
end

const _CV_P3_LAMBDA_NM = collect(range(262.9, 263.6; length=8))

function _cv_p3_expt()
    trace = PlasmaTrace(_cv_p3_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = _CV_P3_LAMBDA_NM .* 1e-9
    effects = (
        KSmearing(half_angle_deg=2.0, nodes=3),
        GradientSmearing(_cv_p3_template(); leaf="electrons.e.T_eV", width=30.0, nodes=3),
        WavelengthCalibration(lambda_offset_nm=0.02),
        SpectralIRF(fwhm_nm=0.08),
        TemporalIRF(TabulatedKernel((-1, 0, 2), (0.2, 0.5, 0.3))),
        Sensitivity(scale=1.0e-6, lambda_nm=[262.0, 263.25, 264.5], response=[0.8, 1.0, 0.9]),
        RationalBackground(numerator=(20.0, 2.0), denominator=(0.3,)),
    )
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m, effects=effects)
end

function _cv_p3_pspec(expt::Experiment)
    ksmearing_idx = findfirst(e -> e isa KSmearing, expt.effects)
    spectral_idx = findfirst(e -> e isa SpectralIRF, expt.effects)
    sens_idx = findfirst(e -> e isa Sensitivity, expt.effects)
    rational_idx = findfirst(e -> e isa RationalBackground, expt.effects)

    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0, time_varying=true)
    v_fD = FittedVariable("f_D"; bounds=(0.3, 0.9), init=0.7)
    v_half = FittedVariable("half_angle"; bounds=(0.001, 0.2), init=2.0 * pi / 180)
    v_sig = FittedVariable("sig_spec"; bounds=(1e-11, 2e-10), init=3.5e-11)
    v_scale = FittedVariable("scale"; bounds=(1e-7, 1e-5), init=1e-6)
    v_numer = FittedVariable("numer0"; bounds=(5.0, 50.0), init=20.0)

    return ParameterSpec(expt;
        variables=[v_Te, v_fD, v_half, v_sig, v_scale, v_numer],
        bindings=["T_e" => "electrons.e.T_eV", "f_D" => "ions.D.fraction",
            "half_angle" => (ksmearing_idx, :half_angle),
            "sig_spec" => (spectral_idx, :sigma),
            "scale" => (sens_idx, :scale),
            "numer0" => (rational_idx, :numerator, 1)])
end

function _cv_p3_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

function _cv_p3_problem()
    expt = _cv_p3_expt()
    pspec = _cv_p3_pspec(expt)
    working = _cv_p3_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig)
    problem = FitProblem(working, pspec, data)
    theta = theta0 .+ 0.02 .* (1:length(theta0))
    return problem, theta
end

@testset "CV3: _gauss_newton_floor on the real P3 diagonal (not P1)" begin
    problem, theta = _cv_p3_problem()
    Jac = jacobian(problem, theta)
    d = _gauss_newton_diagonal(problem, Jac)
    @test maximum(d) > 0   # D21: a vacuous oracle must never pass

    dg = _gauss_newton_floor(d)
    @test dg !== nothing
    Pinv = 1 ./ (2 .* dg)
    @test all(isfinite, Pinv)
    @test maximum(dg) / minimum(dg) <= 1 / PRECOND_FLOOR_REL
    @test all(dg .>= d)   # the floor only ever raises an entry, never lowers one
    @info "CV3" spread_d = maximum(d) / minimum(d) spread_dg = maximum(dg) / minimum(dg) n_bound = count(<(PRECOND_FLOOR_REL * maximum(d)), d)
end
