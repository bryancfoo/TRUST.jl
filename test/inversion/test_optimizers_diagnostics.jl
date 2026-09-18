using Test
using OTIS
using OTIS: jvp!, jtvp!, jacobian, _gauss_newton_diagonal

# CV1, CV2 against dev/phase3C_instructions.md Section 4's Test table (spec D13, D16), gated
# "optimizer diagnostics". Both rows use P1, test_optimizers.jl's own one-bin noise-free fixture
# (D46), rebuilt here rather than imported across the @safetestset boundary -- the same discipline
# test_jacobian.jl's P2/P3 already follow.

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
