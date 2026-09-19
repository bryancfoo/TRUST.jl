using Test
using OTIS
using OTIS: _cgls, _build_procedure, _PROCEDURE_NAMES, _from_unconstrained,
    jtvp!, jvp!, jacobian, _full_residuals, _gauss_newton_diagonal, _gauss_newton_floor,
    PRECOND_FLOOR_REL
using LinearAlgebra: norm
import ForwardDiff

# OP1 to OP19 against dev/phase3_instructions.md Section 7's OP table (spec D27-D30) and Section
# 8's carried item 1 (OP19, spec D29), gated "optimizers". D46's P1 (one bin, 24 wavelengths,
# noise-free) is the reference problem for OP1-OP9: the minimum sits exactly at the generating
# parameters, so 1e-6 recovery is a statement about the optimizer, not about noise.
#
# P1's plasma recipe fixes T_D = 200 eV, u_D = 0 km/s at the template level (D46), but a zero
# truth makes "1e-6 relative on the recovered x" undefined for u_D (dividing by zero) -- so this
# file's P1 fixture builds its synthetic data from an explicit, nonzero truth for both variables
# (T_D = 250 eV, u_D = 40 km/s) via `flatten(pspec, ...)`, independent of the `FittedVariable`s'
# own `init`. Flagged under rule 11 zero: D46 does not give OP1's "+-25% of the truth" a concrete
# recipe that survives a zero truth.

const _OP_T_TRUTH = 250.0
const _OP_U_TRUTH = 40.0

# **Chunk 10 carried item 1** (D29, D43 item 3), **superseded at Step 3.C.3 (D38)**: chunk 9 found
# P1's `sigma = ones(...)` (D46) makes chi2's gradient in y-space enormous -- `|grad O|` is already
# ~2.6e5 at a mere 2% deviation in T_D from the truth -- and the old raw-gradient rule's shipped
# `step_size` overshot the very first step into the logit's saturated region on it. Chunk 9 worked
# around that here with a tuned `_OP_NGD_STEP_SIZE` override on OP5/OP7/OP8/OP9; Level 2 then
# measured that the enormous gradient is not a P1 artefact of unit sigma -- it is `dx/dy` carrying
# the bound width, present on any realistically weighted problem too -- and revised the raw-gradient
# default to `1.0e-8`, small enough that OP5/OP7/OP8/OP9 no longer needed the override. **Step
# 3.C.3 changes the update rule itself, not merely the tuning**: `step_size` is now a step *length*
# in the normalized gradient direction (D38), which cannot overshoot the way a raw-gradient
# multiplier could, by construction, on any problem's gradient scale -- so OP5/OP7/OP8/OP9 still
# use the shipped default with no override, for a structural reason now rather than a numerical
# coincidence. OP19 remains the row that would catch a regression to a default that saturates on
# its first step.

# ------------------------------------------------------------------------------------------
# P1 (D46): the one-bin problem, noise-free
# ------------------------------------------------------------------------------------------

function _op_p1_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _op_p1_expt()
    trace = PlasmaTrace(_op_p1_template(); t_ns=[0.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=24)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
end

function _op_p1_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0)
    v_u = FittedVariable("u_D"; bounds=(-200.0, 200.0), init=0.0)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

function _op_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

# Builds P1's FitProblem with noise-free data generated at (T_D, u_D) = (_OP_T_TRUTH, _OP_U_TRUTH).
function _op_p1_problem()
    expt = _op_p1_expt()
    pspec = _op_p1_pspec(expt)
    working = _op_working(expt, pspec)
    theta_truth = flatten(pspec, ["T_D" => _OP_T_TRUTH, "u_D" => _OP_U_TRUTH])
    truth = unflatten(working, pspec, theta_truth)
    image = measure(truth)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=ones(size(image)))
    return FitProblem(working, pspec, data)
end

_op_p1_start(; t_frac::Real=1.25, u_frac::Real=0.75) = (problem) -> flatten(problem.pspec,
    ["T_D" => _OP_T_TRUTH * t_frac, "u_D" => _OP_U_TRUTH * u_frac])

# ------------------------------------------------------------------------------------------
# P2 (D46): the ten-bin problem -- OP4, OP14
# ------------------------------------------------------------------------------------------

function _op_p2_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _op_p2_expt()
    trace = PlasmaTrace(_op_p2_template(); t_ns=collect(range(0.0, 3.0; length=10)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=32)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
end

function _op_p2_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

# Builds P2 with a smoothly time-varying truth (a linear ramp on both T_D and u_D), a regularizer
# with one D2 term on "T_D" at weight = 1e-3 (OP4), and returns (problem, theta_perturbed).
function _op_p2_problem()
    expt = _op_p2_expt()
    pspec = _op_p2_pspec(expt)
    working = _op_working(expt, pspec)
    nt = nbins(pspec.trace)
    t_ramp = range(220.0, 260.0; length=nt)
    u_ramp = range(-20.0, 20.0; length=nt)
    theta_truth = flatten(pspec, ["T_D" => collect(t_ramp), "u_D" => collect(u_ramp)])
    truth = unflatten(working, pspec, theta_truth)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig)

    term = RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)
    reg = Regularizer([term]; weight=1e-3)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    # 0.02, not the 0.1 an earlier draft used: at 0.1 the very first LBFGS step overshoot into
    # the logit's saturated region on this problem's particular gradient direction (Optim's line
    # search accepted it because the curvature condition is trivially satisfied once the
    # gradient is exactly zero there); 0.05 and below converge cleanly (checked by sweeping
    # 0.1 down to 0.001). Not a bug in solve() -- LM converges from the same 0.1 start fine, and
    # LBFGS itself converges fine from 0.05 -- just an unlucky arbitrary choice of test
    # perturbation for this direction. Flagged under rule 11 zero.
    theta_start = theta_truth .+ 0.02 .* (1:length(theta_truth))
    return problem, theta_start
end

# P2's own problem, but with a `mask` on the data half the pixels (CR3, D30: the criterion is per
# pixel, so the same physical setup at a different `problem.npixels` must give a proportionally
# different `f_abstol`). Otherwise identical to `_op_p2_problem` -- same truth, same regularizer,
# same start perturbation.
function _op_p2_problem_masked(mask::AbstractMatrix{Bool})
    expt = _op_p2_expt()
    pspec = _op_p2_pspec(expt)
    working = _op_working(expt, pspec)
    nt = nbins(pspec.trace)
    t_ramp = range(220.0, 260.0; length=nt)
    u_ramp = range(-20.0, 20.0; length=nt)
    theta_truth = flatten(pspec, ["T_D" => collect(t_ramp), "u_D" => collect(u_ramp)])
    truth = unflatten(working, pspec, theta_truth)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig, mask=mask)

    term = RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)
    reg = Regularizer([term]; weight=1e-3)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    theta_start = theta_truth .+ 0.02 .* (1:length(theta_truth))
    return problem, theta_start
end

# ------------------------------------------------------------------------------------------
# OP1, OP2, OP3: the three tight optimizers recover P1's generating parameters
# ------------------------------------------------------------------------------------------

@testset "OP1: LBFGS recovers the generating parameters (P1)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    result = solve(problem, LBFGS(); start=start)
    errT = abs(result.values["T_D"][1] - _OP_T_TRUTH) / _OP_T_TRUTH
    errU = abs(result.values["u_D"][1] - _OP_U_TRUTH) / abs(_OP_U_TRUTH)
    @info "OP1" errT errU result.status result.iterations
    @test errT < 1e-6
    @test errU < 1e-6
end

@testset "OP2: LevenbergMarquardt recovers the generating parameters (P1)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    result = solve(problem, LevenbergMarquardt(); start=start)
    errT = abs(result.values["T_D"][1] - _OP_T_TRUTH) / _OP_T_TRUTH
    errU = abs(result.values["u_D"][1] - _OP_U_TRUTH) / abs(_OP_U_TRUTH)
    @info "OP2" errT errU result.status result.iterations
    @test errT < 1e-6
    @test errU < 1e-6
end

@testset "OP3: noisy_descent_lbfgs recovers the generating parameters (P1)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    seq = OptimizerSequence(NoisyGradientDescent(steps=300, seed=20260907), LBFGS())
    result = solve(problem, seq; start=start)
    errT = abs(result.values["T_D"][1] - _OP_T_TRUTH) / _OP_T_TRUTH
    errU = abs(result.values["u_D"][1] - _OP_U_TRUTH) / abs(_OP_U_TRUTH)
    @info "OP3" errT errU result.status result.iterations
    @test errT < 1e-6
    @test errU < 1e-6
end

# ------------------------------------------------------------------------------------------
# OP4: L-BFGS and LM agree on a regularized ten-bin problem
# ------------------------------------------------------------------------------------------

@testset "OP4: LBFGS and LM agree on a regularized problem (P2)" begin
    problem, start = _op_p2_problem()
    # Step 3.C.2 (D30): the shipped LBFGS default now stops on a per-pixel objective-decrease
    # criterion, deliberately much looser than the old g_abstol = 1e-8, so it no longer lands at
    # the same bitwise-tight minimizer LM's own (untouched, still tight) g_abstol/x_reltol do.
    # OP4 asks whether two different optimizers agree on the same minimum, not whether the
    # shipped default is tight -- that question is CR1's -- so it runs LBFGS with an explicitly
    # tight criterion here (rule 11 zero: "OP rows updated where a default moved", the Files list
    # for this step).
    tight = LBFGS(; f_abstol_per_pixel=1e-12, g_abstol_per_pixel=1e-12, patience=20, iterations=5000)
    r1 = solve(problem, tight; start=start)
    r2 = solve(problem, LevenbergMarquardt(); start=start)
    relerr = maximum(abs.(r1.theta .- r2.theta) ./ max.(abs.(r2.theta), 1.0))
    @info "OP4" relerr r1.status r2.status
    @test isapprox(r1.theta, r2.theta; rtol=1e-6, atol=1e-6)
end

# ------------------------------------------------------------------------------------------
# OP5 to OP9: NoisyGradientDescent alone (P1)
# ------------------------------------------------------------------------------------------

@testset "OP5: zero noise is plain gradient descent (P1)" begin
    problem = _op_p1_problem()
    start = _op_p1_start(; t_frac=1.05, u_frac=0.9)(problem)
    # Step 3.C.3 (D38): under the *normalized* step the shipped default step_size is a fixed
    # length in y, chosen (measurement 8) for real progress on realistically-sized problems in a
    # bounded step budget -- it is not "small enough" in the sense this row needs, and no fixed
    # step_size can be, once the run gets within O(step_size) of the minimum: a normalized step
    # cannot shrink the way the old raw-gradient*alpha step did, so it eventually overshoots and
    # the objective increases at that step (measured: alpha = 5e-4 already violates monotonicity
    # once, at step 199 of 200, on this fixture). OP5's own question is whether zero-noise descent
    # is monotone for *a* small enough step size, not whether the shipped default is that size --
    # the same "OP rows updated where a default moved" license Section 6's Files list gives OP4 --
    # so this row keeps its own small, explicit step_size (chosen with margin below the smallest
    # value measured to violate monotonicity) instead of the shipped default; the assertion itself
    # is unchanged.
    opt = NoisyGradientDescent(noise=0.0, steps=200, step_size=2e-4)
    result = solve(problem, opt; start=start)
    objs = [t.objective for t in result.trajectory]
    @test all(diff(objs) .< 0)
end

@testset "OP6: a fixed seed is bitwise reproducible (P1)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    opt = NoisyGradientDescent(seed=20260907, steps=200)
    r1 = solve(problem, opt; start=start)
    r2 = solve(problem, opt; start=start)
    @test r1.theta == r2.theta
    @test r1.objective === r2.objective
    @test length(r1.trajectory) == length(r2.trajectory)
    @test all(a === b for (a, b) in zip(r1.trajectory, r2.trajectory))
end

@testset "OP7: the returned point is the best seen, not the last (P1)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    opt = NoisyGradientDescent(noise=5.0, noise_decay=5.0, steps=100, seed=20260907)
    result = solve(problem, opt; start=start)
    best = minimum(t.objective for t in result.trajectory)
    @test result.objective === best
    @test result.objective < result.trajectory[end].objective
end

@testset "OP8: the explorer reaches the basin (P1)" begin
    problem = _op_p1_problem()
    pspec = problem.pspec
    v_T, v_u = pspec.variables[1], pspec.variables[2]
    start = flatten(pspec, ["T_D" => v_T.lower + 1e-3 * (v_T.upper - v_T.lower),
        "u_D" => v_u.upper - 1e-3 * (v_u.upper - v_u.lower)])
    chi2_start = chi2(problem, start)
    opt = NoisyGradientDescent(seed=20260907)
    result = solve(problem, opt; start=start)
    @info "OP8" chi2_start result.chi2 (result.chi2 / chi2_start)
    @test result.chi2 <= 1e-2 * chi2_start
end

@testset "OP9: every iterate respects its bounds (P1, forced exploration)" begin
    problem = _op_p1_problem()
    pspec = problem.pspec
    v_T, v_u = pspec.variables[1], pspec.variables[2]
    start = flatten(pspec, ["T_D" => v_T.lower + 1e-3 * (v_T.upper - v_T.lower),
        "u_D" => v_u.upper - 1e-3 * (v_u.upper - v_u.lower)])

    opt = NoisyGradientDescent(noise=5.0, noise_decay=20.0, steps=300, seed=20260907)
    y_trace = Vector{Float64}[]
    solve(problem, opt; start=start, y_trace=y_trace)

    @test length(y_trace) == opt.steps
    for y in y_trace
        xT = _from_unconstrained(y[1], v_T.lower, v_T.upper)
        xu = _from_unconstrained(y[2], v_u.lower, v_u.upper)
        @test v_T.lower < xT < v_T.upper
        @test v_u.lower < xu < v_u.upper
    end
end

# ------------------------------------------------------------------------------------------
# OP19: the shipped default actually explores (chunk 10 carried item 1)
# ------------------------------------------------------------------------------------------

# P1's own build (D46), but with sigma from a realistic noise model instead of D46's unit sigma:
# this is D29's "realistic, 1 bin" fixture, the one that showed the old 1e-2 default never moves
# either weighting while 1e-8 does.
function _op_p1_realistic_problem()
    expt = _op_p1_expt()
    pspec = _op_p1_pspec(expt)
    working = _op_working(expt, pspec)
    theta_truth = flatten(pspec, ["T_D" => _OP_T_TRUTH, "u_D" => _OP_U_TRUTH])
    truth = unflatten(working, pspec, theta_truth)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig)
    return FitProblem(working, pspec, data)
end

@testset "OP19: the shipped default actually explores (realistic sigma, no override)" begin
    problem = _op_p1_realistic_problem()
    start = _op_p1_start()(problem)
    chi2_start = chi2(problem, start)
    opt = NoisyGradientDescent()   # no step_size override: this is the shipped default
    result = solve(problem, opt; start=start)
    @info "OP19" chi2_start result.chi2 (result.chi2 / chi2_start)
    @test result.chi2 <= 1e-2 * chi2_start
end
# Rewritten at Step 3.C.3 (D38): step_size is now a step *length* in y, chosen by measurement 8's
# sweep (dev/reports/phase3C_report.md), not a multiplier on the raw gradient. This is the row
# that catches a shipped default which saturates on its very first step -- precisely EPW's own
# LBFGS failure (D36), on a different optimizer: chunk 10 found the old raw-gradient default
# overshot P1's very first step into the logit's saturated region, and this row's whole purpose is
# to never let that regress silently. The margin (chi2 falls to at most 1e-2 of its start) is the
# same bar the old (pre-normalization) row used; it is loose on purpose -- OP8's own tighter 1e-2
# bound on a *better*-conditioned start already exercises the explorer's accuracy, and this row's
# job is only to confirm the shipped default moves the fit meaningfully, not to pin its exact
# endpoint.

# ------------------------------------------------------------------------------------------
# OP10: OptimizerSequence warm-starts from the best, not the last
# ------------------------------------------------------------------------------------------

@testset "OP10: OptimizerSequence warm-starts from the best, not the last" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    ngd = NoisyGradientDescent(noise=5.0, noise_decay=5.0, steps=50, seed=20260907)
    result1 = solve(problem, ngd; start=start)
    @test result1.objective < result1.trajectory[end].objective   # precondition: best != last

    seq = OptimizerSequence(ngd, LBFGS())
    result_seq = solve(problem, seq; start=start)
    n1 = length(result1.trajectory)
    @test result_seq.trajectory[n1+1].objective === result1.objective
end

# ------------------------------------------------------------------------------------------
# OP11, OP12: the four procedure names
# ------------------------------------------------------------------------------------------

@testset "OP11: the four procedure names build" begin
    @test _build_procedure("lbfgs") isa LBFGS
    @test _build_procedure("levenberg_marquardt") isa LevenbergMarquardt
    @test _build_procedure("noisy_descent") isa NoisyGradientDescent

    seq = _build_procedure("noisy_descent_lbfgs")
    @test seq isa OptimizerSequence
    @test seq.members[1] isa NoisyGradientDescent
    @test seq.members[2] isa LBFGS

    seq2 = _build_procedure(["lbfgs", "levenberg_marquardt"])
    @test seq2 isa OptimizerSequence
    @test seq2.members[1] isa LBFGS
    @test seq2.members[2] isa LevenbergMarquardt
end

@testset "OP12: an unknown procedure is a loud error" begin
    @test_throws "unknown procedure" _build_procedure("bogus")
end

# ------------------------------------------------------------------------------------------
# OP13: the Optim call shape
# ------------------------------------------------------------------------------------------

@testset "OP13: the Optim call shape (LBFGS)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    result = solve(problem, LBFGS(); start=start)
    @test result.f_calls > 0
    @test result.g_calls > 0
    @test length(result.trajectory) == result.iterations + 1
    @test result.status in (:gradient_norm, :f_tolerance, :x_tolerance, :iterations)
end

# ------------------------------------------------------------------------------------------
# OP14: LM's CGLS solves the damped normal equations
# ------------------------------------------------------------------------------------------

@testset "OP14: LM's CGLS solves the damped system (P2)" begin
    problem, theta = _op_p2_problem()
    ntheta = problem.pspec.ntheta

    r = _full_residuals(problem, theta)
    Jac = jacobian(problem, theta)
    gvec = zeros(ntheta)
    jtvp!(gvec, problem, Jac, r)
    mu = 1e-2

    delta, niter = _cgls(problem, Jac, r, gvec, mu, 1e-6, 100)

    Jdense = ForwardDiff.jacobian(th -> _full_residuals(problem, th), theta)
    resid = transpose(Jdense) * (Jdense * delta) .+ mu .* delta .+ gvec
    relerr = norm(resid) / norm(gvec)
    @info "OP14" relerr niter
    @test relerr <= 1e-5
end

# ------------------------------------------------------------------------------------------
# OP15: LM's mu update (Nielsen's rule)
# ------------------------------------------------------------------------------------------

@testset "OP15: LM's mu update -- a rejected step" begin
    problem = _op_p1_problem()
    start = _op_p1_start(; t_frac=3.0, u_frac=1.0)(problem)  # far enough that the GN step overshoots
    mu_trace = Float64[]
    opt = LevenbergMarquardt(iterations=2, mu_init=1e-8)
    result = solve(problem, opt; start=start, mu_trace=mu_trace)
    @info "OP15 rejected" mu_trace result.theta start
    @test length(result.trajectory) == 1   # no accept: the one entry is the start
    @test result.theta == start
    @test mu_trace[2] == mu_trace[1] * 2.0   # rejected: mu *= nu, nu started at 2.0
end

@testset "OP15: LM's mu update -- an accepted step" begin
    problem = _op_p1_problem()
    start = _op_p1_start(; t_frac=1.02, u_frac=1.0)(problem)  # close, conservative step
    mu_trace = Float64[]
    opt = LevenbergMarquardt(iterations=2)
    result = solve(problem, opt; start=start, mu_trace=mu_trace)
    @info "OP15 accepted" mu_trace result.theta start
    @test length(result.trajectory) >= 2
    @test result.theta != start
    @test mu_trace[2] <= mu_trace[1]    # accepted: mu *= max(1/3, 1-(2rho-1)^3) <= mu
end

# ------------------------------------------------------------------------------------------
# OP16: LM is seeded
# ------------------------------------------------------------------------------------------

@testset "OP16: LM is seeded (bitwise reproducible)" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    opt = LevenbergMarquardt(seed=20260908)
    r1 = solve(problem, opt; start=start)
    r2 = solve(problem, opt; start=start)
    @test r1.theta == r2.theta
    @test r1.objective === r2.objective
    @test all(a === b for (a, b) in zip(r1.trajectory, r2.trajectory))
end

# ------------------------------------------------------------------------------------------
# OP17: solve re-throws a RationalBackground pole
# ------------------------------------------------------------------------------------------

@testset "OP17: solve re-throws a RationalBackground pole" begin
    template = Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
    trace = PlasmaTrace(template; t_ns=[0.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=24)) .* 1e-9
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(RationalBackground(numerator=(20.0, 2.0), denominator=(0.1,)),))

    v_b1 = FittedVariable("b1"; bounds=(-5.0, 5.0), init=0.1)
    pspec = ParameterSpec(expt; variables=[v_b1], bindings=["b1" => (1, :denominator, 1)])
    working = _op_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=ones(size(image)))
    problem = FitProblem(working, pspec, data)

    bad_start = flatten(pspec, ["b1" => 2.0])   # |b1| = 2 > 1: the pole sits inside s in [-1, 1]
    @test_throws "RationalBackground" solve(problem, LBFGS(); start=bad_start)
end

# ------------------------------------------------------------------------------------------
# OP18: show
# ------------------------------------------------------------------------------------------

@testset "OP18: show" begin
    @test occursin("LBFGS", sprint(show, LBFGS()))
    @test occursin("LevenbergMarquardt", sprint(show, LevenbergMarquardt()))
    @test occursin("NoisyGradientDescent", sprint(show, NoisyGradientDescent()))
    @test occursin("OptimizerSequence", sprint(show, OptimizerSequence(NoisyGradientDescent(), LBFGS())))

    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    result = solve(problem, LBFGS(); start=start)
    @test occursin("FitResult", sprint(show, result))
    str = sprint(show, MIME("text/plain"), result)
    @test occursin("lbfgs", str)
end

# ------------------------------------------------------------------------------------------
# CR1 to CR7: the convergence criterion, the caps and the loud status (Step 3.C.2, D30-D34)
# ------------------------------------------------------------------------------------------

@testset "CR1: the primary criterion is observable in the trajectory (P2)" begin
    problem, start = _op_p2_problem()
    opt = LBFGS()
    result = solve(problem, opt; start=start)
    @info "CR1" result.status result.converged result.iterations
    @test result.status == :f_tolerance
    @test result.converged
    threshold = opt.f_abstol_per_pixel * problem.npixels
    traj = result.trajectory
    n = length(traj)
    @test n > opt.patience
    for k in (n-opt.patience+1):n
        dO = traj[k-1].objective - traj[k].objective
        @test dO < threshold
    end
end

@testset "CR2: patience arithmetic (successive_f_tol = patience - 1)" begin
    problem, start = _op_p2_problem()
    r4 = solve(problem, LBFGS(; patience=4); start=start)
    r5 = solve(problem, LBFGS(; patience=5); start=start)
    @info "CR2" r4.iterations r5.iterations
    @test r5.iterations >= r4.iterations
end

@testset "CR3: the criterion is per pixel" begin
    problem, _ = _op_p2_problem()
    nlambda, nt = size(problem.data.image)
    mask = trues(nlambda, nt)
    mask[(nlambda÷2+1):end, :] .= false
    problem_half, _ = _op_p2_problem_masked(mask)
    @test problem_half.npixels == problem.npixels ÷ 2

    opt = LBFGS()
    f_abstol_full = opt.f_abstol_per_pixel * problem.npixels
    f_abstol_half = opt.f_abstol_per_pixel * problem_half.npixels
    @info "CR3" problem.npixels problem_half.npixels f_abstol_full f_abstol_half
    @test f_abstol_full / f_abstol_half == problem.npixels / problem_half.npixels
end

@testset "CR4: the saturation guard" begin
    problem = _op_p1_problem()
    pspec = problem.pspec
    v_T, v_u = pspec.variables[1], pspec.variables[2]
    start = flatten(pspec, ["T_D" => v_T.lower + 1e-20 * (v_T.upper - v_T.lower),
        "u_D" => v_u.upper - 1e-20 * (v_u.upper - v_u.lower)])
    result = solve(problem, LBFGS(); start=start)
    @info "CR4" result.status result.converged result.termination result.iterations
    @test result.status == :saturated
    @test !result.converged
    @test result.termination == :GradientNorm   # Optim.g_converged fired underneath
end

@testset "CR5: a near-bound stop by the primary criterion is still converged" begin
    problem = _op_p1_problem()
    pspec = problem.pspec
    v_T, v_u = pspec.variables[1], pspec.variables[2]
    start = flatten(pspec, ["T_D" => v_T.lower + 1e-3 * (v_T.upper - v_T.lower),
        "u_D" => v_u.upper - 1e-3 * (v_u.upper - v_u.lower)])
    opt = LBFGS(; f_abstol_per_pixel=1.0e6, patience=1)   # trivially satisfied every iteration
    result = solve(problem, opt; start=start)
    @info "CR5" result.status result.converged result.iterations
    @test result.status == :f_tolerance
    @test result.converged
end

@testset "CR6: :converged_at_start, and :saturated wins when the start is also saturated" begin
    problem = _op_p1_problem()
    start = _op_p1_start()(problem)
    result0 = solve(problem, LBFGS(); start=start)
    @test result0.status != :saturated

    result1 = solve(problem, LBFGS(); start=result0.theta)
    @info "CR6 (normal)" result1.status result1.iterations length(result1.trajectory)
    @test result1.status == :converged_at_start
    @test result1.iterations == 0
    @test length(result1.trajectory) == 1

    pspec = problem.pspec
    v_T, v_u = pspec.variables[1], pspec.variables[2]
    sat_start = flatten(pspec, ["T_D" => v_T.lower + 1e-20 * (v_T.upper - v_T.lower),
        "u_D" => v_u.upper - 1e-20 * (v_u.upper - v_u.lower)])
    sat_result0 = solve(problem, LBFGS(); start=sat_start)
    @test sat_result0.status == :saturated

    sat_result1 = solve(problem, LBFGS(); start=sat_result0.theta)
    @info "CR6 (saturated wins)" sat_result1.status sat_result1.iterations
    @test sat_result1.status == :saturated
end

# ------------------------------------------------------------------------------------------
# PC1 to PC5: the preconditioner and its guard (Step 3.C.3, D36-D37)
# ------------------------------------------------------------------------------------------

@testset "PC1: preconditioning reaches the same minimizer (P2)" begin
    problem, start = _op_p2_problem()
    # Same reasoning as OP4 (Section 6's own "OP rows updated where a default moved" license,
    # exercised here for a *new* row): the question is whether two descent paths reach the same
    # minimum, not whether the shipped (loose) default does -- so both runs use a tight criterion,
    # otherwise the shipped default's own patience/f_abstol slack would dominate any comparison at
    # 1e-6 and the row would be testing the criterion's looseness, not the preconditioner.
    tight_on = LBFGS(; f_abstol_per_pixel=1e-12, g_abstol_per_pixel=1e-12, patience=20,
        iterations=5000, precondition=true)
    tight_off = LBFGS(; f_abstol_per_pixel=1e-12, g_abstol_per_pixel=1e-12, patience=20,
        iterations=5000, precondition=false)
    r_on = solve(problem, tight_on; start=start)
    r_off = solve(problem, tight_off; start=start)
    relerr = maximum(abs.(r_on.theta .- r_off.theta) ./ max.(abs.(r_off.theta), 1.0))
    @info "PC1" relerr r_on.status r_off.status r_on.iterations r_off.iterations
    @test isapprox(r_on.theta, r_off.theta; rtol=1e-6, atol=1e-6)
end

@testset "PC2: preconditioning costs no correctness (P2)" begin
    problem, start = _op_p2_problem()
    opt = LBFGS()   # precondition = true, the shipped default
    r_on = solve(problem, opt; start=start)
    r_off = solve(problem, LBFGS(; precondition=false); start=start)
    threshold = opt.f_abstol_per_pixel * problem.npixels
    @info "PC2" r_on.chi2 r_off.chi2 (r_on.chi2 - r_off.chi2) threshold r_on.status r_off.status
    @test r_on.chi2 <= r_off.chi2 + threshold
end

@testset "PC3: the floor binds and is relative" begin
    # A deliberately near-singular diagonal, built directly rather than from a real problem: one
    # entry near zero, one exactly zero (EPW's own failure mode, chunk 5), one at the scale of the
    # rest.
    d = [1e-20, 0.0, 1.0, 1e10]
    dg = _gauss_newton_floor(d)
    @test dg !== nothing
    Pinv = 1 ./ (2 .* dg)
    @info "PC3" dg Pinv dynamic_range = (maximum(dg) / minimum(dg))
    @test all(isfinite, Pinv)
    @test maximum(dg) / minimum(dg) <= 1 / PRECOND_FLOOR_REL
    @test all(dg .>= d)   # the floor only ever raises an entry, never lowers one

    # The two degenerate cases (D37): every entry zero, and a non-finite entry present.
    @test _gauss_newton_floor(zeros(5)) === nothing
    @test _gauss_newton_floor([1.0, NaN, 2.0]) === nothing
    @test _gauss_newton_floor([1.0, Inf, 2.0]) === nothing
end

@testset "PC4: preconditioned CGLS reaches the same delta (P2)" begin
    problem, theta = _op_p2_problem()
    ntheta = problem.pspec.ntheta

    r = _full_residuals(problem, theta)
    Jac = jacobian(problem, theta)
    gvec = zeros(ntheta)
    jtvp!(gvec, problem, Jac, r)
    mu = 1e-2

    d = _gauss_newton_diagonal(problem, Jac)
    dg = _gauss_newton_floor(d)
    @test dg !== nothing

    delta_off, niter_off = _cgls(problem, Jac, r, gvec, mu, 1e-10, 2000)
    delta_on, niter_on = _cgls(problem, Jac, r, gvec, mu, 1e-10, 2000; precond=dg)

    relerr = maximum(abs.(delta_on .- delta_off) ./ max.(abs.(delta_off), 1.0))
    @info "PC4" relerr niter_off niter_on
    @test isapprox(delta_on, delta_off; rtol=1e-8, atol=1e-8)
end

@testset "PC5: precond_refresh is honoured" begin
    problem, start = _op_p2_problem()
    trace_often = []
    r_often = solve(problem, LBFGS(; precond_refresh=1); start=start, precond_trace=trace_often)
    trace_rarely = []
    r_rarely = solve(problem, LBFGS(; precond_refresh=10_000); start=start, precond_trace=trace_rarely)
    @info "PC5" r_often.converged r_rarely.converged r_often.iterations r_rarely.iterations n_rebuilds_refresh1 = length(trace_often) n_rebuilds_refresh10000 = length(trace_rarely)
    @test r_often.converged
    @test r_rarely.converged
    @test length(trace_often) != length(trace_rarely)
    @test length(trace_often) > length(trace_rarely)
end
