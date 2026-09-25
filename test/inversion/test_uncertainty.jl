using Test
using OTIS
using OTIS: jvp!, jtvp!, jacobian, _full_residuals, gradient!, _free_entries, _theta_layout,
    _to_unconstrained, _from_unconstrained, _dx_dy, _chain_class, BOUND_PROXIMITY_TOL
import Random
using LinearAlgebra: dot, cholesky, Symmetric

# UN1-UN6, CS1-CS3 against dev/phase4A_instructions.md Section 10.1's table (spec D43, D45, D46),
# gated "uncertainty". The fixture is D35's MC fixture (test_joint.jl), copied here and NOT shared
# across the @safetestset boundary (chunk 6's own rule, inherited from Phase 3), with `dlam_b`
# removed: channel b's WavelengthCalibration stays at 0.01 nm, unbound. ntheta = 8.

# ------------------------------------------------------------------------------------------
# The fixture (D35, copied, dlam_b removed)
# ------------------------------------------------------------------------------------------

function _un_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _un_experiments()
    trace = PlasmaTrace(_un_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_a = collect(range(262.9, 263.6; length=16)) .* 1e-9
    lambda_b = collect(range(262.2, 262.9; length=12)) .* 1e-9
    expt_a = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_a,
        effects=(SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6), ConstantBackground(level=20.0)))
    expt_b = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_b,
        effects=(WavelengthCalibration(lambda_offset_nm=0.01), SpectralIRF(fwhm_nm=0.12),
            Sensitivity(scale=2.0e-6), ConstantBackground(level=15.0)))
    return expt_a, expt_b
end

# T_D (time-varying, shared plasma), T_e (scalar, shared plasma), scale_a/scale_b (each channel's
# own Sensitivity), bg (bound into both channels' ConstantBackground.level). No dlam_b: channel
# b's WavelengthCalibration is unbound (chunk 9's own fixture, Section 10.1).
function _un_pspec(expt_a::Experiment, expt_b::Experiment)
    sens_a = findfirst(e -> _chain_class(e) === :Sensitivity, expt_a.effects)
    bg_a = findfirst(e -> e isa ConstantBackground, expt_a.effects)
    sens_b = findfirst(e -> _chain_class(e) === :Sensitivity, expt_b.effects)
    bg_b = findfirst(e -> e isa ConstantBackground, expt_b.effects)

    v_TD = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0)
    v_scale_a = FittedVariable("scale_a"; bounds=(1e-7, 1e-5), init=1e-6)
    v_scale_b = FittedVariable("scale_b"; bounds=(1e-7, 1e-5), init=1e-6)
    v_bg = FittedVariable("bg"; bounds=(1.0, 60.0), init=20.0)

    return ParameterSpec((expt_a, expt_b);
        variables=[v_TD, v_Te, v_scale_a, v_scale_b, v_bg],
        bindings=["T_D" => "ions.D.T_eV", "T_e" => "electrons.e.T_eV",
            "scale_a" => (channel=1, effect=sens_a, field=:scale),
            "scale_b" => (channel=2, effect=sens_b, field=:scale),
            "bg" => (channel=1, effect=bg_a, field=:level),
            "bg" => (channel=2, effect=bg_b, field=:level)])
end

_un_regularizer(pspec::ParameterSpec) = Regularizer(
    [RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)];
    weight=1e-3)

function _un_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

function _un_full_problem()
    expt_a, expt_b = _un_experiments()
    pspec = _un_pspec(expt_a, expt_b)
    reg = _un_regularizer(pspec)
    working_a = _un_working(expt_a, pspec)
    working_b = _un_working(expt_b, pspec)
    theta0 = flatten(pspec)
    rng = Random.Xoshiro(20260923)
    truth_a = unflatten(working_a, pspec, theta0; channel=1)
    data_a = synthesize(truth_a, GaussianNoise(a=1.0, b=0.02); rng=rng)
    truth_b = unflatten(working_b, pspec, theta0; channel=2)
    data_b = synthesize(truth_b, GaussianNoise(a=1.0, b=0.02); rng=rng)
    return working_a, working_b, pspec, reg, data_a, data_b, theta0
end

function _un_joint(weights::Vector{Float64}=[0.3, 2.0]; regularizer_on::Bool=true)
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _un_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); weights=weights, regularizer=regularizer_on ? reg : nothing)
    return jp, theta0
end

# Channel a alone, carrying its own regularizer (UN1's "channel a alone (with the regularizer)").
function _un_channel_a_alone()
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _un_full_problem()
    return FitProblem(working_a, pspec, data_a; channel=1, name="a", regularizer=reg), theta0
end

# A second fixture, D35's original pspec with dlam_b bound to channel b's calibration offset --
# CS2's third loud error needs a spec that actually binds the offset chunk 9's own fixture leaves
# free.
function _un_pspec_with_dlam(expt_a::Experiment, expt_b::Experiment)
    sens_a = findfirst(e -> _chain_class(e) === :Sensitivity, expt_a.effects)
    bg_a = findfirst(e -> e isa ConstantBackground, expt_a.effects)
    calib_b = findfirst(e -> _chain_class(e) === :WavelengthCalibration, expt_b.effects)
    sens_b = findfirst(e -> _chain_class(e) === :Sensitivity, expt_b.effects)
    bg_b = findfirst(e -> e isa ConstantBackground, expt_b.effects)

    v_TD = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0)
    v_scale_a = FittedVariable("scale_a"; bounds=(1e-7, 1e-5), init=1e-6)
    v_scale_b = FittedVariable("scale_b"; bounds=(1e-7, 1e-5), init=1e-6)
    v_dlam_b = FittedVariable("dlam_b"; bounds=(-0.2e-9, 0.2e-9), init=0.01e-9)
    v_bg = FittedVariable("bg"; bounds=(1.0, 60.0), init=20.0)

    return ParameterSpec((expt_a, expt_b);
        variables=[v_TD, v_Te, v_scale_a, v_scale_b, v_dlam_b, v_bg],
        bindings=["T_D" => "ions.D.T_eV", "T_e" => "electrons.e.T_eV",
            "scale_a" => (channel=1, effect=sens_a, field=:scale),
            "scale_b" => (channel=2, effect=sens_b, field=:scale),
            "dlam_b" => (channel=2, effect=calib_b, field=:offset),
            "bg" => (channel=1, effect=bg_a, field=:level),
            "bg" => (channel=2, effect=bg_b, field=:level)])
end

function _un_joint_with_dlam()
    expt_a, expt_b = _un_experiments()
    pspec = _un_pspec_with_dlam(expt_a, expt_b)
    reg = _un_regularizer(pspec)
    working_a = _un_working(expt_a, pspec)
    working_b = _un_working(expt_b, pspec)
    theta0 = flatten(pspec)
    rng = Random.Xoshiro(20260923)
    truth_a = unflatten(working_a, pspec, theta0; channel=1)
    data_a = synthesize(truth_a, GaussianNoise(a=1.0, b=0.02); rng=rng)
    truth_b = unflatten(working_b, pspec, theta0; channel=2)
    data_b = synthesize(truth_b, GaussianNoise(a=1.0, b=0.02); rng=rng)
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); weights=[0.3, 2.0], regularizer=reg)
    return jp, theta0
end

# The dense transcription from jvp! on unit vectors (MC4's own route).
function _un_dense_jvp(problem::AbstractFitProblem, Jac, ntheta::Int, nrows::Int)
    J_dense = zeros(nrows, ntheta)
    v = zeros(ntheta)
    out = zeros(nrows)
    for k in 1:ntheta
        fill!(v, 0.0)
        v[k] = 1.0
        jvp!(out, problem, Jac, v)
        J_dense[:, k] = out
    end
    return J_dense
end

# theta*: LM from theta0 .+ 0.02 .* (1:8), then polish (Section 10.1).
const _UN_JP, _UN_THETA0 = _un_joint()
const _UN_THETA_START = _UN_THETA0 .+ 0.02 .* (1:length(_UN_THETA0))
const _UN_R_LM = solve(_UN_JP, LevenbergMarquardt(); start=_UN_THETA_START)
const _UN_POLISH = polish(_UN_JP, _UN_R_LM)
const _UN_THETA_STAR = _UN_POLISH.result.theta

# ------------------------------------------------------------------------------------------
# UN1: normal_matrices against the dense transcription
# ------------------------------------------------------------------------------------------

@testset "UN1: normal_matrices against the dense transcription J" begin
    ntheta = _UN_JP.pspec.ntheta
    nrows = _UN_JP.npixels + _UN_JP.regularizer.nrows
    Jac = jacobian(_UN_JP, _UN_THETA_STAR)
    J_dense = _un_dense_jvp(_UN_JP, Jac, ntheta, nrows)
    JtJ_dense = J_dense' * J_dense
    Jd_dense = @view J_dense[1:_UN_JP.npixels, :]
    JdtJd_dense = Jd_dense' * Jd_dense

    JtJ, JdtJd = normal_matrices(_UN_JP, _UN_THETA_STAR)
    @test isapprox(JtJ, JtJ_dense; rtol=1e-12)
    @test isapprox(JdtJd, JdtJd_dense; rtol=1e-12)

    ch_a, theta0a = _un_channel_a_alone()
    theta_a = theta0a .+ 0.02 .* (1:length(theta0a))
    ntheta_a = ch_a.pspec.ntheta
    nrows_a = ch_a.npixels + ch_a.regularizer.nrows
    Jac_a = jacobian(ch_a, theta_a)
    J_dense_a = _un_dense_jvp(ch_a, Jac_a, ntheta_a, nrows_a)
    JtJ_dense_a = J_dense_a' * J_dense_a
    Jd_dense_a = @view J_dense_a[1:ch_a.npixels, :]
    JdtJd_dense_a = Jd_dense_a' * Jd_dense_a

    JtJ_a, JdtJd_a = normal_matrices(ch_a, theta_a)
    @test isapprox(JtJ_a, JtJ_dense_a; rtol=1e-12)
    @test isapprox(JdtJd_a, JdtJd_dense_a; rtol=1e-12)
end

# ------------------------------------------------------------------------------------------
# UN2: no regularizer -- JdtJd == JtJ bitwise; sigma_samp == sigma_post
# ------------------------------------------------------------------------------------------

@testset "UN2: no regularizer, JdtJd == JtJ bitwise, sigma_samp agrees with sigma_post" begin
    jp_noreg, theta0 = _un_joint([0.3, 2.0]; regularizer_on=false)
    theta_start = theta0 .+ 0.02 .* (1:length(theta0))
    r = solve(jp_noreg, LevenbergMarquardt(); start=theta_start)
    pol = polish(jp_noreg, r)
    theta_star = pol.result.theta

    JtJ, JdtJd = normal_matrices(jp_noreg, theta_star)
    @test JtJ == JdtJd

    lap = laplace(jp_noreg, theta_star)
    for name in keys(lap.sigma_post)
        for i in eachindex(lap.sigma_post[name])
            a, b = lap.sigma_post[name][i], lap.sigma_samp[name][i]
            if isfinite(a)
                @test isapprox(a, b; rtol=1e-10)
            else
                @test isnan(b)
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# UN3: gn_decrease against the dense formula; polish's own history
# ------------------------------------------------------------------------------------------

@testset "UN3: gn_decrease against the dense formula; polish reaches tol within 10 steps" begin
    ntheta = _UN_JP.pspec.ntheta
    JtJ, _ = normal_matrices(_UN_JP, _UN_THETA_STAR)
    f = _free_entries(_UN_JP.pspec, _UN_THETA_STAR)
    g = zeros(ntheta)
    gradient!(g, _UN_JP, _UN_THETA_STAR)
    gf = g[f]
    dense_decrease = dot(gf, cholesky(Symmetric(JtJ[f, f])) \ gf) / 4
    @test isapprox(gn_decrease(_UN_JP, _UN_THETA_STAR), dense_decrease; rtol=1e-10)

    r10 = solve(_UN_JP, LBFGS(iterations=10); start=_UN_THETA_START)
    pol = polish(_UN_JP, r10)

    @test length(pol.decreases) >= 2
    @test pol.decreases[end] <= 1e-6
    @test length(pol.decreases) <= 10
    @test all(pol.decreases .>= 0)
    @info "UN3 decrease history" pol.decreases

    # at least one step actually taken: the trajectory grew past r10's own last entry
    @test length(pol.result.trajectory) > length(r10.trajectory)

    # objective non-increasing at every recorded (accepted) step
    objs = [row.objective for row in pol.result.trajectory]
    @test all(diff(objs) .<= 0)

    # polish keeps the original solve's own status/converged/iterations/f_calls/g_calls/termination
    @test pol.result.status == r10.status
    @test pol.result.converged == r10.converged
    @test pol.result.iterations == r10.iterations
    @test pol.result.f_calls == r10.f_calls
    @test pol.result.g_calls == r10.g_calls
    @test pol.result.termination == r10.termination
    @test pol.result.elapsed >= r10.elapsed
end

# ------------------------------------------------------------------------------------------
# UN4: laplace's formulas; sigma_samp != sigma_post; the near-bound case
# ------------------------------------------------------------------------------------------

@testset "UN4: laplace's formulas, discrimination from sigma_post, the near-bound case" begin
    JtJ, JdtJd = normal_matrices(_UN_JP, _UN_THETA_STAR)
    f = _free_entries(_UN_JP.pspec, _UN_THETA_STAR)
    C_dense = inv(cholesky(Symmetric(JtJ[f, f])))
    cov_samp_dense = C_dense * JdtJd[f, f] * C_dense

    lap = laplace(_UN_JP, _UN_THETA_STAR)
    @test isapprox(lap.cov_y_samp[f, f], cov_samp_dense; rtol=1e-10)

    entry_var, entry_bin = _theta_layout(_UN_JP.pspec)
    pspec = _UN_JP.pspec
    for e in 1:pspec.ntheta
        f[e] || continue
        v = pspec.variables[entry_var[e]]
        pos = v.time_varying ? entry_bin[e] : 1
        x = _from_unconstrained(_UN_THETA_STAR[e], v.lower, v.upper)
        dxdy = abs(_dx_dy(x, v.lower, v.upper))
        @test isapprox(lap.sigma_samp[v.name][pos], dxdy * sqrt(lap.cov_y_samp[e, e]); rtol=1e-10)
        @test isapprox(lap.sigma_post[v.name][pos], dxdy * sqrt(lap.cov_y_post[e, e]); rtol=1e-10)
    end

    # sigma_samp != sigma_post by more than 1e-6 relative somewhere (regularizer on)
    worst = 0.0
    for name in keys(lap.sigma_post)
        for i in eachindex(lap.sigma_post[name])
            a, b = lap.sigma_post[name][i], lap.sigma_samp[name][i]
            isfinite(a) || continue
            worst = max(worst, abs(a - b) / abs(a))
        end
    end
    @test worst > 1e-6
    @info "UN4 sigma_samp vs sigma_post worst relative difference" worst

    # the near-bound case: T_D's bin-2 entry pinned within BOUND_PROXIMITY_TOL of its upper bound
    v_TD = pspec.variables[findfirst(v -> v.name == "T_D", pspec.variables)]
    entries2 = pspec.bin_entries[2]
    slots2 = pspec.bin_slots[2]
    idx = entries2[findfirst(k -> pspec.variables[slots2[k]].name == "T_D", eachindex(slots2))]
    x_nb = v_TD.upper - 1e-4 * (v_TD.upper - v_TD.lower)
    theta_nb = copy(_UN_THETA_STAR)
    theta_nb[idx] = _to_unconstrained(x_nb, v_TD.lower, v_TD.upper)

    lap_nb = laplace(_UN_JP, theta_nb; max_decrease=Inf)
    @test isnan(lap_nb.sigma_samp["T_D"][2])
    @test isnan(lap_nb.sigma_post["T_D"][2])
    @test lap_nb.near_bound["T_D"][2]
    @test !any(lap_nb.near_bound["T_D"][[1, 3, 4]])
    @test all(isfinite, lap_nb.sigma_samp["T_D"][[1, 3, 4]])
    for name in keys(lap_nb.near_bound)
        name == "T_D" && continue
        @test !any(lap_nb.near_bound[name])
    end
end

# ------------------------------------------------------------------------------------------
# UN5: the refusals
# ------------------------------------------------------------------------------------------

@testset "UN5: laplace and calibration_scan refuse at a non-minimum" begin
    @test_throws r"not at a minimum" laplace(_UN_JP, _UN_THETA_START)
    @test_throws r"polish first" calibration_scan(_UN_JP, _UN_THETA_START, [0.0, 0.01]; channel=2)
end

# ------------------------------------------------------------------------------------------
# UN6: LaplaceResult's layout; show
# ------------------------------------------------------------------------------------------

@testset "UN6: LaplaceResult's sigma_*/near_bound have FitResult.values' keys and lengths" begin
    lap = laplace(_UN_JP, _UN_THETA_STAR)
    var_names = Set(v.name for v in _UN_JP.pspec.variables if v.vary)
    @test Set(keys(lap.sigma_post)) == var_names
    @test Set(keys(lap.sigma_samp)) == var_names
    @test Set(keys(lap.near_bound)) == var_names
    for v in _UN_JP.pspec.variables
        v.vary || continue
        @test length(lap.sigma_post[v.name]) == length(_UN_POLISH.result.values[v.name])
        @test length(lap.sigma_samp[v.name]) == length(_UN_POLISH.result.values[v.name])
        @test length(lap.near_bound[v.name]) == length(_UN_POLISH.result.values[v.name])
    end

    io = IOBuffer()
    show(io, lap)
    @test !isempty(String(take!(io)))
    show(io, MIME"text/plain"(), lap)
    @test !isempty(String(take!(io)))
end

# ------------------------------------------------------------------------------------------
# CS1: the scan on channel 2
# ------------------------------------------------------------------------------------------

@testset "CS1: calibration_scan on channel 2 at [-0.01, 0.0, 0.01] nm" begin
    scan = calibration_scan(_UN_JP, _UN_THETA_STAR, [-0.01, 0.0, 0.01]; channel=2)

    @test scan.shifts_nm == [-0.01, 0.0, 0.01]
    @test scan.results[2].theta == _UN_THETA_STAR
    for name in keys(scan.delta)
        @test all(iszero, scan.delta[name][:, 2])
    end
    @test all(scan.decrease .<= 1e-6)
    @test scan.chi2[1] > scan.chi2[2] < scan.chi2[3]
    @info "CS1 chi2 at each shift" scan.chi2
end

# ------------------------------------------------------------------------------------------
# CS2: the loud errors
# ------------------------------------------------------------------------------------------

@testset "CS2: calibration_scan's loud errors" begin
    @test_throws r"0\.0" calibration_scan(_UN_JP, _UN_THETA_STAR, [-0.01, 0.01]; channel=2)
    @test_throws r"no WavelengthCalibration" calibration_scan(_UN_JP, _UN_THETA_STAR, [0.0, 0.01]; channel=1)

    jp_dlam, theta0_dlam = _un_joint_with_dlam()
    theta_start_dlam = theta0_dlam .+ 0.02 .* (1:length(theta0_dlam))
    r_dlam = solve(jp_dlam, LevenbergMarquardt(); start=theta_start_dlam)
    pol_dlam = polish(jp_dlam, r_dlam)
    @test_throws r"cannot also fit it" calibration_scan(jp_dlam, pol_dlam.result.theta, [0.0, 0.01]; channel=2)
end

# ------------------------------------------------------------------------------------------
# CS3: s_sys and the layout
# ------------------------------------------------------------------------------------------

@testset "CS3: s_sys == vec(maximum(abs, delta; dims = 2)); delta's shape" begin
    scan = calibration_scan(_UN_JP, _UN_THETA_STAR, [-0.01, 0.0, 0.01]; channel=2)
    for v in _UN_JP.pspec.variables
        @test size(scan.delta[v.name]) == (length(_UN_POLISH.result.values[v.name]), 3)
        @test scan.s_sys[v.name] == vec(maximum(abs, scan.delta[v.name]; dims=2))
    end
end
