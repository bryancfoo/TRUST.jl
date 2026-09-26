using Test
using OTIS
using OTIS: jvp!, jtvp!, jacobian, _full_residuals, gradient!, _free_entries, _theta_layout,
    _to_unconstrained, _from_unconstrained, _dx_dy, _chain_class, BOUND_PROXIMITY_TOL,
    _detrended_spread
import Random
using LinearAlgebra: dot, cholesky, Symmetric
using Statistics: median

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

# ------------------------------------------------------------------------------------------
# SX1-SX5, RE14 against dev/phase4A_instructions.md Section 10.2's table (spec D51, D52),
# gated "uncertainty". The SX fixture (deck-based, mktempdir(), noise-free data with
# sigma = sigma(GaussianNoise(; a = 121.0, b = 141.75), image)): SXA, a two-channel deck with
# T_D and u_D free and time-varying, T_e fixed; SXB, the same deck with T_e a free scalar bound
# to electrons.e.T_eV. Both cameras' true times coincide (channel b's t_shift_ns = 0.02 undoes
# its own recorded-axis offset), so the pooled fine axis is exactly the 12 shared slice times.
# ------------------------------------------------------------------------------------------

_sx_template() = Plasma(; n_e_cm3=3e19,
    electrons=[electrons(T_eV=400.0)],
    ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])

_sx_TD(t_ns) = 180.0 + 60.0 * t_ns
_sx_uD(t_ns) = 20.0 + 15.0 * t_ns
_sx_path(p::AbstractString) = replace(p, "\\" => "/")

function _sx_write(path::AbstractString, t_ns_recorded::AbstractVector, t_shift_ns::Real,
    lambda_lo::Real, lambda_hi::Real, nlambda::Integer, scale::Real, background::Bool)
    t_true_ns = t_ns_recorded .+ t_shift_ns
    trace = PlasmaTrace(_sx_template(); t_ns=collect(t_true_ns),
        leaves=("ions.D.T_eV" => _sx_TD, "ions.D.u_kms" => _sx_uD))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda = collect(range(lambda_lo, lambda_hi; length=nlambda)) .* 1e-9
    effects = background ? (Sensitivity(scale=scale), ConstantBackground(level=25.0)) :
              (Sensitivity(scale=scale),)
    expt = Experiment(; trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
    image = measure(expt)
    sig = sigma(GaussianNoise(; a=121.0, b=141.75), image)
    sd = StreakData(; lambda_m=lambda, t_s=t_ns_recorded .* 1e-9, image=image, sigma=sig)
    write_streak(path, sd; source="sx_test")
    return sd
end

_sx_write_a(path::AbstractString) = _sx_write(path, collect(range(0.05, 1.15; length=12)), 0.0,
    262.0, 264.5, 24, 1.0e-6, true)
_sx_write_b(path::AbstractString) = _sx_write(path, collect(range(0.03, 1.13; length=12)), 0.02,
    262.5, 264.0, 16, 2.0e-6, false)

# regions_block/intervals_block/te_value/te_var_block/regularizer_block let one builder make
# SXA, SXB, and the no-regularizer one-window deck SX4 needs.
function _sx_deck_text(path_a::AbstractString, path_b::AbstractString, outdir::AbstractString;
    regions_block::AbstractString="[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 3\n",
    intervals_block::AbstractString="",
    te_value::AbstractString="400.0", te_var_block::AbstractString="",
    regularizer_block::AbstractString="[regularizer]\nweight = 1.0\nterms = [\n  { select = [\"T_D\", \"u_D\"], operator = \"D2\", normalization = \"relative\" },\n]\n")
    return """
    deck_version = "otis-deck-2"
    channels = ["a", "b"]

    $regions_block
    $intervals_block

    [channel.a.data]
    file = "$(_sx_path(path_a))"
    [channel.a.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"
    [channel.a.instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [channel.b.data]
    file = "$(_sx_path(path_b))"
    t_shift_ns = 0.02
    [channel.b.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"
    [channel.b.instrument]
    sensitivity = { scale = 2.0e-6 }

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }
    $te_var_block

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = $te_value

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = "u_D"

    $regularizer_block

    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_sx_path(outdir))"
    """
end

function _sx_build(dir::AbstractString, text::AbstractString, deckname::AbstractString)
    path = joinpath(dir, deckname)
    write(path, text)
    deck = read_deck(path)
    problem, opt, start = OTIS.build_problem(deck)
    r = solve(problem, opt; start=start)
    pol = polish(problem, r)
    return deck, problem, pol.result
end

const _SX_DIR = mktempdir()
const _SX_PATH_A = joinpath(_SX_DIR, "sx_a.jld2")
const _SX_PATH_B = joinpath(_SX_DIR, "sx_b.jld2")
const _SX_SD_A = _sx_write_a(_SX_PATH_A)
const _SX_SD_B = _sx_write_b(_SX_PATH_B)

const _SX_TEXT_A = _sx_deck_text(_SX_PATH_A, _SX_PATH_B, joinpath(_SX_DIR, "out_a"))
const _SX_DECK_A, _SX_PROBLEM_A, _SX_RESULT_A = _sx_build(_SX_DIR, _SX_TEXT_A, "sxa_deck.toml")
const _SX_SCATTER_A = slice_scatter(_SX_DECK_A, _SX_RESULT_A)

const _SX_TEXT_B = _sx_deck_text(_SX_PATH_A, _SX_PATH_B, joinpath(_SX_DIR, "out_b");
    te_value="\"T_e\"", te_var_block="T_e = { bounds = [100.0, 1000.0], init = 380.0 }")
const _SX_DECK_B, _SX_PROBLEM_B, _SX_RESULT_B = _sx_build(_SX_DIR, _SX_TEXT_B, "sxb_deck.toml")
const _SX_SCATTER_B = slice_scatter(_SX_DECK_B, _SX_RESULT_B)

# The coarse chan_data, built the same way build_problem does internally, for SX1's own
# window-count cross-check (D48 item 2's own check, reused here rather than reimplemented).
const _SX_CHAN_LIST = OTIS._channel_list(_SX_DECK_A.sections)
const _SX_RAW, _SX_PROBES, _SX_COLLECTIONS = OTIS._read_channels(_SX_DECK_A.sections, _SX_CHAN_LIST, true)
const _SX_REGIONS = OTIS._build_averaging_regions(_SX_DECK_A.sections["fit_regions"])
const _SX_COARSE = OTIS._average_channels(_SX_RAW, _SX_CHAN_LIST, _SX_REGIONS)

@testset "SX1: the grid" begin
    sc = _SX_SCATTER_A
    @test sc.m == [4]
    @test length(sc.t_fine) == 12
    @test isapprox(sc.t_fine, collect(0.05:0.1:1.15) .* 1e-9; rtol=1e-12)
    @test sc.counts == ones(Int, 12, 2)
    @test sc.coarse_bin == repeat(1:3, inner=4)
    for c in 1:2
        for k in 1:3
            @test sum(sc.counts[sc.coarse_bin.==k, c]) == _SX_COARSE[c].meta["regions_window_counts"][k]
        end
    end
end

@testset "SX2: exact recovery, noise-free" begin
    sc = _SX_SCATTER_A
    t_fine_ns = sc.t_fine .* 1e9
    truth_TD = _sx_TD.(t_fine_ns)
    truth_uD = _sx_uD.(t_fine_ns)
    @test isapprox(sc.values["T_D"], truth_TD; rtol=1e-8)
    @test isapprox(sc.values["u_D"], truth_uD; rtol=1e-8)
    @test all(sc.decrease .<= 1e-6)
    worst_TD = maximum(abs.(sc.values["T_D"] .- truth_TD) ./ truth_TD)
    worst_uD = maximum(abs.(sc.values["u_D"] .- truth_uD) ./ truth_uD)
    @info "SX2 worst relative recovery error" worst_TD worst_uD
    for name in ("T_D", "u_D")
        med = median(sc.values[name])
        @test all(sc.s_stat[name] .<= 1e-8 * med)
    end
    @info "SX2 s_stat" s_stat_TD = sc.s_stat["T_D"] s_stat_uD = sc.s_stat["u_D"]
end

@testset "SX3: _detrended_spread's closed form" begin
    @test isapprox(_detrended_spread([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 5.0]), sqrt(1.35); rtol=1e-14)
    @test isnan(_detrended_spread([0.0, 1.0], [1.0, 3.0]))
    @test isnan(_detrended_spread([0.0], [1.0]))
end

@testset "SX4: SXB -- T_e absent; window 5 matches its own one-window deck" begin
    sc = _SX_SCATTER_B
    @test Set(keys(sc.values)) == Set(["T_D", "u_D"])

    resultB = _SX_RESULT_B
    Te_star = resultB.values["T_e"][1]
    TD_bin2 = resultB.values["T_D"][2]
    uD_bin2 = resultB.values["u_D"][2]

    text5 = _sx_deck_text(_SX_PATH_A, _SX_PATH_B, joinpath(_SX_DIR, "out_window5");
        regions_block="[[fit.regions]]\nt_range_ns = [0.4, 0.5]\nnbins = 1\n",
        te_value="\"T_e\"",
        te_var_block="T_e = { bounds = [100.0, 1000.0], init = $Te_star, vary = false }",
        regularizer_block="")
    deck5, problem5, _ = _sx_build(_SX_DIR, text5, "window5_deck.toml")
    start5 = flatten(problem5.pspec, ["T_D" => TD_bin2, "u_D" => uD_bin2])
    fit5 = solve(problem5, LevenbergMarquardt(); start=start5)
    pol5 = polish(problem5, fit5)

    rel_TD = abs(pol5.result.values["T_D"][1] - sc.values["T_D"][5]) / abs(sc.values["T_D"][5])
    rel_uD = abs(pol5.result.values["u_D"][1] - sc.values["u_D"][5]) / abs(sc.values["u_D"][5])
    @test rel_TD <= 1e-12
    @test rel_uD <= 1e-12
    @info "SX4 window 5 relative difference against its own one-window deck" rel_TD rel_uD
end

@testset "SX5: the three refusals" begin
    text_noregions = """
    deck_version = "otis-deck-2"

    [data]
    file = "$(_sx_path(_SX_PATH_A))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"

    [instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = "u_D"

    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_sx_path(joinpath(_SX_DIR, "out_noregions")))"
    """
    deck_nr, problem_nr, result_nr = _sx_build(_SX_DIR, text_noregions, "noregions_deck.toml")
    @test_throws r"has no \[\[fit\.regions\]\]" slice_scatter(deck_nr, result_nr)

    text_intervals = _sx_deck_text(_SX_PATH_A, _SX_PATH_B, joinpath(_SX_DIR, "out_intervals");
        intervals_block="[fit.intervals.foo]\nt_range_ns = [-1.0, 2.0]\n")
    deck_iv, problem_iv, result_iv = _sx_build(_SX_DIR, text_intervals, "intervals_deck.toml")
    @test_throws r"intervals are not supported" slice_scatter(deck_iv, result_iv)

    @test_throws r"was not fitted from this deck" slice_scatter(_SX_DECK_B, _SX_RESULT_A)
end
