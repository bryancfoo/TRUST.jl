using Test
using OTIS
using OTIS: jvp!, jtvp!, _full_residuals, _gauss_newton_diagonal, _channel_nparams, channel_names,
    _collect_kernel_bindings, _kernel_effects, _widen_trace, _make_cache, _jacobian_bin!,
    _chain_order, JACOBIAN_CHUNK
import ForwardDiff
import Random
using LinearAlgebra: dot

# MC1 to MC12 against dev/phase4A_instructions.md Section 9.1's table (spec D31, D34, D35, D36),
# gated "joint fit". The MC fixture (D35) is built here, not imported across the @safetestset
# boundary (D46's own rule, inherited from Phase 3).

# ------------------------------------------------------------------------------------------
# The MC fixture (D35)
# ------------------------------------------------------------------------------------------

function _mc_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

# Channel a: 16 wavelengths, SpectralIRF + Sensitivity + ConstantBackground. Channel b: 12
# wavelengths, WavelengthCalibration + SpectralIRF + Sensitivity + ConstantBackground. Both share
# one trace and one probe/collection (D35).
function _mc_experiments()
    trace = PlasmaTrace(_mc_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
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
# own Sensitivity), dlam_b (channel b's WavelengthCalibration.offset), bg (bound into BOTH
# channels' ConstantBackground.level -- the shared-binding case, D35).
function _mc_pspec(expt_a::Experiment, expt_b::Experiment)
    sens_a = findfirst(e -> e isa Sensitivity, expt_a.effects)
    bg_a = findfirst(e -> e isa ConstantBackground, expt_a.effects)
    calib_b = findfirst(e -> e isa WavelengthCalibration, expt_b.effects)
    sens_b = findfirst(e -> e isa Sensitivity, expt_b.effects)
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

_mc_regularizer(pspec::ParameterSpec) = Regularizer(
    [RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)];
    weight=1e-3)

function _mc_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

_mc_perturb(theta0::AbstractVector, scale::Real=0.02) = theta0 .+ scale .* (1:length(theta0))

function _mc_data(working::Experiment, pspec::ParameterSpec, theta::AbstractVector, channel::Int, rng)
    truth = unflatten(working, pspec, theta; channel=channel)
    return synthesize(truth, GaussianNoise(a=1.0, b=0.02); rng=rng)
end

# Builds the full (unmasked) two-channel fixture: (working_a, working_b, pspec, reg, data_a,
# data_b, theta0). One seeded rng (Xoshiro(20260923), D35) draws channel a's noise then channel
# b's, in that order.
function _mc_full_problem()
    expt_a, expt_b = _mc_experiments()
    pspec = _mc_pspec(expt_a, expt_b)
    reg = _mc_regularizer(pspec)
    working_a = _mc_working(expt_a, pspec)
    working_b = _mc_working(expt_b, pspec)
    theta0 = flatten(pspec)
    rng = Random.Xoshiro(20260923)
    data_a = _mc_data(working_a, pspec, theta0, 1, rng)
    data_b = _mc_data(working_b, pspec, theta0, 2, rng)
    return working_a, working_b, pspec, reg, data_a, data_b, theta0
end

# The masked-channel pair (D35): `reference`, a plain single-channel FitProblem carrying `reg`;
# `joint`, the same channel a plus a fully masked channel b, on the same pspec.
function _mc_pair()
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    data_b_allmasked = StreakData(lambda_m=data_b.lambda, t_s=data_b.t, image=data_b.image,
        sigma=data_b.sigma, mask=falses(size(data_b.image)))

    reference = FitProblem(working_a, pspec, data_a; channel=1, regularizer=reg)
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b_allmasked; channel=2, name="b", allow_empty_mask=true)
    joint = JointFitProblem((ch_a, ch_b); regularizer=reg)

    theta = _mc_perturb(theta0)
    return reference, joint, pspec, theta
end

# A minimal, independent single-channel problem: MC1's "a different pspec" fixture.
function _mc_other_problem()
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
    trace = PlasmaTrace(template; t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.9, 263.6; length=16)) .* 1e-9
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(Sensitivity(scale=1.0e-6),))
    v_T = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0)
    pspec = ParameterSpec(expt; variables=[v_T], bindings=["T_e" => "electrons.e.T_eV"])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
    theta0 = flatten(pspec)
    data = synthesize(unflatten(working, pspec, theta0), GaussianNoise(a=1.0, b=0.02);
        rng=Random.Xoshiro(1))
    return FitProblem(working, pspec, data; channel=2, name="other")
end

# ------------------------------------------------------------------------------------------
# MC1: JointFitProblem's seven checks, FitProblem's name check, allow_empty_mask (D31)
# ------------------------------------------------------------------------------------------

@testset "MC1: JointFitProblem's seven checks, FitProblem's name check, allow_empty_mask (D31)" begin
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")

    @test_throws r"at least one channel" JointFitProblem(())

    @test_throws r"weights for 2 channels" JointFitProblem((ch_a, ch_b); weights=[1.0])

    @test_throws r"finite and positive" JointFitProblem((ch_a, ch_b); weights=[1.0, -1.0])
    @test_throws r"finite and positive" JointFitProblem((ch_a, ch_b); weights=[1.0, NaN])
    @test_throws r"finite and positive" JointFitProblem((ch_a, ch_b); weights=[1.0, Inf])

    other = _mc_other_problem()
    @test_throws r"is not === channel 1's" JointFitProblem((ch_a, other))

    ch_b_wrong_channel = FitProblem(working_b, pspec, data_b; channel=1, name="b")
    @test_throws r"expected 2" JointFitProblem((ch_a, ch_b_wrong_channel))

    ch_b_with_reg = FitProblem(working_b, pspec, data_b; channel=2, name="b", regularizer=reg)
    @test_throws r"carries its own regularizer" JointFitProblem((ch_a, ch_b_with_reg))

    ch_b_same_name = FitProblem(working_b, pspec, data_b; channel=2, name="a")
    @test_throws r"names must be distinct" JointFitProblem((ch_a, ch_b_same_name))

    data_a_empty = StreakData(lambda_m=data_a.lambda, t_s=data_a.t, image=data_a.image,
        sigma=data_a.sigma, mask=falses(size(data_a.image)))
    ch_a_empty = FitProblem(working_a, pspec, data_a_empty; channel=1, name="a", allow_empty_mask=true)
    data_b_empty = StreakData(lambda_m=data_b.lambda, t_s=data_b.t, image=data_b.image,
        sigma=data_b.sigma, mask=falses(size(data_b.image)))
    ch_b_empty = FitProblem(working_b, pspec, data_b_empty; channel=2, name="b", allow_empty_mask=true)
    @test_throws r"every channel is empty" JointFitProblem((ch_a_empty, ch_b_empty))

    jp = JointFitProblem((ch_a, ch_b))
    @test jp isa JointFitProblem
    @test channel_names(jp) == ["a", "b"]

    @test_throws r"name must match" FitProblem(working_a, pspec, data_a; name="Bad")

    @test_throws r"the mask excludes every pixel" FitProblem(working_b, pspec, data_b_empty; channel=2, name="b")
    @test ch_b_empty.npixels == 0
    @test isempty(ch_b_empty.pixel_index)
end

# ------------------------------------------------------------------------------------------
# MC2 (star): the masked-channel test, products -- === elementwise, everything (D35)
# ------------------------------------------------------------------------------------------

@testset "MC2: the masked-channel test, products (bitwise, D35)" begin
    reference, joint, pspec, theta = _mc_pair()
    ntheta = pspec.ntheta

    @test all(residuals(reference, theta) .=== residuals(joint, theta))
    @test all(_full_residuals(reference, theta) .=== _full_residuals(joint, theta))
    @test objective(reference, theta) === objective(joint, theta)

    g_ref = zeros(ntheta)
    g_joint = zeros(ntheta)
    gradient!(g_ref, reference, theta)
    gradient!(g_joint, joint, theta)
    @test all(g_ref .=== g_joint)

    Jac_ref = jacobian(reference, theta)
    Jac_joint = jacobian(joint, theta)
    nrows = reference.npixels + reference.regularizer.nrows

    idx_bin = pspec.bin_entries[1][1]   # T_D's own bin-1 entry, the only per-bin variable
    idx_bg = pspec.offsets[findfirst(v -> v.name == "bg", pspec.variables)]
    idx_scale_a = pspec.offsets[findfirst(v -> v.name == "scale_a", pspec.variables)]

    rng = Random.Xoshiro(20260924)
    unit_vectors = Vector{Float64}[]
    for idx in (idx_bin, idx_bg, idx_scale_a)
        v = zeros(ntheta)
        v[idx] = 1.0
        push!(unit_vectors, v)
    end
    push!(unit_vectors, randn(rng, ntheta))

    for v in unit_vectors
        out_ref = zeros(nrows)
        out_joint = zeros(nrows)
        jvp!(out_ref, reference, Jac_ref, v)
        jvp!(out_joint, joint, Jac_joint, v)
        @test all(out_ref .=== out_joint)
    end

    for _ in 1:3
        w = randn(rng, nrows)
        gt_ref = zeros(ntheta)
        gt_joint = zeros(ntheta)
        jtvp!(gt_ref, reference, Jac_ref, w)
        jtvp!(gt_joint, joint, Jac_joint, w)
        @test all(gt_ref .=== gt_joint)
    end

    d_ref = _gauss_newton_diagonal(reference, Jac_ref)
    d_joint = _gauss_newton_diagonal(joint, Jac_joint)
    @test all(d_ref .=== d_joint)
end

# ------------------------------------------------------------------------------------------
# MC3 (star): the masked-channel test, fits -- === (D35)
# ------------------------------------------------------------------------------------------

@testset "MC3: the masked-channel test, fits (LBFGS, LevenbergMarquardt, D35)" begin
    for opt in (LBFGS(iterations=5), LevenbergMarquardt(iterations=3))
        reference, joint, pspec, _ = _mc_pair()
        result_ref = solve(reference, opt)
        result_joint = solve(joint, opt)

        @test all(result_ref.theta .=== result_joint.theta)
        @test result_ref.chi2 === result_joint.chi2
        @test result_ref.R === result_joint.R
        @test result_ref.objective === result_joint.objective
        @test length(result_ref.trajectory) == length(result_joint.trajectory)
        for (row_ref, row_joint) in zip(result_ref.trajectory, result_joint.trajectory)
            @test row_ref === row_joint
        end
        @test Set(keys(result_ref.values)) == Set(keys(result_joint.values))
        for k in keys(result_ref.values)
            @test all(result_ref.values[k] .=== result_joint.values[k])
        end
    end
end

# ------------------------------------------------------------------------------------------
# MC4-MC6: the joint Jacobian, weights = [0.3, 2.0], regularizer on
# ------------------------------------------------------------------------------------------

function _mc4_fixture()
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); weights=[0.3, 2.0], regularizer=reg)
    theta = _mc_perturb(theta0)
    ntheta = pspec.ntheta
    nrows = jp.npixels + reg.nrows
    return jp, theta, ntheta, nrows
end

# The dense transcription from jvp! on unit vectors (MC4's own oracle route, distinct from JT1's
# from-FitJacobian's-fields route, since JointFitJacobian nests one FitJacobian per channel).
function _mc_dense_jvp(jp::JointFitProblem, Jac, ntheta::Int, nrows::Int)
    J_dense = zeros(nrows, ntheta)
    v = zeros(ntheta)
    out = zeros(nrows)
    for k in 1:ntheta
        fill!(v, 0.0)
        v[k] = 1.0
        jvp!(out, jp, Jac, v)
        J_dense[:, k] = out
    end
    return J_dense
end

@testset "MC4: the joint Jacobian against brute force (weights = [0.3, 2.0], regularizer on)" begin
    jp, theta, ntheta, nrows = _mc4_fixture()
    Jac = jacobian(jp, theta)

    J_bruteforce = ForwardDiff.jacobian(th -> _full_residuals(jp, th), theta)
    @test maximum(abs, J_bruteforce) > 0

    J_dense = _mc_dense_jvp(jp, Jac, ntheta, nrows)

    worst = 0.0
    worst_col = 0
    for k in 1:ntheta
        colref = @view J_bruteforce[:, k]
        colnew = @view J_dense[:, k]
        floor_k = 1e-10 * maximum(abs, colref)
        relerr = maximum(abs.(colnew .- colref) ./ max.(abs.(colref), floor_k))
        if relerr > worst
            worst = relerr
            worst_col = k
        end
        @test relerr <= 1e-10
    end
    @info "MC4 worst relative error" worst worst_col
end

@testset "MC5: jtvp! is jvp!'s transpose (MC4's problem)" begin
    jp, theta, ntheta, nrows = _mc4_fixture()
    Jac = jacobian(jp, theta)
    rng = Random.Xoshiro(20260925)
    for _ in 1:3
        v = randn(rng, ntheta)
        w = randn(rng, nrows)
        Jv = zeros(nrows)
        jvp!(Jv, jp, Jac, v)
        Jtw = zeros(ntheta)
        jtvp!(Jtw, jp, Jac, w)
        @test isapprox(dot(w, Jv), dot(Jtw, v); rtol=1e-12)
    end
end

@testset "MC6: the Gauss-Newton diagonal equals the dense transcription's column sums of squares" begin
    jp, theta, ntheta, nrows = _mc4_fixture()
    Jac = jacobian(jp, theta)
    J_dense = _mc_dense_jvp(jp, Jac, ntheta, nrows)
    d = _gauss_newton_diagonal(jp, Jac)
    for k in 1:ntheta
        @test isapprox(d[k], sum(abs2, view(J_dense, :, k)); rtol=1e-12)
    end
end

# ------------------------------------------------------------------------------------------
# MC7: the weights scale rows exactly (weights = [4.0, 1.0])
# ------------------------------------------------------------------------------------------

@testset "MC7: the weights scale rows exactly (weights = [4.0, 1.0])" begin
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); weights=[4.0, 1.0], regularizer=reg)
    theta = _mc_perturb(theta0)

    r_unweighted_a = residuals(ch_a, theta)
    r_unweighted_b = residuals(ch_b, theta)
    r_joint = residuals(jp, theta)

    ra = view(r_joint, jp.row_ranges[1])
    rb = view(r_joint, jp.row_ranges[2])
    @test all(ra .=== 2 .* r_unweighted_a)
    @test all(rb .=== r_unweighted_b)

    chi2_a = chi2(ch_a, theta)
    chi2_b = chi2(ch_b, theta)
    @test isapprox(chi2(jp, theta), 4 * chi2_a + chi2_b; rtol=1e-14)
end

# ------------------------------------------------------------------------------------------
# MC8: unflatten per channel (D33)
# ------------------------------------------------------------------------------------------

@testset "MC8: unflatten per channel (D33)" begin
    expt_a, expt_b = _mc_experiments()
    pspec = _mc_pspec(expt_a, expt_b)
    working_a = _mc_working(expt_a, pspec)
    working_b = _mc_working(expt_b, pspec)
    theta = flatten(pspec, ["scale_a" => 3e-6, "scale_b" => 5e-6, "dlam_b" => 0.05e-9, "bg" => 30.0])

    fitted_a = unflatten(working_a, pspec, theta; channel=1)
    sens_a_idx = findfirst(e -> e isa Sensitivity, fitted_a.effects)
    bg_a_idx = findfirst(e -> e isa ConstantBackground, fitted_a.effects)
    @test isapprox(fitted_a.effects[sens_a_idx].scale, 3e-6; rtol=1e-12)
    @test isapprox(fitted_a.effects[bg_a_idx].level, 30.0; rtol=1e-12)
    @test findfirst(e -> e isa WavelengthCalibration, fitted_a.effects) === nothing

    fitted_b = unflatten(working_b, pspec, theta; channel=2)
    sens_b_idx = findfirst(e -> e isa Sensitivity, fitted_b.effects)
    bg_b_idx = findfirst(e -> e isa ConstantBackground, fitted_b.effects)
    calib_b_idx = findfirst(e -> e isa WavelengthCalibration, fitted_b.effects)
    @test isapprox(fitted_b.effects[sens_b_idx].scale, 5e-6; rtol=1e-12)
    @test isapprox(fitted_b.effects[bg_b_idx].level, 30.0; rtol=1e-12)
    @test isapprox(fitted_b.effects[calib_b_idx].offset, 0.05e-9; rtol=1e-12)

    # neither channel's own Sensitivity carries the other channel's scale
    @test !isapprox(fitted_a.effects[sens_a_idx].scale, 5e-6; rtol=1e-6)
    @test !isapprox(fitted_b.effects[sens_b_idx].scale, 3e-6; rtol=1e-6)

    # MC8's aligned-layout case (Level 2 amendment, dev/phase4A_instructions.md Section 9.1,
    # 2026-09-24, correcting the discrimination check's revert (a) gap found in chunk 6's own
    # report: the main MC fixture staggers channel b's layout with a leading
    # WavelengthCalibration, so every coincidental same-index collision lands on a differently
    # named field and revert (a) never corrupts a value). Two experiments with ONE SHARED effect
    # layout, index for index: channel a's Sensitivity is left unbound (fixed at construction,
    # scale = 1e-6); channel b's Sensitivity sits at the SAME index and is fitted through
    # scale_b2. Under the correct channel filter, channel a has no :effect binding of its own,
    # so unflatten returns it unchanged; under revert (a) the channel filter is gone and
    # scale_b2's binding (effect index 1, field :scale, no competing channel-1 binding on that
    # field to make `only(matching)` throw) is picked up silently, corrupting channel a's own
    # Sensitivity.scale to channel b's value.
    trace_aligned = PlasmaTrace(_mc_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_aligned = collect(range(262.9, 263.6; length=16)) .* 1e-9
    expt_a2 = Experiment(trace=trace_aligned, probe=probe, collection=collection, lambda_m=lambda_aligned,
        effects=(Sensitivity(scale=1e-6),))
    expt_b2 = Experiment(trace=trace_aligned, probe=probe, collection=collection, lambda_m=lambda_aligned,
        effects=(Sensitivity(scale=2e-6),))

    v_scale_b2 = FittedVariable("scale_b2"; bounds=(1e-7, 1e-5), init=5e-6)
    pspec_aligned = ParameterSpec((expt_a2, expt_b2); variables=[v_scale_b2],
        bindings=["scale_b2" => (channel=2, effect=1, field=:scale)])

    working_a2 = _mc_working(expt_a2, pspec_aligned)
    theta_aligned = flatten(pspec_aligned, ["scale_b2" => 5e-6])

    fitted_a2 = unflatten(working_a2, pspec_aligned, theta_aligned; channel=1)
    @test isapprox(fitted_a2.effects[1].scale, 1e-6; rtol=1e-12)
end

# ------------------------------------------------------------------------------------------
# MC9: FitResult's per-channel summaries (D36)
# ------------------------------------------------------------------------------------------

@testset "MC9: FitResult's per-channel summaries (D36)" begin
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); weights=[1.3, 0.7], regularizer=reg)

    result = solve(jp, LBFGS(iterations=3))
    theta_star = result.theta

    @test length(result.channels) == 2
    ca, cb = result.channels
    @test ca.name == "a"
    @test cb.name == "b"
    @test ca.weight == 1.3
    @test cb.weight == 0.7
    @test ca.chi2 === chi2(ch_a, theta_star)
    @test cb.chi2 === chi2(ch_b, theta_star)
    @test isapprox(result.chi2, 1.3 * ca.chi2 + 0.7 * cb.chi2; rtol=1e-14)
    @test ca.npixels == ch_a.npixels
    @test cb.npixels == ch_b.npixels
    @test result.npixels == ch_a.npixels + ch_b.npixels

    ntheta = pspec.ntheta
    @test ca.nparams == ntheta - 2   # scale_b, dlam_b: channel-2-only bindings
    @test cb.nparams == ntheta - 1   # scale_a: a channel-1-only binding

    expected_reduced_a = ca.npixels > ca.nparams ? ca.chi2 / (ca.npixels - ca.nparams) : NaN
    expected_reduced_b = cb.npixels > cb.nparams ? cb.chi2 / (cb.npixels - cb.nparams) : NaN
    @test isequal(ca.reduced_chi2, expected_reduced_a)
    @test isequal(cb.reduced_chi2, expected_reduced_b)
    @test isapprox(ca.reduced_chi2_simple, ca.chi2 / ca.npixels; rtol=1e-14)
    @test isapprox(cb.reduced_chi2_simple, cb.chi2 / cb.npixels; rtol=1e-14)

    @test length(result.experiments) == 2
    @test result.experiment === result.experiments[1]

    str = sprint(show, MIME("text/plain"), result)
    @test occursin("a: weight", str)
    @test occursin("b: weight", str)
end

# ------------------------------------------------------------------------------------------
# MC10: every optimizer runs on a JointFitProblem (D34)
# ------------------------------------------------------------------------------------------

@testset "MC10: every optimizer runs on a JointFitProblem (D34)" begin
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); regularizer=reg)

    for opt in (LBFGS(iterations=3), LevenbergMarquardt(iterations=2),
        NoisyGradientDescent(steps=3),
        OptimizerSequence(NoisyGradientDescent(steps=2), LBFGS(iterations=2)))
        result = solve(jp, opt)
        @test result isa FitResult
        @test length(result.channels) == 2
        @test length(result.experiments) == 2
    end
end

# ------------------------------------------------------------------------------------------
# MC11: weight_scan over a JointFitProblem (D34)
# ------------------------------------------------------------------------------------------

@testset "MC11: weight_scan over a JointFitProblem (D34)" begin
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); regularizer=reg)

    scan = weight_scan(jp, [1e-4, 1e-2], LBFGS(iterations=3))
    @test scan isa WeightScanResult
    @test length(scan.results) == 2
    for r in scan.results
        @test length(r.channels) == 2
    end
    @test length(scan.unregularized.channels) == 2
end

# ------------------------------------------------------------------------------------------
# MC12: the per-bin dual pass allocates zero for a channel-2 kernel global (JT16's form)
# ------------------------------------------------------------------------------------------

@testset "MC12: the per-bin dual pass allocates zero for a channel-2 kernel global (JT16's form)" begin
    working_a, working_b, pspec, reg, data_a, data_b, theta0 = _mc_full_problem()
    problem_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    theta = _mc_perturb(theta0)

    expt = unflatten(problem_b.experiment, pspec, theta; channel=2)
    nvar = length(pspec.variables)
    nlambda = length(problem_b.experiment.lambda)

    kernel_effects0 = _kernel_effects(expt)
    nkernel_eff = length(kernel_effects0)

    calib_idx = findfirst(e -> e isa WavelengthCalibration, expt.effects)
    kb2 = _collect_kernel_bindings(pspec, expt, nkernel_eff; channel=2)
    @test any(b -> b.field === :offset, kb2[calib_idx])
    kb1 = _collect_kernel_bindings(pspec, expt, nkernel_eff; channel=1)
    @test isempty(kb1[calib_idx])

    kernel_bindings = kb2
    kernel_global_var = Int[]
    for i in 1:nvar
        v = pspec.variables[i]
        (v.vary && !v.time_varying) || continue
        affects_P = false
        for b in pspec.bindings
            b.slot == i || continue
            if b.kind === :leaf
                affects_P = true
            elseif b.kind === :effect && b.channel == 2 && 1 <= b.effect <= nkernel_eff && _chain_order(expt.effects[b.effect]) == 1
                affects_P = true
            end
        end
        affects_P && push!(kernel_global_var, i)
    end
    G = length(kernel_global_var)
    @test G >= 1
    kernel_global_idx = [pspec.offsets[i] for i in kernel_global_var]
    kernel_global_of_var = zeros(Int, nvar)
    for (gk, i) in enumerate(kernel_global_var)
        kernel_global_of_var[i] = gk
    end

    DFT = ForwardDiff.Dual{OTIS._JacobianTag,Float64,JACOBIAN_CHUNK}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[1]))
    kernel_globals = Array{Float64,3}(undef, nlambda, length(pspec.trace.axis), G)
    probe0, collection0, lambda0 = expt.probe, expt.collection, expt.lambda
    prefactor_flag = OTIS._prefactor_flag(expt.prefactor)

    _jb!() = _jacobian_bin!(block, kernel_globals, 1, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
        probe0, collection0, lambda0, prefactor_flag, theta)
    _jb!()   # warm
    bytes = @allocated _jb!()
    @test bytes == 0
end
