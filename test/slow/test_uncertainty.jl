using Test
using OTIS
using OTIS: _leaf_si_path
using Random: Xoshiro
using Statistics: median

# CS4 against dev/phase4A_instructions.md Section 10.1's slow-tier row (spec D46), gated
# "uncertainty (slow)". The reproductions' cut-down EPW problem, epw_ablation_problem(; nlambda =
# 128, nt = 32), the EPW deck's own three variables (logn, T_e, m_e), with a
# WavelengthCalibration(lambda_offset_nm = 0.0) added to the fitting experiment's effects,
# unbound -- so the scan below is a pure calibration probe, not itself fitted.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

function _cs4_expt0(problem_nt)
    effects = (problem_nt.effects..., WavelengthCalibration(lambda_offset_nm=0.0))
    return Experiment(trace=problem_nt.trace, probe=problem_nt.probe, collection=problem_nt.collection,
        lambda_m=problem_nt.lambda, effects=effects)
end

function _cs4_pspec(expt0::Experiment)
    v_logn = FittedVariable("logn"; bounds=(18.5, 20.5), init=19.176, time_varying=true)
    v_Te = FittedVariable("T_e"; bounds=(100.0, 2000.0), init=350.0, time_varying=true)
    v_me = FittedVariable("m_e"; bounds=(2.0, 5.5), init=2.5, time_varying=true)
    return ParameterSpec(expt0; variables=[v_logn, v_Te, v_me], dependents=["ne_cm3" => "10^logn"],
        bindings=["ne_cm3" => "n_e_cm3", "T_e" => "electrons.e.T_eV", "m_e" => "electrons.e.f.m"])
end

# Reads a leaf's own true values off `trace` in the boundary key's own unit (the reproductions'
# own `_fr_truth`, test/slow/test_reproduce.jl, copied here under the @safetestset boundary rule).
function _cs4_truth(trace::PlasmaTrace, boundary_key::AbstractString)
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    i === nothing && error("_cs4_truth: leaf \"$si_path\" not present in this trace")
    return collect(Float64, trace.values[i, :]) ./ factor
end

# theta*(0): LM, then polish, started at the truth -- the EPW deck's own inits stall (Pitfall 31),
# and this is a synthetic sensitivity test, not a recovery test.
function _cs4_problem_and_star()
    problem_nt = epw_ablation_problem(; nlambda=128, nt=32)
    expt0 = _cs4_expt0(problem_nt)
    pspec = _cs4_pspec(expt0)
    working = Experiment(; trace=pspec.trace, probe=expt0.probe, collection=expt0.collection,
        lambda_m=expt0.lambda, effects=expt0.effects, prefactor=expt0.prefactor)

    theta_truth = flatten(pspec, [
        "logn" => log10.(_cs4_truth(pspec.trace, "n_e_cm3")),
        "T_e" => _cs4_truth(pspec.trace, "electrons.e.T_eV"),
        "m_e" => _cs4_truth(pspec.trace, "electrons.e.f.m")])

    truth_expt = unflatten(working, pspec, theta_truth)
    sd = synthesize(truth_expt, problem_nt.noise; rng=Xoshiro(problem_nt.seed), source="cs4_slow")

    reg = Regularizer([
            RegularizerTerm(pspec; select=["logn", "T_e"], operator=:D2, normalization=:relative, weight=1.0),
            RegularizerTerm(pspec; select=["m_e"], operator=:D2, normalization=:unit, weight=1.0)];
        weight=2.0e-4)

    problem = FitProblem(working, pspec, sd; regularizer=reg)

    t0 = time()
    r = solve(problem, LevenbergMarquardt(); start=theta_truth)
    pol = polish(problem, r)
    build_elapsed = time() - t0
    return problem, pol.result.theta, build_elapsed, r
end

@testset "CS4: the calibration scan on the cut-down EPW problem (128 x 32)" begin
    problem, theta_star, build_elapsed, r0 = _cs4_problem_and_star()
    @info "CS4: theta*(0) build" build_elapsed r0.status r0.iterations gn_decrease(problem, theta_star)

    shifts = [-0.2, -0.1, 0.0, 0.1, 0.2]
    t0 = time()
    scan = calibration_scan(problem, theta_star, shifts; channel=1)
    elapsed = time() - t0
    @info "CS4: calibration_scan elapsed (5 points)" elapsed per_point = elapsed / length(shifts)

    # Fixed by D49 (Phase 4.A remediation round 2, 2026-09-25): the -0.1 nm point's stall was a
    # forward-model bug, not the optimizer -- see dev/phase4A_instructions.md Section 10.1R.
    @test all(scan.decrease .<= 1e-6)
    zero_idx = findfirst(iszero, scan.shifts_nm)
    @test scan.results[zero_idx].theta == theta_star

    # Reported, never asserted (Section 10.1, the roadmap's own Test): the median movement of
    # logn and T_e per shift, their s_sys, and whether n_e moves monotonically with the shift
    # while T_e moves far less -- the EPW's red-shifted peak sits at a larger |Δλ| for larger n_e.
    med_dlogn = [median(scan.delta["logn"][:, k]) for k in eachindex(shifts)]
    med_dTe = [median(scan.delta["T_e"][:, k]) for k in eachindex(shifts)]
    for (k, s) in enumerate(shifts)
        @info "CS4 report: shift" shift_nm = s median_dlogn = med_dlogn[k] median_dTe = med_dTe[k] chi2 = scan.chi2[k] status = scan.status[k]
    end
    @info "CS4 report: s_sys" s_sys_logn = scan.s_sys["logn"] s_sys_Te = scan.s_sys["T_e"] s_sys_me = scan.s_sys["m_e"]
    monotone_ne = issorted(med_dlogn) || issorted(med_dlogn; rev=true)
    ratio = maximum(abs, med_dTe) / max(maximum(abs, med_dlogn), eps())
    @info "CS4 report: does n_e move monotonically with the shift, and by how much more than T_e" monotone_ne ratio_dTe_over_dlogn = ratio
end
