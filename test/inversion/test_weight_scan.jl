using Test
using OTIS
using OTIS: _with_weight
using Random: Xoshiro

# WS1 to WS10 against dev/phase3_instructions.md Section 8 (spec D31), gated "weight scan". P2
# (D46) is duplicated here rather than imported from test_optimizers.jl, since chunks do not
# import across a @safetestset boundary (D46's own rule): a two-term D2 regularizer, one term per
# fitted variable at its own bound-width normalization, so a scan has something to scan.
#
# Unlike test_optimizers.jl's own P2 (deliberately noise-free sigma weights on a noise-free
# image, D46's OP1-OP9 recipe), this file's data actually carries `synthesize`d noise: the
# unregularized fit here needs a genuine, non-machine-precision RSS against `truth` for WS5/WS6
# to be a meaningful comparison rather than a division by a near-zero denominator.
const _WS_SEED = 20260914

const _WS_WEIGHTS = [1e-4, 1e-3, 1e-2, 1e-1]

function _ws_p2_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _ws_p2_expt()
    trace = PlasmaTrace(_ws_p2_template(); t_ns=collect(range(0.0, 3.0; length=10)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=32)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
end

function _ws_p2_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

function _ws_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

# Builds P2 with a smoothly time-varying truth (a linear ramp on both T_D and u_D), a two-term D2
# regularizer (T_D and u_D each at absolute/bound-width normalization), and returns
# (problem, theta_start, truth), truth a Dict of the generating traces for WS5.
function _ws_p2_problem()
    expt = _ws_p2_expt()
    pspec = _ws_p2_pspec(expt)
    working = _ws_working(expt, pspec)
    nt = nbins(pspec.trace)
    t_ramp = range(220.0, 260.0; length=nt)
    u_ramp = range(-20.0, 20.0; length=nt)
    theta_truth = flatten(pspec, ["T_D" => collect(t_ramp), "u_D" => collect(u_ramp)])
    truth_expt = unflatten(working, pspec, theta_truth)
    data = synthesize(truth_expt, GaussianNoise(a=1.0, b=0.02); rng=Xoshiro(_WS_SEED))

    term_T = RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:absolute, weight=1.0)
    term_u = RegularizerTerm(pspec; select=["u_D"], operator=:D2, normalization=:absolute, weight=1.0)
    reg = Regularizer([term_T, term_u]; weight=1.0)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    truth = Dict("T_D" => collect(t_ramp), "u_D" => collect(u_ramp))
    theta_start = theta_truth .+ 0.02 .* (1:length(theta_truth))
    return problem, theta_start, truth
end

# A second P2 build with a ridge (:I) term on T_D alone, reference 0 -- a large Lambda pulls T_D
# toward zero, far from its ~220-260 eV truth, so this fixture can force a per-parameter RSS ratio
# above one on demand (WS6), rather than hoping the D2 fixture above produces one by chance (a D2
# term does not penalize the linear truth at all, since D2 of a straight line is zero).
function _ws_p2_ridge_problem()
    expt = _ws_p2_expt()
    pspec = _ws_p2_pspec(expt)
    working = _ws_working(expt, pspec)
    nt = nbins(pspec.trace)
    t_ramp = range(220.0, 260.0; length=nt)
    u_ramp = range(-20.0, 20.0; length=nt)
    theta_truth = flatten(pspec, ["T_D" => collect(t_ramp), "u_D" => collect(u_ramp)])
    truth_expt = unflatten(working, pspec, theta_truth)
    data = synthesize(truth_expt, GaussianNoise(a=1.0, b=0.02); rng=Xoshiro(_WS_SEED))

    term = RegularizerTerm(pspec; select=["T_D"], operator=:I, normalization=:absolute, weight=1.0)
    reg = Regularizer([term]; weight=1.0)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    truth = Dict("T_D" => collect(t_ramp))
    theta_start = theta_truth .+ 0.02 .* (1:length(theta_truth))
    return problem, theta_start, truth
end

# ------------------------------------------------------------------------------------------
# WS1: one row per requested weight
# ------------------------------------------------------------------------------------------

@testset "WS1: one row per requested weight" begin
    problem, start, truth = _ws_p2_problem()
    scan = weight_scan(problem, _WS_WEIGHTS, LBFGS(); start=start)
    @test length(scan.weights) == 4
    @test scan.weights == sort(_WS_WEIGHTS)
    @test length(scan.chi2) == 4
    @test length(scan.R) == 4
    @test length(scan.chi2_normalized) == 4
    @test length(scan.R_normalized) == 4
    @test length(scan.results) == 4
end

# ------------------------------------------------------------------------------------------
# WS2: chi2 non-decreasing and R non-increasing in Lambda, to the optimizer's own tolerance
# ------------------------------------------------------------------------------------------

@testset "WS2: chi2 is non-decreasing and R non-increasing in Lambda" begin
    problem, start, truth = _ws_p2_problem()
    scan = weight_scan(problem, _WS_WEIGHTS, LBFGS(); start=start)
    @info "WS2" scan.chi2 scan.R
    for k in 1:3
        slack_chi2 = 1e-6 * scan.chi2[1]
        slack_R = 1e-6 * scan.R[1]
        @test scan.chi2[k+1] >= scan.chi2[k] - slack_chi2
        @test scan.R[k+1] <= scan.R[k] + slack_R
    end
end

# ------------------------------------------------------------------------------------------
# WS3: the normalization, and the unregularized fit is not a row of the table
# ------------------------------------------------------------------------------------------

@testset "WS3: normalized by the unregularized fit, which is not itself a row" begin
    problem, start, truth = _ws_p2_problem()
    scan = weight_scan(problem, _WS_WEIGHTS, LBFGS(); start=start)
    @test scan.chi2_normalized == scan.chi2 ./ scan.chi2_unreg
    @test scan.R_normalized == scan.R ./ scan.R_unreg
    @test length(scan.weights) == length(_WS_WEIGHTS)   # the unregularized fit adds no row
    @test scan.chi2_unreg == scan.unregularized.chi2
    @test scan.R_unreg == scan.unregularized.R
end

# ------------------------------------------------------------------------------------------
# WS4: the warm start reproduces the cold start
# ------------------------------------------------------------------------------------------

@testset "WS4: the warm start reproduces the cold start" begin
    problem, start, truth = _ws_p2_problem()
    scan = weight_scan(problem, _WS_WEIGHTS, LBFGS(); start=start)

    w = _WS_WEIGHTS[1]
    cold_problem = _with_weight(problem, w)
    cold_result = solve(cold_problem, LBFGS(); start=flatten(problem.pspec))

    warm_theta = scan.results[1].theta
    @info "WS4" warm_theta cold_result.theta
    @test isapprox(warm_theta, cold_result.theta; rtol=1e-4, atol=1e-4)
end

# ------------------------------------------------------------------------------------------
# WS5: the per-parameter RSS
# ------------------------------------------------------------------------------------------

@testset "WS5: the per-parameter RSS" begin
    problem, start, truth = _ws_p2_problem()
    scan = weight_scan(problem, _WS_WEIGHTS, LBFGS(); start=start, truth=truth)

    for (name, tvals) in truth
        expected_unreg = sum(abs2, tvals .- scan.unregularized.values[name])
        @test isapprox(scan.rss_unreg[name], expected_unreg; rtol=1e-12)
        expected = [sum(abs2, tvals .- r.values[name]) for r in scan.results]
        @test isapprox(scan.rss[name], expected; rtol=1e-12)
        @test isapprox(scan.rss_normalized[name], expected ./ expected_unreg; rtol=1e-12)
    end
end

# ------------------------------------------------------------------------------------------
# WS6: a ratio above one is allowed and is not bounded by any assertion
# ------------------------------------------------------------------------------------------

@testset "WS6: a ratio above one is allowed" begin
    problem, start, truth = _ws_p2_ridge_problem()
    scan = weight_scan(problem, [1e-4, 1e-2, 1.0, 100.0], LBFGS(); start=start, truth=truth)
    @info "WS6" scan.rss_normalized["T_D"]
    @test scan.rss_normalized["T_D"][end] > 1.0
    @test all(isfinite, scan.rss_normalized["T_D"])
end

# ------------------------------------------------------------------------------------------
# WS7: the scan chooses nothing
# ------------------------------------------------------------------------------------------

@testset "WS7: the scan chooses nothing" begin
    names_lower = lowercase.(String.(fieldnames(WeightScanResult)))
    for bad in ("opt", "best", "chosen", "argmin", "recommend", "select")
        @test !any(n -> occursin(bad, n), names_lower)
    end
    doc = string(@doc weight_scan)
    @test occursin("Step 4.3", doc)
end

# ------------------------------------------------------------------------------------------
# WS8: bad weights
# ------------------------------------------------------------------------------------------

@testset "WS8: bad weights" begin
    problem, start, truth = _ws_p2_problem()
    @test_throws "weights must be positive and distinct" weight_scan(
        problem, [1e-3, 1e-3, 1e-2], LBFGS(); start=start)
    @test_throws "weights must be positive and distinct" weight_scan(
        problem, [0.0, 1e-2], LBFGS(); start=start)
    @test_throws "weights must be positive and distinct" weight_scan(
        problem, [-1e-3, 1e-2], LBFGS(); start=start)
end

# ------------------------------------------------------------------------------------------
# WS9: lcurveplot takes it -- two plain vectors of equal length, never invoked (Makie is not
# loaded in the fast suite)
# ------------------------------------------------------------------------------------------

@testset "WS9: lcurveplot takes it" begin
    problem, start, truth = _ws_p2_problem()
    scan = weight_scan(problem, _WS_WEIGHTS, LBFGS(); start=start)
    @test scan.chi2_normalized isa Vector{Float64}
    @test scan.R_normalized isa Vector{Float64}
    @test length(scan.chi2_normalized) == length(scan.R_normalized)
end

# ------------------------------------------------------------------------------------------
# WS10: no process is spawned
# ------------------------------------------------------------------------------------------

@testset "WS10: no process is spawned" begin
    path = joinpath(@__DIR__, "..", "..", "src", "inversion", "weight_scan.jl")
    found = String[]
    for line in eachline(path)
        for banned in ("Distributed", "@spawn", "addprocs", "@threads", "Threads.@spawn")
            occursin(banned, line) && push!(found, "$banned: $(strip(line))")
        end
    end
    @test isempty(found)
end
