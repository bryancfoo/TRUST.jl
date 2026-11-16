using Test
using TRUST
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const EX_NLAMBDA = 128
const EX_NT = 32

@testset "EX1-EX4: EPW ablation" begin
    problem = epw_ablation_problem(; nlambda=EX_NLAMBDA, nt=EX_NT)
    expt = build_experiment(problem)
    M = measure(expt)
    lambda = expt.lambda

    window = (lambda .>= lambda_to_si(232.0, :lambda_nm)) .& (lambda .<= lambda_to_si(250.0, :lambda_nm))
    lam_win = lambda[window]

    peaks_nm = Float64[]
    peak_vals = Float64[]
    for j in 1:EX_NT
        col = M[window, j]
        i = argmax(col)
        push!(peaks_nm, lambda_from_si(lam_win[i], :lambda_nm))
        push!(peak_vals, col[i])
    end

    # EX1: the peak lies in [239, 250] nm in every bin.
    @test all(p -> 239.0 <= p <= 250.0, peaks_nm)

    # EX2: the bluest column peak is at 239.5 +/- 1.0 nm, at a bin with t in [3.0, 3.9] ns.
    t_ns = t_from_si.(expt.t, :t_ns)
    jmin = argmin(peaks_nm)
    @info "EX2" bluest_nm = peaks_nm[jmin] t_ns = t_ns[jmin]
    @test isapprox(peaks_nm[jmin], 239.5; atol=1.0)
    @test 3.0 - 0.45 <= t_ns[jmin] <= 3.9 + 0.45

    # EX3: the last bin's peak is red of the bluest by more than 2.0 nm.
    @info "EX3" excursion = peaks_nm[end] - peaks_nm[jmin]
    @test peaks_nm[end] - peaks_nm[jmin] > 2.0

    # EX4: the last bin's peak value exceeds the bluest bin's by more than 1.3x.
    @info "EX4" ratio = peak_vals[end] / peak_vals[jmin]
    @test peak_vals[end] / peak_vals[jmin] > 1.3
end

@testset "EX5-EX8: IAW multi-flow" begin
    problem = iaw_multiflow_problem(; nlambda=EX_NLAMBDA, nt=EX_NT)
    expt = build_experiment(problem)
    M = measure(expt)
    lambda = expt.lambda

    argmax_nm = Float64[]
    peak_vals = Float64[]
    for j in 1:EX_NT
        i = argmax(view(M, :, j))
        push!(argmax_nm, lambda_from_si(lambda[i], :lambda_nm))
        push!(peak_vals, M[i, j])
    end

    # EX5
    @info "EX5" bin1 = argmax_nm[1] bin32 = argmax_nm[end]
    @test isapprox(argmax_nm[1], 262.83; atol=0.05)
    @test isapprox(argmax_nm[end], 263.36; atol=0.05)

    # EX6: non-decreasing to within one wavelength bin.
    dlam_nm = (lambda_from_si(lambda[end], :lambda_nm) - lambda_from_si(lambda[1], :lambda_nm)) / (length(lambda) - 1)
    for j in 2:EX_NT
        @test argmax_nm[j] >= argmax_nm[j-1] - dlam_nm
    end

    # EX7: total excursion exceeds 0.40 nm.
    excursion = maximum(argmax_nm) - minimum(argmax_nm)
    @info "EX7" excursion
    @test excursion > 0.40

    # EX8: peak value grows by more than 2.5x from bin 1 to bin 32.
    @info "EX8" ratio = peak_vals[end] / peak_vals[1]
    @test peak_vals[end] / peak_vals[1] > 2.5
end

@testset "EX9-EX12: IAW shock" begin
    problem = iaw_shock_problem(; nlambda=EX_NLAMBDA, nt=EX_NT)

    # EX9: columns 1 to 9 are bitwise identical to column 1, WITHOUT the TemporalIRF (the
    # pre-shock plasma is constant, so this is exact and also proves plasma(trace, j) is
    # deterministic; with the TemporalIRF on, the shock at t_s = 1.15 ns mixes neighbouring
    # bins and this only holds to about bin 7).
    effects_no_temporal = (SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6))
    expt_no_t = Experiment(trace=problem.trace, probe=problem.probe, collection=problem.collection,
        lambda_m=problem.lambda, effects=effects_no_temporal)
    M9 = measure(expt_no_t)
    for j in 2:9
        @test all(M9[:, j] .=== M9[:, 1])
    end

    expt = build_experiment(problem)
    M = measure(expt)
    lambda = expt.lambda
    t_ns = t_from_si.(expt.t, :t_ns)
    @info "shock t_ns near shock" t9 = t_ns[9] t10 = t_ns[10]

    argmax_nm = Float64[]
    colsum = Float64[]
    for j in 1:EX_NT
        i = argmax(view(M, :, j))
        push!(argmax_nm, lambda_from_si(lambda[i], :lambda_nm))
        push!(colsum, sum(view(M, :, j)))
    end

    # EX10: the column sum jumps by more than 3x from bin 9 to bin 10.
    @info "EX10" ratio = colsum[10] / colsum[9]
    @test colsum[10] / colsum[9] > 3

    # EX11: the argmax moves by more than 0.1 nm from bin 9 to bin 10.
    @info "EX11" move = abs(argmax_nm[10] - argmax_nm[9])
    @test abs(argmax_nm[10] - argmax_nm[9]) > 0.1

    # EX12: after the shock, the column sum falls monotonically from bin 10 to bin 32.
    @test issorted(colsum[10:EX_NT]; rev=true)
end

@testset "EX13: all three build, run, and land in a sensible noise regime" begin
    # Closed finding (Step 2.8, 2026-09-11, second revision): the shock problem's pre-shock IAW
    # line is narrow and cold, and the 0.08 nm SpectralIRF crushes it far harder than the broad
    # post-shock feature that becomes the global max after the chain, so D24's 20 341 -- the peak
    # of the unconvolved streak times that problem's sensitivity scale -- is not what this test
    # measures. The bands are per problem because the three examples are not required to land at
    # the same counts, only each in a regime where the Poisson noise model is sensible.
    bands = (epw_ablation=(15_000, 25_000), iaw_multiflow=(15_000, 25_000), iaw_shock=(7_000, 13_000))
    problems = (epw_ablation_problem(; nlambda=EX_NLAMBDA, nt=EX_NT),
        iaw_multiflow_problem(; nlambda=EX_NLAMBDA, nt=EX_NT),
        iaw_shock_problem(; nlambda=EX_NLAMBDA, nt=EX_NT))
    for problem in problems
        expt = build_experiment(problem)
        M = measure(expt)
        @test all(isfinite, M)
        sd = synthesize(expt, problem.noise; rng=Xoshiro(problem.seed))
        @test all(isfinite, sd.image)
        @test all(sd.sigma .> 0)
        peak = maximum(sd.image)
        @info "EX13 peak counts" name = problem.name peak
        lo, hi = getfield(bands, Symbol(problem.name))
        @test lo <= peak <= hi
    end
end

@testset "EX14: the EPW problem's m stays inside the table's domain" begin
    problem = epw_ablation_problem(; nlambda=EX_NLAMBDA, nt=EX_NT)
    for j in 1:nbins(problem.trace)
        p = plasma(problem.trace, j)
        m = parameters(p.electrons[1].f).m
        @test 1.5 <= m <= 8.0
    end
end
