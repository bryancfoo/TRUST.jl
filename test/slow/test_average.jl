using Test
using OTIS
using OTIS: _leaf_si_path
using Random: Xoshiro
using Statistics: median

# SA1, SA2 against dev/phase4A_instructions.md Section 7 (spec D20 item 2, as amended;
# Section 7's SA1 row, made decision-free 2026-09-23), gated "streak averaging (slow)". A
# synthetic 1024-slice EPW streak averaged to 32 bins, fitted through the deck path end to end --
# the row that proves the whole path (synthesize, average, build, fit) works at experimental
# resolution.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const _SA_EXAMPLES_DIR = joinpath(@__DIR__, "..", "..", "examples")

"""
One leaf's ground-truth row from a `PlasmaTrace`, converted from SI into the boundary-keyword
unit a deck fits it through -- copied from `test/slow/test_reproduce.jl`'s `_fr_truth`, which
cannot be imported across the `@safetestset` boundary (spec Section 7, SA1).
"""
function _sa_truth(trace::PlasmaTrace, boundary_key::AbstractString)
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    i === nothing && error("_sa_truth: leaf \"$si_path\" not present in this trace")
    return collect(Float64, trace.values[i, :]) ./ factor
end

# Linear interpolation of (x, y) at each of xq, clamped at the ends (xq is expected to lie
# strictly inside [x[1], x[end]] here: the averaging region [2.0, 5.6] ns matches
# epw_ablation_problem's own t_ns range exactly, so every averaged bin center falls inside it).
function _sa_interp(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    out = Vector{Float64}(undef, length(xq))
    for (i, xi) in enumerate(xq)
        j = searchsortedlast(x, xi)
        if j <= 0
            out[i] = y[1]
        elseif j >= length(x)
            out[i] = y[end]
        else
            frac = (xi - x[j]) / (x[j+1] - x[j])
            out[i] = y[j] + frac * (y[j+1] - y[j])
        end
    end
    return out
end

function _sa_epw()
    @info "SA1: synthesizing the 256x1024 EPW streak"
    flush(stderr)
    problem_nt = epw_ablation_problem(; nlambda=256, nt=1024)
    expt_truth = build_experiment(problem_nt)
    sd = synthesize(expt_truth, problem_nt.noise; rng=Xoshiro(problem_nt.seed),
        source="epw_ablation_256x1024")
    @info "SA1: synthesis done"
    flush(stderr)

    return mktempdir() do dir
        sd_path = joinpath(dir, "epw_ablation_256x1024_streak.jld2")
        write_streak(sd_path, sd; source="epw_ablation_256x1024")
        @info "SA1: streak written" sd_path
        flush(stderr)

        deck0 = read_deck(joinpath(_SA_EXAMPLES_DIR, "epw_ablation.toml"))
        sections = deepcopy(deck0.sections)
        sections["data"] = OTIS._validate_data_section(Dict{String,Any}(
            "file" => sd_path,
            "lambda_range_nm" => [220.0, 260.0],
            "regions" => [Dict{String,Any}("t_range_ns" => [2.0, 5.6], "nbins" => 32)]))
        deck = OTIS.Deck(deck0.deck_version, sections)
        problem, optimizer, start = OTIS.build_problem(deck)
        @info "SA1: problem built" npixels = problem.npixels naxis = length(problem.experiment.t)
        flush(stderr)

        t0 = time()
        result = solve(problem, optimizer; start=start)
        elapsed = time() - t0
        @info "SA1: fit done" elapsed_s = elapsed converged = result.converged status = result.status
        flush(stderr)

        truth_ne = _sa_interp(problem_nt.trace.axis, _sa_truth(problem_nt.trace, "n_e_cm3"),
            problem.experiment.t)
        truth_Te = _sa_interp(problem_nt.trace.axis, _sa_truth(problem_nt.trace, "electrons.e.T_eV"),
            problem.experiment.t)

        fit_ne = 10 .^ result.values["logn"]
        fit_Te = result.values["T_e"]

        rel_ne = median(abs.(fit_ne .- truth_ne) ./ truth_ne)
        rel_Te = median(abs.(fit_Te .- truth_Te) ./ truth_Te)

        (; result, rel_ne, rel_Te, elapsed)
    end
end

const _SA_EPW = _sa_epw()

@testset "SA1: a 1024-slice EPW streak averaged to 32 bins fits to convergence" begin
    @info "SA1: medians" rel_ne = _SA_EPW.rel_ne rel_Te = _SA_EPW.rel_Te converged = _SA_EPW.result.converged
    @test _SA_EPW.result.converged == true
    @test _SA_EPW.rel_ne <= 0.2
    @test _SA_EPW.rel_Te <= 0.2
end

@testset "SA2: the fit's wall time (reported, never asserted)" begin
    @info "SA2: wall time" elapsed_s = _SA_EPW.elapsed
    @test true
end
