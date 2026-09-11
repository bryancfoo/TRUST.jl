# foo2026 Section III.C: the IAW shock problem, full resolution. Spec D25.
# Run: julia --project=. examples/iaw_shock.jl

using OTIS
using Random: Xoshiro

include(joinpath(@__DIR__, "problems.jl"))

function main()
    problem = iaw_shock_problem()
    expt = build_experiment(problem)

    t_ideal = @elapsed P_ideal = ideal_streak(expt)
    t_measure = @elapsed M = measure(expt)
    rng = Xoshiro(problem.seed)
    t_synth = @elapsed sd = synthesize(expt, problem.noise; rng=rng, source="iaw_shock")

    outdir = joinpath(dirname(@__DIR__), "runs")
    mkpath(outdir)
    streak_path = joinpath(outdir, "iaw_shock_streak.jld2")
    trace_path = joinpath(outdir, "iaw_shock_trace.jld2")
    write_streak(streak_path, sd; source="iaw_shock", probe=problem.probe, collection=problem.collection)
    write_trace(trace_path, problem.trace; source="iaw_shock")

    @info "iaw_shock" grid = size(P_ideal) peak_ideal_m_inv = maximum(P_ideal) peak_measured_counts = maximum(M) peak_sigma = maximum(sd.sigma) median_sigma = sort(vec(sd.sigma))[length(sd.sigma)÷2] t_ideal_s = t_ideal t_measure_s = t_measure t_synthesize_s = t_synth streak_bytes = filesize(streak_path) trace_bytes = filesize(trace_path)
    return sd
end

main()
