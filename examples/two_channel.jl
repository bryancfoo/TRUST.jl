# Phase 4.A Step 4.A.3c: the synthetic two-channel problem, full resolution. Spec D41.
# Run: julia --project=. examples/two_channel.jl

using TRUST

include(joinpath(@__DIR__, "problems.jl"))

function main()
    problem = two_channel_problem()

    t_epw = @elapsed sd_epw = synthesize_channel(problem, :epw)
    t_iaw = @elapsed sd_iaw = synthesize_channel(problem, :iaw)

    outdir = joinpath(dirname(@__DIR__), "runs")
    mkpath(outdir)
    epw_path = joinpath(outdir, "two_channel_epw_streak.jld2")
    iaw_path = joinpath(outdir, "two_channel_iaw_streak.jld2")
    write_streak(epw_path, sd_epw; source="two_channel", probe=problem.probe, collection=problem.collection)
    write_streak(iaw_path, sd_iaw; source="two_channel", probe=problem.probe, collection=problem.collection)

    @info "two_channel: epw" grid = size(sd_epw.image) peak_counts = maximum(sd_epw.image) t_synthesize_s = t_epw streak_bytes = filesize(epw_path)
    @info "two_channel: iaw" grid = size(sd_iaw.image) peak_counts = maximum(sd_iaw.image) t_synthesize_s = t_iaw streak_bytes = filesize(iaw_path)
    return sd_epw, sd_iaw
end

main()
