# Phase 4.B Step 4.B.5: the three shapes' synthetic two-channel streaks, full resolution. Spec D39.
# Run: julia --project=. examples/shapes.jl

using TRUST

include(joinpath(@__DIR__, "problems.jl"))

function main()
    outdir = joinpath(dirname(@__DIR__), "runs")
    mkpath(outdir)

    for name in (:kappa, :generalized_normal, :skew_normal)
        problem = shapes_problem(name)
        for camera in (:epw, :iaw)
            t = @elapsed sd = synthesize_channel(problem, camera)
            path = joinpath(outdir, "shapes_$(name)_$(camera)_streak.jld2")
            write_streak(path, sd; source="shapes_$(name)", probe=problem.probe, collection=problem.collection)
            @info "shapes: $(name) $(camera)" grid = size(sd.image) peak_counts = maximum(sd.image) t_synthesize_s = t streak_bytes = filesize(path)
        end
    end
    return nothing
end

main()
