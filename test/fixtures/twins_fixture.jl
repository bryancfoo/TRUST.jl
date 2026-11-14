# Phase G.2 spec D10 (OQ-69 item 7): the gridded twins' checks, shared by the fast suite's GW rows (cut-down,
# test/instrument/test_examples_twins.jl) and the slow tier's (full size, test/slow/test_examples_twins.jl).
# The twin is examples/problems.jl's twin_cameras at size_grid's defaults; its oracle is the parametric
# problem's own measured streak, which shares no code with the comb (conventions.md 8.2).

using Test
using TRUST

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

# Each grid's node count at size_grid's defaults (spacing 0.2, extent 4, width 1.2; the user's U1), in the
# order of twin_sources (spec D10). Exact: the rule of spec D6 on the problem's own extremes of T and u, the
# same at every size, since the extremes fall on bins every axis holds.
const _GW_NODES = Dict("epw_ablation" => [77], "iaw_multiflow" => [107, 133], "iaw_shock" => [123, 185],
                       "bimaxwellian_hydrogen" => [91, 101])

# The pass mark, the user's (OQ-69 item 7): the twin's noise-free streak within a tenth of σ of the parametric
# one at every pixel. Deliberately in the data's units, not the grid's.
const _GW_MARK = 0.1

function _gw_check(problem)
    cams = twin_cameras(problem)
    for cam in cams
        nodes = [length(g.w0) for g in cam.grids]
        w = twin_worst_sigma(cam)
        errc = [maximum(e) for e in cam.projection.error_cauchy]
        @info "GW: $(problem.name), camera $(cam.camera)" nodes worst_sigma = w.worst at = w.at max_error_cauchy = errc
        @test nodes == _GW_NODES[problem.name]
        @test all(all, cam.projection.converged)
        @test w.worst <= _GW_MARK
    end
    return cams
end
