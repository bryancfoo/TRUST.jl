# Step G.2.1: the gridded twins at full size (Phase G.2 spec D8, D10; OQ-69 item 7), the slow tier's namesake
# of test/instrument/test_examples_twins.jl. Gate "gridded twins (slow)": GW1 to GW4 at 512 × 256 (the
# two-camera problem at 256 × 1024 and 256 × 960, its own default sizes), the same checks and the same pinned
# node counts as the fast rows (test/fixtures/twins_fixture.jl).

include(joinpath(@__DIR__, "..", "fixtures", "twins_fixture.jl"))

@testset "GW1 (full size): the EPW ablation twin" begin
    _gw_check(epw_ablation_problem())
end

@testset "GW2 (full size): the IAW multi-flow twin" begin
    _gw_check(iaw_multiflow_problem())
end

@testset "GW3 (full size): the IAW shock twin" begin
    _gw_check(iaw_shock_problem())
end

@testset "GW4 (full size): the bi-Maxwellian hydrogen twin, both cameras" begin
    cams = _gw_check(bimaxwellian_hydrogen_problem())
    @test all(cams[1].grids[s] === cams[2].grids[s] for s in 1:2)
end
