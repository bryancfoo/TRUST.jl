# Step G.2.1: the gridded twins at cut-down size (Phase G.2 spec D8, D10; OQ-69 item 7), in the slow tier since
# Phase G.4's Step G.4.0 (the user, 2026-11-16, OQ-71 (6); until then the fast suite's gate "gridded twins"). Gate
# "gridded twins cut-down (slow)": GW1 to GW5, unchanged. Each problem's fitted species written on grids sized by
# size_grid at its defaults and projected bin by bin (examples/problems.jl's twin_cameras); the twin's noise-free
# measured streak within 0.1 σ of the parametric one at every pixel of every camera (test/fixtures/twins_fixture.jl).
# Cut-down is 128 × 32; the two-camera problem 128 × 32 on its EPW camera and 128 × 30 on its IAW camera (Phase G.2
# spec D8). The full-size rows are test/slow/test_examples_twins.jl's.

include(joinpath(@__DIR__, "..", "fixtures", "twins_fixture.jl"))

@testset "GW1: the EPW ablation twin" begin
    _gw_check(epw_ablation_problem(; nlambda=128, nt=32))
end

@testset "GW2: the IAW multi-flow twin" begin
    _gw_check(iaw_multiflow_problem(; nlambda=128, nt=32))
end

@testset "GW3: the IAW shock twin" begin
    _gw_check(iaw_shock_problem(; nlambda=128, nt=32))
end

@testset "GW4: the bi-Maxwellian hydrogen twin, both cameras on one pair of grids" begin
    cams = _gw_check(bimaxwellian_hydrogen_problem(; nlambda_epw=128, nlambda_iaw=128, nt_epw=32, nt_iaw=30))
    @test [c.camera for c in cams] == [:epw, :iaw]
    @test all(cams[1].grids[s] === cams[2].grids[s] for s in 1:2)
    tw = cams[1].gridded.trace
    @test tw.template.electrons.names == ("e",) && tw.template.ions.names == ("H",)
end

@testset "GW5: the bi-Maxwellian hydrogen problem as specified" begin
    p = bimaxwellian_hydrogen_problem()
    @test p.name == "bimaxwellian_hydrogen"
    @test p.epw.seed == 20261113 && p.iaw.seed == 20261114
    @test p.epw.t_shift_ns == 0.0 && p.iaw.t_shift_ns == 0.05
    @test p.epw.t_ns_recorded[[1, end]] == [2.0, 5.8] && length(p.epw.t_ns_recorded) == 1024
    @test p.iaw.t_ns_recorded[[1, end]] == [1.95, 5.75] && length(p.iaw.t_ns_recorded) == 960
    @test p.epw.lambda[[1, end]] == [220e-9, 260e-9] && p.iaw.lambda[[1, end]] == [262.0e-9, 264.5e-9]
    @test p.template.electrons.names == ("cold", "hot") && p.template.ions.names == ("H", "Hb")
    @test all(sp.mass === 1.008 * TRUST.ATOMIC_MASS_UNIT && sp.charge === 1.0 for sp in p.template.ions)
    # OQ-69 item 7's profiles, at t = 2 and 4 ns, through the trace (eV, km/s, fractions)
    tr = PlasmaTrace(p.template; t_ns=[2.0, 4.0], leaves=p.leaves)
    q2, q4 = plasma(tr, 1), plasma(tr, 2)
    eV(T) = T_from_si(T, :T_eV)
    @test eV(q2.electrons["cold"].T) ≈ 300.0 && eV(q4.electrons["cold"].T) ≈ 300 + 200 * (1 - exp(-2.0))
    @test eV(q2.electrons["hot"].T) ≈ 1500.0 && eV(q4.electrons["hot"].T) ≈ 1000 + 500 * exp(-2.0 / 1.35)
    @test q4.electrons["hot"].fraction ≈ 0.3 && q4.electrons["cold"].fraction ≈ 0.7
    @test q2.electrons["hot"].fraction ≈ 0.05 + 0.25 * exp(-4.0)
    @test eV(q4.ions["H"].T) ≈ 150 + 350 * (1 - exp(-2.0)) && q4.ions["H"].u == 0.0
    @test eV(q4.ions["Hb"].T) ≈ 100 + 300 * 2.0 / 3.8 && u_from_si(q4.ions["Hb"].u, :u_kms) ≈ 300 * exp(-2.0 / 1.2)
    @test q4.ions["H"].fraction == 0.7 && q4.ions["Hb"].fraction == 0.3
    @test u_from_si(q2.ions["Hb"].u, :u_kms) ≈ 300.0
end
