using Test
using OTIS
using OTIS: SG_TABLE, _cheb_slice, _cheb_table_eval,
            _sg_principal_value, _cheb_nodes, _cheb_coeffs, _cheb_eval, _cheb_map
using Random
import ForwardDiff

# --- TB1: the Chebyshev convention (write first, per spec Pitfall 8) ---

@testset "TB1: Chebyshev convention pins the halving" begin
    rng = Random.Xoshiro(20260908)
    for n in (8, 16)
        nodes = _cheb_nodes(n)
        values = [cos(3 * acos(t)) for t in nodes]   # T_3(t)
        c = _cheb_coeffs(values)
        @test isapprox(c[4], 1.0; atol = 1e-14)       # c-tilde_3, 1-based index 4
        others = [c[j] for j in eachindex(c) if j != 4]
        @test maximum(abs, others) <= 1e-14

        for _ in 1:200
            t = 2 * rand(rng) - 1
            @test isapprox(_cheb_eval(c, 0, n, t), cos(3 * acos(t)); atol = 1e-14)
        end
    end
end

@testset "TB2: interpolation exact at the nodes" begin
    n = 24
    nodes = _cheb_nodes(n)
    testf(t) = exp(0.5 * t) * cos(2t)
    values = testf.(nodes)
    c = _cheb_coeffs(values)
    for t in nodes
        @test isapprox(_cheb_eval(c, 0, n, t), testf(t); atol = 1e-13)
    end
end

@testset "TB3: table structure and convergence" begin
    @test SG_TABLE.edges[1] == 0.0
    @test SG_TABLE.edges[end] == SG_TABLE.xi_max
    @test issorted(SG_TABLE.edges) && allunique(SG_TABLE.edges)
    @test SG_TABLE.p_lo == 1.5
    @test SG_TABLE.p_hi == 8.0
    @test length(SG_TABLE.nxi) == length(SG_TABLE.offset) == length(SG_TABLE.edges) - 1
    L = sum(SG_TABLE.nxi .+ 1)
    @test L == length(SG_TABLE.coeffs) ÷ (SG_TABLE.np + 1)
    # Corrected 2026-09-08, round 1: assert on the converged flags (spec D9), not on
    # nxi[p] < cap -- a panel that reaches its cap and passes the trailing-coefficient check
    # there is a fine panel; only a flag that is false (a direction stopped while still
    # failing) is the finding OQ-31 item 2 asks for.
    @test length(SG_TABLE.xi_converged) == length(SG_TABLE.nxi)
    @test all(SG_TABLE.xi_converged)
    @test SG_TABLE.p_converged
end

const _rng_seed = 20260908

function _tb_points()
    rng = Random.Xoshiro(_rng_seed)
    edges = SG_TABLE.edges
    xi_max = SG_TABLE.xi_max
    m_lo, m_hi = SG_TABLE.p_lo, SG_TABLE.p_hi
    near_edge(xi) = any(abs(xi - e) < 1e-9 for e in edges)
    xis = Float64[]
    ms = Float64[]
    while length(xis) < 400
        xi = rand(rng) * xi_max
        m = m_lo + rand(rng) * (m_hi - m_lo)
        near_edge(xi) && continue
        push!(xis, xi)
        push!(ms, m)
    end
    while length(xis) < 500
        xi = rand(rng) * 0.5
        m = 1.5 + rand(rng) * 0.5
        near_edge(xi) && continue
        push!(xis, xi)
        push!(ms, m)
    end
    return xis, ms
end

const _TB_XI, _TB_M = _tb_points()

@testset "TB4: value certification against the generator" begin
    worst = 0.0
    for i in eachindex(_TB_XI)
        xi = _TB_XI[i]
        m = _TB_M[i]
        f = SuperGaussian(m)
        tv = _cheb_table_eval(SG_TABLE, f.slice, xi)
        rv = _sg_principal_value(m, xi)
        worst = max(worst, abs(tv - rv) / max(abs(rv), 1e-8))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
end

@testset "TB5: m=2 against Zprime" begin
    f = SuperGaussian(2.0)
    for i in eachindex(_TB_XI)
        xi = _TB_XI[i]
        tv = _cheb_table_eval(SG_TABLE, f.slice, xi)
        zp = real(Zprime(xi))
        @test isapprox(tv, zp; rtol = 1e-8, atol = 1e-8)
    end
end

@testset "TB6: the deterministic edge set" begin
    xis = vcat(SG_TABLE.edges...)
    ms = (SG_TABLE.p_lo, 2.0, 4.75, SG_TABLE.p_hi)
    for xi in xis, m in ms
        f = SuperGaussian(m)
        tv = _cheb_table_eval(SG_TABLE, f.slice, xi)
        rv = _sg_principal_value(m, xi)
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
end

function _tb7_data()
    h = 1e-4
    fd = Vector{Float64}(undef, length(_TB_XI))
    ad = Vector{Float64}(undef, length(_TB_XI))
    for i in eachindex(_TB_XI)
        xi = _TB_XI[i]
        m = _TB_M[i]
        f = SuperGaussian(m)
        ad[i] = ForwardDiff.derivative(x -> _cheb_table_eval(SG_TABLE, f.slice, x), xi)
        fd[i] = (_sg_principal_value(m, xi + h) - _sg_principal_value(m, xi - h)) / (2h)
    end
    return ad, fd
end

const _TB7_AD, _TB7_FD = _tb7_data()
const _TB7_FLOOR = 1e-6 * maximum(abs, _TB7_FD)

@testset "TB7: dR/dxi against central differences of the generator" begin
    for i in eachindex(_TB_XI)
        @test isapprox(_TB7_AD[i], _TB7_FD[i]; rtol = 1e-6, atol = _TB7_FLOOR)
    end
end

@testset "TB8: dR/dm against central differences of the generator" begin
    h = 1e-4
    fd = Vector{Float64}(undef, length(_TB_XI))
    ad = Vector{Float64}(undef, length(_TB_XI))
    for i in eachindex(_TB_XI)
        xi = _TB_XI[i]
        m = _TB_M[i]
        ad[i] = ForwardDiff.derivative(mm -> _cheb_table_eval(SG_TABLE, _cheb_slice(SG_TABLE, mm), xi), m)
        fd[i] = (_sg_principal_value(m + h, xi) - _sg_principal_value(m - h, xi)) / (2h)
    end
    extra_xi = (0.3, 1.0, 3.0, 7.0)
    extra_fd = Float64[]
    extra_ad = Float64[]
    for m0 in (SG_TABLE.p_lo, SG_TABLE.p_hi), xi in extra_xi
        push!(extra_ad, ForwardDiff.derivative(mm -> _cheb_table_eval(SG_TABLE, _cheb_slice(SG_TABLE, mm), xi), m0))
        push!(extra_fd, (_sg_principal_value(m0 + h, xi) - _sg_principal_value(m0 - h, xi)) / (2h))
    end
    all_fd = vcat(fd, extra_fd)
    floor_ = 1e-6 * maximum(abs, all_fd)
    for i in eachindex(_TB_XI)
        @test isapprox(ad[i], fd[i]; rtol = 1e-6, atol = floor_)
    end
    for i in eachindex(extra_ad)
        @test isapprox(extra_ad[i], extra_fd[i]; rtol = 1e-6, atol = floor_)
    end
end

@testset "TB9: panel joins" begin
    # Adjacent panels share their endpoint node exactly (spec D15): compare the value AT the
    # node from each side (prevfloat(edge) picked up by the panel ending there, edge itself by
    # the panel starting there), not at an offset -- an offset of h picks up ~2h*R'(edge), which
    # swamps a round-off-level tolerance long before it says anything about the shared node.
    interior_edges = SG_TABLE.edges[2:end-1]
    for edge in interior_edges, m in (1.5, 2.0, 5.0, 8.0)
        f = SuperGaussian(m)
        v_below = _cheb_table_eval(SG_TABLE, f.slice, prevfloat(edge))
        v_above = _cheb_table_eval(SG_TABLE, f.slice, edge)
        @test isapprox(v_below, v_above; rtol = 1e-12, atol = 1e-12)

        d_below = ForwardDiff.derivative(x -> _cheb_table_eval(SG_TABLE, f.slice, x), edge - 1e-8)
        d_above = ForwardDiff.derivative(x -> _cheb_table_eval(SG_TABLE, f.slice, x), edge + 1e-8)
        # Clarified 2026-09-08, round 2: the floor is TB7's own set-level floor, `_TB7_FLOOR`,
        # not one recomputed per edge (conventions.md 8.4: one factor times the largest
        # finite-difference derivative over the grid, so a point where the derivative happens to
        # be small is not held to round-off).
        @test isapprox(d_below, d_above; rtol = 1e-6, atol = _TB7_FLOOR)
    end
end

@testset "TB10: the join with the series at |xi| = xi_max" begin
    for m in (1.5, 2.0, 5.0, 8.0)
        f = SuperGaussian(m)
        tv = _cheb_table_eval(SG_TABLE, f.slice, SG_TABLE.xi_max)
        series_val, converged = OTIS._cauchy_series(f, SG_TABLE.xi_max)
        @test converged
        @test isapprox(tv, series_val; rtol = 1e-8, atol = 1e-8)
    end
end

@testset "TB11: xi = 0 sanity bound" begin
    # Rewritten 2026-09-08, round 1, then cut back to one clause round 2 (spec D12, and the
    # note under Section 4's test table). Round 1 kept clause (c), a coarse sanity bound at
    # xi=0 that may NOT drive tau (only TB7/TB8 may, per D12), and added two evenness clauses,
    # (a) and (b). Round 2 removed (a) and (b): _cheb_table_eval never applies abs() -- it takes
    # |xi| >= 0 by contract, and the abs() that makes the shape's Cauchy integral even lives in
    # cauchy_dfdv, which is Step 1.SG.3's. Written against _cheb_table_eval, (a) compared
    # abs(xi) with abs(-xi) -- the same Float64 -- and asserted v == v, and (b) exercised
    # ForwardDiff's own abs() rule inside the test's closure; neither could test this package at
    # this step. The real evenness test is DP8 in Step 1.SG.3, against cauchy_dfdv, where the
    # identity is the package's own and can fail.
    ms = (1.5, 2.0, 5.0, 8.0)

    # (c) a coarse sanity bound on |R'(0)|, 100x TB7's floor, deliberately loose and barred
    # from driving tau (D12): a genuinely broken panel selection or transposed tensor would
    # put this at O(1), not at a few times TB7's floor.
    for m in ms
        f = SuperGaussian(m)
        d0 = ForwardDiff.derivative(x -> _cheb_table_eval(SG_TABLE, f.slice, abs(x)), 0.0)
        @test abs(d0) <= 100 * _TB7_FLOOR
    end
end

@testset "TB12: allocations" begin
    f = SuperGaussian(3.0)
    _cheb_table_eval(SG_TABLE, f.slice, 1.3)
    @test (@allocated _cheb_table_eval(SG_TABLE, f.slice, 1.3)) == 0

    dxi = ForwardDiff.Dual{:TestTag}(1.3, 1.0)
    _cheb_table_eval(SG_TABLE, f.slice, dxi)
    @test (@allocated _cheb_table_eval(SG_TABLE, f.slice, dxi)) == 0

    dm = ForwardDiff.Dual{:TestTag}(3.0, 1.0)
    fd = SuperGaussian(dm)
    _cheb_table_eval(SG_TABLE, fd.slice, 1.3)
    @test (@allocated _cheb_table_eval(SG_TABLE, fd.slice, 1.3)) == 0

    v = (m = 3.5,)
    reconstruct(f, v)
    @test (@allocated reconstruct(f, v)) == 0
end

@testset "TB13: the error map" begin
    # Clarified 2026-09-08, round 2: the assertion is TB4's, verbatim -- isapprox(tv, rv;
    # rtol = 1e-8, atol = 1e-8), nothing tighter. The normalized ratio abs(tv-rv)/max(abs(rv),
    # 1e-8) is a pure relative test with the floor folded into the denominator instead of the
    # tolerance, which demands far more accuracy than the target near the ξ≈0.92, m=2 crossing
    # (where rv is near zero) and is unsatisfiable there for reasons that are not a defect. It is
    # computed and reported below for the error map, never asserted on.
    panel_worst = zeros(length(SG_TABLE.edges) - 1)
    for i in eachindex(_TB_XI)
        xi = _TB_XI[i]
        m = _TB_M[i]
        f = SuperGaussian(m)
        tv = _cheb_table_eval(SG_TABLE, f.slice, xi)
        rv = _sg_principal_value(m, xi)
        p = clamp(searchsortedlast(SG_TABLE.edges, xi), 1, length(SG_TABLE.edges) - 1)
        relerr = abs(tv - rv) / max(abs(rv), 1e-8)
        panel_worst[p] = max(panel_worst[p], relerr)
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    # The 100 points near xi=0, m in [1.5,2] are _TB_XI[401:500], already asserted above (they
    # are part of eachindex(_TB_XI)); this loop only recomputes the normalized ratio for the
    # report, so it is reported and not asserted a second time.
    near0_worst = 0.0
    for i in 401:length(_TB_XI)
        xi = _TB_XI[i]
        m = _TB_M[i]
        f = SuperGaussian(m)
        tv = _cheb_table_eval(SG_TABLE, f.slice, xi)
        rv = _sg_principal_value(m, xi)
        near0_worst = max(near0_worst, abs(tv - rv) / max(abs(rv), 1e-8))
    end
    @info "TB13 error map: worst per panel" panel_worst
    @info "TB13 worst near xi=0, m in [1.5,2]" near0_worst
end
