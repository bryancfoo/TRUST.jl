# Step 3.C.4: SG_DM_TABLE, the super-Gaussian's m-partial table, and the fast dual-m call
# (spec D3, D4, D6, D7, D8, D9). conventions.md 8.1: this adds tests to a component Phase 1.SG
# already covers, so it gets its own file. DM1-DM15, exactly as
# dev/phase3C_instructions.md Section 7's Test table specifies them.

using Test
using TRUST
using TRUST: SG_TABLE, SG_DM_TABLE, _cheb_slice, _cheb_table_eval, _sg_dphi_dm_reference,
            _sg_x0, _sg_C, _sg_uncached, _widen_shape, _spectrum_eltype
using Random
import ForwardDiff
using SpecialFunctions: gamma, digamma, expint
using QuadGK: quadgk

# --- DM1, DM2: structure and convergence flags ---

@testset "DM1: the table's structure" begin
    @test issorted(SG_DM_TABLE.edges) && allunique(SG_DM_TABLE.edges)
    @test SG_DM_TABLE.edges[1] == 0.0
    @test SG_DM_TABLE.edges[end] == SG_DM_TABLE.xi_max
    @test length(SG_DM_TABLE.nxi) == length(SG_DM_TABLE.offset) == length(SG_DM_TABLE.edges) - 1
    LM = sum(SG_DM_TABLE.nxi .+ 1)
    @test LM == length(SG_DM_TABLE.coeffs) ÷ (SG_DM_TABLE.np + 1)
    @test SG_DM_TABLE.p_lo == SG_TABLE.p_lo
    @test SG_DM_TABLE.p_hi == SG_TABLE.p_hi
    @test SG_DM_TABLE.xi_max == SG_TABLE.xi_max
end

@testset "DM2: no direction stopped while failing" begin
    # Corrected 2026-09-08 round 1, reused here (spec D4): a cap merely reached is not a
    # finding, only a flag that is false -- a direction that stopped while still failing the
    # trailing-coefficient check -- is. See TB3 for the Cauchy table's own version of this row.
    @test all(SG_DM_TABLE.xi_converged)
    @test SG_DM_TABLE.p_converged
end

# --- DM3, DM4: the table's own accuracy, and that the certification covers the domain ---

const _DM_SEED = 20260918

function _dm_points()
    rng = Random.Xoshiro(_DM_SEED)
    edges = SG_DM_TABLE.edges
    xi_max = SG_DM_TABLE.xi_max
    m_lo, m_hi = SG_DM_TABLE.p_lo, SG_DM_TABLE.p_hi
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
        m = m_lo + rand(rng) * 0.5
        near_edge(xi) && continue
        push!(xis, xi)
        push!(ms, m)
    end
    return xis, ms
end

const _DM_X, _DM_M = _dm_points()

@testset "DM3: the table against the generator" begin
    worst = 0.0
    worst_at = (0.0, 0.0)
    for i in eachindex(_DM_X)
        x = _DM_X[i]
        m = _DM_M[i]
        slice = _cheb_slice(SG_DM_TABLE, m)
        tv = _cheb_table_eval(SG_DM_TABLE, slice, x)
        rv = _sg_dphi_dm_reference(m, x)
        err = abs(tv - rv) / max(abs(rv), 1e-9)
        if err > worst
            worst = err
            worst_at = (x, m)
        end
        @test isapprox(tv, rv; rtol = 1e-9, atol = 1e-9)
    end
    @info "DM3 worst normalized error" worst worst_at
end

@testset "DM4: DM3 covers the domain" begin
    npanels = length(SG_DM_TABLE.edges) - 1
    counts = zeros(Int, npanels)
    for x in _DM_X
        p = clamp(searchsortedlast(SG_DM_TABLE.edges, x), 1, npanels)
        counts[p] += 1
    end
    @test all(>=(10), counts)

    # The four corners, explicitly, at the table's own tolerance.
    corners = ((0.0, SG_DM_TABLE.p_lo), (0.0, SG_DM_TABLE.p_hi),
               (SG_DM_TABLE.xi_max, SG_DM_TABLE.p_lo), (SG_DM_TABLE.xi_max, SG_DM_TABLE.p_hi))
    for (x, m) in corners
        slice = _cheb_slice(SG_DM_TABLE, m)
        tv = _cheb_table_eval(SG_DM_TABLE, slice, x)
        rv = _sg_dphi_dm_reference(m, x)
        @test isapprox(tv, rv; rtol = 1e-9, atol = 1e-9)
    end

    # Both edges in m, across every panel boundary in xi.
    for x in SG_DM_TABLE.edges, m in (SG_DM_TABLE.p_lo, SG_DM_TABLE.p_hi)
        slice = _cheb_slice(SG_DM_TABLE, m)
        tv = _cheb_table_eval(SG_DM_TABLE, slice, x)
        rv = _sg_dphi_dm_reference(m, x)
        @test isapprox(tv, rv; rtol = 1e-9, atol = 1e-9)
    end
end

# --- DM5: the independent m = 2 oracle (rule 8.2: expint shares nothing with _dgamma_da) ---

@testset "DM5: the independent oracle at m = 2" begin
    m = 2.0
    a = 2 / m   # = 1
    x0 = _sg_x0(m)
    C = _sg_C(m)
    x0p_over_x0 = (5 * digamma(5 / m) - 3 * digamma(3 / m)) / (2 * m^2)
    Cp_over_C = (9 * digamma(3 / m) - 5 * digamma(5 / m)) / (2 * m^2)
    slice = _cheb_slice(SG_DM_TABLE, m)
    for x in _DM_X
        t = abs(x / x0)^m
        Gamma_a = exp(-t) * log(t) + expint(t)     # Gamma_a(1, t), independent of _dgamma_da
        Gamma_t = -t^(a - 1) * exp(-t)
        dt_dm = t * (log(abs(x / x0)) - m * x0p_over_x0)
        Gamma_val = gamma(a, t)
        closed = Cp_over_C * C * Gamma_val + C * (Gamma_a * (-2 / m^2) + Gamma_t * dt_dm)
        tv = _cheb_table_eval(SG_DM_TABLE, slice, x)
        @test isapprox(tv, closed; rtol = 1e-9, atol = 1e-9)
    end
end

# --- DM6: the BigFloat oracle, computing both integrals itself (rule 8.2, 8.6, 8.8) ---

_dm_x0_big(m) = sqrt(3 * gamma(3 / big(m)) / (2 * gamma(5 / big(m))))
_dm_C_big(m) = 1 / (2 * _dm_x0_big(m) * gamma(3 / big(m)))

function _dm_oracle_big(x, m)
    mb = big(m)
    a = 2 / mb
    x0 = _dm_x0_big(m)
    C = _dm_C_big(m)
    xb = big(x)
    t = abs(xb / x0)^mb
    Gamma_val, _ = quadgk(s -> s^(a - 1) * exp(-s), t, t + 1, big(Inf);
                           rtol = big(1e-30), atol = big(1e-60))
    Gamma_a, _ = quadgk(s -> s^(a - 1) * log(s) * exp(-s), t, t + 1, big(Inf);
                         rtol = big(1e-30), atol = big(1e-60))
    Gamma_t = -t^(a - 1) * exp(-t)
    x0p_over_x0 = (5 * digamma(5 / mb) - 3 * digamma(3 / mb)) / (2 * mb^2)
    Cp_over_C = (9 * digamma(3 / mb) - 5 * digamma(5 / mb)) / (2 * mb^2)
    dt_dm = t * (log(abs(xb / x0)) - mb * x0p_over_x0)
    return Cp_over_C * C * Gamma_val + C * (Gamma_a * (-2 / mb^2) + Gamma_t * dt_dm)
end

@testset "DM6: the BigFloat oracle" begin
    # Twelve points spanning the panels and both m edges (rule 8.6: a handful, never a grid).
    pts = ((0.1, 1.5), (0.1, 8.0), (0.3, 1.5), (0.3, 8.0), (0.4, 3.5), (0.7, 5.0),
           (1.5, 1.5), (1.5, 8.0), (3.0, 3.5), (6.0, 5.0), (9.0, 1.5), (9.0, 8.0))
    setprecision(BigFloat, 128) do
        for (x, m) in pts
            oracle = Float64(_dm_oracle_big(x, m))
            ref = _sg_dphi_dm_reference(m, x)
            @test isapprox(ref, oracle; rtol = 1e-9)
        end
    end
end

# --- DM7: panel joins ---

@testset "DM7: the interpolant is continuous at every panel join" begin
    interior_edges = SG_DM_TABLE.edges[2:end-1]
    for edge in interior_edges, m in (1.5, 2.0, 5.0, 8.0)
        slice = _cheb_slice(SG_DM_TABLE, m)
        v_below = _cheb_table_eval(SG_DM_TABLE, slice, prevfloat(edge))
        v_above = _cheb_table_eval(SG_DM_TABLE, slice, edge)
        @test isapprox(v_below, v_above; rtol = 1e-9, atol = 1e-9)
    end
end

# --- DM8: the row the whole design turns on ---

@testset "DM8: the assembled m-partial, through the real call" begin
    ms = (1.5, 2.0, 3.5, 5.0, 8.0)
    xs = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 9.9)
    for m in ms, x in xs
        fast = ForwardDiff.derivative(mm -> SuperGaussian(mm)(x), m)
        ref = ForwardDiff.derivative(mm -> _sg_uncached(mm)(x), m)
        @test isapprox(fast, ref; rtol = 1e-9, atol = 1e-9)
    end
end

# --- DM9: x = 0 under a dual m, in closed form (OQ-9 trap 2) ---

@testset "DM9: x = 0 under a dual m" begin
    for m in (1.5, 2.0, 3.5, 5.0, 8.0)
        ad = ForwardDiff.derivative(mm -> SuperGaussian(mm)(0.0), m)
        a = 2 / m
        C = _sg_C(m)
        Cp_over_C = (9 * digamma(3 / m) - 5 * digamma(5 / m)) / (2 * m^2)
        closed = Cp_over_C * C * gamma(a) - (2 * C / m^2) * gamma(a) * digamma(a)
        @test isapprox(ad, closed; rtol = 1e-9)
    end
end

# --- DM10: outside the domain, the reference runs (loose by design: this is the quadrature
# fallback against itself through a different call shape) ---

@testset "DM10: outside the domain, the reference runs" begin
    for (m, x) in ((1.2, 2.0), (3.0, 12.0))
        fast = ForwardDiff.derivative(mm -> SuperGaussian(mm)(x), m)
        ref = ForwardDiff.derivative(mm -> _sg_uncached(mm)(x), m)
        @test isfinite(fast)
        @test isapprox(fast, ref; rtol = 1e-8, atol = 1e-8)
    end
end

# --- DM11, DM12: the deterministic half of the whole phase ---

@testset "DM11: phi at a dual m allocates nothing" begin
    fd = SuperGaussian(ForwardDiff.Dual{:DM11}(3.0, 1.0))
    _call() = fd(1.3)
    _call()
    @test (@allocated _call()) == 0
end

@testset "DM12: spectrum! at a dual m allocates nothing" begin
    probe = ProbeBeam(lambda_probe_nm = 263.25)
    collection = Collection(theta_deg = 60, k_model = :exact)
    lambda = collect(range(235e-9, 255e-9; length = 512))
    m_dual = ForwardDiff.Dual(3.0, 1.0)
    p = Plasma(n_e_cm3 = 5e19,
               electrons = (electrons(T_eV = 500.0, f = SuperGaussian(m_dual)),),
               ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
    P = similar(lambda, _spectrum_eltype(p, probe, collection, lambda))
    _spec!() = spectrum!(P, p, probe, collection, lambda)
    _spec!()
    @test (@allocated _spec!()) == 0
end

# --- DM13: _widen_shape copies both slices ---

@testset "DM13: _widen_shape copies both slices" begin
    f = SuperGaussian(3.0)
    g = _widen_shape(f, ForwardDiff.Dual{:DM13a,Float64,1})
    @test typeof(g).parameters[2] == typeof(f).parameters[2]
    @test typeof(g).parameters[3] == typeof(f).parameters[3] == 0
    @test g.dm_slice === ()
    @test all(ForwardDiff.value.(g.slice) .=== f.slice)

    fd = SuperGaussian(ForwardDiff.Dual{:DM13b}(3.0, 1.0))
    gd = _widen_shape(fd, ForwardDiff.Dual{:DM13c,typeof(fd.m),1})
    @test typeof(gd).parameters[3] == typeof(fd).parameters[3]
    @test length(gd.dm_slice) == length(fd.dm_slice)
    @test all(ForwardDiff.value.(gd.dm_slice) .=== fd.dm_slice)
    @test all(ForwardDiff.value.(gd.slice) .=== fd.slice)

    f0 = _sg_uncached(3.0)
    g0 = _widen_shape(f0, ForwardDiff.Dual{:DM13d,Float64,1})
    @test g0.slice === ()
    @test g0.dm_slice === ()
end

# --- DM14: reconstruct allocates nothing at a dual m ---

@testset "DM14: reconstruct allocates nothing at a dual m" begin
    f = SuperGaussian(3.0)
    v = (m = ForwardDiff.Dual{:DM14}(3.5, 1.0),)
    _rc() = reconstruct(f, v)
    _rc()
    @test (@allocated _rc()) == 0
end

# --- DM15: a nested dual falls back, and is right (a safety net, not a product) ---

@testset "DM15: a nested dual in m falls back, and is right" begin
    x = 1.3
    m = 3.0
    h = 1e-4
    g(mm) = ForwardDiff.derivative(m2 -> SuperGaussian(m2)(x), mm)
    ad = ForwardDiff.derivative(m1 -> ForwardDiff.derivative(m2 -> SuperGaussian(m2)(x), m1), m)
    fd = (g(m + h) - g(m - h)) / (2h)
    @test isapprox(ad, fd; rtol = 1e-5)
end
