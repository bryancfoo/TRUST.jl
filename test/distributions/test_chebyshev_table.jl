using Test
using OTIS
using OTIS: _build_cheb_table, _cheb_table_eval, _cheb_slice, _cheb_panel_index, ChebyshevTable,
            _sg_principal_value, SG_PANEL_EDGES, SG_FIT_TOL, SG_M_LO, SG_M_HI, SG_DEGREE_START,
            SG_DEGREE_CAP_XI, SG_DEGREE_CAP_M, SG_BISECT_MAX, SG_TABLE
using Random
import ForwardDiff

# --- CT1: the general builder reproduces SG_TABLE, built through its own caps ---

@testset "CT1: the general builder reproduces SG_TABLE" begin
    t = _build_cheb_table(_sg_principal_value, SG_PANEL_EDGES, SG_FIT_TOL, SG_M_LO, SG_M_HI;
                           degree_start = SG_DEGREE_START, cap_xi = SG_DEGREE_CAP_XI,
                           cap_p = SG_DEGREE_CAP_M, bisect_max = SG_BISECT_MAX)
    @test t.edges == SG_TABLE.edges
    @test t.nxi == SG_TABLE.nxi
    @test t.offset == SG_TABLE.offset
    @test t.np == SG_TABLE.np
    @test all(t.coeffs .=== SG_TABLE.coeffs)
    @test t.xi_converged == SG_TABLE.xi_converged
    @test t.p_converged == SG_TABLE.p_converged
    @test t.p_lo == SG_TABLE.p_lo
    @test t.p_hi == SG_TABLE.p_hi
    @test t.xi_lo == SG_TABLE.xi_lo
    @test t.xi_max == SG_TABLE.xi_max
    @test t.tau == SG_TABLE.tau
    @test t.degree_start == SG_DEGREE_START
    @test t.cap_xi == SG_DEGREE_CAP_XI
    @test t.cap_p == SG_DEGREE_CAP_M
    @test t.bisect_max == SG_BISECT_MAX
end

# --- CT2: the caps are honoured (spec D15; run before writing the row so the toy discriminates) ---

_ct_pole(p, x) = 1 / (1.05 - p * x)

@testset "CT2: the caps are honoured" begin
    t8 = _build_cheb_table(_ct_pole, (0.0, 1.0), 1e-12, 0.0, 1.0;
                            degree_start = 8, cap_xi = 32, cap_p = 8, bisect_max = 4)
    t64 = _build_cheb_table(_ct_pole, (0.0, 1.0), 1e-12, 0.0, 1.0;
                             degree_start = 8, cap_xi = 32, cap_p = 64, bisect_max = 4)
    @test t8.p_converged == false
    @test t8.np == 8
    @test t64.np > 8
end

# --- CT3, CT4, CT5: a signed domain, built once and shared ---

_ct_odd(p, x) = sin(3x + p) + p * x^3

const _CT_SIGNED_TABLE = _build_cheb_table(_ct_odd, (-2.0, -1.0, 0.0, 1.0, 2.0), 1e-11, -1.0, 1.0;
                                            degree_start = 8, cap_xi = 64, cap_p = 48,
                                            bisect_max = 4)

@testset "CT3: a signed domain" begin
    @test _CT_SIGNED_TABLE.xi_lo == -2.0
    @test _CT_SIGNED_TABLE.xi_max == 2.0

    rng = Random.Xoshiro(20260926)
    for _ in 1:50
        x = -2.0 + 4.0 * rand(rng)
        p = -1.0 + 2.0 * rand(rng)
        slice = _cheb_slice(_CT_SIGNED_TABLE, p)
        tv = _cheb_table_eval(_CT_SIGNED_TABLE, slice, x)
        rv = _ct_odd(p, x)
        @test isapprox(tv, rv; rtol = 1e-9, atol = 1e-10)
    end

    # The function is not even in x: a table wrongly evaluated at abs(x) would fail here.
    slice_check = _cheb_slice(_CT_SIGNED_TABLE, 0.5)
    v_pos = _cheb_table_eval(_CT_SIGNED_TABLE, slice_check, 1.3)
    v_neg = _cheb_table_eval(_CT_SIGNED_TABLE, slice_check, -1.3)
    @test abs(v_pos - v_neg) > 0.1
end

@testset "CT4: panel index in a signed table" begin
    edges = _CT_SIGNED_TABLE.edges
    n = length(edges)
    for (i, edge) in enumerate(edges)
        below = _cheb_panel_index(_CT_SIGNED_TABLE, prevfloat(edge))
        at = _cheb_panel_index(_CT_SIGNED_TABLE, edge)
        if i == 1
            @test below == 1
            @test at == 1
        elseif i == n
            @test below == n - 1
            @test at == n - 1
        else
            @test below == i - 1
            @test at == i
        end
    end
    @test _cheb_panel_index(_CT_SIGNED_TABLE, edges[1] - 10.0) == 1
    @test _cheb_panel_index(_CT_SIGNED_TABLE, edges[end] + 10.0) == n - 1
end

@testset "CT5: allocations" begin
    slice = _cheb_slice(_CT_SIGNED_TABLE, 0.5)
    _cheb_table_eval(_CT_SIGNED_TABLE, slice, 1.3)
    @test (@allocated _cheb_table_eval(_CT_SIGNED_TABLE, slice, 1.3)) == 0

    dx = ForwardDiff.Dual{:TestTag}(1.3, 1.0)
    _cheb_table_eval(_CT_SIGNED_TABLE, slice, dx)
    @test (@allocated _cheb_table_eval(_CT_SIGNED_TABLE, slice, dx)) == 0

    dp = ForwardDiff.Dual{:TestTag}(0.5, 1.0)
    dslice = _cheb_slice(_CT_SIGNED_TABLE, dp)
    _cheb_table_eval(_CT_SIGNED_TABLE, dslice, 1.3)
    @test (@allocated _cheb_table_eval(_CT_SIGNED_TABLE, dslice, 1.3)) == 0
end
