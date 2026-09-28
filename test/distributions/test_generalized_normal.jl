# Step 4.B.3: GeneralizedNormal, the shape (spec D31-D34, D42). GN1-GN25, gate
# "GeneralizedNormal". Section 10.2's formulas, in the code's normalization; every oracle
# independent of the code under test (conventions.md 8.2): a quadrature, a BigFloat evaluation,
# Zprime, ForwardDiff or a finite difference, never the function itself.

using Test
using OTIS
using OTIS: GN_TABLE, GN_DEGREE_START, GN_DEGREE_CAP_XI, GN_DEGREE_CAP_M, GN_BISECT_MAX,
            _gn_uncached, _gn_cauchy_reference, _gn_principal_value, _cheb_slice, _cheb_table_eval,
            _basefloat, _cauchy_upper, _widen_shape, _cauchy_series, SHAPE_REGISTRY
using SpecialFunctions: loggamma, digamma, gamma
using QuadGK: quadgk
using Random
import ForwardDiff

include(joinpath(@__DIR__, "..", "shape_contract.jl"))

# ------------------------------------------------------------------------------------------
# GN1-GN7: the shape (Build steps 2 and 4; both use the constructor, which already needs
# GN_TABLE, exactly as Kappa's own file does)
# ------------------------------------------------------------------------------------------

@testset "GN1: the three normalizations" begin
    for m in (1.5, 2.0, 3.7, 8.0, 12.0)
        check_shape_normalizations(GeneralizedNormal(m))
    end
end

@testset "GN2: x0 and C against a 128-bit BigFloat oracle (MPFR's gamma)" begin
    setprecision(BigFloat, 128) do
        for m in (1.5, 1.6, 2.0, 2.5, 3.7, 8.0, 12.0, 30.0)
            f = GeneralizedNormal(m)
            mb = big(m)
            x0b = sqrt(gamma(1 / mb) / (2 * gamma(3 / mb)))
            Cb = mb / (2 * x0b * gamma(1 / mb))
            @test isapprox(f.x0, Float64(x0b); rtol = 1e-14)
            @test isapprox(f.C, Float64(Cb); rtol = 1e-14)
        end
    end
    f2 = GeneralizedNormal(2.0)
    @test isapprox(f2.x0, 1.0; rtol = 1e-15)
    @test isapprox(f2.C, 1 / sqrt(pi); rtol = 1e-15)
end

@testset "GN3: normalization on a fine grid (an independent quadrature)" begin
    worst1 = 0.0
    worst2 = 0.0
    ms = collect(1.5:0.005:8.5)
    for m in ms
        f = _gn_uncached(m)
        I0, _ = quadgk(x -> f(x), -Inf, 0.0, Inf; rtol = 1e-13, atol = 1e-15)
        I2, _ = quadgk(x -> x^2 * f(x), -Inf, 0.0, Inf; rtol = 1e-13, atol = 1e-15)
        worst1 = max(worst1, abs(I0 - 1))
        worst2 = max(worst2, abs(I2 - 0.5))
        @test isapprox(I0, 1; rtol = 1e-12)
        @test isapprox(I2, 0.5; rtol = 1e-12)
    end
    @info "GN3 worst |I0 - 1|, |I2 - 1/2| over the grid" worst1 worst2
end

@testset "GN4: the Maxwellian at m = 2" begin
    f2 = GeneralizedNormal(2.0)
    for x in -6.0:0.01:6.0
        @test isapprox(f2(x), exp(-x^2) / sqrt(pi); rtol = 1e-14, atol = 1e-300)
        @test isapprox(dfdx(f2, x), -2x * exp(-x^2) / sqrt(pi); rtol = 1e-14, atol = 1e-300)
    end
    ad0 = ForwardDiff.derivative(x -> dfdx(f2, x), 0.0)
    @test isapprox(ad0, -2 / sqrt(pi); rtol = 1e-14)
    worst = 0.0
    for xi in -10.0:0.05:10.0
        cv = cauchy_dfdv(f2, xi)
        zp = Zprime(xi)
        worst = max(worst, abs(real(cv) - real(zp)), abs(imag(cv) - imag(zp)))
        @test isapprox(real(cv), real(zp); atol = 1e-9, rtol = 0)
        @test isapprox(imag(cv), imag(zp); atol = 1e-9, rtol = 0)
    end
    @info "GN4 worst |cauchy_dfdv(m=2) - Zprime|" worst

    # (b) Phase 4.B spec D47 item 1: a dual m exactly at 2 with a nonzero partial reads the
    # x = 0 branch's m-partial correctly only when the guard compares the primal value.
    md = ForwardDiff.Dual{:GN4}(2.0, 1.0)
    xd = ForwardDiff.Dual{:GN4}(0.0, 1.0)
    @test isapprox(ForwardDiff.partials(dfdx(GeneralizedNormal(md), xd))[1], -2 / sqrt(pi); rtol = 1e-14)
end

@testset "GN5: dfdx vs ForwardDiff, and x=0 under a dual m" begin
    xs = [x for x in -6.0:0.05:6.0 if x != 0.0]
    for m in (1.6, 2.0, 3.7, 10.0)
        f = GeneralizedNormal(m)
        @test dfdx(f, 0.0) == 0
        for x in xs
            ad = ForwardDiff.derivative(y -> f(y), x)
            @test isapprox(ad, dfdx(f, x); rtol = 1e-13, atol = 1e-15)
        end
    end
    m0 = 3.7
    ad0 = ForwardDiff.derivative(mm -> GeneralizedNormal(mm)(0.0), m0)
    @test isfinite(ad0)
    f0 = _gn_uncached(m0)
    dlnx0dm0 = (3 * digamma(3 / m0) - digamma(1 / m0)) / (2 * m0^2)
    closed0 = f0.C * (1 / m0 - dlnx0dm0 + digamma(1 / m0) / m0^2)
    @test isapprox(ad0, closed0; rtol = 1e-12)
end

@testset "GN6: moments, the derived quantities, and errors" begin
    for m in (1.5, 3.7, 10.0)
        f = GeneralizedNormal(m)
        for n in 0:6
            ref, _ = quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-12, atol = 1e-14)
            @test isapprox(moment(f, n), ref; rtol = 1e-11, atol = 1e-12)
        end
        for n in (1, 3, 5)
            @test moment(f, n) === 0.0
        end
    end
    for m in (1.5, 4.0, 10.0)
        f = GeneralizedNormal(m)
        expected = gamma(5 / m) * gamma(1 / m) / gamma(3 / m)^2 - 3
        @test isapprox(excess_kurtosis(f), expected; rtol = 1e-12)
    end
    @test isapprox(excess_kurtosis(GeneralizedNormal(2.0)), 0.0; atol = 1e-13)
    @test_throws ArgumentError moment(GeneralizedNormal(3.7), -1)
end

@testset "GN7: not the projected shape" begin
    fgn2 = GeneralizedNormal(2.0)
    fsg2 = SuperGaussian(2.0)
    for x in (0.0, 0.7, 2.0)
        @test isapprox(fgn2(x), fsg2(x); rtol = 1e-13)
    end
    fgn4 = GeneralizedNormal(4.0)
    fsg4 = SuperGaussian(4.0)
    diff4 = fgn4(0.0) - fsg4(0.0)
    @info "GN7 GeneralizedNormal(4.0)(0) - SuperGaussian(4.0)(0)" diff4
    @test abs(diff4) > 1e-2
end

# ------------------------------------------------------------------------------------------
# The table (Build step 3): GN8, GN9, GN11-GN14, GN25, then GN10 (the BigFloat oracle)
# ------------------------------------------------------------------------------------------

@testset "GN8: table structure" begin
    @test GN_TABLE.p_lo == 2.0
    @test GN_TABLE.p_hi == 8.0
    @test GN_TABLE.xi_lo == 0.0
    @test GN_TABLE.xi_max == 10.0
    @test all(GN_TABLE.xi_converged)
    @test GN_TABLE.p_converged
    @test GN_TABLE.np <= GN_DEGREE_CAP_M
    @test GN_TABLE.degree_start == GN_DEGREE_START
    @test GN_TABLE.cap_xi == GN_DEGREE_CAP_XI
    @test GN_TABLE.cap_p == GN_DEGREE_CAP_M
    @test GN_TABLE.bisect_max == GN_BISECT_MAX
    L = sum(GN_TABLE.nxi .+ 1)
    @test length(GN_TABLE.coeffs) == L * (GN_TABLE.np + 1)
    @info "GN8 table" edges = GN_TABLE.edges nxi = GN_TABLE.nxi np = GN_TABLE.np L = L
end

const _GN_RNG = Random.Xoshiro(20260927)
const _GN9_POINTS = [(2.0 + 6.0 * rand(_GN_RNG), 10.0 * rand(_GN_RNG)) for _ in 1:60]
const _GN9_NEARZERO = [(2.0 + 6.0 * rand(_GN_RNG), 0.05 * rand(_GN_RNG)) for _ in 1:40]

@testset "GN9: the table against its generator" begin
    worst_main = 0.0
    for (m, xi) in _GN9_POINTS
        slice = _cheb_slice(GN_TABLE, m)
        tv = _cheb_table_eval(GN_TABLE, slice, xi)
        rv = _gn_principal_value(m, xi)
        worst_main = max(worst_main, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    worst_near0 = 0.0
    for (m, xi) in _GN9_NEARZERO
        slice = _cheb_slice(GN_TABLE, m)
        tv = _cheb_table_eval(GN_TABLE, slice, xi)
        rv = _gn_principal_value(m, xi)
        worst_near0 = max(worst_near0, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    for edge in GN_TABLE.edges, m in (2.0, 5.0, 8.0)
        slice = _cheb_slice(GN_TABLE, m)
        tv = _cheb_table_eval(GN_TABLE, slice, edge)
        rv = _gn_principal_value(m, edge)
        worst_main = max(worst_main, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    @info "GN9 worst |table - generator|, main range and near xi=0" worst_main worst_near0
end

@testset "GN11: the Maxwellian edge against Zprime" begin
    worst = 0.0
    slice2 = _cheb_slice(GN_TABLE, 2.0)
    for xi in 0.0:0.05:10.0
        tv = _cheb_table_eval(GN_TABLE, slice2, xi)
        zp = real(Zprime(xi))
        worst = max(worst, abs(tv - zp))
        @test isapprox(tv, zp; atol = 1e-9, rtol = 0)
    end
    @info "GN11 worst |table(m=2) - Re Zprime|" worst
end

function _gn12_data()
    h = 1e-4
    ad = Vector{Float64}(undef, length(_GN9_POINTS))
    fd = Vector{Float64}(undef, length(_GN9_POINTS))
    for (i, (m, xi)) in enumerate(_GN9_POINTS)
        slice = _cheb_slice(GN_TABLE, m)
        ad[i] = ForwardDiff.derivative(x -> _cheb_table_eval(GN_TABLE, slice, x), xi)
        fd[i] = (_gn_principal_value(m, xi + h) - _gn_principal_value(m, xi - h)) / (2h)
    end
    return ad, fd
end
const _GN12_AD, _GN12_FD = _gn12_data()
const _GN12_FLOOR = 1e-6 * maximum(abs, _GN12_FD)

@testset "GN12: dR/dxi against central differences of the generator" begin
    for i in eachindex(_GN12_AD)
        @test isapprox(_GN12_AD[i], _GN12_FD[i]; rtol = 1e-6, atol = _GN12_FLOOR)
    end
end

const _GN13_M = [2.0005 + (7.9995 - 2.0005) * rand(_GN_RNG) for _ in 1:12]
const _GN13_XI = [10.0 * rand(_GN_RNG) for _ in 1:12]

@testset "GN13: dR/dm against central differences, and the chain rule" begin
    h = 1e-4
    ad = Vector{Float64}(undef, 12)
    fd = Vector{Float64}(undef, 12)
    for i in 1:12
        m = _GN13_M[i]
        xi = _GN13_XI[i]
        ad[i] = ForwardDiff.derivative(mm -> _cheb_table_eval(GN_TABLE, _cheb_slice(GN_TABLE, mm), xi), m)
        fd[i] = (_gn_principal_value(m + h, xi) - _gn_principal_value(m - h, xi)) / (2h)
    end
    floor_ = 1e-6 * maximum(abs, fd)
    worst_ratio = 0.0
    for i in 1:12
        worst_ratio = max(worst_ratio, abs(ad[i] - fd[i]) / floor_)
        @test isapprox(ad[i], fd[i]; rtol = 1e-6, atol = floor_)
    end
    @info "GN13 worst ratio to its floor" worst_ratio floor_

    for (m, xi) in ((2.5, 1.0), (3.7, 4.0), (6.0, 8.0), (7.9, 9.5))
        dRdm = ForwardDiff.derivative(mm -> _cheb_table_eval(GN_TABLE, _cheb_slice(GN_TABLE, mm), xi), m)
        lhs = ForwardDiff.derivative(mm -> real(cauchy_dfdv(GeneralizedNormal(mm), xi)), m)
        @test isapprox(lhs, dRdm; rtol = 1e-12)
    end

    # (c) Phase 4.B spec D47 item 1: the chain at the table's m edges, along +m and -m, must
    # take the table (=== the slice's own derivative), not the fallback a dual-at-a-tie
    # comparison would otherwise divert it to.
    for (m, xi) in ((2.0, 1.0), (8.0, 1.0), (8.0, 9.5)), s in (1.0, -1.0)
        lhs = s * ForwardDiff.derivative(t -> real(cauchy_dfdv(GeneralizedNormal(s * t), xi)), s * m)
        rhs = s * ForwardDiff.derivative(t -> _cheb_table_eval(GN_TABLE, _cheb_slice(GN_TABLE, s * t), xi), s * m)
        @test lhs === rhs
    end
end

@testset "GN14: panel joins" begin
    interior_edges = GN_TABLE.edges[2:end-1]
    for edge in interior_edges, m in (2.0, 5.0, 8.0)
        slice = _cheb_slice(GN_TABLE, m)
        v_below = _cheb_table_eval(GN_TABLE, slice, prevfloat(edge))
        v_above = _cheb_table_eval(GN_TABLE, slice, edge)
        @test isapprox(v_below, v_above; rtol = 1e-12, atol = 1e-12)

        d_below = ForwardDiff.derivative(x -> _cheb_table_eval(GN_TABLE, slice, x), edge - 1e-8)
        d_above = ForwardDiff.derivative(x -> _cheb_table_eval(GN_TABLE, slice, x), edge + 1e-8)
        @test isapprox(d_below, d_above; rtol = 1e-6, atol = _GN12_FLOOR)
    end
end

@testset "GN25: the table-series join at xi = 10" begin
    worst = 0.0
    for m in (2.0, 2.5, 3.0, 5.0, 8.0)
        f = GeneralizedNormal(m)
        tv = _cheb_table_eval(GN_TABLE, _cheb_slice(GN_TABLE, m), 10.0)
        sv, converged = _cauchy_series(f, nextfloat(10.0))
        @test converged
        worst = max(worst, abs(tv - sv))
        @test isapprox(tv, sv; atol = 1e-10)
    end
    @info "GN25 worst |table(xi=10) - series(nextfloat(10))|" worst
end

# ------------------------------------------------------------------------------------------
# GN26 (Phase 4.C spec D3): the constructor's guard compares the primal
# ------------------------------------------------------------------------------------------

@testset "GN26: the constructor's guard compares the primal (Phase 4.C spec D3)" begin
    for s in (1.0, -1.0)
        g = GeneralizedNormal(ForwardDiff.Dual(1.5, s))
        @test ForwardDiff.value(g.m) === 1.5
        @test ForwardDiff.partials(g.m)[1] === s
    end
    for bad in (prevfloat(1.5), ForwardDiff.Dual(prevfloat(1.5), 1.0), ForwardDiff.Dual(prevfloat(1.5), -1.0),
                Inf, NaN)
        @test_throws "m must be finite and at least 1.5" GeneralizedNormal(bad)
    end
end

# --- GN10: the independent BigFloat oracle, D34's code verbatim ---

function _gn_dphi_big(m::BigFloat, x::BigFloat)
    x0 = sqrt(gamma(1 / m) / (2 * gamma(3 / m)))
    C = m / (2 * x0 * gamma(1 / m))
    iszero(x) && return zero(x)
    t = abs(x / x0)
    return -(m / x0) * sign(x) * t^(m - 1) * C * exp(-t^m)
end
function _gn_R_big(m::Real, xi::Real)
    setprecision(BigFloat, 64) do
        mb, xb = BigFloat(m), BigFloat(xi)
        g(u) = (_gn_dphi_big(mb, xb + u) - _gn_dphi_big(mb, xb - u)) / u
        a = abs(xb)
        pts = iszero(a) ? (big(0), big(6), big(Inf)) :
              a < 6 ? (big(0), a, big(6), big(Inf)) : (big(0), big(6), a, big(Inf))
        Float64(quadgk(g, pts...; rtol = BigFloat(1e-16), atol = BigFloat(1e-25))[1])
    end
end

@testset "GN10(a): the table against the BigFloat oracle" begin
    worst = 0.0
    for m in (2.5, 6.0), xi in (0.3, 1.7, 6.0)
        tv = _cheb_table_eval(GN_TABLE, _cheb_slice(GN_TABLE, m), xi)
        rv = _gn_R_big(m, xi)
        worst = max(worst, abs(tv - rv))
        @test isapprox(tv, rv; atol = 1e-9)
    end
    @info "GN10(a) worst |table - BigFloat oracle|" worst
end

@testset "GN10(b): the fallback against the BigFloat oracle" begin
    worst = 0.0
    for (m, xi) in ((1.5, 0.0), (1.5, 0.05), (1.75, 1e-3), (1.75, 1.5))
        f = GeneralizedNormal(m)
        cv = cauchy_dfdv(f, xi)
        rv = _gn_R_big(m, xi)
        worst = max(worst, abs(real(cv) - rv))
        @test isapprox(real(cv), rv; atol = 1e-9)
    end
    @info "GN10(b) worst |fallback - BigFloat oracle|" worst
end

# ------------------------------------------------------------------------------------------
# The public constructor and the rest of the interface (Build step 4): GN15-GN20
# ------------------------------------------------------------------------------------------

@testset "GN15: the branches of cauchy_dfdv" begin
    f = GeneralizedNormal(3.5)

    # (a) inside the table's domain
    for xi in (-3.3, 0.0, 0.7, 9.9)
        QT = _basefloat(xi)
        expected = _cheb_table_eval(GN_TABLE, f.slice, abs(xi)) + im * QT(pi) * dfdx(f, xi)
        @test cauchy_dfdv(f, xi) === expected
    end

    # (b) |xi| outside the table, series converges
    xi_b = 12.0
    sv, conv_b = _cauchy_series(f, xi_b)
    @test conv_b
    @test cauchy_dfdv(f, xi_b) === sv + im * _basefloat(xi_b)(pi) * dfdx(f, xi_b)

    # (c) m outside the table, series does not converge
    for f_c in (GeneralizedNormal(1.7), GeneralizedNormal(12.0))
        xi_c = 1.3
        sv_c, conv_c = _cauchy_series(f_c, xi_c)
        @test !conv_c
        @test cauchy_dfdv(f_c, xi_c) === _gn_cauchy_reference(f_c, xi_c)
    end

    # (d) off the real axis
    @test cauchy_dfdv(f, 1.3 + 0.2im) === _cauchy_upper(f, 1.3 + 0.2im)
    @test_throws "requires Im xi >= 0" cauchy_dfdv(f, 1.3 - 0.2im)

    # (e) the uncached form throws
    fu = _gn_uncached(3.5)
    @test_throws "carries no table slice" cauchy_dfdv(fu, 1.0)

    # (f) evenness
    for xi in (0.4, 3.0, 9.0)
        @test real(cauchy_dfdv(f, -xi)) == real(cauchy_dfdv(f, xi))
        @test imag(cauchy_dfdv(f, -xi)) == -imag(cauchy_dfdv(f, xi))
    end

    # (g) Phase 4.B spec D47 item 1: a dual xi exactly at xi_max must stay in the table.
    @test ForwardDiff.derivative(x -> real(cauchy_dfdv(f, x)), 10.0) ===
          ForwardDiff.derivative(x -> _cheb_table_eval(GN_TABLE, f.slice, x), 10.0)
end

const _GN_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _GN_COLLECTION = Collection(theta_deg = 60, k_model = :exact)
const _GN_LAMBDA_S = 245e-9  # test_shape_cost.jl's SC fixture's own wavelength

_gn_iaw_plasma(f) = Plasma(n_e_cm3 = 5e19,
                            electrons = (electrons(T_eV = 500.0, f = f),),
                            ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))

function _gn_dphi_dm_closed(m::Real, x::Real)
    f = _gn_uncached(m)
    dlnx0dm = (3 * digamma(3 / m) - digamma(1 / m)) / (2 * m^2)
    if iszero(x)
        return f.C * (1 / m - dlnx0dm + digamma(1 / m) / m^2)
    end
    t = abs(x / f.x0)
    return f(x) * (1 / m - dlnx0dm + digamma(1 / m) / m^2 - t^m * (log(t) - m * dlnx0dm))
end

@testset "GN16: phi's m-partial, and the spectral density's" begin
    xs = collect(-4.0:0.05:4.0)
    ms = (1.6, 2.0, 3.0, 6.0, 12.0)
    worst_phi = 0.0
    for m in ms
        for x in xs
            ad = ForwardDiff.derivative(mm -> _gn_uncached(mm)(x), m)
            closed = _gn_dphi_dm_closed(m, x)
            worst_phi = max(worst_phi, abs(ad - closed))
            @test isapprox(ad, closed; rtol = 1e-11, atol = 1e-14)
        end
    end
    @info "GN16 worst phi m-partial error" worst_phi

    ms2 = (1.7, 2.2, 3.5, 6.0, 9.0)
    fd = Dict{Float64,Float64}()
    for m in ms2
        h = 1e-4 * m
        plasma_p = _gn_iaw_plasma(GeneralizedNormal(m + h))
        plasma_m = _gn_iaw_plasma(GeneralizedNormal(m - h))
        (k, om, _) = wavevector(_GN_PROBE, _GN_COLLECTION, _GN_LAMBDA_S, plasma_p)
        (km, omm, _) = wavevector(_GN_PROBE, _GN_COLLECTION, _GN_LAMBDA_S, plasma_m)
        sp = spectral_density(plasma_p, k, om)
        sm = spectral_density(plasma_m, km, omm)
        fd[m] = (sp - sm) / (2h)
    end
    floor_ = 1e-6 * maximum(abs, values(fd))
    worst_S = 0.0
    worst_ratio = 0.0
    for m in ms2
        fmd = GeneralizedNormal(ForwardDiff.Dual{:GN16}(m, 1.0))
        plasma = _gn_iaw_plasma(fmd)
        (k, om, _) = wavevector(_GN_PROBE, _GN_COLLECTION, _GN_LAMBDA_S, plasma)
        s = spectral_density(plasma, k, om)
        ad = ForwardDiff.partials(s)[1]
        worst_S = max(worst_S, abs(ad - fd[m]))
        worst_ratio = max(worst_ratio, abs(ad - fd[m]) / (1e-6 * abs(fd[m]) + floor_))
        @test isapprox(ad, fd[m]; rtol = 1e-6, atol = floor_)
    end
    @info "GN16 worst spectral-density m-derivative error, and ratio to its allowance" worst_S worst_ratio floor_
end

@testset "GN17: allocations" begin
    f = GeneralizedNormal(3.5)
    reconstruct(f, (m = 3.9,))
    @test (@allocated reconstruct(f, (m = 3.9,))) == 0

    md = ForwardDiff.Dual{:GN17}(3.5, 1.0)
    reconstruct(f, (m = md,))
    @test (@allocated reconstruct(f, (m = md,))) == 0

    _cheb_table_eval(GN_TABLE, f.slice, 1.3)
    @test (@allocated _cheb_table_eval(GN_TABLE, f.slice, 1.3)) == 0

    xd = ForwardDiff.Dual{:GN17}(1.3, 1.0)
    _cheb_table_eval(GN_TABLE, f.slice, xd)
    @test (@allocated _cheb_table_eval(GN_TABLE, f.slice, xd)) == 0

    fmd = GeneralizedNormal(ForwardDiff.Dual{:GN17}(3.5, 1.0))
    _cheb_table_eval(GN_TABLE, fmd.slice, 1.3)
    @test (@allocated _cheb_table_eval(GN_TABLE, fmd.slice, 1.3)) == 0

    f(0.7)
    @test (@allocated f(0.7)) == 0
    dfdx(f, 0.7)
    @test (@allocated dfdx(f, 0.7)) == 0

    xd2 = ForwardDiff.Dual{:GN17}(0.7, 1.0)
    f(xd2)
    @test (@allocated f(xd2)) == 0
    dfdx(f, xd2)
    @test (@allocated dfdx(f, xd2)) == 0

    fmd(0.7)
    @test (@allocated fmd(0.7)) == 0
    dfdx(fmd, 0.7)
    @test (@allocated dfdx(fmd, 0.7)) == 0

    # the guard: f(0.0) and dfdx(f, 0.0) at an m-dual shape
    fmd(0.0)
    @test (@allocated fmd(0.0)) == 0
    dfdx(fmd, 0.0)
    @test (@allocated dfdx(fmd, 0.0)) == 0
end

@testset "GN18: _widen_shape copies" begin
    f = GeneralizedNormal(3.5)
    D = ForwardDiff.Dual{:GN18,Float64,1}
    g = @inferred _widen_shape(f, D)
    @test g isa GeneralizedNormal
    @test g.m === D(f.m)
    @test g.x0 === D(f.x0)
    @test g.C === D(f.C)
    @test all(iszero, ForwardDiff.partials(g.m))
    @test all(iszero, ForwardDiff.partials(g.x0))
    @test all(iszero, ForwardDiff.partials(g.C))
    @test g.slice == D.(f.slice)
end

@testset "GN19: promote first" begin
    f32 = GeneralizedNormal(2.7f0)
    f64 = GeneralizedNormal(Float64(2.7f0))
    @test typeof(f32) === typeof(f64)
    @test f32.m === f64.m
    @test f32.x0 === f64.x0
    @test f32.C === f64.C
    @test f32.slice === f64.slice
end

@testset "GN20: parameters, reconstruct, show, and the constructor's throws" begin
    f = GeneralizedNormal(3.5)
    @test parameters(f) == (m = 3.5,)
    g = reconstruct(f, (m = 6.0,))
    @test g.slice === _cheb_slice(GN_TABLE, 6.0)
    @test sprint(show, f) == "GeneralizedNormal(m = 3.5)"
    @test_throws "m must be finite and at least 1.5" GeneralizedNormal(1.49)
    @test_throws "m must be finite and at least 1.5" GeneralizedNormal(1.0)
    @test_throws "m must be finite and at least 1.5" GeneralizedNormal(Inf)
    @test_throws "m must be finite and at least 1.5" GeneralizedNormal(NaN)
    @test GeneralizedNormal(1.5) isa GeneralizedNormal
end

# ------------------------------------------------------------------------------------------
# Registration (Build step 5): GN21, SC1-SC5 (in test_shape_cost.jl)
# ------------------------------------------------------------------------------------------

function _gn21_deck(datafile, outdir)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    m_e = { bounds = [1.8, 8.0], init = 3.0 }
    T_D = { bounds = [50.0, 800.0], init = 200.0 }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0
    shape = { type = "GeneralizedNormal", m = "m_e" }

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = 40.0

    [instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "levenberg_marquardt"

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

function _gn21_write_streak(path)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = GeneralizedNormal(4.0))],
        ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0, u_kms = 40.0)])
    trace = PlasmaTrace(template; t_ns = [0.0])
    expt = Experiment(trace = trace, probe = ProbeBeam(lambda_probe_nm = 263.25),
        collection = Collection(theta_deg = 60.0, k_model = :exact),
        lambda_m = collect(range(262.0, 264.5; length = 24)) .* 1e-9,
        effects = (Sensitivity(scale = 1.0e-6), ConstantBackground(level = 25.0)))
    image = measure(expt)
    sd = StreakData(lambda_m = expt.lambda, t_s = expt.t, image = image, sigma = ones(size(image)))
    write_streak(path, sd; source = "test")
    return sd
end

@testset "GN21: registration" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _gn21_write_streak(datafile)
        outdir = joinpath(dir, "out")

        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _gn21_deck(datafile, outdir))
        deck = read_deck(deckpath)
        @test deck.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "GeneralizedNormal"

        problem, optimizer, start = OTIS.build_problem(deck)
        b = only(b for b in problem.pspec.bindings if b.name == "m_e")
        @test b.kind === :leaf
        @test string(b.path) == "electrons.e.f.m"

        eff = effective_deck(deck, problem, optimizer)
        @test occursin("GeneralizedNormal", eff)
        efftmp = joinpath(dir, "effective.toml")
        write(efftmp, eff)
        deck2 = read_deck(efftmp)
        @test deck2.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "GeneralizedNormal"

        # an unknown shape key is refused
        bad = replace(_gn21_deck(datafile, outdir), "m = \"m_e\"" => "kappa = 3.0")
        badpath = joinpath(dir, "bad.toml")
        write(badpath, bad)
        @test_throws "unknown key \"kappa\"" read_deck(badpath)

        # a PlasmaTrace with a GeneralizedNormal electron round-trips
        template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = GeneralizedNormal(3.5))],
            ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0)])
        trace = PlasmaTrace(template; t_ns = [0.0, 1.0])
        tracepath = joinpath(dir, "trace.jld2")
        write_trace(tracepath, trace)
        trace2 = read_trace(tracepath; as = :trace)
        p1 = parameters(trace.template.electrons["e"].f)
        p2 = parameters(trace2.template.electrons["e"].f)
        @test p1.m === p2.m
    end
end

# ------------------------------------------------------------------------------------------
# The fits and the regression (Build step 6): GN22, GN23, GN24
# ------------------------------------------------------------------------------------------

# KP22's fixture, with the shape, variable and binding replaced (spec D34).
function _gn_p1_problem(m_truth::Real)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = GeneralizedNormal(4.0))],
        ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 200.0, u_kms = 0.0)])
    trace = PlasmaTrace(template; t_ns = [0.0])
    expt = Experiment(trace = trace, probe = ProbeBeam(lambda_probe_nm = 263.25),
        collection = Collection(theta_deg = 60.0, k_model = :exact),
        lambda_m = collect(range(262.0, 264.5; length = 24)) .* 1e-9,
        effects = (Sensitivity(scale = 1.0e-6), ConstantBackground(level = 25.0)))
    vm = FittedVariable("m_e"; bounds = (1.8, 8.0), init = 4.0)
    vT = FittedVariable("T_D"; bounds = (50.0, 800.0), init = 200.0)
    pspec = ParameterSpec(expt; variables = [vm, vT],
        bindings = ["m_e" => "electrons.e.f.m", "T_D" => "ions.D.T_eV"])
    working = Experiment(; trace = pspec.trace, probe = expt.probe, collection = expt.collection,
        lambda_m = expt.lambda, effects = expt.effects, prefactor = expt.prefactor)
    theta = flatten(pspec, ["m_e" => m_truth, "T_D" => 250.0])
    image = measure(unflatten(working, pspec, theta))
    data = StreakData(lambda_m = working.lambda, t_s = working.t, image = image,
        sigma = ones(size(image)))
    return FitProblem(working, pspec, data), pspec
end

@testset "GN22: the P1-sized recovery" begin
    for (m_truth, m_start) in ((3.0, 4.0), (5.0, 3.0), (2.4, 4.0))
        problem, pspec = _gn_p1_problem(m_truth)
        start = flatten(pspec, ["m_e" => m_start, "T_D" => 200.0])
        result = solve(problem, LevenbergMarquardt(); start = start)
        m_hat = result.values["m_e"][1]
        TD_hat = result.values["T_D"][1]
        errM = abs(m_hat - m_truth) / m_truth
        errT = abs(TD_hat - 250.0) / 250.0
        @info "GN22" m_truth m_start errM errT result.converged result.iterations
        @test errM < 1e-6
        @test errT < 1e-6
        @test result.converged
    end
end

@testset "GN23: the IAW feature's change with m, pinned" begin
    plasma_M = _gn_iaw_plasma(Maxwellian())
    (k, _, _) = wavevector(_GN_PROBE, _GN_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k * cs)

    sM = [spectral_density(plasma_M, k, om) for om in omegas]
    idx = argmax(sM)
    @info "GN23 argmax index and omega0/(k*cs)" idx (omegas[idx] / (k * cs))
    @test idx == 309

    omega0 = omegas[idx]
    s_M0 = spectral_density(plasma_M, k, omega0)
    ms = (2.5, 3.0, 4.0, 6.0, 8.0)
    expected = (0.7253490515815485, 0.5629633110179512, 0.4176815714180117,
                0.3279848011058526, 0.299045761691713)
    ratios = Float64[]
    for m in ms
        plasma_m = _gn_iaw_plasma(GeneralizedNormal(m))
        s_m = spectral_density(plasma_m, k, omega0)
        push!(ratios, s_m / s_M0)
    end
    @info "GN23 ratios as computed" ratios expected
    for (r, e) in zip(ratios, expected)
        @test isapprox(r, e; rtol = 1e-7)
    end
    @test all(ratios[i] > ratios[i+1] for i in 1:length(ratios)-1)
end

@testset "GN24: every fixture stays inside the table" begin
    # GN16's fixture
    worst16 = 0.0
    for m in (1.7, 2.2, 3.5, 6.0, 9.0)
        plasma = _gn_iaw_plasma(GeneralizedNormal(m))
        (k, om, _) = wavevector(_GN_PROBE, _GN_COLLECTION, _GN_LAMBDA_S, plasma)
        sp_e = plasma.electrons["e"]
        worst16 = max(worst16, abs(xi(sp_e, k, om)))
    end
    @test worst16 <= GN_TABLE.xi_max
    @info "GN24 GN16 fixture worst |xi_e|" worst16

    # GN23's fixture
    plasma_M = _gn_iaw_plasma(Maxwellian())
    (k23, _, _) = wavevector(_GN_PROBE, _GN_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k23 * cs)
    idx = argmax([spectral_density(plasma_M, k23, om) for om in omegas])
    omega0 = omegas[idx]
    worst23 = 0.0
    for m in (2.5, 3.0, 4.0, 6.0, 8.0)
        plasma_m = _gn_iaw_plasma(GeneralizedNormal(m))
        sp_e = plasma_m.electrons["e"]
        worst23 = max(worst23, abs(xi(sp_e, k23, omega0)))
    end
    @test worst23 <= GN_TABLE.xi_max
    @info "GN24 GN23 fixture worst |xi_e|" worst23

    # GN22's fixture, all three truths, all 24 wavelengths
    worst22 = 0.0
    for m_truth in (3.0, 5.0, 2.4)
        problem, pspec = _gn_p1_problem(m_truth)
        theta = flatten(pspec, ["m_e" => m_truth, "T_D" => 250.0])
        expt_truth = unflatten(problem.experiment, pspec, theta)
        plasma_truth = plasma(expt_truth.trace, 1)
        sp_e = plasma_truth.electrons["e"]
        for lam in expt_truth.lambda
            (k22, om22, _) = wavevector(expt_truth.probe, expt_truth.collection, lam, plasma_truth)
            worst22 = max(worst22, abs(xi(sp_e, k22, om22)))
        end
    end
    @test worst22 <= GN_TABLE.xi_max
    @info "GN24 GN22 fixture worst |xi_e|" worst22
end
