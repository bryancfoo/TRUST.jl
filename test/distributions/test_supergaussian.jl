using Test
using OTIS
using OTIS: _sg_cauchy_reference, _cauchy_series, _require_high_moments, _upper_gamma,
            _dgamma_da, _sg_x0, _sg_C, _sg_table_eval, SG_TABLE, _cauchy_upper, _basefloat
using ForwardDiff: derivative
import ForwardDiff
using SpecialFunctions: gamma, digamma
using QuadGK: quadgk
using Random

include(joinpath(@__DIR__, "..", "shape_contract.jl"))

@testset "SG1: normalizations" begin
    for m in 1.5:0.5:8.0
        check_shape_normalizations(SuperGaussian(m))
    end
end

@testset "SG2: phi and dfdx reduce to Maxwellian at m=2" begin
    f = SuperGaussian(2.0)
    fm = Maxwellian()
    for x in (0.0, 0.3, -0.3, 1.0, -1.0, 3.0, -3.0)
        @test isapprox(f(x), fm(x); rtol = 1e-15, atol = 1e-15)
        @test isapprox(dfdx(f, x), dfdx(fm, x); rtol = 1e-15, atol = 1e-15)
    end
end

@testset "SG3: moment vs quadrature" begin
    for m in (1.5, 2.0, 5.0)
        f = SuperGaussian(m)
        for n in 0:10
            ref = quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-12, atol = 1e-14)[1]
            @test isapprox(moment(f, n), ref; rtol = 1e-10, atol = 1e-10)
        end
    end
end

@testset "SG4: moment at m=2 equals Maxwellian's" begin
    f = SuperGaussian(2.0)
    fm = Maxwellian()
    for n in 0:4
        @test isapprox(moment(f, n), moment(fm, n); rtol = 1e-14, atol = 1e-14)
    end
end

@testset "SG5: excess_kurtosis and skewness" begin
    expected = Dict(1.5 => 0.430936001385, 2.0 => 0.0, 3.0 => -0.370202724829,
                     5.0 => -0.621648480580)
    for m in (1.5, 2.0, 3.0, 5.0)
        f = SuperGaussian(m)
        @test isapprox(excess_kurtosis(f), expected[m]; rtol = 1e-10, atol = 1e-10)
        @test isapprox(skewness(f), 0; atol = 1e-12)
    end
end

@testset "SG6: _sg_cauchy_reference vs Zprime at m=2" begin
    f = SuperGaussian(2.0)
    xis = (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, 12.0,
           0.5 + 0.5im, 1.0 + 0.2im, 3.0 + 1.0im, 12.0 + 0.5im, 2.0im)
    for xi in xis
        @test isapprox(_sg_cauchy_reference(f, xi), Zprime(xi); rtol = 1e-12, atol = 1e-12)
    end
end

@testset "SG7: Cauchy(0) closed form" begin
    printed = Dict(1.5 => -2.507969491287, 2.0 => -2.0, 3.0 => -1.612266101542,
                    4.0 => -1.458959743484, 5.0 => -1.380065550198, 8.0 => -1.282273636068)
    for m in (1.5, 2.0, 3.0, 4.0, 5.0, 8.0)
        f = SuperGaussian(m)
        closed = -gamma(1 / m) / (f.x0^2 * gamma(3 / m))
        val = real(_sg_cauchy_reference(f, 0.0))
        @test isapprox(val, closed; rtol = 1e-10, atol = 1e-10)
        @test isapprox(val, printed[m]; rtol = 1e-10, atol = 1e-10)
    end
end

@testset "SG8: series vs reference where it converges" begin
    for m in (1.5, 2.0, 3.0, 5.0)
        f = SuperGaussian(m)
        for xi in (9.0, 10.0)
            val, converged = _cauchy_series(f, xi)
            @test converged
            ref = real(_sg_cauchy_reference(f, xi))
            @test isapprox(val, ref; rtol = 1e-9, atol = 1e-9)
        end
    end
end

@testset "SG9: series stopping rule" begin
    for m in (1.5, 2.0, 3.0, 5.0)
        f = SuperGaussian(m)
        _, conv0 = _cauchy_series(f, 0.0)
        @test conv0 == false
        _, conv3 = _cauchy_series(f, 3.0)
        @test conv3 == false
    end
    for m in (1.5, 2.0, 3.0, 5.0, 8.0)
        f = SuperGaussian(m)
        _, conv10 = _cauchy_series(f, 10.0)
        @test conv10 == true
    end
end

@testset "SG10: _require_high_moments on Maxwellian" begin
    @test_throws "Laurent series needs moments up to n = 39" _cauchy_series(Maxwellian(), 10.0)
end

# --- SG11: BigFloat oracle, sharing no machinery with src/ (conventions.md 8.2) ---

_x0_big(m) = sqrt(3 * gamma(3 / big(m)) / (2 * gamma(5 / big(m))))
_C_big(m) = 1 / (2 * _x0_big(m) * gamma(3 / big(m)))
_dphi_big(x, m) = -(big(m) * _C_big(m) / _x0_big(m)^2) * x * exp(-abs(x / _x0_big(m))^big(m))
# phi''(x) in closed form (differentiate _dphi_big once more by hand), rather than a finite
# difference: independent of _cauchy_real's own removable-singularity handling either way, and
# avoids the extreme BigFloat precision an h = 1e-25 finite difference would otherwise need.
function _d2phi_big(x, m)
    x0 = _x0_big(m); C = _C_big(m); mb = big(m)
    t = abs(x / x0)^mb
    return -(mb * C / x0^2) * exp(-t) * (1 - mb * t)
end

function _cauchy_ref_big(xi_re::Real, m)
    xib = big(xi_re)
    g(t) = iszero(t) ? 2 * _d2phi_big(xib, m) : (_dphi_big(xib + t, m) - _dphi_big(xib - t, m)) / t
    pv, _ = quadgk(g, big(0), big(6), big(40), big(Inf); rtol = big(1e-14), atol = big(1e-60))
    return pv + im * big(pi) * _dphi_big(xib, m)
end

function _cauchy_ref_big(xi::Complex, m)
    xib = big(real(xi)) + im * big(imag(xi))
    val, _ = quadgk(x -> _dphi_big(x, m) / (x - xib), big(-Inf), big(0), big(Inf);
                     rtol = big(1e-14), atol = big(1e-60))
    return val
end

@testset "SG11: reference vs BigFloat oracle" begin
    setprecision(BigFloat, 128) do
        for m in (1.5, 3.0, 5.0), xi in (0.0, 0.5, 1.0, 2.0, 3.0)
            f = SuperGaussian(m)
            oracle = _cauchy_ref_big(xi, m)
            val = _sg_cauchy_reference(f, xi)
            @test isapprox(real(val), Float64(real(oracle)); rtol = 1e-9, atol = 1e-9)
            @test isapprox(imag(val), Float64(imag(oracle)); rtol = 1e-9, atol = 1e-9)
        end
        for m in (1.5, 3.0, 5.0), xi in (0.5 + 0.5im, 2.0 + 1.0im)
            f = SuperGaussian(m)
            oracle = _cauchy_ref_big(xi, m)
            val = _sg_cauchy_reference(f, xi)
            @test isapprox(real(val), Float64(real(oracle)); rtol = 1e-9, atol = 1e-9)
            @test isapprox(imag(val), Float64(imag(oracle)); rtol = 1e-9, atol = 1e-9)
        end
    end
end

@testset "SG12: reference vs series at |xi| = 10" begin
    for m in (1.5, 2.0, 3.0, 5.0, 8.0)
        f = SuperGaussian(m)
        for xi in (10.0, -10.0)
            ref = real(_sg_cauchy_reference(f, xi))
            series_val, converged = _cauchy_series(f, xi)
            @test converged
            @test isapprox(ref, series_val; rtol = 1e-9, atol = 1e-9)
        end
    end
end

@testset "SG13: continuity through the series switch" begin
    # Both branches cauchy_dfdv can take (series, reference) must individually agree with the
    # reference to the target across the switch region, so that neither introduces a jump.
    switch = Dict(1.5 => 7.85, 2.0 => 6.08, 3.0 => 4.72, 5.0 => 3.86)
    for m in (1.5, 2.0, 3.0, 5.0)
        f = SuperGaussian(m)
        xi0 = switch[m]
        for dx in -0.2:0.005:0.2
            xi = xi0 + dx
            cur = real(cauchy_dfdv(f, xi))
            ref = real(_sg_cauchy_reference(f, xi))
            @test isapprox(cur, ref; rtol = 1e-9, atol = 1e-9)
        end
    end
end

@testset "SG14: dphi/dx vs dfdx" begin
    for m in (1.5, 2.0, 3.5, 5.0), x in (0.0, 0.1, 0.5, 1.0, 2.0)
        f = SuperGaussian(m)
        d = derivative(y -> f(y), x)
        @test isapprox(d, dfdx(f, x); rtol = 1e-14, atol = 1e-14)
    end
end

@testset "SG15: dphi/dx, dphi/dm vs central differences" begin
    ms = (1.5, 2.0, 3.5, 5.0)
    xs = (0.0, 0.1, 0.5, 1.0, 2.0)
    h = 1e-5
    fd_x = Dict{Tuple{Float64,Float64},Float64}()
    fd_m = Dict{Tuple{Float64,Float64},Float64}()
    for m in ms, x in xs
        fd_x[(m, x)] = (SuperGaussian(m)(x + h) - SuperGaussian(m)(x - h)) / (2h)
        fd_m[(m, x)] = (SuperGaussian(m + h)(x) - SuperGaussian(m - h)(x)) / (2h)
    end
    floor_x = 1e-8 * maximum(abs, values(fd_x))
    floor_m = 1e-8 * maximum(abs, values(fd_m))
    for m in ms, x in xs
        f = SuperGaussian(m)
        adx = derivative(y -> f(y), x)
        adm = derivative(mm -> SuperGaussian(mm)(x), m)
        @test isapprox(adx, fd_x[(m, x)]; rtol = 1e-8, atol = floor_x)
        @test isapprox(adm, fd_m[(m, x)]; rtol = 1e-8, atol = floor_m)
    end
end

@testset "SG16: joint (x, m) gradient" begin
    h = 1e-5
    for m in (2.0, 3.5), x in (0.0, 0.7, 2.0)
        g = ForwardDiff.gradient(p -> SuperGaussian(p[2])(p[1]), [x, m])
        fdx = (SuperGaussian(m)(x + h) - SuperGaussian(m)(x - h)) / (2h)
        fdm = (SuperGaussian(m + h)(x) - SuperGaussian(m - h)(x)) / (2h)
        @test isapprox(g[1], fdx; rtol = 1e-8, atol = 1e-8 * max(abs(fdx), 1e-12))
        @test isapprox(g[2], fdm; rtol = 1e-8, atol = 1e-8 * max(abs(fdm), 1e-12))
    end
end

@testset "SG17: _upper_gamma dispatch" begin
    a0, t0 = 0.7, 1.3
    v_plain = _upper_gamma(a0, t0)
    @test isfinite(v_plain)
    @test isapprox(v_plain, gamma(a0, t0); rtol = 1e-14)

    da = ForwardDiff.Dual{:TagA}(a0, 1.0)
    dt = ForwardDiff.Dual{:TagA}(t0, 1.0)
    v_both = _upper_gamma(da, dt)
    @test isfinite(ForwardDiff.value(v_both))
    @test isapprox(ForwardDiff.value(v_both), gamma(a0, t0); rtol = 1e-14)

    v_da = _upper_gamma(da, t0)
    @test isfinite(ForwardDiff.value(v_da))
    @test isapprox(ForwardDiff.value(v_da), gamma(a0, t0); rtol = 1e-14)

    v_dt = _upper_gamma(a0, dt)
    @test isfinite(ForwardDiff.value(v_dt))
    @test isapprox(ForwardDiff.value(v_dt), gamma(a0, t0); rtol = 1e-14)

    nested = ForwardDiff.derivative(
        aa -> ForwardDiff.derivative(tt -> _upper_gamma(aa, tt), t0), a0)
    @test isfinite(nested)
end

@testset "SG18: _dgamma_da" begin
    for a in (0.4, 0.9, 1.1)
        @test isapprox(_dgamma_da(a, 0.0), gamma(a) * digamma(a); rtol = 1e-12, atol = 1e-14)
    end
    pts = ((0.4, 1e-6), (0.5714, 0.01), (0.9, 0.3), (1.0, 1.0), (1.1, 3.0),
           (0.4, 3.0), (1.1, 1e-6), (0.9, 1.0))
    setprecision(BigFloat, 128) do
        for (a, t) in pts
            ref, _ = quadgk(s -> s^(big(a) - 1) * log(s) * exp(-s), big(t), big(t) + 1, big(Inf);
                             rtol = big(1e-16), atol = big(1e-60))
            @test isapprox(Float64(ref), _dgamma_da(a, t); rtol = 1e-12, atol = 1e-14)
        end
    end
end

_richardson_dm(g, m, h) = (4 * ((g(m + h) - g(m - h)) / (2h)) -
                            (g(m + 2h) - g(m - 2h)) / (4h)) / 3

@testset "SG19: dR/dm through the D8 breakpoint" begin
    h = 1e-4
    for m in (2.0, 3.5)
        oracles = Float64[]
        for xi in (0.5, 1.0, 2.0)
            g(mm) = real(_sg_cauchy_reference(SuperGaussian(mm), xi))
            push!(oracles, _richardson_dm(g, m, h))
        end
        floor_ = 1e-6 * maximum(abs, oracles)
        for (i, xi) in enumerate((0.5, 1.0, 2.0))
            ad = ForwardDiff.derivative(mm -> real(_sg_cauchy_reference(SuperGaussian(mm), xi)), m)
            @test isapprox(ad, oracles[i]; rtol = 1e-6, atol = floor_)
        end
    end
end

@testset "SG20: errors" begin
    @test_throws "SuperGaussian: m must be positive" SuperGaussian(0.0)
    @test_throws "SuperGaussian: m must be positive" SuperGaussian(-1.0)
    f = SuperGaussian(3.0)
    @test_throws "no single analytic continuation off the real axis" f(0.5 + 0.1im)
    @test_throws "no single analytic continuation off the real axis" dfdx(f, 0.5 + 0.1im)
    @test_throws "requires Im xi >= 0" cauchy_dfdv(f, 0.5 - 0.5im)
    @test_throws "n must be non-negative" moment(f, -1)
end

@testset "SG21: parameters, reconstruct, show" begin
    f = SuperGaussian(3.5)
    @test parameters(f) === (m = 3.5,)
    g = reconstruct(f, (m = 4.0,))
    @test g.m == 4.0
    io = IOBuffer()
    show(io, f)
    s = String(take!(io))
    @test s == "SuperGaussian(m = 3.5)"
    @test !occursin("x0", s)
    @test !occursin("slice", s)
end

# --- Round 4 (2026-09-08): the three dual-path defects Level 1's review found ---

@testset "R4.1: SuperGaussian(::Float32) promotes to Float64 (round 4, D3)" begin
    # As of Step 2.0 (Phase 2 spec, item 1) the constructor promotes m to FT BEFORE computing
    # x0(m) and C(m), so 3.0f0 -- which round-trips exactly through Float32 -- now gives a shape
    # that agrees with SuperGaussian(3.0) to round-off, not merely to Float32 precision; HK1
    # below is the bitwise version of this test. The rtol = 1e-6 comparisons here still hold
    # (bitwise agreement implies them) and are kept as a looser, independent check.
    f32 = SuperGaussian(3.0f0)
    f64 = SuperGaussian(3.0)
    @test typeof(f32).parameters[1] === Float64
    @test f32.m === f64.m
    for x in (0.0, 0.5, -0.5, 2.5, -2.5)
        @test isapprox(f32(x), f64(x); rtol = 1e-6)
        @test isapprox(dfdx(f32, x), dfdx(f64, x); rtol = 1e-6)
    end
    for n in 0:4
        @test isapprox(moment(f32, n), moment(f64, n); rtol = 1e-6, atol = 1e-6)
    end
end

@testset "R4.2: phi at a dual x with a plain m allocates nothing (round 4, D5)" begin
    f = SuperGaussian(3.0)
    xd = ForwardDiff.Dual{:R4}(1.0, 1.0)
    _call_dual() = f(xd)
    _call_dual()   # warm up
    @test (@allocated _call_dual()) == 0
    v = _call_dual()
    @test isfinite(ForwardDiff.value(v))
    @test isapprox(ForwardDiff.value(v), f(1.0); rtol = 1e-14)
    @test isapprox(ForwardDiff.partials(v)[1], dfdx(f, 1.0); rtol = 1e-12)
end

@testset "R4.3: phi at x=0 keeps a concrete return type under a dual (round 4, D5)" begin
    f = SuperGaussian(3.0)
    x0d = ForwardDiff.Dual{:R4}(0.0, 1.0)
    v = @inferred f(x0d)
    @test ForwardDiff.value(v) == f.C * gamma(2 / f.m)
    @test ForwardDiff.partials(v)[1] == 0.0
end

# --- Step 1.SG.3: cauchy_dfdv served from the table (spec D14, Section 5) ---

_dp_rng = Random.Xoshiro(20260908)
_dp_xis = [(2 * rand(_dp_rng) - 1) * SG_TABLE.xi_max for _ in 1:200]
_dp_ms  = [SG_TABLE.m_lo + rand(_dp_rng) * (SG_TABLE.m_hi - SG_TABLE.m_lo) for _ in 1:200]

@testset "DP1: cauchy_dfdv against _sg_cauchy_reference (seeded)" begin
    worst_re = 0.0
    worst_im = 0.0
    for (xi_pt, m) in zip(_dp_xis, _dp_ms)
        f = SuperGaussian(m)
        cur = cauchy_dfdv(f, xi_pt)
        ref = _sg_cauchy_reference(f, xi_pt)
        worst_re = max(worst_re, abs(real(cur) - real(ref)) / max(abs(real(ref)), 1e-8))
        worst_im = max(worst_im, abs(imag(cur) - imag(ref)) / max(abs(imag(ref)), 1e-12))
        @test isapprox(real(cur), real(ref); rtol = 1e-8, atol = 1e-8)
        @test isapprox(imag(cur), imag(ref); rtol = 1e-12, atol = 1e-12)
    end
    @info "DP1 worst normalized error" worst_re worst_im
end

@testset "DP2: cauchy_dfdv against Zprime at m=2" begin
    f = SuperGaussian(2.0)
    worst_re = 0.0
    worst_im = 0.0
    for xi_pt in _dp_xis
        cur = cauchy_dfdv(f, xi_pt)
        ref = Zprime(xi_pt)
        worst_re = max(worst_re, abs(real(cur) - real(ref)) / max(abs(real(ref)), 1e-8))
        worst_im = max(worst_im, abs(imag(cur) - imag(ref)) / max(abs(imag(ref)), 1e-8))
        @test isapprox(real(cur), real(ref); rtol = 1e-8, atol = 1e-8)
        @test isapprox(imag(cur), imag(ref); rtol = 1e-8, atol = 1e-8)
    end
    @info "DP2 worst normalized error" worst_re worst_im
end

@testset "DP3: the branches are the ones D14 names" begin
    xmax = SG_TABLE.xi_max
    mlo = SG_TABLE.m_lo
    mhi = SG_TABLE.m_hi

    # (a) inside the table's domain, at both sides of the m and xi edges: the table branch, exact
    for m in (mlo, 2.0, mhi), xi_pt in (0.0, 1.0, xmax)
        f = SuperGaussian(m)
        QT = _basefloat(xi_pt)
        expected = _sg_table_eval(SG_TABLE, f.slice, abs(xi_pt)) + im * QT(pi) * dfdx(f, xi_pt)
        @test cauchy_dfdv(f, xi_pt) == expected
    end

    # (b) just outside xi_max: the series branch, exact
    let f = SuperGaussian(3.0), xi_pt = xmax + 1e-6
        series_val, converged = _cauchy_series(f, xi_pt)
        @test converged
        QT = _basefloat(xi_pt)
        expected = series_val + im * QT(pi) * dfdx(f, xi_pt)
        @test cauchy_dfdv(f, xi_pt) == expected
    end

    # (c) m outside the table's domain, both sides: the reference, exact (xi = 2 never converges
    # the series -- D6's switch curve starts at |xi| >= 3.47)
    for m in (1.4999, 8.0001)
        f = SuperGaussian(m)
        xi_pt = 2.0
        @test cauchy_dfdv(f, xi_pt) == _sg_cauchy_reference(f, xi_pt)
    end

    # (d) m at the table's edges exactly: the table branch
    for m in (mlo, mhi)
        f = SuperGaussian(m)
        xi_pt = 2.0
        QT = _basefloat(xi_pt)
        expected = _sg_table_eval(SG_TABLE, f.slice, abs(xi_pt)) + im * QT(pi) * dfdx(f, xi_pt)
        @test cauchy_dfdv(f, xi_pt) == expected
    end
end

@testset "DP4: the upper half-plane and below it" begin
    f = SuperGaussian(3.0)
    for xi_pt in (0.5 + 0.5im, 2.0 + 1.0im, 5.0 + 0.3im)
        @test cauchy_dfdv(f, xi_pt) == _cauchy_upper(f, xi_pt)
    end
    for (m, xi_pt) in ((1.5, 0.5 + 0.5im), (3.0, 0.5 + 0.5im), (1.5, 2.0 + 1.0im), (3.0, 2.0 + 1.0im))
        fm = SuperGaussian(m)
        oracle = _cauchy_ref_big(xi_pt, m)
        val = cauchy_dfdv(fm, xi_pt)
        @test isapprox(real(val), Float64(real(oracle)); rtol = 1e-9, atol = 1e-9)
        @test isapprox(imag(val), Float64(imag(oracle)); rtol = 1e-9, atol = 1e-9)
    end
    @test_throws "requires Im xi >= 0" cauchy_dfdv(f, 0.5 - 0.5im)
end

@testset "DP5: AD through the dispatch" begin
    ms = (1.75, 2.0, 3.5, 6.0)
    xis = (0.1, 0.5, 0.92, 2.0, 5.0, 9.0)
    h = 1e-4
    fd_xi = Dict{Tuple{Float64,Float64},Float64}()
    fd_m = Dict{Tuple{Float64,Float64},Float64}()
    for m in ms, xi_pt in xis
        fd_xi[(m, xi_pt)] = (real(_sg_cauchy_reference(SuperGaussian(m), xi_pt + h)) -
                              real(_sg_cauchy_reference(SuperGaussian(m), xi_pt - h))) / (2h)
        fd_m[(m, xi_pt)] = (real(_sg_cauchy_reference(SuperGaussian(m + h), xi_pt)) -
                             real(_sg_cauchy_reference(SuperGaussian(m - h), xi_pt))) / (2h)
    end
    floor_xi = 1e-6 * maximum(abs, values(fd_xi))
    floor_m = 1e-6 * maximum(abs, values(fd_m))
    worst_xi = 0.0
    worst_m = 0.0
    for m in ms, xi_pt in xis
        f = SuperGaussian(m)
        ad_xi = ForwardDiff.derivative(x -> real(cauchy_dfdv(f, x)), xi_pt)
        ad_m = ForwardDiff.derivative(mm -> real(cauchy_dfdv(SuperGaussian(mm), xi_pt)), m)
        worst_xi = max(worst_xi, abs(ad_xi - fd_xi[(m, xi_pt)]))
        worst_m = max(worst_m, abs(ad_m - fd_m[(m, xi_pt)]))
        @test isapprox(ad_xi, fd_xi[(m, xi_pt)]; rtol = 1e-6, atol = floor_xi)
        @test isapprox(ad_m, fd_m[(m, xi_pt)]; rtol = 1e-6, atol = floor_m)
    end
    @info "DP5 worst absolute error against its floor" worst_xi floor_xi worst_m floor_m
end

@testset "DP6: allocations through the dispatch, and susceptibility" begin
    f = SuperGaussian(3.0)
    xi_plain = 2.0
    xi_dual = ForwardDiff.Dual{:DP6}(2.0, 1.0)
    f_dual_m = SuperGaussian(ForwardDiff.Dual{:DP6}(3.0, 1.0))

    _cd_plain() = cauchy_dfdv(f, xi_plain)
    _cd_dualxi() = cauchy_dfdv(f, xi_dual)
    _cd_dualm() = cauchy_dfdv(f_dual_m, xi_plain)
    _cd_plain(); _cd_dualxi(); _cd_dualm()
    @test (@allocated _cd_plain()) == 0
    @test (@allocated _cd_dualxi()) == 0
    @test (@allocated _cd_dualm()) == 0

    plasma = Plasma(n_e_cm3 = 5e19,
                     electrons = (electrons(T_eV = 500, f = SuperGaussian(3.0)),),
                     ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
    spe = plasma.electrons["e"]
    k = 1.0e7
    vth = thermal_speed(spe)
    omega = k * 2.0 * vth
    xi_s = xi(spe, k, omega)
    expected = -(alpha(spe, k)^2 / 2) * cauchy_dfdv(spe.f, xi_s)
    val = susceptibility(spe, k, omega)
    @test isapprox(val, expected; rtol = 1e-14)

    _susc() = susceptibility(spe, k, omega)
    _susc()
    @test (@allocated _susc()) == 0
end

@testset "DP7: continuity across the xi_max join" begin
    xmax = SG_TABLE.xi_max
    worst = 0.0
    for m in (1.5, 2.0, 5.0)
        f = SuperGaussian(m)
        for dxi in -0.05:0.005:0.05
            xi_pt = xmax + dxi
            cur = real(cauchy_dfdv(f, xi_pt))
            ref = real(_sg_cauchy_reference(f, xi_pt))
            worst = max(worst, abs(cur - ref) / max(abs(ref), 1e-8))
            @test isapprox(cur, ref; rtol = 1e-8, atol = 1e-8)
        end
    end
    @info "DP7 worst normalized error at the xi_max join" worst
end

@testset "DP8: evenness, where the abs actually lives" begin
    rng8 = Random.Xoshiro(20260908)
    xis8 = [rand(rng8) * SG_TABLE.xi_max for _ in 1:8]
    for m in (1.5, 2.0, 5.0, 8.0)
        f = SuperGaussian(m)
        for xi_pt in xis8
            plus = cauchy_dfdv(f, xi_pt)
            minus = cauchy_dfdv(f, -xi_pt)
            @test isapprox(real(minus), real(plus); rtol = 1e-14)
            @test isapprox(imag(minus), -imag(plus); rtol = 1e-14)
        end
    end
    for m in (1.5, 2.0, 5.0, 8.0)
        f = SuperGaussian(m)
        for a in (0.05, 0.3, 1.7, 6.0)
            dplus = ForwardDiff.derivative(x -> real(cauchy_dfdv(f, x)), a)
            dminus = ForwardDiff.derivative(x -> real(cauchy_dfdv(f, x)), -a)
            @test isapprox(dminus, -dplus; rtol = 1e-14)
        end
    end
end

# --- Step 2.0 (2026-09-09): the constructor promotes m before computing x0(m) and C(m) ---

@testset "HK1: SuperGaussian(3.0f0) equals SuperGaussian(3.0) bitwise" begin
    # 3.0f0 round-trips exactly through Float32, so m_p = FT(3.0f0) === 3.0 exactly; since the
    # constructor now promotes BEFORE computing x0 and C (Phase 2 spec Step 2.0 item 1), every
    # cached field is computed in Float64 arithmetic from that same value, not widened from a
    # Float32-precision result. This is the whole point of the reorder: a relative tolerance
    # would hide the old (wrong) behaviour, so the comparison here is bitwise.
    f32 = SuperGaussian(3.0f0)
    f64 = SuperGaussian(3.0)
    @test typeof(f32) === typeof(f64)
    @test f32.m === f64.m
    @test f32.x0 === f64.x0
    @test f32.C === f64.C
    @test f32.slice === f64.slice
end

@testset "HK2: the Float32 path agrees bitwise in phi, dfdx and moment" begin
    # 2.5f0, 3.0f0 and 0.5f0 are exactly representable in Float32 and round-trip exactly to
    # Float64 (Phase 1's SP12 pattern, reused here for the same reason).
    for m32 in (2.5f0, 3.0f0, 0.5f0)
        f32 = SuperGaussian(m32)
        f64 = SuperGaussian(Float64(m32))
        @test typeof(f32) === typeof(f64)
        for x in (0.0, 0.3, -0.3, 1.0, 2.5)
            v32, v64 = f32(x), f64(x)
            @test v32 === v64
            @test typeof(v32) === Float64
            d32, d64 = dfdx(f32, x), dfdx(f64, x)
            @test d32 === d64
            @test typeof(d32) === Float64
        end
        for n in 0:4
            mo32, mo64 = moment(f32, n), moment(f64, n)
            @test mo32 === mo64
            @test typeof(mo32) === Float64
        end
    end
end
