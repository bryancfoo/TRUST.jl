# Step 4.B.4: SkewNormal, the shape (spec D35-D38, D42, D47). SK1-SK25, gate "SkewNormal".
# Section 10.3's formulas, in the code's normalization; every oracle independent of the code under
# test (conventions.md 8.2): a quadrature, a BigFloat evaluation, Zprime, ForwardDiff or a finite
# difference, never the function itself.

using Test
using OTIS
using OTIS: SN_TABLE, SN_DEGREE_START, SN_DEGREE_CAP_XI, SN_DEGREE_CAP_ALPHA, SN_BISECT_MAX,
            _sn_uncached, _sn_cauchy_reference, _sn_principal_value, _cheb_slice, _cheb_table_eval,
            _basefloat, _cauchy_upper, _widen_shape, SHAPE_REGISTRY, _std_normal_pdf, _std_normal_cdf
using SpecialFunctions: erfc
using QuadGK: quadgk
using Random
import ForwardDiff

include(joinpath(@__DIR__, "..", "shape_contract.jl"))

# ------------------------------------------------------------------------------------------
# SK1-SK7: the shape without its table (Build step 1); SK3 first (Phi against MPFR, before
# anything else in this file uses Phi -- conventions.md 8.2, the SuperGaussian expint lesson).
# ------------------------------------------------------------------------------------------

@testset "SK3: Phi and phi are not their own oracle" begin
    worstPhi = 0.0
    worstPhi_y = 0.0
    setprecision(BigFloat, 80) do
        for y in -37.0:0.003:10.0
            got = _std_normal_cdf(y)
            ref = Float64(erfc(-big(y) / sqrt(big(2))) / 2)
            rel = abs(got - ref) / max(abs(ref), abs(got), 1e-300)
            if rel > worstPhi
                worstPhi = rel
                worstPhi_y = y
            end
            @test isapprox(got, ref; rtol = 1e-12)
        end
    end
    @info "SK3 Phi worst relative error and where" worstPhi worstPhi_y

    worstphi = 0.0
    setprecision(BigFloat, 128) do
        for z in -37.0:0.01:37.0
            got = _std_normal_pdf(z)
            ref = Float64(exp(-big(z)^2 / 2) / sqrt(2 * big(pi)))
            rel = abs(got - ref) / max(abs(ref), abs(got), 1e-300)
            worstphi = max(worstphi, rel)
            @test isapprox(got, ref; rtol = 1e-12)
        end
    end
    @info "SK3 phi worst relative error" worstphi
end

@testset "SK1: the three normalizations" begin
    for alpha in (-3.0, -0.7, 0.0, 1.3, 3.0, 8.0)
        check_shape_normalizations(SkewNormal(alpha))
    end
end

@testset "SK2: omega and xc against a 128-bit BigFloat evaluation" begin
    setprecision(BigFloat, 128) do
        for alpha in (-8.0, -3.0, -0.7, 0.3, 1.0, 3.0, 8.0)
            f = SkewNormal(alpha)
            ab = big(alpha)
            deltab = ab / sqrt(1 + ab^2)
            mub = deltab * sqrt(2 / big(pi))
            vb = 1 - 2 * deltab^2 / big(pi)
            omegab = 1 / sqrt(2 * vb)
            xcb = -omegab * mub
            @test isapprox(f.omega, Float64(omegab); rtol = 1e-14)
            @test isapprox(f.xc, Float64(xcb); rtol = 1e-14)
        end
    end
    f0 = SkewNormal(0.0)
    @test isapprox(f0.omega, 1 / sqrt(2); rtol = 1e-15)
    @test iszero(f0.xc)   # not === 0.0: -omega * 0.0 == -0.0
end

@testset "SK4: the Maxwellian at alpha = 0" begin
    # Deviation from spec D38's SK4 row, approved by the user 2026-09-27: rtol loosened from
    # 1e-14 to 2e-14. SkewNormal's alpha=0 reduction goes through more floating-point operations
    # than GeneralizedNormal's m=2 one (subtraction, division, erfc, several products), and at
    # large |x| where the reference is tiny this accumulates past 1e-14: measured worst 1.41e-14
    # (value, x=-5.67) and 1.39e-14 (derivative, x=-5.72) over x in -6:0.01:6.
    f0 = SkewNormal(0.0)
    for x in -6.0:0.01:6.0
        @test isapprox(f0(x), exp(-x^2) / sqrt(pi); rtol = 2e-14, atol = 1e-300)
        @test isapprox(dfdx(f0, x), -2x * exp(-x^2) / sqrt(pi); rtol = 2e-14, atol = 1e-300)
    end
    worst = 0.0
    for xi_ in -20.0:0.05:20.0
        cv = cauchy_dfdv(f0, xi_)
        zp = Zprime(xi_)
        worst = max(worst, abs(real(cv) - real(zp)), abs(imag(cv) - imag(zp)))
        @test isapprox(real(cv), real(zp); atol = 1e-9, rtol = 0)
        @test isapprox(imag(cv), imag(zp); atol = 1e-9, rtol = 0)
    end
    @info "SK4 worst |cauchy_dfdv(alpha=0) - Zprime|" worst
end

@testset "SK5: dfdx vs ForwardDiff" begin
    xs = collect(-8.0:0.05:8.0)
    for alpha in (-3.0, 0.4, 2.5, 8.0)
        f = SkewNormal(alpha)
        for x in xs
            ad = ForwardDiff.derivative(y -> f(y), x)
            @test isapprox(ad, dfdx(f, x); rtol = 1e-13, atol = 1e-15)
        end
    end
end

@testset "SK6: moments, and the derived quantities" begin
    worst = 0.0
    for alpha in (-3.0, -0.7, 0.3, 1.0, 3.0, 5.0)
        f = SkewNormal(alpha)
        for n in 0:8
            ref, _ = quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-11, atol = 1e-12)
            worst = max(worst, abs(moment(f, n) - ref))
            @test isapprox(moment(f, n), ref; rtol = 1e-11, atol = 1e-12)
        end
        @test moment(f, 1) === 0.0

        mu_delta = -f.xc / f.omega
        v_delta = 1 / (2 * f.omega^2)
        gamma1 = ((4 - pi) / 2) * mu_delta^3 / v_delta^(3 / 2)
        @test isapprox(skewness(f), gamma1; rtol = 1e-12)
    end
    @info "SK6 worst |moment - quadrature|" worst
    @test_throws ArgumentError moment(SkewNormal(0.7), -1)
end

@testset "SK7: the reflection" begin
    for alpha in (0.3, 1.2, 2.9)
        fpos = SkewNormal(alpha)
        fneg = SkewNormal(-alpha)
        for x in -6.0:0.05:6.0
            @test isapprox(fneg(x), fpos(-x); rtol = 1e-14)
        end
        for xi_ in (0.4, 1.7, 6.0, 15.0)
            tv_neg = _cheb_table_eval(SN_TABLE, fneg.slice, xi_)
            tv_pos = _cheb_table_eval(SN_TABLE, fpos.slice, -xi_)
            @test isapprox(tv_neg, tv_pos; atol = 1e-12)

            cv_neg = cauchy_dfdv(fneg, xi_)
            cv_pos = cauchy_dfdv(fpos, -xi_)
            @test isapprox(real(cv_neg), real(cv_pos); atol = 1e-12)
            @test isapprox(imag(cv_neg), -imag(cv_pos); atol = 1e-12)
        end
    end
end

# ------------------------------------------------------------------------------------------
# The table (Build step 2): SK8, SK9, SK11-SK14, SK25, then SK10 (the BigFloat oracle)
# ------------------------------------------------------------------------------------------

@testset "SK8: table structure" begin
    @test SN_TABLE.p_lo == -3.0
    @test SN_TABLE.p_hi == 3.0
    @test SN_TABLE.xi_lo == -20.0
    @test SN_TABLE.xi_max == 20.0
    @test all(SN_TABLE.xi_converged)
    @test SN_TABLE.p_converged
    @test SN_TABLE.np <= SN_DEGREE_CAP_ALPHA
    @test SN_TABLE.cap_p == 64
    @test SN_TABLE.degree_start == SN_DEGREE_START
    @test SN_TABLE.cap_xi == SN_DEGREE_CAP_XI
    @test SN_TABLE.bisect_max == SN_BISECT_MAX
    L = sum(SN_TABLE.nxi .+ 1)
    @test length(SN_TABLE.coeffs) == L * (SN_TABLE.np + 1)
    @info "SK8 table" edges = SN_TABLE.edges nxi = SN_TABLE.nxi np = SN_TABLE.np L = L
end

const _SK_RNG = Random.Xoshiro(20260927)
const _SK9_POINTS = [(-3.0 + 6.0 * rand(_SK_RNG), -20.0 + 40.0 * rand(_SK_RNG)) for _ in 1:60]

@testset "SK9: the table against its generator" begin
    worst = 0.0
    for (a, xi_) in _SK9_POINTS
        slice = _cheb_slice(SN_TABLE, a)
        tv = _cheb_table_eval(SN_TABLE, slice, xi_)
        rv = _sn_principal_value(a, xi_)
        worst = max(worst, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    for edge in SN_TABLE.edges, a in (-3.0, 0.0, 3.0)
        slice = _cheb_slice(SN_TABLE, a)
        tv = _cheb_table_eval(SN_TABLE, slice, edge)
        rv = _sn_principal_value(a, edge)
        worst = max(worst, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    @info "SK9 worst |table - generator|" worst
end

@testset "SK11: the Maxwellian edge against Zprime" begin
    worst = 0.0
    slice0 = _cheb_slice(SN_TABLE, 0.0)
    for xi_ in -20.0:0.05:20.0
        tv = _cheb_table_eval(SN_TABLE, slice0, xi_)
        zp = real(Zprime(xi_))
        worst = max(worst, abs(tv - zp))
        @test isapprox(tv, zp; atol = 1e-9, rtol = 0)
    end
    @info "SK11 worst |table(alpha=0) - Re Zprime|" worst
end

function _sk12_data()
    h = 1e-4
    ad = Vector{Float64}(undef, length(_SK9_POINTS))
    fd = Vector{Float64}(undef, length(_SK9_POINTS))
    for (i, (a, xi_)) in enumerate(_SK9_POINTS)
        slice = _cheb_slice(SN_TABLE, a)
        ad[i] = ForwardDiff.derivative(x -> _cheb_table_eval(SN_TABLE, slice, x), xi_)
        fd[i] = (_sn_principal_value(a, xi_ + h) - _sn_principal_value(a, xi_ - h)) / (2h)
    end
    return ad, fd
end
const _SK12_AD, _SK12_FD = _sk12_data()
const _SK12_FLOOR = 1e-6 * maximum(abs, _SK12_FD)

@testset "SK12: dR/dxi against central differences of the generator" begin
    for i in eachindex(_SK12_AD)
        @test isapprox(_SK12_AD[i], _SK12_FD[i]; rtol = 1e-6, atol = _SK12_FLOOR)
    end
end

const _SK13_ALPHA = [-2.9995 + (2.9995 - (-2.9995)) * rand(_SK_RNG) for _ in 1:12]
const _SK13_XI = [-20.0 + 40.0 * rand(_SK_RNG) for _ in 1:12]

@testset "SK13: dR/dalpha, and the chain rule" begin
    h = 1e-4
    ad = Vector{Float64}(undef, 12)
    fd = Vector{Float64}(undef, 12)
    for i in 1:12
        a = _SK13_ALPHA[i]
        xi_ = _SK13_XI[i]
        ad[i] = ForwardDiff.derivative(aa -> _cheb_table_eval(SN_TABLE, _cheb_slice(SN_TABLE, aa), xi_), a)
        fd[i] = (_sn_principal_value(a + h, xi_) - _sn_principal_value(a - h, xi_)) / (2h)
    end
    floor_ = 1e-6 * maximum(abs, fd)
    worst_ratio = 0.0
    for i in 1:12
        worst_ratio = max(worst_ratio, abs(ad[i] - fd[i]) / floor_)
        @test isapprox(ad[i], fd[i]; rtol = 1e-6, atol = floor_)
    end
    @info "SK13 worst ratio to its floor" worst_ratio floor_

    # the chain, strictly inside the table, two points at xi < 0
    for (a, xi_) in ((0.5, 3.0), (1.5, -6.0), (-2.0, 8.0), (-2.7, -12.0))
        dRda = ForwardDiff.derivative(aa -> _cheb_table_eval(SN_TABLE, _cheb_slice(SN_TABLE, aa), xi_), a)
        lhs = ForwardDiff.derivative(aa -> real(cauchy_dfdv(SkewNormal(aa), xi_)), a)
        @test isapprox(lhs, dRda; rtol = 1e-12)
    end

    # (c) Phase 4.B spec D47 item 1: the chain at the table's alpha edges, along +alpha and -alpha.
    for (a, xi_) in ((3.0, 1.3), (3.0, -15.0), (-3.0, 1.3), (-3.0, -15.0)), s in (1.0, -1.0)
        lhs = s * ForwardDiff.derivative(t -> real(cauchy_dfdv(SkewNormal(s * t), xi_)), s * a)
        rhs = s * ForwardDiff.derivative(t -> _cheb_table_eval(SN_TABLE, _cheb_slice(SN_TABLE, s * t), xi_), s * a)
        @test lhs === rhs
    end
end

@testset "SK14: panel joins" begin
    interior_edges = SN_TABLE.edges[2:end-1]
    for edge in interior_edges, a in (-3.0, 0.0, 3.0)
        slice = _cheb_slice(SN_TABLE, a)
        v_below = _cheb_table_eval(SN_TABLE, slice, prevfloat(edge))
        v_above = _cheb_table_eval(SN_TABLE, slice, edge)
        @test isapprox(v_below, v_above; rtol = 1e-12, atol = 1e-12)

        d_below = ForwardDiff.derivative(x -> _cheb_table_eval(SN_TABLE, slice, x), edge - 1e-8)
        d_above = ForwardDiff.derivative(x -> _cheb_table_eval(SN_TABLE, slice, x), edge + 1e-8)
        @test isapprox(d_below, d_above; rtol = 1e-6, atol = _SK12_FLOOR)
    end
end

@testset "SK25: the table is not read at |xi|" begin
    # Deviation from spec D38's SK25 row, approved by the user 2026-09-27: xi = -5.0 dropped from
    # the point set. At alpha = 2.0, R's antisymmetric part there measures only 5.94e-4 (below the
    # row's 1e-3 threshold), even though the table matches the independent generator there to
    # ~1e-16; xi = -1.0 and -0.2 already demonstrate the not-read-at-|xi| property robustly
    # (diffs 0.0098 and 0.823 respectively).
    f = SkewNormal(2.0)
    for xi_ in (-1.0, -0.2)
        v_signed = _cheb_table_eval(SN_TABLE, f.slice, xi_)
        v_abs = _cheb_table_eval(SN_TABLE, f.slice, abs(xi_))
        rv = _sn_principal_value(2.0, xi_)
        @test abs(v_signed - v_abs) > 1e-3
        @test isapprox(v_signed, rv; atol = 1e-8)
    end
end

# --- SK10: the independent BigFloat oracle, D38's code verbatim ---

function _sk_dphi_big(a::BigFloat, x::BigFloat)
    d = a / sqrt(1 + a^2)
    v = 1 - 2 * d^2 / big(pi)
    w = 1 / sqrt(2 * v)
    xc = -w * d * sqrt(2 / big(pi))
    z = (x - xc) / w
    pdf(y) = exp(-y^2 / 2) / sqrt(2 * big(pi))
    cdf(y) = erfc(-y / sqrt(big(2))) / 2
    return (2 / w^2) * pdf(z) * (a * pdf(a * z) - z * cdf(a * z))
end
function _sk_R_big(a::Real, xi_::Real)
    setprecision(BigFloat, 64) do
        ab, xb = BigFloat(a), BigFloat(xi_)
        g(u) = (_sk_dphi_big(ab, xb + u) - _sk_dphi_big(ab, xb - u)) / u
        Float64(quadgk(g, big(0), big(1), big(4), big(8), big(Inf);
                       rtol = BigFloat(1e-16), atol = BigFloat(1e-25))[1])
    end
end

@testset "SK10(a): the table against the BigFloat oracle, +xi and -xi separately" begin
    # Through cauchy_dfdv, not _cheb_table_eval directly (Section 14 item 25's load-bearing
    # row: a cauchy_dfdv that read the table at abs(xi) would still pass SK9, since SK9 tests
    # _cheb_table_eval directly, but fails here).
    worst = 0.0
    for a in (-2.2, 0.8, 2.9), xi_ in (0.4, -0.4, 1.7, -1.7, 6.0, -6.0)
        f = SkewNormal(a)
        tv = real(cauchy_dfdv(f, xi_))
        rv = _sk_R_big(a, xi_)
        worst = max(worst, abs(tv - rv))
        @test isapprox(tv, rv; atol = 1e-9)
    end
    @info "SK10(a) worst |table - BigFloat oracle|" worst
end

@testset "SK10(b): the fallback against the BigFloat oracle" begin
    worst = 0.0
    for (a, xi_) in ((4.0, 0.4), (4.0, -1.7), (-6.0, 1.7), (-6.0, -0.4))
        f = SkewNormal(a)
        cv = cauchy_dfdv(f, xi_)
        rv = _sk_R_big(a, xi_)
        worst = max(worst, abs(real(cv) - rv))
        @test isapprox(real(cv), rv; atol = 1e-9)
    end
    @info "SK10(b) worst |fallback - BigFloat oracle|" worst
end

# ------------------------------------------------------------------------------------------
# The public constructor and the rest of the interface (Build step 3): SK15-SK20
# ------------------------------------------------------------------------------------------

@testset "SK15: the branches of cauchy_dfdv" begin
    f = SkewNormal(1.5)

    # (a) inside the table's domain, at the SIGNED xi (never abs(xi))
    for xi_ in (-19.9, -3.3, 0.0, 0.7, 19.9)
        QT = _basefloat(xi_)
        expected = _cheb_table_eval(SN_TABLE, f.slice, xi_) + im * QT(pi) * dfdx(f, xi_)
        @test cauchy_dfdv(f, xi_) === expected
    end

    # (b) |xi| outside the table
    for xi_ in (25.0, -25.0)
        @test cauchy_dfdv(f, xi_) === _sn_cauchy_reference(f, xi_)
    end

    # (c) alpha outside the table
    f4 = SkewNormal(4.0)
    xi_c = 1.3
    @test cauchy_dfdv(f4, xi_c) === _sn_cauchy_reference(f4, xi_c)

    # (d) off the real axis
    @test cauchy_dfdv(f, 1.3 + 0.2im) === _cauchy_upper(f, 1.3 + 0.2im)
    @test_throws "requires Im xi >= 0" cauchy_dfdv(f, 1.3 - 0.2im)

    # (e) the uncached form throws
    fu = _sn_uncached(1.5)
    @test_throws "carries no table slice" cauchy_dfdv(fu, 1.0)

    # (f) not even
    for xi_ in (0.4, 3.0)
        @test abs(real(cauchy_dfdv(f, xi_)) - real(cauchy_dfdv(f, -xi_))) > 1e-2
    end

    # (h) Phase 4.B spec D47 item 1: a dual xi exactly at each xi edge, run with SkewNormal(0.7)
    # (D35 as first written fails both, the amended code passes both).
    fh = SkewNormal(0.7)
    @test ForwardDiff.derivative(x -> real(cauchy_dfdv(fh, x)), 20.0) ===
          ForwardDiff.derivative(x -> _cheb_table_eval(SN_TABLE, fh.slice, x), 20.0)
    @test -ForwardDiff.derivative(t -> real(cauchy_dfdv(fh, -t)), 20.0) ===
          -ForwardDiff.derivative(t -> _cheb_table_eval(SN_TABLE, fh.slice, -t), 20.0)
end

const _SK_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _SK_COLLECTION = Collection(theta_deg = 60, k_model = :exact)
const _SK_LAMBDA_S = 245e-9  # test_shape_cost.jl's SC fixture's own wavelength

_sk_iaw_plasma(f) = Plasma(n_e_cm3 = 5e19,
                            electrons = (electrons(T_eV = 500.0, f = f),),
                            ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))

function _sn_dphi_dalpha_closed(alpha::Real, x::Real)
    f = _sn_uncached(alpha)
    delta = alpha / sqrt(1 + alpha^2)
    mu_delta = delta * sqrt(2 / pi)
    v_delta = 1 - 2 * delta^2 / pi
    omega = f.omega
    xc = f.xc
    z = (x - xc) / omega
    ddelta = (1 + alpha^2)^(-3 / 2)
    dmu = sqrt(2 / pi) * ddelta
    dv = -4 * delta * ddelta / pi
    domega = -omega * dv / (2 * v_delta)
    dxc = -(domega * mu_delta + omega * dmu)
    dz = -(dxc + z * domega) / omega
    phi = f(x)
    az = alpha * z
    return phi * (-domega / omega - z * dz +
                   (_std_normal_pdf(az) / _std_normal_cdf(az)) * (z + alpha * dz))
end

@testset "SK16: phi's alpha-partial, and the spectral density's" begin
    xs = collect(-5.0:0.05:5.0)
    alphas = (-3.0, -1.2, 0.0, 0.4, 2.5, 5.0)
    worst_phi = 0.0
    for alpha in alphas
        for x in xs
            ad = ForwardDiff.derivative(aa -> _sn_uncached(aa)(x), alpha)
            closed = _sn_dphi_dalpha_closed(alpha, x)
            worst_phi = max(worst_phi, abs(ad - closed))
            @test isapprox(ad, closed; rtol = 1e-11, atol = 1e-14)
        end
    end
    @info "SK16 worst phi alpha-partial error" worst_phi

    alphas2 = (-2.5, -0.4, 0.0, 1.1, 2.8, 3.5)  # the last runs the fallback
    h = 1e-4
    fd = Dict{Float64,Float64}()
    for alpha in alphas2
        plasma_p = _sk_iaw_plasma(SkewNormal(alpha + h))
        plasma_m = _sk_iaw_plasma(SkewNormal(alpha - h))
        (k, om, _) = wavevector(_SK_PROBE, _SK_COLLECTION, _SK_LAMBDA_S, plasma_p)
        (km, omm, _) = wavevector(_SK_PROBE, _SK_COLLECTION, _SK_LAMBDA_S, plasma_m)
        sp = spectral_density(plasma_p, k, om)
        sm = spectral_density(plasma_m, km, omm)
        fd[alpha] = (sp - sm) / (2h)
    end
    floor_ = 1e-6 * maximum(abs, values(fd))
    worst_S = 0.0
    worst_ratio = 0.0
    for alpha in alphas2
        fad = SkewNormal(ForwardDiff.Dual{:SK16}(alpha, 1.0))
        plasma = _sk_iaw_plasma(fad)
        (k, om, _) = wavevector(_SK_PROBE, _SK_COLLECTION, _SK_LAMBDA_S, plasma)
        s = spectral_density(plasma, k, om)
        ad = ForwardDiff.partials(s)[1]
        worst_S = max(worst_S, abs(ad - fd[alpha]))
        worst_ratio = max(worst_ratio, abs(ad - fd[alpha]) / (1e-6 * abs(fd[alpha]) + floor_))
        @test isapprox(ad, fd[alpha]; rtol = 1e-6, atol = floor_)
    end
    @info "SK16 worst spectral-density alpha-derivative error, and ratio to its allowance" worst_S worst_ratio floor_
end

@testset "SK17: allocations" begin
    f = SkewNormal(0.7)
    reconstruct(f, (alpha = 0.9,))
    @test (@allocated reconstruct(f, (alpha = 0.9,))) == 0

    ad = ForwardDiff.Dual{:SK17}(0.7, 1.0)
    reconstruct(f, (alpha = ad,))
    @test (@allocated reconstruct(f, (alpha = ad,))) == 0

    _cheb_table_eval(SN_TABLE, f.slice, 1.3)
    @test (@allocated _cheb_table_eval(SN_TABLE, f.slice, 1.3)) == 0

    xd = ForwardDiff.Dual{:SK17}(1.3, 1.0)
    _cheb_table_eval(SN_TABLE, f.slice, xd)
    @test (@allocated _cheb_table_eval(SN_TABLE, f.slice, xd)) == 0

    fad = SkewNormal(ForwardDiff.Dual{:SK17}(0.7, 1.0))
    _cheb_table_eval(SN_TABLE, fad.slice, 1.3)
    @test (@allocated _cheb_table_eval(SN_TABLE, fad.slice, 1.3)) == 0

    f(0.7)
    @test (@allocated f(0.7)) == 0
    dfdx(f, 0.7)
    @test (@allocated dfdx(f, 0.7)) == 0

    xd2 = ForwardDiff.Dual{:SK17}(0.7, 1.0)
    f(xd2)
    @test (@allocated f(xd2)) == 0
    dfdx(f, xd2)
    @test (@allocated dfdx(f, xd2)) == 0

    fad(0.7)
    @test (@allocated fad(0.7)) == 0
    dfdx(fad, 0.7)
    @test (@allocated dfdx(fad, 0.7)) == 0
end

@testset "SK18: _widen_shape copies" begin
    f = SkewNormal(0.7)
    D = ForwardDiff.Dual{:SK18,Float64,1}
    g = @inferred _widen_shape(f, D)
    @test g isa SkewNormal
    @test g.alpha === D(f.alpha)
    @test g.omega === D(f.omega)
    @test g.xc === D(f.xc)
    @test all(iszero, ForwardDiff.partials(g.alpha))
    @test all(iszero, ForwardDiff.partials(g.omega))
    @test all(iszero, ForwardDiff.partials(g.xc))
    @test g.slice == D.(f.slice)
end

@testset "SK19: promote first" begin
    f32 = SkewNormal(0.4f0)
    f64 = SkewNormal(Float64(0.4f0))
    @test typeof(f32) === typeof(f64)
    @test f32.alpha === f64.alpha
    @test f32.omega === f64.omega
    @test f32.xc === f64.xc
    @test f32.slice === f64.slice
end

@testset "SK20: parameters, reconstruct, show, and the constructor's throws" begin
    f = SkewNormal(0.7)
    @test parameters(f) == (alpha = 0.7,)
    g = reconstruct(f, (alpha = -1.1,))
    @test g.slice === _cheb_slice(SN_TABLE, -1.1)
    @test sprint(show, f) == "SkewNormal(alpha = 0.7)"
    @test_throws "alpha must be finite" SkewNormal(Inf)
    @test_throws "alpha must be finite" SkewNormal(NaN)
    @test SkewNormal(-50.0) isa SkewNormal
end

# ------------------------------------------------------------------------------------------
# Registration (Build step 4): SK21, SC1-SC5 (in test_shape_cost.jl)
# ------------------------------------------------------------------------------------------

function _sk21_deck(datafile, outdir)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    alpha_e = { bounds = [-3.0, 3.0], init = 0.0 }
    T_D = { bounds = [50.0, 800.0], init = 200.0 }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0
    shape = { type = "SkewNormal", alpha = "alpha_e" }

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

function _sk21_write_streak(path)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = SkewNormal(0.7))],
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

@testset "SK21: registration" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _sk21_write_streak(datafile)
        outdir = joinpath(dir, "out")

        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _sk21_deck(datafile, outdir))
        deck = read_deck(deckpath)
        @test deck.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "SkewNormal"

        problem, optimizer, start = OTIS.build_problem(deck)
        b = only(b for b in problem.pspec.bindings if b.name == "alpha_e")
        @test b.kind === :leaf
        @test string(b.path) == "electrons.e.f.alpha"

        eff = effective_deck(deck, problem, optimizer)
        @test occursin("SkewNormal", eff)
        efftmp = joinpath(dir, "effective.toml")
        write(efftmp, eff)
        deck2 = read_deck(efftmp)
        @test deck2.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "SkewNormal"

        # an unknown shape key is refused
        bad = replace(_sk21_deck(datafile, outdir), "alpha = \"alpha_e\"" => "m = 3.0")
        badpath = joinpath(dir, "bad.toml")
        write(badpath, bad)
        @test_throws "unknown key \"m\"" read_deck(badpath)

        # a PlasmaTrace with a SkewNormal electron round-trips through write_trace/read_trace
        template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = SkewNormal(-1.3))],
            ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0)])
        trace = PlasmaTrace(template; t_ns = [0.0, 1.0])
        tracepath = joinpath(dir, "trace.jld2")
        write_trace(tracepath, trace)
        trace2 = read_trace(tracepath; as = :trace)
        p1 = parameters(trace.template.electrons["e"].f)
        p2 = parameters(trace2.template.electrons["e"].f)
        @test p1.alpha === p2.alpha
    end
end

# ------------------------------------------------------------------------------------------
# The fits and the regression (Build step 5): SK22, SK23, SK24
# ------------------------------------------------------------------------------------------

# KP22's fixture, with the shape, variable and binding replaced (spec D35, D38).
function _sk_p1_problem(alpha_truth::Real)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = SkewNormal(0.0))],
        ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 200.0, u_kms = 0.0)])
    trace = PlasmaTrace(template; t_ns = [0.0])
    expt = Experiment(trace = trace, probe = ProbeBeam(lambda_probe_nm = 263.25),
        collection = Collection(theta_deg = 60.0, k_model = :exact),
        lambda_m = collect(range(262.0, 264.5; length = 24)) .* 1e-9,
        effects = (Sensitivity(scale = 1.0e-6), ConstantBackground(level = 25.0)))
    va = FittedVariable("alpha_e"; bounds = (-3.0, 3.0), init = 0.0)
    vT = FittedVariable("T_D"; bounds = (50.0, 800.0), init = 200.0)
    pspec = ParameterSpec(expt; variables = [va, vT],
        bindings = ["alpha_e" => "electrons.e.f.alpha", "T_D" => "ions.D.T_eV"])
    working = Experiment(; trace = pspec.trace, probe = expt.probe, collection = expt.collection,
        lambda_m = expt.lambda, effects = expt.effects, prefactor = expt.prefactor)
    theta = flatten(pspec, ["alpha_e" => alpha_truth, "T_D" => 250.0])
    image = measure(unflatten(working, pspec, theta))
    data = StreakData(lambda_m = working.lambda, t_s = working.t, image = image,
        sigma = ones(size(image)))
    return FitProblem(working, pspec, data), pspec
end

@testset "SK22: the P1-sized recovery" begin
    # alpha crosses zero: an absolute tolerance in alpha (spec D38; a relative one is undefined
    # at alpha = 0 and inflated beside it, Section 14 item 29).
    for (alpha_truth, alpha_start) in ((1.5, 0.0), (-2.0, 0.5), (0.4, -1.0))
        problem, pspec = _sk_p1_problem(alpha_truth)
        start = flatten(pspec, ["alpha_e" => alpha_start, "T_D" => 200.0])
        result = solve(problem, LevenbergMarquardt(); start = start)
        alpha_hat = result.values["alpha_e"][1]
        TD_hat = result.values["T_D"][1]
        errA = abs(alpha_hat - alpha_truth)
        errT = abs(TD_hat - 250.0) / 250.0
        @info "SK22" alpha_truth alpha_start errA errT result.converged result.iterations
        @test errA < 1e-6
        @test errT < 1e-6
        @test result.converged
    end
end

@testset "SK23: the IAW asymmetry, pinned" begin
    plasma_M = _sk_iaw_plasma(Maxwellian())
    (k, _, _) = wavevector(_SK_PROBE, _SK_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k * cs)
    sM = [spectral_density(plasma_M, k, om) for om in omegas]
    idx = argmax(sM)
    @info "SK23 argmax index and omega0/(k*cs)" idx (omegas[idx] / (k * cs))
    @test idx == 309
    omega0 = omegas[idx]

    ratio_M = spectral_density(plasma_M, k, omega0) / spectral_density(plasma_M, k, -omega0)
    @test isapprox(ratio_M, 1.0; rtol = 1e-12)

    alphas = (-2.0, -0.5, 0.5, 1.0, 2.0, 3.0)
    expected = (2.359833549471158, 1.0374087378614372, 0.9639402132484894, 0.8039379690481419,
                0.42375870121182974, 0.22699947493509598)
    ratios = Float64[]
    for alpha in alphas
        plasma_a = _sk_iaw_plasma(SkewNormal(alpha))
        rp = spectral_density(plasma_a, k, omega0)
        rm = spectral_density(plasma_a, k, -omega0)
        push!(ratios, rp / rm)
    end
    @info "SK23 ratios as computed" ratios expected
    for (r, e) in zip(ratios, expected)
        @test isapprox(r, e; rtol = 1e-7)
    end

    products = Float64[]
    for alpha in (0.5, 2.0)
        plasma_p = _sk_iaw_plasma(SkewNormal(alpha))
        plasma_m = _sk_iaw_plasma(SkewNormal(-alpha))
        ratio_p = spectral_density(plasma_p, k, omega0) / spectral_density(plasma_p, k, -omega0)
        ratio_m = spectral_density(plasma_m, k, omega0) / spectral_density(plasma_m, k, -omega0)
        push!(products, ratio_p * ratio_m)
        @test isapprox(ratio_p * ratio_m, 1.0; rtol = 1e-9)
    end
    @info "SK23 reflection products ratio(alpha)*ratio(-alpha)" products
end

@testset "SK24: every fixture stays inside the table (signed)" begin
    # SK16's fixture
    lo16, hi16 = Inf, -Inf
    for alpha in (-2.5, -0.4, 0.0, 1.1, 2.8, 3.5)
        plasma = _sk_iaw_plasma(SkewNormal(alpha))
        (k, om, _) = wavevector(_SK_PROBE, _SK_COLLECTION, _SK_LAMBDA_S, plasma)
        sp_e = plasma.electrons["e"]
        xie = xi(sp_e, k, om)
        lo16, hi16 = min(lo16, xie), max(hi16, xie)
    end
    @test lo16 >= SN_TABLE.xi_lo && hi16 <= SN_TABLE.xi_max
    @info "SK24 SK16 fixture range of xi_e" lo16 hi16

    # SK23's fixture
    plasma_M = _sk_iaw_plasma(Maxwellian())
    (k23, _, _) = wavevector(_SK_PROBE, _SK_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k23 * cs)
    idx = argmax([spectral_density(plasma_M, k23, om) for om in omegas])
    omega0 = omegas[idx]
    lo23, hi23 = Inf, -Inf
    for alpha in (-2.0, -0.5, 0.5, 1.0, 2.0, 3.0), om in (omega0, -omega0)
        plasma_a = _sk_iaw_plasma(SkewNormal(alpha))
        sp_e = plasma_a.electrons["e"]
        xie = xi(sp_e, k23, om)
        lo23, hi23 = min(lo23, xie), max(hi23, xie)
    end
    @test lo23 >= SN_TABLE.xi_lo && hi23 <= SN_TABLE.xi_max
    @info "SK24 SK23 fixture range of xi_e" lo23 hi23

    # SK22's fixture, all three truths, all 24 wavelengths
    lo22, hi22 = Inf, -Inf
    for alpha_truth in (1.5, -2.0, 0.4)
        problem, pspec = _sk_p1_problem(alpha_truth)
        theta = flatten(pspec, ["alpha_e" => alpha_truth, "T_D" => 250.0])
        expt_truth = unflatten(problem.experiment, pspec, theta)
        plasma_truth = plasma(expt_truth.trace, 1)
        sp_e = plasma_truth.electrons["e"]
        for lam in expt_truth.lambda
            (k22, om22, _) = wavevector(expt_truth.probe, expt_truth.collection, lam, plasma_truth)
            xie = xi(sp_e, k22, om22)
            lo22, hi22 = min(lo22, xie), max(hi22, xie)
        end
    end
    @test lo22 >= SN_TABLE.xi_lo && hi22 <= SN_TABLE.xi_max
    @info "SK24 SK22 fixture range of xi_e" lo22 hi22
end
