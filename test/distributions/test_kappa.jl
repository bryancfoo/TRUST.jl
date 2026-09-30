# Step 4.B.2b: Kappa, the shape (spec D16-D20, D24). KP1-KP24, gate "Kappa". Section 10.1's
# formulas, in the code's normalization; every oracle independent of the code under test
# (conventions.md 8.2): a quadrature, a BigFloat evaluation, the integer-kappa residue closed
# form, Zprime, ForwardDiff or a finite difference, never the function itself.

using Test
using OTIS
using OTIS: KAPPA_TABLE, KAPPA_DEGREE_START, KAPPA_DEGREE_CAP_XI, KAPPA_DEGREE_CAP_S,
            KAPPA_BISECT_MAX, _kappa_uncached, _kappa_cauchy_reference, _kappa_principal_value,
            _cheb_slice, _cheb_table_eval, _basefloat, _cauchy_upper, _widen_shape, SHAPE_REGISTRY
using SpecialFunctions: loggamma, digamma, gamma
using QuadGK: quadgk
using Random
import ForwardDiff

include(joinpath(@__DIR__, "..", "shape_contract.jl"))

# ------------------------------------------------------------------------------------------
# KP1-KP7: the shape without its table (Build step 1)
# ------------------------------------------------------------------------------------------

@testset "KP1: the three normalizations" begin
    for kappa in (2.0, 2.5, 3.7, 10.0, 1000.0)
        check_shape_normalizations(Kappa(kappa))
    end
end

@testset "KP2: C against a 128-bit BigFloat oracle (MPFR's gamma)" begin
    setprecision(BigFloat, 128) do
        for kappa in (1.6, 2.0, 2.5, 3.7, 10.0, 100.0)
            f = Kappa(kappa)
            kb = big(kappa)
            Cb = gamma(kb) / (sqrt(big(pi)) * sqrt(kb - 3 // 2) * gamma(kb - 1 // 2))
            @test isapprox(f.C, Float64(Cb); rtol = 1e-13)
        end
        f1000 = Kappa(1000.0)
        kb = big(1000.0)
        Cb = gamma(kb) / (sqrt(big(pi)) * sqrt(kb - 3 // 2) * gamma(kb - 1 // 2))
        @test isapprox(f1000.C, Float64(Cb); rtol = 5e-12)
    end
    f2 = Kappa(2.0)
    @test isapprox(f2.C, 2 * sqrt(2) / pi; rtol = 1e-15)
end

@testset "KP3: C normalization on a fine grid (an independent quadrature)" begin
    worst = 0.0
    kappas = vcat(collect(1.6:0.005:12.0), [15.0, 20.0, 50.0, 100.0, 300.0])
    for kappa in kappas
        f = Kappa(kappa)
        a = sqrt(kappa - 1.5)
        I, _ = quadgk(x -> exp(-kappa * log1p((x / a)^2)), -Inf, 0.0, Inf; rtol = 1e-13, atol = 1e-15)
        val = f.C * I
        worst = max(worst, abs(val - 1))
        @test isapprox(val, 1; rtol = 1e-13)
    end
    @info "KP3 worst |C*I - 1| over the grid" worst
end

@testset "KP4: reduction to the Maxwellian as kappa -> infinity" begin
    for kappa in (1e2, 1e3, 1e4)
        f = Kappa(kappa)
        @test abs(kappa * (f.C * sqrt(pi) - 1) - 3 / 8) <= 1 / kappa
        maxdiff = maximum(abs(f(x) - exp(-x^2) / sqrt(pi)) for x in -6:0.001:6)
        @test abs(kappa * maxdiff - 0.2116) <= 0.005
    end
end

@testset "KP5: dfdx vs ForwardDiff, and x=0 under a dual kappa" begin
    xs = collect(-8.0:0.05:8.0)
    for kappa in (2.0, 3.7, 10.0)
        f = Kappa(kappa)
        for x in xs
            ad = ForwardDiff.derivative(y -> f(y), x)
            @test isapprox(ad, dfdx(f, x); rtol = 1e-13, atol = 1e-15)
        end
    end
    kappa0 = 3.7
    f0 = Kappa(kappa0)
    ad0 = ForwardDiff.derivative(k -> Kappa(k)(0.0), kappa0)
    @test isfinite(ad0)
    closed = f0.C * (digamma(kappa0) - digamma(kappa0 - 0.5) - 1 / (2 * f0.a^2))
    @test isapprox(ad0, closed; rtol = 1e-12)
end

@testset "KP6: moments, the divergence boundary, and the derived quantities" begin
    for kappa in (3.7, 10.0)
        f = Kappa(kappa)
        for n in 0:4
            ref, _ = quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-12, atol = 1e-14)
            @test isapprox(moment(f, n), ref; rtol = 1e-11, atol = 1e-12)
        end
    end
    let kappa = 2.0, f = Kappa(kappa)
        for n in 0:2
            ref, _ = quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-12, atol = 1e-14)
            @test isapprox(moment(f, n), ref; rtol = 1e-11, atol = 1e-12)
        end
    end
    @test moment(Kappa(2.0), 4) === Inf
    @test moment(Kappa(2.5), 4) === Inf
    @test moment(Kappa(3.0), 6) === Inf
    @test isfinite(moment(Kappa(3.7), 6))
    for kappa in (2.0, 2.5, 3.0, 3.7, 10.0), n in (1, 3, 5)
        @test moment(Kappa(kappa), n) === 0.0
    end
    @test_throws ArgumentError moment(Kappa(3.7), -1)
    for kappa in (3.0, 3.7, 10.0)
        @test isapprox(excess_kurtosis(Kappa(kappa)), 3 / (kappa - 2.5); rtol = 1e-12)
    end
    @test excess_kurtosis(Kappa(2.0)) === Inf
    @test skewness(Kappa(3.7)) == 0
end

@testset "KP7: the label -- the 3-D marginal identity" begin
    for kappa in (2.0, 3.5, 7.0), x in (0.0, 0.7, 2.5)
        a = sqrt(kappa - 1.5)
        lhs, _ = quadgk(r -> 2 * pi * r * (1 + (x^2 + r^2) / a^2)^(-(kappa + 1)), 0, Inf;
                         rtol = 1e-13, atol = 1e-300)
        rhs = (pi * a^2 / kappa) * (1 + x^2 / a^2)^(-kappa)
        @test isapprox(lhs, rhs; rtol = 1e-12)
    end
end

# ------------------------------------------------------------------------------------------
# The oracle (Build step 2): _kp_residue, D17, checked before it is used (KP10 (a),(b),(d))
# ------------------------------------------------------------------------------------------

"""
    _kp_D(n::Int, xi::T) where {T<:Real} -> Complex{T}

D17's D(xi) for integer kappa = n >= 2, generic in T (Float64 or BigFloat), written from Section
10.1 with no package function but `loggamma`: the rising-factorial coefficients p_j = (n)_j/j! are
built in exact BigInt arithmetic (always integers) and converted to T only at the end, so no
precision is lost regardless of T. Test-local only (conventions.md 8.2): an oracle must not share
machinery with the code under test.
"""
function _kp_D(n::Int, xi::T) where {T<:Real}
    n >= 2 || throw(ArgumentError("_kp_D: n must be an integer >= 2; got $n"))
    kappa = T(n)
    a = sqrt(kappa - 3 * one(T) / 2)
    C = exp(loggamma(kappa) - loggamma(kappa - one(T) / 2)) / (sqrt(T(pi)) * a)
    pint = Vector{BigInt}(undef, n)
    pint[1] = 1
    for j in 1:n-1
        pint[j+1] = div(pint[j] * (n + j - 1), j)
    end
    p = T.(pint)
    ia = a * im
    total = zero(Complex{T})
    for j in 0:n-1
        total += p[j+1] * (n - j) * (2 * ia)^(-n - j) * (ia - xi)^(j - n - 1)
    end
    return C * a^(2n) * (-1)^(n - 1) * total
end

"""
    _kp_residue(n::Int, xi::T) where {T<:Real} -> T

R(xi) = Re(2*pi*i*D(xi)), the real part the table serves (D17). Test-local, generic in T.
"""
_kp_residue(n::Int, xi::T) where {T<:Real} = real(2 * T(pi) * im * _kp_D(n, xi))

@testset "KP10(a): _kp_residue against the reference quadrature, before it is used" begin
    worst = 0.0
    for n in (2, 3, 4, 5), xi in (0.3, 1.7, 6.0)
        ref = real(_kappa_cauchy_reference(_kappa_uncached(n), xi))
        val = _kp_residue(n, xi)
        worst = max(worst, abs(val - ref))
        @test isapprox(val, ref; rtol = 1e-9, atol = 1e-12)
    end
    @info "KP10(a) worst |residue - reference|" worst
end

"""
    _kp_d2phi_big(x, kappa, a, C)

phi''(x) in closed form, independent BigFloat arithmetic, for the removable u=0 singularity in
KP10(b)'s symmetric-difference quadrature (SG11's `_cauchy_ref_big` precedent).
"""
function _kp_d2phi_big(x, kappa, a, C)
    t = 1 + (x / a)^2
    dtdx = 2x / a^2
    return -(2 * kappa * C / a^2) * (t^(-(kappa + 1)) + x * (-(kappa + 1)) * t^(-(kappa + 2)) * dtdx)
end

@testset "KP10(b): _kp_residue in BigFloat against a BigFloat quadrature" begin
    setprecision(BigFloat, 128) do
        worst = big(0.0)
        for n in (2, 3, 4), xi0 in (0.3, 1.7, 6.0)
            xib = big(xi0)
            kappa = big(n)
            a = sqrt(kappa - 3 // 2)
            C = exp(loggamma(kappa) - loggamma(kappa - 1 // 2)) / (sqrt(big(pi)) * a)
            dphi(x) = -(2 * kappa * C / a^2) * x * (1 + (x / a)^2)^(-(kappa + 1))
            g(u) = iszero(u) ? 2 * _kp_d2phi_big(xib, kappa, a, C) : (dphi(xib + u) - dphi(xib - u)) / u
            pv, _ = quadgk(g, big(0), big(6), big(Inf); rtol = big(1e-25), atol = big(1e-40))
            val = _kp_residue(n, xib)
            worst = max(worst, abs(val - pv))
            @test isapprox(val, pv; atol = big(1e-30))
        end
        @info "KP10(b) worst |residue - BigFloat quadrature|" Float64(worst)
    end
end

@testset "KP10(d): the Plemelj check" begin
    worst = 0.0
    for n in (2, 3, 4, 5), xi in (0.3, 1.7, 6.0)
        Dval = _kp_D(n, xi)
        im_part = imag(2 * pi * im * Dval)
        f = Kappa(Float64(n))
        val = im_part + pi * dfdx(f, xi)
        worst = max(worst, abs(val))
        @test isapprox(val, 0.0; atol = 1e-13)
    end
    @info "KP10(d) worst Plemelj residual" worst
end

# ------------------------------------------------------------------------------------------
# The table (Build step 3): KP8, KP9, KP10(c), KP11-KP14
# ------------------------------------------------------------------------------------------

@testset "KP8: table structure" begin
    @test KAPPA_TABLE.p_lo == 0.0
    @test KAPPA_TABLE.p_hi == 0.5
    @test KAPPA_TABLE.xi_lo == 0.0
    @test KAPPA_TABLE.xi_max == 20.0
    @test all(KAPPA_TABLE.xi_converged)
    @test KAPPA_TABLE.p_converged
    @test KAPPA_TABLE.np <= KAPPA_DEGREE_CAP_S
    @test KAPPA_TABLE.degree_start == KAPPA_DEGREE_START
    @test KAPPA_TABLE.cap_xi == KAPPA_DEGREE_CAP_XI
    @test KAPPA_TABLE.cap_p == KAPPA_DEGREE_CAP_S
    @test KAPPA_TABLE.bisect_max == KAPPA_BISECT_MAX
    L = sum(KAPPA_TABLE.nxi .+ 1)
    @test length(KAPPA_TABLE.coeffs) == L * (KAPPA_TABLE.np + 1)
    @test length(KAPPA_TABLE.offset) == length(KAPPA_TABLE.nxi)  # offset has one entry per panel, not per slice row
    @info "KP8 table" edges = KAPPA_TABLE.edges nxi = KAPPA_TABLE.nxi np = KAPPA_TABLE.np L = L
end

const _KP_RNG = Random.Xoshiro(20260926)
const _KP9_POINTS = [(0.5 * rand(_KP_RNG), 20.0 * rand(_KP_RNG)) for _ in 1:60]

@testset "KP9: the table against its generator" begin
    worst = 0.0
    for (s, xi) in _KP9_POINTS
        slice = _cheb_slice(KAPPA_TABLE, s)
        tv = _cheb_table_eval(KAPPA_TABLE, slice, xi)
        rv = _kappa_principal_value(s, xi)
        worst = max(worst, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    for edge in KAPPA_TABLE.edges, s in (0.0, 0.25, 0.5)
        slice = _cheb_slice(KAPPA_TABLE, s)
        tv = _cheb_table_eval(KAPPA_TABLE, slice, edge)
        rv = _kappa_principal_value(s, edge)
        worst = max(worst, abs(tv - rv))
        @test isapprox(tv, rv; rtol = 1e-8, atol = 1e-8)
    end
    @info "KP9 worst |table - generator|" worst
end

@testset "KP10(c): the table against the residue oracle" begin
    worst = 0.0
    xis401 = collect(0.0:0.05:20.0)
    kp9_xis = [xi for (_, xi) in _KP9_POINTS]
    for n in (2, 3, 4, 5, 10)
        s = 1 / n
        slice = _cheb_slice(KAPPA_TABLE, s)
        for xi in vcat(xis401, kp9_xis)
            tv = _cheb_table_eval(KAPPA_TABLE, slice, xi)
            rv = _kp_residue(n, xi)
            worst = max(worst, abs(tv - rv))
            @test isapprox(tv, rv; atol = 1e-9, rtol = 0)
        end
    end
    @info "KP10(c) worst |table - residue|" worst
end

@testset "KP11: the Maxwellian edge against Zprime" begin
    worst = 0.0
    slice0 = _cheb_slice(KAPPA_TABLE, 0.0)
    for xi in 0.0:0.05:20.0
        tv = _cheb_table_eval(KAPPA_TABLE, slice0, xi)
        zp = real(Zprime(xi))
        worst = max(worst, abs(tv - zp))
        @test isapprox(tv, zp; atol = 1e-9, rtol = 0)
    end
    @info "KP11 worst |table(s=0) - Re Zprime|" worst
end

function _kp12_data()
    h = 1e-4
    ad = Vector{Float64}(undef, length(_KP9_POINTS))
    fd = Vector{Float64}(undef, length(_KP9_POINTS))
    for (i, (s, xi)) in enumerate(_KP9_POINTS)
        slice = _cheb_slice(KAPPA_TABLE, s)
        ad[i] = ForwardDiff.derivative(x -> _cheb_table_eval(KAPPA_TABLE, slice, x), xi)
        fd[i] = (_kappa_principal_value(s, xi + h) - _kappa_principal_value(s, xi - h)) / (2h)
    end
    return ad, fd
end
const _KP12_AD, _KP12_FD = _kp12_data()
const _KP12_FLOOR = 1e-6 * maximum(abs, _KP12_FD)

@testset "KP12: dR/dxi against central differences of the generator" begin
    for i in eachindex(_KP12_AD)
        @test isapprox(_KP12_AD[i], _KP12_FD[i]; rtol = 1e-6, atol = _KP12_FLOOR)
    end
end

const _KP13_S = [2e-4 + (0.4998 - 2e-4) * rand(_KP_RNG) for _ in 1:12]
const _KP13_XI = [20.0 * rand(_KP_RNG) for _ in 1:12]

@testset "KP13: dR/ds against central differences, and the chain rule" begin
    h = 1e-4
    ad = Vector{Float64}(undef, 12)
    fd = Vector{Float64}(undef, 12)
    for i in 1:12
        s = _KP13_S[i]
        xi = _KP13_XI[i]
        ad[i] = ForwardDiff.derivative(ss -> _cheb_table_eval(KAPPA_TABLE, _cheb_slice(KAPPA_TABLE, ss), xi), s)
        fd[i] = (_kappa_principal_value(s + h, xi) - _kappa_principal_value(s - h, xi)) / (2h)
    end
    floor_ = 1e-6 * maximum(abs, fd)
    worst_ratio = 0.0
    for i in 1:12
        worst_ratio = max(worst_ratio, abs(ad[i] - fd[i]) / floor_)
        @test isapprox(ad[i], fd[i]; rtol = 1e-6, atol = floor_)
    end
    @info "KP13 worst ratio to its floor" worst_ratio floor_

    for (kappa, xi) in ((2.5, 1.0), (4.0, 5.0), (7.0, 10.0), (15.0, 15.0))
        s = 1 / kappa
        dRds = ForwardDiff.derivative(ss -> _cheb_table_eval(KAPPA_TABLE, _cheb_slice(KAPPA_TABLE, ss), xi), s)
        lhs = ForwardDiff.derivative(k -> real(cauchy_dfdv(Kappa(k), xi)), kappa)
        rhs = -(1 / kappa^2) * dRds
        @test isapprox(lhs, rhs; rtol = 1e-12)
    end
end

@testset "KP14: panel joins" begin
    interior_edges = KAPPA_TABLE.edges[2:end-1]
    for edge in interior_edges, s in (0.0, 0.25, 0.5)
        slice = _cheb_slice(KAPPA_TABLE, s)
        v_below = _cheb_table_eval(KAPPA_TABLE, slice, prevfloat(edge))
        v_above = _cheb_table_eval(KAPPA_TABLE, slice, edge)
        @test isapprox(v_below, v_above; rtol = 1e-12, atol = 1e-12)

        d_below = ForwardDiff.derivative(x -> _cheb_table_eval(KAPPA_TABLE, slice, x), edge - 1e-8)
        d_above = ForwardDiff.derivative(x -> _cheb_table_eval(KAPPA_TABLE, slice, x), edge + 1e-8)
        @test isapprox(d_below, d_above; rtol = 1e-6, atol = _KP12_FLOOR)
    end
end

# ------------------------------------------------------------------------------------------
# The public constructor and the rest of the interface (Build step 4): KP15-KP20
# ------------------------------------------------------------------------------------------

@testset "KP15: the branches of cauchy_dfdv" begin
    f = Kappa(3.5)

    # (a) inside the table's domain
    for xi in (-3.3, 0.0, 0.7, 19.9)
        QT = _basefloat(xi)
        expected = _cheb_table_eval(KAPPA_TABLE, f.slice, abs(xi)) + im * QT(pi) * dfdx(f, xi)
        @test cauchy_dfdv(f, xi) === expected
    end

    # (b) |xi| outside the table
    xi_b = 25.0
    @test cauchy_dfdv(f, xi_b) === _kappa_cauchy_reference(f, xi_b)

    # (c) kappa outside the table (s > p_hi)
    f18 = Kappa(1.8)
    xi_c = 1.3
    @test cauchy_dfdv(f18, xi_c) === _kappa_cauchy_reference(f18, xi_c)

    # (d) off the real axis
    @test cauchy_dfdv(f, 1.3 + 0.2im) === _cauchy_upper(f, 1.3 + 0.2im)
    @test_throws "requires Im xi >= 0" cauchy_dfdv(f, 1.3 - 0.2im)

    # (e) the uncached form throws
    fu = _kappa_uncached(3.5)
    @test_throws "carries no table slice" cauchy_dfdv(fu, 1.0)

    # (f) evenness
    for xi in (0.4, 3.0, 12.0)
        @test real(cauchy_dfdv(f, -xi)) == real(cauchy_dfdv(f, xi))
        @test imag(cauchy_dfdv(f, -xi)) == -imag(cauchy_dfdv(f, xi))
    end
end

const _KP_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _KP_COLLECTION = Collection(theta_deg = 60, k_model = :exact)
const _KP_LAMBDA_S = 245e-9  # test_shape_cost.jl's SC fixture's own wavelength

_kp_iaw_plasma(f) = Plasma(n_e_cm3 = 5e19,
                            electrons = (electrons(T_eV = 500.0, f = f),),
                            ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))

@testset "KP16: phi's kappa-partial, and the spectral density's" begin
    xs = collect(-10.0:0.1:10.0)
    kappas = (2.2, 3.5, 10.0, 100.0)
    worst_phi = 0.0
    for kappa in kappas
        f = Kappa(kappa)
        for x in xs
            ad = ForwardDiff.derivative(k -> Kappa(k)(x), kappa)
            a = f.a
            closed = f(x) * (digamma(kappa) - digamma(kappa - 0.5) - 1 / (2a^2) -
                              log1p((x / a)^2) + kappa * x^2 / (a^2 * (a^2 + x^2)))
            worst_phi = max(worst_phi, abs(ad - closed))
            @test isapprox(ad, closed; rtol = 1e-11, atol = 1e-14)
        end
    end
    @info "KP16 worst phi kappa-partial error" worst_phi

    kappas2 = (2.2, 3.5, 6.0, 15.0, 60.0)
    fd = Dict{Float64,Float64}()
    for kappa in kappas2
        h = 1e-4 * kappa
        plasma_p = _kp_iaw_plasma(Kappa(kappa + h))
        plasma_m = _kp_iaw_plasma(Kappa(kappa - h))
        (k, om, _) = wavevector(_KP_PROBE, _KP_COLLECTION, _KP_LAMBDA_S, plasma_p)
        (km, omm, _) = wavevector(_KP_PROBE, _KP_COLLECTION, _KP_LAMBDA_S, plasma_m)
        sp = spectral_density(plasma_p, k, om)
        sm = spectral_density(plasma_m, km, omm)
        fd[kappa] = (sp - sm) / (2h)
    end
    floor_ = 1e-6 * maximum(abs, values(fd))
    worst_S = 0.0
    for kappa in kappas2
        fkd = Kappa(ForwardDiff.Dual{:KP16}(kappa, 1.0))
        plasma = _kp_iaw_plasma(fkd)
        (k, om, _) = wavevector(_KP_PROBE, _KP_COLLECTION, _KP_LAMBDA_S, plasma)
        s = spectral_density(plasma, k, om)
        ad = ForwardDiff.partials(s)[1]
        worst_S = max(worst_S, abs(ad - fd[kappa]))
        @test isapprox(ad, fd[kappa]; rtol = 1e-6, atol = floor_)
    end
    @info "KP16 worst spectral-density kappa-derivative error" worst_S floor_
end

@testset "KP17: allocations" begin
    f = Kappa(3.5)
    reconstruct(f, (kappa = 3.9,))
    @test (@allocated reconstruct(f, (kappa = 3.9,))) == 0

    kd = ForwardDiff.Dual{:KP17}(3.5, 1.0)
    reconstruct(f, (kappa = kd,))
    @test (@allocated reconstruct(f, (kappa = kd,))) == 0

    _cheb_table_eval(KAPPA_TABLE, f.slice, 1.3)
    @test (@allocated _cheb_table_eval(KAPPA_TABLE, f.slice, 1.3)) == 0

    xd = ForwardDiff.Dual{:KP17}(1.3, 1.0)
    _cheb_table_eval(KAPPA_TABLE, f.slice, xd)
    @test (@allocated _cheb_table_eval(KAPPA_TABLE, f.slice, xd)) == 0

    fkd = Kappa(ForwardDiff.Dual{:KP17}(3.5, 1.0))
    _cheb_table_eval(KAPPA_TABLE, fkd.slice, 1.3)
    @test (@allocated _cheb_table_eval(KAPPA_TABLE, fkd.slice, 1.3)) == 0

    f(0.7)
    @test (@allocated f(0.7)) == 0
    dfdx(f, 0.7)
    @test (@allocated dfdx(f, 0.7)) == 0

    xd2 = ForwardDiff.Dual{:KP17}(0.7, 1.0)
    f(xd2)
    @test (@allocated f(xd2)) == 0
    dfdx(f, xd2)
    @test (@allocated dfdx(f, xd2)) == 0

    fkd(0.7)
    @test (@allocated fkd(0.7)) == 0
    dfdx(fkd, 0.7)
    @test (@allocated dfdx(fkd, 0.7)) == 0
end

@testset "KP18: _widen_shape copies" begin
    f = Kappa(3.5)
    D = ForwardDiff.Dual{:KP18,Float64,1}
    g = @inferred _widen_shape(f, D)
    @test g isa Kappa
    @test g.kappa === D(f.kappa)
    @test g.a === D(f.a)
    @test g.C === D(f.C)
    @test all(iszero, ForwardDiff.partials(g.kappa))
    @test all(iszero, ForwardDiff.partials(g.a))
    @test all(iszero, ForwardDiff.partials(g.C))
    @test g.slice == D.(f.slice)
end

@testset "KP19: promote first" begin
    f32 = Kappa(2.7f0)
    f64 = Kappa(Float64(2.7f0))
    @test typeof(f32) === typeof(f64)
    @test f32.kappa === f64.kappa
    @test f32.a === f64.a
    @test f32.C === f64.C
    @test f32.slice === f64.slice
end

@testset "KP20: parameters, reconstruct, show, and the constructor's throws" begin
    f = Kappa(3.5)
    @test parameters(f) == (kappa = 3.5,)
    g = reconstruct(f, (kappa = 6.0,))
    @test g.slice === _cheb_slice(KAPPA_TABLE, inv(6.0))
    @test sprint(show, f) == "Kappa(kappa = 3.5)"
    @test_throws "kappa must be finite and greater than 3/2" Kappa(1.5)
    @test_throws "kappa must be finite and greater than 3/2" Kappa(1.0)
    @test_throws "kappa must be finite and greater than 3/2" Kappa(Inf)
    @test_throws "kappa must be finite and greater than 3/2" Kappa(NaN)
end

# ------------------------------------------------------------------------------------------
# Registration (Build step 5): KP21, SC5 (in test_shape_cost.jl)
# ------------------------------------------------------------------------------------------

function _kp21_write_streak(path)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = Kappa(4.0))],
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

function _kp21_deck(datafile, outdir)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    kappa_e = { bounds = [2.0, 20.0], init = 4.0 }
    T_D = { bounds = [50.0, 800.0], init = 200.0 }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0
    shape = { type = "Kappa", kappa = "kappa_e" }

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

@testset "KP21: registration" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _kp21_write_streak(datafile)
        outdir = joinpath(dir, "out")

        # (a) the deck parses, binds kappa_e to electrons.e.f.kappa, and the effective deck
        # round-trips through read_deck with kappa present.
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _kp21_deck(datafile, outdir))
        deck = read_deck(deckpath)
        @test deck.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "Kappa"

        problem, optimizer, start = OTIS.build_problem(deck)
        b = only(b for b in problem.pspec.bindings if b.name == "kappa_e")
        @test b.kind === :leaf
        @test string(b.path) == "electrons.e.f.kappa"

        eff = effective_deck(deck, problem, optimizer)
        @test occursin("kappa", eff)
        efftmp = joinpath(dir, "effective.toml")
        write(efftmp, eff)
        deck2 = read_deck(efftmp)
        @test deck2.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "Kappa"

        # (b) an unknown shape key is refused
        bad = replace(_kp21_deck(datafile, outdir), "kappa = \"kappa_e\"" => "m = 3.0")
        badpath = joinpath(dir, "bad.toml")
        write(badpath, bad)
        @test_throws "unknown key \"m\"" read_deck(badpath)

        # (c) a PlasmaTrace with a Kappa electron round-trips through write_trace/read_trace
        template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = Kappa(3.5))],
            ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0)])
        trace = PlasmaTrace(template; t_ns = [0.0, 1.0])
        tracepath = joinpath(dir, "trace.jld2")
        write_trace(tracepath, trace)
        trace2 = read_trace(tracepath; as = :trace)
        p1 = parameters(trace.template.electrons["e"].f)
        p2 = parameters(trace2.template.electrons["e"].f)
        @test p1.kappa === p2.kappa
    end
end

# ------------------------------------------------------------------------------------------
# The fits and the regression (Build step 6): KP22, KP23, KP24
# ------------------------------------------------------------------------------------------

# D20's fixture, verbatim.
function _kp_p1_problem(kappa_truth::Real)
    template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = Kappa(4.0))],
        ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 200.0, u_kms = 0.0)])
    trace = PlasmaTrace(template; t_ns = [0.0])
    expt = Experiment(trace = trace, probe = ProbeBeam(lambda_probe_nm = 263.25),
        collection = Collection(theta_deg = 60.0, k_model = :exact),
        lambda_m = collect(range(262.0, 264.5; length = 24)) .* 1e-9,
        effects = (Sensitivity(scale = 1.0e-6), ConstantBackground(level = 25.0)))
    vk = FittedVariable("kappa_e"; bounds = (2.0, 20.0), init = 4.0)
    vT = FittedVariable("T_D"; bounds = (50.0, 800.0), init = 200.0)
    pspec = ParameterSpec(expt; variables = [vk, vT],
        bindings = ["kappa_e" => "electrons.e.f.kappa", "T_D" => "ions.D.T_eV"])
    working = Experiment(; trace = pspec.trace, probe = expt.probe, collection = expt.collection,
        lambda_m = expt.lambda, effects = expt.effects, prefactor = expt.prefactor)
    theta = flatten(pspec, ["kappa_e" => kappa_truth, "T_D" => 250.0])
    image = measure(unflatten(working, pspec, theta))
    data = StreakData(lambda_m = working.lambda, t_s = working.t, image = image,
        sigma = ones(size(image)))
    return FitProblem(working, pspec, data), pspec
end

@testset "KP22: the P1-sized recovery" begin
    for (kappa_truth, kappa_start) in ((3.0, 5.0), (6.5, 3.0))
        problem, pspec = _kp_p1_problem(kappa_truth)
        start = flatten(pspec, ["kappa_e" => kappa_start, "T_D" => 200.0])
        result = solve(problem, LevenbergMarquardt(); start = start)
        kappa_hat = result.values["kappa_e"][1]
        TD_hat = result.values["T_D"][1]
        errK = abs(kappa_hat - kappa_truth) / kappa_truth
        errT = abs(TD_hat - 250.0) / 250.0
        @info "KP22" kappa_truth kappa_start errK errT result.converged result.iterations
        @test errK < 1e-6
        @test errT < 1e-6
        @test result.converged
    end
end

@testset "KP23: the IAW feature's change with kappa, pinned" begin
    plasma_M = _kp_iaw_plasma(Maxwellian())
    (k, _, _) = wavevector(_KP_PROBE, _KP_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k * cs)

    sM = [spectral_density(plasma_M, k, om) for om in omegas]
    idx = argmax(sM)
    @info "KP23 argmax index and omega0/(k*cs)" idx (omegas[idx] / (k * cs))
    @test idx == 309

    omega0 = omegas[idx]
    s_M0 = spectral_density(plasma_M, k, omega0)
    kappas = (2, 3, 5, 10, 100)
    expected = (1.4029662293469545, 1.2899189845634842, 1.167863406617755,
                1.0796969675875066, 1.0075397389958305)
    ratios = Float64[]
    for kappa in kappas
        plasma_k = _kp_iaw_plasma(Kappa(Float64(kappa)))
        s_k = spectral_density(plasma_k, k, omega0)
        push!(ratios, s_k / s_M0)
    end
    @info "KP23 ratios as computed" ratios expected
    for (r, e) in zip(ratios, expected)
        @test isapprox(r, e; rtol = 1e-7)
    end
    @test all(ratios[i] > ratios[i+1] for i in 1:length(ratios)-1)
end

@testset "KP24: every fixture stays inside the table" begin
    # KP16's fixture
    worst16 = 0.0
    for kappa in (2.2, 3.5, 6.0, 15.0, 60.0)
        plasma = _kp_iaw_plasma(Kappa(kappa))
        (k, om, _) = wavevector(_KP_PROBE, _KP_COLLECTION, _KP_LAMBDA_S, plasma)
        sp_e = plasma.electrons["e"]
        worst16 = max(worst16, abs(xi(sp_e, k, om)))
    end
    @test worst16 <= KAPPA_TABLE.xi_max
    @info "KP24 KP16 fixture worst |xi_e|" worst16

    # KP23's fixture
    plasma_M = _kp_iaw_plasma(Maxwellian())
    (k23, _, _) = wavevector(_KP_PROBE, _KP_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k23 * cs)
    idx = argmax([spectral_density(plasma_M, k23, om) for om in omegas])
    omega0 = omegas[idx]
    worst23 = 0.0
    for kappa in (2, 3, 5, 10, 100)
        plasma_k = _kp_iaw_plasma(Kappa(Float64(kappa)))
        sp_e = plasma_k.electrons["e"]
        worst23 = max(worst23, abs(xi(sp_e, k23, omega0)))
    end
    @test worst23 <= KAPPA_TABLE.xi_max
    @info "KP24 KP23 fixture worst |xi_e|" worst23

    # KP22's fixture, both truths, all 24 wavelengths
    worst22 = 0.0
    for kappa_truth in (3.0, 6.5)
        problem, pspec = _kp_p1_problem(kappa_truth)
        theta = flatten(pspec, ["kappa_e" => kappa_truth, "T_D" => 250.0])
        expt_truth = unflatten(problem.experiment, pspec, theta)
        plasma_truth = plasma(expt_truth.trace, 1)
        sp_e = plasma_truth.electrons["e"]
        for lam in expt_truth.lambda
            (k22, om22, _) = wavevector(expt_truth.probe, expt_truth.collection, lam, plasma_truth)
            worst22 = max(worst22, abs(xi(sp_e, k22, om22)))
        end
    end
    @test worst22 <= KAPPA_TABLE.xi_max
    @info "KP24 KP22 fixture worst |xi_e|" worst22
end

# ------------------------------------------------------------------------------------------
# KP25 (Phase 4.C spec D3): a dual exactly at the table's edges stays in the table
# ------------------------------------------------------------------------------------------

@testset "KP25: a dual exactly at the table's edges stays in the table (Phase 4.C spec D3)" begin
    for xi in (1.0, 9.5), s in (1.0, -1.0)
        lhs = s * ForwardDiff.derivative(t -> real(cauchy_dfdv(Kappa(s * t), xi)), s * 2.0)
        rhs = s * ForwardDiff.derivative(t -> _cheb_table_eval(KAPPA_TABLE, _cheb_slice(KAPPA_TABLE, inv(s * t)), xi), s * 2.0)
        @test lhs === rhs
    end
    f = Kappa(3.0)
    for xi0 in (20.0, -20.0), s in (1.0, -1.0)
        lhs = s * ForwardDiff.derivative(x -> real(cauchy_dfdv(f, s * x)), s * xi0)
        rhs = s * ForwardDiff.derivative(x -> _cheb_table_eval(KAPPA_TABLE, f.slice, abs(s * x)), s * xi0)
        @test lhs === rhs
    end
end

# ------------------------------------------------------------------------------------------
# KP26 (Phase 4.D spec D15): the constructor's guard compares the primal
# ------------------------------------------------------------------------------------------

@testset "KP26: the constructor's guard compares the primal (Phase 4.D spec D15)" begin
    for s in (1.0, -1.0)
        @test_throws "kappa must be finite and greater than 3/2" Kappa(ForwardDiff.Dual(1.5, s))
    end
end
