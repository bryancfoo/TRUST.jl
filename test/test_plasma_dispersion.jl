using Test
using OTIS
using QuadGK: quadgk
using SpecialFunctions: dawson
using ForwardDiff: derivative
using ChainRulesCore: frule, rrule, NoTangent

# The oracle is quadrature, never erfcx (OQ-23): these reference functions share no machinery
# with src/plasma_dispersion.jl.

# Im ξ > 0: BigFloat quadrature of the definition. atol = 1e-60 alongside rtol (deviation,
# approved 2026-09-07): at some grid points the integral is legitimately near zero, where a
# pure relative tolerance is nearly unsatisfiable and forces runaway adaptive subdivision;
# 1e-60 is far below the 1e-12 rtol the T1 test itself checks against, so it costs no accuracy.
Z_ref_upper(xi) = quadgk(x -> exp(-x^2) / (x - xi), BigFloat(-Inf), BigFloat(0), BigFloat(Inf);
                         rtol = 1e-30, atol = 1e-60)[1] / sqrt(BigFloat(pi))

# Im ξ < 0: the reflection Z(ξ) + Z(-ξ) = 2i√π e^{-ξ²}, with the upper-half-plane quadrature.
Z_ref_lower(xi) = 2im * sqrt(BigFloat(pi)) * exp(-xi^2) - Z_ref_upper(-xi)

# Im ξ = 0: Plemelj, principal value by singularity subtraction, in BigFloat.
# Z(x) = (1/√π) [ PV + i·π·e^{-x²} ]  -- the i·π residue, before the 1/√π prefactor (Pitfall 7).
# At x = 0 the integrand is exactly odd, so the PV integral is analytically zero; atol = 1e-60
# (deviation, approved 2026-09-07) lets quadgk stop there instead of chasing an unsatisfiable
# relative tolerance against a near-zero estimate (measured: ~118s -> ~11s for that one point).
function Z_ref_real(x::Real)
    xb = BigFloat(x)
    L = BigFloat(30)
    g(u) = iszero(u) ? -2xb * exp(-xb^2) : (exp(-(xb + u)^2) - exp(-xb^2)) / u
    pv = quadgk(g, -L, BigFloat(0), L; rtol = 1e-30, atol = 1e-60)[1]
    return (pv + im * BigFloat(pi) * exp(-xb^2)) / sqrt(BigFloat(pi))
end

function Z_ref(xi::Complex)
    if imag(xi) > 0
        Z_ref_upper(xi)
    elseif imag(xi) < 0
        Z_ref_lower(xi)
    else
        Z_ref_real(real(xi))
    end
end

# the reference grid: used ONLY by T1, where every point costs a BigFloat quadrature
const ref_re_grid = (-4.0, -1.0, 0.0, 1.0, 4.0)
const ref_im_grid = (0.0, 0.5, -0.5, 2.0, -2.0)

# the full grid: used by the cheap exact-identity tests (T7), which cost nothing
const re_grid = (-4.0, -2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0, 4.0)
const im_grid = (0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0)

@testset "T1: Z vs BigFloat quadrature, all quadrants and the real axis" begin
    worst = Dict(:upper => 0.0, :lower => 0.0, :real => 0.0)
    for re in ref_re_grid, b in ref_im_grid
        xi = complex(re, b)
        zref = Z_ref(xi)
        z = Z(xi)
        relerr = Float64(abs(z - zref) / abs(zref))
        quadrant = b > 0 ? :upper : (b < 0 ? :lower : :real)
        worst[quadrant] = max(worst[quadrant], relerr)
        @test isapprox(z, zref; rtol = 1e-12)
    end
    @info "T1 worst relative error of Z vs BigFloat quadrature, per quadrant" worst[:upper] worst[:lower] worst[:real]
end

@testset "T2: Rw vs 1 - 2x dawson(x) (Sheffield 5.2.7)" begin
    for x in 0.0:0.1:6.0
        Rw = -real(Zprime(x)) / 2
        Rw_ref = 1 - 2x * dawson(x)
        @test isapprox(Rw, Rw_ref; rtol = 1e-12)
    end
end

@testset "T3: Iw vs sqrt(pi) x exp(-x^2) (Sheffield 5.2.8)" begin
    for x in 0.0:0.1:6.0
        Iw = -imag(Zprime(x)) / 2
        Iw_ref = sqrt(pi) * x * exp(-x^2)
        @test isapprox(Iw, Iw_ref; rtol = 1e-12, atol = 1e-300)
    end
end

@testset "T4: Landau sign (Iw(x) > 0, Im Z(x) > 0 for x > 0)" begin
    for x in 0.1:0.1:5.0
        @test (-imag(Zprime(x)) / 2) > 0
        @test imag(Z(x)) > 0
    end
end

@testset "T5: small-x expansion (Sheffield 5.2.9)" begin
    for x in (0.05, 0.1, 0.2, 0.3)
        Rw = -real(Zprime(x)) / 2
        S3 = 1 - 2x^2 * (1 - (2 / 3) * x^2 + (4 / 15) * x^4)
        @test abs(Rw - S3) <= 0.25 * x^8
    end
end

@testset "T6: large-x expansion (Sheffield 5.2.10)" begin
    for x in (4.0, 5.0, 6.0, 8.0)
        Rw = -real(Zprime(x)) / 2
        S3 = -(1 / (2x^2)) * (1 + 3 / (2x^2) + 15 / (4x^4))
        @test abs(Rw - S3) <= 15 / x^8
    end
end

@testset "T7: Zprime(xi) == -2(1 + xi*Z(xi)), exact" begin
    for re in re_grid, b in im_grid
        xi = complex(re, b)
        @test Zprime(xi) == -2 * (1 + xi * Z(xi))
    end
end

@testset "T8: AD derivative(Z) vs Zprime" begin
    for x in (0.1, 0.5, 1.0, 2.0, 3.0)
        dre = derivative(t -> real(Z(t)), x)
        dim = derivative(t -> imag(Z(t)), x)
        zp = Zprime(x)
        @test isapprox(dre, real(zp); rtol = 1e-15)
        @test isapprox(dim, imag(zp); rtol = 1e-15)
    end
end

@testset "T9: AD derivative(Z) vs central differences" begin
    h = 1e-5
    for x in (0.1, 0.5, 1.0, 2.0, 3.0)
        dre = derivative(t -> real(Z(t)), x)
        dim = derivative(t -> imag(Z(t)), x)
        cd_re = (real(Z(x + h)) - real(Z(x - h))) / (2h)
        cd_im = (imag(Z(x + h)) - imag(Z(x - h))) / (2h)
        @test isapprox(dre, cd_re; rtol = 1e-8)
        @test isapprox(dim, cd_im; rtol = 1e-8)
    end
end

@testset "T10: AD derivative(Zprime) vs analytic Z'' and central differences" begin
    h = 1e-5
    for x in (0.1, 0.5, 1.0, 2.0, 3.0)
        dre = derivative(t -> real(Zprime(t)), x)
        dim = derivative(t -> imag(Zprime(t)), x)
        zpp = -2 * (Z(x) + x * Zprime(x))
        @test isapprox(dre, real(zpp); rtol = 1e-15)
        @test isapprox(dim, imag(zpp); rtol = 1e-15)

        cd_re = (real(Zprime(x + h)) - real(Zprime(x - h))) / (2h)
        cd_im = (imag(Zprime(x + h)) - imag(Zprime(x - h))) / (2h)
        @test isapprox(dre, cd_re; rtol = 1e-8)
        @test isapprox(dim, cd_im; rtol = 1e-8)
    end
end

@testset "T11: Complex{Dual} method, holomorphy" begin
    for t0 in (-1.0, 0.0, 1.0)
        # derivative along the real direction: equals Zprime(xi)
        dre1 = derivative(t -> real(Z(complex(t, 0.5))), t0)
        dim1 = derivative(t -> imag(Z(complex(t, 0.5))), t0)
        zp1 = Zprime(complex(t0, 0.5))
        @test isapprox(dre1, real(zp1); rtol = 1e-15)
        @test isapprox(dim1, imag(zp1); rtol = 1e-15)

        # derivative along the imaginary direction: equals i*Zprime(xi)
        dre2 = derivative(t -> real(Z(complex(0.5, t))), t0)
        dim2 = derivative(t -> imag(Z(complex(0.5, t))), t0)
        zp2 = im * Zprime(complex(0.5, t0))
        @test isapprox(dre2, real(zp2); rtol = 1e-15)
        @test isapprox(dim2, imag(zp2); rtol = 1e-15)
    end
end

@testset "T12: ChainRulesCore rules (D8)" begin
    for xi in (0.5, 1.0 + 0.3im)
        dxi = 0.3 - 0.7im
        Ω̄ = 0.2 + 1.1im

        y, dy = frule((NoTangent(), dxi), Z, xi)
        @test isapprox(y, Z(xi); rtol = 1e-15)
        @test isapprox(dy, Zprime(xi) * dxi; rtol = 1e-15)

        y2, pb = rrule(Z, xi)
        @test isapprox(y2, Z(xi); rtol = 1e-15)
        _, xbar = pb(Ω̄)
        @test isapprox(xbar, conj(Zprime(xi)) * Ω̄; rtol = 1e-15)

        y3, dy3 = frule((NoTangent(), dxi), Zprime, xi)
        zpp = -2 * (Z(xi) + xi * Zprime(xi))
        @test isapprox(y3, Zprime(xi); rtol = 1e-15)
        @test isapprox(dy3, zpp * dxi; rtol = 1e-15)

        y4, pb2 = rrule(Zprime, xi)
        @test isapprox(y4, Zprime(xi); rtol = 1e-15)
        _, xbar2 = pb2(Ω̄)
        @test isapprox(xbar2, conj(zpp) * Ω̄; rtol = 1e-15)
    end
end

@testset "T13: allocation-free Z and Zprime" begin
    # @allocated at global scope lies (Pitfall 15): wrap in a function and warm it up first.
    _bench_Z(xi) = Z(xi)
    _bench_Zprime(xi) = Zprime(xi)
    xi = 0.7 + 0.3im
    _bench_Z(xi)
    _bench_Zprime(xi)
    @test (@allocated _bench_Z(xi)) == 0
    @test (@allocated _bench_Zprime(xi)) == 0
end
