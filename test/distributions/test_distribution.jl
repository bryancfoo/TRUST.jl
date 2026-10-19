using Test
using TRUST
using ForwardDiff: derivative

include(joinpath(@__DIR__, "..", "shape_contract.jl"))

# A shape defined only by its call, with no dfdx or cauchy_dfdv override, so that everything
# else in the interface is exercised through the generic fallbacks (spec G3, OQ-3 item 1).
struct PlainMaxwellian <: TRUST.AbstractVelocityDistribution end
(::PlainMaxwellian)(x) = exp(-x^2) / sqrt(pi)

@testset "G1: AD through the generic Cauchy fallback vs analytic Z''" begin
    f = PlainMaxwellian()
    for xi0 in (0.3, 0.9241388, 1.0, 2.0, 3.0)
        dre = derivative(t -> real(cauchy_dfdv(f, t)), xi0)
        dim = derivative(t -> imag(cauchy_dfdv(f, t)), xi0)
        zpp = -2 * (Z(xi0) + xi0 * Zprime(xi0))
        @test isapprox(dre, real(zpp); rtol = 1e-8)
        @test isapprox(dim, imag(zpp); rtol = 1e-8)
    end
end

@testset "G2: AD through the generic Cauchy fallback vs central differences" begin
    f = PlainMaxwellian()
    h = 1e-5
    for xi0 in (0.3, 0.9241388, 1.0, 2.0, 3.0)
        dre = derivative(t -> real(cauchy_dfdv(f, t)), xi0)
        dim = derivative(t -> imag(cauchy_dfdv(f, t)), xi0)
        cd_re = (real(cauchy_dfdv(f, xi0 + h)) - real(cauchy_dfdv(f, xi0 - h))) / (2h)
        cd_im = (imag(cauchy_dfdv(f, xi0 + h)) - imag(cauchy_dfdv(f, xi0 - h))) / (2h)
        @test isapprox(dre, cd_re; rtol = 1e-7)
        @test isapprox(dim, cd_im; rtol = 1e-7)
    end
end

@testset "G3: a shape defined only by its call is usable" begin
    f = PlainMaxwellian()
    check_shape_normalizations(f)
    @test isapprox(moment(f, 0), 1; atol = 1e-10)
    @test isapprox(moment(f, 1), 0; atol = 1e-10)
    @test isapprox(moment(f, 2), 0.5; atol = 1e-10)
    @test isapprox(dfdx(f, 0.5), -2 * 0.5 * exp(-0.5^2) / sqrt(pi); rtol = 1e-10)
    @test isapprox(cauchy_dfdv(f, 0.5), Zprime(0.5); atol = 1e-10)
end

@testset "G4: dfdx throws for a shape without a complex method" begin
    f = PlainMaxwellian()
    @test_throws "is not available: the generic derivative uses ForwardDiff" dfdx(f, 0.5 + 0.3im)
end

@testset "G5: cauchy_dfdv throws in the lower half-plane without a complex dfdx" begin
    f = PlainMaxwellian()
    @test_throws "is not available: the generic derivative uses ForwardDiff" cauchy_dfdv(f, 0.5 - 0.3im)
end
