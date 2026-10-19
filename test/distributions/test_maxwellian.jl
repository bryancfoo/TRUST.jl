using Test
using TRUST
using TRUST: _cauchy_real, _cauchy_upper, _cauchy_lower
using ForwardDiff: derivative

include(joinpath(@__DIR__, "..", "shape_contract.jl"))

const f = Maxwellian()

@testset "M1: normalizations" begin
    check_shape_normalizations(f)
end

@testset "M2: cauchy_dfdv(Maxwellian(), xi) === Zprime(xi), exact" begin
    for xi in (-2.0, -0.5, 0.0, 0.5, 2.0, 0.3 + 0.7im, -1.0 - 0.2im)
        @test cauchy_dfdv(f, xi) === Zprime(xi)
    end
end

@testset "M3: generic fallback vs Zprime, real xi" begin
    for x in (-4.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.7071, 0.9241388, 1.0, 1.5, 2.0, 3.0, 4.0)
        @test isapprox(_cauchy_real(f, x), Zprime(x); atol = 1e-10)
    end
end

@testset "M4: generic fallback vs Zprime, Im xi > 0" begin
    for re in (-3.0, -1.0, 0.0, 1.0, 3.0), b in (0.1, 0.3, 1.0, 2.0)
        xi = complex(re, b)
        @test isapprox(_cauchy_upper(f, xi), Zprime(xi); atol = 1e-10)
    end
end

@testset "M5: generic fallback vs Zprime, Im xi < 0" begin
    for re in (-3.0, -1.0, 0.0, 1.0, 3.0), b in (-0.1, -0.3, -1.0, -2.0)
        xi = complex(re, b)
        @test isapprox(_cauchy_lower(f, xi), Zprime(xi); atol = 1e-10)
    end
end

@testset "M6: moments vs analytic values" begin
    analytic = (1.0, 0.0, 0.5, 0.0, 0.75)
    for n in 0:4
        @test isapprox(moment(f, n), analytic[n + 1]; atol = 1e-10)
    end
end

@testset "M7: skewness and excess_kurtosis" begin
    @test isapprox(skewness(f), 0; atol = 1e-10)
    @test isapprox(excess_kurtosis(f), 0; atol = 1e-10)
end

@testset "M8: moment out of range throws" begin
    @test_throws "moment is defined for n = 0 to 4; got n = 5" moment(f, 5)
    @test_throws "moment is defined for n = 0 to 4; got n = -1" moment(f, -1)
end

@testset "M9: dfdx vs ForwardDiff derivative of the call" begin
    for x in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
        @test isapprox(dfdx(f, x), derivative(f, x); rtol = 1e-14)
    end
end

@testset "M10: parameters and reconstruct" begin
    @test parameters(f) === NamedTuple()
    @test reconstruct(f, NamedTuple()) === f
end

@testset "M11: complex argument support" begin
    x = 0.5 + 0.3im
    @test isfinite(f(x))
    @test isfinite(dfdx(f, x))
end
