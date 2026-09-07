using Test
using QuadGK: quadgk

"""
    check_shape_normalizations(f; atol = 1e-10, rtol = 1e-12, limits = (-Inf, 0.0, Inf))

Shared contract test for every `AbstractVelocityDistribution`: `∫φ = 1`, `∫xφ = 0`,
`∫x²φ = 1/2` (`conventions.md` 1.5), each checked by independent quadrature. Included, not
run as its own `@safetestset`, by every shape's own test file (spec D14).
"""
function check_shape_normalizations(f; atol = 1e-10, rtol = 1e-12, limits = (-Inf, 0.0, Inf))
    @testset "$(typeof(f))" begin
        @test isapprox(quadgk(x -> f(x), limits...; rtol = rtol, atol = 1e-14)[1], 1;
                        atol = atol)
        @test isapprox(quadgk(x -> x * f(x), limits...; rtol = rtol, atol = 1e-14)[1], 0;
                        atol = atol)
        @test isapprox(quadgk(x -> x^2 * f(x), limits...; rtol = rtol, atol = 1e-14)[1], 0.5;
                        atol = atol)
    end
end
