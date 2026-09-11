using Test
using OTIS
using OTIS: _interpolate_response
import ForwardDiff
using Random

@testset "SR1: a flat response equals the scalar" begin
    lambda = collect(range(235e-9, 255e-9; length=64))
    e_flat = Sensitivity(scale=2.5, lambda_m=collect(range(230e-9, 260e-9; length=8)),
        response=ones(8))
    e_scalar = Sensitivity(scale=2.5)
    op = OTIS._prepare(e_flat, lambda, [0.0])
    rng = Random.Xoshiro(20260911)
    P = rand(rng, 64, 4) .+ 0.1
    R_flat = apply!(similar(P), P, e_flat, op)
    R_scalar = apply!(similar(P), P, e_scalar, nothing)
    @test all(R_flat .=== R_scalar)
end

@testset "SR2: interpolation against a hand-written oracle" begin
    lam_cal = collect(range(200e-9, 300e-9; length=16))
    resp_cal = collect(range(1.0, 16.0; length=16))   # a linear ramp
    lambda = collect(range(210e-9, 290e-9; length=128))
    got = _interpolate_response(lam_cal, resp_cal, lambda)
    # The ramp is linear, so the hand oracle is the same linear function evaluated directly.
    slope = (resp_cal[end] - resp_cal[1]) / (lam_cal[end] - lam_cal[1])
    oracle = resp_cal[1] .+ slope .* (lambda .- lam_cal[1])
    @test isapprox(got, oracle; rtol=1e-14)
end

@testset "SR3: normalization" begin
    lam_cal = collect(range(200e-9, 300e-9; length=8))
    resp_cal = [1.0, 2.0, 3.7, 2.5, 1.8, 0.9, 0.5, 0.2]
    e = Sensitivity(scale=1.0, lambda_m=lam_cal, response=resp_cal)
    @test e.response_norm == 3.7
    @test maximum(e.response) == 1.0
    @test isapprox(e.response .* e.response_norm, resp_cal; rtol=1e-15)
end

@testset "SR4: a curve that does not span the window throws" begin
    lam_cal = collect(range(240e-9, 250e-9; length=8))
    resp_cal = ones(8)
    e = Sensitivity(scale=1.0, lambda_m=lam_cal, response=resp_cal)

    lambda_short_blue = collect(range(235e-9, 245e-9; length=16))   # starts before lam_cal
    @test_throws "not spanned" OTIS._prepare(e, lambda_short_blue, [0.0])

    lambda_short_red = collect(range(245e-9, 255e-9; length=16))    # ends after lam_cal
    @test_throws "not spanned" OTIS._prepare(e, lambda_short_red, [0.0])
end

@testset "SR5: islinear still holds, with a response" begin
    lam_cal = collect(range(230e-9, 260e-9; length=8))
    resp_cal = [0.2, 0.5, 0.9, 1.0, 0.95, 0.7, 0.4, 0.15]
    lambda = collect(range(235e-9, 255e-9; length=64))
    e = Sensitivity(scale=3.0, lambda_m=lam_cal, response=resp_cal)
    op = OTIS._prepare(e, lambda, [0.0])
    @test islinear(e)

    rng = Random.Xoshiro(20260912)
    P = rand(rng, 64, 4) .+ 0.1
    Q = rand(rng, 64, 4) .+ 0.1
    a, b = 0.37, -1.9
    lhs = apply!(similar(P), a .* P .+ b .* Q, e, op)
    rhs = a .* apply!(similar(P), P, e, op) .+ b .* apply!(similar(P), Q, e, op)
    @test isapprox(lhs, rhs; rtol=1e-14)
end

@testset "SR6: AD in scale with a response present" begin
    lam_cal = collect(range(230e-9, 260e-9; length=8))
    resp_cal = [0.2, 0.5, 0.9, 1.0, 0.95, 0.7, 0.4, 0.15]
    lambda = collect(range(235e-9, 255e-9; length=16))
    rng = Random.Xoshiro(20260913)
    P = rand(rng, 16, 2) .+ 0.1

    function f(x)
        e = Sensitivity(scale=x, lambda_m=lam_cal, response=resp_cal)
        op = OTIS._prepare(e, lambda, [0.0])
        return sum(apply!(similar(P, typeof(x)), P, e, op))
    end
    x0 = 2.0
    h = 1e-6
    ad = ForwardDiff.derivative(f, x0)
    fd = (f(x0 * (1 + h)) - f(x0 * (1 - h))) / (2h * x0)
    floor_ = 1e-6 * abs(fd)
    @test isapprox(ad, fd; rtol=1e-6, atol=floor_)
end
