using Test
using TRUST
import ForwardDiff
using Random

_bg_plasma() = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))
_bg_probe() = ProbeBeam(lambda_probe_nm=263.25)
_bg_collection() = Collection(theta_deg=60, k_model=:exact)

@testset "BG1: ConstantBackground adds a constant" begin
    rng = Random.Xoshiro(20260913)
    P = rand(rng, 16, 4)
    e = ConstantBackground(level=25.0)
    R = apply!(copy(P), P, e, nothing)
    @test R == P .+ 25
end

@testset "BG2: PolynomialBackground's reference and scale" begin
    lambda = collect(range(235e-9, 255e-9; length=17))
    e1 = PolynomialBackground(coefficients=(1.0,))
    B1 = TRUST._prepare(e1, lambda, [0.0])
    @test isapprox(B1[1], 1.0; atol=1e-15)
    @test isapprox(B1[end], 1.0; atol=1e-15)

    econst = ConstantBackground(level=5.0)
    epoly = PolynomialBackground(coefficients=(5.0,))
    Bpoly = TRUST._prepare(epoly, lambda, [0.0])
    P = zeros(length(lambda), 1)
    r1 = apply!(copy(P), P, econst, nothing)
    r2 = apply!(copy(P), P, epoly, Bpoly)
    @test r1 == r2

    elin = PolynomialBackground(coefficients=(0.0, 1.0))
    Blin = TRUST._prepare(elin, lambda, [0.0])
    lam0 = (lambda[1] + lambda[end]) / 2
    dlam = (lambda[end] - lambda[1]) / 2
    s = (lambda .- lam0) ./ dlam
    @test isapprox(Blin, s; atol=1e-14)
    @test isapprox(s[1], -1.0; atol=1e-14)
    @test isapprox(s[(end+1)÷2], 0.0; atol=1e-14)
    @test isapprox(s[end], 1.0; atol=1e-14)
end

@testset "BG3: ImageBackground" begin
    img = reshape(collect(1.0:64.0), 16, 4)
    e = ImageBackground(image=img, scale=2.0)
    P = zeros(16, 4)
    R = apply!(copy(P), P, e, nothing)
    @test isapprox(R, 2.0 .* img; atol=1e-14)

    bad_img = zeros(8, 4)
    e_bad = ImageBackground(image=bad_img)
    @test_throws "background image" apply!(copy(P), P, e_bad, nothing)
end

@testset "BG4: the affine identity" begin
    # effect(aP+bQ) - a*effect(P) - b*effect(Q) == (1-a-b)*B, B the effect's own additive term
    # -- not spatially uniform for PolynomialBackground (varies by wavelength) or
    # ImageBackground (varies by pixel), so the check compares against B directly rather than
    # asserting the difference is constant.
    rng = Random.Xoshiro(20260914)
    P = rand(rng, 16, 4) .+ 0.1
    Q = rand(rng, 16, 4) .+ 0.1
    a, b = 0.37, -1.9
    lambda = collect(range(235e-9, 255e-9; length=16))

    e1 = ConstantBackground(level=25.0)
    B1 = fill(e1.level, size(P))
    lhs1 = apply!(copy(a .* P .+ b .* Q), a .* P .+ b .* Q, e1, nothing)
    rhs1 = a .* apply!(copy(P), P, e1, nothing) .+ b .* apply!(copy(Q), Q, e1, nothing)
    @test isapprox(lhs1 .- rhs1, (1 - a - b) .* B1; atol=1e-13)

    e2 = PolynomialBackground(coefficients=(1.0, 0.5))
    op2 = TRUST._prepare(e2, lambda, [0.0])
    B2 = repeat(op2, 1, size(P, 2))
    lhs2 = apply!(copy(a .* P .+ b .* Q), a .* P .+ b .* Q, e2, op2)
    rhs2 = a .* apply!(copy(P), P, e2, op2) .+ b .* apply!(copy(Q), Q, e2, op2)
    @test isapprox(lhs2 .- rhs2, (1 - a - b) .* B2; atol=1e-13)

    e3 = ImageBackground(image=rand(rng, 16, 4))
    B3 = e3.scale .* e3.image
    lhs3 = apply!(copy(a .* P .+ b .* Q), a .* P .+ b .* Q, e3, nothing)
    rhs3 = a .* apply!(copy(P), P, e3, nothing) .+ b .* apply!(copy(Q), Q, e3, nothing)
    @test isapprox(lhs3 .- rhs3, (1 - a - b) .* B3; atol=1e-13)
end

@testset "BG5: parameters / reconstruct" begin
    e1 = ConstantBackground(level=25.0)
    @test reconstruct(e1, parameters(e1)).level == e1.level

    e2 = PolynomialBackground(coefficients=(1.0, 2.0, 3.0))
    @test reconstruct(e2, parameters(e2)).coefficients == e2.coefficients

    e3 = ImageBackground(image=rand(4, 4), scale=1.5)
    r3 = reconstruct(e3, parameters(e3))
    @test r3.scale == e3.scale
    @test r3.image === e3.image
end

@testset "BG6: a dual parameter reaches the streak" begin
    template = _bg_plasma()
    probe = _bg_probe(); collection = _bg_collection()
    lambda = collect(range(235e-9, 255e-9; length=16))
    trace = PlasmaTrace(template; t_ns=[0.0, 1.0])

    level_dual = ForwardDiff.Dual(25.0, 1.0)
    expt1 = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda,
        effects=(ConstantBackground(level=level_dual),))
    M1 = measure(expt1)
    @test eltype(M1) <: ForwardDiff.Dual

    scale_dual = ForwardDiff.Dual(1.0, 1.0)
    expt2 = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda,
        effects=(ImageBackground(image=zeros(16, 2), scale=scale_dual),))
    M2 = measure(expt2)
    @test eltype(M2) <: ForwardDiff.Dual

    coef_dual = (ForwardDiff.Dual(1.0, 1.0), 2.0)
    expt3 = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda,
        effects=(PolynomialBackground(coefficients=coef_dual),))
    M3 = measure(expt3)
    @test eltype(M3) <: ForwardDiff.Dual
end

# --- Step 2.3b: RationalBackground (spec D46) ---

@testset "RB1: rat11 and rat21 against hand-written formulas" begin
    lambda = collect(range(235e-9, 255e-9; length=128))
    lam0 = (first(lambda) + last(lambda)) / 2
    dlam = (last(lambda) - first(lambda)) / 2
    s = (lambda .- lam0) ./ dlam

    rat11 = RationalBackground(numerator=(2.0, 0.5), denominator=(0.4,))
    op11 = TRUST._prepare(rat11, lambda, [0.0])
    oracle11 = (2.0 .+ 0.5 .* s) ./ (1 .+ 0.4 .* s)
    @test isapprox(op11, oracle11; rtol=1e-14)

    rat21 = RationalBackground(numerator=(2.0, 0.5, -0.3), denominator=(0.4,))
    op21 = TRUST._prepare(rat21, lambda, [0.0])
    oracle21 = (2.0 .+ 0.5 .* s .- 0.3 .* s .^ 2) ./ (1 .+ 0.4 .* s)
    @test isapprox(op21, oracle21; rtol=1e-14)
end

@testset "RB2: Q = 0 is PolynomialBackground, bitwise" begin
    lambda = collect(range(235e-9, 255e-9; length=64))
    rat = RationalBackground(numerator=(1.0, 2.0, 3.0), denominator=())
    poly = PolynomialBackground(coefficients=(1.0, 2.0, 3.0))
    op_rat = TRUST._prepare(rat, lambda, [0.0])
    op_poly = TRUST._prepare(poly, lambda, [0.0])
    @test all(op_rat .=== op_poly)
end

@testset "RB3: the scaled variable is D14's" begin
    # atol = 1e-14, matching BG2's own precedent for this identical s[1]/s[end] boundary check:
    # (lambda[1]-lam0)/dlam is not exactly -1.0 in Float64 at these wavelength magnitudes.
    lambda = collect(range(235e-9, 255e-9; length=129))   # odd length: exact middle bin
    rat = RationalBackground(numerator=(0.0, 1.0), denominator=())   # B = s
    op = TRUST._prepare(rat, lambda, [0.0])
    @test isapprox(op[1], -1.0; atol=1e-14)
    @test isapprox(op[(end+1)÷2], 0.0; atol=1e-14)
    @test isapprox(op[end], 1.0; atol=1e-14)
end

@testset "RB4: the pole guard" begin
    lambda = collect(range(235e-9, 255e-9; length=128))
    rat_pole = RationalBackground(numerator=(1.0,), denominator=(1.2,))   # 1+1.2s = 0 at s=-1/1.2, inside [-1,1]
    @test_throws "pole" TRUST._prepare(rat_pole, lambda, [0.0])

    rat_ok = RationalBackground(numerator=(1.0,), denominator=(0.5,))   # min |1+0.5s| = 0.5 at s=-1
    op = TRUST._prepare(rat_ok, lambda, [0.0])
    @test all(isfinite, op)
end

@testset "RB5: the MATLAB conversion" begin
    lambda = collect(range(235e-9, 255e-9; length=64))
    p1, p2, q1 = 0.7, 2.0, 2.5
    lam0 = (first(lambda) + last(lambda)) / 2
    dlam = (last(lambda) - first(lambda)) / 2
    s = (lambda .- lam0) ./ dlam
    matlab_form = (p1 .* s .+ p2) ./ (s .+ q1)

    rat = RationalBackground(numerator=(p2 / q1, p1 / q1), denominator=(1 / q1,))
    op = TRUST._prepare(rat, lambda, [0.0])
    @test isapprox(op, matlab_form; rtol=1e-14)
end

@testset "RB6: the affine identity, parameters/reconstruct, and a dual coefficient" begin
    rng = Random.Xoshiro(20260915)
    lambda = collect(range(235e-9, 255e-9; length=32))
    P = rand(rng, 32, 4) .+ 0.1
    Q = rand(rng, 32, 4) .+ 0.1
    a, b = 0.37, -1.9

    e = RationalBackground(numerator=(1.0, 0.5), denominator=(0.3,))
    op = TRUST._prepare(e, lambda, [0.0])
    B = repeat(op, 1, size(P, 2))
    lhs = apply!(copy(a .* P .+ b .* Q), a .* P .+ b .* Q, e, op)
    rhs = a .* apply!(copy(P), P, e, op) .+ b .* apply!(copy(Q), Q, e, op)
    @test isapprox(lhs .- rhs, (1 - a - b) .* B; atol=1e-13)

    r = reconstruct(e, parameters(e))
    @test r.numerator == e.numerator
    @test r.denominator == e.denominator

    template = _bg_plasma()
    probe = _bg_probe(); collection = _bg_collection()
    trace = PlasmaTrace(template; t_ns=[0.0, 1.0])
    num_dual = (ForwardDiff.Dual(1.0, 1.0), 0.5)
    e_dual = RationalBackground(numerator=num_dual, denominator=(0.3,))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda,
        effects=(e_dual,))
    M = measure(expt)
    @test eltype(M) <: ForwardDiff.Dual
end
