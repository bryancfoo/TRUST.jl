using Test
using TRUST
using TRUST: _rand_poisson, _point_spread
using Random: Xoshiro
using Statistics: mean, var

@testset "NZ1: _rand_poisson is Poisson" begin
    N = 200_000
    for lambda in (0.5, 3.0, 18.0, 60.0, 300.0)
        rng = Xoshiro(20260905)
        draws = [_rand_poisson(rng, lambda) for _ in 1:N]
        m = mean(draws)
        v = var(draws)
        third = mean((draws .- lambda) .^ 3)
        mean_tol = 4 * sqrt(lambda / N)
        var_tol = 4 * lambda * sqrt(2 / (N - 1))
        @info "NZ1" lambda m v third mean_tol var_tol
        @test isapprox(m, lambda; atol=mean_tol)
        @test isapprox(v, lambda; atol=var_tol)
        @test isapprox(third, lambda; rtol=0.10)
    end
end

function _nz_streak(; nlambda=32, nt=8)
    template = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    trace = PlasmaTrace(template; t_ns=collect(range(2.0, 3.0; length=nt)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=nlambda))
    effects = (Sensitivity(scale=2.0e-4), ConstantBackground(level=25.0))
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
end

@testset "NZ2: the variance identity" begin
    expt = _nz_streak()
    model = StreakCameraNoise(sigma_px=0.0)
    Sbar = measure(expt)
    K = 400
    images = Array{Float64}(undef, size(Sbar)..., K)
    for k in 1:K
        rng = Xoshiro(20260904 + k)
        sd = synthesize(expt, model; rng=rng)
        images[:, :, k] = sd.image
    end
    sigma2 = sigma(model, Sbar) .^ 2
    ratios = Float64[]
    for j in axes(Sbar, 2), l in axes(Sbar, 1)
        s2 = var(images[l, j, :])
        r = s2 / sigma2[l, j]
        push!(ratios, r)
        @test isapprox(r, 1.0; rtol=0.25)
    end
    pooled = mean(ratios)
    @info "NZ2 pooled variance ratio" pooled
    @test isapprox(pooled, 1.0; rtol=0.03)
end

@testset "NZ3: the mean is unbiased" begin
    expt = _nz_streak()
    model = StreakCameraNoise(sigma_px=0.0)
    Sbar = measure(expt)
    sig = sigma(model, Sbar)
    K = 400
    acc = zeros(size(Sbar))
    for k in 1:K
        rng = Xoshiro(20260906 + k)
        sd = synthesize(expt, model; rng=rng)
        acc .+= (sd.image .- Sbar) ./ sig
    end
    acc ./= K
    pooled = mean(acc)
    @info "NZ3 pooled bias" pooled
    @test isapprox(pooled, 0.0; atol=0.02)
end

@testset "NZ4: sigma against its closed form" begin
    Sbar = [10.0 20.0; 30.0 40.0]
    # sigma_px = 0.0 since Phase 4.B spec D9: the closed form below is the pre-spread formula,
    # which `sigma` returns only without a spread; NZ12 is the spread's own closed-form row.
    model_cam = StreakCameraNoise(camera_gain=135.0, noise_factor=1.05, sigma_px=0.0, read_noise=11.0)
    oracle_cam = sqrt.(Sbar .* 135.0 .* 1.05 .+ 11.0^2)
    @test isapprox(sigma(model_cam, Sbar), oracle_cam; rtol=1e-14)

    model_gauss = GaussianNoise(a=4.0, b=0.5)
    oracle_gauss = sqrt.(4.0 .+ 0.5 .* Sbar)
    @test isapprox(sigma(model_gauss, Sbar), oracle_gauss; rtol=1e-14)
end

@testset "NZ5: the point spread conserves counts" begin
    rng = Xoshiro(20260907)
    img = rand(rng, 32, 16) .* 100 .+ 1
    for sigma_px in (1.0, 2.5)
        out = _point_spread(img, sigma_px)
        @test isapprox(sum(out), sum(img); rtol=1e-12)
    end
end

@testset "NZ6: GaussianNoise" begin
    expt = _nz_streak()
    model = GaussianNoise(a=100.0, b=4.0)
    Sbar = measure(expt)
    sig2 = sigma(model, Sbar) .^ 2
    K = 400
    images = Array{Float64}(undef, size(Sbar)..., K)
    for k in 1:K
        rng = Xoshiro(20260908 + k)
        sd = synthesize(expt, model; rng=rng)
        images[:, :, k] = sd.image
    end
    ratios = Float64[]
    for j in axes(Sbar, 2), l in axes(Sbar, 1)
        push!(ratios, var(images[l, j, :]) / sig2[l, j])
    end
    pooled = mean(ratios)
    @info "NZ6 GaussianNoise pooled variance ratio" pooled
    @test isapprox(pooled, 1.0; rtol=0.03)
end

@testset "NZ7: reproducibility" begin
    expt = _nz_streak()
    model = StreakCameraNoise()
    sd1 = synthesize(expt, model; rng=Xoshiro(42))
    sd2 = synthesize(expt, model; rng=Xoshiro(42))
    @test all(sd1.image .=== sd2.image)
end

@testset "NZ8: the residual statistic" begin
    expt = _nz_streak(nlambda=64, nt=16)
    model = StreakCameraNoise(sigma_px=0.0)
    rng = Xoshiro(20260906)
    sd = synthesize(expt, model; rng=rng, source="NZ8")
    Sbar = measure(expt)
    resid = (sd.image .- Sbar) ./ sd.sigma
    n = length(resid)
    m = mean(resid)
    v = var(resid)
    @info "NZ8" m v n
    @test isapprox(m, 0.0; atol=4 / sqrt(n))
    @test isapprox(v, 1.0; atol=4 * sqrt(2 / n))
end

@testset "NZ9: sigma is positive everywhere, including at zero signal" begin
    Sbar = zeros(4, 3)
    model = StreakCameraNoise(read_noise=11.0)
    s = sigma(model, Sbar)
    @test all(>(0), s)
    @test all(isapprox.(s, 11.0; rtol=1e-14))
end

@testset "NZ10: every noise error of D28" begin
    @test_throws "expected signal is negative" sigma(StreakCameraNoise(), [-1.0 2.0])
    @test_throws "a must be strictly positive" GaussianNoise(a=0.0)
    @test_throws "lambda must be non-negative" _rand_poisson(Xoshiro(1), -1.0)
end

@testset "NZ11: the zero branch is today's expression, bitwise" begin
    S = [10.0 20.0 30.0; 40.0 50.0 60.0]
    oracle = sqrt.(S .* 135.0 .* 1.05 .+ 11.0^2)
    for sigma_px in (0.0, -1.0)
        model = StreakCameraNoise(sigma_px=sigma_px)
        @test all(sigma(model, S) .=== oracle)
    end
end

@testset "NZ12: the spread's own closed form, an independent oracle" begin
    S = [10.0 * (l + 2j) for l in 1:5, j in 1:4]
    model = StreakCameraNoise(sigma_px=1.3)
    w(n, s) = (W = [exp(-(l - a)^2 / (2s^2)) for l in 1:n, a in 1:n]; W ./ sum(W; dims=1))
    wl, wt = w(5, 1.3), w(4, 1.3)
    oracle = [sqrt(sum(wl[l, a]^2 * wt[j, b]^2 * S[a, b] * 135.0 * 1.05 for a in 1:5, b in 1:4) + 11.0^2)
              for l in 1:5, j in 1:4]
    @test isapprox(sigma(model, S), oracle; rtol=1e-13)
end

@testset "NZ13: the realized noise against the realized sigma, the spread on" begin
    expt = _nz_streak()
    model = StreakCameraNoise()
    Sbar = measure(expt)
    K = 400
    images = Array{Float64}(undef, size(Sbar)..., K)
    for k in 1:K
        rng = Xoshiro(20260926 + k)
        sd = synthesize(expt, model; rng=rng)
        images[:, :, k] = sd.image
    end
    sigma2 = sigma(model, Sbar) .^ 2
    ratios = Float64[]
    worst = 0.0
    worst_pixel = (0, 0)
    for j in axes(Sbar, 2), l in axes(Sbar, 1)
        s2 = var(images[l, j, :])
        r = s2 / sigma2[l, j]
        push!(ratios, r)
        dev = abs(r - 1.0)
        if dev > worst
            worst = dev
            worst_pixel = (l, j)
        end
        @test isapprox(r, 1.0; rtol=0.25)
    end
    pooled = mean(ratios)
    @info "NZ13 pooled variance ratio, worst pixel deviation" pooled worst worst_pixel
    @test isapprox(pooled, 1.0; rtol=0.03)
end
