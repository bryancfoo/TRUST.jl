using Test
using OTIS
using CairoMakie
using Random: Xoshiro

@testset "MK1: the extension loads" begin
    @test Base.get_extension(OTIS, :OTISMakieExt) !== nothing
end

function _mk_experiment(; nlambda=64, nt=16)
    template = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    trace = PlasmaTrace(template; t_ns=collect(range(2.0, 3.0; length=nt)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=nlambda))
    effects = (Sensitivity(scale=2.0e-4), ConstantBackground(level=25.0))
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects), trace
end

@testset "MK2: streakplot on a synthesized streak, and it saves" begin
    expt, _ = _mk_experiment()
    model = StreakCameraNoise(sigma_px=1.0)
    sd = synthesize(expt, model; rng=Xoshiro(20260911))

    fig = streakplot(sd)
    @test fig isa Makie.Figure

    path = joinpath(mktempdir(), "streak.png")
    save(path, fig)
    @test isfile(path)
    @test filesize(path) > 1024

    fig2 = Makie.Figure()
    ax = Makie.Axis(fig2[1, 1])
    plt = streakplot!(ax, sd)
    @test plt !== nothing
end

@testset "MK3: spectrumplot, standalone and into an existing axis" begin
    expt, _ = _mk_experiment()
    M = measure(expt)
    lambda_nm = lambda_from_si.(expt.lambda, :lambda_nm)

    fig = spectrumplot(lambda_nm, M[:, 1])
    @test fig isa Makie.Figure

    fig2 = Makie.Figure()
    ax = Makie.Axis(fig2[1, 1])
    plt = spectrumplot!(ax, lambda_nm, M[:, 2])
    @test plt !== nothing
end

@testset "MK4: traceplot, with and without a sigma band, and an unknown leaf throws" begin
    _, trace = _mk_experiment()

    fig = traceplot(trace, "electrons.e.T")
    @test fig isa Makie.Figure

    sigma_band = fill(20.0, nbins(trace))
    fig2 = traceplot(trace, "electrons.e.T"; sigma=sigma_band)
    @test fig2 isa Makie.Figure

    fig3 = Makie.Figure()
    ax = Makie.Axis(fig3[1, 1])
    plt = traceplot!(ax, trace, "ions.i.T")
    @test plt !== nothing

    @test_throws "has no species named" traceplot(trace, "electrons.bogus.T")
end

@testset "MK5: lcurveplot on two plain vectors" begin
    chi2 = collect(range(100.0, 1.0; length=8))
    R = collect(range(1.0, 100.0; length=8))

    fig = lcurveplot(chi2, R)
    @test fig isa Makie.Figure

    fig2 = Makie.Figure()
    ax = Makie.Axis(fig2[1, 1])
    plt = lcurveplot!(ax, chi2, R; weights=collect(1:8))
    @test plt !== nothing
end

@testset "MK6: streakplot blanks masked pixels unless masked = false (Phase 4.N spec D18)" begin
    image = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    mask = [true false; true true; false true]
    sd = StreakData(lambda_nm=[262.0, 263.0, 264.0], t_ns=[0.0, 1.0], image=image, mask=mask)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    z = streakplot!(ax, sd)[3][]
    @test isnan.(z) == permutedims(.!mask)
    @test z[permutedims(mask)] == permutedims(image)[permutedims(mask)]
    z_all = streakplot!(ax, sd; masked=false)[3][]
    @test z_all == permutedims(image)
    @test streakplot(sd) isa Makie.Figure
    @test streakplot(sd; masked=false) isa Makie.Figure
end
