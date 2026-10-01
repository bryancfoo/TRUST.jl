using Test
using OTIS
using OTIS: _SubsampledAxis, _kernel_effects, _chain_order
import ForwardDiff

# Step 4.F.2: the pixel's integration, sampled (Phase 4.F spec D10-D12, OQ-53, the user's decision
# of 2026-09-30). PI1-PI5 and PI7, gate "pixel integration"; PI6 and PI8, which compile the KB chain's
# dual pass, are in the slow tier (test/slow/test_pixel_integration.jl), the phase's share (spec D24).
# PixelIntegration(subsamples = N) averages the spectrum over N points inside each pixel before every
# detector effect; N = 1 is today's model bitwise.

include(joinpath(@__DIR__, "..", "fixtures", "sampling_fixture.jl"))

# ------------------------------------------------------------------------------------------
# PI1: the effect, its traits and its refusals
# ------------------------------------------------------------------------------------------

@testset "PI1: the effect, its traits and its refusals" begin
    e = PixelIntegration(subsamples=4)
    @test e.subsamples == 4
    @test PixelIntegration().subsamples == 1
    @test isbitstype(PixelIntegration)
    @test_throws "subsamples must be a positive integer; got 0" PixelIntegration(subsamples=0)
    @test stage(e) === :kernel
    @test _chain_order(e) == 1
    @test islinear(e) == false
    @test parameters(e) == NamedTuple()
    @test reconstruct(e, NamedTuple()) === e
    @test sprint(show, e) == "PixelIntegration(4 subsamples)"

    # innermost whatever the order given, every other effect's order kept
    expt = _pi_kb_expt(4)
    @test [nameof(typeof(x)) for x in expt.effects] ==
          [:KSmearing, :GradientSmearing, :WavelengthCalibration, :PixelIntegration, :SpectralIRF, :Sensitivity]
    @test length(_kernel_effects(expt)) == 4

    trace = PlasmaTrace(_pi_gn_plasma(); t_ns=[4.6])
    @test_throws "only one PixelIntegration is allowed" Experiment(; trace=trace, probe=_PI_PROBE,
        collection=_PI_COLL, lambda_m=_PI_LAMBDA, effects=(PixelIntegration(subsamples=2), PixelIntegration(subsamples=3)))
    @test_throws "needs at least two pixels" Experiment(; trace=trace, probe=_PI_PROBE,
        collection=_PI_COLL, lambda_m=[240e-9], effects=(PixelIntegration(subsamples=2),))
end

# ------------------------------------------------------------------------------------------
# PI2: one sub-sample is today's model, bitwise (the KB chain's residuals and Jacobian: PI8, slow)
# ------------------------------------------------------------------------------------------

@testset "PI2: one sub-sample is today's model, bitwise" begin
    @test all(measure(_pi_gn_expt(PixelIntegration(subsamples=1))) .=== measure(_pi_gn_expt()))
    @test all(ideal_streak(_pi_gn_expt(PixelIntegration(subsamples=1))) .=== ideal_streak(_pi_gn_expt()))
end

# ------------------------------------------------------------------------------------------
# PI3: the sample positions and weights
# ------------------------------------------------------------------------------------------

@testset "PI3: the sample positions and weights" begin
    lam = collect(range(250e-9, 251e-9; length=11))           # uniform, pitch 0.1 nm
    d = lam[2] - lam[1]
    p = _pi_gn_plasma()
    s = samples(PixelIntegration(subsamples=4), p, _PI_PROBE, _PI_COLL, lam)
    @test length(s) == 4
    @test all(x -> x[1] == 0.25, s)
    @test all(x -> x[2] === p && x[3] === _PI_PROBE && x[4] === _PI_COLL, s)
    for (i, x) in enumerate(s)
        c = (i - 2.5) / 4
        @test all(isapprox(x[5][l], lam[l] + c * d; rtol=1e-15) for l in 2:10)
    end

    # a non-uniform axis: the centred spacing inside, one-sided at the two ends
    nu = [1.0, 2.0, 4.0, 7.0] .* 1e-9
    a = _SubsampledAxis{Float64,Vector{Float64}}(nu, 0.25)
    @test a[1] == nu[1] + 0.25 * (nu[2] - nu[1])
    @test a[2] == nu[2] + 0.25 * ((nu[3] - nu[1]) / 2)
    @test a[4] == nu[4] + 0.25 * (nu[4] - nu[3])
    @test size(a) == size(nu)

    # N = 1: weight one, offset zero
    s1 = samples(PixelIntegration(), p, _PI_PROBE, _PI_COLL, lam)
    @test s1[1][1] === 1.0
    @test all(s1[1][5][l] === lam[l] for l in eachindex(lam))

    # a dual axis (a fitted calibration) stays inferable
    dl = ForwardDiff.Dual{Nothing}.(lam, 1.0)
    @test length(@inferred samples(PixelIntegration(subsamples=3), p, _PI_PROBE, _PI_COLL, dl)) == 3

    # the kernel's own fold (measure.jl) is these samples' fold, bitwise
    for n in (2, 3, 4)
        kernel = OTIS._accumulate!(zeros(11), p, _PI_PROBE, _PI_COLL, lam, (PixelIntegration(subsamples=n),), 1.0, true)
        folded = zeros(11)
        for (wq, pq, probeq, collq, axis) in samples(PixelIntegration(subsamples=n), p, _PI_PROBE, _PI_COLL, lam)
            folded .+= wq .* spectrum(pq, probeq, collq, collect(axis))
        end
        @test all(kernel .=== folded)
    end
end

# ------------------------------------------------------------------------------------------
# PI4: the fold, against an explicit average over every node and sub-sample
# ------------------------------------------------------------------------------------------

@testset "PI4: the fold, against an explicit average" begin
    for n in (3, 4)
        expt = _pi_kb_expt(n)
        P = ideal_streak(expt)
        ks = expt.effects[1]; gs = expt.effects[2]; wc = expt.effects[3]
        lam = expt.lambda
        lref = (first(lam) + last(lam)) / 2
        cal = lam .+ wc.offset .+ wc.dispersion_scale .* (lam .- lref)
        pitch = [(cal[min(l + 1, end)] - cal[max(l - 1, 1)]) / (min(l + 1, length(cal)) - max(l - 1, 1)) for l in eachindex(cal)]
        j = 2
        pj = plasma(expt.trace, j)
        ref = zeros(length(lam))
        for (wq, oq) in zip(ks.weights, ks.offsets), (wg, og) in zip(gs.weights, gs.offsets), i in 1:n
            coll = Collection{Float64}(_PI_COLL.theta + oq, :exact, 0.0, 0.0, 0, 0.0)
            pg = OTIS._with_leaf(pj, gs.ref, OTIS._leaf_value(pj, gs.ref) + og)
            axis = cal .+ ((i - (n + 1) / 2) / n) .* pitch
            ref .+= (wq * wg / n) .* spectrum(pg, _PI_PROBE, coll, axis)
        end
        @test maximum(abs.(P[:, j] .- ref)) <= 1e-13 * maximum(abs, ref)
    end
end

# ------------------------------------------------------------------------------------------
# PI5: the resolved fixture -- Phase 4.E's bin 25, counts 1 to 16 against 64
# ------------------------------------------------------------------------------------------

@testset "PI5: the resolved fixture" begin
    col(n) = measure(_pi_gn_expt(PixelIntegration(subsamples=n)))[:, 1]
    ref = col(64)
    err = Dict(n => maximum(abs.(col(n) .- ref)) / maximum(ref) for n in (1, 2, 4, 8, 16))
    @info "PI5: max|M_N - M_64| / max(M_64)" err1 = err[1] err2 = err[2] err4 = err[4] err8 = err[8] err16 = err[16]
    @test err[1] > 0.5          # the centre-sampled model misses the line by more than half the peak
    @test err[2] < err[1]
    @test err[4] > err[2]       # 0.25-pixel steps still alias a 0.198-pixel line
    @test err[8] < err[2]
    @test err[8] <= 0.02
    @test err[16] < err[8]
    @test err[16] <= 1.5e-3
end

# ------------------------------------------------------------------------------------------
# PI7: the deck key
# ------------------------------------------------------------------------------------------

# On the resonance-width check's one-bin deck (test/fixtures/sampling_fixture.jl), whose chain the
# RW rows build too.
@testset "PI7: the deck key" begin
    mktempdir() do dir
        _rw_streak(dir)
        w(name, text) = (p = joinpath(dir, name); write(p, text); p)
        deck_with(line) = read_deck(w("d.toml", _rw_deck_text(dir; line=line)))

        deck = deck_with("pixel_integration = { subsamples = 4 }")
        @test deck.sections["instrument"]["pixel_integration"] == Dict{String,Any}("subsamples" => 4)
        problem, opt, start = OTIS.build_problem(deck)
        @test [nameof(typeof(e)) for e in problem.experiment.effects] == [:PixelIntegration, :SpectralIRF, :Sensitivity]
        @test problem.experiment.effects[1] === PixelIntegration(subsamples=4)
        @test read_deck(w("eff.toml", effective_deck(deck, problem, opt))) == deck

        one = OTIS.build_problem(deck_with("pixel_integration = { subsamples = 1 }"))
        none = OTIS.build_problem(deck_with(""))
        @test all(residuals(one[1], one[3]) .=== residuals(none[1], none[3]))

        @test_throws "must be an integer" deck_with("pixel_integration = { subsamples = 1.5 }")
        @test_throws "must be an integer" deck_with("pixel_integration = { subsamples = \"n\" }")
        @test_throws "subsamples must be a positive integer; got 0" OTIS.build_problem(
            deck_with("pixel_integration = { subsamples = 0 }"))
    end
end
