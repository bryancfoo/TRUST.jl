using Test
using OTIS
using OTIS: _chain_order, _chain_class, _kernel_rank, _kernel_effects, ACCEPTED_UNITS

# Step 4.G.1: the tabulated k-smearing (Phase 4.G spec D3-D9, OQ-56, decided 2026-10-03). TK1-TK8,
# gate "tabulated k-smearing"; TK9, which compiles the KB chain's dual pass with the rays in it, is in
# the slow tier (test/slow/test_tabulated_k_smearing.jl). TabulatedKSmearing averages the spectrum over
# an explicit list of collection angles, offsets from the Collection's theta, with the user's weights.

include(joinpath(@__DIR__, "..", "fixtures", "tabulated_k_smearing_fixture.jl"))

# ------------------------------------------------------------------------------------------
# TK1: the effect, its traits and its show
# ------------------------------------------------------------------------------------------

@testset "TK1: the effect, its traits and its show" begin
    e = _tk_rays()
    @test e isa TabulatedKSmearing{10,Float64}
    @test isbitstype(typeof(e))
    @test e.offsets === Tuple(theta_to_si.(collect(_TK_OFFSETS_DEG), :theta_deg))
    @test e.weights === _TK_WEIGHTS            # their sum is one to within 10 ulps: kept as given
    @test all(isapprox(_TK_OFFSETS_DEG[q], _TK_THETA_DEG[q] - 60.01; atol=1e-12) for q in 1:10)
    rad = TabulatedKSmearing(offsets_rad=collect(e.offsets), weights=collect(_TK_WEIGHTS))
    @test rad.offsets === e.offsets && rad.weights === e.weights

    # the weights are divided by their sum when it is not one; a zero weight is a node that adds nothing
    @test TabulatedKSmearing(offsets_deg=[-1.0, 0.0, 1.0], weights=[1, 2, 1]).weights === (0.25, 0.5, 0.25)
    @test TabulatedKSmearing(offsets_deg=[0.0], weights=[0.3]).weights === (1.0,)
    @test TabulatedKSmearing(offsets_deg=[-1.0, 1.0], weights=[0.0, 2.0]).weights === (0.0, 1.0)
    @test TabulatedKSmearing(offsets_rad=(0.0, 0.01), weights=(1.0, 1.0)).weights === (0.5, 0.5)

    @test stage(e) === :kernel
    @test _chain_order(e) == 1
    @test islinear(e) == false
    @test _chain_class(e) === :none
    @test _kernel_rank(e) == 0
    @test parameters(e) == NamedTuple()
    @test reconstruct(e, NamedTuple()) === e
    @test sprint(show, e) == "TabulatedKSmearing(10 nodes, -6.37 to 6.11 deg)"
    @test first(keys(ACCEPTED_UNITS.offsets)) === :offsets_deg
    @test values(ACCEPTED_UNITS.offsets) == values(ACCEPTED_UNITS.theta)

    # two may appear, and one beside a KSmearing, in the order given (the class is :none)
    expt = _tk_expt(:epw, _tk_uniform(), e, _tk_rays())
    @test [nameof(typeof(x)) for x in expt.effects] ==
          [:KSmearing, :TabulatedKSmearing, :TabulatedKSmearing, :SpectralIRF, :Sensitivity]
    @test length(_kernel_effects(expt)) == 3
end

# ------------------------------------------------------------------------------------------
# TK2: the refusals, each with its message
# ------------------------------------------------------------------------------------------

@testset "TK2: the refusals" begin
    @test_throws "offsets and weights must have the same length; got 2 and 1" TabulatedKSmearing(
        offsets_deg=[0.0, 1.0], weights=[1.0])
    @test_throws "at least one node is needed; got none" TabulatedKSmearing(offsets_deg=Float64[], weights=Float64[])
    @test_throws "weights must be non-negative; got -0.5 at node 2" TabulatedKSmearing(
        offsets_deg=[0.0, 1.0], weights=[1.0, -0.5])
    @test_throws "the weights must not all be zero" TabulatedKSmearing(offsets_deg=[0.0, 1.0], weights=[0.0, 0.0])
    @test_throws "offsets must be strictly increasing; got 0.0 then 0.0 rad at nodes 1 and 2" TabulatedKSmearing(
        offsets_deg=[0.0, 0.0], weights=[1.0, 1.0])
    @test_throws "offsets must be strictly increasing" TabulatedKSmearing(offsets_deg=[1.0, 0.0], weights=[1.0, 1.0])
    @test_throws "every offset must be finite; got NaN rad at node 2" TabulatedKSmearing(
        offsets_deg=[0.0, NaN], weights=[1.0, 1.0])
    @test_throws "every offset must be finite; got Inf rad at node 1" TabulatedKSmearing(
        offsets_rad=[Inf, 0.0], weights=[1.0, 1.0])
    @test_throws "every weight must be finite; got Inf at node 2" TabulatedKSmearing(
        offsets_deg=[0.0, 1.0], weights=[1.0, Inf])
    @test_throws "offsets_deg must be a vector of real numbers; got [\"a\"]" TabulatedKSmearing(
        offsets_deg=["a"], weights=[1.0])
    @test_throws "weights must be a vector of real numbers; got 1.0" TabulatedKSmearing(offsets_deg=[0.0], weights=1.0)
    @test_throws "more than one unit keyword given for offsets" TabulatedKSmearing(
        offsets_deg=[0.0], offsets_rad=[0.0], weights=[1.0])
    @test_throws "no unit keyword given for offsets" TabulatedKSmearing(weights=[1.0])
    @test_throws UndefKeywordError TabulatedKSmearing(offsets_deg=[0.0])
    @test_throws "unrecognised keyword(s) (:nodes,)" TabulatedKSmearing(offsets_deg=[0.0], weights=[1.0], nodes=3)
end

# ------------------------------------------------------------------------------------------
# TK3: built from a KSmearing's own offsets and weights, it is that KSmearing, bitwise
# ------------------------------------------------------------------------------------------

@testset "TK3: a KSmearing's own nodes give its streak, bitwise" begin
    k = _tk_uniform()
    t = TabulatedKSmearing(offsets_rad=k.offsets, weights=k.weights)
    @test t.offsets === k.offsets
    @test t.weights === k.weights          # they sum to 1 + 2.2e-16: kept, not divided
    @test all(measure(_tk_expt(:epw, t)) .=== measure(_tk_expt(:epw, k)))
    @test all(ideal_streak(_tk_expt(:epw, t)) .=== ideal_streak(_tk_expt(:epw, k)))
end

# ------------------------------------------------------------------------------------------
# TK4: one node at offset zero is the unsmeared model, bitwise
# ------------------------------------------------------------------------------------------

@testset "TK4: one node at offset zero is the unsmeared model, bitwise" begin
    none = measure(_tk_expt(:epw))
    @test all(measure(_tk_expt(:epw, TabulatedKSmearing(offsets_deg=[0.0], weights=[1.0]))) .=== none)
    @test all(measure(_tk_expt(:epw, TabulatedKSmearing(offsets_rad=[0.0], weights=[0.3]))) .=== none)
    @test all(ideal_streak(_tk_expt(:epw, TabulatedKSmearing(offsets_deg=[0.0], weights=[1.0]))) .===
              ideal_streak(_tk_expt(:epw)))
end

# ------------------------------------------------------------------------------------------
# TK5: the ten rays' samples, and their allocation
# ------------------------------------------------------------------------------------------

@testset "TK5: the ten rays' samples" begin
    e = _tk_rays()
    p = _tk_template()
    s = samples(e, p, _TK_PROBE, _TK_COLL, _TK_EPW)
    @test length(s) == 10
    @test all(s[q][1] === e.weights[q] for q in 1:10)
    @test all(s[q][2] === p && s[q][3] === _TK_PROBE && s[q][5] === _TK_EPW for q in 1:10)
    @test all(s[q][4].theta === _TK_COLL.theta + e.offsets[q] for q in 1:10)
    @test all(s[q][4].k_model === :exact && s[q][4].npixels == 0 for q in 1:10)
    @test length(@inferred samples(e, p, _TK_PROBE, _TK_COLL, _TK_EPW)) == 10

    _st() = samples(e, p, _TK_PROBE, _TK_COLL, _TK_EPW)
    _st()
    @test (@allocated _st()) == 0
end

# ------------------------------------------------------------------------------------------
# TK6: the ten rays against the equal-variance KSmearing, and against no smearing
# ------------------------------------------------------------------------------------------

@testset "TK6: the ten rays against the equal-variance KSmearing" begin
    rel(a, b) = maximum(abs.(a .- b)) / maximum(b)
    for (window, lo, hi, slo, shi) in ((:epw, 4.7e-4, 1.9e-3, 1.6e-2, 6.2e-2), (:iaw, 2.9e-5, 1.2e-4, 2.1e-3, 8.5e-3))
        none = measure(_tk_expt(window))
        uniform = measure(_tk_expt(window, _tk_uniform()))
        rays = measure(_tk_expt(window, _tk_rays()))
        d = rel(rays, uniform)
        s = rel(rays, none)
        @info "TK6: $(window), max|rays - uniform| and max|rays - none| over the peak" d s
        @test lo <= d <= hi
        @test slo <= s <= shi
        @test d < s / 10
    end
end

# ------------------------------------------------------------------------------------------
# TK7: beside a KSmearing, the fold over every pair of nodes
# ------------------------------------------------------------------------------------------

@testset "TK7: composed with a KSmearing" begin
    k = KSmearing(half_angle_deg=1.0, nodes=3)
    e = _tk_rays()
    expt = _tk_expt(:epw, k, e)
    P = ideal_streak(expt)
    j = 3
    pj = plasma(expt.trace, j)
    ref = zeros(length(_TK_EPW))
    for (wk, ok) in zip(k.weights, k.offsets), (wt, ot) in zip(e.weights, e.offsets)
        coll = Collection{Float64}(_TK_COLL.theta + ok + ot, :exact, 0.0, 0.0, 0, 0.0)
        ref .+= (wk * wt) .* spectrum(pj, _TK_PROBE, coll, _TK_EPW)
    end
    err = maximum(abs.(P[:, j] .- ref)) / maximum(abs, ref)
    @info "TK7: the fold against the explicit double sum" err
    @test err <= 1e-13
end

# ------------------------------------------------------------------------------------------
# TK8: the deck key
# ------------------------------------------------------------------------------------------

@testset "TK8: the deck key" begin
    mktempdir() do dir
        _tk_streak(dir)
        w(name, text) = (p = joinpath(dir, name); write(p, text); p)

        # read, built, round-tripped through the effective deck and the result file
        deck = read_deck(w("d.toml", _tk_deck_text(dir)))
        @test deck.sections["instrument"]["tabulated_k_smearing"] ==
              Dict{String,Any}("offsets_deg" => collect(_TK_OFFSETS_DEG), "weights" => collect(_TK_WEIGHTS))
        problem, opt, start = OTIS.build_problem(deck)
        effects = problem.experiment.effects
        @test [nameof(typeof(x)) for x in effects] == [:TabulatedKSmearing, :SpectralIRF, :Sensitivity]
        @test effects[1].offsets === _tk_rays().offsets && effects[1].weights === _tk_rays().weights
        eff = effective_deck(deck, problem, opt)
        @test read_deck(w("eff.toml", eff)) == deck
        o = objective(problem, start)
        result = OTIS._build_result(problem, start, o, [(iteration=0.0, objective=o, chi2=o, R=0.0)], true,
            :f_tolerance, 0, 1, 0, 0.0, "none")
        write_result(joinpath(dir, "r.jld2"), result; problem=problem, deck=eff)
        r = read_result(joinpath(dir, "r.jld2"))
        @test r.deck == eff
        @test r.channels[1].effects[1].type == "TabulatedKSmearing"
        @test isempty(r.channels[1].effects[1].parameters)
        @test read_deck(w("back.toml", r.deck)) == deck

        # the channel form, beside a k_smearing: refused, since the two would smear twice (amended at
        # Step 4.H.0, 2026-10-05, with the user's approval, Phase 4.H spec D4; until then this row built
        # both, k_smearing first; in Julia the two still compose, TK7)
        @test_throws "read_deck: in [channel.epw]: read_deck: section \"instrument\" gives both \"k_smearing\" and \"tabulated_k_smearing\"" read_deck(
            w("c.toml", _tk_deck_text(dir; form=:channel,
                line=_TK_DECK_LINE * "\nk_smearing = { half_angle_deg = 1.0, nodes = 3 }")))

        # refused at read_deck: a string, a number for a list, an unknown key
        rd(line) = read_deck(w("x.toml", _tk_deck_text(dir; line=line)))
        @test_throws "takes numbers only: offsets_deg and weights cannot hold a variable reference" rd(
            "tabulated_k_smearing = { offsets_deg = [\"T_e\"], weights = [1.0] }")
        @test_throws "key \"offsets_deg\" in section \"instrument.tabulated_k_smearing\" must be a list" rd(
            "tabulated_k_smearing = { offsets_deg = 1.0, weights = [1.0] }")
        @test_throws "unknown key \"nodes\" in section \"instrument.tabulated_k_smearing\"" rd(
            "tabulated_k_smearing = { offsets_deg = [0.0], weights = [1.0], nodes = 3 }")
        @test_throws "in [channel.epw]: read_deck: section \"instrument.tabulated_k_smearing\" takes numbers only" read_deck(
            w("y.toml", _tk_deck_text(dir; form=:channel,
                line="tabulated_k_smearing = { offsets_deg = [0.0], weights = [\"x\"] }")))

        # refused when build_problem constructs the effect, with the constructor's message
        bp(line) = OTIS.build_problem(rd(line))
        @test_throws "offsets and weights must have the same length; got 2 and 1" bp(
            "tabulated_k_smearing = { offsets_deg = [0.0, 1.0], weights = [1.0] }")
        @test_throws "offsets and weights must have the same length; got 0 and 1" bp(
            "tabulated_k_smearing = { weights = [1.0] }")
        @test_throws "at least one node is needed; got none" bp("tabulated_k_smearing = {}")
        @test_throws "weights must be non-negative; got -1.0 at node 1" bp(
            "tabulated_k_smearing = { offsets_deg = [0.0, 1.0], weights = [-1.0, 2.0] }")
        @test_throws "the weights must not all be zero" bp(
            "tabulated_k_smearing = { offsets_deg = [0.0, 1.0], weights = [0.0, 0.0] }")
        @test_throws "offsets must be strictly increasing" bp(
            "tabulated_k_smearing = { offsets_deg = [1.0, 0.0], weights = [1.0, 1.0] }")
        @test_throws "every offset must be finite; got Inf rad at node 2" bp(
            "tabulated_k_smearing = { offsets_deg = [0.0, inf], weights = [1.0, 1.0] }")
    end
end
