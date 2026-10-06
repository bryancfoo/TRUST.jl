using Test
using OTIS
using OTIS: JACOBIAN_CHUNK, _convective_gain, _spectrum_at, _widen_plasma, _epw_resonance_widths,
    _resonance_widths
import ForwardDiff

# Step 4.H.1: the convective gain in the fit path (Phase 4.H spec D5-D8, D11; OQ-57, decided
# 2026-10-05). GF1-GF7, gate "gain in fit"; GF8-GF10, which compile the brute-force Jacobian and the dual
# pass of a chain nothing earlier builds, or fit, are in the slow tier (test/slow/test_gain_in_fit.jl).
# `Experiment` carries the gain `spectrum` has computed since Step 2.3b, and every evaluation of the
# forward model -- measure, the Jacobian's dual pass, the resonance-width check -- takes it; the deck
# asks for it in [geometry].

include(joinpath(@__DIR__, "..", "fixtures", "tabulated_k_smearing_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "sampling_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "gain_in_fit_fixture.jl"))

# ------------------------------------------------------------------------------------------
# GF1: the field, its validation, its show, and unflatten's rebuild
# ------------------------------------------------------------------------------------------

@testset "GF1: the gain on Experiment" begin
    @test fieldnames(Experiment)[8:9] == (:prefactor, :gain)
    @test Experiment(; trace=_tk_trace(), probe=_GF_PROBE, collection=_TK_COLL, lambda_m=_TK_IAW).gain === :none
    @test _gf_expt(:iaw, :convective).gain === :convective
    @test _gf_expt(:iaw, :weak).gain === :weak
    @test_throws "Experiment: gain must be :none, :convective or :weak; got :strong" _gf_expt(:iaw, :strong)

    text(e) = sprint(show, MIME"text/plain"(), e)
    @test occursin("\n  gain:       :convective\n", text(_gf_expt(:iaw, :convective)))
    @test !occursin("gain", text(_gf_expt(:iaw, :none)))

    # unflatten's rebuild (an effect binding makes it rebuild) keeps the gain
    expt = _gf_expt(:iaw, :convective; chain=true)
    pspec = ParameterSpec(expt; variables=[FittedVariable("s"; bounds=(1e-5, 1e-3), init=2.0e-4)],
        bindings=["s" => (findfirst(e -> e isa Sensitivity, expt.effects), :scale)])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, gain=expt.gain)
    rebuilt = unflatten(working, pspec, flatten(pspec))
    @test rebuilt !== working
    @test rebuilt.gain === :convective
end

# ------------------------------------------------------------------------------------------
# GF2: with no kernel-stage effect, measure is spectrum(...; gain) per bin, bitwise
# ------------------------------------------------------------------------------------------

@testset "GF2: measure is spectrum with the gain, bin by bin" begin
    for window in (:epw, :iaw)
        lam = window === :epw ? _TK_EPW : _TK_IAW
        none = measure(_gf_expt(window, :none))
        for gain in (:convective, :weak)
            M = measure(_gf_expt(window, gain))
            @test all(all(M[:, j] .=== spectrum(plasma(_tk_trace(), j), _GF_PROBE, _TK_COLL, lam; gain=gain))
                      for j in 1:4)
        end
        # the default is today's model, and the gain is visible on this fixture, so the rows above see it
        @test all(none .=== measure(Experiment(; trace=_tk_trace(), probe=_GF_PROBE, collection=_TK_COLL,
            lambda_m=lam)))
        d = maximum(abs.(measure(_gf_expt(window, :convective)) .- none)) / maximum(none)
        @info "GF2: $(window), max|gain - none| over the peak" d
        @test d > 1e-3
    end
end

# ------------------------------------------------------------------------------------------
# GF3: a PixelIntegration fold carries the gain into every sub-sample, bitwise
# ------------------------------------------------------------------------------------------

@testset "GF3: the PixelIntegration fold" begin
    for n in (1, 4)
        e = PixelIntegration(subsamples=n)
        P = measure(_gf_expt(:epw, :convective, e))
        for j in (1, 4)
            pj = plasma(_tk_trace(), j)
            folded = zeros(length(_TK_EPW))
            for (wq, pq, probeq, collq, axis) in samples(e, pj, _GF_PROBE, _TK_COLL, _TK_EPW)
                folded .+= wq .* spectrum(pq, probeq, collq, collect(axis); gain=:convective)
            end
            @test all(P[:, j] .=== folded)
        end
    end
end

# ------------------------------------------------------------------------------------------
# GF4: the ten rays, each with its own gain
# ------------------------------------------------------------------------------------------

@testset "GF4: the ten rays with the gain on" begin
    e = _tk_rays()
    for window in (:epw, :iaw)
        lam = window === :epw ? _TK_EPW : _TK_IAW
        P = measure(_gf_expt(window, :convective, e))
        j = 3
        pj = plasma(_tk_trace(), j)
        ref = zeros(length(lam))
        for (wt, ot) in zip(e.weights, e.offsets)
            ref .+= wt .* spectrum(pj, _GF_PROBE, Collection(theta_rad=_TK_COLL.theta + ot, k_model=:exact), lam;
                gain=:convective)
        end
        err = maximum(abs.(P[:, j] .- ref)) / maximum(abs, ref)
        @info "GF4: $(window), the fold against the weighted sum of spectrum at each ray" err
        @test err <= 1e-13
    end
end

# ------------------------------------------------------------------------------------------
# GF5: the gain's branch is concrete under the Jacobian's dual (conventions.md 5.14)
# ------------------------------------------------------------------------------------------

@testset "GF5: the gain's branch is concrete under a dual" begin
    DFT = ForwardDiff.Dual{OTIS._JacobianTag,Float64,JACOBIAN_CHUNK}
    pd = _widen_plasma(_gf_template(), DFT)
    cd = Collection{DFT}(DFT(_TK_COLL.theta), :exact, DFT(0.0), DFT(0.0), 0, DFT(0.0))
    lam = 263.0e-9
    for (p, coll) in ((pd, _TK_COLL), (_gf_template(), cd), (pd, cd))
        k, omega, k_s = wavevector(_GF_PROBE, coll, lam, p)
        d = dielectric(p, k, omega)
        rg = Base.infer_return_type(_convective_gain,
            Tuple{Symbol,typeof(d),typeof(k),typeof(k_s),typeof(_GF_PROBE),typeof(coll)})
        rs = Base.infer_return_type(_spectrum_at,
            Tuple{typeof(p),typeof(_GF_PROBE),typeof(coll),Float64,Bool,Symbol})
        @test isconcretetype(rg) && rg === DFT
        @test isconcretetype(rs) && rs === DFT
        # and each arm returns that type at run time
        @test typeof(_convective_gain(:none, d, k, k_s, _GF_PROBE, coll)) === DFT
        @test typeof(_convective_gain(:convective, d, k, k_s, _GF_PROBE, coll)) === DFT
    end
end

# ------------------------------------------------------------------------------------------
# GF6: the deck key
# ------------------------------------------------------------------------------------------

@testset "GF6: the deck key" begin
    mktempdir() do dir
        _gf_streak(dir)
        w(name, text) = (p = joinpath(dir, name); write(p, text); p)
        on = _GF_ENERGETICS * "\ngain = \"convective\""

        # the shorthand: read, built, round-tripped through the effective deck and the result file
        deck = read_deck(w("d.toml", _gf_deck_text(dir; geometry=on)))
        @test deck.sections["geometry"]["gain"] == "convective"
        problem, opt, start = OTIS.build_problem(deck)
        @test problem.experiment.gain === :convective
        eff = effective_deck(deck, problem, opt)
        @test occursin("gain = \"convective\"", eff)
        @test read_deck(w("eff.toml", eff)) == deck
        o = objective(problem, start)
        result = OTIS._build_result(problem, start, o, [(iteration=0.0, objective=o, chi2=o, R=0.0)], true,
            :f_tolerance, 0, 1, 0, 0.0, "none")
        @test result.experiments[1].gain === :convective
        write_result(joinpath(dir, "r.jld2"), result; problem=problem, deck=eff)
        r = read_result(joinpath(dir, "r.jld2"))
        @test r.deck == eff
        @test read_deck(w("back.toml", r.deck)) == deck

        # the default is filled when the key is omitted, and carried by the effective deck
        plain = read_deck(w("p.toml", _gf_deck_text(dir)))
        @test plain.sections["geometry"]["gain"] == "none"
        pproblem, popt, _ = OTIS.build_problem(plain)
        @test pproblem.experiment.gain === :none
        @test occursin("gain = \"none\"", effective_deck(plain, pproblem, popt))
        @test read_deck(w("n.toml", _gf_deck_text(dir; geometry="gain = \"none\""))) == plain

        # the channel form: its own geometry carries the key; :weak
        cdeck = read_deck(w("c.toml", _gf_deck_text(dir; form=:channel, geometry=_GF_ENERGETICS * "\ngain = \"weak\"")))
        @test cdeck.sections["channels"][1]["geometry"]["gain"] == "weak"
        cproblem, copt, _ = OTIS.build_problem(cdeck)
        @test OTIS._channel_problem(cproblem, 1).experiment.gain === :weak
        @test read_deck(w("ceff.toml", effective_deck(cdeck, cproblem, copt))) == cdeck

        # refused at read_deck, each naming the section and the key
        rd(geometry; form=:shorthand) = read_deck(w("x.toml", _gf_deck_text(dir; form=form, geometry=geometry)))
        @test_throws "read_deck: key \"gain\" in section \"geometry\" must be \"none\", \"convective\" or \"weak\"; got \"strong\"" rd(
            _GF_ENERGETICS * "\ngain = \"strong\"")
        @test_throws "must be \"none\", \"convective\" or \"weak\"; got true" rd(_GF_ENERGETICS * "\ngain = true")
        @test_throws "read_deck: key \"gain\" in section \"geometry\" is \"convective\", which needs the probe's intensity: give intensity_wcm2 or intensity_wm2" rd(
            "diameter_um = 100.0\ngain = \"convective\"")
        @test_throws "read_deck: key \"gain\" in section \"geometry\" is \"weak\", which needs the probe's beam diameter: give diameter_um or diameter_m" rd(
            "intensity_wcm2 = 5.55e14\ngain = \"weak\"")
        @test_throws "read_deck: in [channel.epw]: read_deck: key \"gain\" in section \"geometry\" is \"convective\", which needs the probe's intensity" rd(
            "gain = \"convective\""; form=:channel)
    end
end

# ------------------------------------------------------------------------------------------
# GF7: the resonance-width check measures the model that was fitted, gain included
# ------------------------------------------------------------------------------------------

@testset "GF7: the resonance-width check takes the gain" begin
    mktempdir() do dir
        _rw_streak(dir)
        text = replace(_rw_deck_text(dir), "theta_deg = 60.0" => "theta_deg = 60.0\n" * _GF_ENERGETICS * "\ngain = \"convective\"")
        path = joinpath(dir, "g.toml")
        write(path, text)
        problem, _, start = OTIS.build_problem(read_deck(path))
        o = objective(problem, start)
        r = OTIS._build_result(problem, start, o, [(iteration=0.0, objective=o, chi2=o, R=0.0)], true,
            :f_tolerance, 0, 1, 0, 0.0, "none")
        expt = r.experiments[1]
        @test expt.gain === :convective
        p = plasma(expt.trace, 1)
        with = _epw_resonance_widths(p, expt.probe, expt.collection, expt.lambda, true, :convective)
        without = _epw_resonance_widths(p, expt.probe, expt.collection, expt.lambda, true, :none)
        @info "GF7: the bin's resonance width in pixels, with and without the gain" with without
        @test length(with) == length(without) == 1
        @test _resonance_widths(r)[1][1] == with[1]
        @test with[1] != without[1]
    end
end
