using Test
using TRUST
using TRUST: _variable_channels, _stage_deck, _staged_fit

# Step 4.F.1's slow rows (Phase 4.F spec D9, D24), gated "staged fit (slow)": ST3, the staged fit
# through fit(deck), which fits, and ST4, the reach, whose two-channel problem with a channel-only
# binding is compiled for it alone. fit(deck) pins BLAS to one thread for the rest of the process
# (conventions.md 6.6); this file restores the count it found, as TS2 does, for the slow files that
# follow.

include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))

const _STS_NBLAS = pin_blas_threads()
pin_blas_threads(_STS_NBLAS)

# ------------------------------------------------------------------------------------------
# ST3: fit(deck) with procedure = "staged", against an independent assembly of every part
# ------------------------------------------------------------------------------------------

@testset "ST3: the staged fit through fit(deck)" begin
    mktempdir() do dir
        _st_streaks(dir)
        path = _st_write(dir, "d.toml", _st_text(dir))
        deck = read_deck(path)
        result = fit(deck)
        r = read_result(joinpath(dir, "out", "result.jld2"))

        @test r.stages.names == ["epw", "iaw"]
        @test r.stages.sources == Dict("logn" => "epw", "T_e" => "epw", "m_e" => "epw", "T_D" => "iaw")
        @test isempty(r.stages.channels["epw"].frozen) && isempty(r.stages.channels["iaw"].frozen)

        # each stage is the fit of the single-channel deck examples/problems.jl derives from text
        plain = _st_text(dir; optimizer="procedure = \"lbfgs\"")
        for name in ("epw", "iaw")
            p, opt, st = TRUST.build_problem(read_deck(_st_write(dir, "$name.toml", single_channel_deck_text(plain, name))))
            alone = solve(p, opt; start=st)
            @test r.stages.channels[name].theta == alone.theta
            @test r.stages.channels[name].chi2 == alone.chi2
            @test r.stages.channels[name].variables["logn"].values == alone.values["logn"]
        end

        # the joint start is the stages' values, assembled by name through flatten
        problem, _, _ = TRUST.build_problem(deck)
        vals = [v.name => r.stages.channels[r.stages.sources[v.name]].variables[v.name].values
                for v in problem.pspec.variables]
        @test r.stages.start == flatten(problem.pspec, vals)

        # the joint fit and polish, rerun from that start
        joint = solve(problem, LevenbergMarquardt(); start=r.stages.start)
        pol = polish(problem, joint; tol=1e-6)
        @test result.theta == pol.result.theta
        @test r.fit.theta == pol.result.theta
        @test r.stages.polish_decreases == pol.decreases
        @test r.fit.gn_decrease == pol.decreases[end]

        # a fit that is not staged writes no stages
        fit(read_deck(_st_write(dir, "plain.toml", plain)))
        @test read_result(joinpath(dir, "out", "result.jld2")).stages === nothing
    end
end

# ------------------------------------------------------------------------------------------
# ST4: which channels a variable reaches, the frozen rule and the source rule (no fit)
# ------------------------------------------------------------------------------------------

@testset "ST4: the reach, the frozen variable and the start_from refusal" begin
    mktempdir() do dir
        _st_streaks(dir)
        # s_iaw reaches the IAW channel alone, through the dependent sx bound to its sensitivity
        text = _st_text(dir)
        text = replace(text, "sensitivity = { scale = 1.0e-6 }" => "sensitivity = { scale = \"sx\" }")
        text = replace(text, "ne_cm3 = \"10^logn\"" => "ne_cm3 = \"10^logn\"\nsx = \"s_iaw * 1.0e-6\"")
        text = replace(text, "T_D  = { bounds" =>
            "s_iaw = { bounds = [0.5, 2.0], init = 1.0 }\nT_D  = { bounds")
        text = replace(text, "[channel.iaw.data]" => "[channel.iaw]\nweight = 2.5\n\n[channel.iaw.data]")
        deck = read_deck(_st_write(dir, "reach.toml", text))
        problem, _, _ = TRUST.build_problem(deck)
        names = [v.name for v in problem.pspec.variables]
        reach = Dict(zip(names, _variable_channels(problem.pspec, 2)))
        @test reach == Dict("T_D" => [1, 2], "T_e" => [1, 2], "s_iaw" => [2], "logn" => [1, 2], "m_e" => [1, 2])

        sdeck = _stage_deck(deck, 1, "lbfgs", ["s_iaw"])
        @test [c["name"] for c in sdeck.sections["channels"]] == ["epw"]
        @test sdeck.sections["optimizer"]["procedure"] == "lbfgs"
        @test !haskey(sdeck.sections["optimizer"], "staged")
        @test sdeck.sections["fit_variables"]["s_iaw"]["vary"] == false
        @test deck.sections["fit_variables"]["s_iaw"]["vary"] == true
        sdeck2 = _stage_deck(deck, 2, "lbfgs", String[])
        @test [c["name"] for c in sdeck2.sections["channels"]] == ["iaw"]
        @test sdeck2.sections["channels"][1]["weight"] == 1.0
        @test deck.sections["channels"][2]["weight"] == 2.5

        opts = (stage_procedure="lbfgs", joint_optimizer=LevenbergMarquardt(), polish_tol=1e-6)
        @test_throws "start_from takes \"s_iaw\" from channel \"epw\", whose stage freezes it" _staged_fit(deck;
            opts..., start_from=Dict{String,Any}("s_iaw" => "epw"))

        unused = replace(text, "T_D  = { bounds" => "unused = { bounds = [0.0, 1.0], init = 0.5 }\nT_D  = { bounds")
        deck_u = read_deck(_st_write(dir, "unused.toml", unused))
        @test_throws "the fitted variable \"unused\" reaches no channel's model" _staged_fit(deck_u;
            opts..., start_from=Dict{String,Any}())
    end
end

pin_blas_threads(_STS_NBLAS)
