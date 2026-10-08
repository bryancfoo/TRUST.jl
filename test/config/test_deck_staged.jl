using Test
using OTIS

# Step 4.F.1: the staged fit as a deck option (Phase 4.F spec D3-D9, OQ-54, the user's decision of
# 2026-09-30). ST1 and ST2, gate "staged fit": the grammar and the refusals. ST3, the staged fit
# through fit(deck), and ST4, the reach, are in the slow tier (test/slow/test_deck_staged.jl): ST3
# fits, and ST4 compiles a two-channel problem with a channel-only binding (spec D24). procedure =
# "staged" fits each channel alone, then every channel jointly from those fits, then polishes; the
# stages' results go into the result file.

include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))

# ------------------------------------------------------------------------------------------
# ST1: the grammar, its defaults, the joint optimizer, the round trip
# ------------------------------------------------------------------------------------------

@testset "ST1: the staged table, its defaults and the round trip" begin
    mktempdir() do dir
        _st_streaks(dir)
        deck = read_deck(_st_write(dir, "d.toml", _st_text(dir)))
        @test deck.sections["optimizer"]["procedure"] == "staged"
        @test deck.sections["optimizer"]["staged"] == Dict{String,Any}("stage_procedure" => "lbfgs",
            "joint_procedure" => "levenberg_marquardt", "polish_tol" => 1e-6,
            "start_from" => Dict{String,Any}("T_D" => "iaw"))

        bare = read_deck(_st_write(dir, "bare.toml", _st_text(dir; optimizer="procedure = \"staged\"")))
        @test bare.sections["optimizer"]["staged"] == Dict{String,Any}(
            "stage_procedure" => "noisy_descent_lbfgs", "joint_procedure" => "levenberg_marquardt",
            "polish_tol" => 1e-6, "start_from" => Dict{String,Any}())

        problem, optimizer, _ = OTIS.build_problem(deck)
        @test problem isa JointFitProblem
        @test optimizer isa LevenbergMarquardt
        eff = effective_deck(deck, problem, optimizer)
        @test read_deck(_st_write(dir, "eff.toml", eff)) == deck

        # the committed example deck reads as written (read_deck opens no data file)
        ex = read_deck(joinpath(@__DIR__, "..", "..", "examples", "two_channel_staged.toml"))
        @test ex.sections["optimizer"]["staged"] == deck.sections["optimizer"]["staged"]
    end
end

# ------------------------------------------------------------------------------------------
# ST2: the refusals, each at read_deck
# ------------------------------------------------------------------------------------------

@testset "ST2: the refusals" begin
    mktempdir() do dir
        _st_streaks(dir)
        base = _st_text(dir)
        rd(name, text) = read_deck(_st_write(dir, name, text))

        one = _st_text(dir; optimizer=_ST_STAGED)
        one = replace(one, "channels = [\"epw\", \"iaw\"]" => "channels = [\"epw\"]")
        one = replace(one, r"\[channel\.iaw\.data\][\s\S]*?\[fit\.variables\]" => "[fit.variables]")
        @test_throws "this deck has one, \"epw\"" rd("one.toml", one)

        shorthand = """
        deck_version = "otis-deck-2"
        [data]
        file = "$(_st_path(joinpath(dir, "epw.jld2")))"
        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0
        [fit.variables]
        T_e = { bounds = [100.0, 2000.0], init = 800.0 }
        [plasma]
        n_e_cm3 = 1.0e19
        [plasma.electrons.e]
        T_eV = "T_e"
        [optimizer]
        procedure = "staged"
        """
        @test_throws "this deck is in the shorthand form" rd("short.toml", shorthand)

        @test_throws "is read only with procedure = \"staged\"" rd("ro.toml",
            _st_text(dir; optimizer="procedure = \"lbfgs\"\nstaged = { polish_tol = 1e-6 }"))
        @test_throws "\"optimizer.staged.stage_procedure\" cannot be \"staged\"" rd("nest.toml",
            replace(base, "stage_procedure = \"lbfgs\"" => "stage_procedure = \"staged\""))
        @test_throws "unknown procedure \"bogus\"" rd("bogus.toml",
            replace(base, "stage_procedure = \"lbfgs\"" => "stage_procedure = \"lbfgs\", joint_procedure = \"bogus\""))
        @test_throws "unknown key \"bogus\" in section \"optimizer.staged\"" rd("key.toml",
            replace(base, "stage_procedure = \"lbfgs\"" => "stage_procedure = \"lbfgs\", bogus = 1"))
        @test_throws "\"optimizer.staged.polish_tol\" must be a finite number > 0" rd("tol.toml",
            replace(base, "stage_procedure = \"lbfgs\"" => "stage_procedure = \"lbfgs\", polish_tol = 0.0"))
        @test_throws "names \"nope\", which is not a fitted variable" rd("var.toml",
            replace(base, "T_D = \"iaw\"" => "nope = \"iaw\""))
        @test_throws "must name a channel of \"channels\"" rd("chan.toml",
            replace(base, "T_D = \"iaw\"" => "T_D = \"xyz\""))
        frozen = replace(base, "init = 300.0, time_varying = true }" => "init = 300.0, time_varying = true, vary = false }")
        @test_throws "names \"T_D\", which is not a fitted variable" rd("frozen.toml", frozen)
    end
end

# ------------------------------------------------------------------------------------------
# ST5: the stage decks inherit [fit.start] and [fit.hold]; start_from cannot name a held variable
# (Phase 4.I spec D12, Step 4.I.1)
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "fixtures", "start_hold_fixture.jl"))

@testset "ST5: the stage decks inherit the start and hold tables" begin
    mktempdir() do dir
        _st_streaks(dir)
        sp = _sh_start_file(dir)
        text = _sh_tables(_st_text(dir); start=(sp, ["logn"]), hold=(sp, ["T_e"]))
        deck = read_deck(_st_write(dir, "d.toml", text))
        joint, _, _ = OTIS.build_problem(deck)
        for c in 1:2
            sdeck = OTIS._stage_deck(deck, c, "lbfgs", String[])
            @test sdeck.sections["fit_start"] == deck.sections["fit_start"]
            @test sdeck.sections["fit_hold"] == deck.sections["fit_hold"]
            p, _, st = OTIS.build_problem(sdeck)
            @test !only(v for v in p.pspec.variables if v.name == "T_e").vary
            @test p.pspec.ntheta == joint.pspec.ntheta == 12
        end
        @test_throws "read_deck: \"optimizer.staged.start_from\" names \"T_D\", which [fit.hold] holds, so no stage fits it" read_deck(
            _st_write(dir, "h.toml", _sh_tables(_st_text(dir); hold=(sp, ["T_D"]))))
    end
end
