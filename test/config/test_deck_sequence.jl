using Test
using OTIS
using OTIS: _staged_fit

# Step 4.I.3: the sequential stage (Phase 4.I spec D22 to D24, D26; OQ-59 item 4), gated "sequential
# staging". SQ1 and SQ2: the grammar, its round trip, and every refusal, none of which fits; SQ3 to
# SQ5, which fit, are in the slow tier (test/slow/test_deck_sequence.jl).

include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "start_hold_fixture.jl"))

# The staged fixture's deck with [optimizer.staged]'s `extra` entries set (a Dict of key => value).
_sq_text(dir, extra::AbstractDict) = _sh_edit(raw -> merge!(raw["optimizer"]["staged"], extra), _st_text(dir))

# ------------------------------------------------------------------------------------------
# SQ1: sequence and hold read, kept only when given, and round-trip
# ------------------------------------------------------------------------------------------

@testset "SQ1: sequence and hold, read and round-tripped" begin
    mktempdir() do dir
        _st_streaks(dir)
        plain = read_deck(_st_write(dir, "plain.toml", _st_text(dir)))
        @test !haskey(plain.sections["optimizer"]["staged"], "sequence")
        @test !haskey(plain.sections["optimizer"]["staged"], "hold")
        text = _sq_text(dir, Dict("sequence" => ["epw", "iaw"], "hold" => Dict("iaw" => ["logn", "T_e", "m_e"])))
        deck = read_deck(_st_write(dir, "seq.toml", text))
        st = deck.sections["optimizer"]["staged"]
        @test st["sequence"] == Any["epw", "iaw"]
        @test st["hold"] == Dict{String,Any}("iaw" => Any["logn", "T_e", "m_e"])
        problem, optimizer, _ = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)
        @test occursin("sequence = [\"epw\", \"iaw\"]", eff)
        @test read_deck(_st_write(dir, "eff.toml", eff)) == deck
        reversed = read_deck(_st_write(dir, "rev.toml", _sq_text(dir, Dict("sequence" => ["iaw", "epw"]))))
        @test reversed.sections["optimizer"]["staged"]["sequence"] == Any["iaw", "epw"]
    end
end

# ------------------------------------------------------------------------------------------
# SQ2: the refusals -- read_deck's, and _staged_fit's before any stage runs
# ------------------------------------------------------------------------------------------

@testset "SQ2: the refusals" begin
    mktempdir() do dir
        _st_streaks(dir)
        rd(name, extra) = read_deck(_st_write(dir, name, _sq_text(dir, extra)))
        @test_throws "read_deck: \"optimizer.staged.sequence\" must name every channel of \"channels\" once, in the order the stages run; got [\"epw\"] for channels [\"epw\", \"iaw\"]" rd(
            "a.toml", Dict("sequence" => ["epw"]))
        @test_throws "must name every channel of \"channels\" once" rd("b.toml", Dict("sequence" => ["epw", "epw"]))
        @test_throws "must name every channel of \"channels\" once" rd("c.toml", Dict("sequence" => ["epw", "xyz"]))
        @test_throws "read_deck: \"optimizer.staged.sequence\" must be a non-empty list of channel names" rd(
            "d.toml", Dict("sequence" => String[]))
        @test_throws "read_deck: \"optimizer.staged.hold\" needs \"sequence\"" rd("e.toml",
            Dict("hold" => Dict("iaw" => ["logn"])))
        @test_throws "read_deck: \"optimizer.staged.hold\" names stage \"xyz\", which is not a channel of \"sequence\"" rd(
            "f.toml", Dict("sequence" => ["epw", "iaw"], "hold" => Dict("xyz" => ["logn"])))
        @test_throws "read_deck: \"optimizer.staged.hold.iaw\" names \"nope\", which is not a variable a stage fits" rd(
            "g.toml", Dict("sequence" => ["epw", "iaw"], "hold" => Dict("iaw" => ["nope"])))
        @test_throws "read_deck: \"optimizer.staged.hold.iaw\" must be a non-empty list of variable names" rd(
            "h.toml", Dict("sequence" => ["epw", "iaw"], "hold" => Dict("iaw" => String[])))
        sp = _sh_start_file(dir)
        held = _sh_tables(_sq_text(dir, Dict("sequence" => ["epw", "iaw"], "hold" => Dict("iaw" => ["T_e"]))); hold=(sp, ["T_e"]))
        @test_throws "read_deck: \"optimizer.staged.hold.iaw\" names \"T_e\", which is not a variable a stage fits" read_deck(
            _st_write(dir, "i.toml", held))

        # the refusals that need the bindings: _staged_fit's, raised before any stage is fitted
        first_holds = read_deck(_st_write(dir, "j.toml", _sq_text(dir, Dict("sequence" => ["epw", "iaw"], "hold" => Dict("epw" => ["logn"])))))
        @test_throws "fit: [optimizer.staged] hold: stage \"epw\" holds \"logn\", which no earlier stage of the sequence fits" _staged_fit(first_holds)
        reach = _sh_edit(_sq_text(dir, Dict("sequence" => ["epw", "iaw"], "hold" => Dict("epw" => ["s_iaw"])))) do raw
            raw["channel"]["iaw"]["instrument"]["sensitivity"]["scale"] = "s_iaw"
            raw["fit"]["variables"]["s_iaw"] = Dict("bounds" => [1.0e-8, 1.0e-4], "init" => 1.0e-6)
        end
        @test_throws "fit: [optimizer.staged] hold: stage \"epw\" holds \"s_iaw\", which does not reach its channel's model" _staged_fit(
            read_deck(_st_write(dir, "k.toml", reach)))
        from_held = _sq_text(dir, Dict("sequence" => ["iaw", "epw"], "hold" => Dict("epw" => ["T_D"])))
        from_held = _sh_edit(raw -> raw["optimizer"]["staged"]["start_from"] = Dict("T_D" => "epw"), from_held)
        @test_throws "fit: [optimizer.staged] start_from takes \"T_D\" from stage \"epw\", which holds it" _staged_fit(
            read_deck(_st_write(dir, "l.toml", from_held)))
    end
end
