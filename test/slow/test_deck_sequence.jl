using Test
using OTIS
using OTIS: _stage_deck, _build_problem

# Step 4.I.3's slow rows (Phase 4.I spec D22 to D26), gated "sequential staging (slow)": SQ3 to SQ5
# fit the staged fixture's two-channel deck (test/fixtures/staged_fixture.jl, ST3's) with a sequence.
# The fast rows are test/config/test_deck_sequence.jl's. fit(deck) pins BLAS to one thread; this file
# restores the count it found, as TS2 does.

include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "start_hold_fixture.jl"))

const _SQS_NBLAS = pin_blas_threads()
pin_blas_threads(_SQS_NBLAS)

_sqs_text(dir, extra::AbstractDict) = _sh_edit(raw -> merge!(raw["optimizer"]["staged"], extra), _st_text(dir))

# ------------------------------------------------------------------------------------------
# SQ3: the IAW stage holds the EPW stage's electrons, bitwise (acceptance (h))
# ------------------------------------------------------------------------------------------

@testset "SQ3: a later stage holds an earlier stage's series" begin
    mktempdir() do dir
        _st_streaks(dir)
        deck = read_deck(_st_write(dir, "d.toml",
            _sqs_text(dir, Dict("sequence" => ["epw", "iaw"], "hold" => Dict("iaw" => ["logn", "T_e", "m_e"])))))
        result = fit(deck)
        r = read_result(joinpath(dir, "out", "result.jld2"))
        epw, iaw = r.stages.channels["epw"], r.stages.channels["iaw"]
        @test r.stages.sequence == ["epw", "iaw"]
        @test r.stages.names == ["epw", "iaw"]
        @test iaw.held == ["T_e", "logn", "m_e"] && isempty(epw.held)
        for v in ("logn", "T_e", "m_e")
            @test all(iaw.variables[v].values .=== epw.variables[v].values)
            @test iaw.variables[v].vary == 0
        end
        @test length(iaw.theta) == 4                            # T_D's four bins alone
        # the IAW stage, assembled independently: the electrons held at the EPW stage's values, T_D
        # (fitted by both) warm-started from the EPW stage's
        held = Dict{String,Vector{Float64}}(v => epw.variables[v].values for v in ("logn", "T_e", "m_e"))
        p, opt, st = _build_problem(_stage_deck(deck, 2, "lbfgs", String[]); held=held,
            start=Dict{String,Any}("T_D" => epw.variables["T_D"].values))
        @test solve(p, opt; start=st).theta == iaw.theta
        # the joint start: the electrons from the EPW stage, T_D from the IAW stage (start_from)
        @test r.stages.sources == Dict("logn" => "epw", "T_e" => "epw", "m_e" => "epw", "T_D" => "iaw")
        @test r.fit.start == r.stages.start
        @test result.theta == r.fit.theta
    end
end

# ------------------------------------------------------------------------------------------
# SQ4: with a sequence and no hold, a later stage starts from the earlier stage's series
# ------------------------------------------------------------------------------------------

@testset "SQ4: the warm start" begin
    mktempdir() do dir
        _st_streaks(dir)
        deck = read_deck(_st_write(dir, "d.toml", _sqs_text(dir, Dict("sequence" => ["epw", "iaw"]))))
        fit(deck)
        r = read_result(joinpath(dir, "out", "result.jld2"))
        epw, iaw = r.stages.channels["epw"], r.stages.channels["iaw"]
        @test isempty(iaw.held)
        warm = Dict{String,Any}(v => epw.variables[v].values for v in ("logn", "T_e", "m_e", "T_D"))
        sdeck = _stage_deck(deck, 2, "lbfgs", String[])
        p, opt, st = _build_problem(sdeck; start=warm)
        @test st == flatten(p.pspec, collect(warm))
        @test st != OTIS.build_problem(sdeck)[3]
        @test solve(p, opt; start=st).theta == iaw.theta
        # every variable both stages fit takes its joint start from the last of them, the IAW stage
        @test r.stages.sources == Dict("logn" => "iaw", "T_e" => "iaw", "m_e" => "iaw", "T_D" => "iaw")
    end
end

# ------------------------------------------------------------------------------------------
# SQ5: a stage and the final fit at their iteration cap each warn once (acceptance (i))
# ------------------------------------------------------------------------------------------

@testset "SQ5: the cap warning, for each stage and the final fit" begin
    mktempdir() do dir
        _st_streaks(dir)
        text = _sh_edit(_st_text(dir)) do raw
            raw["optimizer"]["staged"]["stage_procedure"] = "levenberg_marquardt"
            raw["optimizer"]["levenberg_marquardt"] = Dict("iterations" => 1)
        end
        deck = read_deck(_st_write(dir, "cap.toml", text))
        result = @test_logs (:warn, r"^fit: stage \"epw\" stopped at its iteration cap after 1 iterations") (:warn,
            r"^fit: stage \"iaw\" stopped at its iteration cap after 1 iterations") (:warn,
            r"^fit: the final fit stopped at its iteration cap after 1 iterations") match_mode=:any min_level=Base.CoreLogging.Warn fit(deck)
        @test result.status === :iterations
        r = read_result(joinpath(dir, "out", "result.jld2"))
        @test r.stages.channels["epw"].status == "iterations" && r.stages.channels["iaw"].status == "iterations"
    end
end

pin_blas_threads(_SQS_NBLAS)
