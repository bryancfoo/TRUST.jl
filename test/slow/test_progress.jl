using Test
using OTIS

# Step 4.K.1's slow row (Phase 4.K spec D13), gated "progress (slow)": PG9 fits the staged fixture's
# two-channel deck twice through fit(deck), with the progress lines off and on, so it is kept out of the
# phase's share of the fast suite (conventions.md 8.6). The fast rows are
# test/inversion/test_progress.jl's. fit(deck) pins BLAS to one thread for the rest of the process
# (conventions.md 6.6); this file restores the count it found, as test_deck_staged.jl does.

include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "progress_fixture.jl"))

const _PGS_NBLAS = pin_blas_threads()
pin_blas_threads(_PGS_NBLAS)

# ------------------------------------------------------------------------------------------
# PG9: the staged deck's banners, each before its optimizer's header, and the fit unchanged
# ------------------------------------------------------------------------------------------

@testset "PG9: the staged deck's banners, and its fit unchanged" begin
    mktempdir() do dir
        _st_streaks(dir)
        on = _ST_STAGED * "\nlbfgs = { progress = 1 }\nlevenberg_marquardt = { progress = 1 }"
        deck0 = read_deck(_st_write(dir, "off.toml", _st_text(dir)))
        deck1 = read_deck(_st_write(dir, "on.toml", _st_text(dir; optimizer=on)))
        out0, r0 = _pg_capture(() -> fit(deck0))
        out1, r1 = _pg_capture(() -> fit(deck1))
        @test isempty(out0)
        _pg_test_bitwise(r1, r0)

        lines = _pg_lines(out1)
        k = findall(l -> startswith(l, "progress: "), lines)
        @test lines[k] == ["progress: stage \"epw\"", "progress: stage \"iaw\"", "progress: joint"]
        @test k[1] == 1
        @test startswith(lines[k[1]+1], "progress lbfgs: start ") && startswith(lines[k[2]+1], "progress lbfgs: start ")
        @test startswith(lines[k[3]+1], "progress levenberg_marquardt: start ")
        @test count(l -> startswith(l, "progress lbfgs: done "), lines) == 2
        @test count(l -> match(_PG_DONE, l) !== nothing, lines) == 3
        # polish prints nothing: the joint fit's final line is the last
        @test match(_PG_DONE, lines[end])[:name] == "levenberg_marquardt"
        @test all(l -> startswith(l, "progress: ") || occursin(": start npixels=", l) ||
                       match(_PG_LINE, l) !== nothing || match(_PG_DONE, l) !== nothing, lines)
    end
end

pin_blas_threads(_PGS_NBLAS)
