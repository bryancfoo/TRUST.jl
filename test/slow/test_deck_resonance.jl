using Test
using TRUST

# Step 4.F.2's slow row (Phase 4.F spec D17, D24), gated "resonance width (slow)": RW6, fit(deck)
# itself, a fit and so in the slow tier. fit(deck) pins BLAS to one thread for the rest of the
# process (conventions.md 6.6); this file restores the count it found, as TS2 does.

include(joinpath(@__DIR__, "..", "fixtures", "sampling_fixture.jl"))

const _RWS_NBLAS = pin_blas_threads()
pin_blas_threads(_RWS_NBLAS)

# ------------------------------------------------------------------------------------------
# RW6: fit(deck) on the one-bin deck at bin 25's fitted values
# ------------------------------------------------------------------------------------------

@testset "RW6: fit(deck) warns, and changes nothing" begin
    mktempdir() do dir
        _rw_streak(dir)
        path = joinpath(dir, "f.toml")
        write(path, _rw_deck_text(dir))
        deck = read_deck(path)
        result = @test_logs (:warn, _RW_SHORT_MSG) match_mode=:any min_level=Base.CoreLogging.Warn fit(deck)
        problem, opt, start = TRUST.build_problem(deck)
        @test result.theta == solve(problem, opt; start=start).theta
        @test read_result(joinpath(dir, "out", "result.jld2")).fit.theta == result.theta
    end
end

pin_blas_threads(_RWS_NBLAS)
