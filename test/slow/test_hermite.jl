using Test
using OTIS
using OTIS: _hermite_lobe, _warn_hermite_lobe, HERMITE_NEGATIVE_MASS_WARN
import Random

# Step 5.A.1's slow rows (Phase 5.A spec D12), gated "Hermite (slow)": HE12 fits the recovery's EPW streak from
# a skew-normal and from a flattened electron distribution with HermiteExpansion's c3 and c4 free (OQ-66 (l)), and
# checks fit(deck)'s lobe warning on the two fits (OQ-66 (h)); HE15 fits a deck with the shape through fit(deck)
# and checks that it warns. In the slow tier because each compiles the fit path for the new shape type, about
# 26 s together (the user's decision of 2026-10-16, Phase 5.A summary U3). The fast rows are
# test/distributions/test_hermite.jl's. fit(deck) pins BLAS to one thread; this file restores the count it found.

include(joinpath(@__DIR__, "..", "fixtures", "hermite_fixture.jl"))

const _HES_NBLAS = pin_blas_threads()
pin_blas_threads(_HES_NBLAS)

function _he_recovery(truth_f)
    problem, _, start = _he_recovery_problem(truth_f)
    return solve(problem, LevenbergMarquardt(); start = start)
end

# ------------------------------------------------------------------------------------------
# HE12: the P1-sized recovery, and the warning on its two fits (OQ-66 (l), (h))
# ------------------------------------------------------------------------------------------

@testset "HE12: the P1-sized recovery" begin
    # Deliberately loose (spec D12): the expansion is truncated (noise-free skewness -2.8 %) and the data noisy;
    # over 20 noise seeds the skewness error read at most 7.4 % and the excess kurtosis 0.029 to 0.051,
    # GeneralizedNormal(3.0)'s c4 at most -0.0076.
    fit_sn = _he_recovery(SkewNormal(1.0))
    fit_gn = _he_recovery(GeneralizedNormal(3.0))
    sn = skewness(SkewNormal(1.0))
    skew_hat = 12 * sqrt(2) * fit_sn.values["c3_e"][1]
    exk_hat = 96 * fit_sn.values["c4_e"][1]
    @info "HE12" skew_hat sn (skew_hat - sn) / sn exk_hat excess_kurtosis(SkewNormal(1.0)) fit_sn.values["T_e"][1] fit_gn.values["c4_e"][1] fit_sn.iterations fit_gn.iterations
    flush(stderr)
    @test fit_sn.converged
    @test abs(skew_hat - sn) / sn < 0.10
    @test exk_hat > 0
    @test fit_gn.converged
    @test fit_gn.values["c4_e"][1] < 0
    # the warning on the fits: the flattened core's is negative past the threshold, the skew-normal's is not
    gn = HermiteExpansion(c3 = fit_gn.values["c3_e"][1], c4 = fit_gn.values["c4_e"][1])
    msg = _HE_WARN_HEAD * "\n  electrons.e: bin 1 (largest negative mass 0.00232)"
    entries = @test_logs (:warn, msg) min_level = Base.CoreLogging.Warn _warn_hermite_lobe(fit_gn, "the final fit")
    @test length(entries) == 1
    @test entries[1].negative_mass === _hermite_lobe(gn).negative_mass
    @test entries[1].negative_mass > HERMITE_NEGATIVE_MASS_WARN
    @test isempty(@test_logs min_level = Base.CoreLogging.Warn _warn_hermite_lobe(fit_sn, "the final fit"))
end

# ------------------------------------------------------------------------------------------
# HE15: fit(deck) warns (spec D10)
# ------------------------------------------------------------------------------------------

@testset "HE15: fit(deck) warns on a negative lobe" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _he11_write_streak(datafile)
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _he11_deck(datafile, joinpath(dir, "out")))
        # the data's c3 = 0.04 with the deck's c5 = 0.002 is negative over 5.5e-3 of the distribution in both bins
        msg = _HE_WARN_HEAD * "\n  electrons.e: bins 1, 2 (largest negative mass 0.00547)"
        r = @test_logs (:warn, msg) match_mode = :any min_level = Base.CoreLogging.Warn fit(deckpath)
        @test r.converged
        @test isfile(joinpath(dir, "out", "result.jld2"))
    end
    pin_blas_threads(_HES_NBLAS)
end
