using Test
using TRUST
using TRUST: _epw_resonance_widths, _recommended_subsamples, _resonance_widths,
    _narrow_resonance_entries, _warn_narrow_resonance, RESONANCE_WIDTH_WARN

# Step 4.F.2: the resonance-width check (Phase 4.F spec D13-D16, OQ-53, the user's decision of
# 2026-09-30). RW1-RW5, gate "resonance width", no fit; RW6, fit(deck) itself, is in the slow tier
# (test/slow/test_deck_resonance.jl). After its other warnings fit(deck) measures each bin's EPW resonance
# against the wavelength pixel and warns below RESONANCE_WIDTH_WARN, naming the bins and the count to
# refit with; a report that changes nothing.

include(joinpath(@__DIR__, "..", "fixtures", "sampling_fixture.jl"))

const _RW_IAW = collect(range(262.0e-9, 264.5e-9; length=256))  # the two-channel IAW window

# ------------------------------------------------------------------------------------------
# RW1: Phase 4.E's bin 25 -- one resonance, located, and measured against a finer grid
# ------------------------------------------------------------------------------------------

@testset "RW1: the narrow line, located and measured" begin
    p = _rw_gn(5.4658, 467.6205, 19.793557, 497.4972)
    w = _epw_resonance_widths(p, _RW_PROBE, _RW_COLL, _RW_EPW, true)
    @test length(w) == 1

    # the locator: Re(eps) changes sign twice on the pixel axis; one pair lies above omega_pe
    omega_pe = sqrt(p.n_e * ELEMENTARY_CHARGE^2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS))
    ko = [wavevector(_RW_PROBE, _RW_COLL, l, p) for l in _RW_EPW]
    re = [real(dielectric(p, k[1], k[2]).epsilon) for k in ko]
    flips = [l for l in 1:255 if (re[l] < 0) != (re[l+1] < 0)]
    @test flips == [145, 187]
    @test abs(ko[145][2]) > omega_pe && abs(ko[146][2]) > omega_pe
    @test abs(ko[187][2]) < omega_pe

    # the width against the same half-maximum rule on a grid sixteen times finer
    dl = _RW_EPW[146] - _RW_EPW[145]
    g = range(_RW_EPW[145] - 8dl, _RW_EPW[146] + 8dl; length=17 * 1024 + 1)
    S = spectrum(p, _RW_PROBE, _RW_COLL, collect(g))
    k = argmax(S)
    a = findlast(i -> S[i] <= S[k] / 2, 1:k)
    b = findfirst(i -> S[i] <= S[k] / 2, k:length(S)) + k - 1
    oracle = (g[b] - g[a]) / dl
    @info "RW1: the line's width in pixels" package = w[1] fine_grid = oracle
    @test abs(w[1] - oracle) <= 1e-2 * oracle
    @test w[1] < RESONANCE_WIDTH_WARN
end

# ------------------------------------------------------------------------------------------
# RW2: a resolved line, no resonance, and the IAW window
# ------------------------------------------------------------------------------------------

@testset "RW2: a resolved line, no resonance, the IAW window" begin
    resolved = _epw_resonance_widths(_rw_gn(4.6111, 520.0, 19.81, 497.0), _RW_PROBE, _RW_COLL, _RW_EPW, true)
    @test length(resolved) == 1
    @test resolved[1] >= RESONANCE_WIDTH_WARN
    maxwellian = Plasma(n_e_cm3=3.198e19, electrons=(electrons(T_eV=1303.4),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0),))
    @test isempty(_epw_resonance_widths(maxwellian, _RW_PROBE, _RW_COLL, _RW_EPW, true))
    @test isempty(_epw_resonance_widths(_rw_gn(5.4658, 467.6205, 19.793557, 497.4972), _RW_PROBE, _RW_COLL, _RW_IAW, true))
end

# ------------------------------------------------------------------------------------------
# RW3: the count the warning recommends
# ------------------------------------------------------------------------------------------

@testset "RW3: the recommended count" begin
    @test [_recommended_subsamples(w) for w in (0.03, 0.1983, 0.25, 0.27, 0.5, 0.99, 1.0, 1.24, Inf)] ==
          [128, 16, 8, 8, 4, 4, 1, 1, 1]
end

# ------------------------------------------------------------------------------------------
# RW4-RW5: the warning, on a one-bin EPW deck at bin 25's fitted values (no fit)
# ------------------------------------------------------------------------------------------

# A FitResult at the deck's inits, no fit: the check reads only result.experiments and the names.
function _rw_result(dir::AbstractString, name::AbstractString, text::AbstractString)
    path = joinpath(dir, name)
    write(path, text)
    problem, _, start = TRUST.build_problem(read_deck(path))
    o = objective(problem, start)
    row = (iteration=0.0, objective=o, chi2=o, R=0.0)
    return TRUST._build_result(problem, start, o, [row], true, :f_tolerance, 0, 1, 0, 0.0, "none")
end

@testset "RW4: the warning's text, shorthand and channel form" begin
    mktempdir() do dir
        _rw_streak(dir)
        r = _rw_result(dir, "a.toml", _rw_deck_text(dir))
        entries = @test_logs (:warn, _RW_SHORT_MSG) min_level=Base.CoreLogging.Warn _warn_narrow_resonance(r)
        @test length(entries) == 1
        @test entries[1].channel == "main" && entries[1].bin == 1 && entries[1].subsamples == 1
        @test isapprox(entries[1].fwhm_px, 0.1983; atol=5e-4)

        rc = _rw_result(dir, "c.toml", _rw_deck_text(dir; form=:channel))
        msg_c = replace(_RW_SHORT_MSG, "  main:" => "  epw:", "[instrument]" => "[channel.epw.instrument]")
        @test_logs (:warn, msg_c) min_level=Base.CoreLogging.Warn _warn_narrow_resonance(rc)
    end
end

@testset "RW5: silent once the channel samples finely enough" begin
    mktempdir() do dir
        _rw_streak(dir)
        r16 = _rw_result(dir, "a16.toml", _rw_deck_text(dir; line="pixel_integration = { subsamples = 16 }"))
        @test isempty(@test_logs min_level=Base.CoreLogging.Warn _warn_narrow_resonance(r16))
        r8 = _rw_result(dir, "a8.toml", _rw_deck_text(dir; line="pixel_integration = { subsamples = 8 }"))
        msg8 = replace(_RW_SHORT_MSG, "1 per pixel now" => "8 per pixel now")
        @test_logs (:warn, msg8) min_level=Base.CoreLogging.Warn _warn_narrow_resonance(r8)
        rmax = _rw_result(dir, "m.toml", replace(_rw_deck_text(dir; line="pixel_integration = { subsamples = 1 }"),
            "shape = { type = \"GeneralizedNormal\", m = \"m_e\" }" => ""))
        @test all(>=(RESONANCE_WIDTH_WARN), _resonance_widths(rmax)[1])
        @test isempty(@test_logs min_level=Base.CoreLogging.Warn _warn_narrow_resonance(rmax))
    end
end
