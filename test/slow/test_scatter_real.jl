using Test
using OTIS

# Step 4.N.1's slow rows (Phase 4.N spec D8; OQ-64 (d) to (f)), gated "scatter real deck (slow)": SI5 to
# SI7 scatter the readable fixture's RD deck (two cameras, u_D on the interval "late" = coarse bins 2 and
# 3 of 3, T_e held at a two-knot series file, two dependents, two scalars, a frozen variable), whose
# one-bin window problems nothing in the fast suite compiles. The fast rows are
# test/config/test_scatter_real.jl's. fit(deck) pins BLAS to one thread for the rest of the process
# (conventions.md 6.6); this file restores the count it found, as test_readable.jl does.

include(joinpath(@__DIR__, "..", "fixtures", "readable_fixture.jl"))

const _SIS_NBLAS = pin_blas_threads()
pin_blas_threads(_SIS_NBLAS)

const _SIS_DIR = mktempdir()
const _SIS_PA = _rd_streak_a(joinpath(_SIS_DIR, "a.jld2"))
const _SIS_PB = _rd_streak_b(joinpath(_SIS_DIR, "b.jld2"); masked=2)
const _SIS_HOLD = write_start(joinpath(_SIS_DIR, "hold.jld2"); axis_ns=[0.0, 1.2],
    values=Dict("T_e" => [390.0, 410.0]), units=Dict("T_e" => "T_eV"), source="si_test")
# (e)'s short file (Phase 4.N spec D8): knots just outside the first and last coarse bins' times (0.2 and
# 1.0 ns), so that the coarse fit builds, and the two windows at each end (0.05, 0.15, 1.05 and 1.15 ns)
# outside it
const _SIS_SHORT = write_start(joinpath(_SIS_DIR, "short.jld2"); axis_ns=[0.19, 1.01],
    values=Dict("T_e" => [392.0, 408.0]), units=Dict("T_e" => "T_eV"), source="si_test")

# The RD deck with the hold file `hold`, its problem and its polished fit (fit(deck), then polish).
function _sis_fit(name, hold)
    deck = read_deck(_rd_write(_SIS_DIR, "$name.toml",
        _rd_deck_text(_SIS_PA, _SIS_PB, hold, joinpath(_SIS_DIR, "out_$name"))))
    problem = first(OTIS.build_problem(deck))
    return deck, problem, polish(problem, _rd_fit(deck)).result
end

const _SIS_DECK, _SIS_PROBLEM, _SIS_POL = _sis_fit("rd", _SIS_HOLD)
const _SIS_SCATTER = slice_scatter(_SIS_DECK, _SIS_POL)
const _SIS_DECK_SHORT, _, _SIS_POL_SHORT = _sis_fit("short", _SIS_SHORT)
const _SIS_SCATTER_SHORT = slice_scatter(_SIS_DECK_SHORT, _SIS_POL_SHORT)

# The RD deck cut to the one window [lo, hi] ns (Phase 4.M's RD9 route): no interval and no regularizer,
# u_D frozen at its init when `uD_free` is false, T_e held at `te` through a one-value series file, and
# the scalars s_b and bg frozen at the polished fit's values, as slice_scatter holds them (Pitfall 42).
function _sis_window_text(lo, hi, te, uD_free, dir)
    hold = write_start(joinpath(dir, "te.jld2"); axis_ns=[0.0], values=Dict("T_e" => [te]),
        units=Dict("T_e" => "T_eV"), source="si_test")
    text = _rd_deck_text(_SIS_PA, _SIS_PB, hold, joinpath(dir, "out"))
    uD = uD_free ? "u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }" :
         "u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true, vary = false }"
    return replace(text, _RD_REGIONS => "[[fit.regions]]\nt_range_ns = [$lo, $hi]\nnbins = 1",
        "[fit.intervals.late]\nt_range_ns = [0.5, 1.2]\n" => "",
        "u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true, interval = \"late\" }" => uD,
        "s_b = { bounds = [1.0e-6, 4.0e-6], init = 2.2e-6 }" =>
            "s_b = { bounds = [1.0e-6, 4.0e-6], init = $(_SIS_POL.values["s_b"][1]), vary = false }",
        "bg = { bounds = [26.0, 40.0], init = 30.0 }" =>
            "bg = { bounds = [26.0, 40.0], init = $(_SIS_POL.values["bg"][1]), vary = false }",
        "[regularizer]\nweight = 1.0\nterms = [\n  { select = [\"T_D\"], operator = \"D2\", normalization = \"relative\" },\n]\n" => "")
end

# ------------------------------------------------------------------------------------------
# SI5: the RD deck scatters; u_D NaN exactly outside its interval, its statistics NaN there (OQ-64 (d))
# ------------------------------------------------------------------------------------------

@testset "SI5: the RD deck scatters: u_D frozen outside its interval, NaN in the record there" begin
    sc = _SIS_SCATTER
    @test sc.m == [4]
    @test sc.coarse_bin == repeat(1:3, inner=4)
    @test Set(keys(sc.values)) == Set(keys(sc.p16)) == Set(["T_D", "u_D"])
    @test Set(keys(sc.held)) == Set(["T_e"])
    bin1 = sc.coarse_bin .== 1
    @test isnan.(sc.values["u_D"]) == bin1
    @test !any(sc.near_bound["u_D"][bin1])
    @test all(isfinite, sc.values["T_D"])
    for stat in (sc.s_stat, sc.p16, sc.median, sc.p84)
        @test isnan(stat["u_D"][1]) && all(isfinite, stat["u_D"][2:3])
        @test all(isfinite, stat["T_D"])
    end
    @test sc.n_near_bound["u_D"][1] == 0
    # a hold file without the held name: refused with _series_values' own message
    bad = write_start(joinpath(_SIS_DIR, "bad.jld2"); axis_ns=[0.0, 1.2], values=Dict("T_x" => [1.0, 2.0]),
        source="si_test")
    deck_bad = read_deck(_rd_write(_SIS_DIR, "bad.toml", _rd_deck_text(_SIS_PA, _SIS_PB, bad,
        joinpath(_SIS_DIR, "out_bad"))))
    @test_throws "build_problem: [fit.hold] names \"T_e\", which file \"$(_rd_path(bad))\" does not hold; it holds [\"T_x\"]" slice_scatter(
        deck_bad, _SIS_POL)
end

# ------------------------------------------------------------------------------------------
# SI6: the held values, interpolated at each window's time; beyond the file, the coarse bin's (OQ-64 (e))
# ------------------------------------------------------------------------------------------

@testset "SI6: T_e held at the series at each window's time, the coarse bin's value beyond the file" begin
    sc = _SIS_SCATTER
    hold = read_start(_SIS_HOLD)
    for j in 1:12
        @test sc.held["T_e"][j] === OTIS._interpolate_series(hold.axis, hold.values["T_e"], sc.t_fine[j])
    end
    sc2, r2 = _SIS_SCATTER_SHORT, _SIS_POL_SHORT
    short = read_start(_SIS_SHORT)
    outside = [j for j in 1:12 if !(first(short.axis) <= sc2.t_fine[j] <= last(short.axis))]
    @test outside == [1, 2, 11, 12]
    for j in 1:12
        k = sc2.coarse_bin[j]
        want = j in outside ? r2.values["T_e"][k] :
               OTIS._interpolate_series(short.axis, short.values["T_e"], sc2.t_fine[j])
        @test sc2.held["T_e"][j] === want
    end
    @test sc2.held["T_e"][1] === sc2.held["T_e"][2] != sc2.held["T_e"][3]
end

# ------------------------------------------------------------------------------------------
# SI7: the readable file of the scatter: data, models, chi2, the new datasets, otis-readable-2 (OQ-64 (f))
# ------------------------------------------------------------------------------------------

@testset "SI7: write_readable with the scatter: the windows' data and models, the quantiles, held, the version" begin
    sc, s = _SIS_SCATTER, _SIS_DECK.sections
    chan_list = OTIS._channel_list(s)
    raw, _, _ = OTIS._read_channels(s, chan_list, true)
    regions = OTIS._build_averaging_regions(s["fit_regions"])
    fine = [AveragingRegion(rg.lo, rg.hi, rg.nbins * sc.m[i], "") for (i, rg) in enumerate(regions)]
    windows = OTIS._average_channels(raw, chan_list, fine)
    mktempdir() do dir
        path = write_readable(joinpath(dir, "sc.h5"), _SIS_POL, _SIS_PROBLEM; scatter=sc)
        OTIS.jldopen(path, "r") do f
            @test f["layout_version"] == "otis-readable-2"
            @test Set(first.(_rd_leaves(f))) == _rd_expected_keys(_SIS_POL, ("a", "b"); deck=false, gn=false,
                scatter=sc)
            for (c, name) in enumerate(("a", "b"))
                @test isequal(f["scatter/channels/$name/data"], windows[c].image)
                @test isequal(f["scatter/channels/$name/sigma"], windows[c].sigma)
                @test f["scatter/channels/$name/mask"] == UInt8.(windows[c].mask)
            end
            # each window's chi2 from its data, model, sigma and mask (both weights 1)
            ca, _ = _rd_chi2_bins(f, "scatter/channels/a")
            cb, _ = _rd_chi2_bins(f, "scatter/channels/b")
            @test isapprox(f["scatter/chi2"], ca .+ cb; rtol=1e-12)
            for name in ("T_D", "u_D")
                @test isequal(f["scatter/params/$name"], sc.values[name])
                @test isequal(f["scatter/sigma_stat/$name"], sc.s_stat[name])
                @test isequal(f["scatter/p16/$name"], sc.p16[name])
                @test isequal(f["scatter/median/$name"], sc.median[name])
                @test isequal(f["scatter/p84/$name"], sc.p84[name])
            end
            @test isequal(f["scatter/held/T_e"], sc.held["T_e"])
            # the model: window 2 (bin 1, u_D at its init) and window 7 (bin 2, u_D fitted), each
            # re-evaluated at its own values through its own one-window deck, T_e at its held value
            for (j, lo, hi, uD_free) in ((2, 0.1, 0.2, false), (7, 0.6, 0.7, true))
                wdir = mkpath(joinpath(dir, "w$j"))
                text = _sis_window_text(lo, hi, sc.held["T_e"][j], uD_free, wdir)
                p = first(OTIS.build_problem(read_deck(_rd_write(wdir, "w.toml", text))))
                vals = uD_free ? ["T_D" => sc.values["T_D"][j], "u_D" => sc.values["u_D"][j]] :
                       ["T_D" => sc.values["T_D"][j]]
                theta = flatten(p.pspec, vals)
                for (c, name) in enumerate(("a", "b"))
                    ch = OTIS._channel_problem(p, c)
                    m = measure(unflatten(ch.experiment, p.pspec, theta; channel=c))
                    @test isapprox(f["scatter/channels/$name/model"][:, j], vec(m); rtol=1e-12)
                end
            end
        end
    end
end

pin_blas_threads(_SIS_NBLAS)
