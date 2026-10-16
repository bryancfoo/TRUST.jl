using Test
using OTIS

# Step 4.M.1: the readable fit output (Phase 4.M spec D5 to D16; OQ-63, decided 2026-10-14). RD1 to
# RD10, gate "readable", every row on synthetic data (Phase E's A29). fit(deck) writes readable.h5
# (otis-readable-2) beside result.jld2 unless [output] readable = false, and write_readable writes the
# same file from a script, with the Laplace sigma and the scatter's windows when given. These rows fit
# test_uncertainty.jl's SX deck on noisy streaks (camera b's first two wavelength rows masked), whose
# problem types the "uncertainty" gate has already compiled; the RD deck's rows (an interval, a held
# series, dependents, scalars, a frozen variable, a near-bound entry) are test/slow/test_readable.jl's.
# RD14 is Phase 4.L's (spec D9, OQ-62 item 8): a key-off fit removes an earlier run's readable.h5.
# Phase 4.N (spec D9) changed RD7's two version literals to otis-readable-2 and RD10's field list to the
# twenty fields ScatterResult now has; the scatter's new datasets are in the fixture's key set.

include(joinpath(@__DIR__, "..", "fixtures", "readable_fixture.jl"))

const _RD_DIR = mktempdir()
const _RD_SA = _rd_streak_a(joinpath(_RD_DIR, "a.jld2"); seed=41)
const _RD_SB = _rd_streak_b(joinpath(_RD_DIR, "b.jld2"); seed=42, masked=2)
const _RD_ON = joinpath(_RD_DIR, "out_on")
const _RD_OFF = joinpath(_RD_DIR, "out_off")
const _RD_TEXT = _rd_sx_text(_RD_SA, _RD_SB, _RD_ON)
const _RD_DECK = read_deck(_rd_write(_RD_DIR, "on.toml", _RD_TEXT))
const _RD_DECK_OFF = read_deck(_rd_write(_RD_DIR, "off.toml", _rd_sx_text(_RD_SA, _RD_SB, _RD_OFF; readable=false)))
const _RD_RESULT = _rd_fit(_RD_DECK)
const _RD_RESULT_OFF = _rd_fit(_RD_DECK_OFF)
const _RD_PROBLEM = first(OTIS.build_problem(_RD_DECK))
const _RD_POL = polish(_RD_PROBLEM, _RD_RESULT).result
const _RD_LAP = laplace(_RD_PROBLEM, _RD_POL.theta)
const _RD_SCATTER = slice_scatter(_RD_DECK, _RD_POL)
const _RD_SINGLE_OUT = joinpath(_RD_DIR, "out_single")
const _RD_SINGLE_DECK = read_deck(_rd_write(_RD_DIR, "single.toml", _rd_single_text(_RD_SA, _RD_SINGLE_OUT)))
const _RD_SINGLE_RESULT = _rd_fit(_RD_SINGLE_DECK)

# ------------------------------------------------------------------------------------------
# RD1: the [output] key, its default, its refusal and the effective deck (OQ-63 (e))
# ------------------------------------------------------------------------------------------

@testset "RD1: the readable key, its default, its refusal and the effective deck" begin
    @test _RD_DECK.sections["output"] == Dict{String,Any}("dir" => _rd_path(_RD_ON), "readable" => true)
    @test _RD_DECK_OFF.sections["output"] == Dict{String,Any}("dir" => _rd_path(_RD_OFF), "readable" => false)
    for (value, shown) in (("\"yes\"", "\"yes\""), ("1", "1"), ("\"true\"", "\"true\""))
        path = _rd_write(_RD_DIR, "bad.toml", replace(_RD_TEXT, "[output]\n" => "[output]\nreadable = $value\n"))
        @test_throws "read_deck: key \"readable\" in section \"output\" must be true or false; got $shown" read_deck(path)
    end
    eff_on = read(joinpath(_RD_ON, "effective_deck.toml"), String)
    eff_off = read(joinpath(_RD_OFF, "effective_deck.toml"), String)
    @test _RD_TOML.parse(eff_on)["output"] == Dict("dir" => _rd_path(_RD_ON), "readable" => true)
    @test _RD_TOML.parse(eff_off)["output"] == Dict("dir" => _rd_path(_RD_OFF), "readable" => false)
    @test read_deck(_rd_write(_RD_DIR, "eff_on.toml", eff_on)) == _RD_DECK
    @test read_deck(_rd_write(_RD_DIR, "eff_off.toml", eff_off)) == _RD_DECK_OFF
end

# ------------------------------------------------------------------------------------------
# RD2: fit(deck) writes readable.h5 unless readable = false; result.jld2 the same either way (OQ-63 (e))
# ------------------------------------------------------------------------------------------

@testset "RD2: fit(deck) writes readable.h5 unless readable = false; result.jld2 the same either way" begin
    @test sort(readdir(_RD_ON)) == ["effective_deck.toml", "readable.h5", "result.jld2"]
    @test sort(readdir(_RD_OFF)) == ["effective_deck.toml", "result.jld2"]
    eff_on = read(joinpath(_RD_ON, "effective_deck.toml"), String)
    on = _rd_dict(joinpath(_RD_ON, "result.jld2"))
    off = _rd_dict(joinpath(_RD_OFF, "result.jld2"))
    rd = _rd_dict(joinpath(_RD_ON, "readable.h5"))
    @test rd["source"] == "fit"
    @test rd["deck"] == eff_on
    @test rd["fit/gn_decrease"] === on["fit/gn_decrease"]
    # U1 (the user, 2026-10-14): every dataset by name, bitwise, but the three clock datasets, and the
    # deck text differing only by its [output] table
    @test Set(keys(on)) == Set(keys(off))
    clock = ("meta/created", "meta/created_unix", "fit/elapsed")
    @test sort([k for k in keys(on) if !(k in clock) && k != "deck" && !isequal(on[k], off[k])]) == String[]
    @test on["deck"] == eff_on
    @test on["deck"] != off["deck"]
    @test replace(off["deck"], _rd_path(_RD_OFF) => _rd_path(_RD_ON), "readable = false" => "readable = true") == on["deck"]
end

# ------------------------------------------------------------------------------------------
# RD3: data, sigma and mask are the problem's, model the result file's, bitwise (OQ-63 (a))
# ------------------------------------------------------------------------------------------

@testset "RD3: data, sigma and mask the problem's, model the result file's, bitwise" begin
    rd = _rd_dict(joinpath(_RD_ON, "readable.h5"))
    re = _rd_dict(joinpath(_RD_ON, "result.jld2"))
    for (c, name) in enumerate(("a", "b"))
        ch = OTIS._channel_problem(_RD_PROBLEM, c)
        @test isequal(rd["channels/$name/data"], ch.data.image)
        @test isequal(rd["channels/$name/sigma"], ch.sigma)
        @test rd["channels/$name/mask"] == UInt8.(ch.mask)
        @test isequal(rd["channels/$name/model"], re["channels/$name/model"])
    end
    @test count(iszero, rd["channels/a/mask"]) == 0
    @test count(iszero, rd["channels/b/mask"]) == 6
end

# ------------------------------------------------------------------------------------------
# RD4: chi2 per bin over the fitted pixels, recomputed from the file (OQ-63 (b))
# ------------------------------------------------------------------------------------------

@testset "RD4: chi2 per bin over the fitted pixels, recomputed from data, model, sigma and mask" begin
    re = _rd_dict(joinpath(_RD_ON, "result.jld2"))
    OTIS.jldopen(joinpath(_RD_ON, "readable.h5"), "r") do f
        for name in ("a", "b")
            chi2, n = _rd_chi2_bins(f, "channels/$name")
            @test isapprox(f["channels/$name/chi2"], chi2; rtol=1e-12)
            @test f["channels/$name/npixels"] == n
            @test isapprox(f["channels/$name/chi2_total"], sum(chi2); rtol=1e-12)
            @test isapprox(f["channels/$name/chi2_total"], re["channels/$name/chi2"]; rtol=1e-12)
            @test sum(f["channels/$name/npixels"]) == re["channels/$name/npixels"]
            @test f["channels/$name/weight"] === re["channels/$name/weight"]
        end
    end
end

# ------------------------------------------------------------------------------------------
# RD5: the parameters, their units and flags, the axes and fit/ (OQ-63 (c))
# ------------------------------------------------------------------------------------------

@testset "RD5: params bitwise FitResult.values, units, flags, the axes and fit/" begin
    r = _RD_RESULT
    rd = _rd_dict(joinpath(_RD_ON, "readable.h5"))
    @test isempty(OTIS._near_bound_entries(r))
    for (name, unit) in (("T_D", "T_eV"), ("u_D", "u_kms"))
        @test rd["params/$name"] isa Vector{Float64}
        @test isequal(rd["params/$name"], r.values[name])
        @test rd["units/$name"] == unit
        @test rd["flags/$name"] == ["F", "F", "F"]
    end
    @test isequal(rd["t_ns"], t_from_si.(r.pspec.axis, :t_ns))
    @test isapprox(rd["t_ns"], r.pspec.axis .* 1e9; rtol=1e-15)
    for (c, name) in enumerate(("a", "b"))
        lam = r.experiments[c].lambda
        @test isequal(rd["channels/$name/lambda_nm"], lambda_from_si.(lam, :lambda_nm))
        @test isapprox(rd["channels/$name/lambda_nm"], lam .* 1e9; rtol=1e-15)
    end
    @test rd["fit/chi2"] === r.chi2
    @test rd["fit/chi2_per_n"] === r.chi2 / r.npixels
    @test rd["fit/R"] === r.R
    @test rd["fit/objective"] === r.objective
    @test rd["fit/status"] == string(r.status)
    @test rd["fit/converged"] === UInt8(r.converged)
    @test rd["fit/termination"] == string(r.termination)
    @test rd["fit/iterations"] === Int64(r.iterations)
    @test rd["fit/npixels"] === Int64(r.npixels)
    @test rd["fit/ntheta"] === Int64(r.pspec.ntheta)
end

# ------------------------------------------------------------------------------------------
# RD6: each channel under its own name; a single-channel fit's one channel is main (OQ-63 (d))
# ------------------------------------------------------------------------------------------

@testset "RD6: each channel under channels/<name>; a single-channel fit's is main" begin
    rd = _rd_dict(joinpath(_RD_ON, "readable.h5"))
    @test size(rd["channels/a/data"]) == (24, 3)
    @test size(rd["channels/b/data"]) == (16, 3)
    @test length(rd["channels/a/lambda_nm"]) == 24
    @test length(rd["channels/b/lambda_nm"]) == 16
    problem = first(OTIS.build_problem(_RD_SINGLE_DECK))
    OTIS.jldopen(joinpath(_RD_SINGLE_OUT, "readable.h5"), "r") do f
        @test collect(keys(f["channels"])) == ["main"]
        @test isequal(f["channels/main/data"], problem.data.image)
        @test size(f["channels/main/model"]) == (24, 12)
        @test !haskey(f, "data") && !haskey(f, "model") && !haskey(f, "lambda_nm")
        @test isequal(f["bin_lo_ns"], f["t_ns"]) && isequal(f["bin_hi_ns"], f["t_ns"])
    end
end

# ------------------------------------------------------------------------------------------
# RD7: read back through JLD2: the shape, the version, the key set, the bin windows (OQ-63 (f))
# ------------------------------------------------------------------------------------------

@testset "RD7: read back: the orientation, the version, the key set and the bin windows" begin
    OTIS.jldopen(joinpath(_RD_ON, "readable.h5"), "r") do f
        @test f["layout_version"] == "otis-readable-2"
        @test f["README"] isa String && startswith(f["README"], "otis-readable-2: ")
        @test size(f["channels/a/data"]) == (length(f["channels/a/lambda_nm"]), length(f["t_ns"]))
        @test Set(keys(f)) == Set(["layout_version", "README", "source", "deck", "t_ns", "bin_lo_ns",
            "bin_hi_ns", "channels", "params", "units", "flags", "fit"])
        leaves = _rd_leaves(f)
        @test Set(first.(leaves)) == _rd_expected_keys(_RD_RESULT, ("a", "b"); deck=true, gn=true)
        @test all(v isa _RD_PLAIN for v in last.(leaves))
        @test isapprox(f["bin_lo_ns"], [0.0, 0.4, 0.8]; atol=1e-12)
        @test isapprox(f["bin_hi_ns"], [0.4, 0.8, 1.2]; atol=1e-12)
        @test all(f["bin_lo_ns"] .<= f["t_ns"] .<= f["bin_hi_ns"])
    end
    OTIS.jldopen(joinpath(_RD_SINGLE_OUT, "readable.h5"), "r") do f
        @test Set(first.(_rd_leaves(f))) == _rd_expected_keys(_RD_SINGLE_RESULT, ("main",); deck=true, gn=true)
    end
    mktempdir() do dir
        path = write_readable(joinpath(dir, "all.h5"), _RD_POL, _RD_PROBLEM; deck="the deck",
            laplace=_RD_LAP, scatter=_RD_SCATTER)
        @test path == joinpath(dir, "all.h5")
        OTIS.jldopen(path, "r") do f
            @test Set(keys(f)) == Set(["layout_version", "README", "source", "deck", "t_ns", "bin_lo_ns",
                "bin_hi_ns", "channels", "params", "units", "flags", "fit", "laplace", "scatter"])
            leaves = _rd_leaves(f)
            @test Set(first.(leaves)) == _rd_expected_keys(_RD_POL, ("a", "b"); deck=true, gn=true,
                laplace=_RD_LAP, scatter=_RD_SCATTER)
            @test all(v isa _RD_PLAIN for v in last.(leaves))
            @test f["deck"] == "the deck"
            @test size(f["scatter/channels/b/data"]) == (16, length(f["scatter/t_ns"]))
        end
    end
end

# ------------------------------------------------------------------------------------------
# RD8: laplace/ only when given, raw; write_readable's arguments and refusals (OQ-63 (g))
# ------------------------------------------------------------------------------------------

@testset "RD8: laplace/ only when given, the raw sigma_post; the arguments and the refusals" begin
    r, lap = _RD_POL, _RD_LAP
    mktempdir() do dir
        rd = _rd_dict(write_readable(joinpath(dir, "lap.h5"), r, _RD_PROBLEM; laplace=lap))
        @test rd["source"] == "write_readable"
        @test !haskey(rd, "deck")
        for name in ("T_D", "u_D")
            @test isequal(rd["laplace/sigma/$name"], lap.sigma_post[name])
            @test isequal(rd["laplace/sigma_samp/$name"], lap.sigma_samp[name])
            @test rd["laplace/near_bound/$name"] == UInt8.(lap.near_bound[name])
        end
        @test rd["laplace/decrease"] === lap.decrease
        @test rd["fit/gn_decrease"] === lap.decrease
        bare = _rd_dict(write_readable(joinpath(dir, "bare.h5"), r, _RD_PROBLEM; deck=""))
        @test !any(startswith(k, "laplace/") || startswith(k, "scatter/") for k in keys(bare))
        @test !haskey(bare, "fit/gn_decrease")
        @test !haskey(bare, "deck")   # U4: an empty deck text is not written (h5py cannot read "")

        # fit(deck)'s own result beside a problem rebuilt from the deck (spec D7, U3): the same file
        # but for source and gn_decrease
        eff = read(joinpath(_RD_ON, "effective_deck.toml"), String)
        again = _rd_dict(write_readable(joinpath(dir, "again.h5"), _RD_RESULT, _RD_PROBLEM; deck=eff))
        fitted = _rd_dict(joinpath(_RD_ON, "readable.h5"))
        @test Set(keys(again)) == setdiff(Set(keys(fitted)), ["fit/gn_decrease"])
        @test sort([k for k in keys(again) if k != "source" && !isequal(again[k], fitted[k])]) == String[]

        wrong = copy(lap.theta)
        wrong[1] += 0.01
        lap_wrong = LaplaceResult(wrong, lap.decrease, lap.free, lap.cov_y_post, lap.cov_y_samp,
            lap.sigma_post, lap.sigma_samp, lap.near_bound)
        @test_throws "write_readable: laplace was computed at a different θ than result.theta" write_readable(
            joinpath(dir, "w.h5"), r, _RD_PROBLEM; laplace=lap_wrong)
        single = first(OTIS.build_problem(_RD_SINGLE_DECK))
        @test_throws "write_readable: problem has channels [\"main\"], but result has [\"a\", \"b\"]" write_readable(
            joinpath(dir, "w.h5"), r, single)
        te_free = read_deck(_rd_write(dir, "te.toml", _rd_sx_text(_RD_SA, _RD_SB, joinpath(dir, "te"); te_free=true)))
        @test_throws "write_readable: problem has ntheta = 7, but result has 6" write_readable(
            joinpath(dir, "w.h5"), r, first(OTIS.build_problem(te_free)))
    end
end

# ------------------------------------------------------------------------------------------
# RD9: scatter/ only when given: the windows' values, data and models (OQ-63 (h))
# ------------------------------------------------------------------------------------------

@testset "RD9: scatter/ only when given: the windows' values, their data and their models" begin
    sc, s = _RD_SCATTER, _RD_DECK.sections
    chan_list = OTIS._channel_list(s)
    raw, _, _ = OTIS._read_channels(s, chan_list, true)
    regions = OTIS._build_averaging_regions(s["fit_regions"])
    fine = [AveragingRegion(rg.lo, rg.hi, rg.nbins * sc.m[i], "") for (i, rg) in enumerate(regions)]
    windows = OTIS._average_channels(raw, chan_list, fine)
    mktempdir() do dir
        @test !haskey(_rd_dict(write_readable(joinpath(dir, "bare.h5"), _RD_POL, _RD_PROBLEM)), "scatter/t_ns")
        path = write_readable(joinpath(dir, "sc.h5"), _RD_POL, _RD_PROBLEM; scatter=sc)
        OTIS.jldopen(path, "r") do f
            @test isequal(f["scatter/t_ns"], t_from_si.(sc.t_fine, :t_ns))
            @test f["scatter/bin"] == sc.coarse_bin
            @test f["scatter/status"] == string.(sc.status)
            @test isequal(f["scatter/decrease"], sc.decrease)
            @test isequal(f["scatter/chi2"], sc.chi2)
            @test Set(keys(f["scatter/params"])) == Set(["T_D", "u_D"])
            for name in ("T_D", "u_D")
                @test isequal(f["scatter/params/$name"], sc.values[name])
                @test f["scatter/near_bound/$name"] == UInt8.(sc.near_bound[name])
                @test isequal(f["scatter/sigma_stat/$name"], sc.s_stat[name])
            end
            for (c, name) in enumerate(("a", "b"))
                @test isequal(f["scatter/channels/$name/data"], windows[c].image)
                @test isequal(f["scatter/channels/$name/sigma"], windows[c].sigma)
                @test f["scatter/channels/$name/mask"] == UInt8.(windows[c].mask)
            end
            # each window's chi2 from its data, model, sigma and mask (both weights 1)
            ca, _ = _rd_chi2_bins(f, "scatter/channels/a")
            cb, _ = _rd_chi2_bins(f, "scatter/channels/b")
            @test isapprox(f["scatter/chi2"], ca .+ cb; rtol=1e-12)
            # the model: windows 5 and 12 re-evaluated at their own values, each through its own
            # one-window deck (test_uncertainty.jl's SX4 route)
            for (j, lo, hi) in ((5, 0.4, 0.5), (12, 1.1, 1.2))
                text = _rd_sx_text(_RD_SA, _RD_SB, joinpath(dir, "w$j");
                    regions="[[fit.regions]]\nt_range_ns = [$lo, $hi]\nnbins = 1", regularizer="")
                p = first(OTIS.build_problem(read_deck(_rd_write(dir, "w$j.toml", text))))
                theta = flatten(p.pspec, ["T_D" => sc.values["T_D"][j], "u_D" => sc.values["u_D"][j]])
                for (c, name) in enumerate(("a", "b"))
                    ch = OTIS._channel_problem(p, c)
                    m = measure(unflatten(ch.experiment, p.pspec, theta; channel=c))
                    @test isapprox(f["scatter/channels/$name/model"][:, j], vec(m); rtol=1e-12)
                end
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# RD10: ScatterResult's new fields; the eleven-argument constructor; result.jld2's scatter/* unchanged
# ------------------------------------------------------------------------------------------

@testset "RD10: ScatterResult's new fields, its old constructor, and the result file unchanged" begin
    sc, r = _RD_SCATTER, _RD_POL
    @test fieldnames(ScatterResult) == (:theta, :m, :t_fine, :coarse_bin, :counts, :status, :decrease,
        :values, :near_bound, :s_stat, :n_near_bound, :data, :sigma, :mask, :model, :chi2, :held, :p16,
        :median, :p84)
    @test size.(sc.data) == size.(sc.sigma) == size.(sc.mask) == size.(sc.model) == [(24, 12), (16, 12)]
    @test length(sc.chi2) == 12
    old = ScatterResult(sc.theta, sc.m, sc.t_fine, sc.coarse_bin, sc.counts, sc.status, sc.decrease,
        sc.values, sc.near_bound, sc.s_stat, sc.n_near_bound)
    @test isempty(old.data) && isempty(old.sigma) && isempty(old.mask) && isempty(old.model) && isempty(old.chi2)
    @test sprint(show, sc) == sprint(show, old) == "ScatterResult(12 fine windows over 1 region(s), 2 free variable(s))"
    mktempdir() do dir
        write_result(joinpath(dir, "new.jld2"), r; deck="d", scatter=sc)
        write_result(joinpath(dir, "old.jld2"), r; deck="d", scatter=old)
        a = OTIS.jldopen(f -> Dict(_rd_leaves(f, "scatter")), joinpath(dir, "new.jld2"), "r")
        b = OTIS.jldopen(f -> Dict(_rd_leaves(f, "scatter")), joinpath(dir, "old.jld2"), "r")
        want = Set(["scatter/m", "scatter/t_fine", "scatter/coarse_bin", "scatter/counts", "scatter/status",
            "scatter/decrease"])
        for name in ("T_D", "u_D"), k in ("values", "near_bound", "s_stat", "n_near_bound")
            push!(want, "scatter/variables/$name/$k")
        end
        @test Set(keys(a)) == Set(keys(b)) == want
        @test all(isequal(a[k], b[k]) for k in keys(a))
        @test propertynames(read_result(joinpath(dir, "new.jld2")).scatter) == (:m, :t_fine, :coarse_bin,
            :counts, :status, :decrease, :values, :near_bound, :s_stat, :n_near_bound)
        @test_throws "write_readable: scatter holds the windows of 0 channel(s), but result has 2" write_readable(
            joinpath(dir, "w.h5"), r, _RD_PROBLEM; scatter=old)
        wrong = copy(sc.theta)
        wrong[1] += 0.01
        @test_throws "write_readable: scatter was computed at a different θ than result.theta" write_readable(
            joinpath(dir, "w.h5"), r, _RD_PROBLEM; scatter=ScatterResult(wrong, sc.m, sc.t_fine,
                sc.coarse_bin, sc.counts, sc.status, sc.decrease, sc.values, sc.near_bound, sc.s_stat,
                sc.n_near_bound))
    end
end

# ------------------------------------------------------------------------------------------
# RD14: a key-off fit removes an earlier run's readable.h5 (Phase 4.L spec D9; OQ-62 item 8)
# ------------------------------------------------------------------------------------------

@testset "RD14: readable = false removes an earlier run's readable.h5; result.jld2 the same either way" begin
    dir = joinpath(_RD_DIR, "out_rd14")
    on_deck = read_deck(_rd_write(_RD_DIR, "rd14_on.toml", _rd_sx_text(_RD_SA, _RD_SB, dir)))
    off_deck = read_deck(_rd_write(_RD_DIR, "rd14_off.toml", _rd_sx_text(_RD_SA, _RD_SB, dir; readable=false)))
    _rd_fit(on_deck)
    @test sort(readdir(dir)) == ["effective_deck.toml", "readable.h5", "result.jld2"]
    on = _rd_dict(joinpath(dir, "result.jld2"))
    _rd_fit(off_deck)
    @test sort(readdir(dir)) == ["effective_deck.toml", "result.jld2"]
    off = _rd_dict(joinpath(dir, "result.jld2"))
    # RD2's comparison: every dataset by name, bitwise, but the three clock datasets and the deck text,
    # which differs only by its [output] table
    @test Set(keys(on)) == Set(keys(off))
    clock = ("meta/created", "meta/created_unix", "fit/elapsed")
    @test sort([k for k in keys(on) if !(k in clock) && k != "deck" && !isequal(on[k], off[k])]) == String[]
    @test replace(off["deck"], "readable = false" => "readable = true") == on["deck"]
    # into the same directory again, now without a companion: nothing raised, nothing written
    @test _rd_fit(off_deck) isa FitResult
    @test sort(readdir(dir)) == ["effective_deck.toml", "result.jld2"]
end
