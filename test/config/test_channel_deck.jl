using Test
using OTIS
using JLD2: jldopen

# CD1 to CD12 against dev/phase4A_instructions.md Section 9.2 (spec D40, chunk 7), gated
# "channel deck": the channel deck and the result file. The CD fixture (amended 2026-09-24; Level
# 2 computed the oracle numbers below with average_regions itself) is two tiny synthetic streaks,
# cameras "a" and "b", with different slice times over one shared span. No slice sits on a window
# edge, so no membership depends on round-off.

# ------------------------------------------------------------------------------------------
# The fixture
# ------------------------------------------------------------------------------------------

function _cd_write_streak_a(path::AbstractString)
    lambda_m = collect(range(262.9, 263.6; length=16)) .* 1e-9
    t_s = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1] .* 1e-9
    image = fill(30.0, 16, 6)
    sigma = fill(1.0, 16, 6)
    mask = trues(16, 6)
    mask[16, :] .= false
    sd = StreakData(; lambda_m=lambda_m, t_s=t_s, image=image, sigma=sigma, mask=mask)
    write_streak(path, sd; source="test")
    return sd
end

function _cd_write_streak_b(path::AbstractString)
    lambda_m = collect(range(262.2, 262.9; length=12)) .* 1e-9
    t_s = [0.05, 0.35, 0.65, 0.95] .* 1e-9
    image = fill(30.0, 12, 4)
    sigma = fill(1.0, 12, 4)
    sd = StreakData(; lambda_m=lambda_m, t_s=t_s, image=image, sigma=sigma)
    write_streak(path, sd; source="test")
    return sd
end

function _cd_write_streak_masked(path::AbstractString)
    lambda_m = collect(range(262.9, 263.6; length=16)) .* 1e-9
    t_s = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1] .* 1e-9
    image = fill(30.0, 16, 6)
    sigma = fill(1.0, 16, 6)
    mask = falses(16, 6)
    sd = StreakData(; lambda_m=lambda_m, t_s=t_s, image=image, sigma=sigma, mask=mask)
    write_streak(path, sd; source="test")
    return sd
end

const _CD_PLASMA_BLOCK = """
[plasma]
n_e_cm3 = 3.0e19

[plasma.electrons.e]
fraction = 1.0
T_eV = 400.0

[plasma.ions.D]
mass_amu = 2.014
charge = 1
fraction = 1.0
T_eV = "T_D"
u_kms = 0.0
"""

_cd_path(p::AbstractString) = replace(p, "\\" => "/")

# Writes `text` to `dir/name` and returns the path -- a thin wrapper so `read_deck(_cd_write(...))`
# reads as one expression, since `write` itself returns a byte count, not a path.
function _cd_write(dir::AbstractString, name::AbstractString, text::AbstractString)
    path = joinpath(dir, name)
    write(path, text)
    return path
end

# The two-channel deck (D40's design, Section 9.2's fixture): channels = ["a", "b"], one shared
# [[fit.regions]] of two 0.6 ns windows, channel b's weight 2.5, T_D shared, scale_a/scale_b
# per-channel, bg shared across both channels' Constant backgrounds (CD7's first half).
function _cd_deck_text(path_a::AbstractString, path_b::AbstractString, outdir::AbstractString;
    b_extra::AbstractString="",
    regions_block::AbstractString="[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 2\n")
    return """
    deck_version = "otis-deck-2"
    channels = ["a", "b"]

    $regions_block

    [channel.a.data]
    file = "$(_cd_path(path_a))"
    [channel.a.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    [channel.a.instrument]
    sensitivity = { scale = "scale_a" }
    background = [{ type = "Constant", level = "bg" }]

    [channel.b]
    weight = 2.5
    [channel.b.data]
    file = "$(_cd_path(path_b))"
    $b_extra
    [channel.b.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    [channel.b.instrument]
    wavelength_calibration = { lambda_offset_nm = 0.01 }
    sensitivity = { scale = "scale_b" }
    background = [{ type = "Constant", level = "bg" }]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0 }
    scale_a = { bounds = [1e-7, 1e-5], init = 1e-6 }
    scale_b = { bounds = [1e-7, 1e-5], init = 1e-6 }
    bg = { bounds = [1.0, 60.0], init = 20.0 }

    $_CD_PLASMA_BLOCK

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "lbfgs"
    lbfgs = { iterations = 3 }

    [output]
    dir = "$(_cd_path(outdir))"
    """
end

# The one-channel shorthand, camera a alone with the same region and content (CD5, CD11).
function _cd_shorthand_deck_text(path_a::AbstractString, outdir::AbstractString)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(_cd_path(path_a))"

    [[fit.regions]]
    t_range_ns = [0.0, 1.2]
    nbins = 2

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [instrument]
    sensitivity = { scale = "scale_a" }
    background = [{ type = "Constant", level = "bg" }]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0 }
    scale_a = { bounds = [1e-7, 1e-5], init = 1e-6 }
    bg = { bounds = [1.0, 60.0], init = 20.0 }

    $_CD_PLASMA_BLOCK

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "lbfgs"
    lbfgs = { iterations = 3 }

    [output]
    dir = "$(_cd_path(outdir))"
    """
end

# The one-channel channel form, camera a alone, same content as the shorthand above (CD5).
function _cd_one_channel_deck_text(path_a::AbstractString, outdir::AbstractString;
    weight_line::AbstractString="")
    return """
    deck_version = "otis-deck-2"
    channels = ["a"]

    [[fit.regions]]
    t_range_ns = [0.0, 1.2]
    nbins = 2

    [channel.a]
    $weight_line
    [channel.a.data]
    file = "$(_cd_path(path_a))"
    [channel.a.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    [channel.a.instrument]
    sensitivity = { scale = "scale_a" }
    background = [{ type = "Constant", level = "bg" }]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0 }
    scale_a = { bounds = [1e-7, 1e-5], init = 1e-6 }
    bg = { bounds = [1.0, 60.0], init = 20.0 }

    $_CD_PLASMA_BLOCK

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "lbfgs"
    lbfgs = { iterations = 3 }

    [output]
    dir = "$(_cd_path(outdir))"
    """
end

# The oracle (Section 9.2, computed 2026-09-24 by Level 2 with average_regions itself):
#   camera a alone: t̄ = [0.3, 0.9] ns, |W| = [3, 3]
#   camera b alone: t̄ = [0.2, 0.8] ns, |W| = [2, 2]
#   pooled: bin 1 = (3*0.3 + 2*0.2)/5 = 0.26 ns; bin 2 = (3*0.9 + 2*0.8)/5 = 0.86 ns
#   camera b shifted by t_shift_ns = 0.3: t̄ = [0.35, 0.8] ns, |W| = [1, 2] (the 1.25 ns slice
#     falls outside [0.0, 1.2] ns and is dropped)
#   pooled, b shifted: bin 1 = (3*0.3 + 1*0.35)/4 = 0.3125 ns; bin 2 unchanged, 0.86 ns
const _CD_POOLED_AXIS_NS = [0.26, 0.86]
const _CD_POOLED_AXIS_SHIFTED_NS = [0.3125, 0.86]

# ------------------------------------------------------------------------------------------
# CD1: the top-level key check; [channel.x] in a shorthand deck is mixing
# ------------------------------------------------------------------------------------------

@testset "CD1: the top-level key check; mixing" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        _cd_write_streak_a(path_a)
        outdir = joinpath(dir, "out")

        bad = _cd_shorthand_deck_text(path_a, outdir) * "\n[bogus]\nx = 1.0\n"
        @test_throws "unknown top-level key" read_deck(_cd_write(dir, "bad_top.toml", bad))

        mixed = _cd_shorthand_deck_text(path_a, outdir) *
                "\nchannels = [\"epw\"]\n[channel.epw]\nweight = 1.0\n"
        @test_throws "mixes the channel form" read_deck(_cd_write(dir, "mixed.toml", mixed))
    end
end

# ------------------------------------------------------------------------------------------
# CD2: the channel form's own errors
# ------------------------------------------------------------------------------------------

@testset "CD2: the channel form's own errors" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        base = _cd_deck_text(path_a, path_b, outdir)

        dup = replace(base, "channels = [\"a\", \"b\"]" => "channels = [\"a\", \"a\"]")
        @test_throws "appears more than once" read_deck(_cd_write(dir, "d_dup.toml", dup))

        illformed = replace(base, "channels = [\"a\", \"b\"]" => "channels = [\"A\", \"b\"]")
        @test_throws r"must match" read_deck(_cd_write(dir, "d_illformed.toml", illformed))

        # list names a channel with no table
        no_table = replace(base, "channels = [\"a\", \"b\"]" => "channels = [\"a\", \"b\", \"c\"]")
        @test_throws "disagree" read_deck(_cd_write(dir, "d_no_table.toml", no_table))

        # a table not listed: rename [channel.b...] headers to [channel.c...]
        extra_table = replace(base, "[channel.b]" => "[channel.c]")
        extra_table = replace(extra_table, "[channel.b.data]" => "[channel.c.data]")
        extra_table = replace(extra_table, "[channel.b.geometry]" => "[channel.c.geometry]")
        extra_table = replace(extra_table, "[channel.b.instrument]" => "[channel.c.instrument]")
        @test_throws "disagree" read_deck(_cd_write(dir, "d_extra_table.toml", extra_table))

        unknown_key = replace(base, "[channel.b]\nweight = 2.5" => "[channel.b]\nweight = 2.5\nbogus = 1.0")
        @test_throws "unknown key \"bogus\" in section \"channel.b\"" read_deck(
            _cd_write(dir, "d_unknown.toml", unknown_key))

        missing_data = replace(base, "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\n" => "")
        @test_throws "\"channel.a\" is missing section \"data\"" read_deck(
            _cd_write(dir, "d_missing_data.toml", missing_data))

        missing_geometry = replace(base,
            "[channel.a.geometry]\nlambda_probe_nm = 263.25\ntheta_deg = 60.0\n" => "")
        @test_throws "\"channel.a\" is missing section \"geometry\"" read_deck(
            _cd_write(dir, "d_missing_geom.toml", missing_geometry))

        weight_zero = replace(base, "weight = 2.5" => "weight = 0.0")
        @test_throws "must be a finite number > 0" read_deck(_cd_write(dir, "d_w0.toml", weight_zero))

        weight_neg = replace(base, "weight = 2.5" => "weight = -1.0")
        @test_throws "must be a finite number > 0" read_deck(_cd_write(dir, "d_wneg.toml", weight_neg))

        weight_nan = replace(base, "weight = 2.5" => "weight = nan")
        @test_throws "must be a finite number > 0" read_deck(_cd_write(dir, "d_wnan.toml", weight_nan))

        weight_string = replace(base, "weight = 2.5" => "weight = \"w\"")
        @test_throws "not a variable reference" read_deck(_cd_write(dir, "d_wstr.toml", weight_string))
    end
end

# ------------------------------------------------------------------------------------------
# CD3: the regions; the four cross-section errors
# ------------------------------------------------------------------------------------------

@testset "CD3: the regions; the four cross-section errors" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        base = _cd_deck_text(path_a, path_b, outdir)

        # "regions" inside a data table is refused, naming [[fit.regions]]
        old_key = replace(base,
            "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\n" =>
            "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\nregions = []\n")
        @test_throws "moved from [data] to [[fit.regions]]" read_deck(
            _cd_write(dir, "d_oldkey.toml", old_key))

        # t_range_ns together with [[fit.regions]]
        with_trange = replace(base,
            "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\n" =>
            "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\nt_range_ns = [0.0, 1.0]\n")
        @test_throws "and the deck gives [[fit.regions]]" read_deck(
            _cd_write(dir, "d_trange.toml", with_trange))

        # t_shift_ns without [[fit.regions]]: a one-channel deck, so "2+ channels needs regions"
        # does not also fire.
        one_shift = replace(_cd_one_channel_deck_text(path_a, outdir),
            "[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 2\n\n" => "")
        one_shift = replace(one_shift,
            "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\n" =>
            "[channel.a.data]\nfile = \"$(_cd_path(path_a))\"\nt_shift_ns = 0.1\n")
        @test_throws "without [[fit.regions]]" read_deck(_cd_write(dir, "d_shift.toml", one_shift))

        # a region's interval also declared in [fit.intervals]
        interval_clash = replace(base,
            "[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 2\n" =>
            "[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 2\ninterval = \"pre\"\n") *
            "\n[fit.intervals.pre]\nt_range_ns = [0.0, 1.2]\n"
        @test_throws "which [fit.intervals] also declares" read_deck(
            _cd_write(dir, "d_interval.toml", interval_clash))

        # two or more channels without [[fit.regions]]
        no_regions = replace(base, "[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 2\n" => "")
        @test_throws "needs [[fit.regions]]" read_deck(_cd_write(dir, "d_noregions.toml", no_regions))
    end
end

# ------------------------------------------------------------------------------------------
# CD4: a two-channel deck builds a JointFitProblem, pooled axis against the oracle
# ------------------------------------------------------------------------------------------

@testset "CD4: a two-channel deck builds a JointFitProblem; the pooled axis" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        deck = read_deck(_cd_write(dir, "deck.toml", _cd_deck_text(path_a, path_b, outdir)))

        problem, optimizer, start = OTIS.build_problem(deck)
        @test problem isa JointFitProblem
        @test OTIS.channel_names(problem) == ["a", "b"]
        @test problem.weights == [1.0, 2.5]

        oracle = _CD_POOLED_AXIS_NS .* 1e-9
        @test isapprox(problem.pspec.trace.axis, oracle; rtol=1e-14)
        for ch in problem.channels
            @test ch.data.t == problem.pspec.trace.axis
        end

        scale_a_bindings = [b for b in problem.pspec.bindings if b.name == "scale_a"]
        scale_b_bindings = [b for b in problem.pspec.bindings if b.name == "scale_b"]
        @test length(scale_a_bindings) == 1
        @test scale_a_bindings[1].channel == 1
        @test length(scale_b_bindings) == 1
        @test scale_b_bindings[1].channel == 2

        # CD7's first half: bg is bound into both channels' backgrounds.
        bg_bindings = [b for b in problem.pspec.bindings if b.name == "bg"]
        @test length(bg_bindings) == 2
        @test Set(b.channel for b in bg_bindings) == Set([1, 2])
    end
end

# ------------------------------------------------------------------------------------------
# CD5: a one-channel channel-form deck builds a plain FitProblem, === the shorthand's
# ------------------------------------------------------------------------------------------

@testset "CD5: a one-channel channel-form deck builds a plain FitProblem" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        _cd_write_streak_a(path_a)
        outdir1 = joinpath(dir, "out1")
        outdir2 = joinpath(dir, "out2")

        deck_channel = read_deck(_cd_write(dir, "one_channel.toml",
            _cd_one_channel_deck_text(path_a, outdir1)))
        deck_short = read_deck(_cd_write(dir, "shorthand.toml",
            _cd_shorthand_deck_text(path_a, outdir2)))

        problem_c, _, start_c = OTIS.build_problem(deck_channel)
        problem_s, _, start_s = OTIS.build_problem(deck_short)

        @test problem_c isa FitProblem
        @test problem_c.name == "a"
        @test problem_s.name == "main"

        r_c = residuals(problem_c, start_c)
        r_s = residuals(problem_s, start_s)
        @test length(r_c) == length(r_s)
        @test all(r_c .=== r_s)
    end
end

# ------------------------------------------------------------------------------------------
# CD6: the channel form round-trips
# ------------------------------------------------------------------------------------------

@testset "CD6: the channel form round-trips" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        deck1 = read_deck(_cd_write(dir, "deck.toml", _cd_deck_text(path_a, path_b, outdir)))

        problem1, optimizer1, start1 = OTIS.build_problem(deck1)
        eff = effective_deck(deck1, problem1, optimizer1)
        @test occursin("channel.a.", eff)
        @test occursin("channel.b.", eff)

        deck2 = read_deck(_cd_write(dir, "eff.toml", eff))
        @test deck2 == deck1
        problem2, _, start2 = OTIS.build_problem(deck2)

        @test OTIS.channel_names(problem2) == OTIS.channel_names(problem1)
        @test problem2.weights == problem1.weights
        @test problem2.pspec.trace.axis == problem1.pspec.trace.axis

        names1 = OTIS.channel_names(problem1)
        names2 = OTIS.channel_names(problem2)
        for (b1, b2) in zip(problem1.pspec.bindings, problem2.pspec.bindings)
            @test b1.name == b2.name
            @test b1.factor == b2.factor
            @test b1.unit_key == b2.unit_key
            if b1.kind === :effect
                @test names1[b1.channel] == names2[b2.channel]
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# CD7: a shared variable gets two bindings; the one-factor rule spans channels
# ------------------------------------------------------------------------------------------

@testset "CD7: the one-factor rule spans channels" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")

        base = _cd_deck_text(path_a, path_b, outdir)
        clash = replace(base,
            "sensitivity = { scale = \"scale_a\" }" =>
            "sensitivity = { scale = \"scale_a\" }\nspectral_irf = { fwhm_nm = \"dlam\" }")
        clash = replace(clash,
            "wavelength_calibration = { lambda_offset_nm = 0.01 }" =>
            "wavelength_calibration = { lambda_offset_nm = \"dlam\" }")
        # dlam's placeholder value must construct a legal SpectralIRF (fwhm_nm > 0) before
        # ParameterSpec's binding-conflict check is ever reached.
        clash = replace(clash, "bg = { bounds = [1.0, 60.0], init = 20.0 }" =>
            "bg = { bounds = [1.0, 60.0], init = 20.0 }\ndlam = { bounds = [1e-11, 1e-9], init = 1e-10 }")
        deck = read_deck(_cd_write(dir, "clash.toml", clash))
        # the one-conversion-factor rule is checked in ParameterSpec's constructor (_resolve_bindings),
        # reached from build_problem -- read_deck alone does not resolve bindings.
        err = try
            OTIS.build_problem(deck)
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("different units", msg)
        @test occursin("fwhm_nm", msg)
        @test occursin("lambda_offset_nm", msg)
    end
end

# ------------------------------------------------------------------------------------------
# CD8: an entirely masked channel is refused; two channels without regions is refused
# ------------------------------------------------------------------------------------------

@testset "CD8: an entirely masked channel is refused" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a_masked.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_masked(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        deck = read_deck(_cd_write(dir, "deck.toml", _cd_deck_text(path_a, path_b, outdir)))
        @test_throws "channel \"a\" has no usable pixel" OTIS.build_problem(deck)
    end
end

# ------------------------------------------------------------------------------------------
# CD9: a channel's t_shift_ns moves which windows its slices fall in
# ------------------------------------------------------------------------------------------

@testset "CD9: t_shift_ns moves camera b's windows; the pooled axis follows" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir1 = joinpath(dir, "out1")
        outdir2 = joinpath(dir, "out2")

        deck_noshift = read_deck(_cd_write(dir, "noshift.toml",
            _cd_deck_text(path_a, path_b, outdir1)))
        problem_noshift, _, _ = OTIS.build_problem(deck_noshift)
        @test problem_noshift.channels[2].data.meta["regions_window_counts"] == [2, 2]
        @test isapprox(problem_noshift.pspec.trace.axis, _CD_POOLED_AXIS_NS .* 1e-9; rtol=1e-14)

        deck_shift = read_deck(_cd_write(dir, "shift.toml",
            _cd_deck_text(path_a, path_b, outdir2; b_extra="t_shift_ns = 0.3\n")))
        problem_shift, _, _ = OTIS.build_problem(deck_shift)
        @test problem_shift.channels[2].data.meta["regions_window_counts"] == [1, 2]
        @test isapprox(problem_shift.pspec.trace.axis, _CD_POOLED_AXIS_SHIFTED_NS .* 1e-9; rtol=1e-14)
    end
end

# ------------------------------------------------------------------------------------------
# CD10: fit(deck) writes otis-result-3 with channel groups
# ------------------------------------------------------------------------------------------

@testset "CD10: fit(deck) writes otis-result-3 with channel groups" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        deck = read_deck(_cd_write(dir, "deck.toml", _cd_deck_text(path_a, path_b, outdir)))

        result = fit(deck)
        respath = joinpath(outdir, "result.jld2")
        @test isfile(respath)
        r = read_result(respath)
        @test r.layout_version == "otis-result-3"
        @test Set(c.name for c in r.channels) == Set(["a", "b"])

        problem2, _, _ = OTIS.build_problem(deck)
        for summ in r.channels
            c = summ.name == "a" ? 1 : 2
            ch = problem2.channels[c]
            D = ch.data.image
            sig = ch.sigma
            M = summ.model
            for idx in CartesianIndices(D)
                if ch.mask[idx]
                    @test isapprox(summ.residual[idx], (M[idx] - D[idx]) / sig[idx]; rtol=1e-12)
                else
                    @test isnan(summ.residual[idx])
                end
            end
            # Carried (chunk 7's review): the stored model is the right channel's own -- a
            # write_result that stored the wrong channel's model would still pass the check
            # above, since it recomputes (model - D)/sigma from that SAME stored model.
            @test isapprox(summ.residual[ch.pixel_index], residuals(ch, result.theta); rtol=1e-12)
        end

        a_summ = r.channels[findfirst(c -> c.name == "a", r.channels)]
        @test all(isnan, a_summ.residual[16, :])
        @test all(==(0x00), a_summ.mask[16, :])
        for k in eachindex(a_summ.effects)
            @test a_summ.effects[k].type isa String
        end
    end
end

# ------------------------------------------------------------------------------------------
# CD11: write_result without problem omits residual/mask; a shorthand fit writes channels/main
# ------------------------------------------------------------------------------------------

@testset "CD11: write_result without problem omits residual and mask" begin
    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        path_b = joinpath(dir, "b.jld2")
        _cd_write_streak_a(path_a)
        _cd_write_streak_b(path_b)
        outdir = joinpath(dir, "out")
        deck = read_deck(_cd_write(dir, "deck.toml", _cd_deck_text(path_a, path_b, outdir)))
        problem, optimizer, start = OTIS.build_problem(deck)
        result = solve(problem, optimizer; start=start)

        path = joinpath(dir, "no_problem.jld2")
        write_result(path, result; deck="dummy")
        r = read_result(path)
        for summ in r.channels
            @test summ.residual === nothing
            @test summ.mask === nothing
        end
    end

    mktempdir() do dir
        path_a = joinpath(dir, "a.jld2")
        _cd_write_streak_a(path_a)
        outdir = joinpath(dir, "out")
        deck = read_deck(_cd_write(dir, "deck.toml", _cd_shorthand_deck_text(path_a, outdir)))
        fit(deck)
        r = read_result(joinpath(outdir, "result.jld2"))
        @test length(r.channels) == 1
        @test r.channels[1].name == "main"
        @test r.channels[1].residual !== nothing
    end
end

# ------------------------------------------------------------------------------------------
# CD12: an old result-file version is refused, naming both versions
# ------------------------------------------------------------------------------------------

@testset "CD12: an otis-result-2 file is refused by version" begin
    mktempdir() do dir
        path = joinpath(dir, "old.jld2")
        jldopen(path, "w") do f
            f["layout_version"] = "otis-result-2"
        end
        err = try
            read_result(path)
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("otis-result-2", msg)
        @test occursin("otis-result-3", msg)
    end
end
