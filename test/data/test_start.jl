using Test
using OTIS
using OTIS: _interpolate_series
using JLD2: jldopen

# Step 4.I.1: the start file (Phase 4.I spec D5, D6, D8, D9, D13; OQ-59 item 2), gated "start file".
# SF1 to SF3: the otis-start-1 layout, write_start/read_start and the interpolation; SF4 to SF6: a
# deck's [fit.start] -- build_problem's start, and every refusal read_deck and build_problem make; SF7:
# a result file of the same deck as its start.

include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "start_hold_fixture.jl"))

# ------------------------------------------------------------------------------------------
# SF1: write_start and read_start round-trip bitwise
# ------------------------------------------------------------------------------------------

@testset "SF1: write_start and read_start round-trip" begin
    mktempdir() do dir
        path = _sh_start_file(dir)
        jldopen(path, "r") do f
            @test f["layout_version"] == "otis-start-1"
            @test f["meta/source"] == "test"
            @test f["variables/T_e/unit"] == "T_eV"
            @test !haskey(f, "variables/logn/unit")
        end
        st = read_start(path)
        @test st.layout_version == "otis-start-1"
        @test st.source == "test"
        @test st.axis == _SH_AXIS_NS .* 1e-9
        @test all(st.values["T_e"] .=== _SH_SERIES["T_e"])
        @test all(st.values["logn"] .=== _SH_SERIES["logn"])
        @test st.values["m_e"] == [3.5]
        @test st.units == Dict("T_e" => "T_eV")

        # axis_s: the axis as given, bitwise
        ax = [1.0e-9, 2.0e-9, 4.0e-9]
        p2 = write_start(joinpath(dir, "s.jld2"); axis_s=ax, values=["a" => [1.0, 2.0, 3.0]])
        @test p2 == joinpath(dir, "s.jld2")
        st2 = read_start(p2)
        @test all(st2.axis .=== ax)
        @test isempty(st2.units) && st2.source == ""
    end
end

# ------------------------------------------------------------------------------------------
# SF2: the writer's and the reader's refusals
# ------------------------------------------------------------------------------------------

@testset "SF2: write_start and read_start refuse" begin
    mktempdir() do dir
        p = joinpath(dir, "x.jld2")
        v = ["a" => [1.0, 2.0]]
        @test_throws "give exactly one of axis_s or axis_ns" write_start(p; values=v)
        @test_throws "give exactly one of axis_s or axis_ns" write_start(p; axis_s=[1.0, 2.0], axis_ns=[1.0, 2.0], values=v)
        @test_throws "unrecognised keyword" write_start(p; axis_us=[1.0, 2.0], values=v)
        @test_throws "the axis must be strictly increasing" write_start(p; axis_s=[2.0, 1.0], values=v)
        @test_throws "is not a valid variable name" write_start(p; axis_s=[1.0, 2.0], values=["2a" => [1.0, 2.0]])
        @test_throws "\"a\" has 3 values; the axis has 2" write_start(p; axis_s=[1.0, 2.0], values=["a" => [1.0, 2.0, 3.0]])
        @test_throws "\"a\" has a value that is not finite" write_start(p; axis_s=[1.0, 2.0], values=["a" => [1.0, NaN]])
        @test_throws "\"a\" is given twice" write_start(p; axis_s=[1.0, 2.0], values=["a" => [1.0, 2.0], "a" => [3.0]])
        @test_throws "units names \"b\", which values does not" write_start(p; axis_s=[1.0, 2.0], values=v, units=Dict("b" => "T_eV"))

        jldopen(p, "w") do f
            f["layout_version"] = "otis-start-0"
        end
        @test_throws "has layout_version \"otis-start-0\"; expected \"otis-start-1\"" read_start(p)
        jldopen(p, "w") do f
            f["layout_version"] = "otis-start-1"
            f["axis"] = [1.0, 2.0]
        end
        @test_throws "is missing the required dataset \"variables\"" read_start(p)
        jldopen(p, "w") do f
            f["layout_version"] = "otis-start-1"
            f["axis"] = [1.0, 2.0]
            f["variables/a/values"] = [1.0, 2.0, 3.0]
        end
        @test_throws "\"a\" has 3 values; the axis has 2" read_start(p)
    end
end

# ------------------------------------------------------------------------------------------
# SF3: the interpolation, exact at the axis's own points
# ------------------------------------------------------------------------------------------

@testset "SF3: _interpolate_series" begin
    ax = [1.0, 2.0, 4.0]
    v = [10.0, 0.1, 5.0]
    for k in 1:3
        @test _interpolate_series(ax, v, ax[k]) === v[k]
    end
    @test _interpolate_series(ax, v, 1.5) === 10.0 + (1.5 - 1.0) / (2.0 - 1.0) * (0.1 - 10.0)
    @test _interpolate_series(ax, v, 3.0) === 0.1 + (3.0 - 2.0) / (4.0 - 2.0) * (5.0 - 0.1)
    @test _interpolate_series(ax, v, 3.0) ≈ 2.55 rtol = 1e-15
end

# ------------------------------------------------------------------------------------------
# SF4: [fit.start]: build_problem's start is flatten of the interpolated series, bitwise
# ------------------------------------------------------------------------------------------

@testset "SF4: build_problem's start from [fit.start]" begin
    mktempdir() do dir
        _st_streaks(dir)
        text = _sh_epw_text(dir)
        p0, _, s0 = OTIS.build_problem(read_deck(_st_write(dir, "plain.toml", text)))
        sp = _sh_start_file(dir)
        deck = read_deck(_st_write(dir, "start.toml", _sh_tables(text; start=(sp, ["T_e", "logn", "m_e"]))))
        @test deck.sections["fit_start"] == Dict{String,Any}("file" => _st_path(sp), "variables" => Any["T_e", "logn", "m_e"])
        problem, _, start = OTIS.build_problem(deck)
        pspec = problem.pspec
        ax = read_start(sp).axis
        oracle = [nm => [_sh_interp(ax, _SH_SERIES[nm], t) for t in pspec.axis] for nm in ("T_e", "logn", "m_e")]
        @test all(start .=== flatten(pspec, oracle))
        @test start != s0
        # a variable not named keeps its init: T_D's entries are flatten(pspec)'s
        i = findfirst(v -> v.name == "T_D", pspec.variables)
        rng = pspec.offsets[i]:(pspec.offsets[i]+pspec.widths[i]-1)
        @test all(start[rng] .=== s0[rng])
        @test pspec.ntheta == p0.pspec.ntheta
        # the effective deck carries the table and re-reads to the same Deck
        eff = effective_deck(deck, problem, LBFGS())
        @test occursin("[fit.start]", eff)
        @test read_deck(_st_write(dir, "eff.toml", eff)) == deck
    end
end

# ------------------------------------------------------------------------------------------
# SF5: the refusals at read_deck
# ------------------------------------------------------------------------------------------

@testset "SF5: [fit.start] and [fit.hold] refused at read_deck" begin
    mktempdir() do dir
        _st_streaks(dir)
        text = _sh_epw_text(dir)
        sp = _st_path(joinpath(dir, "start.jld2"))
        rd(name, t) = read_deck(_st_write(dir, name, t))
        fitset(key, value) = _sh_edit(raw -> raw["fit"][key] = value, text)
        @test_throws "read_deck: \"fit.start\" must be a table" rd("a.toml", fitset("start", 1))
        @test_throws "unknown key \"files\" in section \"fit.start\"" rd("b.toml",
            fitset("start", Dict("files" => sp, "variables" => ["T_e"])))
        @test_throws "read_deck: \"fit.hold\" needs \"file\"" rd("c.toml",
            fitset("hold", Dict("variables" => ["T_e"])))
        @test_throws "read_deck: \"fit.start.variables\" must be a non-empty list of variable names" rd("d.toml",
            fitset("start", Dict("file" => sp, "variables" => String[])))
        @test_throws "read_deck: \"fit.start.variables\" names \"T_e\" twice" rd("e.toml",
            _sh_tables(text; start=(sp, ["T_e", "T_e"])))
        @test_throws "read_deck: \"fit.start.variables\" names \"nope\", which is not a fitted variable" rd("f.toml",
            _sh_tables(text; start=(sp, ["nope"])))
        frozen = _sh_edit(raw -> raw["fit"]["variables"]["T_D"]["vary"] = false, text)
        @test_throws "read_deck: \"fit.hold.variables\" names \"T_D\", which is not a fitted variable" rd("g.toml",
            _sh_tables(frozen; hold=(sp, ["T_D"])))
        @test_throws "read_deck: \"logn\" is named in both [fit.start] and [fit.hold]" rd("h.toml",
            _sh_tables(text; start=(sp, ["T_e", "logn"]), hold=(sp, ["logn"])))
    end
end

# ------------------------------------------------------------------------------------------
# SF6: the refusals at build_problem
# ------------------------------------------------------------------------------------------

@testset "SF6: [fit.start] and [fit.hold] refused at build_problem" begin
    mktempdir() do dir
        _st_streaks(dir)
        text = _sh_epw_text(dir)
        bp(name, t) = OTIS.build_problem(read_deck(_st_write(dir, name, t)))
        sp = _sh_start_file(dir)

        # a file of neither layout (the EPW streak itself)
        @test_throws "build_problem: [fit.start] file \"$(_st_path(joinpath(dir, "epw.jld2")))\" has layout_version \"otis-streak-1\"; expected \"otis-result-3\" (a result file) or \"otis-start-1\"" bp("a.toml",
            _sh_tables(text; start=(joinpath(dir, "epw.jld2"), ["T_e"])))
        # a name the file does not hold
        @test_throws "build_problem: [fit.start] names \"T_D\", which file \"$(_st_path(sp))\" does not hold; it holds [\"T_e\", \"logn\", \"m_e\"]" bp("b.toml",
            _sh_tables(text; start=(sp, ["T_D"])))
        # a unit that differs from the deck's binding keyword
        sk = _sh_start_file(dir; name="keV.jld2", units=Dict("T_e" => "T_keV"))
        @test_throws "build_problem: [fit.hold]: file \"$(_st_path(sk))\" gives \"T_e\" in \"T_keV\", but this deck binds it through \"T_eV\"" bp("c.toml",
            _sh_tables(text; hold=(sk, ["T_e"])))
        # a scalar variable given a series
        sc = _sh_edit(text) do raw
            raw["channel"]["epw"]["instrument"]["sensitivity"]["scale"] = "s_epw"
            raw["fit"]["variables"]["s_epw"] = Dict("bounds" => [1.0e-5, 1.0e-3], "init" => 2.0e-4)
        end
        ss =_sh_start_file(dir; name="scalar.jld2", values=Dict("s_epw" => fill(2.0e-4, 7)), units=Dict{String,String}())
        @test_throws "build_problem: [fit.start]: \"s_epw\" is a scalar variable, but file \"$(_st_path(ss))\" holds a series of 7 values for it" bp("d.toml",
            _sh_tables(sc; start=(ss, ["s_epw"])))
        # a bin outside the file's range: no extrapolation
        so = _sh_start_file(dir; name="short.jld2", axis_ns=collect(range(2.1, 5.0; length=7)))
        @test_throws "build_problem: [fit.start]: \"T_e\" needs a value at bin 4 (t = 5.16667 ns), outside file \"$(_st_path(so))\"'s time range [2.1, 5] ns; a series is never extrapolated" bp("e.toml",
            _sh_tables(text; start=(so, ["T_e"])))
        # a value outside the bounds, and one on a bound: refused, never clamped
        sb = _sh_start_file(dir; name="bound.jld2", values=Dict("T_e" => fill(2500.0, 7)))
        @test_throws "build_problem: [fit.hold]: \"T_e\" is 2500.0 at bin 1 (t = 2.63333 ns) from file \"$(_st_path(sb))\", not strictly inside its bounds (100.0, 2000.0)" bp("f.toml",
            _sh_tables(text; hold=(sb, ["T_e"])))
        se = _sh_start_file(dir; name="edge.jld2", values=Dict("T_e" => fill(2000.0, 7)))
        @test_throws "not strictly inside its bounds (100.0, 2000.0)" bp("g.toml",
            _sh_tables(text; start=(se, ["T_e"])))
        sv = _sh_start_file(dir; name="scalarval.jld2", values=Dict("s_epw" => [2.0e-2]), units=Dict{String,String}())
        @test_throws "build_problem: [fit.start]: \"s_epw\" is 0.02 in file \"$(_st_path(sv))\", not strictly inside its bounds (1.0e-5, 0.001)" bp("h.toml",
            _sh_tables(sc; start=(sv, ["s_epw"])))
    end
end

# ------------------------------------------------------------------------------------------
# SF7: a result file of the deck itself as its start returns its theta, to the logit's round trip
# ------------------------------------------------------------------------------------------

@testset "SF7: an otis-result-3 file as [fit.start]" begin
    mktempdir() do dir
        _st_streaks(dir)
        text = _sh_epw_text(dir)
        problem, _, start = OTIS.build_problem(read_deck(_st_write(dir, "plain.toml", text)))
        theta = start .+ 0.1 .* sin.(1:length(start))             # four time-varying variables, off their inits
        r = OTIS._build_result(problem, theta, 0.0, OTIS._TRAJECTORY_ROW[], true, :converged_at_start, 0, 0, 0,
            0.0, "none")
        rpath = write_result(joinpath(dir, "result.jld2"), r)
        names = ["T_D", "T_e", "logn", "m_e"]
        p2, _, start2 = OTIS.build_problem(read_deck(_st_write(dir, "own.toml", _sh_tables(text; start=(rpath, names)))))
        # four ulps of the larger bound's magnitude, carried through dy/dx = 1/_dx_dy (spec D9)
        worst = 0.0
        for (iv, v) in enumerate(p2.pspec.variables), k in 0:(p2.pspec.widths[iv]-1)
            i = p2.pspec.offsets[iv] + k
            x = OTIS._from_unconstrained(theta[i], v.lower, v.upper)
            tol = 4 * eps(max(abs(v.lower), abs(v.upper))) / OTIS._dx_dy(x, v.lower, v.upper)
            worst = max(worst, abs(start2[i] - theta[i]) / tol)
        end
        @info "SF7: the start against the file's theta, in units of the round trip's bound" worst
        @test worst <= 1
        @test start2 != start
    end
end
