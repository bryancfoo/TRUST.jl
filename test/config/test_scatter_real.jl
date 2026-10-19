using Test
using TRUST
using TRUST: _quantile_sorted
using Random: Xoshiro
using Statistics: quantile

# Step 4.N.1's fast rows (Phase 4.N spec D8; OQ-64 (a) to (c)), gated "scatter real deck": SI1 to SI4 on
# the readable fixture's SX deck (test_uncertainty.jl's SX chains on noisy streaks, camera b's first two
# wavelength rows masked), whose problem type the "uncertainty" and "readable" gates have compiled. The
# deck here averages onto two regions, [0, 0.8] ns in two bins of four windows and [0.8, 1.2] ns in four
# bins of one window each, so one scatter holds both kinds of bin. The rows on a deck with an interval
# and a held series are test/slow/test_scatter_real.jl's.

include(joinpath(@__DIR__, "..", "fixtures", "readable_fixture.jl"))

const _SI_NBLAS = pin_blas_threads(1)

const _SI_DIR = mktempdir()
const _SI_SA = _rd_streak_a(joinpath(_SI_DIR, "a.jld2"); seed=41)
const _SI_SB = _rd_streak_b(joinpath(_SI_DIR, "b.jld2"); seed=42, masked=2)
const _SI_REGIONS = "[[fit.regions]]\nt_range_ns = [0.0, 0.8]\nnbins = 2\n\n[[fit.regions]]\nt_range_ns = [0.8, 1.2]\nnbins = 4"
const _SI_TEXT = _rd_sx_text(_SI_SA, _SI_SB, joinpath(_SI_DIR, "out"); regions=_SI_REGIONS)
const _SI_DECK = read_deck(_rd_write(_SI_DIR, "si.toml", _SI_TEXT))
const _SI_PROBLEM, _SI_OPT, _SI_START = TRUST.build_problem(_SI_DECK)
const _SI_RESULT = polish(_SI_PROBLEM, solve(_SI_PROBLEM, _SI_OPT; start=_SI_START)).result
const _SI_SCATTER = slice_scatter(_SI_DECK, _SI_RESULT)

# ------------------------------------------------------------------------------------------
# SI1: the quantiles of every bin and free variable are Statistics.quantile's, bitwise (OQ-64 (a))
# ------------------------------------------------------------------------------------------

@testset "SI1: p16, median and p84 per bin are Statistics.quantile's; a one-window bin gives its value" begin
    sc = _SI_SCATTER
    @test sc.m == [4, 1]
    @test sc.coarse_bin == [1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6]
    @test Set(keys(sc.values)) == Set(keys(sc.p16)) == Set(keys(sc.median)) == Set(keys(sc.p84)) ==
          Set(["T_D", "u_D"])
    @test isempty(sc.held)
    for name in ("T_D", "u_D"), k in 1:6
        J = findall(==(k), sc.coarse_bin)
        q = quantile(sc.values[name][J], (0.16, 0.5, 0.84))
        @test sc.p16[name][k] === q[1]
        @test sc.median[name][k] === q[2]
        @test sc.p84[name][k] === q[3]
        if length(J) == 1
            @test sc.p16[name][k] === sc.median[name][k] === sc.p84[name][k] === sc.values[name][only(J)]
        else
            @test sc.p16[name][k] < sc.median[name][k] < sc.p84[name][k]
        end
    end
end

# ------------------------------------------------------------------------------------------
# SI2: _quantile_sorted against Statistics.quantile, bitwise, on every size of 1 to 12 (U1)
# ------------------------------------------------------------------------------------------

@testset "SI2: _quantile_sorted is Statistics.quantile, bitwise, ties and near-ties included" begin
    rng = Xoshiro(20261015)
    n_checked = 0
    n_differ = 0
    for m in 1:12, trial in 1:50
        x = sort(200.0 .+ 30.0 .* randn(rng, m))
        if trial > 40 && m > 1
            x[2] = x[1] * (1 + 1e-12)       # a near-tie: the branch where a ≈ b
        end
        sort!(x)
        for q in (0.16, 0.5, 0.84)
            n_checked += 1
            n_differ += _quantile_sorted(x, q) !== quantile(x, q; sorted=true)
        end
    end
    @test n_checked == 1800
    @test n_differ == 0
    @test _quantile_sorted([3.5], 0.84) === 3.5
    @test _quantile_sorted([1.0, 2.0, 3.0, 4.0], 0.5) === 2.5
    @test_throws "_quantile_sorted: no value to take a quantile of" _quantile_sorted(Float64[], 0.5)
end

# ------------------------------------------------------------------------------------------
# SI3: a result from another deck is refused, intervals and held series counted (OQ-64 (b))
# ------------------------------------------------------------------------------------------

@testset "SI3: a FitResult from a deck with another variable set is refused" begin
    late = replace(_SI_TEXT, "u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }" =>
        "u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true, interval = \"late\" }\n\n" *
        "[fit.intervals.late]\nt_range_ns = [0.4, 1.2]")
    deck_late = read_deck(_rd_write(_SI_DIR, "late.toml", late))
    @test_throws "slice_scatter: result was not fitted from this deck" slice_scatter(deck_late, _SI_RESULT)

    hold = write_start(joinpath(_SI_DIR, "hold.jld2"); axis_ns=[0.0, 1.2],
        values=Dict("u_D" => [20.0, 38.0]), units=Dict("u_D" => "u_kms"), source="si_test")
    held = replace(_SI_TEXT, "[plasma]\n" =>
        "[fit.hold]\nfile = \"$(_rd_path(hold))\"\nvariables = [\"u_D\"]\n\n[plasma]\n")
    deck_held = read_deck(_rd_write(_SI_DIR, "held.toml", held))
    @test_throws "slice_scatter: result was not fitted from this deck" slice_scatter(deck_held, _SI_RESULT)

    te_free = read_deck(_rd_write(_SI_DIR, "te.toml", _rd_sx_text(_SI_SA, _SI_SB, joinpath(_SI_DIR, "te");
        regions=_SI_REGIONS, te_free=true)))
    @test_throws "slice_scatter: result was not fitted from this deck" slice_scatter(te_free, _SI_RESULT)
end

# ------------------------------------------------------------------------------------------
# SI4: the constructors, the result file's scatter/* unchanged, the readable file's new datasets (OQ-64 (c))
# ------------------------------------------------------------------------------------------

@testset "SI4: the eleven- and sixteen-field constructors; result.jld2's scatter/*; readable.h5's new datasets" begin
    sc, r = _SI_SCATTER, _SI_RESULT
    @test fieldnames(ScatterResult) == (:theta, :m, :t_fine, :coarse_bin, :counts, :status, :decrease,
        :values, :near_bound, :s_stat, :n_near_bound, :data, :sigma, :mask, :model, :chi2, :held, :p16,
        :median, :p84)
    old11 = ScatterResult(sc.theta, sc.m, sc.t_fine, sc.coarse_bin, sc.counts, sc.status, sc.decrease,
        sc.values, sc.near_bound, sc.s_stat, sc.n_near_bound)
    old16 = ScatterResult(sc.theta, sc.m, sc.t_fine, sc.coarse_bin, sc.counts, sc.status, sc.decrease,
        sc.values, sc.near_bound, sc.s_stat, sc.n_near_bound, sc.data, sc.sigma, sc.mask, sc.model, sc.chi2)
    @test isempty(old11.data) && isempty(old11.chi2)
    for o in (old11, old16)
        @test o.held isa Dict{String,Vector{Float64}} && isempty(o.held)
        @test isempty(o.p16) && isempty(o.median) && isempty(o.p84)
    end
    @test old16.data === sc.data && old16.chi2 === sc.chi2
    mktempdir() do dir
        write_result(joinpath(dir, "new.jld2"), r; deck="d", scatter=sc)
        write_result(joinpath(dir, "old.jld2"), r; deck="d", scatter=old16)
        a = TRUST.jldopen(f -> Dict(_rd_leaves(f, "scatter")), joinpath(dir, "new.jld2"), "r")
        b = TRUST.jldopen(f -> Dict(_rd_leaves(f, "scatter")), joinpath(dir, "old.jld2"), "r")
        want = Set(["scatter/m", "scatter/t_fine", "scatter/coarse_bin", "scatter/counts", "scatter/status",
            "scatter/decrease"])
        for name in ("T_D", "u_D"), k in ("values", "near_bound", "s_stat", "n_near_bound")
            push!(want, "scatter/variables/$name/$k")
        end
        @test Set(keys(a)) == Set(keys(b)) == want
        @test all(isequal(a[k], b[k]) for k in keys(a))

        @test_throws "write_readable: scatter holds no quantiles for [\"T_D\", \"u_D\"]; pass the ScatterResult slice_scatter returned" write_readable(
            joinpath(dir, "w.h5"), r, _SI_PROBLEM; scatter=old16)
        rd = _rd_dict(write_readable(joinpath(dir, "sc.h5"), r, _SI_PROBLEM; scatter=sc))
        @test rd["layout_version"] == "trust-readable-2"
        @test startswith(rd["README"], "trust-readable-2: ")
        for name in ("T_D", "u_D")
            @test rd["scatter/p16/$name"] == sc.p16[name]
            @test rd["scatter/median/$name"] == sc.median[name]
            @test rd["scatter/p84/$name"] == sc.p84[name]
        end
        @test !any(startswith(k, "scatter/held/") for k in keys(rd))
        @test Set(keys(rd)) == _rd_expected_keys(r, ("a", "b"); deck=false, gn=false, scatter=sc)
    end
end

pin_blas_threads(_SI_NBLAS)
