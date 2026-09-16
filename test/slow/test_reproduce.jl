using Test
using OTIS
using OTIS: _with_weight, _leaf_si_path
using Random: Xoshiro

# FR1 to FR8 against dev/phase3_instructions.md Section 10 (spec D38), the slow tier's own
# reproduction of foo2026's three test problems on the Phase 2 cut-down grid, nlambda = 128,
# nt = 32 -- the grid examples/problems.jl's own D25 feature numbers are pinned on. Every row of
# this file is reported with its numbers (D38), not only passed: @info lines below carry the
# numbers a human reads even where nothing asserts on them (FR3, FR8).
#
# Each cut-down FitProblem is built from the SAME deck chunk 12 already wrote for the
# full-resolution examples (examples/*.toml): read_deck, then a copy of its own
# `sections["data"]["file"]` pointed at a freshly synthesized 128 x 32 streak instead of the
# full-resolution one (OTIS.Deck/OTIS.build_problem directly -- a "script deck" in
# conventions.md 7.4's sense, everything but the data file exactly the committed deck). This
# reuses the exact fitted-variable bounds/inits, dependents, instrument chain, optimizer and
# regularizer terms the full-resolution reproduction runs, rather than a second hand-built copy
# that could quietly drift from it.
#
# examples/epw_ablation.toml, examples/iaw_multiflow.toml and examples/iaw_shock.toml are chunk
# 12's own names for D38's three reproduction decks (rule 11 zero) -- not the
# examples/fit_*.toml names dev/phase3_instructions.md Section 9 anticipated when it was written,
# before chunk 12 ran; reported in this chunk's hand-off, not renamed here.
#
# The Lambda lists below are shortened from D38's own seven-decade default (Section 10: "if a
# slow-tier problem exceeds about three minutes, shorten the Lambda list, not the grid"). Measured
# on this machine, a single fit costs roughly 140 s for the EPW problem (its SuperGaussian shape
# is evaluated through the widened table, not the Maxwellian closed form), 74 s for the shock
# problem and 31 s for multi-flow -- so even the unregularized fit alone exceeds the three-minute
# guideline for EPW, and shortening the list cannot close that gap by itself. The counts below are
# a practical compromise, not a claim that they meet the guideline; the actual elapsed time is
# measured and reported for every problem (@info "... scan"), not asserted.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const _FR_EXAMPLES_DIR = joinpath(@__DIR__, "..", "..", "examples")

_fr_mean(d::Dict{String,Float64}) = sum(values(d)) / length(d)

"""
Prints the scan table one row at a time (`@printf`-style, not a bare `@info` on the vectors):
Julia's non-interactive logging truncates a short `Vector{Float64}` to
`"3-element Vector{Float64}: …"` when stdout is redirected to a file rather than a terminal (as it
is under `julia ... > log`, exactly this file's own slow-tier run), so a report built from that
log would be missing the very numbers D38 asks for. One row per weight avoids it.
"""
function _fr_print_scan_table(label::AbstractString, scan::WeightScanResult, elapsed::Float64)
    @info "$label: weight_scan elapsed" elapsed_s = elapsed
    for (w, c, r) in zip(scan.weights, scan.chi2_normalized, scan.R_normalized)
        @info "$label: scan row" Lambda = w chi2_over_chi2_unreg = c R_over_R_unreg = r
    end
end

"""
Prints a `Dict{String,Float64}` one entry at a time, for the same reason as
`_fr_print_scan_table`: Julia's non-interactive `Dict` display also elides entries (shown as a
bare `⋮ => ⋮` row) once there are more than a handful, which silently dropped two of multi-flow's
five per-parameter RSS entries under `julia ... > log`.
"""
function _fr_print_dict(label::AbstractString, d::Dict{String,Float64})
    for k in sort(collect(keys(d)))
        @info "$label: per-parameter row" variable = k rss_normalized = d[k]
    end
end

"""
One leaf's ground-truth row from a `PlasmaTrace` (e.g. `problem_nt.trace`, examples/problems.jl),
converted from SI into the same boundary-keyword unit a deck fits it through (`_leaf_si_path`,
the exact conversion `_resolve_bindings` itself uses) -- eV for `"electrons.e.T_eV"`, cm^-3 for
`"n_e_cm3"`, km/s for `"ions.C.u_kms"`, dimensionless where the key carries no unit.
"""
function _fr_truth(trace::PlasmaTrace, boundary_key::AbstractString)
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    i === nothing && error("_fr_truth: leaf \"$si_path\" not present in this trace")
    return collect(Float64, trace.values[i, :]) ./ factor
end

# The cut-down deck: read_deck's own validated sections, `[data].file` replaced by the 128x32
# streak this file just synthesized. `read_deck`'s own reference/key validation already ran once
# when chunk 12 built and committed the full-resolution deck; nothing here re-derives it.
function _fr_deck_for_cutdown(name::AbstractString, sd_path::AbstractString)
    deck = read_deck(joinpath(_FR_EXAMPLES_DIR, "$name.toml"))
    sections = deepcopy(deck.sections)
    sections["data"] = merge(sections["data"], Dict{String,Any}("file" => sd_path))
    return OTIS.Deck(deck.deck_version, sections)
end

_fr_total_variation(v::AbstractVector) = sum(abs(v[j+1] - v[j]) for j in 1:(length(v)-1))

"""
The regularized fit D38 calls "shown", at a weight that need not be one of the scan's own
decades (EPW's 2e-4, multi-flow's 5e-3 are not). If `weight` already is one of `scan`'s own
weights, its own result is reused rather than refit. Otherwise, one more fit at exactly `weight`,
warm-started from the scan's nearest neighbour in log10(Lambda) -- reusing `scan`'s own
unregularized fit and `rss_unreg` (D31's normalization is "by the unregularized fit's own RSS for
that variable", the SAME fit, not a fresh cold-start one).
"""
function _fr_shown(problem::FitProblem, scan::WeightScanResult, weight::Float64, optimizer)
    idx = findfirst(==(weight), scan.weights)
    idx !== nothing && return scan.results[idx]
    logs = log10.(scan.weights)
    nearest = scan.results[argmin(abs.(logs .- log10(weight)))]
    return solve(_with_weight(problem, weight), optimizer; start=nearest.theta)
end

function _fr_rss_normalized(result::FitResult, truth::Dict{String,Vector{Float64}}, scan::WeightScanResult)
    return Dict{String,Float64}(v => sum(abs2, truth[v] .- result.values[v]) / scan.rss_unreg[v]
                                 for v in keys(truth))
end

function _fr_scan_monotonic(scan::WeightScanResult)
    for k in 1:(length(scan.weights)-1)
        slack_chi2 = 1e-6 * scan.chi2[1]
        slack_R = 1e-6 * scan.R[1]
        @test scan.chi2[k+1] >= scan.chi2[k] - slack_chi2
        @test scan.R[k+1] <= scan.R[k] + slack_R
    end
end

# ------------------------------------------------------------------------------------------
# EPW ablation: FR1, FR2, FR3, FR8
# ------------------------------------------------------------------------------------------

const _FR_EPW_LAMBDA = [1.0e-5, 1.0e-3, 1.0e-1] # shortened from D38's seven decades; see header
const _FR_EPW_LAMBDA_SHOWN = 2.0e-4

function _fr_epw()
    problem_nt = epw_ablation_problem(; nlambda=128, nt=32)
    expt_truth = build_experiment(problem_nt)
    sd = synthesize(expt_truth, problem_nt.noise; rng=Xoshiro(problem_nt.seed), source="epw_ablation_128x32")

    return mktempdir() do dir
        sd_path = joinpath(dir, "epw_ablation_128x32_streak.jld2")
        write_streak(sd_path, sd; source="epw_ablation_128x32")
        deck = _fr_deck_for_cutdown("epw_ablation", sd_path)
        problem, optimizer, start = OTIS.build_problem(deck)

        truth = Dict{String,Vector{Float64}}(
            "logn" => log10.(_fr_truth(problem_nt.trace, "n_e_cm3")),
            "T_e" => _fr_truth(problem_nt.trace, "electrons.e.T_eV"),
            "m_e" => _fr_truth(problem_nt.trace, "electrons.e.f.m"))

        t0 = time()
        scan = weight_scan(problem, _FR_EPW_LAMBDA, optimizer; start=start, truth=truth)
        elapsed = time() - t0

        shown = _fr_shown(problem, scan, _FR_EPW_LAMBDA_SHOWN, optimizer)
        rss_shown = _fr_rss_normalized(shown, truth, scan)

        (; scan, shown, rss_shown, elapsed)
    end
end

const _FR_EPW = _fr_epw()

@testset "FR1: the EPW problem fits (scan structure)" begin
    scan = _FR_EPW.scan
    _fr_print_scan_table("FR1 EPW", scan, _FR_EPW.elapsed)
    @test length(scan.weights) == length(_FR_EPW_LAMBDA)
    _fr_scan_monotonic(scan)
end

@testset "FR2: the EPW average RSS ratio is below one at Lambda = 2e-4" begin
    _fr_print_dict("FR2/FR3 EPW", _FR_EPW.rss_shown)
    avg = _fr_mean(_FR_EPW.rss_shown)
    @info "FR2 EPW average RSS, normalized, at Lambda=2e-4" average = avg
    @test avg < 1.0
end

@testset "FR3: the EPW per-parameter table (reported, not asserted)" begin
    @test Set(keys(_FR_EPW.rss_shown)) == Set(("logn", "T_e", "m_e"))
end

@testset "FR8: m_e in the EPW problem's first bins (reported, not asserted)" begin
    m_e = _FR_EPW.shown.values["m_e"]
    first5 = m_e[1:5]
    @info "FR8 EPW m_e, first five bins (alpha_e = 0.564 at t = 2.0 ns, noncollective there)" first5
    @test length(m_e) >= 5
end

# ------------------------------------------------------------------------------------------
# IAW multi-flow: FR4
# ------------------------------------------------------------------------------------------

const _FR_MULTIFLOW_LAMBDA = [1.0e-6, 1.0e-4, 1.0e-3, 1.0e-1] # shortened from seven decades; see header
const _FR_MULTIFLOW_LAMBDA_SHOWN = 5.0e-3

function _fr_multiflow()
    problem_nt = iaw_multiflow_problem(; nlambda=128, nt=32)
    expt_truth = build_experiment(problem_nt)
    sd = synthesize(expt_truth, problem_nt.noise; rng=Xoshiro(problem_nt.seed), source="iaw_multiflow_128x32")

    return mktempdir() do dir
        sd_path = joinpath(dir, "iaw_multiflow_128x32_streak.jld2")
        write_streak(sd_path, sd; source="iaw_multiflow_128x32")
        deck = _fr_deck_for_cutdown("iaw_multiflow", sd_path)
        problem, optimizer, start = OTIS.build_problem(deck)
        nt = nbins(problem.pspec.trace)

        truth = Dict{String,Vector{Float64}}(
            "f_D" => _fr_truth(problem_nt.trace, "ions.D.fraction"),
            "T_D" => _fr_truth(problem_nt.trace, "ions.D.T_eV"),
            "T_C" => _fr_truth(problem_nt.trace, "ions.C.T_eV"),
            "u_D" => zeros(nt), # ion D's flow is fixed at 0 km/s throughout (examples/problems.jl)
            "u_C" => _fr_truth(problem_nt.trace, "ions.C.u_kms"))

        t0 = time()
        scan = weight_scan(problem, _FR_MULTIFLOW_LAMBDA, optimizer; start=start, truth=truth)
        elapsed = time() - t0

        shown = _fr_shown(problem, scan, _FR_MULTIFLOW_LAMBDA_SHOWN, optimizer)
        rss_shown = _fr_rss_normalized(shown, truth, scan)

        (; scan, shown, rss_shown, elapsed)
    end
end

const _FR_MULTIFLOW = _fr_multiflow()

@testset "FR4: the multi-flow problem fits (structure), average RSS ratio below one" begin
    scan = _FR_MULTIFLOW.scan
    _fr_print_scan_table("FR4 multi-flow", scan, _FR_MULTIFLOW.elapsed)
    @test length(scan.weights) == length(_FR_MULTIFLOW_LAMBDA)
    _fr_scan_monotonic(scan)
    _fr_print_dict("FR4 multi-flow", _FR_MULTIFLOW.rss_shown)
    avg = _fr_mean(_FR_MULTIFLOW.rss_shown)
    @info "FR4 multi-flow average RSS, normalized, at Lambda=5e-3" average = avg
    @test avg < 1.0
end

# ------------------------------------------------------------------------------------------
# IAW shock: FR5, FR6, FR7
# ------------------------------------------------------------------------------------------
#
# No `truth` is passed to this problem's `weight_scan` -- D38/Section 10 asks for no per-parameter
# RSS table here (only FR2/FR3 for EPW and the RSS clause folded into FR4 for multi-flow); FR5-FR7
# are structural/total-variation checks that read `.values` directly. This also sidesteps a real
# shape mismatch a first draft of this file got wrong: `FitResult.values["u_pre"]` is the FULL
# nt-bin row (parameters.jl's `_value_table`), holding `NaN` outside `u_pre`'s own "pre" interval
# and its fitted value inside it -- NOT a short vector truncated to the interval's own bins. A
# `truth` entry sliced to the interval's length therefore cannot be compared against it elementwise
# (weight_scan.jl's own `tvals .- r.values[name]` assumes equal lengths); this is what "covers only
# the pre-shock bins" in FR7 means, and FR6's total variation below slices `.values[v]` to each
# variable's own bins for the same reason before differencing.

const _FR_SHOCK_LAMBDA = [1.0e-5, 1.0e-3, 1.0e-2, 1.0e-1] # shortened from seven decades; see header
const _FR_SHOCK_LAMBDA_A = 1.0e-2
const _FR_SHOCK_LAMBDA_B = 1.0e-1

function _fr_shock()
    problem_nt = iaw_shock_problem(; nlambda=128, nt=32)
    expt_truth = build_experiment(problem_nt)
    sd = synthesize(expt_truth, problem_nt.noise; rng=Xoshiro(problem_nt.seed), source="iaw_shock_128x32")

    return mktempdir() do dir
        sd_path = joinpath(dir, "iaw_shock_128x32_streak.jld2")
        write_streak(sd_path, sd; source="iaw_shock_128x32")
        deck = _fr_deck_for_cutdown("iaw_shock", sd_path)
        problem, optimizer, start = OTIS.build_problem(deck)

        pre_iv = problem.pspec.intervals[findfirst(iv -> iv.name == "pre", problem.pspec.intervals)]
        post_iv = problem.pspec.intervals[findfirst(iv -> iv.name == "post", problem.pspec.intervals)]

        t0 = time()
        scan = weight_scan(problem, _FR_SHOCK_LAMBDA, optimizer; start=start)
        elapsed = time() - t0

        idx_a = findfirst(==(_FR_SHOCK_LAMBDA_A), scan.weights)
        idx_b = findfirst(==(_FR_SHOCK_LAMBDA_B), scan.weights)

        (; scan, elapsed, idx_a, idx_b, pre_bins=pre_iv.bins, post_bins=post_iv.bins,
            nt=nbins(problem.pspec.trace))
    end
end

const _FR_SHOCK = _fr_shock()

@testset "FR5: the shock problem fits at Lambda = 1e-2 and 1e-1" begin
    scan = _FR_SHOCK.scan
    _fr_print_scan_table("FR5 shock", scan, _FR_SHOCK.elapsed)
    @test _FR_SHOCK.idx_a !== nothing
    @test _FR_SHOCK.idx_b !== nothing
    @test scan.results[_FR_SHOCK.idx_a] isa FitResult
    @test scan.results[_FR_SHOCK.idx_b] isa FitResult
end

@testset "FR6: the shock's regularized traces are smoother at Lambda = 1e-1 than at Lambda = 0" begin
    scan = _FR_SHOCK.scan
    reg = scan.results[_FR_SHOCK.idx_b]
    unreg = scan.unregularized
    own_bins = Dict("f_D" => 1:_FR_SHOCK.nt, "T_D" => 1:_FR_SHOCK.nt, "T_C" => 1:_FR_SHOCK.nt,
        "u_pre" => _FR_SHOCK.pre_bins, "u_post" => _FR_SHOCK.post_bins)
    for v in ("f_D", "T_D", "T_C", "u_pre", "u_post")
        bins = own_bins[v]
        tv_reg = _fr_total_variation(reg.values[v][bins])
        tv_unreg = _fr_total_variation(unreg.values[v][bins])
        @info "FR6 shock total variation" variable = v tv_unreg tv_reg
        @test tv_reg < tv_unreg
    end
end

@testset "FR7: the shock's interval form runs" begin
    scan = _FR_SHOCK.scan
    reg = scan.results[_FR_SHOCK.idx_a]
    other_of_pre = setdiff(1:_FR_SHOCK.nt, _FR_SHOCK.pre_bins)
    other_of_post = setdiff(1:_FR_SHOCK.nt, _FR_SHOCK.post_bins)
    @info "FR7 shock interval bins" pre_bins=_FR_SHOCK.pre_bins post_bins=_FR_SHOCK.post_bins
    @test all(isfinite, reg.values["u_pre"][_FR_SHOCK.pre_bins])
    @test all(!isfinite, reg.values["u_pre"][other_of_pre])
    @test all(isfinite, reg.values["u_post"][_FR_SHOCK.post_bins])
    @test all(!isfinite, reg.values["u_post"][other_of_post])
end
