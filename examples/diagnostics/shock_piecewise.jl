# Step 4.C.2: the shock fitted piecewise, against the whole-axis fit, at cut-down. Spec
# dev/phase4C_instructions.md D12, D15, D17, D18 and D25 (OQ-47).
#
# A MEASUREMENT, not a fix: nothing here changes a deck, a weight or an optimizer default. It
# reproduces the numbers behind the slow tier's FR9 so that anyone can rerun them, and records the
# effect of the gap's half-width delta once. Two things are measured, both at cut-down (128 x 32),
# both as FR9 does: examples/iaw_shock.toml (the Fig. 5 reproduction, every D2 term on the whole
# axis) is fitted over a four-point weight scan [1e-5, 1e-3, 1e-2, 1e-1], and
# examples/iaw_shock_piecewise.toml (every D2 term on pre_smooth or post_smooth, so the bins within
# delta of the shock carry no regularizer row) is fitted at Lambda = 1e-1, warm-started from the
# whole-axis Lambda = 1e-1 minimizer (the theta layout is the same, and checked here).
#   1. FR9's comparison at three noise seeds (20260903, the problem's own; 20260928; 20260929), at
#      the deck's delta = 0.1 ns;
#   2. the effect of delta in {0.1, 0.05, 0.2, 0.3} ns on the problem's own seed. delta is FIXED at
#      0.1 ns in the deck (spec D12) and is never tuned to a number: this table only records what
#      moving it does. At cut-down any delta in (0.012, 0.129) ns removes bin 10 alone, so 0.05
#      and 0.1 are one problem there, and the last line says whether their fits are bitwise equal.
# Per fit it prints the status, iterations and elapsed time; per quantity (f_D, T_D, T_C, u_ion) the
# jump-window RSS against the truth for the piecewise, whole-axis and unregularized fits, their
# ratio, the same ratio over all 32 bins, and the 10-90 % widths of the jump (the truth's own on
# the bins, whole-axis, piecewise, unregularized; "undefined" when a crossing is missing from the
# window); then the mean ratio over f_D, T_D and T_C, FR9's asserted quantity. The metric is
# examples/problems.jl's (iaw_shock_truth, jump_window, jump_rss_ratios, jump_width).
#
# Run (from the slow tier's own environment, as FR9 runs):
#   julia --project=test/slow examples/diagnostics/shock_piecewise.jl
#
# Budget: 10 minutes (hard limit 30); the first measurement took about three on the machine it was
# written on. Its output goes to the phase report, never to a committed file. Nothing here writes a
# result file or a figure, and nothing is run at full resolution (spec D25): at 512 x 256 each gap
# bin is a third of the 0.05 ns temporal IRF and the comparison is not well determined.

using TRUST
using Random: Xoshiro
using Printf
using LinearAlgebra: BLAS

include(joinpath(@__DIR__, "..", "problems.jl"))

const _EXAMPLES_DIR = joinpath(@__DIR__, "..")

const SEEDS = [20260903, 20260928, 20260929]
const OWN_SEED = 20260903                  # iaw_shock_problem's own, the seed FR5 to FR9 fit
const DELTAS_OWN = [0.1, 0.05, 0.2, 0.3]   # ns; 0.1 is the committed deck's
const DELTAS_OTHER = [0.1]
const SCAN_WEIGHTS = [1.0e-5, 1.0e-3, 1.0e-2, 1.0e-1]
const SHOWN_WEIGHT = 1.0e-1
const SHOCK_TIME_NS = 1.15
const QUANTITIES = ("f_D", "T_D", "T_C", "u_ion")

# One line at a time, flushed as it is printed (a killed run loses only what was in flight).
function say(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
    return nothing
end

# `_fr_deck_for_cutdown`'s body, for any deck path: the deck's own validated sections, with
# [data].file replaced by the cut-down streak this script just wrote.
function deck_on(path::AbstractString, sd_path::AbstractString)
    deck = read_deck(path)
    sections = deepcopy(deck.sections)
    sections["data"] = merge(sections["data"], Dict{String,Any}("file" => sd_path))
    return TRUST.Deck(deck.deck_version, sections)
end

# The committed piecewise deck's text with its two regularizer-interval lines rewritten to
# [0.0, 1.15 - delta] and [1.15 + delta, 4.0] (delta rounded to six digits). Each line must be
# found exactly once, or the rewrite is refused.
function piecewise_text(delta::Real)
    text = read(joinpath(_EXAMPLES_DIR, "iaw_shock_piecewise.toml"), String)
    hi = round(SHOCK_TIME_NS - delta; digits=6)
    lo = round(SHOCK_TIME_NS + delta; digits=6)
    for (pattern, line) in ((r"^pre_smooth\s*=.*$"m, "pre_smooth  = { t_range_ns = [0.0, $hi] }"),
                            (r"^post_smooth\s*=.*$"m, "post_smooth = { t_range_ns = [$lo, 4.0] }"))
        count(_ -> true, eachmatch(pattern, text)) == 1 ||
            error("piecewise_text: expected exactly one line matching $pattern in iaw_shock_piecewise.toml")
        text = replace(text, pattern => line)
    end
    return text
end

# ... written to a temporary file (in `dir`), whose path is returned.
function piecewise_deck_path(dir::AbstractString, delta::Real)
    path = joinpath(dir, "iaw_shock_piecewise_delta_$(delta).toml")
    write(path, piecewise_text(delta))
    return path
end

same_layout(a, b) = a.ntheta == b.ntheta && a.offsets == b.offsets &&
    a.bin_entries == b.bin_entries && a.global_entries == b.global_entries

width_str(t_ns, u, a, b, J) = (w = jump_width(t_ns, u, a, b, J); w === nothing ? "undefined" : @sprintf("%.4f", w))

bits(theta) = reinterpret(UInt64, collect(Float64, theta))

"""
    run_seed(seed, deltas) -> (; thetas, means)

Synthesize the cut-down shock streak with `Xoshiro(seed)`, build the committed whole-axis deck on
it, scan the four weights, then for each `delta` build the piecewise deck on the same streak, check
its theta layout is the whole-axis deck's, solve warm-started from the scan's Lambda = 1e-1
minimizer, and print the comparison. `thetas` and `means` are keyed by `delta`.
"""
function run_seed(seed::Integer, deltas)
    say("=" ^ 100)
    say("seed ", seed, ": synthesize 128 x 32; whole-axis deck iaw_shock.toml; weight scan over ", SCAN_WEIGHTS)
    problem_nt = iaw_shock_problem(; nlambda=128, nt=32)
    sd = synthesize(build_experiment(problem_nt), problem_nt.noise; rng=Xoshiro(seed),
        source="iaw_shock_128x32_seed$seed")
    thetas = Dict{Float64,Vector{Float64}}()
    means = Dict{Float64,Float64}()
    mktempdir() do dir
        sd_path = joinpath(dir, "iaw_shock_128x32_seed$(seed)_streak.jld2")
        write_streak(sd_path, sd; source="iaw_shock_128x32_seed$seed")
        whole_problem, optimizer, start = TRUST.build_problem(
            deck_on(joinpath(_EXAMPLES_DIR, "iaw_shock.toml"), sd_path))
        nt = nbins(whole_problem.pspec.trace)

        t0 = time()
        scan = weight_scan(whole_problem, SCAN_WEIGHTS, optimizer; start=start)
        say(@sprintf("  scan elapsed %.1f s", time() - t0))
        for (k, w) in enumerate(scan.weights)
            r = scan.results[k]
            say(@sprintf("  scan row: Lambda = %-8g chi2 = %.6e R = %.6e status = %s iterations = %d converged = %s",
                w, r.chi2, r.R, r.status, r.iterations, r.converged))
        end
        u = scan.unregularized
        say(@sprintf("  unregularized: chi2 = %.6e status = %s iterations = %d converged = %s",
            u.chi2, u.status, u.iterations, u.converged))
        whole = scan.results[findfirst(==(SHOWN_WEIGHT), scan.weights)]

        t_ns = whole_problem.pspec.axis_boundary
        J = jump_window(t_ns)
        truth = iaw_shock_truth(problem_nt.trace)
        steps = iaw_shock_steps()
        say("  the jump window J = ", J, " (t = ", @sprintf("%.3f", t_ns[first(J)]), " to ",
            @sprintf("%.3f", t_ns[last(J)]), " ns); the whole-axis deck has ",
            whole_problem.regularizer.nrows, " regularizer rows")

        for delta in deltas
            problem, opt, _ = TRUST.build_problem(deck_on(piecewise_deck_path(dir, delta), sd_path))
            layout = same_layout(problem.pspec, whole_problem.pspec)
            covered = union((problem.pspec.intervals[t.interval].bins for t in problem.regularizer.terms)...)
            gap = setdiff(1:nt, covered)
            say(@sprintf("  delta = %g ns: gap bins %s, %d regularizer rows (whole-axis %d), theta layout as the whole-axis deck's: %s",
                delta, gap, problem.regularizer.nrows, whole_problem.regularizer.nrows, layout))
            layout || error("run_seed: the piecewise deck's theta layout differs from the whole-axis deck's; the warm start is invalid")
            t1 = time()
            pw = solve(problem, opt; start=whole.theta)
            wall = time() - t1
            say(@sprintf("    piecewise fit: status = %s iterations = %d converged = %s elapsed = %.2f s (wall %.2f s) chi2 = %.6e R = %.6e; whole-axis fit: status = %s iterations = %d converged = %s elapsed = %.2f s",
                pw.status, pw.iterations, pw.converged, pw.elapsed, wall, pw.chi2, pw.R,
                whole.status, whole.iterations, whole.converged, whole.elapsed))
            for v in QUANTITIES
                a, b = steps[v]
                rss_pw = jump_rss(pw.values[v], truth[v], J)
                rss_whole = jump_rss(whole.values[v], truth[v], J)
                rss_unreg = jump_rss(u.values[v], truth[v], J)
                all_ratio = jump_rss(pw.values[v], truth[v], 1:nt) / jump_rss(whole.values[v], truth[v], 1:nt)
                say(@sprintf("    %-5s window RSS: piecewise %.6g whole-axis %.6g unregularized %.6g; ratio %.4f; ratio over all %d bins %.4f; widths (ns): truth %s whole-axis %s piecewise %s unregularized %s",
                    v, rss_pw, rss_whole, rss_unreg, rss_pw / rss_whole, nt, all_ratio,
                    width_str(t_ns, truth[v], a, b, J), width_str(t_ns, whole.values[v], a, b, J),
                    width_str(t_ns, pw.values[v], a, b, J), width_str(t_ns, u.values[v], a, b, J)))
            end
            mean_ratio, ratios = jump_rss_ratios(pw.values, whole.values, truth, J)
            say(@sprintf("    mean ratio (f_D, T_D, T_C) = %.4f  [f_D %.4f, T_D %.4f, T_C %.4f]  (FR9 asserts the delta = 0.1 mean below 0.85)",
                mean_ratio, ratios["f_D"], ratios["T_D"], ratios["T_C"]))
            thetas[delta] = collect(Float64, pw.theta)
            means[delta] = mean_ratio
        end
    end
    return (; thetas, means)
end

function main()
    say("BLAS.get_num_threads() = ", BLAS.get_num_threads(), ", Threads.nthreads() = ", Threads.nthreads())
    say("shock_piecewise.jl: seeds ", SEEDS, "; delta on seed ", OWN_SEED, ": ", DELTAS_OWN, ", on the others: ", DELTAS_OTHER)
    t0 = time()
    # The delta = 0.1 rewrite must be the committed deck: nothing else in this script reads it.
    mktempdir() do dir
        rewritten = read_deck(piecewise_deck_path(dir, 0.1))
        committed = read_deck(joinpath(_EXAMPLES_DIR, "iaw_shock_piecewise.toml"))
        say("the delta = 0.1 rewrite reads as the committed iaw_shock_piecewise.toml: ",
            rewritten.sections == committed.sections)
    end
    results = Dict{Int,Any}()
    for seed in SEEDS
        results[seed] = run_seed(seed, seed == OWN_SEED ? DELTAS_OWN : DELTAS_OTHER)
    end
    th = results[OWN_SEED].thetas
    say("=" ^ 100)
    say("seed ", OWN_SEED, ": delta = 0.05's theta is bitwise delta = 0.1's: ", bits(th[0.05]) == bits(th[0.1]))
    say(@sprintf("total wall time %.1f s", time() - t0))
    return nothing
end

main()
