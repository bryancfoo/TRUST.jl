# foo2026 Section III.C / Fig. 5 / Table I: the IAW shock full-resolution reproduction. Spec D38,
# D31. Run: julia --project=. examples/fit_iaw_shock.jl
#
# NOT RUN as of Chunk 13 (dev/reports/phase3_report.md): a single fit of this deck's own
# `noisy_descent_lbfgs` procedure (NoisyGradientDescent's default 2000 steps) measured at roughly
# 87 minutes at full resolution -- see the Chunk 13 report for the measurement and the user's
# 2026-09-16 direction to report the cost rather than spend several hours of compute running it.
# This script is otherwise complete and ready to run once that is revisited.
#
# Synthesizes the full-resolution (512 x 256) ground truth by running examples/iaw_shock.jl
# (included below; its own `main()` call writes runs/iaw_shock_streak.jld2, the streak
# examples/iaw_shock.toml fits), then read_deck's/build_problem's that deck, fits it
# unregularized, and scans the six decades of D38's Lambda list. Both of the paper's own chosen
# weights for this problem (Fig. 5), Lambda = 1e-2 and 1e-1, are members of that seven-point list,
# so both come directly from the scan's own `results` -- no extra warm-started fit is needed here,
# unlike the other two reproductions. Writes runs/iaw_shock/result.jld2 (the Lambda = 1e-1 fit,
# Fig. 5's main panel -- examples/iaw_shock.toml's own [regularizer].weight, D38 -- with the whole
# scan attached) and runs/iaw_shock/effective_deck.toml.
#
# No `truth` is passed to `weight_scan` here, matching test/slow/test_reproduce.jl's FR5-FR7 (D38
# asks for no per-parameter RSS table for this problem): `FitResult.values["u_pre"]` is the FULL
# nt-bin row (parameters.jl's `_value_table`), NaN outside `u_pre`'s own "pre" interval, not a
# short vector truncated to the interval -- a `truth` entry sliced to the interval's own length
# cannot be compared against it elementwise (weight_scan.jl's `tvals .- r.values[name]` assumes
# equal lengths). The total-variation comparison below slices `.values[v]` to each variable's own
# bins for the same reason before differencing.

using OTIS
using Random: Xoshiro

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "iaw_shock.jl")) # writes runs/iaw_shock_streak.jld2 (its own main())

const LAMBDA_SCAN = [1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
const LAMBDA_A = 1.0e-2
const LAMBDA_B = 1.0e-1 # the deck's own [regularizer].weight -- Fig. 5's main panel

# D34 item 3: refuse to write a result carrying a non-converged fit. `OTIS_ALLOW_UNCONVERGED=1`
# downgrades the refusal to a loud warning for a diagnostic run; it defaults off, and a file
# written under it is a diagnostic, never a reproduction.
const ALLOW_UNCONVERGED = get(ENV, "OTIS_ALLOW_UNCONVERGED", "0") == "1"

function _check_converged(scan::WeightScanResult,
    extra::AbstractVector{<:Pair{<:AbstractString,<:FitResult}})
    offenders = String[]
    scan.unregularized.converged ||
        push!(offenders, "unregularized (status = $(scan.unregularized.status))")
    for (w, r) in zip(scan.weights, scan.results)
        r.converged || push!(offenders, "Lambda = $w (status = $(r.status))")
    end
    for (label, r) in extra
        r.converged || push!(offenders, "$label (status = $(r.status))")
    end
    isempty(offenders) && return nothing
    msg = "not converged, refusing to write a result: " * join(offenders, "; ")
    ALLOW_UNCONVERGED ? (@warn "$msg (OTIS_ALLOW_UNCONVERGED=1: writing anyway)") : error(msg)
    return nothing
end

_total_variation(v::AbstractVector) = sum(abs(v[j+1] - v[j]) for j in 1:(length(v)-1))

function run_reproduction()
    t_start = time()

    deck = read_deck(joinpath(@__DIR__, "iaw_shock.toml"))
    problem, optimizer, start = OTIS.build_problem(deck)

    pre_iv = problem.pspec.intervals[findfirst(iv -> iv.name == "pre", problem.pspec.intervals)]
    post_iv = problem.pspec.intervals[findfirst(iv -> iv.name == "post", problem.pspec.intervals)]
    nt = nbins(problem.pspec.trace)

    t_scan = @elapsed scan = weight_scan(problem, LAMBDA_SCAN, optimizer; start=start)
    @info "iaw_shock: seven-point scan" LAMBDA_SCAN scan.chi2_normalized scan.R_normalized t_scan_s = t_scan

    idx_a = findfirst(==(LAMBDA_A), scan.weights)
    idx_b = findfirst(==(LAMBDA_B), scan.weights)
    result_a, result_b = scan.results[idx_a], scan.results[idx_b]
    @info "iaw_shock: fit at Lambda = 1e-2" chi2 = result_a.chi2 R = result_a.R
    @info "iaw_shock: fit at Lambda = 1e-1 (D38, Fig. 5 main panel)" chi2 = result_b.chi2 R = result_b.R

    own_bins = Dict("f_D" => 1:nt, "T_D" => 1:nt, "T_C" => 1:nt, "u_pre" => pre_iv.bins, "u_post" => post_iv.bins)
    for v in ("f_D", "T_D", "T_C", "u_pre", "u_post")
        bins = own_bins[v]
        tv_unreg = _total_variation(scan.unregularized.values[v][bins])
        tv_a = _total_variation(result_a.values[v][bins])
        tv_b = _total_variation(result_b.values[v][bins])
        @info "iaw_shock: total variation" variable = v tv_unreg tv_at_1e_2 = tv_a tv_at_1e_1 = tv_b
    end

    _check_converged(scan, ["shown (Lambda = $LAMBDA_B)" => result_b])

    outdir = deck.sections["output"]["dir"]
    mkpath(outdir)
    result_path = joinpath(outdir, "result.jld2")
    eff = effective_deck(deck, problem, optimizer)
    write_result(result_path, result_b; deck=eff, source="fit_iaw_shock", scan=scan)
    open(joinpath(outdir, "effective_deck.toml"), "w") do io
        write(io, eff)
    end

    t_total = time() - t_start
    @info "iaw_shock: end to end" t_total_s = t_total result_bytes = filesize(result_path) result_path
    return result_a, result_b, scan
end

run_reproduction()
