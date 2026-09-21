# foo2026 Section III.A / Table I: the EPW ablation full-resolution reproduction. Spec D38, D31.
# Run: julia --project=. examples/fit_epw_ablation.jl
#
# NOT RUN as of Chunk 13 (dev/reports/phase3_report.md): a single fit of this deck's own
# `noisy_descent_lbfgs` procedure (NoisyGradientDescent's default 2000 steps) measured at roughly
# 87 minutes at full resolution -- see the Chunk 13 report for the measurement and the user's
# 2026-09-16 direction to report the cost rather than spend several hours of compute running it.
# This script is otherwise complete and ready to run once that is revisited.
#
# Synthesizes the full-resolution (512 x 256) ground truth by running examples/epw_ablation.jl
# (included below; its own `main()` call writes runs/epw_ablation_streak.jld2, the streak
# examples/epw_ablation.toml fits, with sigma already frozen at synthesis, OQ-14), then
# read_deck's/build_problem's that deck, fits it unregularized, scans the six decades of D38's
# Lambda list (foo2026's own, one point per decade), and fits once more at the paper's own chosen
# Lambda = 2e-4 (D38, Fig. 1) -- warm-started from the scan's nearest neighbour in log10(Lambda),
# reusing the scan's own unregularized fit and rss_unreg (D31's normalization is "by the
# unregularized fit's own RSS for that variable", the SAME fit, not a fresh cold-start one).
# Writes runs/epw_ablation/result.jld2 (the Lambda = 2e-4 fit "shown" by D38, with the whole scan
# attached) and runs/epw_ablation/effective_deck.toml.

using OTIS
using OTIS: _with_weight, _leaf_si_path
using Random: Xoshiro

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "epw_ablation.jl")) # writes runs/epw_ablation_streak.jld2 (its own main())

const LAMBDA_SCAN = [1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
const LAMBDA_SHOWN = 2.0e-4

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

function _leaf_truth(trace::PlasmaTrace, boundary_key::AbstractString)
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    i === nothing && error("_leaf_truth: leaf \"$si_path\" not present in this trace")
    return collect(Float64, trace.values[i, :]) ./ factor
end

function _shown_fit(problem::FitProblem, scan::WeightScanResult, weight::Float64, optimizer)
    idx = findfirst(==(weight), scan.weights)
    idx !== nothing && return scan.results[idx]
    logs = log10.(scan.weights)
    nearest = scan.results[argmin(abs.(logs .- log10(weight)))]
    return solve(_with_weight(problem, weight), optimizer; start=nearest.theta)
end

function _rss_normalized(result::FitResult, truth::Dict{String,Vector{Float64}}, scan::WeightScanResult)
    return Dict{String,Float64}(v => sum(abs2, truth[v] .- result.values[v]) / scan.rss_unreg[v]
                                 for v in keys(truth))
end

function run_reproduction()
    t_start = time()

    deck = read_deck(joinpath(@__DIR__, "epw_ablation.toml"))
    problem, optimizer, start = OTIS.build_problem(deck)

    problem_nt = epw_ablation_problem()
    truth = Dict{String,Vector{Float64}}(
        "logn" => log10.(_leaf_truth(problem_nt.trace, "n_e_cm3")),
        "T_e" => _leaf_truth(problem_nt.trace, "electrons.e.T_eV"),
        "m_e" => _leaf_truth(problem_nt.trace, "electrons.e.f.m"))

    t_scan = @elapsed scan = weight_scan(problem, LAMBDA_SCAN, optimizer; start=start, truth=truth)
    @info "epw_ablation: seven-point scan" LAMBDA_SCAN scan.chi2_normalized scan.R_normalized t_scan_s = t_scan

    t_shown = @elapsed shown = _shown_fit(problem, scan, LAMBDA_SHOWN, optimizer)
    rss_shown = _rss_normalized(shown, truth, scan)
    avg = sum(values(rss_shown)) / length(rss_shown)
    for v in sort(collect(keys(rss_shown)))
        @info "epw_ablation: fit shown (Lambda = 2e-4, D38) per-parameter row" variable = v rss_normalized = rss_shown[v]
    end
    @info "epw_ablation: fit shown (Lambda = 2e-4, D38)" average_rss = avg t_shown_s = t_shown chi2 = shown.chi2 R = shown.R converged = shown.converged iterations = shown.iterations

    _check_converged(scan, ["shown (Lambda = $LAMBDA_SHOWN)" => shown])

    outdir = deck.sections["output"]["dir"]
    mkpath(outdir)
    result_path = joinpath(outdir, "result.jld2")
    eff = effective_deck(deck, problem, optimizer)
    write_result(result_path, shown; deck=eff, source="fit_epw_ablation", scan=scan)
    open(joinpath(outdir, "effective_deck.toml"), "w") do io
        write(io, eff)
    end

    t_total = time() - t_start
    @info "epw_ablation: end to end" t_total_s = t_total result_bytes = filesize(result_path) result_path
    return shown, scan
end

run_reproduction()
