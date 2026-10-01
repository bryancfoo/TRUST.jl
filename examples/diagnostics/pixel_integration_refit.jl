# Step 4.F.3: the refit at the count the resonance-width check asks for. Spec
# dev/phase4F_instructions.md D19 and D20 (OQ-53, the user's decision of 2026-09-30, its (iv) and (v)).
#
# A MEASUREMENT, not a fix: nothing here changes a shape, a table, a deck, the forward model or an
# optimizer, and nothing is built on what it shows. Two measurements, each printed in full:
#
#   (v) Phase 4.D's A3. The cut-down EPW reproduction's reported fit, exactly as the slow tier makes
#       it (test/slow/test_reproduce.jl: epw_ablation_problem at 128 x 32, weight_scan over 1e-5,
#       1e-3 and 1e-1 from the deck's start, then the fit at FR2's weight 2e-4 warm-started from the
#       scan's nearest point), whose SuperGaussian m sits on its upper bound 5.5 in 17 of 32 bins;
#       the width check on it; then the asked count, the deck's start and weight 2e-4, against the
#       same fit at count 1. BLAS at its default thread count throughout, as the slow tier runs it.
#   (iv) The shapes example's GeneralizedNormal case: its two streaks synthesized as
#       examples/shapes.jl does (bitwise runs/shapes_generalized_normal_<camera>_streak.jld2 when
#       those files exist), then fit(deck) on examples/fit_shapes.jl's deck with procedure =
#       "staged" at count 1 (Phase 4.D's fit: 174 iterations, EPW chi2/n 41.2117, bin 25's m 5.4658),
#       the width check's entries, and the same at the count the check asks for. fit(deck) pins BLAS
#       to one thread; the streaks are synthesized before it does.
#
# (v) runs first so that its fits see BLAS's default, as the slow tier's do. Its output goes to the
# phase report, never to a committed file; the streaks and decks it writes live in a temporary
# directory and are deleted.
#
# Run:
#   julia --project=. examples/diagnostics/pixel_integration_refit.jl
#
# Budget: 60 minutes (hard limit 90); Level 2's run took 17.4 minutes on a quiet machine, 13.1 of
# them the count-64 fit of (v) (spec D20).

using OTIS
using OTIS: _with_weight, _leaf_si_path
using Printf
using Random: Xoshiro
using LinearAlgebra: BLAS
import TOML

include(joinpath(@__DIR__, "..", "problems.jl"))

const _EXAMPLES_DIR = normpath(joinpath(@__DIR__, ".."))
const _RUNS_DIR = normpath(joinpath(@__DIR__, "..", "..", "runs"))

function say(args...)
    println(args...)
    flush(stdout)
    return nothing
end

# The fitted m in every bin beside its truth, its EPW resonance's width in pixels ("-" for none) and
# whether it is near a bound; then the width check's entries and the count they ask for.
function report(label, r, truth_m)
    w = OTIS._resonance_widths(r)[1]
    m = r.values["m_e"]
    upper = [e.bin for e in OTIS._near_bound_entries(r) if e.variable == "m_e" && e.bound === :upper]
    say(@sprintf("%s: status %s, %d iterations; EPW chi2/n %.4f; m_e on its upper bound in %d bins %s", label, r.status,
        r.iterations, r.channels[1].chi2 / r.channels[1].npixels, length(upper), string(upper)))
    for j in eachindex(m)
        say(@sprintf("   bin %2d: m %.4f (truth %.4f)  FWHM %s%s", j, m[j], truth_m[j],
            isinf(w[j]) ? "-" : @sprintf("%.3f px", w[j]), r.near_bound["m_e"][j] ? "  [near a bound]" : ""))
    end
    entries = OTIS._narrow_resonance_entries(r, OTIS._resonance_widths(r))
    asked = isempty(entries) ? 1 : maximum(OTIS._recommended_subsamples(e.fwhm_px) for e in entries)
    say(@sprintf("   the width check: %d entries %s; it asks for %d", length(entries),
        join((@sprintf("bin %d %.3f px", e.bin, e.fwhm_px) for e in entries), ", "), asked))
    return asked
end

function _truth_row(trace::PlasmaTrace, boundary_key::AbstractString)
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    return collect(Float64, trace.values[i, :]) ./ factor
end

# ------------------------------------------------------------------------------------------
# (v) Phase 4.D's A3: the cut-down EPW reproduction
# ------------------------------------------------------------------------------------------

function epw_deck(sd_path; count=nothing)
    deck = read_deck(joinpath(_EXAMPLES_DIR, "epw_ablation.toml"))
    s = deepcopy(deck.sections)
    s["data"] = merge(s["data"], Dict{String,Any}("file" => sd_path))
    count === nothing || (s["instrument"]["pixel_integration"] = Dict{String,Any}("subsamples" => count))
    return OTIS.Deck(deck.deck_version, s)
end

function measurement_v(dir)
    say("(v) Phase 4.D's A3: the cut-down EPW reproduction (BLAS ", BLAS.get_num_threads(), " threads)")
    problem_nt = epw_ablation_problem(; nlambda=128, nt=32)
    sd = synthesize(build_experiment(problem_nt), problem_nt.noise; rng=Xoshiro(problem_nt.seed), source="epw_ablation_128x32")
    sd_path = joinpath(dir, "epw_ablation_128x32_streak.jld2")
    write_streak(sd_path, sd; source="epw_ablation_128x32")
    truth = Dict("logn" => log10.(_truth_row(problem_nt.trace, "n_e_cm3")),
        "T_e" => _truth_row(problem_nt.trace, "electrons.e.T_eV"), "m_e" => _truth_row(problem_nt.trace, "electrons.e.f.m"))
    problem, optimizer, start = OTIS.build_problem(epw_deck(sd_path))
    say(@sprintf("   pixel pitch %.6f nm", (problem.experiment.lambda[2] - problem.experiment.lambda[1]) * 1e9))

    t = @elapsed begin
        scan = weight_scan(problem, [1.0e-5, 1.0e-3, 1.0e-1], optimizer; start=start, truth=truth)
        logs = log10.(scan.weights)
        nearest = scan.results[argmin(abs.(logs .- log10(2.0e-4)))]
        shown = solve(_with_weight(problem, 2.0e-4), optimizer; start=nearest.theta)
    end
    say(@sprintf("   [time] the scan and FR2's fit: %.1f s", t))
    asked = report("count 1, FR2's fit (Lambda 2e-4)", shown, truth["m_e"])

    th_truth = flatten(problem.pspec, ["logn" => truth["logn"], "T_e" => truth["T_e"], "m_e" => truth["m_e"]])
    expt_t = unflatten(problem.experiment, problem.pspec, th_truth)
    wt = Float64[]
    for j in 1:nbins(expt_t.trace)
        ws = OTIS._epw_resonance_widths(plasma(expt_t.trace, j), expt_t.probe, expt_t.collection, expt_t.lambda, true)
        push!(wt, isempty(ws) ? Inf : minimum(ws))
    end
    say("   the truth's widths: ", join((isinf(x) ? "-" : @sprintf("%.2f", x) for x in wt), " "))

    asked == 1 && (say("   no count asked: no refit"); return)
    t = @elapsed r1 = solve(_with_weight(problem, 2.0e-4), optimizer; start=start)
    say(@sprintf("   [time] count 1 from the deck's start: %.1f s", t))
    report("count 1, Lambda 2e-4, from the deck's start", r1, truth["m_e"])
    pN, oN, sN = OTIS.build_problem(epw_deck(sd_path; count=asked))
    t = @elapsed rN = solve(_with_weight(pN, 2.0e-4), oN; start=sN)
    say(@sprintf("   [time] count %d from the deck's start: %.1f s", asked, t))
    report("count $asked, Lambda 2e-4, from the deck's start", rN, truth["m_e"])
    return
end

# ------------------------------------------------------------------------------------------
# (iv) the shapes example's GeneralizedNormal case, through fit(deck)
# ------------------------------------------------------------------------------------------

function shapes_deck_path(dir, paths, tag; count=nothing)
    text = with_wavelength_calibration(read(joinpath(_EXAMPLES_DIR, "shapes_generalized_normal.toml"), String))
    text = with_staged_procedure(text; stage_procedure="levenberg_marquardt", start_from=Dict{String,Any}("T_D" => "iaw"))
    raw = TOML.parse(text)
    raw["channel"]["epw"]["data"]["file"] = replace(paths[1], "\\" => "/")
    raw["channel"]["iaw"]["data"]["file"] = replace(paths[2], "\\" => "/")
    raw["output"]["dir"] = replace(joinpath(dir, "out_$tag"), "\\" => "/")
    count === nothing || (raw["channel"]["epw"]["instrument"]["pixel_integration"] = Dict{String,Any}("subsamples" => count))
    buf = IOBuffer()
    TOML.print(buf, raw; sorted=true)
    path = joinpath(dir, "deck_$tag.toml")
    write(path, String(take!(buf)))
    return path
end

function measurement_iv(dir)
    say("(iv) the shapes example's GeneralizedNormal case (synthesis at BLAS ", BLAS.get_num_threads(), " threads)")
    prob = shapes_problem(:generalized_normal)
    paths = String[]
    for camera in (:epw, :iaw)
        sd = synthesize_channel(prob, camera)
        path = joinpath(dir, "gn_$(camera)_streak.jld2")
        write_streak(path, sd; source="pixel_integration_refit", probe=prob.probe, collection=prob.collection)
        push!(paths, path)
        ref = joinpath(_RUNS_DIR, "shapes_generalized_normal_$(camera)_streak.jld2")
        same = isfile(ref) ? (r = read_streak(ref); all(r.image .=== sd.image) && all(r.sigma .=== sd.sigma)) : nothing
        say("   synthesized ", camera, ": ", size(sd.image), "; bitwise runs/shapes_generalized_normal_$(camera)_streak.jld2: ",
            same === nothing ? "no such file" : string(same))
    end
    base = Dict(prob.leaves)
    function one_fit(tag, count)
        t = @elapsed r = fit(shapes_deck_path(dir, paths, tag; count=count))
        say(@sprintf("   [time] fit(deck), count %d: %.1f s", count === nothing ? 1 : count, t))
        truth_m = base["electrons.e.f.m"].(t_from_si.(r.pspec.axis, :t_ns))
        say(@sprintf("   IAW chi2/n %.4f; bins with |m - truth| > 0.2: %s", r.channels[2].chi2 / r.channels[2].npixels,
            string(findall(abs.(r.values["m_e"] .- truth_m) .> 0.2))))
        return report("count $(count === nothing ? 1 : count), the staged fit", r, truth_m)
    end
    asked = one_fit("count1", nothing)
    asked == 1 && (say("   no count asked: no refit"); return)
    one_fit("asked", asked)
    return
end

function main()
    t0 = time()
    mktempdir() do dir
        measurement_v(dir)
        measurement_iv(dir)
    end
    say(@sprintf("[time] total %.1f s", time() - t0))
end

main()
