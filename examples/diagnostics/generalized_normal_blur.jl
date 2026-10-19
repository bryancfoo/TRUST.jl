# Step 4.D.2: is the unmodelled point spread why GeneralizedNormal's EPW chi2/n reads about 41?
# Spec dev/phase4D_instructions.md D17 and D18 (OQ-50 option (a), the user's decision of
# 2026-09-29: "(a) first, then we'll see").
#
# A MEASUREMENT, not a fix: nothing here changes a shape, a table, a deck, a noise model or the
# forward model, and nothing is built on what it shows. On the two-channel shapes problem
# (examples/problems.jl's shapes_problem), GeneralizedNormal read EPW chi2/n = 41.2 against 3.2 for
# Kappa and SkewNormal (Phase 4.B's A22), its m 5.47 in one bin against 4.61. The hypothesis
# (Phase 4.B spec D11) is the mean's blur by StreakCameraNoise's point spread, which the forward
# model does not contain (OQ-40, B1 not ordered). The test: synthesize both cameras again with the
# spread on (sigma_px = 1, the problem as committed) and off (sigma_px = 0), fit each exactly as
# examples/fit_shapes.jl does (the deck with with_wavelength_calibration, staged_joint_fit), and
# compare. Kappa is the control: sigma_px = 0 changes sigma as well as the image (the pre-spread
# formula, larger, and correct for an average of independent slices, OQ-43), so chi2/n moves for
# every shape; only GeneralizedNormal's chi2/n against Kappa's at the same sigma_px says whether the
# blur is what separates them.
#
# Every streak is synthesized first, with BLAS at its default thread count, as examples/shapes.jl
# does, so that the sigma_px = 1 streaks are bitwise examples/shapes.jl's (checked against
# runs/shapes_<name>_<camera>_streak.jld2 when those files exist); then BLAS is pinned to one
# thread and the fits run, as examples/fit_shapes.jl does (conventions.md 6.6: BLAS's thread count
# moves the data's and the fit's last bits).
#
# Per shape and sigma_px it prints: the synthesis check, the staged fit's status, iterations and
# elapsed time, polish's last decrease and gn_decrease at the result, chi2/n per camera and per
# time bin (and without the camera's worst bin), the near-bound entries, the median relative error
# of every variable, and the fitted shape parameter at every bin against its truth; then a summary
# table.
#
# Run:
#   julia --project=. examples/diagnostics/generalized_normal_blur.jl
#
# Budget: 30 minutes (hard limit 60); Level 2's two runs took 10.6 and 7.0 minutes on the machine
# it was written on. Its output goes to the phase report, never to a committed file; the streaks it
# writes live in a temporary directory and are deleted.

using TRUST
using Printf
using Statistics: median
using LinearAlgebra: BLAS

include(joinpath(@__DIR__, "..", "problems.jl"))

const _EXAMPLES_DIR = normpath(joinpath(@__DIR__, ".."))
const _RUNS_DIR = normpath(joinpath(@__DIR__, "..", "..", "runs"))

# name, the deck's shape variable, and its truth leaf (examples/fit_shapes.jl's _SHAPES).
const SHAPES = ((name=:generalized_normal, shape_variable="m_e", shape_key="electrons.e.f.m"),
                (name=:kappa, shape_variable="kappa_e", shape_key="electrons.e.f.kappa"))
const SIGMA_PX = (1.0, 0.0)

function say(args...)
    println(args...)
    flush(stdout)
    return nothing
end

# shapes_problem(name) with both cameras' point spread set to sigma_px; everything else, the seeds
# included, as committed.
function problem_with_spread(name::Symbol, sigma_px::Real)
    p = shapes_problem(name)
    noise = StreakCameraNoise(sigma_px=sigma_px)
    return merge(p, (; epw=merge(p.epw, (; noise=noise)), iaw=merge(p.iaw, (; noise=noise))))
end

# Whether a streak equals examples/shapes.jl's committed-procedure file bitwise (`nothing` when the
# file does not exist).
function same_as_runs(sd::StreakData, name::Symbol, camera::Symbol)
    path = joinpath(_RUNS_DIR, "shapes_$(name)_$(camera)_streak.jld2")
    isfile(path) || return nothing
    ref = read_streak(path)
    return ref.image == sd.image && all(ref.image .=== sd.image) && all(ref.sigma .=== sd.sigma)
end

# examples/fit_shapes.jl's deck: the committed shapes deck, with_wavelength_calibration, its two
# data files pointed at this run's streaks and its output directory at `dir`.
function deck_path(dir, name::Symbol, epw_path, iaw_path)
    text = with_wavelength_calibration(read(joinpath(_EXAMPLES_DIR, "shapes_$(name).toml"), String))
    for (camera, path) in (("epw", epw_path), ("iaw", iaw_path))
        old = "runs/shapes_$(name)_$(camera)_streak.jld2"
        occursin(old, text) || error("deck_path: \"$old\" not found in shapes_$(name).toml")
        text = replace(text, old => replace(path, "\\" => "/"))
    end
    out = replace(joinpath(dir, "out_$(name)"), "\\" => "/")
    text = replace(text, "dir = \"runs/shapes/$(name)/joint\"" => "dir = \"$out\"")
    path = joinpath(dir, "deck_$(name).toml")
    write(path, text)
    return path
end

function main()
    say("BLAS.get_num_threads() = ", BLAS.get_num_threads(), ", Threads.nthreads() = ", Threads.nthreads())
    t_start = time()
    rows = NamedTuple[]
    mktempdir() do dir
        # 1. Synthesize every streak first, BLAS unpinned (examples/shapes.jl's state).
        streaks = Dict{Tuple{Symbol,Float64},Tuple{String,String}}()
        for spec in SHAPES, sp in SIGMA_PX
            p = problem_with_spread(spec.name, sp)
            paths = String[]
            for camera in (:epw, :iaw)
                t = @elapsed sd = synthesize_channel(p, camera)
                path = joinpath(dir, "$(spec.name)_sigma$(Int(sp))_$(camera)_streak.jld2")
                write_streak(path, sd; source="generalized_normal_blur", probe=p.probe, collection=p.collection)
                push!(paths, path)
                same = sp == 1.0 ? same_as_runs(sd, spec.name, camera) : "n/a (sigma_px = 0)"
                say(@sprintf("synthesized %-18s sigma_px = %.0f %s: %s in %.1f s; bitwise runs/shapes_%s_%s_streak.jld2: %s",
                    spec.name, sp, camera, string(size(sd.image)), t, spec.name, camera,
                    same === nothing ? "no such file" : string(same)))
            end
            streaks[(spec.name, sp)] = (paths[1], paths[2])
        end

        # 2. Fit each, BLAS pinned (examples/fit_shapes.jl's state).
        pin_blas_threads()
        say("pinned: BLAS.get_num_threads() = ", BLAS.get_num_threads())
        for spec in SHAPES, sp in SIGMA_PX
            say("=" ^ 100)
            say(@sprintf("%s, sigma_px = %.0f: staged_joint_fit (examples/fit_shapes.jl's procedure)", spec.name, sp))
            epw_path, iaw_path = streaks[(spec.name, sp)]
            sub = mkpath(joinpath(dir, "$(spec.name)_sigma$(Int(sp))"))
            dp = deck_path(sub, spec.name, epw_path, iaw_path)
            t = @elapsed st = staged_joint_fit(dp; shape_variable=spec.shape_variable)
            r = st.result
            t_gn = @elapsed gd = gn_decrease(st.problem, r.theta)
            chi2n = Dict(c.name => c.chi2 / c.npixels for c in r.channels)
            entries = TRUST._near_bound_entries(r)
            say(@sprintf("  status = %s converged = %s iterations = %d elapsed = %.1f s (staged, wall %.1f s); polish's last decrease = %.3e; gn_decrease = %.3e (%.1f s)",
                r.status, r.converged, r.iterations, r.elapsed, t, st.decreases[end], gd, t_gn))
            say(@sprintf("  chi2/n: EPW %.4f, IAW %.4f; stages alone: EPW %.4f, IAW %.4f",
                chi2n["epw"], chi2n["iaw"], st.epw.chi2 / st.epw.npixels, st.iaw.chi2 / st.iaw.npixels))
            # chi2/n per fitted time bin, per camera, from the pixel residuals (M - D)/sigma over the
            # pixels the fit used; and chi2/n with the camera's worst bin left out.
            worst_bins = Dict{String,Int}()
            for (c, label) in ((1, "EPW"), (2, "IAW"))
                ch = TRUST._channel_problem(st.problem, c)
                model = measure(r.experiments[c])
                used = falses(size(ch.data.image))
                used[ch.pixel_index] .= true
                sq = [used[l, j] ? ((model[l, j] - ch.data.image[l, j]) / ch.sigma[l, j])^2 : 0.0
                      for l in axes(used, 1), j in axes(used, 2)]
                per_bin = [sum(sq[:, j]) / max(count(used[:, j]), 1) for j in axes(used, 2)]
                jw = argmax(per_bin)
                worst_bins[label] = jw
                rest = sum(sq[:, setdiff(axes(used, 2), jw)]) / count(used[:, setdiff(axes(used, 2), jw)])
                say(@sprintf("  %s chi2/n per bin: %s", label, join((@sprintf("%.2f", v) for v in per_bin), " ")))
                say(@sprintf("  %s worst bin %d (chi2/n %.2f); chi2/n without it %.4f", label, jw, per_bin[jw], rest))
            end
            say("  near-bound entries: ", isempty(entries) ? "none" : join(("$(e.variable) bin $(e.bin) ($(e.bound))" for e in entries), ", "))
            p = shapes_problem(spec.name)
            base = Dict(p.leaves)
            t_ns = t_from_si.(r.pspec.axis, :t_ns)
            for (label, key, vname) in (("n_e", "n_e_cm3", "ne_cm3"), ("T_e", "electrons.e.T_eV", "T_e"),
                                        (spec.shape_variable, spec.shape_key, spec.shape_variable),
                                        ("T_D", "ions.D.T_eV", "T_D"))
                truth = base[key].(t_ns)
                err = abs.(r.values[vname] .- truth) ./ abs.(truth)
                say(@sprintf("  %-8s median relative error %.4f, worst %.4f at bin %d", label, median(err), maximum(err), argmax(err)))
            end
            truth_s = base[spec.shape_key].(t_ns)
            fitted_s = r.values[spec.shape_variable]
            say("  bin   t (ns)   $(rpad(spec.shape_variable, 8)) truth   fitted   fitted - truth")
            for j in eachindex(t_ns)
                say(@sprintf("  %3d  %7.4f  %14.4f  %7.4f  %+8.4f", j, t_ns[j], truth_s[j], fitted_s[j], fitted_s[j] - truth_s[j]))
            end
            jp = argmax(truth_s)
            say(@sprintf("  at the truth's peak (bin %d): truth %.4f, fitted %.4f; the fitted maximum %.4f at bin %d",
                jp, truth_s[jp], fitted_s[jp], maximum(fitted_s), argmax(fitted_s)))
            jd = argmax(abs.(fitted_s .- truth_s))
            say(@sprintf("  the shape parameter's largest departure: bin %d, fitted %.4f against truth %.4f",
                jd, fitted_s[jd], truth_s[jd]))
            push!(rows, (; name=spec.name, sigma_px=sp, epw=chi2n["epw"], iaw=chi2n["iaw"],
                status=r.status, converged=r.converged, gd, nb=length(entries),
                peak_truth=truth_s[jp], peak_fit=fitted_s[jp], med=median(abs.(fitted_s .- truth_s) ./ abs.(truth_s)),
                worst_epw=worst_bins["EPW"], departure_bin=jd))
        end
    end
    say("=" ^ 100)
    say("summary: shape, sigma_px, EPW chi2/n, IAW chi2/n, status, converged, gn_decrease, near-bound entries, shape parameter at the truth's peak (truth, fitted), its median relative error, EPW's worst chi2 bin, the shape parameter's largest-departure bin")
    for r in rows
        say(@sprintf("  %-18s %.0f  EPW %8.4f  IAW %8.4f  %-12s %-5s %.3e  %d  %.4f %.4f  %.4f  bin %d  bin %d",
            r.name, r.sigma_px, r.epw, r.iaw, r.status, r.converged, r.gd, r.nb, r.peak_truth, r.peak_fit, r.med,
            r.worst_epw, r.departure_bin))
    end
    epw(name, sp) = only(r.epw for r in rows if r.name === name && r.sigma_px == sp)
    say(@sprintf("  EPW chi2/n, GeneralizedNormal over Kappa: sigma_px = 1: %.3f; sigma_px = 0: %.3f",
        epw(:generalized_normal, 1.0) / epw(:kappa, 1.0), epw(:generalized_normal, 0.0) / epw(:kappa, 0.0)))
    say(@sprintf("total wall time %.1f s", time() - t_start))
    return nothing
end

main()
