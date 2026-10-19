# Phase 4.B Step 4.B.5: each new shape fitted from a deck on the two-channel synthetic problem,
# with the uncertainty reports beside it. Spec D39.
# Run: julia --project=. examples/fit_shapes.jl
#
# examples/shapes.jl is not included here (unlike Phase 4.A's uncertainty_two_channel.jl, which
# re-synthesizes its one problem's two streaks every run): synthesizing all three shapes' six
# streaks is its own step (D39 build order item 2) with its own budget, so this script assumes
# runs/shapes_<name>_{epw,iaw}_streak.jld2 already exist -- run examples/shapes.jl once first.
#
# The wavelength calibration is added in code, through with_wavelength_calibration, into a copy
# of each deck written under runs/shapes/<name>/uncertainty/ (not committed, runs/ is gitignored);
# examples/shapes_<name>.toml itself is not edited.
#
# The staged fit is the deck's own since Phase 4.F (spec D8): the copy is also given procedure =
# "staged" in code (with_staged_procedure: each stage with the deck's own Levenberg-Marquardt, T_D
# from the IAW stage), and TRUST._staged_fit runs it -- the function fit(deck) runs, called here
# because the uncertainty reports below need its problem, which fit(deck) does not return.

using TRUST
using Statistics: median

include(joinpath(@__DIR__, "problems.jl"))

const _FS_EXAMPLES_DIR = @__DIR__
const _FS_RUNS_DIR = joinpath(dirname(_FS_EXAMPLES_DIR), "runs", "shapes")

# name, the deck's shape variable (spec D39's table), the physics leaf it truths against, and
# whether its recovery error is reported relative (kappa, m) or absolute (alpha crosses zero).
const _SHAPES = (
    (name=:kappa, shape_variable="kappa_e", shape_key="electrons.e.f.kappa", error_kind=:relative),
    (name=:generalized_normal, shape_variable="m_e", shape_key="electrons.e.f.m", error_kind=:relative),
    (name=:skew_normal, shape_variable="alpha_e", shape_key="electrons.e.f.alpha", error_kind=:absolute),
)

function _fit_one_shape(spec)
    nm = spec.name
    pin_blas_threads()

    outdir = joinpath(_FS_RUNS_DIR, String(nm), "uncertainty")
    mkpath(outdir)
    deck_source = joinpath(_FS_EXAMPLES_DIR, "shapes_$(nm).toml")
    text = with_wavelength_calibration(read(deck_source, String))
    joint_outdir = replace(joinpath(outdir, "joint"), "\\" => "/")
    text = replace(text, "dir = \"runs/shapes/$(nm)/joint\"" => "dir = \"$joint_outdir\"")
    text = with_staged_procedure(text; stage_procedure="levenberg_marquardt",
        start_from=Dict{String,Any}("T_D" => "iaw"))
    deckpath = joinpath(outdir, "deck.toml")
    write(deckpath, text)

    t_staged = @elapsed st = TRUST._staged_fit(read_deck(deckpath))
    @info "fit_shapes[$nm]: staged fit done" elapsed_s = t_staged decrease = st.decreases[end] status = st.result.status
    flush(stderr)
    for c in st.result.channels
        @info "fit_shapes[$nm]: staged fit per-channel" name = c.name chi2_per_n = c.chi2 / c.npixels
    end
    flush(stderr)

    problem = shapes_problem(nm)
    base = Dict(problem.leaves)
    t_ns = t_from_si.(st.result.pspec.axis, :t_ns)

    variable_reports = (
        (label="n_e", truth_key="n_e_cm3", value_key="ne_cm3", bound_key="logn", kind=:relative),
        (label="T_e", truth_key="electrons.e.T_eV", value_key="T_e", bound_key="T_e", kind=:relative),
        (label=spec.shape_variable, truth_key=spec.shape_key, value_key=spec.shape_variable,
            bound_key=spec.shape_variable, kind=spec.error_kind),
        (label="T_D", truth_key="ions.D.T_eV", value_key="T_D", bound_key="T_D", kind=:relative),
    )
    for vr in variable_reports
        truth = base[vr.truth_key].(t_ns)
        fitted = st.result.values[vr.value_key]
        err = vr.kind === :relative ? abs.(fitted .- truth) ./ abs.(truth) : abs.(fitted .- truth)
        nb = count(st.result.near_bound[vr.bound_key])
        @info "fit_shapes[$nm]: variable" variable = vr.label error_kind = vr.kind median_error = median(err) near_bound_count = nb
    end
    flush(stderr)

    t_laplace = @elapsed lap = laplace(st.problem, st.result.theta)
    @info "fit_shapes[$nm]: laplace done" elapsed_s = t_laplace decrease = lap.decrease
    flush(stderr)
    for name in sort(collect(keys(lap.sigma_samp)))
        med_samp = median(lap.sigma_samp[name])
        n_nan = count(isnan, lap.sigma_samp[name])
        @info "fit_shapes[$nm]: laplace per-variable" variable = name median_sigma_samp = med_samp nan_count = n_nan
    end
    flush(stderr)

    t_scan_epw = @elapsed scan_epw = calibration_scan(st.problem, st.result.theta,
        [-0.2, -0.1, 0.0, 0.1, 0.2]; channel=1)
    t_scan_iaw = @elapsed scan_iaw = calibration_scan(st.problem, st.result.theta,
        [-0.02, -0.01, 0.0, 0.01, 0.02]; channel=2)
    for (label, scan, t_scan) in (("EPW", scan_epw, t_scan_epw), ("IAW", scan_iaw, t_scan_iaw))
        @info "fit_shapes[$nm]: $label calibration scan" elapsed_s = t_scan chi2 = scan.chi2
        for name in sort(collect(keys(scan.delta)))
            med_lo = median(abs.(scan.delta[name][:, 1]))
            med_hi = median(abs.(scan.delta[name][:, end]))
            @info "fit_shapes[$nm]: $label scan per-variable" variable = name median_abs_delta_at_lo_shift = med_lo median_abs_delta_at_hi_shift = med_hi
        end
    end
    flush(stderr)

    t_scatter = @elapsed sc = slice_scatter(st.deck, st.result)
    status_counts = Dict{Symbol,Int}()
    for s in sc.status
        status_counts[s] = get(status_counts, s, 0) + 1
    end
    @info "fit_shapes[$nm]: slice_scatter done" elapsed_s = t_scatter windows = length(sc.t_fine) status_counts
    flush(stderr)
    for name in sort(collect(keys(sc.s_stat)))
        med_ratio = median(sc.s_stat[name] ./ (sqrt(sc.m[1]) .* lap.sigma_samp[name]))
        @info "fit_shapes[$nm]: scatter per-variable" variable = name median_s_stat_over_sqrtm_sigma_samp = med_ratio
    end
    flush(stderr)

    eff = effective_deck(st.deck, st.problem, LevenbergMarquardt())
    result_path = joinpath(outdir, "result.jld2")
    write_result(result_path, st.result; problem=st.problem, laplace=lap, calibration_scan=scan_epw,
        scatter=sc, deck=eff, source="fit_shapes_$(nm)")
    r = read_result(result_path)
    @info "fit_shapes[$nm]: result file re-read" result_path has_scatter = (r.scatter !== nothing) has_laplace = (r.laplace !== nothing)
    flush(stderr)

    return (; name=nm, st, lap, scan_epw, scan_iaw, sc, result_path)
end

function run_fit_shapes()
    results = NamedTuple[]
    for spec in _SHAPES
        t_total = @elapsed r = _fit_one_shape(spec)
        @info "fit_shapes[$(spec.name)]: shape total" elapsed_s = t_total
        flush(stderr)
        push!(results, r)
    end
    return results
end

run_fit_shapes()
