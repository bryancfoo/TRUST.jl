# Phase 4.A Step 4.A.4b: the two-channel example with all three uncertainty reports, at full
# resolution. Spec D55, dev/phase4A_instructions.md Section 10.2.
# Run: julia --project=. examples/uncertainty_two_channel.jl
#
# examples/two_channel.toml is not edited (TC3 and TS1 read it): the wavelength calibration is
# added in code, through with_wavelength_calibration, into a copy of the deck written under
# runs/two_channel/uncertainty/ (which is not committed, runs/ is gitignored).

using TRUST
using Statistics: median

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "two_channel.jl")) # writes the two streaks, its own main()

const _UTC_EXAMPLES_DIR = @__DIR__
const _UTC_OUTDIR = joinpath(dirname(_UTC_EXAMPLES_DIR), "runs", "two_channel", "uncertainty")

function run_uncertainty_two_channel()
    pin_blas_threads()
    mkpath(_UTC_OUTDIR)

    text = with_wavelength_calibration(read(joinpath(_UTC_EXAMPLES_DIR, "two_channel.toml"), String))
    joint_outdir = replace(joinpath(_UTC_OUTDIR, "joint"), "\\" => "/")
    text = replace(text, "dir = \"runs/two_channel/joint\"" => "dir = \"$joint_outdir\"")
    deckpath = joinpath(_UTC_OUTDIR, "deck.toml")
    write(deckpath, text)

    t_staged = @elapsed st = staged_joint_fit(deckpath)
    @info "uncertainty_two_channel: staged fit done" elapsed_s = t_staged decrease = st.decreases[end] status = st.result.status
    for c in st.result.channels
        @info "uncertainty_two_channel: staged fit per-channel" name = c.name chi2_per_n = c.chi2 / c.npixels
    end

    t_laplace = @elapsed lap = laplace(st.problem, st.result.theta)
    @info "uncertainty_two_channel: laplace done" elapsed_s = t_laplace decrease = lap.decrease
    for name in sort(collect(keys(lap.sigma_samp)))
        med_samp = median(lap.sigma_samp[name])
        med_ratio = median(lap.sigma_post[name] ./ lap.sigma_samp[name])
        @info "uncertainty_two_channel: laplace per-variable" variable = name median_sigma_samp = med_samp median_sigma_post_over_samp = med_ratio
    end

    t_scan_epw = @elapsed scan_epw = calibration_scan(st.problem, st.result.theta,
        [-0.2, -0.1, 0.0, 0.1, 0.2]; channel=1)
    t_scan_iaw = @elapsed scan_iaw = calibration_scan(st.problem, st.result.theta,
        [-0.02, -0.01, 0.0, 0.01, 0.02]; channel=2)
    for (label, scan, t_scan) in (("EPW", scan_epw, t_scan_epw), ("IAW", scan_iaw, t_scan_iaw))
        per_point = t_scan / length(scan.shifts_nm)
        @info "uncertainty_two_channel: $label calibration scan" elapsed_s = t_scan per_point_s = per_point chi2 = scan.chi2
        for name in sort(collect(keys(scan.delta)))
            med_delta = [median(scan.delta[name][:, k]) for k in eachindex(scan.shifts_nm)]
            @info "uncertainty_two_channel: $label scan per-variable" variable = name median_delta = med_delta median_s_sys = median(scan.s_sys[name])
        end
    end

    t_scatter = @elapsed sc = slice_scatter(st.deck, st.result)
    @info "uncertainty_two_channel: slice_scatter done" elapsed_s = t_scatter windows = length(sc.t_fine) per_window_s = t_scatter / length(sc.t_fine) m = sc.m
    status_counts = Dict{Symbol,Int}()
    for s in sc.status
        status_counts[s] = get(status_counts, s, 0) + 1
    end
    @info "uncertainty_two_channel: scatter statuses" status_counts
    for name in sort(collect(keys(sc.s_stat)))
        med_s_stat = median(sc.s_stat[name])
        # Against the regularized sigma_samp (Section 10.2's own formula for this report), so
        # this reads higher than Level 2's 0.46-0.50, which was measured against the
        # unregularized sigma -- narrower sigma_samp, larger ratio, expected and not a finding.
        med_ratio = median(sc.s_stat[name] ./ (sqrt(sc.m[1]) .* lap.sigma_samp[name]))
        nb = sum(sc.n_near_bound[name])
        @info "uncertainty_two_channel: scatter per-variable" variable = name median_s_stat = med_s_stat median_s_stat_over_sqrtm_sigma_samp = med_ratio near_bound_windows = nb
    end

    eff = effective_deck(st.deck, st.problem, LevenbergMarquardt())
    result_path = joinpath(_UTC_OUTDIR, "result.jld2")
    write_result(result_path, st.result; problem=st.problem, laplace=lap, calibration_scan=scan_epw,
        scatter=sc, deck=eff, source="uncertainty_two_channel")
    iaw_scan_path = joinpath(_UTC_OUTDIR, "iaw_scan.jld2")
    write_result(iaw_scan_path, st.result; problem=st.problem, laplace=lap, calibration_scan=scan_iaw,
        scatter=sc, deck=eff, source="uncertainty_two_channel")

    r1 = read_result(result_path)
    r2 = read_result(iaw_scan_path)
    @info "uncertainty_two_channel: both result files re-read" result_path result1_has_scatter = r1.scatter !== nothing result1_has_laplace = r1.laplace !== nothing result1_scan_channel = r1.calibration_scan.channel iaw_scan_path result2_scan_channel = r2.calibration_scan.channel

    return (; st, lap, scan_epw, scan_iaw, sc, result_path, iaw_scan_path)
end

run_uncertainty_two_channel()
