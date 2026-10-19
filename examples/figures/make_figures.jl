# The committed figure gallery's generator (spec D36-D38). Regenerates the five PNGs of
# examples/figures/ from examples/problems.jl's fixed seeds; never hand-edit the PNGs.
#
# Run by hand:
#   julia --project=examples/figures -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=examples/figures examples/figures/make_figures.jl [outdir]
#
# `outdir` defaults to this directory (the committed location); FG1 points it at an
# `mktempdir()` instead so the test never touches the committed files.

using CairoMakie
using TRUST
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "problems.jl"))

Makie.set_theme!(fontsize=14)

const SHOCK_TS_NS = 1.15
const CHAIN_BIN = 128

_gaussian_pdf(x) = exp(-x^2 / 2) / sqrt(2π)

function _save_info(path, fig)
    t0 = time()
    save(path, fig; px_per_unit=1)
    dt = time() - t0
    @info "figure" name = basename(path) bytes = filesize(path) seconds = dt
    return path
end

function _measured_streak(expt)
    return StreakData(lambda_m=expt.lambda, t_s=expt.t, image=measure(expt))
end

function figure_epw_ablation(outdir, problem)
    expt = build_experiment(problem)
    sd = _measured_streak(expt)

    fig = Figure(size=(1200, 900))
    Label(fig[0, 1:4], "EPW ablation problem (foo2026 Section III.A)"; fontsize=18, tellwidth=false)

    ax_streak = Axis(fig[1, 1:3]; xlabel="t (ns)", ylabel="λ (nm)")
    hm = streakplot!(ax_streak, sd; colormap=:viridis)
    Colorbar(fig[1, 4], hm; label="counts")

    ax_ne = Axis(fig[2, 1]; xlabel="t (ns)", ylabel="n_e (cm⁻³)")
    traceplot!(ax_ne, problem.trace, "n_e")

    ax_Te = Axis(fig[2, 2]; xlabel="t (ns)", ylabel="T_e (eV)")
    traceplot!(ax_Te, problem.trace, "electrons.e.T")

    ax_m = Axis(fig[2, 3]; xlabel="t (ns)", ylabel="m")
    traceplot!(ax_m, problem.trace, "electrons.e.f.m")

    return _save_info(joinpath(outdir, "epw_ablation.png"), fig)
end

function figure_iaw_multiflow(outdir, problem)
    expt = build_experiment(problem)
    sd = _measured_streak(expt)

    fig = Figure(size=(1200, 900))
    Label(fig[0, 1:4], "IAW multi-flow problem (foo2026 Section III.B)"; fontsize=18, tellwidth=false)

    ax_streak = Axis(fig[1, 1:3]; xlabel="t (ns)", ylabel="λ (nm)")
    hm = streakplot!(ax_streak, sd; colormap=:viridis)
    Colorbar(fig[1, 4], hm; label="counts")

    ax_f = Axis(fig[2, 1]; xlabel="t (ns)", ylabel="fraction")
    traceplot!(ax_f, problem.trace, "ions.D.fraction"; label="f_D")
    traceplot!(ax_f, problem.trace, "ions.C.fraction"; label="f_C")
    axislegend(ax_f)

    ax_T = Axis(fig[2, 2]; xlabel="t (ns)", ylabel="T (eV)")
    traceplot!(ax_T, problem.trace, "ions.D.T"; label="T_D")
    traceplot!(ax_T, problem.trace, "ions.C.T"; label="T_C")
    axislegend(ax_T)

    ax_u = Axis(fig[2, 3]; xlabel="t (ns)", ylabel="u_C (km/s)")
    traceplot!(ax_u, problem.trace, "ions.C.u")

    return _save_info(joinpath(outdir, "iaw_multiflow.png"), fig)
end

function figure_iaw_shock(outdir, problem)
    expt = build_experiment(problem)
    sd = _measured_streak(expt)

    fig = Figure(size=(1200, 900))
    Label(fig[0, 1:3], "IAW shock problem (foo2026 Fig. 5)"; fontsize=18, tellwidth=false)

    ax_streak = Axis(fig[1, 1:2]; xlabel="t (ns)", ylabel="λ (nm)")
    hm = streakplot!(ax_streak, sd; colormap=:viridis)
    Colorbar(fig[1, 3], hm; label="counts")
    vlines!(ax_streak, [SHOCK_TS_NS]; linestyle=:dash, color=:red)

    ax_T = Axis(fig[2, 1]; xlabel="t (ns)", ylabel="T (eV)")
    traceplot!(ax_T, problem.trace, "ions.D.T"; label="T_D")
    traceplot!(ax_T, problem.trace, "ions.C.T"; label="T_C")
    axislegend(ax_T)
    vlines!(ax_T, [SHOCK_TS_NS]; linestyle=:dash, color=:red)

    ax_f = Axis(fig[2, 2]; xlabel="t (ns)", ylabel="f_D")
    traceplot!(ax_f, problem.trace, "ions.D.fraction")
    vlines!(ax_f, [SHOCK_TS_NS]; linestyle=:dash, color=:red)

    return _save_info(joinpath(outdir, "iaw_shock.png"), fig)
end

function figure_instrument_chain(outdir, problem; bin=CHAIN_BIN)
    trace = problem.trace
    p = plasma(trace, bin)
    t_bin = trace.axis[bin]
    single_trace = PlasmaTrace([p]; t_s=[t_bin])

    sens = only(filter(e -> e isa Sensitivity, problem.effects))
    scale = sens.scale

    _stage(effects) = vec(measure(Experiment(trace=single_trace, probe=problem.probe,
        collection=problem.collection, lambda_m=problem.lambda, effects=effects)))

    stage_ideal = _stage(()) .* scale
    stage_k = _stage((KSmearing(half_angle_deg=3.0),)) .* scale
    stage_spec = _stage((KSmearing(half_angle_deg=3.0), SpectralIRF(fwhm_nm=0.5))) .* scale
    stage_gain = _stage((KSmearing(half_angle_deg=3.0), SpectralIRF(fwhm_nm=0.5),
        Sensitivity(scale=scale), ConstantBackground(level=25.0)))

    lambda_nm = lambda_from_si.(problem.lambda, :lambda_nm)
    t_ns = t_from_si(t_bin, :t_ns)

    fig = Figure(size=(1200, 800))
    Label(fig[0, 1], "What the instrument chain does to one spectrum (t = $(round(t_ns; digits=2)) ns)";
        fontsize=18, tellwidth=false)

    ax1 = Axis(fig[1, 1]; ylabel="counts, scaled to overlay")
    spectrumplot!(ax1, lambda_nm, stage_ideal; label="ideal")
    spectrumplot!(ax1, lambda_nm, stage_k; label="+ KSmearing")
    spectrumplot!(ax1, lambda_nm, stage_spec; label="+ SpectralIRF")
    spectrumplot!(ax1, lambda_nm, stage_gain; label="+ Sensitivity, ConstantBackground")
    axislegend(ax1)

    ax2 = Axis(fig[2, 1]; xlabel="λ (nm)", ylabel="difference from ideal")
    spectrumplot!(ax2, lambda_nm, stage_k .- stage_ideal; label="+ KSmearing")
    spectrumplot!(ax2, lambda_nm, stage_spec .- stage_ideal; label="+ SpectralIRF")
    spectrumplot!(ax2, lambda_nm, stage_gain .- stage_ideal; label="+ Sensitivity, ConstantBackground")
    axislegend(ax2)

    linkxaxes!(ax1, ax2)

    return _save_info(joinpath(outdir, "instrument_chain.png"), fig)
end

function figure_noise(outdir, problem)
    expt = build_experiment(problem)
    Sbar = measure(expt)
    lambda_nm = lambda_from_si.(expt.lambda, :lambda_nm)
    j = 1

    model = problem.noise
    sd = synthesize(expt, model; rng=Xoshiro(problem.seed))
    sig = sigma(model, Sbar)

    fig = Figure(size=(1200, 800))
    Label(fig[0, 1:2], "Streak-camera noise (Swadling et al. 2022)"; fontsize=18, tellwidth=false)

    ax1 = Axis(fig[1, 1]; xlabel="λ (nm)", ylabel="counts")
    band!(ax1, lambda_nm, Sbar[:, j] .- sig[:, j], Sbar[:, j] .+ sig[:, j];
        color=(:steelblue, 0.25), label="±σ")
    scatter!(ax1, lambda_nm, sd.image[:, j]; markersize=4, color=(:steelblue, 0.6), label="noisy draw")
    spectrumplot!(ax1, lambda_nm, Sbar[:, j]; color=:black, label="S̄")
    axislegend(ax1)

    # E[image] is the blurred mean (the spread is linear); the model's own mean is not blurred
    # (OQ-40, Phase 4.B spec D12).
    z = vec((sd.image .- TRUST._point_spread(Sbar, model.sigma_px)) ./ sig)

    ax2 = Axis(fig[1, 2]; xlabel="(image - E[image]) / σ  [realized σ, sigma_px = $(model.sigma_px)]", ylabel="density")
    hist!(ax2, z; normalization=:pdf, bins=60, color=(:steelblue, 0.5), label="residuals")
    xs = range(-4, 4; length=200)
    lines!(ax2, xs, _gaussian_pdf.(xs); color=:black, label="unit normal")
    axislegend(ax2)

    return _save_info(joinpath(outdir, "noise.png"), fig)
end

function main()
    outdir = length(ARGS) >= 1 ? ARGS[1] : @__DIR__
    mkpath(outdir)

    epw = epw_ablation_problem()
    multiflow = iaw_multiflow_problem()
    shock = iaw_shock_problem()

    figure_epw_ablation(outdir, epw)
    figure_iaw_multiflow(outdir, multiflow)
    figure_iaw_shock(outdir, shock)
    figure_instrument_chain(outdir, epw)
    figure_noise(outdir, epw)

    return nothing
end

main()
