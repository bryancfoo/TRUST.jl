# The two-channel joint fit's committed figure (spec D41 amendment 5, conventions.md 10.4's
# committed-figure exception). Reads the two result files fit_two_channel.jl writes
# (runs/two_channel/joint/result.jld2, the joint fit, and runs/two_channel/staged/result.jld2,
# whose stages are the EPW-only and IAW-only fits, Phase 4.F spec D8) and draws the fitted n_e,
# T_e, m_e and T_D against the truth for the joint fit and each single-channel fit -- the point
# being what each channel alone can and cannot recover (D41). Not part of
# make_figures.jl/FIGURE_NAMES: FG1 asserts exactly five files, and FG4 runs make_figures.jl twice
# with no fit results on disk.
#
# Run by hand, after examples/fit_two_channel.jl has written the two result files:
#   julia --project=examples/figures -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=examples/figures examples/figures/make_two_channel_figure.jl [outdir]
#
# `outdir` defaults to this directory (the committed location).

using CairoMakie
using TRUST

include(joinpath(@__DIR__, "..", "problems.jl"))

Makie.set_theme!(fontsize=14)

const _RESULTS_DIR = joinpath(dirname(dirname(@__DIR__)), "runs", "two_channel")

function _save_info(path, fig)
    t0 = time()
    save(path, fig; px_per_unit=1)
    dt = time() - t0
    @info "figure" name = basename(path) bytes = filesize(path) seconds = dt
    return path
end

# One panel: the truth curve (fine grid, the analytic profile), then the joint, EPW-only and
# IAW-only fits at their own fitted-axis bin centres. `getvalue` picks the variable's or
# dependent's own `values` vector out of a `read_result` NamedTuple or one of its stage records,
# which carry `axis`, `variables` and `dependents` in the same form.
function _panel!(ax, truth_fn, results, getvalue)
    t_truth = range(2.2, 5.6; length=200)
    lines!(ax, collect(t_truth), truth_fn.(t_truth); color=:black, linewidth=2, label="truth")
    colors = (:steelblue, :firebrick, :seagreen)
    for ((name, r), color) in zip(results, colors)
        t_ns = t_from_si.(r.axis, :t_ns)
        scatterlines!(ax, t_ns, getvalue(r); color=color, markersize=5, label=name)
    end
    return ax
end

function main(outdir=@__DIR__)
    mkpath(outdir)
    problem = two_channel_problem()
    base = Dict(problem.leaves)

    joint = read_result(joinpath(_RESULTS_DIR, "joint", "result.jld2"))
    stages = read_result(joinpath(_RESULTS_DIR, "staged", "result.jld2")).stages
    epw = stages.channels["epw"]
    iaw = stages.channels["iaw"]
    results = (("joint", joint), ("EPW-only", epw), ("IAW-only", iaw))

    fig = Figure(size=(1200, 900))
    Label(fig[0, 1:2], "Two-channel joint fit (Phase 4.A Step 4.A.3c, spec D41)"; fontsize=18,
        tellwidth=false)

    ax_ne = Axis(fig[1, 1]; xlabel="t (ns)", ylabel="n_e (cm⁻³)")
    _panel!(ax_ne, base["n_e_cm3"], results, r -> r.dependents["ne_cm3"].values)
    axislegend(ax_ne; position=:rb)

    ax_Te = Axis(fig[1, 2]; xlabel="t (ns)", ylabel="T_e (eV)")
    _panel!(ax_Te, base["electrons.e.T_eV"], results, r -> r.variables["T_e"].values)

    ax_m = Axis(fig[2, 1]; xlabel="t (ns)", ylabel="m_e")
    _panel!(ax_m, base["electrons.e.f.m"], results, r -> r.variables["m_e"].values)

    ax_TD = Axis(fig[2, 2]; xlabel="t (ns)", ylabel="T_D (eV)")
    _panel!(ax_TD, base["ions.D.T_eV"], results, r -> r.variables["T_D"].values)

    return _save_info(joinpath(outdir, "two_channel_fit.png"), fig)
end

main(length(ARGS) >= 1 ? ARGS[1] : @__DIR__)
