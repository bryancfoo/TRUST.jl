# Phase 4.A Step 4.A.3c: the synthetic two-channel problem, full resolution. Spec D41, Section 9.3;
# moved to the deck at Phase 4.F Step 4.F.1 (spec D8).
# Run: julia --project=. examples/fit_two_channel.jl
#
# Two fits, each through fit(deck), each writing its own result file and effective deck:
# examples/two_channel.toml, the joint fit from the deck's own inits (runs/two_channel/joint/), and
# examples/two_channel_staged.toml, procedure = "staged" (runs/two_channel/staged/), whose two
# stages are the EPW-only and IAW-only fits, every variable free (D41: the point of the figure is
# what each channel alone can and cannot recover), and whose final fit is the staged joint fit.
# The figure itself is drawn separately, in the figures environment
# (examples/figures/make_two_channel_figure.jl), which reads the joint fit and the two stages back.

using OTIS
using OTIS: _near_bound_fraction
using Statistics: median

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "two_channel.jl")) # writes runs/two_channel_{epw,iaw}_streak.jld2 (its own main())

const _FTC_EXAMPLES_DIR = @__DIR__

"""
The truth of the four fitted quantities, evaluated directly from the problem's own analytic
profile functions at `t_ns` -- exact, no interpolation, since the profiles are closed-form.
"""
function _truth_at(problem, t_ns::AbstractVector)
    base = Dict(problem.leaves)
    return Dict{String,Vector{Float64}}(
        "ne_cm3" => base["n_e_cm3"].(t_ns),
        "T_e" => base["electrons.e.T_eV"].(t_ns),
        "m_e" => base["electrons.e.f.m"].(t_ns),
        "T_D" => base["ions.D.T_eV"].(t_ns),
    )
end

# The median relative error of each quantity, from its values on the axis `axis_s` (s).
function _medians(values::AbstractDict, axis_s::AbstractVector, problem_nt)
    truth = _truth_at(problem_nt, t_from_si.(axis_s, :t_ns))
    return Dict{String,Float64}(v => median(abs.(values[v] .- truth[v]) ./ truth[v]) for v in keys(truth))
end

function _report(label::AbstractString, result::FitResult, problem_nt)
    med = _medians(result.values, result.pspec.axis, problem_nt)
    for v in sort(collect(keys(med)))
        @info "fit_two_channel: $label per-variable median" variable = v median_rel_error = med[v]
    end
    for c in result.channels
        @info "fit_two_channel: $label per-channel" name = c.name chi2 = c.chi2 npixels = c.npixels nparams = c.nparams reduced_chi2 = c.reduced_chi2 reduced_chi2_simple = c.reduced_chi2_simple chi2_per_n = c.chi2 / c.npixels
    end
    @info "fit_two_channel: $label" status = result.status converged = result.converged iterations = result.iterations elapsed_s = result.elapsed near_bound_fraction = _near_bound_fraction(result)
    return med
end

# A stage of the staged fit, as read_result returns it (stages.channels[name]).
function _report_stage(label::AbstractString, s, problem_nt)
    values = Dict{String,Vector{Float64}}(v => s.variables[v].values for v in ("T_e", "m_e", "T_D"))
    values["ne_cm3"] = s.dependents["ne_cm3"].values
    med = _medians(values, s.axis, problem_nt)
    for v in sort(collect(keys(med)))
        @info "fit_two_channel: $label per-variable median" variable = v median_rel_error = med[v]
    end
    @info "fit_two_channel: $label" status = s.status converged = s.converged iterations = s.iterations elapsed_s = s.elapsed chi2_per_n = s.chi2 / s.npixels near_bound_fraction = s.near_bound_fraction
    return med
end

function run_two_channel()
    problem_nt = two_channel_problem()

    t_joint = @elapsed joint = fit(joinpath(_FTC_EXAMPLES_DIR, "two_channel.toml"))
    med_joint = _report("joint", joint, problem_nt)

    t_staged = @elapsed staged = fit(joinpath(_FTC_EXAMPLES_DIR, "two_channel_staged.toml"))
    med_staged = _report("staged joint", staged, problem_nt)
    r = read_result(joinpath(dirname(_FTC_EXAMPLES_DIR), "runs", "two_channel", "staged", "result.jld2"))
    med_epw = _report_stage("epw-only (stage 1)", r.stages.channels["epw"], problem_nt)
    med_iaw = _report_stage("iaw-only (stage 2)", r.stages.channels["iaw"], problem_nt)

    @info "fit_two_channel: end to end" t_joint_s = t_joint t_staged_s = t_staged

    return (; joint, staged, med_joint, med_staged, med_epw, med_iaw)
end

run_two_channel()
