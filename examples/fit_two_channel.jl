# Phase 4.A Step 4.A.3c: the synthetic two-channel problem, full resolution. Spec D41, Section 9.3.
# Run: julia --project=. examples/fit_two_channel.jl
#
# Runs the joint deck (examples/two_channel.toml) and the two single-channel decks derived from it
# in code (never committed, OTIS._with... no -- single_channel_deck_text, examples/problems.jl),
# each keeping every variable free (D41: the point of the figure is what each channel alone can
# and cannot recover). Writes runs/two_channel/{joint,epw,iaw}/result.jld2 -- fit's own output,
# per deck. The figure itself is drawn separately, in the figures environment
# (examples/figures/make_two_channel_figure.jl), which reads these three result files back.

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

function _medians(result::FitResult, problem_nt)
    t_ns = t_from_si.(result.pspec.axis, :t_ns)
    truth = _truth_at(problem_nt, t_ns)
    return Dict{String,Float64}(v => median(abs.(result.values[v] .- truth[v]) ./ truth[v])
                                 for v in keys(truth))
end

function _report(label::AbstractString, result::FitResult, problem_nt)
    med = _medians(result, problem_nt)
    for v in sort(collect(keys(med)))
        @info "fit_two_channel: $label per-variable median" variable = v median_rel_error = med[v]
    end
    for c in result.channels
        @info "fit_two_channel: $label per-channel" name = c.name chi2 = c.chi2 npixels = c.npixels nparams = c.nparams reduced_chi2 = c.reduced_chi2 reduced_chi2_simple = c.reduced_chi2_simple chi2_per_n = c.chi2 / c.npixels
    end
    @info "fit_two_channel: $label" status = result.status converged = result.converged iterations = result.iterations elapsed_s = result.elapsed near_bound_fraction = _near_bound_fraction(result)
    return med
end

function run_two_channel()
    problem_nt = two_channel_problem()
    text = read(joinpath(_FTC_EXAMPLES_DIR, "two_channel.toml"), String)

    t_joint = @elapsed joint = fit(joinpath(_FTC_EXAMPLES_DIR, "two_channel.toml"))
    med_joint = _report("joint", joint, problem_nt)

    epw_dir = joinpath(dirname(_FTC_EXAMPLES_DIR), "runs", "two_channel", "epw")
    mkpath(epw_dir)
    epw_deck_path = joinpath(epw_dir, "deck.toml")
    write(epw_deck_path, single_channel_deck_text(text, "epw"))
    t_epw = @elapsed epw = fit(epw_deck_path)
    med_epw = _report("epw-only", epw, problem_nt)

    iaw_dir = joinpath(dirname(_FTC_EXAMPLES_DIR), "runs", "two_channel", "iaw")
    mkpath(iaw_dir)
    iaw_deck_path = joinpath(iaw_dir, "deck.toml")
    write(iaw_deck_path, single_channel_deck_text(text, "iaw"))
    t_iaw = @elapsed iaw = fit(iaw_deck_path)
    med_iaw = _report("iaw-only", iaw, problem_nt)

    @info "fit_two_channel: end to end" t_joint_s = t_joint t_epw_s = t_epw t_iaw_s = t_iaw t_sum_single_s = t_epw + t_iaw

    return (; joint, epw, iaw, med_joint, med_epw, med_iaw)
end

run_two_channel()
