using Test
using OTIS
using OTIS: _near_bound_fraction
using Statistics: median

# TS1 against dev/phase4A_instructions.md Section 9.3 (spec D41, chunk 8), gated "two-channel
# problem (slow)": the joint, EPW-only and IAW-only fits at full resolution, TC3's substitution
# (build_problem + solve on the joint deck and on each single-channel deck derived from it).

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const _TS_EXAMPLES_DIR = joinpath(@__DIR__, "..", "..", "examples")

"""
The truth of the four fitted quantities, evaluated directly from the problem's own analytic
profile functions at `t_ns` -- exact, no interpolation, since the profiles are closed-form.
"""
function _ts_truth_at(problem, t_ns::AbstractVector)
    base = Dict(problem.leaves)
    return Dict{String,Vector{Float64}}(
        "ne_cm3" => base["n_e_cm3"].(t_ns),
        "T_e" => base["electrons.e.T_eV"].(t_ns),
        "m_e" => base["electrons.e.f.m"].(t_ns),
        "T_D" => base["ions.D.T_eV"].(t_ns),
    )
end

function _ts_medians(result::FitResult, problem_nt)
    t_ns = t_from_si.(result.pspec.axis, :t_ns)
    truth = _ts_truth_at(problem_nt, t_ns)
    return Dict{String,Float64}(v => median(abs.(result.values[v] .- truth[v]) ./ truth[v])
                                 for v in keys(truth))
end

function _ts_worst_bin(result::FitResult, problem_nt, variable::AbstractString)
    t_ns = t_from_si.(result.pspec.axis, :t_ns)
    truth = _ts_truth_at(problem_nt, t_ns)[variable]
    fitv = result.values[variable]
    rel = abs.(fitv .- truth) ./ truth
    i = argmax(rel)
    return (; t_ns=t_ns[i], fit=fitv[i], truth=truth[i], rel=rel[i])
end

function _ts_report(label::AbstractString, result::FitResult, problem_nt)
    med = _ts_medians(result, problem_nt)
    @info "TS1: $label medians" ne_cm3 = med["ne_cm3"] T_e = med["T_e"] m_e = med["m_e"] T_D = med["T_D"]
    for c in result.channels
        @info "TS1: $label per-channel" name = c.name chi2_per_n = c.chi2 / c.npixels reduced_chi2 = c.reduced_chi2 npixels = c.npixels nparams = c.nparams
    end
    @info "TS1: $label" status = result.status converged = result.converged iterations = result.iterations elapsed_s = result.elapsed near_bound_fraction = _near_bound_fraction(result)
    return med
end

function _ts_two_channel()
    @info "TS1: synthesizing the full-resolution two-channel problem"
    flush(stderr)
    problem_nt = two_channel_problem()
    sd_epw = synthesize_channel(problem_nt, :epw)
    sd_iaw = synthesize_channel(problem_nt, :iaw)
    @info "TS1: synthesis done" epw_size = size(sd_epw.image) iaw_size = size(sd_iaw.image)
    flush(stderr)

    return mktempdir() do dir
        path_epw = joinpath(dir, "two_channel_epw_streak.jld2")
        path_iaw = joinpath(dir, "two_channel_iaw_streak.jld2")
        write_streak(path_epw, sd_epw; source="two_channel", probe=problem_nt.probe, collection=problem_nt.collection)
        write_streak(path_iaw, sd_iaw; source="two_channel", probe=problem_nt.probe, collection=problem_nt.collection)

        text = read(joinpath(_TS_EXAMPLES_DIR, "two_channel.toml"), String)
        text = replace(text, "runs/two_channel_epw_streak.jld2" => replace(path_epw, "\\" => "/"))
        text = replace(text, "runs/two_channel_iaw_streak.jld2" => replace(path_iaw, "\\" => "/"))

        joint_dir = joinpath(dir, "joint")
        text_joint = replace(text, "runs/two_channel/joint" => replace(joint_dir, "\\" => "/"))
        joint_deck_path = joinpath(dir, "joint_deck.toml")
        write(joint_deck_path, text_joint)
        joint_problem, joint_opt, joint_start = OTIS.build_problem(read_deck(joint_deck_path))
        joint = solve(joint_problem, joint_opt; start=joint_start)
        @info "TS1: joint fit done" elapsed_s = joint.elapsed status = joint.status
        flush(stderr)
        med_joint = _ts_report("joint", joint, problem_nt)

        epw_dir = joinpath(dir, "epw")
        text_epw = single_channel_deck_text(text, "epw")
        text_epw = replace(text_epw, "runs/two_channel/epw" => replace(epw_dir, "\\" => "/"))
        epw_deck_path = joinpath(dir, "epw_deck.toml")
        write(epw_deck_path, text_epw)
        epw_problem, epw_opt, epw_start = OTIS.build_problem(read_deck(epw_deck_path))
        epw = solve(epw_problem, epw_opt; start=epw_start)
        @info "TS1: EPW-only fit done" elapsed_s = epw.elapsed status = epw.status
        flush(stderr)
        med_epw = _ts_report("EPW-only", epw, problem_nt)
        epw_td_move = maximum(abs.(epw.values["T_D"] .- 300.0))
        @info "TS1: EPW-only T_D" median = med_epw["T_D"] largest_move_from_300 = epw_td_move

        iaw_dir = joinpath(dir, "iaw")
        text_iaw = single_channel_deck_text(text, "iaw")
        text_iaw = replace(text_iaw, "runs/two_channel/iaw" => replace(iaw_dir, "\\" => "/"))
        iaw_deck_path = joinpath(dir, "iaw_deck.toml")
        write(iaw_deck_path, text_iaw)
        iaw_problem, iaw_opt, iaw_start = OTIS.build_problem(read_deck(iaw_deck_path))
        iaw = solve(iaw_problem, iaw_opt; start=iaw_start)
        @info "TS1: IAW-only fit done" elapsed_s = iaw.elapsed status = iaw.status
        flush(stderr)
        med_iaw = _ts_report("IAW-only", iaw, problem_nt)

        for v in ("ne_cm3", "T_e", "m_e", "T_D")
            wb = _ts_worst_bin(joint, problem_nt, v)
            @info "TS1: joint fit worst bin" variable = v t_ns = wb.t_ns fit = wb.fit truth = wb.truth rel_error = wb.rel
        end

        # TS2 (D53; through fit(deck) since Phase 4.F, spec D9): examples/two_channel_staged.toml,
        # procedure = "staged", on the same two streaks -- the EPW-only and IAW-only stages, then
        # the joint LM + polish started from their values. fit(deck) pins BLAS to one thread
        # (conventions.md 6.6), so the count is restored after it for the slow files that follow.
        staged_dir = joinpath(dir, "staged")
        text_staged = read(joinpath(_TS_EXAMPLES_DIR, "two_channel_staged.toml"), String)
        text_staged = replace(text_staged, "runs/two_channel_epw_streak.jld2" => replace(path_epw, "\\" => "/"))
        text_staged = replace(text_staged, "runs/two_channel_iaw_streak.jld2" => replace(path_iaw, "\\" => "/"))
        text_staged = replace(text_staged, "runs/two_channel/staged" => replace(staged_dir, "\\" => "/"))
        staged_deck_path = joinpath(dir, "staged_deck.toml")
        write(staged_deck_path, text_staged)
        nblas = pin_blas_threads()
        pin_blas_threads(nblas)
        t_staged = @elapsed staged = fit(staged_deck_path)
        pin_blas_threads(nblas)
        st = read_result(joinpath(staged_dir, "result.jld2"))
        @info "TS2: fit(deck), procedure = \"staged\", done" elapsed_s = t_staged decrease = st.stages.polish_decreases[end] blas_threads_restored = nblas
        flush(stderr)
        med_st = _ts_report("staged joint", staged, problem_nt)

        (; joint, epw, iaw, med_joint, med_epw, med_iaw, epw_td_move, staged, st, med_st)
    end
end

const _TS_TWO_CHANNEL = _ts_two_channel()

@testset "TS1: the joint, EPW-only and IAW-only fits at full size" begin
    r = _TS_TWO_CHANNEL
    @test r.joint.converged == true
    @test r.med_joint["ne_cm3"] <= 0.2
    @test r.med_joint["T_e"] <= 0.2
    @test r.med_joint["T_D"] <= 0.2
end

@testset "TS2: the staged fit through fit(deck) (D53; Phase 4.F spec D9)" begin
    staged = _TS_TWO_CHANNEL.staged
    st = _TS_TWO_CHANNEL.st
    @test st.stages.polish_decreases[end] <= 1e-6

    epw_summary = staged.channels[findfirst(c -> c.name == "epw", staged.channels)]
    iaw_summary = staged.channels[findfirst(c -> c.name == "iaw", staged.channels)]
    chi2n_epw_joint = epw_summary.chi2 / epw_summary.npixels
    chi2n_iaw_joint = iaw_summary.chi2 / iaw_summary.npixels
    chi2n_epw_alone = st.stages.channels["epw"].chi2 / st.stages.channels["epw"].npixels
    chi2n_iaw_alone = st.stages.channels["iaw"].chi2 / st.stages.channels["iaw"].npixels
    @test chi2n_epw_joint <= 1.1 * chi2n_epw_alone
    @test chi2n_iaw_joint <= 1.1 * chi2n_iaw_alone

    # The saturation warning (Phase 4.D spec D5) is printed by fit(deck) itself now, beside the
    # recorded failure below (bin 1's m_e on its lower bound 2.0, Phase 4.B's A14). The assertion
    # is unchanged.
    @test sum(count, values(staged.near_bound)) == 0

    @info "TS2 chi2/n" chi2n_epw_joint chi2n_epw_alone chi2n_iaw_joint chi2n_iaw_alone
end
