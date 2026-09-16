using Test
using OTIS
using Random: Xoshiro

# RP1, RP2 against dev/phase3_instructions.md Section 10 (spec D38, D46), gated "reproduce".
# This file is the fast suite's own structural smoke test: it is not the physics. The three
# foo2026 reproductions themselves -- the full-resolution fits and their figures, and the
# 128 x 32 cut-down scans with the qualitative acceptance of D38 -- are examples/*.jl (run by
# hand) and test/slow/test_reproduce.jl (FR1-FR8, the slow tier).

const _RP_SEED = 20260912
const _RP_EXAMPLES_DIR = joinpath(@__DIR__, "..", "..", "examples")

include(joinpath(_RP_EXAMPLES_DIR, "problems.jl"))

# ------------------------------------------------------------------------------------------
# RP1: one smoke fit
# ------------------------------------------------------------------------------------------
#
# The EPW problem (examples/problems.jl) cut down to nlambda = 32, nt = 6, fitting only two of
# its three time-varying quantities (logn, T_e; D38's third, m_e, is left out here on purpose --
# RP1 names exactly two trace variables). `ParameterSpec` rebuilds its own trace as "the union of
# leaves" (D9/D10, parameters.jl's `_rebuild_trace_with_bindings`): every leaf already present in
# the input `Experiment`'s trace is copied row for row, and only the leaves named by `bindings`
# become theta-driven. Building the fit's base `Experiment` from `epw_ablation_problem`'s own
# ground-truth trace therefore carries the true, time-varying super-Gaussian exponent m(t)
# through unchanged even though m_e is not a fitted variable here -- there is no shape mismatch
# between the data and the model beyond the two quantities actually being fit.

function _rp1_pspec(expt0::Experiment)
    v_logn = FittedVariable("logn"; bounds=(18.5, 20.5), init=19.0, time_varying=true)
    v_Te = FittedVariable("T_e"; bounds=(100.0, 2000.0), init=900.0, time_varying=true)
    return ParameterSpec(expt0; variables=[v_logn, v_Te], dependents=["ne_cm3" => "10^logn"],
        bindings=["ne_cm3" => "n_e_cm3", "T_e" => "electrons.e.T_eV"])
end

function _rp1_problem()
    problem_nt = epw_ablation_problem(; nlambda=32, nt=6)
    expt0 = build_experiment(problem_nt)
    pspec = _rp1_pspec(expt0)
    working = Experiment(; trace=pspec.trace, probe=expt0.probe, collection=expt0.collection,
        lambda_m=expt0.lambda, effects=expt0.effects, prefactor=expt0.prefactor)
    sd = synthesize(expt0, problem_nt.noise; rng=Xoshiro(_RP_SEED), source="rp1_smoke")
    fitproblem = FitProblem(working, pspec, sd)
    return fitproblem, flatten(pspec)
end

@testset "RP1: one smoke fit" begin
    fitproblem, start = _rp1_problem()
    chi2_start = chi2(fitproblem, start)

    result = solve(fitproblem, LBFGS(iterations=30); start=start)
    @test result isa FitResult
    @test result.chi2 < chi2_start
    @test haskey(result.values, "logn")
    @test haskey(result.values, "T_e")
    @test haskey(result.values, "ne_cm3")

    mktempdir() do dir
        path = joinpath(dir, "rp1_result.jld2")
        write_result(path, result; source="rp1_smoke")
        back = read_result(path)
        @test back.fit.theta == result.theta
        @test back.fit.chi2 == result.chi2
        @test back.fit.R == result.R
    end
end

# ------------------------------------------------------------------------------------------
# RP2: the three decks parse
# ------------------------------------------------------------------------------------------
#
# `read_deck` on each of D38's three reproduction decks, without touching `[data].file` at all
# (that only happens inside `build_problem`, when it calls `read_streak`) -- so this needs no
# streak file on disk and cannot depend on `examples/*.jl` having been run by hand first. Every
# variable and dependent reference in `[plasma]`/`[instrument]` is already checked by `read_deck`
# itself (D39, DK4-DK6): a broken reference would have thrown here, not merely produced a wrong
# `Deck`. The three decks are named `examples/epw_ablation.toml`, `examples/iaw_multiflow.toml`
# and `examples/iaw_shock.toml` -- chunk 12's own choice under rule 11 zero, not the
# `examples/fit_*.toml` names dev/phase3_instructions.md Section 9 anticipated when it was
# written (reported in this chunk's hand-off).

const _RP_DECKS = (
    epw_ablation=("epw_ablation.toml", ["logn", "T_e", "m_e"], ["ne_cm3"]),
    iaw_multiflow=("iaw_multiflow.toml", ["f_D", "T_D", "T_C", "u_D", "u_C"], ["f_C"]),
    iaw_shock=("iaw_shock.toml", ["f_D", "T_D", "T_C", "u_pre", "u_post"], ["f_C", "u_ion"]),
)

@testset "RP2: the three decks parse" begin
    for (name, variables, dependents) in values(_RP_DECKS)
        path = joinpath(_RP_EXAMPLES_DIR, name)
        deck = read_deck(path)
        @test sort(collect(keys(deck.sections["fit_variables"]))) == sort(variables)
        @test sort(collect(keys(deck.sections["fit_dependent"]))) == sort(dependents)
    end
end
