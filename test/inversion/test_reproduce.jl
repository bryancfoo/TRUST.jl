using Test
using TRUST
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

# ------------------------------------------------------------------------------------------
# RP3: the three decks carry Table I's normalizations
# ------------------------------------------------------------------------------------------
#
# Step 3.C.6a / D5: foo2026 Table I marks n_e, T_e, T_D and T_C "Relative", and D38's own table
# cell was never updated to match its prose -- chunk 12 built the three decks from the stale
# cell, carrying "initial" where Table I (and D5) call for "relative". This pins the correction
# against silent reversion: the term selecting each deck's Table I "Relative" variables reads
# normalization = "relative", the unit terms (m_e, f_D) are still "unit", and the flow terms
# (u_D/u_C, u_pre, u_post) are still "absolute" with scale = 100.0 (Table I's 100 km/s).
#
# `read_deck` leaves `normalization` and `scale` as raw strings/numbers in
# `deck.sections["regularizer"]["terms"]`, a Vector{Dict{String,Any}} -- the conversion to a
# Symbol through NORMALIZATION_REGISTRY happens later, in build_problem's `_build_regularizer`
# (config/deck.jl) -- so this test reads the deck's own strings directly, never a built
# Regularizer. Violations are collected by deck and term (DK10's idiom, chunk 1) so a failure
# names exactly which deck and which term regressed, rather than a bare @test false.

const _RP3_RELATIVE_VARS = (
    epw_ablation=("epw_ablation.toml", ["logn", "T_e"]),
    iaw_multiflow=("iaw_multiflow.toml", ["T_D", "T_C"]),
    iaw_shock=("iaw_shock.toml", ["T_D", "T_C"]),
)

@testset "RP3: the three decks carry Table I's normalizations" begin
    found = String[]
    for (name, relative_vars) in values(_RP3_RELATIVE_VARS)
        path = joinpath(_RP_EXAMPLES_DIR, name)
        terms = read_deck(path).sections["regularizer"]["terms"]

        relative_terms = filter(t -> Set(t["select"]) == Set(relative_vars), terms)
        if length(relative_terms) != 1
            push!(found, "$name: expected exactly one term selecting $relative_vars, found $(length(relative_terms))")
        else
            norm = relative_terms[1]["normalization"]
            norm == "relative" ||
                push!(found, "$name: term selecting $relative_vars has normalization = \"$norm\", expected \"relative\"")
        end

        for t in terms
            if t["normalization"] == "unit" && Set(t["select"]) == Set(relative_vars)
                push!(found, "$name: term selecting $relative_vars is \"unit\", expected \"relative\"")
            end
        end

        unit_terms = filter(t -> t["normalization"] == "unit", terms)
        isempty(unit_terms) &&
            push!(found, "$name: expected at least one \"unit\" term, found none")

        # epw_ablation fits no flow variable, so it has no absolute term at all; the other two
        # decks do (u_D/u_C, u_pre/u_post) -- check whichever absolute terms exist, not that any do.
        absolute_terms = filter(t -> t["normalization"] == "absolute", terms)
        for t in absolute_terms
            scale = get(t, "scale", nothing)
            scale == 100.0 ||
                push!(found, "$name: absolute term selecting $(t["select"]) has scale = $scale, expected 100.0")
        end
    end
    @test isempty(found)
end

# ------------------------------------------------------------------------------------------
# RP4: `_check_converged`'s dispatch, exercised from the real scripts
# ------------------------------------------------------------------------------------------
#
# Phase 3.C remediation round, item 3 (dev/reports/phase3C_report.md, chunk 8, Section 6 item 1):
# `_check_converged`'s signature in all three examples/fit_*.jl scripts,
# `extra::Vector{Pair{String,FitResult}}`, could never be called. `FitResult` is parametric
# (`FitResult{FT,E,P}`) and `Vector` is invariant in its element type, so `["label" => shown]` --
# the literal expression every script passes -- builds a `Vector{Pair{String,
# FitResult{Float64,...}}}`, never a `Vector{Pair{String,FitResult}}` (the bare `FitResult` in the
# old signature binds to the UnionAll itself, not to any instantiation of it). Every call raised
# `MethodError`; since every route to `write_result` in all three scripts passes through this
# guard first, the bug would have silently and permanently blocked Milestone 3.C's own acceptance
# criterion had chunk 8 not run a script to completion for the first time since D34 added the
# refusal. Fixed to `AbstractVector{<:Pair{<:AbstractString,<:FitResult}}`.
#
# The three scripts are plain top-level programs, not modules: each `include`s a full
# forward-model synthesis and then unconditionally fits at 512 x 256 at the bottom of the file, so
# `include`-ing one here to reach its `_check_converged` would run several hours of compute (D20's
# per-command budget forbids it), and moving the helper into src/ is an API change this
# remediation round does not license. `_rp4_extract` instead parses a script's own committed
# source text with `Meta.parse` -- the same parser Julia itself uses to read the file -- and pulls
# out exactly the `_check_converged` function and the `ALLOW_UNCONVERGED` constant it closes over,
# as the literal expressions actually committed, never hand-copied, without evaluating any of the
# file's own `include`s or its closing call.

function _rp4_extract(script_path::AbstractString)
    text = read(script_path, String)
    found = Dict{Symbol,Expr}()
    pos = firstindex(text)
    while pos <= lastindex(text)
        ex, pos = Meta.parse(text, pos)
        ex === nothing && break
        if ex isa Expr && ex.head === :function && ex.args[1].args[1] === :_check_converged
            found[:_check_converged] = ex
        elseif ex isa Expr && ex.head === :const && ex.args[1].args[1] === :ALLOW_UNCONVERGED
            found[:ALLOW_UNCONVERGED] = ex
        end
    end
    haskey(found, :_check_converged) || error("_rp4_extract: _check_converged not found in $script_path")
    haskey(found, :ALLOW_UNCONVERGED) || error("_rp4_extract: ALLOW_UNCONVERGED not found in $script_path")
    return found
end

# Evaluates the extracted expressions into this file's own (safetestset) module and returns the
# resulting function. `withenv` pins ALLOW_UNCONVERGED to its default (env var unset) regardless
# of the session's own environment, since the error-vs-warn branch `_check_converged`'s own body
# takes depends on it, and this row's assertions require the `error` branch deterministically.
function _rp4_load_check_converged(script_path::AbstractString)
    found = _rp4_extract(script_path)
    withenv("TRUST_ALLOW_UNCONVERGED" => nothing) do
        eval(found[:ALLOW_UNCONVERGED])
    end
    eval(found[:_check_converged])
    # Julia 1.12's stricter world-age rules for global bindings (TRUST_TESTS="reproduce" surfaced
    # this as a warning: "access to binding ... in a world prior to its definition world") apply
    # to reading the binding, not only to calling it -- `getfield` itself is ancient and safe to
    # invokelatest, and doing so makes the read, not just the later call, see the new definition.
    return Base.invokelatest(getfield, @__MODULE__, :_check_converged)
end

_rp4_converged(r::FitResult, converged::Bool, status::Symbol) =
    FitResult(r.theta, r.experiment, r.pspec, r.values, r.near_bound, r.chi2, r.R, r.objective,
        r.weight, r.term_values, r.npixels, r.trajectory, converged, status, r.termination,
        r.iterations, r.f_calls, r.g_calls, r.elapsed, r.optimizer)

_rp4_rescan(s::WeightScanResult, unregularized::FitResult, results::Vector{<:FitResult}) =
    WeightScanResult(s.weights, s.chi2, s.R, s.chi2_normalized, s.R_normalized, s.chi2_unreg,
        s.R_unreg, s.rss, s.rss_normalized, s.rss_unreg, results, unregularized)

# A tiny, cheap fixture: two time bins (nt = 1 left `examples/epw_ablation.jl`'s own temporal-IRF
# convolution untested at its edge case, which is not this row's concern, so nt = 2), an `:I`
# (ridge) regularizer term per variable so the row count does not depend on how many bins there
# are the way `:D1`/`:D2` do, two weights, three LBFGS iterations. Nothing here needs to actually
# converge: RP4 overrides every result's own `converged` flag directly rather than trusting the
# solver's, so the fixture only has to be a real, cheap, correctly-typed `FitResult`/
# `WeightScanResult` pair to dispatch against.
function _rp4_tiny_problem()
    problem_nt = epw_ablation_problem(; nlambda=16, nt=2)
    expt0 = build_experiment(problem_nt)
    pspec = _rp1_pspec(expt0)
    working = Experiment(; trace=pspec.trace, probe=expt0.probe, collection=expt0.collection,
        lambda_m=expt0.lambda, effects=expt0.effects, prefactor=expt0.prefactor)
    sd = synthesize(expt0, problem_nt.noise; rng=Xoshiro(_RP_SEED), source="rp4_tiny")
    term_logn = RegularizerTerm(pspec; select=["logn"], operator=:I, normalization=:absolute, weight=1.0)
    term_Te = RegularizerTerm(pspec; select=["T_e"], operator=:I, normalization=:absolute, weight=1.0)
    reg = Regularizer([term_logn, term_Te]; weight=1.0)
    return FitProblem(working, pspec, sd; regularizer=reg), flatten(pspec)
end

@testset "RP4: _check_converged is byte-identical across the three scripts" begin
    scripts = ("fit_epw_ablation.jl", "fit_iaw_multiflow.jl", "fit_iaw_shock.jl")
    parsed = [_rp4_extract(joinpath(_RP_EXAMPLES_DIR, f))[:_check_converged] for f in scripts]
    @test all(==(parsed[1]), parsed[2:end])
end

@testset "RP4: _check_converged accepts and enforces the literal [\"label\" => shown] form" begin
    check_converged = _rp4_load_check_converged(joinpath(_RP_EXAMPLES_DIR, "fit_epw_ablation.jl"))

    problem, start = _rp4_tiny_problem()
    raw = weight_scan(problem, [1.0e-3, 1.0e-1], LBFGS(iterations=3); start=start)

    unreg_ok = _rp4_converged(raw.unregularized, true, :f_tolerance)
    results_ok = FitResult[_rp4_converged(r, true, :f_tolerance) for r in raw.results]
    scan_ok = _rp4_rescan(raw, unreg_ok, results_ok)
    shown_ok = _rp4_converged(raw.results[1], true, :f_tolerance)

    # 1. a scan whose members all converged, plus a converged `extra`, returns normally.
    @test Base.invokelatest(check_converged, scan_ok,
        ["shown (Lambda = $(raw.weights[1]))" => shown_ok]) === nothing

    # 2. a non-converged scan member makes it `error`, naming the offending weight.
    results_bad = copy(results_ok)
    results_bad[2] = _rp4_converged(results_bad[2], false, :iterations)
    scan_bad = _rp4_rescan(raw, unreg_ok, results_bad)
    err = try
        Base.invokelatest(check_converged, scan_bad, ["shown (Lambda = $(raw.weights[1]))" => shown_ok])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Lambda = $(raw.weights[2])", err.msg)

    # 3. a non-converged `extra` member errors too, naming its own label -- through the same
    #    literal `["label" => shown]` argument form that used to MethodError before the fix.
    shown_bad = _rp4_converged(shown_ok, false, :iterations)
    err2 = try
        Base.invokelatest(check_converged, scan_ok, ["shown (bad)" => shown_bad])
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("shown (bad)", err2.msg)
end
