# Step 4.C.2: the regularization split into two intervals around an unregularized gap (Phase 4.C
# spec D12-D16). PW1-PW5, gate "regularizer piecewise". PW1 and PW2 fit a P1-sized step on RG6's
# nonuniform grid; PW3 to PW5 read and build examples/iaw_shock_piecewise.toml. The comparison of
# the shock problem itself is at cut-down only: the slow tier's FR9 and
# examples/diagnostics/shock_piecewise.jl (Phase 4.C spec D25).

using Test
using OTIS
using OTIS: _regularizer_value
using Random: Xoshiro

const _PW_EXAMPLES = joinpath(@__DIR__, "..", "..", "examples")
include(joinpath(_PW_EXAMPLES, "problems.jl"))

# ------------------------------------------------------------------------------------------
# The P1-sized fixture (spec D16): one time-varying flow u_D on RG6's grid, a 150 km/s step between
# bins 5 and 6, two regularizer intervals leaving bins 5 and 6 in no term's window.
# ------------------------------------------------------------------------------------------

const _PW_T_NS = [0.0, 0.31, 0.7, 1.05, 1.5, 1.9, 2.4, 2.75, 3.2, 3.6]   # RG6's grid
const _PW_STEP = 150.0
const _PW_WEIGHT = 1.0e4

function _pw_fixture()
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
    trace = PlasmaTrace(template; t_ns=_PW_T_NS)
    expt = Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact),
        lambda_m=collect(range(262.0, 264.5; length=32)) .* 1e-9,
        effects=(SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6)))
    pre_smooth = FitInterval("pre_smooth", trace.axis; t_range_ns=[0.0, 1.05])    # bins 1:4
    post_smooth = FitInterval("post_smooth", trace.axis; t_range_ns=[2.4, 3.6])   # bins 7:10
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    pspec = ParameterSpec(expt; variables=[v_u], intervals=[pre_smooth, post_smooth],
        bindings=["u_D" => "ions.D.u_kms"])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
    truth = [t < 1.7 ? 0.0 : _PW_STEP for t in _PW_T_NS]
    data = synthesize(unflatten(working, pspec, flatten(pspec, ["u_D" => truth])),
        GaussianNoise(a=1.0, b=0.02); rng=Xoshiro(20260928))
    term(iv) = RegularizerTerm(pspec; select=["u_D"], operator=:D2, normalization=:absolute,
        scale=100.0, interval=iv)
    whole = Regularizer([term(0)]; weight=_PW_WEIGHT)
    pw = Regularizer([term("pre_smooth"), term("post_smooth")]; weight=_PW_WEIGHT)
    return (; pspec, working, data, truth, whole, pw)
end

const _PW = _pw_fixture()

@testset "PW1: two interval terms leave the gap without a row" begin
    pspec = _PW.pspec
    @test _PW.whole.nrows == 8                    # 10 bins, D2
    @test _PW.pw.nrows == 4                       # 2 + 2
    @test [length(t.rows) for t in _PW.pw.terms] == [2, 2]
    covered = union((pspec.intervals[t.interval].bins for t in _PW.pw.terms)...)
    @test setdiff(1:10, covered) == [5, 6]
    base = collect(range(10.0, 60.0; length=10))
    moved = copy(base)
    moved[5] = 250.0
    moved[6] = -250.0
    th_base = flatten(pspec, ["u_D" => base])
    th_moved = flatten(pspec, ["u_D" => moved])
    @test _regularizer_value(_PW.pw, pspec, th_moved) === _regularizer_value(_PW.pw, pspec, th_base)
    @test _regularizer_value(_PW.whole, pspec, th_moved) != _regularizer_value(_PW.whole, pspec, th_base)
end

@testset "PW2: the gap frees the step the whole-axis term smooths" begin
    start = flatten(_PW.pspec)
    rw = solve(FitProblem(_PW.working, _PW.pspec, _PW.data; regularizer=_PW.whole), LevenbergMarquardt(); start=start)
    rp = solve(FitProblem(_PW.working, _PW.pspec, _PW.data; regularizer=_PW.pw), LevenbergMarquardt(); start=start)
    step_w = rw.values["u_D"][6] - rw.values["u_D"][5]
    step_p = rp.values["u_D"][6] - rp.values["u_D"][5]
    @info "PW2 steps" step_w step_p ratio = step_p / step_w status_w = rw.status status_p = rp.status iterations_w = rw.iterations iterations_p = rp.iterations
    @test rw.converged && rp.converged
    @test step_w < 0.6 * _PW_STEP          # measured 67.71: the whole-axis term smooths the step
    @test abs(step_p - _PW_STEP) < 1.5     # measured 149.80: the gap frees it
    @test step_p / step_w > 2.0            # measured 2.212 (2.211 to 2.216 over five seeds)
end

# ------------------------------------------------------------------------------------------
# PW3-PW5: the committed deck
# ------------------------------------------------------------------------------------------

const _PW_DECK = joinpath(_PW_EXAMPLES, "iaw_shock_piecewise.toml")
const _PW_WHOLE_DECK = joinpath(_PW_EXAMPLES, "iaw_shock.toml")

@testset "PW3: the piecewise deck is the shock deck with its regularizer split" begin
    d = read_deck(_PW_DECK)
    w = read_deck(_PW_WHOLE_DECK)
    for key in ("data", "geometry", "fit", "fit_variables", "fit_dependent", "plasma", "instrument",
                "optimizer", "fit_regions")
        @test d.sections[key] == w.sections[key]
    end
    iv = d.sections["fit_intervals"]
    @test sort(collect(keys(iv))) == ["post", "post_smooth", "pre", "pre_smooth"]
    @test iv["pre"]["t_range_ns"] == [0.0, 1.15] && iv["post"]["t_range_ns"] == [1.15, 4.0]
    @test iv["pre_smooth"]["t_range_ns"] == [0.0, 1.05] && iv["post_smooth"]["t_range_ns"] == [1.25, 4.0]
    reg = d.sections["regularizer"]
    @test reg["weight"] == w.sections["regularizer"]["weight"] == 1.0e-1
    @test length(reg["terms"]) == 6
    @test all(t -> t["interval"] in ("pre_smooth", "post_smooth"), reg["terms"])
    @test d.sections["output"]["dir"] == "runs/iaw_shock_piecewise"
end

function _pw_small_streak(dir)
    p = iaw_shock_problem(; nlambda=16, nt=32)
    sd = synthesize(build_experiment(p), p.noise; rng=Xoshiro(p.seed), source="pw4")
    path = joinpath(dir, "pw4_streak.jld2")
    write_streak(path, sd; source="pw4")
    return path
end

function _pw_deck_on(path, sd_path)
    deck = read_deck(path)
    sections = deepcopy(deck.sections)
    sections["data"] = merge(sections["data"], Dict{String,Any}("file" => sd_path))
    return OTIS.Deck(deck.deck_version, sections)
end

@testset "PW4: the piecewise deck builds, its gap at bin 10, theta laid out as the shock deck's" begin
    mktempdir() do dir
        sd_path = _pw_small_streak(dir)
        pw, _, _ = OTIS.build_problem(_pw_deck_on(_PW_DECK, sd_path))
        wh, _, _ = OTIS.build_problem(_pw_deck_on(_PW_WHOLE_DECK, sd_path))
        ivs = Dict(iv.name => iv.bins for iv in pw.pspec.intervals)
        @test ivs == Dict("pre" => 1:9, "post" => 10:32, "pre_smooth" => 1:9, "post_smooth" => 11:32)
        covered = union((pw.pspec.intervals[t.interval].bins for t in pw.regularizer.terms)...)
        @test setdiff(1:32, covered) == [10]
        @test pw.regularizer.nrows == 108 && wh.regularizer.nrows == 118
        @test pw.regularizer.weight == wh.regularizer.weight == 1.0e-1
        @test pw.pspec.ntheta == wh.pspec.ntheta
        @test pw.pspec.offsets == wh.pspec.offsets
        @test pw.pspec.bin_entries == wh.pspec.bin_entries
        @test pw.pspec.global_entries == wh.pspec.global_entries
    end
end

@testset "PW5: a term wider than its variable's interval is refused through the deck" begin
    text = read(_PW_DECK, String)
    text = replace(text, "post_smooth = { t_range_ns = [1.25, 4.0] }" =>
        "post_smooth = { t_range_ns = [1.25, 4.0] }\npre_wide    = { t_range_ns = [0.0, 1.25] }")
    text = replace(text, "{ select = [\"u_pre\"], operator = \"D2\", normalization = \"absolute\", scale = 100.0, interval = \"pre_smooth\" }" =>
        "{ select = [\"u_pre\"], operator = \"D2\", normalization = \"absolute\", scale = 100.0, interval = \"pre_wide\" }")
    mktempdir() do dir
        sd_path = _pw_small_streak(dir)
        path = joinpath(dir, "pw5.toml")
        write(path, text)
        @test occursin("pre_wide", text) && occursin("interval = \"pre_wide\"", text)
        @test_throws "the term selecting \"u_pre\" is on interval \"pre_wide\" (bins 1:10), but \"u_pre\" is defined only on its own interval \"pre\" (bins 1:9)" OTIS.build_problem(_pw_deck_on(path, sd_path))
    end
end
