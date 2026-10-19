using Test
using TRUST
using TRUST: _leaf_si_path
using Random: Xoshiro
using Statistics: median

# CS4 against dev/phase4A_instructions.md Section 10.1's slow-tier row (spec D46), gated
# "uncertainty (slow)". The reproductions' cut-down EPW problem, epw_ablation_problem(; nlambda =
# 128, nt = 32), the EPW deck's own three variables (logn, T_e, m_e), with a
# WavelengthCalibration(lambda_offset_nm = 0.0) added to the fitting experiment's effects,
# unbound -- so the scan below is a pure calibration probe, not itself fitted.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

function _cs4_expt0(problem_nt)
    effects = (problem_nt.effects..., WavelengthCalibration(lambda_offset_nm=0.0))
    return Experiment(trace=problem_nt.trace, probe=problem_nt.probe, collection=problem_nt.collection,
        lambda_m=problem_nt.lambda, effects=effects)
end

function _cs4_pspec(expt0::Experiment)
    v_logn = FittedVariable("logn"; bounds=(18.5, 20.5), init=19.176, time_varying=true)
    v_Te = FittedVariable("T_e"; bounds=(100.0, 2000.0), init=350.0, time_varying=true)
    v_me = FittedVariable("m_e"; bounds=(2.0, 5.5), init=2.5, time_varying=true)
    return ParameterSpec(expt0; variables=[v_logn, v_Te, v_me], dependents=["ne_cm3" => "10^logn"],
        bindings=["ne_cm3" => "n_e_cm3", "T_e" => "electrons.e.T_eV", "m_e" => "electrons.e.f.m"])
end

# Reads a leaf's own true values off `trace` in the boundary key's own unit (the reproductions'
# own `_fr_truth`, test/slow/test_reproduce.jl, copied here under the @safetestset boundary rule).
function _cs4_truth(trace::PlasmaTrace, boundary_key::AbstractString)
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    i === nothing && error("_cs4_truth: leaf \"$si_path\" not present in this trace")
    return collect(Float64, trace.values[i, :]) ./ factor
end

# theta*(0): LM, then polish, started at the truth -- the EPW deck's own inits stall (Pitfall 31),
# and this is a synthetic sensitivity test, not a recovery test.
function _cs4_problem_and_star()
    problem_nt = epw_ablation_problem(; nlambda=128, nt=32)
    expt0 = _cs4_expt0(problem_nt)
    pspec = _cs4_pspec(expt0)
    working = Experiment(; trace=pspec.trace, probe=expt0.probe, collection=expt0.collection,
        lambda_m=expt0.lambda, effects=expt0.effects, prefactor=expt0.prefactor)

    theta_truth = flatten(pspec, [
        "logn" => log10.(_cs4_truth(pspec.trace, "n_e_cm3")),
        "T_e" => _cs4_truth(pspec.trace, "electrons.e.T_eV"),
        "m_e" => _cs4_truth(pspec.trace, "electrons.e.f.m")])

    truth_expt = unflatten(working, pspec, theta_truth)
    sd = synthesize(truth_expt, problem_nt.noise; rng=Xoshiro(problem_nt.seed), source="cs4_slow")

    reg = Regularizer([
            RegularizerTerm(pspec; select=["logn", "T_e"], operator=:D2, normalization=:relative, weight=1.0),
            RegularizerTerm(pspec; select=["m_e"], operator=:D2, normalization=:unit, weight=1.0)];
        weight=2.0e-4)

    problem = FitProblem(working, pspec, sd; regularizer=reg)

    t0 = time()
    r = solve(problem, LevenbergMarquardt(); start=theta_truth)
    pol = polish(problem, r)
    build_elapsed = time() - t0
    return problem, pol.result.theta, build_elapsed, r
end

@testset "CS4: the calibration scan on the cut-down EPW problem (128 x 32)" begin
    problem, theta_star, build_elapsed, r0 = _cs4_problem_and_star()
    @info "CS4: theta*(0) build" build_elapsed r0.status r0.iterations gn_decrease(problem, theta_star)

    shifts = [-0.2, -0.1, 0.0, 0.1, 0.2]
    t0 = time()
    scan = calibration_scan(problem, theta_star, shifts; channel=1)
    elapsed = time() - t0
    @info "CS4: calibration_scan elapsed (5 points)" elapsed per_point = elapsed / length(shifts)

    # Fixed by D49 (Phase 4.A remediation round 2, 2026-09-25): the -0.1 nm point's stall was a
    # forward-model bug, not the optimizer -- see dev/phase4A_instructions.md Section 10.1R.
    @test all(scan.decrease .<= 1e-6)
    zero_idx = findfirst(iszero, scan.shifts_nm)
    @test scan.results[zero_idx].theta == theta_star

    # Reported, never asserted (Section 10.1, the roadmap's own Test): the median movement of
    # logn and T_e per shift, their s_sys, and whether n_e moves monotonically with the shift
    # while T_e moves far less -- the EPW's red-shifted peak sits at a larger |Δλ| for larger n_e.
    med_dlogn = [median(scan.delta["logn"][:, k]) for k in eachindex(shifts)]
    med_dTe = [median(scan.delta["T_e"][:, k]) for k in eachindex(shifts)]
    for (k, s) in enumerate(shifts)
        @info "CS4 report: shift" shift_nm = s median_dlogn = med_dlogn[k] median_dTe = med_dTe[k] chi2 = scan.chi2[k] status = scan.status[k]
    end
    @info "CS4 report: s_sys" s_sys_logn = scan.s_sys["logn"] s_sys_Te = scan.s_sys["T_e"] s_sys_me = scan.s_sys["m_e"]
    monotone_ne = issorted(med_dlogn) || issorted(med_dlogn; rev=true)
    ratio = maximum(abs, med_dTe) / max(maximum(abs, med_dlogn), eps())
    @info "CS4 report: does n_e move monotonically with the shift, and by how much more than T_e" monotone_ne ratio_dTe_over_dlogn = ratio
end

# ------------------------------------------------------------------------------------------
# UC1-UC3 (D54, amending D48 item 3): 60 noise realizations of a single-channel, P1-sized
# synthetic problem, comparing the sampling-form Laplace covariance and the per-slice scatter
# against the truth, and against each other. Gated "uncertainty (slow)".
#
# GaussianNoise, not StreakCameraNoise (Pitfall 38): StreakCameraNoise's one-pixel point spread
# blurs the data's mean with a kernel the forward model does not have, so a coverage test against
# it would be measuring the point spread's bias, not the machinery. GaussianNoise is Swadling's
# sigma formula without the point spread.
# ------------------------------------------------------------------------------------------

_uc_template() = Plasma(; n_e_cm3=3e19,
    electrons=[electrons(T_eV=400.0)],
    ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])

_uc_TD(t_ns) = 180.0 + 60.0 * t_ns
_uc_uD(t_ns) = 20.0 + 15.0 * t_ns

function _uc_raw_expt()
    t_ns = collect(range(0.0, 3.0; length=96))
    trace = PlasmaTrace(_uc_template(); t_ns=t_ns,
        leaves=("ions.D.T_eV" => _uc_TD, "ions.D.u_kms" => _uc_uD))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda = collect(range(262.0, 264.5; length=24)) .* 1e-9
    effects = (Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0))
    return Experiment(; trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
end

const _UC_NOISE = GaussianNoise(; a=11.0^2, b=135.0 * 1.05)

_uc_path(p::AbstractString) = replace(p, "\\" => "/")

function _uc_deck_text(datafile::AbstractString, outdir::AbstractString; regularizer::Bool=true)
    reg_block = regularizer ? """
    [regularizer]
    weight = 30.0
    terms = [
      { select = ["T_D", "u_D"], operator = "D2", normalization = "relative" },
    ]
    """ : ""
    return """
    deck_version = "trust-deck-2"

    [data]
    file = "$(_uc_path(datafile))"

    [[fit.regions]]
    t_range_ns = [0.0, 3.0]
    nbins = 8

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"

    [instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = "u_D"

    $reg_block

    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_uc_path(outdir))"
    """
end

# One noise realization: synthesize at full (96-slice) resolution, write the streak (rewritten
# in place each call), build the regularized and the unregularized problem from the same data,
# theta* started at the truth (a coverage test, not a recovery test, Pitfall 31), laplace both,
# and the per-slice scatter.
function _uc_realization(dir::AbstractString, r::Integer)
    raw_expt = _uc_raw_expt()
    sd = synthesize(raw_expt, _UC_NOISE; rng=Xoshiro(20260925 + r))
    datafile = joinpath(dir, "uc_streak.jld2")
    write_streak(datafile, sd; source="uc_slow")

    deckpath = joinpath(dir, "uc_deck.toml")
    write(deckpath, _uc_deck_text(datafile, joinpath(dir, "out")))
    deck = read_deck(deckpath)
    problem, opt, _ = TRUST.build_problem(deck)

    truth_ns = t_from_si.(problem.pspec.axis, :t_ns)
    start = flatten(problem.pspec, ["T_D" => _uc_TD.(truth_ns), "u_D" => _uc_uD.(truth_ns)])
    fit0 = solve(problem, opt; start=start)
    pol = polish(problem, fit0)
    theta_star = pol.result.theta
    lap = laplace(problem, theta_star)

    deckpath_unreg = joinpath(dir, "uc_deck_unreg.toml")
    write(deckpath_unreg, _uc_deck_text(datafile, joinpath(dir, "out_unreg"); regularizer=false))
    deck_unreg = read_deck(deckpath_unreg)
    problem_unreg, opt_unreg, _ = TRUST.build_problem(deck_unreg)
    fit0_unreg = solve(problem_unreg, opt_unreg; start=theta_star)
    pol_unreg = polish(problem_unreg, fit0_unreg)
    lap_unreg = laplace(problem_unreg, pol_unreg.result.theta)

    sc = slice_scatter(deck, pol.result)

    return (; deck, problem, result=pol.result, theta_star, lap, lap_unreg, sc, truth_ns)
end

const _UC_VARIABLES = ("T_D", "u_D")
const _UC_NBINS = 8

function _uc_run(n_realizations::Integer=60)
    return mktempdir() do dir
        cov1_1 = Dict(v => 0 for v in _UC_VARIABLES); cov1_2 = Dict(v => 0 for v in _UC_VARIABLES)
        cov1_post_1 = Dict(v => 0 for v in _UC_VARIABLES); cov1_post_2 = Dict(v => 0 for v in _UC_VARIABLES)
        cov2_1 = Dict(v => 0 for v in _UC_VARIABLES); cov2_2 = Dict(v => 0 for v in _UC_VARIABLES)
        n1 = Dict(v => 0 for v in _UC_VARIABLES); n2 = Dict(v => 0 for v in _UC_VARIABLES)

        s_stat_sum = Dict(v => zeros(_UC_NBINS) for v in _UC_VARIABLES)
        sigma_unreg_sum = Dict(v => zeros(_UC_NBINS) for v in _UC_VARIABLES)
        sigma_samp_sum = Dict(v => zeros(_UC_NBINS) for v in _UC_VARIABLES)
        theta_sum = Dict(v => zeros(_UC_NBINS) for v in _UC_VARIABLES)
        theta_sumsq = Dict(v => zeros(_UC_NBINS) for v in _UC_VARIABLES)

        polish_max_decrease = 0.0
        fine_max_decrease = 0.0
        m_ref = Int[]
        n_near_bound_total = 0

        t0 = time()
        for r in 1:n_realizations
            rr = _uc_realization(dir, r)
            r == 1 && (m_ref = rr.sc.m)
            polish_max_decrease = max(polish_max_decrease, rr.lap.decrease)
            fine_max_decrease = max(fine_max_decrease, maximum(rr.sc.decrease))
            n_near_bound_total += sum(count, values(rr.sc.near_bound))

            for v in _UC_VARIABLES
                truthv_coarse = v == "T_D" ? _uc_TD.(rr.truth_ns) : _uc_uD.(rr.truth_ns)
                for k in 1:_UC_NBINS
                    d = abs(rr.result.values[v][k] - truthv_coarse[k])
                    s = rr.lap.sigma_samp[v][k]
                    sp = rr.lap.sigma_post[v][k]
                    n1[v] += 1
                    d <= s && (cov1_1[v] += 1)
                    d <= 2s && (cov1_2[v] += 1)
                    d <= sp && (cov1_post_1[v] += 1)
                    d <= 2sp && (cov1_post_2[v] += 1)
                    s_stat_sum[v][k] += rr.sc.s_stat[v][k]
                    sigma_unreg_sum[v][k] += rr.lap_unreg.sigma_samp[v][k]
                    sigma_samp_sum[v][k] += s
                    theta_sum[v][k] += rr.result.values[v][k]
                    theta_sumsq[v][k] += rr.result.values[v][k]^2
                end

                truthv_fine = v == "T_D" ? _uc_TD.(rr.sc.t_fine .* 1e9) : _uc_uD.(rr.sc.t_fine .* 1e9)
                for j in eachindex(rr.sc.t_fine)
                    d = abs(rr.sc.values[v][j] - truthv_fine[j])
                    ss = rr.sc.s_stat[v][rr.sc.coarse_bin[j]]
                    n2[v] += 1
                    d <= ss && (cov2_1[v] += 1)
                    d <= 2ss && (cov2_2[v] += 1)
                end
            end
        end
        elapsed = time() - t0

        rates1 = Dict(v => (cov1_1[v] / n1[v], cov1_2[v] / n1[v]) for v in _UC_VARIABLES)
        rates1_post = Dict(v => (cov1_post_1[v] / n1[v], cov1_post_2[v] / n1[v]) for v in _UC_VARIABLES)
        rates2 = Dict(v => (cov2_1[v] / n2[v], cov2_2[v] / n2[v]) for v in _UC_VARIABLES)

        uc3_ratio = Dict{String,Vector{Float64}}()
        sigma_ratio_med = Dict{String,Float64}()
        empirical_sd_over_sigma = Dict{String,Vector{Float64}}()
        for v in _UC_VARIABLES
            mean_s_stat = s_stat_sum[v] ./ n_realizations
            mean_sigma_unreg = sigma_unreg_sum[v] ./ n_realizations
            mean_sigma_samp = sigma_samp_sum[v] ./ n_realizations
            uc3_ratio[v] = mean_s_stat ./ (sqrt(m_ref[1]) .* mean_sigma_unreg)
            sigma_ratio_med[v] = median(mean_sigma_samp ./ mean_sigma_unreg)
            mean_theta = theta_sum[v] ./ n_realizations
            var_theta = max.(theta_sumsq[v] ./ n_realizations .- mean_theta .^ 2, 0.0)
            empirical_sd_over_sigma[v] = sqrt.(var_theta) ./ mean_sigma_samp
        end

        return (; elapsed, n_realizations, rates1, rates1_post, rates2, uc3_ratio, sigma_ratio_med,
            empirical_sd_over_sigma, polish_max_decrease, fine_max_decrease, m=m_ref,
            n_near_bound_total)
    end
end

const _UC_REPORT = _uc_run(60)

@testset "UC1: the sampling-form Laplace coverage against the truth" begin
    r = _UC_REPORT
    for v in _UC_VARIABLES
        rate1, rate2 = r.rates1[v]
        @test 0.58 <= rate1 <= 0.78
        @test 0.90 <= rate2 <= 0.99
        @info "UC1 sampling-form coverage" variable = v within_1sigma = rate1 within_2sigma = rate2
    end
    # Reported, never asserted: the posterior form's own coverage.
    for v in _UC_VARIABLES
        rate1p, rate2p = r.rates1_post[v]
        @info "UC1 posterior-form coverage (reported only)" variable = v within_1sigma = rate1p within_2sigma = rate2p
    end
end

@testset "UC2: the fine-window scatter's own coverage against the truth" begin
    r = _UC_REPORT
    for v in _UC_VARIABLES
        rate1, rate2 = r.rates2[v]
        @test 0.58 <= rate1 <= 0.78
        @test 0.90 <= rate2 <= 0.99
        @info "UC2 coverage" variable = v within_1sigma = rate1 within_2sigma = rate2
    end
end

@testset "UC3: mean(s_stat) / (sqrt(m) * mean(sigma_unreg)), per bin" begin
    # sqrt(m) uses the fine windows' own count (D54's own caveat): 8 of the 88 fine windows hold
    # two raw slices instead of one, so the expected ratio here is sqrt(12/11) ~ 1.04, not
    # exactly 1.
    r = _UC_REPORT
    for v in _UC_VARIABLES
        ratios = r.uc3_ratio[v]
        @test all(0.8 .<= ratios .<= 1.25)
        med = median(ratios)
        @test 0.9 <= med <= 1.15
        @info "UC3 ratio" variable = v ratios median = med
    end
    @info "UC coverage tests: wall time and fixture" elapsed_s = r.elapsed per_realization_s = r.elapsed / r.n_realizations m = r.m polish_max_decrease = r.polish_max_decrease fine_max_decrease = r.fine_max_decrease n_near_bound_total = r.n_near_bound_total
    for v in _UC_VARIABLES
        @info "UC report: sigma_samp/sigma_unreg median (regularizer active)" variable = v median_ratio = r.sigma_ratio_med[v]
        @info "UC report: empirical SD / mean sigma_samp per bin" variable = v ratios = r.empirical_sd_over_sigma[v]
    end
end
