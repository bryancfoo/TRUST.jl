using Test
using OTIS
using Random: Xoshiro

# TC1 to TC3 against dev/phase4A_instructions.md Section 9.3 (spec D41, chunk 8), gated
# "two-channel problem". No row here fits anything (D41 amendment 4) -- TC1 checks the problem
# and the synthesis axis, TC2 the degeneracy pair, TC3 the example deck's wiring.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const _TC_CUTDOWN = (nlambda_epw=64, nlambda_iaw=64, nt_epw=128, nt_iaw=120, nbins=8)
const P = two_channel_problem(; _TC_CUTDOWN...)

# ------------------------------------------------------------------------------------------
# TC1: the problem as specified
# ------------------------------------------------------------------------------------------

@testset "TC1: two_channel_problem as specified" begin
    # recorded axes' ends and lengths
    @test P.epw.t_ns_recorded[1] == 2.0
    @test P.epw.t_ns_recorded[end] == 5.8
    @test length(P.epw.t_ns_recorded) == 128
    @test P.iaw.t_ns_recorded[1] == 1.95
    @test P.iaw.t_ns_recorded[end] == 5.75
    @test length(P.iaw.t_ns_recorded) == 120

    # the windows
    @test P.epw.lambda[1] == 220e-9
    @test P.epw.lambda[end] == 260e-9
    @test length(P.epw.lambda) == 64
    @test P.iaw.lambda[1] == 262.0e-9
    @test P.iaw.lambda[end] == 264.5e-9
    @test length(P.iaw.lambda) == 64

    # the seeds and the shift
    @test P.epw.seed == 20260924
    @test P.iaw.seed == 20260925
    @test P.epw.t_shift_ns == 0.0
    @test P.iaw.t_shift_ns == 0.05

    # the region
    @test length(P.regions) == 1
    @test P.regions[1].nbins == 8
    @test isapprox(P.regions[1].lo, t_to_si(2.2, :t_ns); rtol=1e-14)
    @test isapprox(P.regions[1].hi, t_to_si(5.6, :t_ns); rtol=1e-14)

    # every profile positive and finite on each camera's true axis (D41 amendment 1's guard)
    base = Dict(P.leaves)
    ne_f, Te_f, m_f, TD_f = base["n_e_cm3"], base["electrons.e.T_eV"], base["electrons.e.f.m"], base["ions.D.T_eV"]
    true_epw = P.epw.t_ns_recorded .+ P.epw.t_shift_ns
    true_iaw = P.iaw.t_ns_recorded .+ P.iaw.t_shift_ns
    for t in vcat(true_epw, true_iaw)
        ne, Te, m, TD = ne_f(t), Te_f(t), m_f(t), TD_f(t)
        @test isfinite(ne) && ne > 0
        @test isfinite(Te) && Te > 0
        @test isfinite(m)
        @test isfinite(TD) && TD > 0
    end

    # the IAW streak's recorded axis
    sd_iaw = synthesize_channel(P, :iaw)
    @test isapprox(sd_iaw.t, t_to_si.(P.iaw.t_ns_recorded, :t_ns); rtol=1e-14)

    # synthesize_channel(p, :iaw).image is === elementwise the image of a synthesis built by
    # hand on the TRUE axis -- the only row that sees which axis the IRF acted on.
    trace_true = PlasmaTrace(P.template; t_ns=P.iaw.t_ns_recorded .+ 0.05, leaves=P.leaves)
    expt_true = Experiment(; trace=trace_true, probe=P.probe, collection=P.collection,
        lambda_m=P.iaw.lambda, effects=P.iaw.effects)
    sd_true = synthesize(expt_true, P.iaw.noise; rng=Xoshiro(20260925))
    @test all(sd_iaw.image .=== sd_true.image)
end

# ------------------------------------------------------------------------------------------
# TC2: the degeneracy pair, as a ratio (D41 amendment 2's procedure)
# ------------------------------------------------------------------------------------------

# Each camera synthesized on its true axis with its own seed, relabelled, averaged onto the
# region (the IAW with t_shift_ns = 0.05) and pooled (D39). The noise-free model is then
# evaluated directly on the pooled axis, with the averaged sigma and mask, for the truth and for
# a perturbed n_e or T_D. Delta-chi2 per bin is the masked sum of squared standardized
# differences, divided by the region's nbins.
function _tc2_pooled(p)
    sd_epw = synthesize_channel(p, :epw)
    sd_iaw = synthesize_channel(p, :iaw)
    avg_epw = average_regions(sd_epw, p.regions)
    avg_iaw = average_regions(sd_iaw, p.regions; t_shift_ns=p.iaw.t_shift_ns)
    wc_epw = avg_epw.meta["regions_window_counts"]
    wc_iaw = avg_iaw.meta["regions_window_counts"]
    pooled_t_ns = (wc_epw .* avg_epw.t .+ wc_iaw .* avg_iaw.t) ./ (wc_epw .+ wc_iaw) .* 1e9
    return (; avg_epw, avg_iaw, pooled_t_ns)
end

function _tc2_leaves(p, ne_mult::Real, td_mult::Real)
    base = Dict(p.leaves)
    ne_f, TD_f = base["n_e_cm3"], base["ions.D.T_eV"]
    return ("n_e_cm3" => t -> ne_f(t) * ne_mult, "electrons.e.T_eV" => base["electrons.e.T_eV"],
        "electrons.e.f.m" => base["electrons.e.f.m"], "ions.D.T_eV" => t -> TD_f(t) * td_mult)
end

function _tc2_model(p, name::Symbol, pooled_t_ns::AbstractVector, ne_mult::Real, td_mult::Real)
    c = getproperty(p, name)
    trace = PlasmaTrace(p.template; t_ns=pooled_t_ns, leaves=_tc2_leaves(p, ne_mult, td_mult))
    expt = Experiment(; trace=trace, probe=p.probe, collection=p.collection, lambda_m=c.lambda,
        effects=c.effects)
    return measure(expt)
end

function _tc2_dchi2_per_bin(p, name::Symbol, pooled_t_ns::AbstractVector, sigma_avg, mask_avg,
    ne_mult::Real, td_mult::Real)
    M0 = _tc2_model(p, name, pooled_t_ns, 1.0, 1.0)
    M1 = _tc2_model(p, name, pooled_t_ns, ne_mult, td_mult)
    total = 0.0
    for idx in eachindex(M0)
        mask_avg[idx] || continue
        total += ((M1[idx] - M0[idx]) / sigma_avg[idx])^2
    end
    return total / length(pooled_t_ns)
end

@testset "TC2: the degeneracy pair, as a ratio" begin
    pooled = _tc2_pooled(P)
    ne_up, ne_dn = 10.0^0.05, 10.0^(-0.05)

    epw_logn_up = _tc2_dchi2_per_bin(P, :epw, pooled.pooled_t_ns, pooled.avg_epw.sigma, pooled.avg_epw.mask, ne_up, 1.0)
    epw_logn_dn = _tc2_dchi2_per_bin(P, :epw, pooled.pooled_t_ns, pooled.avg_epw.sigma, pooled.avg_epw.mask, ne_dn, 1.0)
    iaw_logn_up = _tc2_dchi2_per_bin(P, :iaw, pooled.pooled_t_ns, pooled.avg_iaw.sigma, pooled.avg_iaw.mask, ne_up, 1.0)
    iaw_logn_dn = _tc2_dchi2_per_bin(P, :iaw, pooled.pooled_t_ns, pooled.avg_iaw.sigma, pooled.avg_iaw.mask, ne_dn, 1.0)

    epw_td_up = _tc2_dchi2_per_bin(P, :epw, pooled.pooled_t_ns, pooled.avg_epw.sigma, pooled.avg_epw.mask, 1.0, 1.2)
    epw_td_dn = _tc2_dchi2_per_bin(P, :epw, pooled.pooled_t_ns, pooled.avg_epw.sigma, pooled.avg_epw.mask, 1.0, 0.8)
    iaw_td_up = _tc2_dchi2_per_bin(P, :iaw, pooled.pooled_t_ns, pooled.avg_iaw.sigma, pooled.avg_iaw.mask, 1.0, 1.2)
    iaw_td_dn = _tc2_dchi2_per_bin(P, :iaw, pooled.pooled_t_ns, pooled.avg_iaw.sigma, pooled.avg_iaw.mask, 1.0, 0.8)

    logn_ratio_up = epw_logn_up / iaw_logn_up
    logn_ratio_dn = epw_logn_dn / iaw_logn_dn
    td_ratio_up = iaw_td_up / epw_td_up
    td_ratio_dn = iaw_td_dn / epw_td_dn

    @info "TC2: measured numbers" epw_logn_up epw_logn_dn iaw_logn_up iaw_logn_dn logn_ratio_up logn_ratio_dn epw_td_up epw_td_dn iaw_td_up iaw_td_dn td_ratio_up td_ratio_dn

    @test epw_logn_up >= 30 * iaw_logn_up
    @test epw_logn_dn >= 30 * iaw_logn_dn
    @test epw_td_up <= 1e-3
    @test epw_td_dn <= 1e-3
    @test iaw_td_up >= 1e4 * epw_td_up
    @test iaw_td_dn >= 1e4 * epw_td_dn
end

# ------------------------------------------------------------------------------------------
# TC3: the example deck's wiring, no fit
# ------------------------------------------------------------------------------------------

@testset "TC3: the example deck's wiring, no fit" begin
    mktempdir() do dir
        sd_epw = synthesize_channel(P, :epw)
        sd_iaw = synthesize_channel(P, :iaw)
        path_epw = joinpath(dir, "epw_streak.jld2")
        path_iaw = joinpath(dir, "iaw_streak.jld2")
        write_streak(path_epw, sd_epw; source="test", probe=P.probe, collection=P.collection)
        write_streak(path_iaw, sd_iaw; source="test", probe=P.probe, collection=P.collection)

        text = read(joinpath(@__DIR__, "..", "..", "examples", "two_channel.toml"), String)
        text = replace(text, "runs/two_channel_epw_streak.jld2" => replace(path_epw, "\\" => "/"))
        text = replace(text, "runs/two_channel_iaw_streak.jld2" => replace(path_iaw, "\\" => "/"))
        text = replace(text, "nbins = 32" => "nbins = 8")

        deck_path = joinpath(dir, "deck.toml")
        write(deck_path, text)
        deck = read_deck(deck_path)

        problem, _, _ = OTIS.build_problem(deck)
        @test problem isa JointFitProblem
        @test OTIS.channel_names(problem) == ["epw", "iaw"]
        @test problem.weights == [1.0, 1.0]

        avg_epw = average_regions(sd_epw, P.regions)
        avg_iaw = average_regions(sd_iaw, P.regions; t_shift_ns=P.iaw.t_shift_ns)
        @test problem.channels[2].data.meta["regions_window_counts"] ==
              avg_iaw.meta["regions_window_counts"]

        wc_epw = avg_epw.meta["regions_window_counts"]
        wc_iaw = avg_iaw.meta["regions_window_counts"]
        pooled = (wc_epw .* avg_epw.t .+ wc_iaw .* avg_iaw.t) ./ (wc_epw .+ wc_iaw)
        @test isapprox(problem.pspec.trace.axis, pooled; rtol=1e-14)

        @test isapprox(problem.channels[1].data.lambda, P.epw.lambda; rtol=1e-14)
        @test isapprox(problem.channels[2].data.lambda, P.iaw.lambda; rtol=1e-14)

        text_epw = single_channel_deck_text(text, "epw")
        deck_epw_path = joinpath(dir, "epw_only.toml")
        write(deck_epw_path, text_epw)
        problem_epw, _, _ = OTIS.build_problem(read_deck(deck_epw_path))
        @test problem_epw isa FitProblem
        @test problem_epw.name == "epw"

        text_iaw = single_channel_deck_text(text, "iaw")
        deck_iaw_path = joinpath(dir, "iaw_only.toml")
        write(deck_iaw_path, text_iaw)
        problem_iaw, _, _ = OTIS.build_problem(read_deck(deck_iaw_path))
        @test problem_iaw isa FitProblem
        @test problem_iaw.name == "iaw"
    end
end
