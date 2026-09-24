using Test
using OTIS
import ForwardDiff

# RS1 to RS9 against dev/phase3_instructions.md Section 5's RS table (spec D15, D16), gated
# "fit residuals". D46's P2 (the ten-bin problem, its own wide TemporalIRF and all) is built here,
# not imported across the @safetestset boundary.

# ------------------------------------------------------------------------------------------
# shared fixtures -- D46's P2
# ------------------------------------------------------------------------------------------

function _rs_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _rs_wrap(trace::PlasmaTrace; effects=())
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=32)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m, effects=effects)
end

# P2 (D46): ten bins, 0 to 3 ns, 32 wavelengths; SpectralIRF + a TemporalIRF deliberately wide
# against the 0.33 ns bin spacing + Sensitivity. Fitted: T_D and u_D, both time-varying.
function _rs_p2()
    trace = PlasmaTrace(_rs_template(); t_ns=collect(range(0.0, 3.0; length=10)))
    return _rs_wrap(trace; effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4),
        Sensitivity(scale=1.0e-6)))
end

function _rs_pspec_p2(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

# A rebuilt experiment/pspec pair wired the way unflatten requires: expt built from pspec's own
# (rebuilt) trace, exactly test_parameters.jl's _pm_working (D12's precondition).
function _rs_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

# StreakData built from unflatten(working, pspec, theta): measure() is the "truth" image, sigma
# from GaussianNoise's own formula applied to that image (varying across pixels, so RS1's divide
# is actually exercised), no noise added -- this chunk tests bookkeeping, not recovery.
function _rs_data(working::Experiment, pspec::ParameterSpec, theta::AbstractVector; mask=nothing)
    truth = unflatten(working, pspec, theta)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    return StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig, mask=mask)
end

# A deterministic, non-constant perturbation of theta -- no RNG needed since nothing here is
# stochastic (D46 reserves seeds for the optimizer/synthetic-data chunks).
_rs_perturb(theta0::AbstractVector) = theta0 .+ 0.05 .* (1:length(theta0))

# ------------------------------------------------------------------------------------------
# RS1: residuals against the definition
# ------------------------------------------------------------------------------------------

@testset "RS1: residuals against the definition" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rs_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)

    theta = _rs_perturb(theta0)
    r = residuals(problem, theta)

    M = measure(unflatten(working, pspec, theta))
    D = data.image
    sig = data.sigma
    oracle = ((vec(M) .- vec(D)) ./ vec(sig))[vec(problem.mask)]
    @test r == oracle
    @test all(r .=== oracle)
end

# ------------------------------------------------------------------------------------------
# RS2: the pixel order, against a deliberately ragged mask
# ------------------------------------------------------------------------------------------

@testset "RS2: the pixel order" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)

    nl, nt = size(measure(unflatten(working, pspec, theta0)))
    mask = BitMatrix(undef, nl, nt)
    for j in 1:nt, l in 1:nl
        mask[l, j] = (mod(l + 2j, 3) != 0)   # a ragged pattern: a different count per column
    end
    @test any(count(mask[:, j]) != count(mask[:, 1]) for j in 1:nt)   # confirm it is ragged

    data = _rs_data(working, pspec, theta0; mask=mask)
    problem = FitProblem(working, pspec, data)
    theta = _rs_perturb(theta0)
    r = residuals(problem, theta)

    M = measure(unflatten(working, pspec, theta))
    D = data.image
    sig = data.sigma
    expected = Float64[]
    for j in 1:nt, l in 1:nl
        mask[l, j] || continue
        push!(expected, (M[l, j] - D[l, j]) / sig[l, j])
    end
    @test length(r) == length(expected)
    @test all(r .=== expected)
    @test problem.npixels == count(mask)
end

# ------------------------------------------------------------------------------------------
# RS3: chi2
# ------------------------------------------------------------------------------------------

@testset "RS3: chi2" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rs_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)

    theta = _rs_perturb(theta0)
    val = chi2(problem, theta)
    expected = sum(abs2, residuals(problem, theta))
    @test isapprox(val, expected; rtol=1e-15)
end

# ------------------------------------------------------------------------------------------
# RS4: the mask's sense -- true means used
# ------------------------------------------------------------------------------------------

@testset "RS4: the mask's sense" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    nl, nt = size(image)

    data_full = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig)
    problem_full = FitProblem(working, pspec, data_full)

    n_bright = max(1, div(nl * nt, 10))
    order = sortperm(vec(image); rev=true)
    bright = Set(order[1:n_bright])
    mask = reshape([!(i in bright) for i in 1:(nl*nt)], nl, nt)
    data_masked = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig, mask=mask)
    problem_masked = FitProblem(working, pspec, data_masked)

    theta = _rs_perturb(theta0)
    @test chi2(problem_masked, theta) < chi2(problem_full, theta)
    @test problem_masked.npixels == count(mask)
    @test problem_full.npixels == nl * nt
end

# ------------------------------------------------------------------------------------------
# RS5: sigma is frozen and nothing recomputes it
# ------------------------------------------------------------------------------------------

@testset "RS5: sigma is frozen and nothing recomputes it" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)

    # (a) a StreakData without sigma throws D39's fragment naming sigma(model, streak)
    data_nosigma = StreakData(lambda_m=working.lambda, t_s=working.t, image=image)
    @test_throws r"has no sigma; make one with sigma\(model, streak\)" FitProblem(working, pspec, data_nosigma)

    # (b) grep: no call to sigma( in src/inversion/. src/config/ does not exist yet at this
    # chunk (it arrives at Step 3.6a), so it is not part of this chunk's tree to search.
    # The one legitimate textual occurrence of "sigma(" in this directory is the D39 message
    # above (a string literal, not a call), so that one line is excluded by content, not by
    # weakening the pattern.
    src_dir = joinpath(@__DIR__, "..", "..", "src", "inversion")
    found_in = String[]
    for (root, _, files) in walkdir(src_dir)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            for line in eachline(path)
                occursin("make one with sigma(model, streak)", line) && continue
                occursin(r"\bsigma\(", line) && push!(found_in, "$path: $(strip(line))")
            end
        end
    end
    @test isempty(found_in)
end

# ------------------------------------------------------------------------------------------
# RS6: chi2 does not move with the fitted scale (OQ-14, closing A22)
# ------------------------------------------------------------------------------------------

@testset "RS6: chi2 does not move with the fitted scale" begin
    expt = _rs_p2()
    @assert expt.effects[3] isa Sensitivity
    v_scale = FittedVariable("scale"; bounds=(1e-7, 1e-5), init=1e-6)
    pspec = ParameterSpec(expt; variables=[v_scale], bindings=["scale" => (3, :scale)])
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rs_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)
    @test problem.sigma === data.sigma

    theta1 = flatten(pspec, ["scale" => 3e-6])
    residuals(problem, theta1)
    @test problem.sigma === data.sigma
    @test problem.sigma === problem.data.sigma
end

# ------------------------------------------------------------------------------------------
# RS7: a dual theta
# ------------------------------------------------------------------------------------------

@testset "RS7: a dual theta" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rs_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)

    theta_plain = _rs_perturb(theta0)
    theta_dual = ForwardDiff.Dual.(theta_plain, 1.0)
    r_dual = residuals(problem, theta_dual)
    r_plain = residuals(problem, theta_plain)
    @test eltype(r_dual) <: ForwardDiff.Dual
    @test isapprox(ForwardDiff.value.(r_dual), r_plain; rtol=1e-15)
end

# ------------------------------------------------------------------------------------------
# RS8: mismatched axes
# ------------------------------------------------------------------------------------------

@testset "RS8: mismatched axes" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)

    shifted_lambda = working.lambda .+ 1e-9   # shifted by 1 nm (SI: m)
    data = StreakData(lambda_m=shifted_lambda, t_s=working.t, image=image, sigma=sig)
    @test_throws r"disagrees with the experiment's" FitProblem(working, pspec, data)
end

# ------------------------------------------------------------------------------------------
# RS9: an all-false mask
# ------------------------------------------------------------------------------------------

@testset "RS10: a mis-paired experiment/pspec trace is refused" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rs_data(working, pspec, theta0)

    # experiment.trace === pspec.trace holds for `working`, but not for the original `expt`
    # (unflatten's precondition, D12; the carried item from chunk 5's review).
    @test working.trace === pspec.trace
    @test expt.trace !== pspec.trace
    @test_throws r"the experiment's trace is not this ParameterSpec's own rebuilt trace" FitProblem(expt, pspec, data)

    # and the correctly-paired problem does respond to theta (the positive half of the finding)
    problem = FitProblem(working, pspec, data)
    chi2_0 = chi2(problem, theta0)
    chi2_1 = chi2(problem, theta0 .+ 2.0)
    @test chi2_0 < 1e-20
    @test chi2_1 > 1.0
end

@testset "RS9: an all-false mask" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    mask = falses(size(image))

    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig, mask=mask)
    @test_throws r"the mask excludes every pixel" FitProblem(working, pspec, data)
end

# ------------------------------------------------------------------------------------------
# RS11 (Step 4.A.3a, chunk 6, A19): FitProblem's channel, name and allow_empty_mask, and the
# name check (D31).
# ------------------------------------------------------------------------------------------

@testset "RS11: FitProblem's channel, name, allow_empty_mask, and the name check (D31)" begin
    expt = _rs_p2()
    pspec = _rs_pspec_p2(expt)
    working = _rs_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rs_data(working, pspec, theta0)

    # defaults: channel 1, name "main"
    problem = FitProblem(working, pspec, data)
    @test problem.channel == 1
    @test problem.name == "main"

    # channel and name are settable
    problem2 = FitProblem(working, pspec, data; channel=2, name="iaw")
    @test problem2.channel == 2
    @test problem2.name == "iaw"

    # the name check
    @test_throws r"name must match" FitProblem(working, pspec, data; name="Bad-Name")
    @test_throws r"name must match" FitProblem(working, pspec, data; name="1abc")

    # allow_empty_mask is the only way a zero-pixel problem exists
    mask = falses(size(data.image))
    data_masked = StreakData(lambda_m=working.lambda, t_s=working.t, image=data.image,
        sigma=data.sigma, mask=mask)
    @test_throws r"the mask excludes every pixel" FitProblem(working, pspec, data_masked)
    problem_empty = FitProblem(working, pspec, data_masked; allow_empty_mask=true, channel=2, name="b")
    @test problem_empty.npixels == 0
    @test isempty(problem_empty.pixel_index)
end
