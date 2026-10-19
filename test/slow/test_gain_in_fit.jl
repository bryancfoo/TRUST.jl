using Test
using TRUST
using TRUST: _kernel_effects, _widen_trace, _make_cache, _jacobian_bin!, JACOBIAN_CHUNK,
    _collect_kernel_bindings, jvp!
const ForwardDiff = TRUST.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

# Step 4.H.1's slow rows (Phase 4.H spec D11), gated "gain in fit (slow)": GF8 and GF9 compile the
# brute-force Jacobian and the per-bin dual pass of a chain nothing earlier builds, and GF10 fits, so
# they are kept out of the phase's share of the fast suite (conventions.md 8.6). The fast rows are
# test/instrument/test_gain_in_fit.jl's.

include(joinpath(@__DIR__, "..", "fixtures", "tabulated_k_smearing_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "gain_in_fit_fixture.jl"))

# ------------------------------------------------------------------------------------------
# GF8: the Jacobian with the gain on, against brute-force ForwardDiff of the residuals
# ------------------------------------------------------------------------------------------

@testset "GF8: the Jacobian with the gain on" begin
    problem, theta = _gf_problem(:convective)
    @test problem.experiment.gain === :convective
    Jac = jacobian(problem, theta)
    J_brute = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
    ntheta = problem.pspec.ntheta
    nres = size(J_brute, 1)
    worst = 0.0
    for k in 1:ntheta
        e = zeros(ntheta); e[k] = 1.0
        c = zeros(nres)
        jvp!(c, problem, Jac, e)
        ref = @view J_brute[:, k]
        floor_k = 1e-10 * maximum(abs, ref)
        worst = max(worst, maximum(abs.(c .- ref) ./ max.(abs.(ref), floor_k)))
    end
    # the gain moves this problem's model, so the comparison above sees it
    off, _ = _gf_problem(:none)
    M_on = measure(unflatten(problem.experiment, problem.pspec, theta))
    M_off = measure(unflatten(off.experiment, off.pspec, theta))
    d = maximum(abs.(M_on .- M_off)) / maximum(M_off)
    @info "GF8: worst relative error against ForwardDiff; the gain's effect on the model over its peak" worst d
    @test worst <= 1e-10
    @test d > 1e-3
end

# ------------------------------------------------------------------------------------------
# GF9: the per-bin dual pass with the gain on allocates nothing
# ------------------------------------------------------------------------------------------

# test_jacobian.jl's JT16 harness on GF8's problem: the bytes one warmed per-bin dual pass allocates
# with the experiment's gain passed, and the number of kernel globals. A function, so that nothing the
# harness closes over is boxed (a closure over a variable the testset assigns twice allocates on its own).
function _gf_dual_pass_bytes(gain::Symbol)
    problem, theta = _gf_problem(gain)
    pspec = problem.pspec
    expt = unflatten(problem.experiment, pspec, theta)
    nvar = length(pspec.variables)
    nlambda = length(problem.experiment.lambda)
    kernel_effects0 = _kernel_effects(expt)
    kernel_bindings = _collect_kernel_bindings(pspec, expt, length(kernel_effects0))
    kernel_global_var = [i for i in 1:nvar if pspec.variables[i].vary && !pspec.variables[i].time_varying]
    G = length(kernel_global_var)
    kernel_global_idx = [pspec.offsets[i] for i in kernel_global_var]
    kernel_global_of_var = zeros(Int, nvar)
    for (gk, i) in enumerate(kernel_global_var)
        kernel_global_of_var[i] = gk
    end
    DFT = ForwardDiff.Dual{TRUST._JacobianTag,Float64,JACOBIAN_CHUNK}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[1]))
    kernel_globals = Array{Float64,3}(undef, nlambda, length(pspec.trace.axis), G)
    flag = TRUST._prefactor_flag(expt.prefactor)
    _jb!() = _jacobian_bin!(block, kernel_globals, 1, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
        expt.probe, expt.collection, expt.lambda, flag, theta, expt.gain)
    _jb!()
    return (@allocated _jb!()), expt.gain, G
end

@testset "GF9: the dual pass with the gain on allocates nothing" begin
    for gain in (:convective, :none)
        bytes, g, G = _gf_dual_pass_bytes(gain)
        @test g === gain
        @test G == 2
        @test bytes == 0
    end
end

# ------------------------------------------------------------------------------------------
# GF10: calibration_scan's rebuild of the experiment keeps the gain
# ------------------------------------------------------------------------------------------

# One bin of the IAW window with the gain on and a WavelengthCalibration whose offset the scan moves;
# n_e and T_e fitted. The data are the model at the start, so the start is a polished minimum.
function _gf_scan_problem()
    trace = PlasmaTrace(_gf_template(); t_ns=[1.0])
    expt = Experiment(; trace=trace, probe=_GF_PROBE, collection=_TK_COLL,
        lambda_m=collect(range(262.9, 263.6; length=16)) .* 1e-9,
        effects=(WavelengthCalibration(lambda_offset_nm=0.0), SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6)),
        gain=:convective)
    pspec = ParameterSpec(expt;
        variables=[FittedVariable("ne"; bounds=(1e19, 1e20), init=3e19),
            FittedVariable("Te"; bounds=(100.0, 1000.0), init=400.0)],
        bindings=["ne" => "n_e_cm3", "Te" => "electrons.e.T_eV"])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, gain=expt.gain)
    theta0 = flatten(pspec)
    image = measure(unflatten(working, pspec, theta0))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image,
        sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    return FitProblem(working, pspec, data), theta0
end

@testset "GF10: calibration_scan keeps the gain" begin
    problem, theta0 = _gf_scan_problem()
    scan = calibration_scan(problem, theta0, [-0.01, 0.0, 0.01])
    @test length(scan.results) == 3
    @test all(r -> r.experiments[1].gain === :convective, scan.results)
    @test scan.results[2].theta == theta0
end
