using Test
using OTIS
using OTIS: FitJacobian, jvp!, jtvp!, Binding, _kernel_effects, _widen_trace, _make_cache,
    _jacobian_bin!, _chain_order, JACOBIAN_CHUNK, _dx_dy, _from_unconstrained
import ForwardDiff
import Random
using LinearAlgebra: dot

# JT1 to JT15 against dev/phase3_instructions.md Section 5's JT table (spec D17-D21), gated
# "fit jacobian". D46's P2 (the ten-bin problem) and P3 (the oracle problem, every effect on)
# are built here, not imported across the @safetestset boundary.

# ------------------------------------------------------------------------------------------
# shared fixtures -- D46's P2 and P3
# ------------------------------------------------------------------------------------------

function _jt_p2_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

# P2 (D46): ten bins, 0 to 3 ns, 32 wavelengths; SpectralIRF + a TemporalIRF deliberately wide
# against the 0.33 ns bin spacing (so JT6's block-banded coupling is real) + Sensitivity.
function _jt_p2_expt()
    trace = PlasmaTrace(_jt_p2_template(); t_ns=collect(range(0.0, 3.0; length=10)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=32)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
end

function _jt_p2_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

# P3 (D46): four bins, eight wavelengths, every effect on. A SuperGaussian(2.5) electron
# species and two ion species; KSmearing/GradientSmearing/ProbeOffset in the kernel; a
# GaussianKernel SpectralIRF, a deliberately asymmetric TabulatedKernel TemporalIRF, a
# Sensitivity with a three-point response curve, and a RationalBackground. Fitted: one trace
# variable (T_e), one scalar plasma variable (f_D), one kernel global (KSmearing's half_angle),
# and three chain globals (SpectralIRF's sigma, Sensitivity's scale, the rational numerator's
# constant term). The wavelength window is placed on the spectral feature itself (not the wider
# 262-264.5 nm window P1/P2 use) so that the SpectralIRF sigma column is well-conditioned rather
# than round-off-dominated -- Level 3's choice of concrete numbers under D46's qualitative recipe,
# flagged in the chunk 6 report.
function _jt_p3_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0, f=SuperGaussian(2.5))],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=0.7),
              ion(name="C", mass_amu=12.0, charge=6, T_eV=200.0, u_kms=0.0, fraction=0.3)])
end

const _JT_P3_LAMBDA_NM = collect(range(262.9, 263.6; length=8))

function _jt_p3_expt()
    trace = PlasmaTrace(_jt_p3_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = _JT_P3_LAMBDA_NM .* 1e-9
    effects = (
        KSmearing(half_angle_deg=2.0, nodes=3),
        GradientSmearing(_jt_p3_template(); leaf="electrons.e.T_eV", width=30.0, nodes=3),
        ProbeOffset(probe_offset_nm=0.02),
        SpectralIRF(fwhm_nm=0.08),
        TemporalIRF(TabulatedKernel((-1, 0, 2), (0.2, 0.5, 0.3))),
        Sensitivity(scale=1.0e-6, lambda_nm=[262.0, 263.25, 264.5], response=[0.8, 1.0, 0.9]),
        RationalBackground(numerator=(20.0, 2.0), denominator=(0.3,)),
    )
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m, effects=effects)
end

function _jt_p3_pspec(expt::Experiment)
    ksmearing_idx = findfirst(e -> e isa KSmearing, expt.effects)
    spectral_idx = findfirst(e -> e isa SpectralIRF, expt.effects)
    sens_idx = findfirst(e -> e isa Sensitivity, expt.effects)
    rational_idx = findfirst(e -> e isa RationalBackground, expt.effects)

    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0, time_varying=true)
    v_fD = FittedVariable("f_D"; bounds=(0.3, 0.9), init=0.7)
    v_half = FittedVariable("half_angle"; bounds=(0.001, 0.2), init=2.0 * pi / 180)
    v_sig = FittedVariable("sig_spec"; bounds=(1e-11, 2e-10), init=3.5e-11)
    v_scale = FittedVariable("scale"; bounds=(1e-7, 1e-5), init=1e-6)
    v_numer = FittedVariable("numer0"; bounds=(5.0, 50.0), init=20.0)

    return ParameterSpec(expt;
        variables=[v_Te, v_fD, v_half, v_sig, v_scale, v_numer],
        bindings=["T_e" => "electrons.e.T_eV", "f_D" => "ions.D.fraction",
            "half_angle" => (ksmearing_idx, :half_angle),
            "sig_spec" => (spectral_idx, :sigma),
            "scale" => (sens_idx, :scale),
            "numer0" => (rational_idx, :numerator, 1)])
end

# A rebuilt experiment/pspec pair wired the way unflatten requires: expt built from pspec's own
# (rebuilt) trace, exactly test_residuals.jl's _rs_working (D12's precondition).
function _jt_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

function _jt_data(working::Experiment, pspec::ParameterSpec, theta::AbstractVector; mask=nothing)
    truth = unflatten(working, pspec, theta)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    return StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig, mask=mask)
end

_jt_perturb(theta0::AbstractVector, scale::Real=0.05) = theta0 .+ scale .* (1:length(theta0))

function _jt_p2_problem(; scheduler::Symbol=:auto)
    expt = _jt_p2_expt()
    pspec = _jt_p2_pspec(expt)
    working = _jt_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _jt_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data; scheduler=scheduler)
    theta = _jt_perturb(theta0)
    return problem, theta
end

function _jt_p3_problem(; scheduler::Symbol=:auto)
    expt = _jt_p3_expt()
    pspec = _jt_p3_pspec(expt)
    working = _jt_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _jt_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data; scheduler=scheduler)
    theta = _jt_perturb(theta0, 0.02)
    return problem, theta
end

# ------------------------------------------------------------------------------------------
# The dense transcription (D21): built from FitJacobian's FIELDS directly, never by calling
# jvp!/jtvp! -- rule 8.2, so a slip in jvp!/jtvp! cannot hide behind its own assembler.
# Returns (J_ideal, J_dense): J_ideal is the ideal streak's dense N_lambda*N_t x ntheta
# Jacobian (dP/dtheta, before the chain, D17's "the ideal streak is block-diagonal" claim);
# J_dense is the full residual Jacobian after the chain and the sigma divide.
# ------------------------------------------------------------------------------------------

function _jt_dense_transcription(Jac::FitJacobian{FT}, problem::FitProblem, ntheta::Int) where {FT}
    pspec = problem.pspec
    nl, nt, _ = size(Jac.kernel_globals)

    dP = zeros(FT, nl, nt, ntheta)
    for j in 1:nt
        bj = pspec.bin_entries[j]
        for (k, idx) in enumerate(bj)
            dP[:, j, idx] .+= Jac.blocks[j][:, k]
        end
    end
    for (k, idx) in enumerate(Jac.kernel_global_idx)
        dP[:, :, idx] .+= Jac.kernel_globals[:, :, k]
    end

    dM = zeros(FT, nl, nt, ntheta)
    for idx in 1:ntheta
        slice = @view dP[:, :, idx]
        Wl = Jac.Klambda === nothing ? slice : Jac.Klambda * slice
        W = Jac.Kt === nothing ? Wl : Wl * transpose(Jac.Kt)
        dM[:, :, idx] = W .* Jac.response
    end
    for (c, idx) in enumerate(Jac.chain_global_idx)
        dM[:, :, idx] .+= Jac.chain_globals[c]
    end

    sig = problem.sigma
    pixel_index = problem.pixel_index
    J_ideal = zeros(FT, nl * nt, ntheta)
    J_dense = zeros(FT, problem.npixels, ntheta)
    for idx in 1:ntheta
        J_ideal[:, idx] = vec(@view dP[:, :, idx])
        col = vec(@view dM[:, :, idx]) ./ vec(sig)
        J_dense[:, idx] = col[pixel_index]
    end
    return J_ideal, J_dense
end

# ------------------------------------------------------------------------------------------
# JT2: the adjoint identity -- no oracle, cheap, sharp. Written first (spec's own advice).
# ------------------------------------------------------------------------------------------

@testset "JT2: the adjoint identity (P2)" begin
    problem, theta = _jt_p2_problem()
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta
    rng = Random.Xoshiro(20260907)
    for _ in 1:5
        v = randn(rng, ntheta)
        w = randn(rng, problem.npixels)
        out1 = zeros(problem.npixels)
        jvp!(out1, problem, Jac, v)
        out2 = zeros(ntheta)
        jtvp!(out2, problem, Jac, w)
        lhs = dot(out1, w)
        rhs = dot(v, out2)
        @test isapprox(lhs, rhs; rtol=1e-12)
    end
end

# ------------------------------------------------------------------------------------------
# JT3: the adjoint identity on P3's deliberately asymmetric TabulatedKernel -- a symmetric
# kernel would let a swapped transpose pass (Pitfall 5).
# ------------------------------------------------------------------------------------------

@testset "JT3: the adjoint identity on an asymmetric kernel (P3)" begin
    problem, theta = _jt_p3_problem()
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta
    rng = Random.Xoshiro(20260907)
    for _ in 1:5
        v = randn(rng, ntheta)
        w = randn(rng, problem.npixels)
        out1 = zeros(problem.npixels)
        jvp!(out1, problem, Jac, v)
        out2 = zeros(ntheta)
        jtvp!(out2, problem, Jac, w)
        @test isapprox(dot(out1, w), dot(v, out2); rtol=1e-12)
    end
end

# ------------------------------------------------------------------------------------------
# JT1: the brute-force oracle -- P3, every effect on. The phase's most load-bearing row.
# ------------------------------------------------------------------------------------------

@testset "JT1: the brute-force oracle (P3, every effect on)" begin
    problem, theta = _jt_p3_problem()
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta

    J_dense_bruteforce = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
    @test maximum(abs, J_dense_bruteforce) > 0   # D21: a vacuous oracle must never pass

    _, J_dense_fields = _jt_dense_transcription(Jac, problem, ntheta)

    worst = 0.0
    worst_col = 0
    for k in 1:ntheta
        colref = @view J_dense_bruteforce[:, k]
        colours = @view J_dense_fields[:, k]
        floor_k = 1e-10 * maximum(abs, colref)
        relerr = maximum(abs.(colours .- colref) ./ max.(abs.(colref), floor_k))
        if relerr > worst
            worst = relerr
            worst_col = k
        end
        @test relerr <= 1e-10
    end
    # Recorded for the report: JT1's worst error, its column and which binding it belongs to.
    @info "JT1 worst relative error" worst worst_col
end

# ------------------------------------------------------------------------------------------
# JT4: jvp!/jtvp! against the dense transcription, all unit vectors (affordable: P3 is 32 x 9).
# ------------------------------------------------------------------------------------------

@testset "JT4: jvp!/jtvp! against the dense transcription (P3)" begin
    problem, theta = _jt_p3_problem()
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta
    _, J_dense = _jt_dense_transcription(Jac, problem, ntheta)

    for k in 1:ntheta
        v = zeros(ntheta)
        v[k] = 1.0
        out = zeros(problem.npixels)
        jvp!(out, problem, Jac, v)
        @test isapprox(out, J_dense[:, k]; rtol=1e-12, atol=1e-12 * maximum(abs, J_dense[:, k]))
    end
    for k in 1:problem.npixels
        w = zeros(problem.npixels)
        w[k] = 1.0
        out = zeros(ntheta)
        jtvp!(out, problem, Jac, w)
        @test isapprox(out, J_dense[k, :]; rtol=1e-12, atol=1e-12 * maximum(abs, J_dense[k, :]))
    end
end

# ------------------------------------------------------------------------------------------
# JT5: the block structure is real (P2) -- the ideal streak is exactly block-diagonal.
# ------------------------------------------------------------------------------------------

@testset "JT5: the block structure is real (P2)" begin
    problem, theta = _jt_p2_problem()
    Jac = jacobian(problem, theta)
    pspec = problem.pspec
    ntheta = pspec.ntheta
    J_ideal, _ = _jt_dense_transcription(Jac, problem, ntheta)
    nl, nt = size(Jac.kernel_globals)[1:2]

    for j in 1:nt
        @test size(Jac.blocks[j], 2) == length(pspec.bin_entries[j])
    end

    J_ideal_3 = reshape(J_ideal, nl, nt, ntheta)
    for j in 1:nt
        for idx in pspec.bin_entries[j]
            for jp in 1:nt
                jp == j && continue
                @test all(iszero, @view J_ideal_3[:, jp, idx])
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# JT6: J^T J is block-BANDED, not block-diagonal (P2's TemporalIRF is deliberately wide).
# ------------------------------------------------------------------------------------------

@testset "JT6: J^T J is block-banded, not block-diagonal (P2)" begin
    problem, theta = _jt_p2_problem()
    Jac = jacobian(problem, theta)
    pspec = problem.pspec
    ntheta = pspec.ntheta
    _, J_dense = _jt_dense_transcription(Jac, problem, ntheta)
    JtJ = transpose(J_dense) * J_dense

    # bin 1's entries (T_D, u_D at j=1) against bin 2's entries (j=2): off-diagonal coupling
    idx1 = pspec.bin_entries[1]
    idx2 = pspec.bin_entries[2]
    diag_scale = maximum(abs, [JtJ[i, i] for i in 1:ntheta])
    coupling = maximum(abs, JtJ[idx1, idx2])
    @test coupling > 1e-8 * diag_scale
end

# ------------------------------------------------------------------------------------------
# JT7: a global entry is seeded in every bin -- zeroing one bin's contribution to a kernel
# global's slice gives the wrong answer; the full assembly (already proven by JT1) does not.
# ------------------------------------------------------------------------------------------

@testset "JT7: a kernel global is seeded in every bin (P3)" begin
    problem, theta = _jt_p3_problem()
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta
    J_dense_bruteforce = ForwardDiff.jacobian(th -> residuals(problem, th), theta)

    @test !isempty(Jac.kernel_global_idx)
    k = 1   # the first kernel global (f_D or half_angle)
    theta_idx = Jac.kernel_global_idx[k]
    full_col = zeros(problem.npixels)
    v = zeros(ntheta); v[theta_idx] = 1.0
    jvp!(full_col, problem, Jac, v)
    oracle_col = J_dense_bruteforce[:, theta_idx]
    @test isapprox(full_col, oracle_col; rtol=1e-10, atol=1e-10 * maximum(abs, oracle_col))

    # a broken assembler that used only bin 1's contribution to this kernel global
    broken = Matrix{Float64}(undef, size(Jac.kernel_globals, 1), size(Jac.kernel_globals, 2))
    fill!(broken, 0.0)
    broken[:, 1] .= Jac.kernel_globals[:, 1, k]   # only bin 1's slice, not the whole N_lambda x N_t
    pixel_index = problem.pixel_index
    sig = problem.sigma
    broken_full = broken .* Jac.response  # (missing the K_lambda/K_t chain, deliberately -- just to get a wrong, nonzero, comparably-scaled column)
    # heuristically compare: the broken column must NOT match the oracle to JT1's tolerance
    broken_col = vec(broken_full)[pixel_index] ./ vec(sig)[pixel_index]
    relerr_broken = maximum(abs.(broken_col .- oracle_col) ./ max.(abs.(oracle_col), 1e-10 * maximum(abs, oracle_col)))
    @test relerr_broken > 1e-6
end

# ------------------------------------------------------------------------------------------
# JT8: a variable bound to both a kernel global and a chain global -- its column is the sum.
# ------------------------------------------------------------------------------------------

@testset "JT8: a variable that is both a kernel and a chain global (P3 variant)" begin
    trace = PlasmaTrace(_jt_p3_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = _JT_P3_LAMBDA_NM .* 1e-9
    effects = (
        KSmearing(half_angle_deg=2.0, nodes=3),
        SpectralIRF(fwhm_nm=0.08),
        TemporalIRF(TabulatedKernel((-1, 0, 2), (0.2, 0.5, 0.3))),
        Sensitivity(scale=1.0e-6, lambda_nm=[262.0, 263.25, 264.5], response=[0.8, 1.0, 0.9]),
    )
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m, effects=effects)
    ksmearing_idx = findfirst(e -> e isa KSmearing, expt.effects)
    sens_idx = findfirst(e -> e isa Sensitivity, expt.effects)

    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0, time_varying=true)
    v_both = FittedVariable("both"; bounds=(1e-7, 1e-5), init=1e-6)
    pspec = ParameterSpec(expt; variables=[v_Te, v_both],
        bindings=["T_e" => "electrons.e.T_eV",
            "both" => (ksmearing_idx, :half_angle),
            "both" => (sens_idx, :scale)])

    working = _jt_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _jt_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)
    theta = _jt_perturb(theta0, 0.02)

    Jac = jacobian(problem, theta)
    @test length(Jac.kernel_global_idx) == 1
    @test length(Jac.chain_global_idx) == 1
    @test Jac.kernel_global_idx[1] == Jac.chain_global_idx[1]

    ntheta = pspec.ntheta
    J_dense_bruteforce = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
    idx = Jac.kernel_global_idx[1]
    v = zeros(ntheta); v[idx] = 1.0
    out = zeros(problem.npixels)
    jvp!(out, problem, Jac, v)
    oracle = J_dense_bruteforce[:, idx]
    @test isapprox(out, oracle; rtol=1e-10, atol=1e-10 * maximum(abs, oracle))
end

# ------------------------------------------------------------------------------------------
# JT9: gradient! against central finite differences (P2 and P3).
# ------------------------------------------------------------------------------------------

@testset "JT9: gradient! against finite differences (P2, P3)" begin
    for (problem, theta) in (_jt_p2_problem(), _jt_p3_problem())
        ntheta = problem.pspec.ntheta
        g = zeros(ntheta)
        gradient!(g, problem, theta)
        h = 1e-6
        fd = zeros(ntheta)
        for k in 1:ntheta
            thp = copy(theta); thp[k] += h * max(abs(theta[k]), 1.0)
            thm = copy(theta); thm[k] -= h * max(abs(theta[k]), 1.0)
            step = thp[k] - thm[k]
            fd[k] = (chi2(problem, thp) - chi2(problem, thm)) / step
        end
        floor_k = 1e-6 * maximum(abs, fd)
        for k in 1:ntheta
            @test isapprox(g[k], fd[k]; rtol=1e-6, atol=max(1e-6, floor_k))
        end
    end
end

# ------------------------------------------------------------------------------------------
# JT10: gradient! does not form a dense J (P2).
# ------------------------------------------------------------------------------------------

@testset "JT10: gradient! does not form a dense J (P2)" begin
    problem, theta = _jt_p2_problem()
    ntheta = problem.pspec.ntheta
    nlambda, nt = size(problem.data.image)
    g = zeros(ntheta)
    gradient!(g, problem, theta)   # warm
    bytes = @allocated gradient!(g, problem, theta)
    n_global = 0   # P2 has no kernel/chain globals
    bound = 10 * (problem.npixels + nlambda * nt * n_global) * 8
    dense_J_bytes = problem.npixels * ntheta * 8
    @info "JT10 allocation" bytes bound dense_J_bytes
    # As specified: below the loose bound. Measured to be well short of a dense J (D43 item 1's
    # neighbourhood; see the chunk 6 report if this is close to or over the bound), and NOT a
    # dense N_pixels x N_theta materialization -- gradient! never forms J.
    @test bytes < bound
end

# ------------------------------------------------------------------------------------------
# JT11: the per-bin dual pass allocates nothing beyond its cache (P2).
# ------------------------------------------------------------------------------------------

@testset "JT11: the per-bin dual pass allocates zero (P2)" begin
    problem, theta = _jt_p2_problem()
    pspec = problem.pspec
    expt = unflatten(problem.experiment, pspec, theta)
    nvar = length(pspec.variables)
    nlambda = length(problem.experiment.lambda)

    kernel_effects0 = _kernel_effects(expt)
    kernel_bindings = Tuple(Union{Nothing,Binding}[nothing for _ in 1:length(kernel_effects0)])
    kernel_global_idx = Int[]
    kernel_global_of_var = zeros(Int, nvar)
    G = 0

    DFT = ForwardDiff.Dual{OTIS._JacobianTag,Float64,JACOBIAN_CHUNK}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[1]))
    kernel_globals = Array{Float64,3}(undef, nlambda, length(pspec.trace.axis), 0)
    probe0, collection0, lambda0 = expt.probe, expt.collection, expt.lambda
    prefactor_flag = OTIS._prefactor_flag(expt.prefactor)

    _jb!() = _jacobian_bin!(block, kernel_globals, 1, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
        probe0, collection0, lambda0, prefactor_flag, theta)
    _jb!()   # warm
    bytes = @allocated _jb!()
    @test bytes == 0
end

# ------------------------------------------------------------------------------------------
# JT12: the chunk is one constant -- the assembled Jacobian at C = 4, 8, 12 agrees with itself.
# ------------------------------------------------------------------------------------------

@testset "JT12: JACOBIAN_CHUNK does not change the answer (P3)" begin
    problem, theta = _jt_p3_problem()
    ntheta = problem.pspec.ntheta
    Jac4 = jacobian(problem, theta; chunk=4)
    Jac8 = jacobian(problem, theta; chunk=8)
    Jac12 = jacobian(problem, theta; chunk=12)
    _, J4 = _jt_dense_transcription(Jac4, problem, ntheta)
    _, J8 = _jt_dense_transcription(Jac8, problem, ntheta)
    _, J12 = _jt_dense_transcription(Jac12, problem, ntheta)
    @test isapprox(J4, J8; rtol=1e-14)
    @test isapprox(J8, J12; rtol=1e-14)
end

# ------------------------------------------------------------------------------------------
# JT13: threads change nothing -- scheduler = :serial and :auto agree bitwise.
# ------------------------------------------------------------------------------------------

@testset "JT13: scheduler = :serial and :auto agree bitwise (P2)" begin
    problem_s, theta = _jt_p2_problem(; scheduler=:serial)
    problem_a, _ = _jt_p2_problem(; scheduler=:auto)
    Jac_s = jacobian(problem_s, theta)
    Jac_a = jacobian(problem_a, theta)
    for j in eachindex(Jac_s.blocks)
        @test Jac_s.blocks[j] == Jac_a.blocks[j]
    end
    @test Jac_s.kernel_globals == Jac_a.kernel_globals
    @test length(Jac_s.chain_globals) == length(Jac_a.chain_globals)
    for c in eachindex(Jac_s.chain_globals)
        @test Jac_s.chain_globals[c] == Jac_a.chain_globals[c]
    end
end

# ------------------------------------------------------------------------------------------
# JT14: an IRF width's column -- P3 with SpectralIRF's sigma bound, against JT1's oracle.
# ------------------------------------------------------------------------------------------

@testset "JT14: an IRF width's column (P3)" begin
    problem, theta = _jt_p3_problem()
    Jac = jacobian(problem, theta)
    ntheta = problem.pspec.ntheta
    J_dense_bruteforce = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
    sig_idx = findfirst(nm -> nm.name == "sig_spec", problem.pspec.variables)
    @test sig_idx !== nothing
    theta_idx = problem.pspec.offsets[sig_idx]
    @test theta_idx in Jac.chain_global_idx

    v = zeros(ntheta); v[theta_idx] = 1.0
    out = zeros(problem.npixels)
    jvp!(out, problem, Jac, v)
    oracle = J_dense_bruteforce[:, theta_idx]
    @test isapprox(out, oracle; rtol=1e-10, atol=1e-10 * maximum(abs, oracle))
end

# ------------------------------------------------------------------------------------------
# JT15: a TabulatedKernel has nothing to bind.
# ------------------------------------------------------------------------------------------

@testset "JT15: a TabulatedKernel has nothing to bind (P3)" begin
    expt = _jt_p3_expt()
    temporal_idx = findfirst(e -> e isa TemporalIRF, expt.effects)
    v = FittedVariable("bad_sigma"; bounds=(1e-11, 1e-9), init=1e-10)
    @test_throws r"has no fittable parameters" ParameterSpec(expt; variables=[v],
        bindings=["bad_sigma" => (temporal_idx, :sigma)])
end
