using Test
using TRUST
using TRUST: jvp!, jtvp!, _gauss_newton_diagonal, _full_residuals, _kernel_effects, _widen_trace, _make_cache,
    _jacobian_bin!, JACOBIAN_CHUNK, _collect_kernel_bindings
using LinearAlgebra: dot
import Random
const ForwardDiff = TRUST.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

# Step 4.J.2's slow rows (Phase 4.J spec D16, D17), gated "continuous charge (slow)": CZ14 and CZ16 compile
# the brute-force Jacobian, the central differences and the per-bin dual pass of a deuterium-carbon problem
# whose carbon charge is fitted per bin on an interval with a D2 term; CZ15 checks that problem's model
# against a plasma built at each bin's charge; CZ17 fits. The fast rows are test/plasma/test_charge.jl's.
# fit(deck) pins BLAS to one thread; this file restores the count it found, as TS2 does.

include(joinpath(@__DIR__, "..", "fixtures", "charge_fixture.jl"))

const _CZS_NBLAS = pin_blas_threads()
pin_blas_threads(_CZS_NBLAS)

# ------------------------------------------------------------------------------------------
# CZ14: the Jacobian with a fitted charge, against brute-force ForwardDiff and central differences
# (OQ-60 (d); spec D16)
# ------------------------------------------------------------------------------------------

@testset "CZ14: the Jacobian with a fitted charge" begin
    problem, theta = _cz_problem()
    Jac = jacobian(problem, theta)
    J_brute = ForwardDiff.jacobian(th -> _full_residuals(problem, th), theta)
    ntheta = problem.pspec.ntheta
    nres = size(J_brute, 1)
    @test ntheta == 17
    @test nres == problem.npixels + problem.regularizer.nrows
    zc = findfirst(v -> v.name == "ZC", problem.pspec.variables)
    zcols = problem.pspec.offsets[zc] .+ (0:3)
    @test maximum(abs, J_brute[:, zcols]) > 0                  # a vacuous oracle must never pass
    J_ours = zeros(nres, ntheta)
    worst_ad = 0.0
    worst_fd = 0.0
    h = 1e-5
    for k in 1:ntheta
        e = zeros(ntheta); e[k] = 1.0
        c = zeros(nres)
        jvp!(c, problem, Jac, e)
        J_ours[:, k] = c
        ref = @view J_brute[:, k]
        worst_ad = max(worst_ad, maximum(abs.(c .- ref) ./ max.(abs.(ref), 1e-10 * maximum(abs, ref))))
        fd = (_full_residuals(problem, theta .+ h .* e) .- _full_residuals(problem, theta .- h .* e)) ./ (2h)
        worst_fd = max(worst_fd, maximum(abs.(c .- fd) ./ max.(abs.(fd), 1e-6 * maximum(abs, fd))))
    end
    @info "CZ14: worst relative error against ForwardDiff and against central differences (h = 1e-5)" worst_ad worst_fd
    @test worst_ad <= 1e-10
    @test worst_fd <= 1e-6
    # the adjoint identity, JT2's form
    rng = Random.Xoshiro(20261010)
    adj = 0.0
    for _ in 1:5
        v = randn(rng, ntheta)
        w = randn(rng, nres)
        o1 = zeros(nres); jvp!(o1, problem, Jac, v)
        o2 = zeros(ntheta); jtvp!(o2, problem, Jac, w)
        adj = max(adj, abs(dot(o1, w) - dot(v, o2)) / abs(dot(o1, w)))
        @test isapprox(dot(o1, w), dot(v, o2); rtol=1e-12)
    end
    # the Gauss-Newton diagonal, MC6's form
    d = _gauss_newton_diagonal(problem, Jac)
    dense = vec(sum(abs2, J_ours; dims=1))
    @test all(k -> isapprox(d[k], dense[k]; rtol=1e-12), 1:ntheta)
    @info "CZ14: the adjoint identity and the diagonal" adj diag = maximum(abs.(d .- dense) ./ dense)
end

# ------------------------------------------------------------------------------------------
# CZ15: the model at each bin is the plasma built at that bin's charge (OQ-60 (d), (f); spec D16)
# ------------------------------------------------------------------------------------------

@testset "CZ15: the model against a plasma built at each bin's values" begin
    problem, theta = _cz_problem()
    pspec = problem.pspec
    M = measure(unflatten(problem.experiment, pspec, theta))
    worst = 0.0
    for j in 1:6
        Z = j >= 3 ? _cz_x(pspec, theta, "ZC", j) : 5.0       # outside "late", the template's charge
        p = _cz_plasma_at(_cz_x(pspec, theta, "ne", j), _cz_x(pspec, theta, "Ti", j), Z, _cz_x(pspec, theta, "xC", 1))
        Mj = measure(_cz_bin_experiment(PlasmaTrace(p; t_ns=[_CZ_T6[j]])))[:, 1]
        @test all(isapprox.(M[:, j], Mj; rtol=1e-12))
        worst = max(worst, maximum(abs.(M[:, j] .- Mj) ./ abs.(Mj)))
    end
    @info "CZ15: the problem's model against a plasma built at each bin's values" worst
end

# ------------------------------------------------------------------------------------------
# CZ16: the per-bin dual pass with a fitted charge allocates nothing (OQ-60 (d); JT16's form)
# ------------------------------------------------------------------------------------------

# test_jacobian.jl's JT16 harness (GF9's form) on CZ14's problem, at bin j. A function, so that nothing
# the harness closes over is boxed.
function _cz_dual_pass_bytes(j::Int)
    problem, theta = _cz_problem()
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
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[j]))
    kernel_globals = Array{Float64,3}(undef, nlambda, nbins(pspec.trace), G)
    flag = TRUST._prefactor_flag(expt.prefactor)
    _jb!() = _jacobian_bin!(block, kernel_globals, j, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
        expt.probe, expt.collection, expt.lambda, flag, theta, expt.gain)
    _jb!()
    return (@allocated _jb!()), G, expt.gain
end

@testset "CZ16: the dual pass with a fitted charge allocates nothing" begin
    for j in (2, 4)                  # bin 2: the charge without partials; bin 4: the charge seeded
        bytes, G, gain = _cz_dual_pass_bytes(j)
        @test G == 1 && gain === :convective
        @test bytes == 0
    end
end

# ------------------------------------------------------------------------------------------
# CZ17: a smooth non-integer charge history recovered from a noisy IAW streak (OQ-60 (e), (g); spec D17)
# ------------------------------------------------------------------------------------------

function _cz_recovery_fits()
    root = mktempdir()
    clean_dir, noisy_dir = joinpath(root, "clean"), joinpath(root, "noisy")
    mkpath(clean_dir); mkpath(noisy_dir)
    scale = _cz_recovery_streak(clean_dir; noisy=false)
    _cz_recovery_streak(noisy_dir; noisy=true)
    nblas = pin_blas_threads()
    clean = fit(_cz_recovery_deck(clean_dir, scale; charge="\"ZC\"", name="free"))
    free_path = _cz_recovery_deck(noisy_dir, scale; charge="\"ZC\"", name="free")
    free = fit(free_path)
    six = fit(_cz_recovery_deck(noisy_dir, scale; charge="6", name="six"))
    four = fit(_cz_recovery_deck(noisy_dir, scale; charge="4", name="four"))
    pin_blas_threads(nblas)
    return (; clean, free, free_path, six, four, noisy_dir)
end

const _CZ_FITS = _cz_recovery_fits()

@testset "CZ17: the charge history recovered; fixed at 6 and at 4 it fits worse" begin
    r = _CZ_FITS
    t_ns = r.free.pspec.axis .* 1e9
    z_true = _cz_z_true.(t_ns)
    @test length(z_true) == _CZ_NBINS
    e_clean = abs.(r.clean.values["ZC"] .- z_true)
    problem, _, _ = TRUST.build_problem(read_deck(r.free_path))
    pol = polish(problem, r.free; tol=1e-6)
    lap = laplace(problem, pol.result.theta)
    sig = lap.sigma_samp["ZC"]
    e_free = abs.(pol.result.values["ZC"] .- z_true)
    chi2n(res) = res.chi2 / res.npixels
    @info "CZ17: the un-noised fit's largest error in the charge" maximum(e_clean)
    @info "CZ17: per bin, the noisy fit's error over (the un-noised error + 4 sampling sigma)" ratio = e_free ./ (e_clean .+ 4 .* sig)
    @info "CZ17: the sampling sigma of ZC, smallest and largest" extrema(sig)
    @info "CZ17: chi2/n with the charge fitted, fixed at 6, fixed at 4" free = chi2n(r.free) six = chi2n(r.six) four = chi2n(r.four)
    @test maximum(e_clean) <= 0.25                          # loose, chosen not measured (spec D17)
    @test count(isfinite, sig) == _CZ_NBINS                 # no bin near a bound
    @test all(e_free .<= e_clean .+ 4 .* sig)
    @test chi2n(r.six) > chi2n(r.free)
    @test chi2n(r.four) > chi2n(r.free)
    # the result file (OQ-60 (g)): the fitted series and the binding text
    TRUST.jldopen(joinpath(r.noisy_dir, "free", "result.jld2"), "r") do f
        @test f["variables/ZC/values"] == r.free.values["ZC"]
        @test occursin("ions.C.charge", f["deck"])
    end
end

pin_blas_threads(_CZS_NBLAS)
