using Test
using OTIS
using OTIS: _kernel_effects, _widen_trace, _make_cache, _jacobian_bin!, JACOBIAN_CHUNK,
    _collect_kernel_bindings, jvp!
const ForwardDiff = OTIS.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

# Step 4.F.2's slow rows (Phase 4.F spec D17, D24), gated "pixel integration (slow)": PI6 and PI8,
# which compile the KB chain's dual pass with a PixelIntegration in it, too slow for the phase's
# share of the fast suite. The fast rows are test/instrument/test_pixel_integration.jl's.

include(joinpath(@__DIR__, "..", "fixtures", "sampling_fixture.jl"))

# ------------------------------------------------------------------------------------------
# PI6: the dual pass -- zero allocations with every kernel-stage parameter bound, and the
# Jacobian against brute-force ForwardDiff (test_jacobian.jl's JT16 and JT17 forms)
# ------------------------------------------------------------------------------------------

@testset "PI6: the dual pass with PixelIntegration in the chain" begin
    problem, theta = _pi_kb_problem(4)
    pspec = problem.pspec
    expt = unflatten(problem.experiment, pspec, theta)
    nvar = length(pspec.variables)
    nlambda = length(problem.experiment.lambda)
    kernel_effects0 = _kernel_effects(expt)
    @test kernel_effects0[end] === PixelIntegration(subsamples=4)
    kernel_bindings = _collect_kernel_bindings(pspec, expt, length(kernel_effects0))
    kernel_global_var = [i for i in 1:nvar if pspec.variables[i].vary && !pspec.variables[i].time_varying]
    G = length(kernel_global_var)
    @test G == 4
    kernel_global_idx = [pspec.offsets[i] for i in kernel_global_var]
    kernel_global_of_var = zeros(Int, nvar)
    for (gk, i) in enumerate(kernel_global_var)
        kernel_global_of_var[i] = gk
    end
    DFT = ForwardDiff.Dual{OTIS._JacobianTag,Float64,JACOBIAN_CHUNK}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[1]))
    kernel_globals = Array{Float64,3}(undef, nlambda, length(pspec.trace.axis), G)
    flag = OTIS._prefactor_flag(expt.prefactor)
    _jb!() = _jacobian_bin!(block, kernel_globals, 1, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
        expt.probe, expt.collection, expt.lambda, flag, theta)
    _jb!()
    @test (@allocated _jb!()) == 0

    # each column against brute-force ForwardDiff of the residuals (test_jacobian.jl's JT17 form)
    Jac = jacobian(problem, theta)
    J_brute = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
    ntheta = pspec.ntheta
    worst = 0.0
    for k in 1:ntheta
        e = zeros(ntheta); e[k] = 1.0
        c = zeros(length(residuals(problem, theta)))
        jvp!(c, problem, Jac, e)
        ref = @view J_brute[:, k]
        floor_k = 1e-10 * maximum(abs, ref)
        worst = max(worst, maximum(abs.(c .- ref) ./ max.(abs.(ref), floor_k)))
    end
    @info "PI6: worst relative error of the Jacobian against ForwardDiff" worst
    @test worst <= 1e-10
end

# ------------------------------------------------------------------------------------------
# PI8: one sub-sample is today's model, bitwise, with every kernel-stage parameter bound
# ------------------------------------------------------------------------------------------

@testset "PI8: one sub-sample is today's model, bitwise, on the KB chain" begin
    # every kernel-stage effect in the chain and bound: residuals and Jacobian blocks
    p1, th = _pi_kb_problem(1)
    p0, th0 = _pi_kb_problem(0)
    @test th == th0
    @test all(residuals(p1, th) .=== residuals(p0, th))
    J1 = jacobian(p1, th)
    J0 = jacobian(p0, th)
    @test all(all(a .=== b) for (a, b) in zip(J1.blocks, J0.blocks))
end
