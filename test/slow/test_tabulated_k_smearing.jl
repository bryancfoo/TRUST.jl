using Test
using OTIS
using OTIS: _kernel_effects, _widen_trace, _make_cache, _jacobian_bin!, JACOBIAN_CHUNK,
    _collect_kernel_bindings, jvp!
const ForwardDiff = OTIS.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

# Step 4.G.1's slow row (Phase 4.G spec D9), gated "tabulated k-smearing (slow)": TK9, which compiles
# the KB chain's dual pass with the ten rays in it, too slow for the phase's share of the fast suite.
# The fast rows are test/instrument/test_tabulated_k_smearing.jl's.

include(joinpath(@__DIR__, "..", "fixtures", "tabulated_k_smearing_fixture.jl"))

# test_jacobian.jl's JT16 harness on the KB chain with the rays (and a PixelIntegration of n
# sub-samples, none for n = 0): the bytes one warmed per-bin dual pass allocates, the kernel effects
# and the number of kernel globals. A function, so that nothing the harness closes over is boxed (a
# closure over a variable the testset assigns twice allocates on its own).
function _tk_dual_pass_bytes(n::Integer)
    problem, theta = _tk_kb_problem(n)
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
    return (@allocated _jb!()), kernel_effects0, G
end

# ------------------------------------------------------------------------------------------
# TK9: the dual pass -- zero allocations with the rays in the chain and every kernel-stage parameter
# bound, with and without a PixelIntegration after them; the Jacobian against brute-force ForwardDiff
# (test_jacobian.jl's JT16 and JT17 forms)
# ------------------------------------------------------------------------------------------

@testset "TK9: the dual pass with the ten rays in the chain" begin
    for n in (0, 4)
        bytes, kernel_effects0, G = _tk_dual_pass_bytes(n)
        @test kernel_effects0[2] === _tk_rays()
        @test length(kernel_effects0) == (n == 0 ? 4 : 5)
        @test G == 4
        @test bytes == 0
    end

    # each column against brute-force ForwardDiff of the residuals (test_jacobian.jl's JT17 form)
    problem, theta = _tk_kb_problem(0)
    Jac = jacobian(problem, theta)
    J_brute = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
    ntheta = problem.pspec.ntheta
    worst = 0.0
    for k in 1:ntheta
        e = zeros(ntheta); e[k] = 1.0
        c = zeros(length(residuals(problem, theta)))
        jvp!(c, problem, Jac, e)
        ref = @view J_brute[:, k]
        floor_k = 1e-10 * maximum(abs, ref)
        worst = max(worst, maximum(abs.(c .- ref) ./ max.(abs.(ref), floor_k)))
    end
    @info "TK9: worst relative error of the Jacobian against ForwardDiff" worst
    @test worst <= 1e-10
end
