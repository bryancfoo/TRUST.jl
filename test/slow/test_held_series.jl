using Test
using TRUST
using TRUST: _full_residuals, jvp!, _kernel_effects, _widen_trace, _make_cache, _jacobian_bin!, JACOBIAN_CHUNK,
    _collect_kernel_bindings
const ForwardDiff = TRUST.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

# Step 4.I.1's slow rows (Phase 4.I spec D13), gated "held series (slow)": HD3 and HD4 compile the
# brute-force Jacobian and the per-bin dual pass of a chain with a held variable, which nothing earlier
# in the fast suite builds, so they are kept out of the phase's share (conventions.md 8.6). The fast
# rows are test/inversion/test_held_series.jl's.

include(joinpath(@__DIR__, "..", "fixtures", "held_series_fixture.jl"))

# ------------------------------------------------------------------------------------------
# HD3: the Jacobian with held variables, against brute-force ForwardDiff
# ------------------------------------------------------------------------------------------

@testset "HD3: the Jacobian with held variables" begin
    problem, theta = _hd_problem()
    Jac = jacobian(problem, theta)
    @test isempty(Jac.chain_globals)               # the held scale has no column (spec D11)
    J_brute = ForwardDiff.jacobian(th -> _full_residuals(problem, th), theta)
    @test maximum(abs, J_brute) > 0
    nrows = size(J_brute, 1)
    worst = 0.0
    for k in 1:problem.pspec.ntheta
        e = zeros(problem.pspec.ntheta); e[k] = 1.0
        c = zeros(nrows)
        jvp!(c, problem, Jac, e)
        ref = @view J_brute[:, k]
        worst = max(worst, maximum(abs.(c .- ref) ./ max.(abs.(ref), 1e-10 * maximum(abs, ref))))
    end
    @info "HD3: worst relative error against ForwardDiff" worst
    @test worst <= 1e-10
end

# ------------------------------------------------------------------------------------------
# HD4: the per-bin dual pass with held variables allocates nothing
# ------------------------------------------------------------------------------------------

# test_jacobian.jl's JT16 harness on HD3's problem, at bin j. A function, so that nothing the harness
# closes over is boxed.
function _hd_dual_pass_bytes(j::Int)
    problem, theta = _hd_problem()
    pspec = problem.pspec
    expt = unflatten(problem.experiment, pspec, theta)
    nvar = length(pspec.variables)
    nlambda = length(expt.lambda)
    kernel_effects0 = _kernel_effects(expt)
    kernel_bindings = _collect_kernel_bindings(pspec, expt, length(kernel_effects0))
    DFT = ForwardDiff.Dual{TRUST._JacobianTag,Float64,JACOBIAN_CHUNK}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[j]))
    kernel_globals = Array{Float64,3}(undef, nlambda, 10, 0)
    kernel_global_idx = Int[]
    kernel_global_of_var = zeros(Int, nvar)
    flag = TRUST._prefactor_flag(expt.prefactor)
    _jb!() = _jacobian_bin!(block, kernel_globals, j, work_trace, vals, col_cache, pspec, nvar, 0,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings, expt.probe,
        expt.collection, expt.lambda, flag, theta, expt.gain)
    _jb!()
    return @allocated _jb!()
end

@testset "HD4: the dual pass with held variables allocates nothing" begin
    @test _hd_dual_pass_bytes(1) == 0              # outside T_D's interval
    @test _hd_dual_pass_bytes(5) == 0              # inside it
end
