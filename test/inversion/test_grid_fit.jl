# Step G.4.2: the gridded fit path (Phase G.4 spec D8 to D13; OQ-71 item 4). Gate "gridded fit": GF1 to GF9.
# grid_variables makes a GaussianGrid's fitted variables with one held at 0; grid_start reads a start from a gridded
# trace; FitProblem's `chunk` is its Jacobian's ForwardDiff width. The fixture is small so that the gate compiles
# quickly: deuterons on an 11-node grid in the IAW window, sixteen wavelengths, four bins. Oracles share no code with
# what they test (conventions.md 8.2): brute-force ForwardDiff of the full residual vector for the Jacobian, the trace's
# own g rows and `measure` on the projected trace for the start, LinearAlgebra's eigenvalues for JᵀJ.

using Test
using TRUST
using TRUST: jvp!, _full_residuals, _with_weight, _kernel_effects, _collect_kernel_bindings, _widen_trace,
    _make_cache, _jacobian_bin!, _chain_order, _grid_node, JACOBIAN_CHUNK, ATOMIC_MASS_UNIT, EV
using LinearAlgebra: Symmetric, eigvals, isposdef
import ForwardDiff

# ------------------------------------------------------------------------------------------
# the fixture (spec D12)
# ------------------------------------------------------------------------------------------

const _GF_M = 2.014 * ATOMIC_MASS_UNIT
const _GF_GRID = GaussianGrid(; mass_kg=_GF_M, T_ref_eV=300.0, nodes=11, spacing=0.5)
_gf_vth(T_eV) = sqrt(2 * T_eV * EV / _GF_M)

# The log-weights of a Maxwellian of temperature T_eV and flow u (m/s) on the grid, in closed form (OQ-68 item 3), with a
# bump at 1.5 thermal speeds of T_ref so that the truth is not a Maxwellian.
function _gf_g(T_eV, u)
    s0 = sqrt(_gf_vth(300.0)^2 - _GF_GRID.sigma^2)
    s = sqrt(_gf_vth(T_eV)^2 - _GF_GRID.sigma^2)
    return [-((_grid_node(_GF_GRID, n) - u) / s)^2 + (_grid_node(_GF_GRID, n) / s0)^2 + log(s0 / s) +
            0.6 * exp(-((_grid_node(_GF_GRID, n) - 1.5 * _gf_vth(300.0)) / (0.6 * _gf_vth(300.0)))^2) for n in 1:11]
end
const _GF_G = reduce(hcat, [_gf_g(T, u) for (T, u) in zip((240.0, 270.0, 300.0, 280.0), (0.0, 2e4, 4e4, 3e4))])

function _gf_expt()
    tmpl = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, f=_GF_GRID)])
    trace = PlasmaTrace(tmpl; t_ns=[0.0, 0.5, 1.1, 1.6], leaves=("ions.D.f" => _GF_G,))
    return Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact), lambda_m=collect(range(262.6, 263.9; length=16)) .* 1e-9,
        effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
end

# The family, T_e scalar beside it (a kernel global, so that a bin's block is its g and one global), time terms on every
# varied parameter and one velocity term (spec D12). `time_weight = 0` keeps the regularizer's type with no time penalty.
function _gf_problem(; hold=:centre, chunk=JACOBIAN_CHUNK, time_weight=1.0, family=nothing)
    expt = _gf_expt()
    fam = family === nothing ? grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-30.0, 30.0), hold=hold) : family
    pspec = ParameterSpec(expt; variables=vcat(fam.variables, [FittedVariable("T_e"; bounds=(200.0, 800.0), init=400.0)]),
        bindings=vcat(fam.bindings, ["T_e" => "electrons.e.T_eV"]))
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection, lambda_m=expt.lambda,
        effects=expt.effects)
    M = measure(expt)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=M, sigma=sqrt.(M) .+ 1.0)
    varied = [v.name for v in fam.variables if v.vary]
    reg = Regularizer([RegularizerTerm(pspec; select=varied, operator=:D2, normalization=:unit, weight=time_weight),
                       VelocityRegularizerTerm(pspec; select=fam.names, grid=_GF_GRID, weight=_gf_vth(300.0)^5)];
                      weight=1e-2)
    problem = FitProblem(working, pspec, data; regularizer=reg, chunk=chunk)
    return problem, fam, expt
end

# The start from the trace, a little away from it, so that every row of the Jacobian is live.
function _gf_theta(problem, fam, expt)
    start = grid_start(expt.trace, "ions.D", fam)
    theta = flatten(problem.pspec, vcat(start, ["T_e" => 420.0]))
    return theta .+ 0.05 .* sin.(1:length(theta))
end

# J_full densely, by jvp! on unit vectors.
function _gf_dense(problem, Jac)
    n = problem.pspec.ntheta
    out = zeros(problem.npixels + problem.regularizer.nrows)
    J = zeros(length(out), n)
    for k in 1:n
        v = zeros(n)
        v[k] = 1.0
        jvp!(out, problem, Jac, v)
        J[:, k] .= out
    end
    return J
end

# ------------------------------------------------------------------------------------------
# GF1, GF2: grid_variables
# ------------------------------------------------------------------------------------------

@testset "GF1: grid_variables on odd, even and symmetric grids" begin
    fam = grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-30.0, 30.0))
    @test fam.names == ["g$k" for k in 1:11]
    @test fam.bindings == ["g$k" => "ions.D.f.g$k" for k in 1:11]
    @test fam.held == 6 && _grid_node(_GF_GRID, 6) == _GF_GRID.u0
    @test [v.vary for v in fam.variables] == [k != 6 for k in 1:11]
    @test all(v -> v.lower == -30.0 && v.upper == 30.0 && v.init == 0.0 && v.time_varying && v.interval == 0,
              fam.variables)
    @test [v.name for v in fam.variables] == fam.names
    # even: the lower of the two middle nodes; symmetric: the middle parameter, the last
    for (nodes, symmetric, np, held) in ((10, false, 10, 5), (11, true, 6, 6), (10, true, 5, 5), (12, true, 6, 6))
        g = GaussianGrid(; mass_kg=_GF_M, T_ref_eV=300.0, nodes=nodes, spacing=0.5, symmetric=symmetric)
        f = grid_variables(g, "electrons.e"; prefix="w_", bounds=(-5, 2))
        @test length(f.variables) == np && f.held == held && f.names == ["w_$k" for k in 1:np]
        @test last(f.bindings[end]) == "electrons.e.f.g$np"
    end
    g10 = GaussianGrid(; mass_kg=_GF_M, T_ref_eV=300.0, nodes=10, spacing=0.5)
    @test _grid_node(g10, 5) < g10.u0 < _grid_node(g10, 6)
    # hold = k; time_varying and interval passed through
    f = grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-30, 30), hold=2, time_varying=false, interval=1)
    @test f.held == 2 && [v.vary for v in f.variables] == [k != 2 for k in 1:11]
    @test all(v -> !v.time_varying && v.interval == 1, f.variables)
    @test grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-30, 30), hold=11).held == 11
end

@testset "GF2: grid_variables refuses" begin
    gv(; kw...) = grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-30.0, 30.0), kw...)
    @test_throws UndefKeywordError grid_variables(_GF_GRID, "ions.D"; prefix="g")
    @test_throws "is not a species key" grid_variables(_GF_GRID, "D"; prefix="g", bounds=(-30, 30))
    @test_throws "is not a species key" grid_variables(_GF_GRID, "plasma.D"; prefix="g", bounds=(-30, 30))
    @test_throws "does not make a variable name" grid_variables(_GF_GRID, "ions.D"; prefix="1g", bounds=(-30, 30))
    @test_throws "lower < 0 < upper" gv(bounds=(0.0, 30.0))
    @test_throws "lower < 0 < upper" gv(bounds=(-30.0, -1.0))
    @test_throws "lower < 0 < upper" gv(bounds=(-30.0, 1.0, 2.0))
    @test_throws "lower < 0 < upper" gv(bounds=(-Inf, 30.0))
    @test_throws "hold must be :centre or an integer in 1:11" gv(hold=:center)
    @test_throws "hold must be :centre or an integer in 1:11" gv(hold=0)
    @test_throws "hold must be :centre or an integer in 1:11" gv(hold=12)
    @test_throws "hold must be :centre or an integer in 1:11" gv(hold=2.0)
    @test_throws "hold must be :centre or an integer in 1:11" gv(hold=true)
end

# ------------------------------------------------------------------------------------------
# GF3, GF4: grid_start
# ------------------------------------------------------------------------------------------

@testset "GF3: grid_start, the held parameter's shift and the round trip" begin
    problem, fam, expt = _gf_problem()
    start = grid_start(expt.trace, "ions.D", fam)
    @test first.(start) == [fam.names[k] for k in 1:11 if k != fam.held]
    for (q, k) in enumerate(k for k in 1:11 if k != fam.held)
        @test last(start[q]) == _GF_G[k, :] .- _GF_G[fam.held, :]
    end
    # the experiment unflatten gives at the start measures what the projected trace does, to the logit's round-off
    theta = flatten(problem.pspec, vcat(start, ["T_e" => 400.0]))
    M = measure(unflatten(problem.experiment, problem.pspec, theta))
    M0 = measure(expt)
    @test maximum(abs.(M .- M0)) <= 1e-12 * maximum(abs, M0)
    @info "GF3 round trip, max |ΔM| / max M" maximum(abs.(M .- M0)) / maximum(abs, M0)
end

@testset "GF4: grid_start refuses" begin
    _, fam, expt = _gf_problem()
    tight = grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-1.0, 1.0))
    @test_throws "\"g1\" at bin 1 is" grid_start(expt.trace, "ions.D", tight)
    @test_throws "not strictly inside its bounds (-1.0, 1.0)" grid_start(expt.trace, "ions.D", tight)
    @test_throws "does not carry a GaussianGrid" grid_start(expt.trace, "electrons.e", fam)
    @test_throws "names no species of the trace" grid_start(expt.trace, "ions.C", fam)
    @test_throws "is not a species key" grid_start(expt.trace, "D", fam)
    g10 = GaussianGrid(; mass_kg=_GF_M, T_ref_eV=300.0, nodes=10, spacing=0.5)
    @test_throws "a grid of 11 parameters, but the family has 10" grid_start(expt.trace, "ions.D",
        grid_variables(g10, "ions.D"; prefix="g", bounds=(-30, 30)))
    @test_throws "it was made for another species" grid_start(expt.trace, "ions.D",
        grid_variables(_GF_GRID, "ions.X"; prefix="g", bounds=(-30, 30)))
    @test_throws "not time-varying" grid_start(expt.trace, "ions.D",
        grid_variables(_GF_GRID, "ions.D"; prefix="g", bounds=(-30, 30), time_varying=false))
end

# ------------------------------------------------------------------------------------------
# GF5, GF6: the Jacobian, and JᵀJ with and without the held node
# ------------------------------------------------------------------------------------------

@testset "GF5: the Jacobian of a gridded problem against brute-force ForwardDiff, at two chunks" begin
    problem8, fam, expt = _gf_problem()
    problem11, _, _ = _gf_problem(; chunk=11)
    theta = _gf_theta(problem8, fam, expt)
    J8 = _gf_dense(problem8, jacobian(problem8, theta))
    J11 = _gf_dense(problem11, jacobian(problem11, theta))
    Jbf = ForwardDiff.jacobian(th -> _full_residuals(problem8, th), theta)
    @test size(J8) == (16 * 4 + problem8.regularizer.nrows, 4 * 10 + 1)
    @test maximum(abs.(J8 .- Jbf)) <= 1e-12 * maximum(abs, Jbf)
    @test maximum(abs.(J11 .- Jbf)) <= 1e-12 * maximum(abs, Jbf)
    @test J11 == J8
    @info "GF5 max |J − J_ForwardDiff| / max |J|, chunk 8" maximum(abs.(J8 .- Jbf)) / maximum(abs, Jbf)
end

@testset "GF6: JᵀJ positive definite with the centre held; singular with every node fitted" begin
    problem, fam, expt = _gf_problem()
    theta = _gf_theta(problem, fam, expt)
    J = _gf_dense(problem, jacobian(problem, theta))
    @test isposdef(Symmetric(J' * J))
    ev = eigvals(Symmetric(J' * J))
    # every node fitted, built by hand (the one row that does: spec D12)
    allfit = (; variables=[FittedVariable("g$k"; bounds=(-30.0, 30.0), init=0.0, time_varying=true) for k in 1:11],
              bindings=fam.bindings, names=fam.names)
    pall, _, _ = _gf_problem(; family=allfit)
    thall = flatten(pall.pspec, vcat(["g$k" => _GF_G[k, :] for k in 1:11], ["T_e" => 420.0]))
    Jall = _gf_dense(pall, jacobian(pall, thall .+ 0.05 .* sin.(1:length(thall))))
    evall = eigvals(Symmetric(Jall' * Jall))
    @info "GF6 smallest over largest eigenvalue of JᵀJ" held = ev[1] / ev[end] every_node_fitted = evall[1] / evall[end] next_two = evall[2:3] ./ evall[end]
    @test abs(evall[1] / evall[end]) < 1e-8 * (ev[1] / ev[end])
end

# ------------------------------------------------------------------------------------------
# GF7: the same fitted distribution whichever node is held
# ------------------------------------------------------------------------------------------

# With no time penalty (the time term's weight 0) every term is unchanged by a constant added to every g of a bin, so
# holding another node writes the same problem down otherwise (spec D12): the fits differ only by where the optimizer
# stops. With time terms on the varied parameters they would not: the time rows read g_n − g_held(t).
@testset "GF7: the same distribution whichever node is held" begin
    weights = Dict{Any,Matrix{Float64}}()
    objectives = Dict{Any,Float64}()
    for hold in (:centre, 2)
        problem, fam, _ = _gf_problem(; hold=hold, chunk=11, time_weight=0.0)
        r = solve(problem, LevenbergMarquardt(; iterations=200); start=flatten(problem.pspec))
        @test r.converged
        ex = unflatten(problem.experiment, problem.pspec, r.theta)
        weights[hold] = reduce(hcat, [collect(plasma(ex.trace, j).ions[1].f.w) ./ plasma(ex.trace, j).ions[1].f.m0
                                      for j in 1:4])
        objectives[hold] = r.objective
    end
    d = maximum(abs.(weights[:centre] .- weights[2])) / maximum(weights[:centre])
    @info "GF7 centre against node 2 held" max_weight_difference = d objectives
    @test d <= 1e-8
    @test isapprox(objectives[:centre], objectives[2]; rtol=1e-10)
end

# ------------------------------------------------------------------------------------------
# GF8: the per-bin dual pass at the whole block allocates nothing
# ------------------------------------------------------------------------------------------

@testset "GF8: the per-bin dual pass at a wide chunk allocates zero" begin
    problem, fam, expt = _gf_problem(; chunk=11)
    theta = _gf_theta(problem, fam, expt)
    pspec = problem.pspec
    ex = unflatten(problem.experiment, pspec, theta)
    nvar = length(pspec.variables)
    nlambda = length(problem.experiment.lambda)
    kernel_effects0 = _kernel_effects(ex)
    kernel_bindings = _collect_kernel_bindings(pspec, ex, length(kernel_effects0))
    kernel_global_var = [i for i in 1:nvar if pspec.variables[i].vary && !pspec.variables[i].time_varying]
    G = length(kernel_global_var)
    @test G == 1 && length(pspec.bin_entries[1]) == 10
    kernel_global_idx = [pspec.offsets[i] for i in kernel_global_var]
    kernel_global_of_var = zeros(Int, nvar)
    kernel_global_of_var[kernel_global_var[1]] = 1
    DFT = ForwardDiff.Dual{TRUST._JacobianTag,Float64,11}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, 10)
    kernel_globals = Array{Float64,3}(undef, nlambda, 4, G)
    prefactor_flag = TRUST._prefactor_flag(ex.prefactor)
    _jb!() = _jacobian_bin!(block, kernel_globals, 2, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings,
        ex.probe, ex.collection, ex.lambda, prefactor_flag, theta)
    _jb!()   # warm
    @test (@allocated _jb!()) == 0
end

# ------------------------------------------------------------------------------------------
# GF9: the chunk reaches the Jacobian, and the default is the width every problem had
# ------------------------------------------------------------------------------------------

@testset "GF9: FitProblem's chunk" begin
    problem8, fam, expt = _gf_problem()
    problem11, _, _ = _gf_problem(; chunk=11)
    @test problem8.chunk == JACOBIAN_CHUNK == 8 && problem11.chunk == 11
    theta = _gf_theta(problem8, fam, expt)
    Jd = jacobian(problem8, theta)
    Je = jacobian(problem8, theta; chunk=8)
    @test Jd.blocks == Je.blocks && Jd.kernel_globals == Je.kernel_globals && Jd.reg_coeffs == Je.reg_coeffs
    @test jacobian(problem11, theta).blocks == jacobian(problem8, theta; chunk=11).blocks
    # the values are the same at any width (GF5), but the dual caches are not: the width a call used shows in what it
    # allocates (spec D12: chunk 8 about 128.6 kB and chunk 11 about 141.4 kB, within 32 B from run to run)
    alloc(f) = (f(); f(); minimum(@allocated(f()) for _ in 1:3))
    a8, a11 = alloc(() -> jacobian(problem8, theta)), alloc(() -> jacobian(problem11, theta))
    @test abs(a11 - alloc(() -> jacobian(problem8, theta; chunk=11))) <= 1024 && a11 - a8 >= 8192
    @test _with_weight(problem11, 0.5).chunk == 11 && _with_weight(problem11, 0.5).regularizer.weight == 0.5
    # a joint problem: each channel at its own chunk, unless one is given
    e1, e2 = expt, Experiment(trace=expt.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=collect(range(262.9, 263.6; length=8)) .* 1e-9, effects=expt.effects)
    pspec = ParameterSpec((e1, e2); variables=fam.variables, bindings=fam.bindings)
    w(e) = Experiment(; trace=pspec.trace, probe=e.probe, collection=e.collection, lambda_m=e.lambda, effects=e.effects)
    d(e) = (M = measure(e); StreakData(lambda_m=e.lambda, t_s=e.t, image=M, sigma=sqrt.(M) .+ 1.0))
    ch1 = FitProblem(w(e1), pspec, d(e1); channel=1, name="one")
    ch2 = FitProblem(w(e2), pspec, d(e2); channel=2, name="two", chunk=11)
    jp = JointFitProblem((ch1, ch2))
    th = flatten(pspec, grid_start(expt.trace, "ions.D", fam))
    J = jacobian(jp, th)
    @test J.channels[1].blocks == jacobian(ch1, th; chunk=8).blocks
    @test J.channels[2].blocks == jacobian(ch2, th; chunk=11).blocks
    @test jacobian(jp, th; chunk=8).channels[2].blocks == jacobian(ch2, th; chunk=8).blocks
    aj, aj8 = alloc(() -> jacobian(jp, th)), alloc(() -> jacobian(jp, th; chunk=8))
    @info "GF9 bytes allocated by one jacobian" a8 a11 joint = aj joint_at_8 = aj8
    @test aj - aj8 >= 4096
    @test_throws "chunk must be a positive integer; got 0" FitProblem(problem8.experiment, problem8.pspec,
        problem8.data; chunk=0)
end
