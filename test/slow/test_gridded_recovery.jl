# Step G.4.4b: the gridded recoveries (Phase G.4 spec Section 7 and D28 to D32; OQ-71 item 6, with the user's U2, D28
# and D29). Gate "gridded recovery (slow)": GV1, the EPW problem's electrons; GV2, the bi-Maxwellian problem's protons on
# its IAW camera; GV3, the null test, the EPW problem with Maxwellian electrons. Each is a fit from the Julia API at 8
# bins, 64 wavelengths on the EPW camera and 512 on the IAW camera (the user's, 2026-11-18): the problem's parametric
# trace, its gridded twin (size_grid's defaults, project_to_grid), grid_variables with the centre held,
# grid_maxwellian_start, time terms (:D2, :unit) on every varied g, one VelocityRegularizerTerm on the family,
# Levenberg–Marquardt with its stall test. The streaks are synthesized with
# GaussianNoise matched to each problem's camera, without the camera's point spread (D28, the user's, to revisit after
# the phase); the EPW and null electrons are on a symmetric grid (D29, the user's), the protons are not. Every tolerance
# is deliberately loose (D32): a cut-down fit stops at another step on another machine or BLAS setting
# (conventions.md 8.6, 6.6), so no row pins a digit; each measured figure is printed with @info.

using Test
using TRUST
using TRUST: normal_matrices, _free_entries, _grid_node
using LinearAlgebra: Symmetric, diag
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))
pin_blas_threads(1)   # a projection's last bits depend on BLAS's thread count (conventions.md 6.6)

const _GV_NT, _GV_NL = 8, 64                 # bins and wavelengths (spec D30)
const _GV_NL_IAW = 512                       # the protons' IAW camera: enough wavelengths that the data constrain the
                                             # recovery (spec D30; the user's of 2026-11-18, to revisit after the phase)
const _GV_BOUNDS = (-100.0, 30.0)            # on g, relative to the held centre (spec D19)
const _GV_LOGN_INIT = 19.176                 # examples/epw_ablation.toml's init for logn: n_e = 1.5e19 cm⁻³
const _GV_LM = LevenbergMarquardt(; iterations=300, lm_rtol=1e-5, lm_rtol_window=10)   # spec D31

# The problem's camera noise without its point spread (spec D28): σ² = a + b S̄ is StreakCameraNoise's σ at sigma_px = 0.
_gv_noise(n::StreakCameraNoise) = GaussianNoise(; a=n.read_noise^2, b=n.camera_gain * n.noise_factor)

# A problem at the recoveries' size: its parametric experiment, its streak, the species' grid and the gridded truth.
function _gv_setup(which::Symbol)
    if which === :ions
        p = bimaxwellian_hydrogen_problem(; nlambda_iaw=_GV_NL_IAW, nt_iaw=_GV_NT)
        trace = camera_trace(p, :iaw)
        tw = gridded_twin(trace, [("ions.H", "ions.Hb")]; names=("H",))   # sized on the IAW camera's own trace
        c = p.iaw
        expt = Experiment(trace=trace, probe=p.probe, collection=p.collection, lambda_m=c.lambda, effects=c.effects)
        data = synthesize(expt, _gv_noise(c.noise); rng=Xoshiro(c.seed))
        return (; expt, data, grid=tw.grids[1], gtrace=tw.trace, set=:ions, name="H", key="ions.H")
    end
    p = which === :epw ? epw_ablation_problem(; nlambda=_GV_NL, nt=_GV_NT) : epw_null_problem(; nlambda=_GV_NL, nt=_GV_NT)
    expt = Experiment(trace=p.trace, probe=p.probe, collection=p.collection, lambda_m=p.lambda, effects=p.effects)
    data = synthesize(expt, _gv_noise(p.noise); rng=Xoshiro(p.seed))
    tw = gridded_twin(p.trace, ["electrons.e"]; symmetric=true)          # spec D29
    return (; expt, data, grid=tw.grids[1], gtrace=tw.trace, set=:electrons, name="e", key="electrons.e")
end

_gv_species(trace, S, j) = getproperty(plasma(trace, j), S.set)[S.name]

# The median over bins of the gridded truth's thermal speed √(2V), the lower of the two middle bins: the velocity
# weight's scale (OQ-70), in m/s.
function _gv_vmed(S)
    v = sort([sqrt(2 * velocity_variance(_gv_species(S.gtrace, S, j).f)) for j in 1:nbins(S.gtrace)])
    return v[cld(length(v), 2)]
end

# The fit problem (spec D31): the family, logn per bin for the electrons as examples/epw_ablation.toml fits it, the time
# terms on every varied g, the velocity term on the family, and the deck's own term on logn; chunk the bin's whole
# block. The experiment carries a copy of the gridded truth: unflatten writes into the spec's trace (Pitfall 11).
function _gv_problem(S, wt, wv)
    fam = grid_variables(S.grid, S.key; prefix="g", bounds=_GV_BOUNDS)
    e = S.expt
    gexpt = Experiment(trace=deepcopy(S.gtrace), probe=e.probe, collection=e.collection, lambda_m=e.lambda,
                       effects=e.effects)
    electrons = S.set === :electrons
    pspec = if electrons
        ParameterSpec(gexpt; variables=vcat(fam.variables,
                                            [FittedVariable("logn"; bounds=(18.5, 20.5), init=_GV_LOGN_INIT, time_varying=true)]),
                      dependents=["ne_cm3" => "10^logn"], bindings=vcat(fam.bindings, ["ne_cm3" => "n_e_cm3"]))
    else
        ParameterSpec(gexpt; variables=fam.variables, bindings=fam.bindings)
    end
    working = Experiment(; trace=pspec.trace, probe=e.probe, collection=e.collection, lambda_m=e.lambda, effects=e.effects)
    varied = [v.name for v in fam.variables if v.vary]
    terms = Any[RegularizerTerm(pspec; select=varied, operator=:D2, normalization=:unit, weight=wt),
                VelocityRegularizerTerm(pspec; select=fam.names, grid=S.grid, weight=wv)]
    electrons && push!(terms, RegularizerTerm(pspec; select="logn", operator=:D2, normalization=:relative, weight=2e-4))
    problem = FitProblem(working, pspec, S.data; regularizer=Regularizer(terms; weight=1.0),
                         chunk=maximum(length, pspec.bin_entries))
    start = grid_maxwellian_start(S.gtrace, S.key, fam)
    electrons && push!(start, "logn" => fill(_GV_LOGN_INIT, nbins(S.gtrace)))
    return problem, fam, flatten(pspec, start)
end

# The data fraction of each parameter of the family at each bin (spec Section 7 item 4): the diagonal of
# (JᵀJ)⁻¹ J_dᵀJ_d, both from normal_matrices, over the entries not near a bound; NaN for the held parameter and an entry
# near a bound. Parameters × bins.
function _gv_data_fraction(problem, fam, theta)
    pspec = problem.pspec
    JtJ, JdtJd = normal_matrices(problem, theta)
    free = _free_entries(pspec, theta)
    fr = fill(NaN, pspec.ntheta)
    fr[free] .= diag(Symmetric(JtJ[free, free]) \ JdtJd[free, free])
    nt = nbins(pspec.trace)
    F = fill(NaN, length(fam.names), nt)
    for (i, v) in enumerate(pspec.variables)
        k = findfirst(==(v.name), fam.names)
        (k === nothing || !v.vary) && continue
        F[k, :] .= fr[pspec.offsets[i] .+ (0:nt-1)]
    end
    return F
end

# A fitted (or start) distribution against the gridded truth, bin by bin: the largest |f − f_truth| over the nodes of
# the core (f_truth at least 1 % of its peak) and the nodes whose parameter's data fraction exceeds one half, as a
# fraction of the peak; the moments; n_e against the parametric truth's.
function _gv_compare(S, problem, theta, F)
    ex = unflatten(problem.experiment, problem.pspec, theta)
    Nv = length(S.grid.w0)
    NP = length(parameters(S.grid))
    nodes = [_grid_node(S.grid, n) for n in 1:Nv]
    par(n) = NP == Nv ? n : min(n, Nv + 1 - n)
    out = NamedTuple[]
    for j in 1:nbins(S.gtrace)
        fr, ft = _gv_species(ex.trace, S, j).f, _gv_species(S.gtrace, S, j).f
        a, b = fr.(nodes), ft.(nodes)
        pk = maximum(b)
        sel = [b[n] >= 0.01 * pk || (F !== nothing && !isnan(F[par(n), j]) && F[par(n), j] > 0.5) for n in 1:Nv]
        push!(out, (; err=maximum(abs.(a .- b)[sel]) / pk,
                    T=velocity_variance(fr) / velocity_variance(ft) - 1,
                    u=(mean_velocity(fr) - mean_velocity(ft)) / sqrt(2 * velocity_variance(ft)),
                    ne=plasma(ex.trace, j).n_e / plasma(S.expt.trace, j).n_e - 1,
                    ndet=F === nothing ? 0 : count(x -> !isnan(x) && x > 0.5, F[:, j])))
    end
    return out
end

# A recovery: the problem, its fit from the Maxwellian start, the data fraction, the fit and the start against the truth.
function _gv_recover(which, wt_scale, wv_scale; polish_it=false)
    S = _gv_setup(which)
    wv = wv_scale * _gv_vmed(S)^5
    problem, fam, theta0 = _gv_problem(S, wt_scale, wv)
    t = @elapsed r = solve(problem, _GV_LM; start=theta0)
    polish_it && (t += @elapsed r = polish(problem, r).result)
    F = _gv_data_fraction(problem, fam, r.theta)
    return (; S, problem, fam, r, t, wv, fit=_gv_compare(S, problem, r.theta, F), start=_gv_compare(S, problem, theta0, nothing))
end

# Per-bin figures as one line each (a vector in @info is abbreviated).
_gv_bins(x) = join((xi isa Integer ? string(xi) : string(round(xi; sigdigits=3)) for xi in x), " ")

function _gv_info(label, R)
    f, s = R.fit, R.start
    @info label status = R.r.status iterations = R.r.iterations seconds = round(R.t; digits=1) chi2n = R.r.chi2 / R.problem.npixels R = R.r.R velocity_weight = R.wv core_error = _gv_bins(b.err for b in f) start_core_error = _gv_bins(b.err for b in s) T_rel = _gv_bins(b.T for b in f) u_err = _gv_bins(b.u for b in f) ne_rel = _gv_bins(b.ne for b in f) data_fraction_nodes = _gv_bins(b.ndet for b in f)
end

# ------------------------------------------------------------------------------------------
# GV1: the EPW problem's electrons, on a symmetric grid (spec D29 to D32)
# ------------------------------------------------------------------------------------------

@testset "GV1: the EPW electrons recovered" begin
    R = _gv_recover(:epw, 0.05, 0.1)
    _gv_info("GV1 EPW electrons", R)
    @test maximum(b -> b.err, R.fit) <= 0.08                     # 0.035; its start 0.129
    @test all(b -> b.ndet >= 1, R.fit)                            # the data fix 1 to 3 nodes in every bin
    # beyond the start: where the truth departs from its moments' Maxwellian, the fit is nearer the truth than the start
    shaped = [j for j in eachindex(R.start) if R.start[j].err > 0.05]
    @test length(shaped) >= 4 && all(j -> R.fit[j].err <= 0.7 * R.start[j].err, shaped)   # at most 0.44 of it
    @test maximum(b -> abs(b.ne), R.fit) <= 0.15                  # 0.040; its start 0.76
    @test maximum(b -> abs(b.T), R.fit) <= 0.10                   # 0.038
end

# ------------------------------------------------------------------------------------------
# GV2: the bi-Maxwellian problem's protons on its IAW camera, core and beam on one grid
# ------------------------------------------------------------------------------------------

@testset "GV2: the protons recovered" begin
    R = _gv_recover(:ions, 0.05, 0.01)
    _gv_info("GV2 protons", R)
    @test maximum(b -> b.err, R.fit) <= 0.15                     # 0.091; its start 0.287
    # beyond the start: the fit's worst bin well inside its start's worst, bin 1, where the beam stands clear of the core
    @test maximum(b -> b.err, R.fit) <= 0.6 * maximum(b -> b.err, R.start)   # 0.32 of it
    @test maximum(b -> abs(b.T), R.fit) <= 0.2                    # 0.097
    @test maximum(b -> abs(b.u), R.fit) <= 0.1                    # 0.048 of the bin's thermal speed
end

# ------------------------------------------------------------------------------------------
# GV3: the null test, fitted exactly as GV1
# ------------------------------------------------------------------------------------------

@testset "GV3: the null test" begin
    # the null problem is the EPW problem's with m = 2: its n_e and T_e are the same functions of t
    pe, pn = epw_ablation_problem(; nlambda=_GV_NL, nt=_GV_NT), epw_null_problem(; nlambda=_GV_NL, nt=_GV_NT)
    @test all(j -> plasma(pn.trace, j).n_e == plasma(pe.trace, j).n_e &&
                   plasma(pn.trace, j).electrons["e"].T == plasma(pe.trace, j).electrons["e"].T, 1:_GV_NT)
    @test all(j -> plasma(pn.trace, j).electrons["e"].f.m == 2.0, 1:_GV_NT)
    R = _gv_recover(:null, 0.05, 0.1; polish_it=true)
    _gv_info("GV3 null", R)
    L = laplace(R.problem, R.r.theta)
    h = R.fam.held
    ex = unflatten(R.problem.experiment, R.problem.pspec, R.r.theta)
    z = Float64[]
    for j in 1:_GV_NT
        gf = collect(Float64, values(parameters(_gv_species(ex.trace, R.S, j).f)))
        gt = collect(Float64, values(parameters(_gv_species(R.S.gtrace, R.S, j).f)))
        for k in eachindex(R.fam.names)
            k == h && continue
            s = L.sigma_post[R.fam.names[k]][j]
            isnan(s) || push!(z, ((gf[k] - gf[h]) - (gt[k] - gt[h])) / s)   # both re-centred on the held node
        end
    end
    @info "GV3 the null test, (g_fit − g_truth)/σ_post" max_abs = maximum(abs, z) entries = length(z) near_bound = count(R.r.near_bound["g$(k)"][j] for k in eachindex(R.fam.names) if k != h for j in 1:_GV_NT)
    @test maximum(abs, z) <= 4.0                                  # 1.86
end
