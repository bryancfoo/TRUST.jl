# Step 3.C.1: what stops a fit. Spec dev/phase3C_instructions.md Section 4, D13-D16, D20.
#
# A DIAGNOSIS, not a fix: nothing here changes an optimizer's default, a criterion or a cap. It
# adds no src/ code of its own -- only calls the g_trace hook, FitResult.termination and
# _gauss_newton_diagonal that Step 3.C.1 added to src/inversion/. The six measurements below are
# Section 4's, in order; each is printed as a table AS IT FINISHES (not buffered to the end), so a
# killed run loses only the measurement in flight. Every part is a function, per ground rule 7 and
# D15, so nothing here measures dynamic dispatch through a non-const global.
#
# Run (from the slow tier's own environment, which already carries CairoMakie and the package):
#   julia --project=test/slow examples/diagnostics/convergence.jl [which] [problem]
#
# `which` selects one measurement ("m1".."m6", default "all"); `problem` restricts a measurement
# to one of "epw_ablation", "iaw_multiflow", "iaw_shock" (default "all"). D15's budget is fifteen
# minutes per command -- run one problem per command for measurements 1, 2 and 5, the ones most
# likely to hit it. The whole script is not expected to finish inside one invocation.
#
# Nothing here writes a result file or a figure; its printed tables are what goes in the report.

using OTIS
using OTIS: _with_weight, _leaf_si_path, jacobian, JACOBIAN_CHUNK, _gauss_newton_diagonal,
    _from_unconstrained, _dx_dy, _count_near_bound
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "problems.jl"))

const _EXAMPLES_DIR = joinpath(@__DIR__, "..")

const CUTDOWN_SPECS = (
    ("epw_ablation", epw_ablation_problem),
    ("iaw_multiflow", iaw_multiflow_problem),
    ("iaw_shock", iaw_shock_problem),
)

const CUTDOWN_LAMBDA = Dict(
    "epw_ablation" => [1.0e-5, 1.0e-3, 1.0e-1],
    "iaw_multiflow" => [1.0e-6, 1.0e-4, 1.0e-3, 1.0e-1],
    "iaw_shock" => [1.0e-5, 1.0e-3, 1.0e-2, 1.0e-1],
)

const G_ABSTOL = 1e-8          # Optim's/LBFGS's shipped default (Section 4 item 1's yardstick)
const RAISED_LBFGS_ITERATIONS = 10_000   # item 2's cap
const RAISED_LM_ITERATIONS = 2_000       # item 2's LM cap
const RAISED_LM_CGLS_MAXITER = 1_000     # item 2's LM inner cap

# ------------------------------------------------------------------------------------------
# Problem builders -- the cut-down problems exactly as test/slow/test_reproduce.jl builds them
# (_fr_deck_for_cutdown's recipe), and the full-resolution multi-flow problem from its own deck
# (its streak already synthesized on disk by an earlier session, runs/iaw_multiflow_streak.jld2).
# ------------------------------------------------------------------------------------------

function _cutdown_deck(name::AbstractString, sd_path::AbstractString)
    deck = read_deck(joinpath(_EXAMPLES_DIR, "$name.toml"))
    sections = deepcopy(deck.sections)
    sections["data"] = merge(sections["data"], Dict{String,Any}("file" => sd_path))
    return OTIS.Deck(deck.deck_version, sections)
end

"""
    _build_cutdown(name, problem_fn; nlambda = 128, nt = 32)

One of the three cut-down (128x32) problems, built the same way test/slow/test_reproduce.jl's
`_fr_deck_for_cutdown` does: `read_deck` on the committed full-resolution deck, `[data].file`
replaced by a freshly synthesized cut-down streak. Returns `(; problem, optimizer, start,
problem_nt)`.
"""
function _build_cutdown(name::AbstractString, problem_fn; nlambda::Int=128, nt::Int=32)
    problem_nt = problem_fn(; nlambda=nlambda, nt=nt)
    expt_truth = build_experiment(problem_nt)
    source = "$(name)_$(nlambda)x$(nt)_diag"
    sd = synthesize(expt_truth, problem_nt.noise; rng=Xoshiro(problem_nt.seed), source=source)
    return mktempdir() do dir
        sd_path = joinpath(dir, "$(source)_streak.jld2")
        write_streak(sd_path, sd; source=source)
        deck = _cutdown_deck(name, sd_path)
        problem, optimizer, start = OTIS.build_problem(deck)
        (; problem, optimizer, start, problem_nt)
    end
end

"""
    _build_fullres_multiflow()

The full-resolution (512x256) multi-flow problem from `examples/iaw_multiflow.toml`, whose
streak (`runs/iaw_multiflow_streak.jld2`) is already on disk. Returns `(problem, optimizer,
start)`.
"""
function _build_fullres_multiflow()
    deck = read_deck(joinpath(_EXAMPLES_DIR, "iaw_multiflow.toml"))
    return OTIS.build_problem(deck)
end

# ------------------------------------------------------------------------------------------
# Shared small helpers
# ------------------------------------------------------------------------------------------

"""
    _theta_bounds(pspec) -> Vector{Tuple{Float64,Float64}}

`(lower, upper)` for every theta index, walking `bin_entries`/`bin_slots` and
`global_entries`/`global_slots` exactly once each -- together the whole of `y`
(`_build_index_map`, `parameters.jl`), the same walk `_count_near_bound` (optimizers.jl) does.
"""
function _theta_bounds(pspec::ParameterSpec)
    bounds = Vector{Tuple{Float64,Float64}}(undef, pspec.ntheta)
    for (j, entries) in enumerate(pspec.bin_entries)
        slots = pspec.bin_slots[j]
        for (k, idx) in enumerate(entries)
            v = pspec.variables[slots[k]]
            bounds[idx] = (v.lower, v.upper)
        end
    end
    for (k, idx) in enumerate(pspec.global_entries)
        v = pspec.variables[pspec.global_slots[k]]
        bounds[idx] = (v.lower, v.upper)
    end
    return bounds
end

"""
    _spread_per_variable(pspec, y, bounds) -> Dict{String,Float64}

`maximum(|dx/dy|) / minimum(|dx/dy|)` per fitted variable name, at `y`.
"""
function _spread_per_variable(pspec::ParameterSpec, y::AbstractVector, bounds)
    nvar = length(pspec.variables)
    groups = [Float64[] for _ in 1:nvar]
    for (j, entries) in enumerate(pspec.bin_entries)
        slots = pspec.bin_slots[j]
        for (k, idx) in enumerate(entries)
            a, b = bounds[idx]
            push!(groups[slots[k]], abs(_dx_dy(_from_unconstrained(y[idx], a, b), a, b)))
        end
    end
    for (k, idx) in enumerate(pspec.global_entries)
        a, b = bounds[idx]
        push!(groups[pspec.global_slots[k]], abs(_dx_dy(_from_unconstrained(y[idx], a, b), a, b)))
    end
    out = Dict{String,Float64}()
    for i in 1:nvar
        isempty(groups[i]) && continue
        out[pspec.variables[i].name] = maximum(groups[i]) / minimum(groups[i])
    end
    return out
end

"""
    _xspace_gradient(bounds, y, gy) -> Vector{Float64}

`gy[i] / (dx/dy)_i` coordinate by coordinate (Section 4 item 1).
"""
function _xspace_gradient(bounds, y::AbstractVector, gy::AbstractVector)
    gx = similar(gy)
    for i in eachindex(gy)
        a, b = bounds[i]
        x = _from_unconstrained(y[i], a, b)
        gx[i] = gy[i] / _dx_dy(x, a, b)
    end
    return gx
end

"""
    _percentile(sorted, p)

`p` in [0, 100], linear index into an already-`sort`ed vector -- cheap and adequate for a
diagnostic table, not a statistics library's own definition.
"""
function _percentile(sorted::AbstractVector, p::Real)
    n = length(sorted)
    n == 0 && return NaN
    k = clamp(round(Int, p / 100 * (n - 1)) + 1, 1, n)
    return sorted[k]
end

"""
    _gradient_report(problem, y) -> NamedTuple

The y-space gradient's inf/2 norms and the x-space gradient's inf norm and 5th/50th/95th
percentiles of |gx|, at `y` (Section 4 item 1).
"""
function _gradient_report(problem::FitProblem, y::AbstractVector)
    bounds = _theta_bounds(problem.pspec)
    g = zeros(problem.pspec.ntheta)
    gradient!(g, problem, y)
    gx = _xspace_gradient(bounds, y, g)
    agx = sort(abs.(gx))
    return (; gnorm_inf=maximum(abs, g), gnorm_2=sqrt(sum(abs2, g)), gx_inf=agx[end],
        gx_p5=_percentile(agx, 5), gx_p50=_percentile(agx, 50), gx_p95=_percentile(agx, 95))
end

"""
    _shown_at(problem, scan, weight, optimizer) -> FitResult

The regularized fit at exactly `weight`, reusing `scan`'s own result if `weight` is one of its
scan points, else one more fit warm-started from `scan`'s nearest neighbour in log10(Lambda) --
`test/slow/test_reproduce.jl`'s own `_fr_shown`.
"""
function _shown_at(problem::FitProblem, scan::WeightScanResult, weight::Float64, optimizer)
    idx = findfirst(==(weight), scan.weights)
    idx !== nothing && return scan.results[idx]
    logs = log10.(scan.weights)
    nearest = scan.results[argmin(abs.(logs .- log10(weight)))]
    return solve(_with_weight(problem, weight), optimizer; start=nearest.theta)
end

_leaf_truth(trace::PlasmaTrace, boundary_key::AbstractString) = begin
    si_path, factor = _leaf_si_path(boundary_key)
    i = findfirst(==(si_path), string.(trace.paths))
    i === nothing && error("_leaf_truth: leaf \"$si_path\" not present in this trace")
    collect(Float64, trace.values[i, :]) ./ factor
end

_total_variation(v::AbstractVector) = sum(abs(v[j+1] - v[j]) for j in 1:(length(v)-1))

_scan_monotonic(scan::WeightScanResult) = (
    all(scan.chi2[k+1] >= scan.chi2[k] - 1e-6 * scan.chi2[1] for k in 1:(length(scan.weights)-1)),
    all(scan.R[k+1] <= scan.R[k] + 1e-6 * scan.R[1] for k in 1:(length(scan.weights)-1)))

function _print_dict(label::AbstractString, d)
    for k in sort(collect(keys(d)))
        @info "$label: entry" key = k value = d[k]
    end
end

"""
    _logged_scan(label, problem, weights, optimizer; truth = nothing) -> WeightScanResult

`weight_scan`'s own chain (unregularized first, each weight warm-started from the previous,
`WeightScanResult`'s own rss/rss_normalized construction), rebuilt here **so that every fit is
logged as it finishes** -- a killed run then loses only the fit in flight, not the whole scan
(D15, ground rule 6). `weight_scan` itself is opaque until it returns, which is why the first
version of this script's measurement 5 lost an entire in-flight shock scan to the fifteen-minute
budget with nothing to show for it.
"""
function _logged_scan(label::AbstractString, problem::FitProblem, weights::Vector{Float64},
    optimizer; truth::Union{Nothing,Dict{String,Vector{Float64}}}=nothing)
    sorted = sort(weights)
    unregularized = solve(_with_weight(problem, 0.0), optimizer; start=flatten(problem.pspec))
    @info "$label: scan point" Lambda = 0.0 label = "unregularized" chi2 = unregularized.chi2 R = unregularized.R status = unregularized.status termination = unregularized.termination iterations = unregularized.iterations
    chi2_unreg = unregularized.chi2
    R_unreg = unregularized.R

    results = Vector{FitResult}(undef, length(sorted))
    current_start = unregularized.theta
    for (k, w) in enumerate(sorted)
        results[k] = solve(_with_weight(problem, w), optimizer; start=current_start)
        current_start = results[k].theta
        r = results[k]
        @info "$label: scan point" Lambda = w chi2_over_chi2_unreg = r.chi2 / chi2_unreg R_over_R_unreg = r.R / R_unreg status = r.status termination = r.termination iterations = r.iterations
    end

    chi2v = [r.chi2 for r in results]
    Rv = [r.R for r in results]
    chi2_normalized = chi2v ./ chi2_unreg
    R_normalized = Rv ./ R_unreg

    rss = Dict{String,Vector{Float64}}()
    rss_normalized = Dict{String,Vector{Float64}}()
    rss_unreg = Dict{String,Float64}()
    if truth !== nothing
        for (name, tvals) in truth
            rss_unreg[name] = sum(abs2, tvals .- unregularized.values[name])
            rss[name] = [sum(abs2, tvals .- r.values[name]) for r in results]
            rss_normalized[name] = rss[name] ./ rss_unreg[name]
        end
    end

    return WeightScanResult(sorted, chi2v, Rv, chi2_normalized, R_normalized, chi2_unreg, R_unreg,
        rss, rss_normalized, rss_unreg, results, unregularized)
end

# ------------------------------------------------------------------------------------------
# Measurement 1: the scale
# ------------------------------------------------------------------------------------------

"""
    measurement1_one(name, problem, optimizer, start)

`‖grad_y O‖` and the x-space gradient at the start, at iteration 100 (recovered by re-running
capped at 100 iterations -- LBFGS is deterministic from a fixed start, so this reproduces the
original run's iterate exactly, CV1's own trick) and at the cap; the change in O over the last
ten iterations before the cap.
"""
function measurement1_one(name::AbstractString, problem::FitProblem, optimizer, start)
    g_trace = []
    t0 = time()
    result = solve(problem, optimizer; start=start, g_trace=g_trace)
    elapsed = time() - t0

    at_start = _gradient_report(problem, start)
    @info "$name measurement1: at start" g_abstol = G_ABSTOL at_start...

    n = length(result.trajectory)
    if n > 101
        r100 = solve(problem, LBFGS(; iterations=100); start=start)
        at100 = _gradient_report(problem, r100.theta)
        @info "$name measurement1: at iteration 100" g_abstol = G_ABSTOL at100...
    else
        @info "$name measurement1: at iteration 100" not_reached = true trajectory_length = n
    end

    at_cap = _gradient_report(problem, result.theta)
    @info "$name measurement1: at the cap" g_abstol = G_ABSTOL iterations = result.iterations status = result.status termination = result.termination at_cap...

    if n > 11
        dO10 = result.trajectory[end-10].objective - result.trajectory[end].objective
        @info "$name measurement1: change in O over the last 10 iterations before the cap" dO10
    else
        @info "$name measurement1: change in O over the last 10 iterations before the cap" not_reached = true trajectory_length = n
    end
    @info "$name measurement1: elapsed" elapsed_s = elapsed
    return nothing
end

# ------------------------------------------------------------------------------------------
# Measurement 2: the caps
# ------------------------------------------------------------------------------------------

function measurement2_lbfgs(name::AbstractString, problem::FitProblem, start;
    iterations::Int=RAISED_LBFGS_ITERATIONS)
    opt = LBFGS(; iterations=iterations)
    t0 = time()
    result = solve(problem, opt; start=start)
    elapsed = time() - t0
    traj = result.trajectory
    o_at(it) = (idx = findfirst(e -> e.iteration == it, traj); idx === nothing ? nothing : traj[idx].objective)
    o1000, o2000, o5000, oend = o_at(1000.0), o_at(2000.0), o_at(5000.0), traj[end].objective
    mean_decrease = length(traj) > 1000 ? (traj[end-1000].objective - oend) / 1000 : nothing
    near_bound_fraction = _count_near_bound(problem.pspec, result.theta) / problem.pspec.ntheta
    @info "$name measurement2: LBFGS to $iterations" converged = result.converged status = result.status termination = result.termination iterations_reached = result.iterations near_bound_fraction O_1000 = o1000 O_2000 = o2000 O_5000 = o5000 O_end = oend mean_decrease_last1000 = mean_decrease elapsed_s = elapsed
    return nothing
end

function measurement2_lm(name::AbstractString, problem::FitProblem, start;
    iterations::Int=RAISED_LM_ITERATIONS, cgls_maxiter::Int=RAISED_LM_CGLS_MAXITER)
    opt = LevenbergMarquardt(; iterations=iterations, cgls_maxiter=cgls_maxiter)
    cgls_iterations = Int[]
    t0 = time()
    result = solve(problem, opt; start=start, cgls_iterations=cgls_iterations)
    elapsed = time() - t0
    meanc = isempty(cgls_iterations) ? nothing : sum(cgls_iterations) / length(cgls_iterations)
    maxc = isempty(cgls_iterations) ? nothing : maximum(cgls_iterations)
    @info "$name measurement2: LM to $iterations (cgls_maxiter=$cgls_maxiter)" converged = result.converged status = result.status iterations_reached = result.iterations mean_cgls_per_outer = meanc max_cgls_per_outer = maxc n_outer_attempted = length(cgls_iterations) elapsed_s = elapsed
    return nothing
end

# ------------------------------------------------------------------------------------------
# Measurement 3: the conditioning
# ------------------------------------------------------------------------------------------

function _conditioning_report(problem::FitProblem, bounds, y::AbstractVector)
    pspec = problem.pspec
    ntheta = pspec.ntheta
    dxdy = [abs(_dx_dy(_from_unconstrained(y[i], bounds[i]...), bounds[i]...)) for i in 1:ntheta]
    spread_all = maximum(dxdy) / minimum(dxdy)
    near_frac = _count_near_bound(pspec, y) / ntheta

    Jac = jacobian(problem, y)
    d = sort(_gauss_newton_diagonal(problem, Jac))
    nzero = count(iszero, d)
    dmin_nonzero = nzero == length(d) ? NaN : d[findfirst(!iszero, d)]
    spread_d = nzero > 0 ? NaN : d[end] / d[1]

    return (; spread_dxdy_all=spread_all, spread_dxdy_per_variable=_spread_per_variable(pspec, y, bounds),
        near_bound_fraction=near_frac, gn_p5=_percentile(d, 5), gn_p50=_percentile(d, 50),
        gn_p95=_percentile(d, 95), gn_spread=spread_d, gn_min_nonzero=dmin_nonzero, gn_zero_count=nzero)
end

function measurement3_one(name::AbstractString, problem::FitProblem, optimizer, start)
    bounds = _theta_bounds(problem.pspec)

    at_start = _conditioning_report(problem, bounds, start)
    @info "$name measurement3: at start" at_start.spread_dxdy_all at_start.near_bound_fraction at_start.gn_p5 at_start.gn_p50 at_start.gn_p95 at_start.gn_spread at_start.gn_zero_count
    _print_dict("$name measurement3: at start, spread(dx/dy) per variable", at_start.spread_dxdy_per_variable)

    result = solve(problem, optimizer; start=start)
    at_cap = _conditioning_report(problem, bounds, result.theta)
    @info "$name measurement3: at the cap" iterations = result.iterations at_cap.spread_dxdy_all at_cap.near_bound_fraction at_cap.gn_p5 at_cap.gn_p50 at_cap.gn_p95 at_cap.gn_spread at_cap.gn_zero_count
    _print_dict("$name measurement3: at the cap, spread(dx/dy) per variable", at_cap.spread_dxdy_per_variable)
    return nothing
end

# ------------------------------------------------------------------------------------------
# Measurement 4: the warm-start chain
# ------------------------------------------------------------------------------------------

"""
    measurement4_epw_exact1(problem, optimizer, start)

The EPW warm-start chain up to Lambda = 1e-5 and 1e-3, exactly as `weight_scan`'s own chain
builds it (unregularized first, each weight warm-started from the previous), capturing
`iterations`, `termination` and `g_trace` at each of the two points that returned ratios of
exactly 1.0 at chunk 14.
"""
function measurement4_epw_exact1(problem::FitProblem, optimizer, start)
    unreg = solve(_with_weight(problem, 0.0), optimizer; start=start)
    @info "EPW exact-1.0 chain: unregularized" chi2 = unreg.chi2 R = unreg.R iterations = unreg.iterations status = unreg.status termination = unreg.termination

    g1 = []
    r1 = solve(_with_weight(problem, 1.0e-5), optimizer; start=unreg.theta, g_trace=g1)
    converged_at_start_1 = !isempty(g1) && g1[1].gnorm_inf <= G_ABSTOL
    @info "EPW exact-1.0 chain: Lambda = 1e-5" chi2_ratio = r1.chi2 / unreg.chi2 R_ratio = r1.R / unreg.R iterations = r1.iterations status = r1.status termination = r1.termination trajectory_length = length(r1.trajectory) g_trace_length = length(g1) start_gnorm_inf = isempty(g1) ? nothing : g1[1].gnorm_inf converged_at_start = converged_at_start_1

    g2 = []
    r2 = solve(_with_weight(problem, 1.0e-3), optimizer; start=r1.theta, g_trace=g2)
    converged_at_start_2 = !isempty(g2) && g2[1].gnorm_inf <= G_ABSTOL
    @info "EPW exact-1.0 chain: Lambda = 1e-3" chi2_ratio = r2.chi2 / unreg.chi2 R_ratio = r2.R / unreg.R iterations = r2.iterations status = r2.status termination = r2.termination trajectory_length = length(r2.trajectory) g_trace_length = length(g2) start_gnorm_inf = isempty(g2) ? nothing : g2[1].gnorm_inf converged_at_start = converged_at_start_2
    return nothing
end

function measurement4_cold_vs_warm(name::AbstractString, problem::FitProblem, optimizer, start,
    lambdas::Vector{Float64})
    scan = weight_scan(problem, lambdas, optimizer; start=start)
    for (i, w) in enumerate(sort(lambdas))
        warm = scan.results[i]
        cold = solve(_with_weight(problem, w), optimizer; start=flatten(problem.pspec))
        @info "$name measurement4: cold vs warm" lambda = w O_warm = warm.objective O_cold = cold.objective chi2_warm = warm.chi2 chi2_cold = cold.chi2 R_warm = warm.R R_cold = cold.R iterations_warm = warm.iterations iterations_cold = cold.iterations status_warm = warm.status status_cold = cold.status
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# Measurement 5: the reference point (raised caps, nothing else changed)
# ------------------------------------------------------------------------------------------

function measurement5_epw(problem::FitProblem, problem_nt, iterations::Int)
    opt = LBFGS(; iterations=iterations)
    truth = Dict{String,Vector{Float64}}(
        "logn" => log10.(_leaf_truth(problem_nt.trace, "n_e_cm3")),
        "T_e" => _leaf_truth(problem_nt.trace, "electrons.e.T_eV"),
        "m_e" => _leaf_truth(problem_nt.trace, "electrons.e.f.m"))
    scan = _logged_scan("epw_ablation measurement5 (FR1/FR2)", problem, CUTDOWN_LAMBDA["epw_ablation"], opt; truth=truth)
    mono_chi2, mono_R = _scan_monotonic(scan)
    shown = _shown_at(problem, scan, 2.0e-4, opt)
    rss = Dict{String,Float64}(v => sum(abs2, truth[v] .- shown.values[v]) / scan.rss_unreg[v] for v in keys(truth))
    avg = sum(values(rss)) / length(rss)
    @info "epw_ablation measurement5 (FR1/FR2): summary" iterations mono_chi2 mono_R
    _print_dict("epw_ablation measurement5 (FR2): RSS normalized at shown Lambda", rss)
    @info "epw_ablation measurement5 (FR2): average RSS" average = avg
    return nothing
end

function measurement5_multiflow(problem::FitProblem, problem_nt, iterations::Int)
    opt = LBFGS(; iterations=iterations)
    nt = nbins(problem.pspec.trace)
    truth = Dict{String,Vector{Float64}}(
        "f_D" => _leaf_truth(problem_nt.trace, "ions.D.fraction"),
        "T_D" => _leaf_truth(problem_nt.trace, "ions.D.T_eV"),
        "T_C" => _leaf_truth(problem_nt.trace, "ions.C.T_eV"),
        "u_D" => zeros(nt),
        "u_C" => _leaf_truth(problem_nt.trace, "ions.C.u_kms"))
    scan = _logged_scan("iaw_multiflow measurement5 (FR4)", problem, CUTDOWN_LAMBDA["iaw_multiflow"], opt; truth=truth)
    mono_chi2, mono_R = _scan_monotonic(scan)
    shown = _shown_at(problem, scan, 5.0e-3, opt)
    rss = Dict{String,Float64}(v => sum(abs2, truth[v] .- shown.values[v]) / scan.rss_unreg[v] for v in keys(truth))
    avg = sum(values(rss)) / length(rss)
    @info "iaw_multiflow measurement5 (FR4): summary" iterations mono_chi2 mono_R
    _print_dict("iaw_multiflow measurement5 (FR4): RSS normalized at shown Lambda", rss)
    @info "iaw_multiflow measurement5 (FR4): average RSS" average = avg
    return nothing
end

function measurement5_shock(problem::FitProblem, iterations::Int)
    opt = LBFGS(; iterations=iterations)
    pspec = problem.pspec
    pre_iv = pspec.intervals[findfirst(iv -> iv.name == "pre", pspec.intervals)]
    post_iv = pspec.intervals[findfirst(iv -> iv.name == "post", pspec.intervals)]
    nt = nbins(pspec.trace)

    scan = _logged_scan("iaw_shock measurement5 (FR5/FR6)", problem, CUTDOWN_LAMBDA["iaw_shock"], opt)
    @info "iaw_shock measurement5 (FR5/FR6): summary" iterations

    idx_b = findfirst(==(1.0e-1), scan.weights)
    reg = scan.results[idx_b]
    unreg = scan.unregularized
    own_bins = Dict("f_D" => 1:nt, "T_D" => 1:nt, "T_C" => 1:nt, "u_pre" => pre_iv.bins, "u_post" => post_iv.bins)
    for v in ("f_D", "T_D", "T_C", "u_pre", "u_post")
        bins = own_bins[v]
        tv_unreg = _total_variation(unreg.values[v][bins])
        tv_reg = _total_variation(reg.values[v][bins])
        @info "iaw_shock measurement5 (FR6 per variable): total variation" variable = v tv_unreg tv_reg smoother = tv_reg < tv_unreg
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# Measurement 6: JACOBIAN_CHUNK's ratio, chain-global cost separated
# ------------------------------------------------------------------------------------------

"""
    _m6_fixture(; nlambda = 128, nt = 32)

A pspec with per-bin entries (T_D, u_D, time-varying) and one kernel-global entry (n_e, not
time-varying, bound to the leaf n_e_cm3 directly) and **no chain-bound effect fitted** -- the
IRFs and Sensitivity are present (so the per-bin convolution cost is real) but none of their own
parameters is a fitted variable, so `chain_globals` is empty and the once-per-assembly chain-stage
cost cannot leak into the per-bin throughput (Section 4 item 6).
"""
function _m6_fixture(; nlambda::Int=128, nt::Int=32)
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=1.0)])
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=nt)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=nlambda)) .* 1e-9
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))

    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    v_n = FittedVariable("n_e"; bounds=(1.0e19, 5.0e19), init=3.0e19)
    pspec = ParameterSpec(expt; variables=[v_T, v_u, v_n],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms", "n_e" => "n_e_cm3"])

    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
    theta = flatten(pspec, ["T_D" => fill(220.0, nt), "u_D" => fill(10.0, nt), "n_e" => 3.2e19])
    image = measure(unflatten(working, pspec, theta))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=ones(size(image)))
    problem = FitProblem(working, pspec, data)
    return problem, theta
end

function _throughput(f, n::Int)
    f()   # warm
    t0 = time()
    for _ in 1:n
        f()
    end
    return (time() - t0) / n
end

function measurement6(; nlambda::Int=128, nt::Int=32, n::Int=20)
    problem, theta = _m6_fixture(; nlambda=nlambda, nt=nt)
    baseline = _throughput(() -> sum(abs2, measure(unflatten(problem.experiment, problem.pspec, theta))), n) / nt
    @info "measurement6: baseline (one plain forward pass, per bin)" nlambda nt n baseline_per_bin_us = baseline * 1e6
    for c in (4, 8, 12)
        jt = _throughput(() -> sum(abs2, jacobian(problem, theta; chunk=c).blocks[1]), n) / nt
        @info "measurement6: JACOBIAN_CHUNK" chunk = c per_bin_us = jt * 1e6 ratio_to_plain_forward_pass = jt / baseline
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------

function main()
    which = isempty(ARGS) ? "all" : lowercase(ARGS[1])
    only_problem = length(ARGS) >= 2 ? ARGS[2] : "all"
    wants(name) = only_problem == "all" || only_problem == name

    if which in ("all", "m1")
        for (name, fn) in CUTDOWN_SPECS
            wants(name) || continue
            b = _build_cutdown(name, fn)
            measurement1_one(name, b.problem, b.optimizer, b.start)
        end
        if wants("iaw_multiflow_fullres")
            problem, optimizer, start = _build_fullres_multiflow()
            measurement1_one("iaw_multiflow_fullres", problem, optimizer, start)
        end
    end
    if which in ("all", "m2-lbfgs")
        for (name, fn) in CUTDOWN_SPECS
            wants(name) || continue
            b = _build_cutdown(name, fn)
            measurement2_lbfgs(name, b.problem, b.start)
        end
    end
    if which in ("all", "m2-lm")
        for (name, fn) in CUTDOWN_SPECS
            wants(name) || continue
            b = _build_cutdown(name, fn)
            measurement2_lm(name, b.problem, b.start)
        end
    end
    if which in ("all", "m3")
        for (name, fn) in CUTDOWN_SPECS
            wants(name) || continue
            b = _build_cutdown(name, fn)
            measurement3_one(name, b.problem, b.optimizer, b.start)
        end
    end
    if which in ("all", "m4")
        if wants("epw_ablation")
            b = _build_cutdown("epw_ablation", epw_ablation_problem)
            measurement4_epw_exact1(b.problem, b.optimizer, b.start)
        end
        for (name, fn) in CUTDOWN_SPECS
            wants(name) || continue
            b = _build_cutdown(name, fn)
            measurement4_cold_vs_warm(name, b.problem, b.optimizer, b.start, CUTDOWN_LAMBDA[name])
        end
    end
    if which in ("all", "m5")
        if wants("epw_ablation")
            b = _build_cutdown("epw_ablation", epw_ablation_problem)
            measurement5_epw(b.problem, b.problem_nt, RAISED_LBFGS_ITERATIONS)
        end
        if wants("iaw_multiflow")
            b = _build_cutdown("iaw_multiflow", iaw_multiflow_problem)
            measurement5_multiflow(b.problem, b.problem_nt, RAISED_LBFGS_ITERATIONS)
        end
        if wants("iaw_shock")
            b = _build_cutdown("iaw_shock", iaw_shock_problem)
            measurement5_shock(b.problem, RAISED_LBFGS_ITERATIONS)
        end
    end
    if which in ("all", "m6")
        measurement6()
    end
    return nothing
end

main()
