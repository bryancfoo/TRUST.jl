# Step 4.C.1: the L-curve's corner, the maximum-curvature estimate of the global weight over a
# weight scan (Phase 4.C spec D4-D9). LC1-LC10, gate "L-curve corner". The oracle for the corner
# is an analytic Tikhonov L-curve's continuous curvature on a fine grid (spec Section 6.3), a
# different formula on a different grid from the estimator's three-point curvature (8.2).

using Test
using TRUST
using Random: Xoshiro

# ------------------------------------------------------------------------------------------
# The analytic L-curve (spec Section 6.3): min ||A x - b||^2 + Lambda ||x||^2 with singular values
# s_i = 10^(-(i-1)/2), i = 1..8, data coefficients beta_i = s_i + 1e-2 (a unit solution plus a flat
# noise floor), nothing of b outside A's range.
# ------------------------------------------------------------------------------------------

const _LC_S = [10.0^(-(i - 1) / 2) for i in 1:8]
const _LC_BETA = _LC_S .+ 1e-2
_lc_chi2(L) = sum((L * _LC_BETA[i] / (_LC_S[i]^2 + L))^2 for i in 1:8)
_lc_R(L) = sum((_LC_S[i] * _LC_BETA[i] / (_LC_S[i]^2 + L))^2 for i in 1:8)

const _LC_W = 10.0 .^ (-7:-1)
const _LC_CHI2 = [_lc_chi2(L) for L in _LC_W]
const _LC_R = [_lc_R(L) for L in _LC_W]

# The oracle: Hansen and O'Leary's continuous curvature of (log chi2, log R) against u = log
# Lambda, derivatives by centred differences on 2000 points over eight decades; returns the
# Lambda of its largest value and the number of its positive local maxima.
function _lc_oracle()
    u = collect(range(-8 * log(10), 0.0; length=2000))
    h = u[2] - u[1]
    x = [log(_lc_chi2(exp(v))) for v in u]
    y = [log(_lc_R(exp(v))) for v in u]
    kap = fill(-Inf, length(u))
    for k in 2:(length(u)-1)
        xp = (x[k+1] - x[k-1]) / (2h)
        yp = (y[k+1] - y[k-1]) / (2h)
        xpp = (x[k+1] - 2x[k] + x[k-1]) / h^2
        ypp = (y[k+1] - 2y[k] + y[k-1]) / h^2
        kap[k] = 2 * (xp * ypp - xpp * yp) / (xp^2 + yp^2)^1.5
    end
    npos = count(k -> kap[k] > 0 && kap[k] >= kap[k-1] && kap[k] >= kap[k+1], 3:(length(u)-2))
    return exp(u[argmax(kap)]), npos
end

@testset "LC1: refusals" begin
    @test_throws "at least three scan points" lcurve_corner([1e-3, 1e-2], [1.0, 2.0], [2.0, 1.0])
    @test_throws "one entry per scan point" lcurve_corner([1e-3, 1e-2, 1e-1], [1.0, 2.0], [3.0, 2.0, 1.0])
    @test_throws "one entry per scan point" lcurve_corner([1e-3, 1e-2, 1e-1], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0];
        converged=[true, true])
    @test_throws "strictly ascending" lcurve_corner([1e-3, 1e-2, 1e-2], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    @test_throws "strictly ascending" lcurve_corner([1e-2, 1e-3, 1e-1], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    @test_throws "weights must be positive" lcurve_corner([0.0, 1e-2, 1e-1], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    @test_throws "chi2 must be positive" lcurve_corner([1e-3, 1e-2, 1e-1], [1.0, 0.0, 3.0], [3.0, 2.0, 1.0])
    @test_throws "R must be positive" lcurve_corner([1e-3, 1e-2, 1e-1], [1.0, 2.0, 3.0], [3.0, 2.0, 0.0])
    @test_throws "R must be positive" lcurve_corner([1e-3, 1e-2, 1e-1], [1.0, 2.0, 3.0], [3.0, NaN, 1.0])
end

@testset "LC2: the analytic corner is bracketed" begin
    L_oracle, npos = _lc_oracle()
    c = lcurve_corner(_LC_W, _LC_CHI2, _LC_R)
    @info "LC2 the oracle's corner and the estimate" L_oracle npos c.weight c.lower c.upper curvature = string(c.curvature)
    @test npos == 1
    @test c.status === :corner
    @test c.index == 4
    @test c.weight == 1e-4
    @test c.lower <= L_oracle <= c.upper
    @test c.index == argmin(abs.(log10.(_LC_W) .- log10(L_oracle)))
    @test !c.at_edge
end

@testset "LC3: the sign -- a mirrored curve negates every curvature and moves the estimate" begin
    c = lcurve_corner(_LC_W, _LC_CHI2, _LC_R)
    m = lcurve_corner(_LC_W, _LC_R, _LC_CHI2)   # (log R, log chi2): the mirror image, reversed turning
    @test all(k -> m.curvature[k] === -c.curvature[k], 2:6)
    @test isnan(m.curvature[1]) && isnan(m.curvature[7])
    @test m.index == 2          # the anti-corner, where a sign error in the cross product lands
    @test c.curvature[4] > 0
end

@testset "LC4: a straight line has no corner" begin
    for (chi2, R) in ((fill(2.0, 5), [5.0, 4.0, 3.0, 2.0, 1.0]),            # vertical in (log chi2, log R)
                      ([1.0, 2.0, 4.0, 8.0, 16.0], [1.0, 2.0, 4.0, 8.0, 16.0]))  # slope one: identical logs
        c = lcurve_corner([1.0, 2.0, 3.0, 4.0, 5.0], chi2, R)
        @test all(iszero, c.curvature[2:4])
        @test c.status === :no_corner
        @test c.weight === nothing && c.index === nothing && c.lower === nothing && c.upper === nothing
        @test c.at_edge === false && c.bracket_converged === nothing
        @test c.weights == [1.0, 2.0, 3.0, 4.0, 5.0]
    end
end

@testset "LC5: a tie goes to the smaller weight" begin
    # A staircase with log(4) - log(2) === log(2) (checked here, since the tie needs it): scan
    # points 2 and 4 turn through the same angle over the same side lengths, so their curvatures
    # are bitwise equal.
    @test log(4.0) - log(2.0) === log(2.0)
    c = lcurve_corner([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 2.0, 2.0, 4.0], [4.0, 2.0, 2.0, 1.0, 1.0])
    @test c.curvature[2] === c.curvature[4]
    @test c.curvature[2] > 0 && c.curvature[3] < 0
    @test c.index == 2 && c.weight == 2.0 && c.lower == 1.0 && c.upper == 3.0
end

@testset "LC6: the edge flag" begin
    @test lcurve_corner(_LC_W[1:5], _LC_CHI2[1:5], _LC_R[1:5]).index == 4          # n - 1
    @test lcurve_corner(_LC_W[1:5], _LC_CHI2[1:5], _LC_R[1:5]).at_edge
    @test lcurve_corner(_LC_W[3:5], _LC_CHI2[3:5], _LC_R[3:5]).at_edge              # n = 3
    @test !lcurve_corner(_LC_W, _LC_CHI2, _LC_R).at_edge
end

@testset "LC7: a common rescaling of chi2 or R changes nothing but round-off" begin
    c = lcurve_corner(_LC_W, _LC_CHI2, _LC_R)
    for (a, b) in ((_LC_CHI2[1], _LC_R[1]), (3.7, 0.29), (1e6, 1e-6))
        n = lcurve_corner(_LC_W, _LC_CHI2 ./ a, _LC_R ./ b)
        @test n.index == c.index
        @test isapprox(n.curvature[2:6], c.curvature[2:6]; rtol=1e-12)
    end
end

@testset "LC8: the bracket's convergence" begin
    @test lcurve_corner(_LC_W, _LC_CHI2, _LC_R).bracket_converged === nothing
    @test lcurve_corner(_LC_W, _LC_CHI2, _LC_R; converged=trues(7)).bracket_converged === true
    for k in 3:5
        flags = trues(7)
        flags[k] = false
        @test lcurve_corner(_LC_W, _LC_CHI2, _LC_R; converged=flags).bracket_converged === false
    end
    flags = trues(7)
    flags[[1, 2, 6, 7]] .= false
    @test lcurve_corner(_LC_W, _LC_CHI2, _LC_R; converged=flags).bracket_converged === true
end

# ------------------------------------------------------------------------------------------
# LC9: a real scan, WS's P2 fixture (test_weight_scan.jl's _ws_p2_problem, duplicated: chunks do
# not import across a @safetestset boundary)
# ------------------------------------------------------------------------------------------

function _lc_ws_problem()
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=10)))
    expt = Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact),
        lambda_m=collect(range(262.0, 264.5; length=32)) .* 1e-9,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    pspec = ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
    nt = nbins(pspec.trace)
    theta_truth = flatten(pspec, ["T_D" => collect(range(220.0, 260.0; length=nt)),
        "u_D" => collect(range(-20.0, 20.0; length=nt))])
    data = synthesize(unflatten(working, pspec, theta_truth), GaussianNoise(a=1.0, b=0.02);
        rng=Xoshiro(20260914))
    reg = Regularizer([RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:absolute),
                       RegularizerTerm(pspec; select=["u_D"], operator=:D2, normalization=:absolute)])
    return FitProblem(working, pspec, data; regularizer=reg), theta_truth .+ 0.02 .* (1:length(theta_truth))
end

@testset "LC9: the scan method on a real scan" begin
    problem, start = _lc_ws_problem()
    scan = weight_scan(problem, [1e-4, 1e-3, 1e-2, 1e-1], LBFGS(); start=start)
    c = lcurve_corner(scan)
    @info "LC9 the WS fixture's corner (reported)" c.status c.weight curvature = string(c.curvature) c.bracket_converged
    v = lcurve_corner(scan.weights, scan.chi2, scan.R; converged=Bool[r.converged for r in scan.results])
    @test all(k -> c.curvature[k] === v.curvature[k], 2:3)
    @test c.status === v.status && c.index === v.index && c.bracket_converged === v.bracket_converged
    @test c.weights == scan.weights
    if c.status === :corner
        @test c.weight == scan.weights[c.index]
        @test c.lower == scan.weights[c.index-1] && c.upper == scan.weights[c.index+1]
        @test c.bracket_converged === all(r.converged for r in scan.results[(c.index-1):(c.index+1)])
    else
        @test c.status === :no_corner
    end
end

@testset "LC10: the estimate is a report; the scan still chooses nothing" begin
    @test fieldnames(WeightScanResult) == (:weights, :chi2, :R, :chi2_normalized, :R_normalized,
        :chi2_unreg, :R_unreg, :rss, :rss_normalized, :rss_unreg, :results, :unregularized)
    names_lower = lowercase.(String.(fieldnames(LCurveCorner)))
    for bad in ("opt", "best", "chosen", "argmin", "recommend", "select")
        @test !any(n -> occursin(bad, n), names_lower)
    end
    srcdir = joinpath(@__DIR__, "..", "..", "src")
    callers = String[]
    for (root, _, files) in walkdir(srcdir), f in files
        endswith(f, ".jl") || continue
        f == "lcurve_corner.jl" && continue
        occursin("lcurve_corner(", read(joinpath(root, f), String)) && push!(callers, f)
    end
    @test isempty(callers)
end
