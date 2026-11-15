# Step G.3.1: the regularizer along velocity (Phase G.3 spec D3 to D10; OQ-70). Gate "velocity regularizer":
# VR1 to VR15. A VelocityRegularizerTerm's :D3 row is the third derivative of the cubic through four neighbouring
# entries times the square root of the stencil's middle interval, in m/s. No row needs a fit or a streak, and only
# VR9 a forward model (its oracle is _gauss_newton_diagonal, which takes a FitProblem): the rows and their Jacobian
# are functions of a ParameterSpec and theta. Every "zero" and "equal" below is relative to the row's own scale
# Σ_a |c_a u_a|, or bitwise (spec D10; Pitfall 1): in SI a row here is of order 1e-17, so an absolute tolerance
# would pass a wrong row. The oracles share no code with the term (conventions.md 8.2): the divided difference by
# its recursive definition, the coefficients in product form and the even-node stencil are the test's own.

using Test
using TRUST
using TRUST: _regularizer_raw_rows, _regularizer_residual_rows, _regularizer_closed_form, _regularizer_value,
    _regularizer_gradient!, _gauss_newton_diagonal, _value_table, _grid_node, ELECTRON_MASS, EV
using Random: Xoshiro
import ForwardDiff

# ------------------------------------------------------------------------------------------
# shared fixtures (spec D10)
# ------------------------------------------------------------------------------------------

# Four bins on an uneven axis, test_regularizer.jl's plasma (so that VR9's forward model is the one the regularizer
# gate has compiled), eight wavelengths. Only VR9 evaluates it; the entries' variables are bound to nothing, since the
# regularizer reads theta through the value table alone.
const _VR_T_NS = [0.0, 0.5, 1.2, 1.6]

function _vr_expt()
    tmpl = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=0.7),
              ion(name="C", mass_amu=12.0, charge=6, T_eV=200.0, u_kms=0.0, fraction=0.3)])
    return Experiment(trace=PlasmaTrace(tmpl; t_ns=_VR_T_NS), probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact), lambda_m=collect(range(262.0, 264.5; length=8)) .* 1e-9)
end

# The electron grid of most rows: 500 eV, nodes a quarter of a thermal speed apart (dv = 3.3e6 m/s), centred at
# 150 km/s, ten nodes.
_vr_grid(; kw...) = GaussianGrid(; mass_kg=ELECTRON_MASS, T_ref_eV=500.0, u0_kms=150.0, nodes=10, spacing=0.25, kw...)
_vr_nodes(g) = [_grid_node(g, n) for n in 1:length(g.w0)]

# Uneven electron velocities in m/s, strictly increasing, spaced 0.24 to 1.3 of the grid's dv.
const _VR_UNEVEN = [-4.1, -3.0, -2.2, -1.1, -0.3, 0.6, 1.4, 2.7, 3.5, 4.9] .* 3.3e6

_vr_names(prefix, N) = ["$prefix$n" for n in 1:N]

# A ParameterSpec whose entries are held at the rows of G (one row per name, one column per bin): the term then
# reads exactly the numbers given, so the value rows test the term's arithmetic and not the logit's round trip.
function _vr_held(pairs...; intervals=FitInterval[], extra=FittedVariable[])
    vars = FittedVariable[]
    for (prefix, G) in pairs, n in 1:size(G, 1)
        push!(vars, FittedVariable("$prefix$n"; bounds=(-1e3, 1e3), init=0.0, time_varying=true, held=G[n, :]))
    end
    append!(vars, extra)
    return ParameterSpec(_vr_expt(); variables=vars, intervals=intervals, bindings=Pair{String,String}[])
end

_vr_raw(pspec, term) = _regularizer_raw_rows(Regularizer([term]), pspec, flatten(pspec))

# The test's own coefficients of row i in product form, and the row's own scale Σ_a |c_a u_a| (Pitfall 1).
_vr_coeffs(v, i) = [6 * sqrt(v[i+2] - v[i+1]) / prod(v[a] - v[b] for b in i:(i+3) if b != a) for a in i:(i+3)]
_vr_scale(u, v, i) = sum(abs.(_vr_coeffs(v, i) .* u[i:(i+3)]))

# Newton's divided difference by its recursive definition, of any order, and the rows it gives: the oracle on
# uneven nodes (VR3, VR8, VR10).
_vr_dd(u, v, lo, hi) = lo == hi ? u[lo] : (_vr_dd(u, v, lo + 1, hi) - _vr_dd(u, v, lo, hi - 1)) / (v[hi] - v[lo])
_vr_expected(u, v) = [6 * sqrt(v[i+2] - v[i+1]) * _vr_dd(u, v, i, i + 3) for i in 1:(length(v) - 3)]

# The worst of |row − expected| / scale over every row of every bin, rows stacked bin by bin.
function _vr_worst(raw, G, v; expected=(u, v) -> zeros(length(v) - 3))
    nr = length(v) - 3
    worst = 0.0
    for j in 1:size(G, 2), i in 1:nr
        u = G[:, j]
        worst = max(worst, abs(raw[(j-1)*nr+i] - expected(u, v)[i]) / _vr_scale(u, v, i))
    end
    return worst
end

# The g of a Maxwellian of temperature T (eV) and flow u (m/s) on `grid`, in closed form: the weights
# dv·G(v_n; u, s) with s² = v_th² − σ², as the default's own are with u₀ and v_ref (spec D10; never a projection).
function _vr_maxwellian_g(grid, T_eV, u)
    vth2 = 2 * T_eV * EV / ELECTRON_MASS
    s = sqrt(vth2 - grid.sigma^2)
    return [log(grid.dv * exp(-((vn - u) / s)^2) / (s * sqrt(pi)) / grid.w0[n]) for (n, vn) in enumerate(_vr_nodes(grid))]
end

# ------------------------------------------------------------------------------------------
# VR1 to VR5: the rows
# ------------------------------------------------------------------------------------------

@testset "VR1: the null space: zero rows for a constant, a linear and a quadratic g, even and uneven, and a Maxwellian's g" begin
    worst = 0.0
    for v in (_vr_nodes(_vr_grid()), _VR_UNEVEN)
        x = v ./ 3.3e6
        for g in (fill(2.7, 10), 1.3 .- 0.8 .* x, 0.4 .- 1.1 .* x .+ 0.35 .* x .^ 2)
            G = repeat(g, 1, 4)
            ps = _vr_held(("g", G))
            raw = _vr_raw(ps, VelocityRegularizerTerm(ps; select=_vr_names("g", 10), velocities=v))
            worst = max(worst, _vr_worst(raw, G, v))
        end
    end
    @info "VR1 worst |row| / scale, polynomials of degree 0 to 2" worst
    @test worst <= 1e-13
    # four Maxwellians of other temperatures and flows, one per bin, on the grid's own default (500 eV, 150 km/s)
    grid = _vr_grid()
    G = hcat(_vr_maxwellian_g(grid, 300.0, -4e5), _vr_maxwellian_g(grid, 800.0, 9e5),
             _vr_maxwellian_g(grid, 1500.0, 0.0), _vr_maxwellian_g(grid, 650.0, -1.2e6))
    ps = _vr_held(("g", G))
    raw = _vr_raw(ps, VelocityRegularizerTerm(ps; select=_vr_names("g", 10), grid=grid))
    worst_m = _vr_worst(raw, G, _vr_nodes(grid))
    @info "VR1 worst |row| / scale, Maxwellians' g" worst_m extrema(G)
    @test worst_m <= 1e-13
end

@testset "VR2: the rows unchanged when a constant is added to every entry" begin
    G = 3 .* randn(Xoshiro(20261114), 10, 4)
    worst = 0.0
    for v in (_vr_nodes(_vr_grid()), _VR_UNEVEN)
        ps0 = _vr_held(("g", G))
        ps1 = _vr_held(("g", G .+ 41.5))
        r0 = _vr_raw(ps0, VelocityRegularizerTerm(ps0; select=_vr_names("g", 10), velocities=v))
        r1 = _vr_raw(ps1, VelocityRegularizerTerm(ps1; select=_vr_names("g", 10), velocities=v))
        worst = max(worst, _vr_worst(r1 .- r0, G .+ 41.5, v))
    end
    @info "VR2 worst |Δrow| / scale under g + 41.5" worst
    @test worst <= 1e-13
end

@testset "VR3: the stencil: even nodes against (u₄ − 3u₃ + 3u₂ − u₁)/dv^(5/2), uneven against the recursive divided difference" begin
    G = randn(Xoshiro(20261115), 10, 4)
    grid = _vr_grid()
    ps = _vr_held(("g", G))
    term = VelocityRegularizerTerm(ps; select=_vr_names("g", 10), grid=grid)
    raw = _vr_raw(ps, term)
    even = (u, v) -> [(u[i+3] - 3 * u[i+2] + 3 * u[i+1] - u[i]) / grid.dv^2.5 for i in 1:(length(v) - 3)]
    worst_even = _vr_worst(raw, G, _vr_nodes(grid); expected=even)
    @info "VR3 worst relative difference, even nodes against the hand stencil" worst_even
    @test worst_even <= 1e-13
    tu = VelocityRegularizerTerm(ps; select=_vr_names("g", 10), velocities=_VR_UNEVEN)
    raw_u = _vr_raw(ps, tu)
    worst_uneven = _vr_worst(raw_u, G, _VR_UNEVEN; expected=_vr_expected)
    @info "VR3 worst relative difference, uneven nodes against the recursive divided difference" worst_uneven
    @test worst_uneven <= 1e-13
    # the closed-form path's raw rows are the generic path's, bitwise (both from _velocity_rows)
    raw_cf = _regularizer_closed_form(Regularizer([tu]), ps, flatten(ps))[1]
    @test all(raw_cf .=== raw_u)
end

@testset "VR4: a cubic's row is 6a√(v_{i+2} − v_{i+1}) on uneven nodes" begin
    V = 3.3e6
    x = _VR_UNEVEN ./ V
    coeffs = ((0.7, -1.2, 0.4, 3.0), (-0.25, 0.0, 2.0, -1.0), (1.9, 0.6, -0.8, 0.1), (0.05, 1.5, 0.0, 0.0))
    G = hcat([a .* x .^ 3 .+ b .* x .^ 2 .+ c .* x .+ d for (a, b, c, d) in coeffs]...)
    ps = _vr_held(("g", G))
    raw = _vr_raw(ps, VelocityRegularizerTerm(ps; select=_vr_names("g", 10), velocities=_VR_UNEVEN))
    worst = 0.0
    for (j, (a, _, _, _)) in enumerate(coeffs)
        cubic = (u, v) -> [6 * (a / V^3) * sqrt(v[i+2] - v[i+1]) for i in 1:(length(v) - 3)]
        worst = max(worst, _vr_worst(raw[(j-1)*7+1:j*7], G[:, j:j], _VR_UNEVEN; expected=cubic))
    end
    @info "VR4 worst relative difference against 6a√(middle interval)" worst
    @test worst <= 1e-13
end

@testset "VR5: refinement: Σ row² in one bin approximates ∫(g‴)² dv, on a grid and at half its spacing (deliberately loose)" begin
    # g = A sin(k v + φ) on [−4h, 4h]; the coarse nodes −5h … 5h (11) and the fine −4.5h … 4.5h (19) both tile it with
    # their middle intervals. k = π/(8h): [−4h, 4h] holds one whole period of (g‴)², so the midpoint sum over the
    # middle intervals is exact and the error is the stencil's alone, Σ row² = I sinc⁶(kh/2), whose distance from I is
    # at most (kh/2)² = (kh)²/4 (sinc y ≥ 1 − y²/6 and Bernoulli; spec D11).
    h = 3.3e6
    A, φ = 2.0, 0.3
    k = pi / (8h)
    a, b = -4h, 4h
    I = A^2 * k^6 * ((b - a) / 2 + (sin(2 * (k * b + φ)) - sin(2 * (k * a + φ))) / (4k))
    S = Float64[]
    for (hh, N) in ((h, 11), (h / 2, 19))
        v = [a - hh + (n - 1) * hh for n in 1:N]
        G = repeat(A .* sin.(k .* v .+ φ), 1, 4)
        ps = _vr_held(("g", G))
        raw = _vr_raw(ps, VelocityRegularizerTerm(ps; select=_vr_names("g", N), velocities=v))
        push!(S, sum(abs2, raw[1:(N - 3)]))
    end
    e_coarse, e_fine = abs(S[1] - I) / I, abs(S[2] - I) / I
    sinc6(y) = (sin(y) / y)^6
    @info "VR5 Σ row² against ∫(g‴)²: relative errors, coarse and fine, and 1 − sinc⁶(kh/2) for each" e_coarse e_fine (1 - sinc6(k * h / 2)) (1 - sinc6(k * h / 4)) I
    @test e_coarse <= (k * h)^2 / 4
    @test e_fine <= (k * h / 2)^2 / 4
    @test abs(S[1] - S[2]) / I <= (k * h)^2 / 4
end

# ------------------------------------------------------------------------------------------
# VR6, VR7: the grid's irrelevance and the symmetric option
# ------------------------------------------------------------------------------------------

@testset "VR6: the grid form is the explicit form at _grid_node's velocities, and two grids on the same nodes give the same rows (bitwise)" begin
    G = randn(Xoshiro(20261116), 10, 4)
    ps = _vr_held(("g", G))
    A = _vr_grid()
    tg = VelocityRegularizerTerm(ps; select=_vr_names("g", 10), grid=A)
    te = VelocityRegularizerTerm(ps; select=_vr_names("g", 10), velocities=_vr_nodes(A))
    @test tg.select == te.select && tg.slots == te.slots && all(tg.velocities .=== te.velocities)
    @test all(_vr_raw(ps, tg) .=== _vr_raw(ps, te))
    # B: the same nodes (A's u₀ and dv), another default (a 1500 eV Maxwellian at −300 km/s) and another width
    vthB = sqrt(2 * 1500.0 * EV / ELECTRON_MASS)
    B = GaussianGrid(; dv_ms=A.dv, u0_ms=A.u0, width=1.0,
                     default_weights=[exp(-((vn + 3e5) / vthB)^2) for vn in _vr_nodes(A)])
    @test B.u0 === A.u0 && B.dv === A.dv && B.sigma != A.sigma && collect(B.w0) != collect(A.w0)
    tB = VelocityRegularizerTerm(ps; select=_vr_names("g", 10), grid=B)
    @test all(_vr_raw(ps, tB) .=== _vr_raw(ps, tg))
end

@testset "VR7: the symmetric grid form runs over all Nv mirrored nodes: its rows are the unsymmetric term's on the mirrored values (bitwise)" begin
    for Nv in (11, 10)
        NP = cld(Nv, 2)
        S = _vr_grid(nodes=Nv, symmetric=true)
        U = _vr_grid(nodes=Nv)
        P = randn(Xoshiro(20261117 + Nv), NP, 4)
        H = vcat([P[min(n, Nv + 1 - n):min(n, Nv + 1 - n), :] for n in 1:Nv]...)
        ps = _vr_held(("p", P), ("h", H))
        ts = VelocityRegularizerTerm(ps; select=_vr_names("p", NP), grid=S)
        tu = VelocityRegularizerTerm(ps; select=_vr_names("h", Nv), grid=U)
        @test ts.select == ["p$(min(n, Nv + 1 - n))" for n in 1:Nv]
        @test length(ts.select) == Nv && all(ts.velocities .=== tu.velocities)
        @test all(_vr_raw(ps, ts) .=== _vr_raw(ps, tu))
    end
end

# ------------------------------------------------------------------------------------------
# VR8 to VR10: the Jacobian, the summed duplicate, R and its gradient
# ------------------------------------------------------------------------------------------

# Fitted entries: g1 … g6 time-varying, `gh` held, `s` scalar, the dependent `d1` (of g2, g5 and t), and p1 … p6
# for a symmetric grid's parameters. θ: g's 24, s 1, p's 24.
function _vr_fitted()
    vars = [FittedVariable("g$n"; bounds=(-30.0, 30.0), init=0.0, time_varying=true) for n in 1:6]
    push!(vars, FittedVariable("gh"; bounds=(-30.0, 30.0), init=0.0, time_varying=true, held=[1.5, -0.5, 2.0, 0.25]))
    push!(vars, FittedVariable("s"; bounds=(-10.0, 10.0), init=0.5))
    append!(vars, [FittedVariable("p$n"; bounds=(-30.0, 30.0), init=0.0, time_varying=true) for n in 1:6])
    ps = ParameterSpec(_vr_expt(); variables=vars, dependents=["d1" => "0.5*g2 - g5 + 0.25*t"],
        bindings=Pair{String,String}[])
    theta = flatten(ps) .+ 1.2 .* sin.(1.7 .* (1:ps.ntheta))
    return ps, theta
end

# The velocity term of VR8: a time-varying entry, a held one, a scalar, a repeated name (g3) and a dependent whose
# chain reaches g5 beside g5's own entry, on uneven electron velocities.
const _VR8_SELECT = ["g1", "g2", "gh", "g3", "s", "g3", "d1", "g4", "g5", "g6"]

# ForwardDiff through the generic rows: the residual rows' Jacobian (one closure, so VR8 and VR10 compile it once).
_vr_jac(reg, pspec, theta) = ForwardDiff.jacobian(th -> _regularizer_residual_rows(reg, pspec, th), theta)

function _vr_dense(nrows, ntheta, rows, cols, coeffs)
    J = zeros(nrows, ntheta)
    for (r, c, v) in zip(rows, cols, coeffs)
        J[r, c] += v
    end
    return J
end

function _vr_worst_relerr(J_ad, J_cf)
    worst = 0.0
    for k in 1:size(J_ad, 2)
        colref = @view J_ad[:, k]
        floor_k = 1e-10 * maximum(abs, colref; init=0.0)
        worst = max(worst, maximum(abs.(J_cf[:, k] .- colref) ./ max.(abs.(colref), floor_k, 1e-300)))
    end
    return worst
end

@testset "VR8: the closed-form Jacobian against ForwardDiff through the generic rows (a held entry, a scalar, a repeated name, a dependent)" begin
    ps, theta = _vr_fitted()
    @test ps.ntheta == 49                       # gh is held: no θ entry
    term = VelocityRegularizerTerm(ps; select=_VR8_SELECT, velocities=_VR_UNEVEN, weight=3.0)
    reg = Regularizer([term]; weight=0.25)
    J_ad = _vr_jac(reg, ps, theta)
    raw, rows, cols, coeffs = _regularizer_closed_form(reg, ps, theta)
    J_cf = _vr_dense(reg.nrows, ps.ntheta, rows, cols, coeffs)
    worst = _vr_worst_relerr(J_ad, J_cf)
    @info "VR8 worst relative error, closed form against ForwardDiff" worst
    @test worst <= 1e-10
    # the held entry is a constant in its rows: the rows are the recursive divided difference of the values the term
    # reads, gh's being its held series
    vt = _value_table(ps, theta)
    U = hcat([[vt[ps.slots[nm], j] for nm in _VR8_SELECT] for j in 1:4]...)
    @test U[3, :] == [1.5, -0.5, 2.0, 0.25]
    worst_v = _vr_worst(raw, U, _VR_UNEVEN; expected=_vr_expected)
    @info "VR8 worst relative difference of the rows against the recursive divided difference" worst_v
    @test worst_v <= 1e-13
end

@testset "VR9: one summed entry per row and θ index: no two triples share a row and a column, and _gauss_newton_diagonal is JᵀJ's diagonal" begin
    ps, theta = _vr_fitted()
    tv = VelocityRegularizerTerm(ps; select=_VR8_SELECT, velocities=_VR_UNEVEN, weight=3.0)
    tsym = VelocityRegularizerTerm(ps; select=_vr_names("p", 6), grid=_vr_grid(nodes=11, symmetric=true))
    reg = Regularizer([tv, tsym]; weight=0.25)
    _, rows, cols, _ = _regularizer_closed_form(reg, ps, theta)
    @test length(unique(zip(rows, cols))) == length(rows)
    expt = _vr_expt()
    working = Experiment(; trace=ps.trace, probe=expt.probe, collection=expt.collection, lambda_m=expt.lambda,
        effects=expt.effects, prefactor=expt.prefactor)
    image = measure(unflatten(working, ps, theta))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    problem = FitProblem(working, ps, data; regularizer=reg)
    J = jacobian(problem, theta)
    d = _gauss_newton_diagonal(problem, J)
    dense = zeros(ps.ntheta)
    for k in 1:ps.ntheta
        e = zeros(ps.ntheta); e[k] = 1.0
        out = zeros(problem.npixels + reg.nrows)
        TRUST.jvp!(out, problem, J, e)
        dense[k] = sum(abs2, out)
    end
    worst = maximum(abs.(d .- dense) ./ dense)
    @info "VR9 worst relative difference, _gauss_newton_diagonal against the dense JᵀJ diagonal" worst
    @test minimum(dense) > 0
    @test worst <= 1e-12
end

@testset "VR10: R against the hand sum, and _regularizer_gradient! against the Jacobian" begin
    ps, theta = _vr_fitted()
    term = VelocityRegularizerTerm(ps; select=_VR8_SELECT, velocities=_VR_UNEVEN, weight=3.0)
    reg = Regularizer([term]; weight=0.25)
    vt = _value_table(ps, theta)
    R_hand = 3.0 * sum(sum(abs2, _vr_expected([vt[ps.slots[nm], j] for nm in _VR8_SELECT], _VR_UNEVEN)) for j in 1:4)
    R = _regularizer_value(reg, ps, theta)
    @info "VR10 relative error, R against the hand sum" abs(R - R_hand) / R_hand
    @test isapprox(R, R_hand; rtol=1e-12)
    g = zeros(ps.ntheta)
    _regularizer_gradient!(g, reg, ps, theta)
    J_ad = _vr_jac(reg, ps, theta)
    g_ad = (2 / reg.weight) .* (J_ad' * _regularizer_residual_rows(reg, ps, theta))
    floor_g = 1e-10 * maximum(abs, g_ad)
    worst = maximum(abs.(g .- g_ad) ./ max.(abs.(g_ad), floor_g))
    @info "VR10 worst relative error, the gradient against (2/Λ) Jᵀρ" worst
    @test worst <= 1e-10
end

# ------------------------------------------------------------------------------------------
# VR11 to VR13: the interval, the counts, both term types
# ------------------------------------------------------------------------------------------

@testset "VR11: an interval term has rows on its bins only, and D13's refusal" begin
    expt = _vr_expt()
    late = FitInterval("late", expt.trace.axis; t_range_ns=[1.0, 2.0])
    G = randn(Xoshiro(20261118), 10, 4)
    q = FittedVariable("q"; bounds=(-30.0, 30.0), init=0.0, time_varying=true, interval=1)
    ps = _vr_held(("g", G); intervals=[late], extra=[q])
    @test ps.intervals[1].bins == 3:4
    whole = _vr_raw(ps, VelocityRegularizerTerm(ps; select=_vr_names("g", 10), velocities=_VR_UNEVEN))
    for iv in ("late", 1)
        t = VelocityRegularizerTerm(ps; select=_vr_names("g", 10), velocities=_VR_UNEVEN, interval=iv)
        @test t.interval == 1 && length(t.rows) == 14
        @test all(_vr_raw(ps, t) .=== whole[15:28])
    end
    sel = ["g1", "g2", "q", "g4", "g5"]
    @test_throws "VelocityRegularizerTerm: the term selecting \"q\" is on the whole axis (bins 1:4), but \"q\" is defined only on its own interval \"late\" (bins 3:4); a term's interval must lie inside every selected variable's own interval" VelocityRegularizerTerm(
        ps; select=sel, velocities=_VR_UNEVEN[1:5])
    @test length(VelocityRegularizerTerm(ps; select=sel, velocities=_VR_UNEVEN[1:5], interval="late").rows) == 4
end

@testset "VR12: Nv − 3 rows per bin, with and without the symmetric option" begin
    for (Nv, sym, NP) in ((10, false, 10), (11, true, 6), (10, true, 5), (4, false, 4))
        ps2 = _vr_held(("g", zeros(NP, 4)))
        t = VelocityRegularizerTerm(ps2; select=_vr_names("g", NP), grid=_vr_grid(nodes=Nv, symmetric=sym))
        @test length(t.select) == Nv && t.rows == 1:(4 * (Nv - 3))
        @test length(_vr_raw(ps2, t)) == 4 * (Nv - 3)
    end
end

@testset "VR13: a Regularizer of a time term, a velocity term and a time term: the rows partition, R adds, the rows carry √(Λw)" begin
    ps, theta = _vr_fitted()
    t1 = RegularizerTerm(ps; select=["g1"], operator=:D2, normalization=:unit, weight=2.0)
    tv = VelocityRegularizerTerm(ps; select=_VR8_SELECT, velocities=_VR_UNEVEN, weight=3.0)
    t2 = RegularizerTerm(ps; select=["g4"], operator=:D1, normalization=:absolute, scale=10.0, weight=0.5)
    reg = Regularizer([t1, tv, t2]; weight=0.25)
    @test typeof(reg.terms) == Tuple{RegularizerTerm, VelocityRegularizerTerm, RegularizerTerm}
    @test reg.terms[1].rows == 1:2 && reg.terms[2].rows == 3:30 && reg.terms[3].rows == 31:33 && reg.nrows == 33
    R_parts = [_regularizer_value(Regularizer([t]; weight=0.25), ps, theta) for t in (t1, tv, t2)]
    @test isapprox(_regularizer_value(reg, ps, theta), sum(R_parts); rtol=1e-14)
    rho = _regularizer_residual_rows(reg, ps, theta)
    raw_all, rows, cols, coeffs = _regularizer_closed_form(reg, ps, theta)
    for (t, w) in zip(reg.terms, (2.0, 3.0, 0.5))
        lone = Regularizer([t]; weight=0.25)
        raw = _regularizer_raw_rows(lone, ps, theta)
        @test all(rho[t.rows] .=== sqrt(0.25 * w) .* raw)
        @test all(raw_all[t.rows] .=== raw)
        _, r1, c1, k1 = _regularizer_closed_form(lone, ps, theta)
        sel = findall(in(t.rows), rows)
        @test rows[sel] == r1 .+ (first(t.rows) - 1) && cols[sel] == c1 && all(coeffs[sel] .=== k1)
    end
end

# ------------------------------------------------------------------------------------------
# VR14, VR15: the refusals and show
# ------------------------------------------------------------------------------------------

@testset "VR14: the refusals" begin
    expt = _vr_expt()
    late = FitInterval("late", expt.trace.axis; t_range_ns=[1.0, 2.0])
    ps = _vr_held(("g", zeros(10, 4)); intervals=[late])
    names10, v10 = _vr_names("g", 10), _VR_UNEVEN
    grid = _vr_grid()
    @test_throws "VelocityRegularizerTerm: give exactly one of `velocities` (the explicit form) and `grid` (the grid form); got both" VelocityRegularizerTerm(
        ps; select=names10, velocities=v10, grid=grid)
    @test_throws "VelocityRegularizerTerm: give exactly one of `velocities` (the explicit form) and `grid` (the grid form); got neither" VelocityRegularizerTerm(
        ps; select=names10)
    @test_throws "VelocityRegularizerTerm: operator must be :D3; got :D2" VelocityRegularizerTerm(
        ps; select=names10, velocities=v10, operator=:D2)
    for nz in (:relative, :initial, :absolute)
        @test_throws "VelocityRegularizerTerm: normalization must be :unit; got :$nz (:relative and :initial would divide by a log-weight, which is zero at the default distribution, and :absolute's fixed divisor is the weight under another name)" VelocityRegularizerTerm(
            ps; select=names10, velocities=v10, normalization=nz)
    end
    @test_throws "VelocityRegularizerTerm: select has 10 names but velocities has 9" VelocityRegularizerTerm(
        ps; select=names10, velocities=v10[1:9])
    @test_throws "VelocityRegularizerTerm: operator :D3 needs at least 4 entries; got 3" VelocityRegularizerTerm(
        ps; select=names10[1:3], velocities=v10[1:3])
    @test_throws "VelocityRegularizerTerm: operator :D3 needs at least 4 entries; got 3" VelocityRegularizerTerm(
        ps; select=names10[1:3], grid=_vr_grid(nodes=3))
    @test_throws "VelocityRegularizerTerm: the grid has 10 parameters (g1 to g10), so select must name 10 variables, in parameter order; got 9" VelocityRegularizerTerm(
        ps; select=names10[1:9], grid=grid)
    @test_throws "VelocityRegularizerTerm: the grid has 6 parameters (g1 to g6), so select must name 6 variables, in parameter order; got 10" VelocityRegularizerTerm(
        ps; select=names10, grid=_vr_grid(nodes=11, symmetric=true))
    @test_throws "VelocityRegularizerTerm: velocity 3 is not finite; got NaN" VelocityRegularizerTerm(
        ps; select=names10[1:5], velocities=[1.0, 2.0, NaN, 4.0, 5.0])
    @test_throws "VelocityRegularizerTerm: velocity 5 is not finite; got Inf" VelocityRegularizerTerm(
        ps; select=names10[1:5], velocities=[1.0, 2.0, 3.0, 4.0, Inf])
    @test_throws "VelocityRegularizerTerm: the velocities must be strictly increasing; velocity 4 (3.0 m/s) is not above velocity 3 (3.0 m/s)" VelocityRegularizerTerm(
        ps; select=names10[1:5], velocities=[1.0, 2.0, 3.0, 3.0, 5.0])
    @test_throws "VelocityRegularizerTerm: the velocities must be strictly increasing; velocity 2 (-1.0 m/s) is not above velocity 1 (1.0 m/s)" VelocityRegularizerTerm(
        ps; select=names10[1:5], velocities=[1.0, -1.0, 3.0, 4.0, 5.0])
    for bad in ("nope", "time_varying")
        @test_throws "VelocityRegularizerTerm: \"$bad\" is not a fitted variable or a dependent; valid names are [" VelocityRegularizerTerm(
            ps; select=[names10[1:4]; bad], velocities=v10[1:5])
    end
    @test_throws "VelocityRegularizerTerm: no interval named \"early\"" VelocityRegularizerTerm(
        ps; select=names10, velocities=v10, interval="early")
    @test_throws "VelocityRegularizerTerm: interval index 2 is out of range 1:1" VelocityRegularizerTerm(
        ps; select=names10, velocities=v10, interval=2)
end

@testset "VR15: show names the operator" begin
    ps, theta = _vr_fitted()
    tv = VelocityRegularizerTerm(ps; select=_vr_names("p", 6), grid=_vr_grid(nodes=11, symmetric=true), weight=2.5)
    @test sprint(show, tv) == "VelocityRegularizerTerm(p1 … p1 (11 entries, -1.643e7 to 1.673e7 m/s); :D3, :unit, weight = 2.5)"
    t1 = RegularizerTerm(ps; select=["g1"], operator=:D2, normalization=:unit, weight=2.0)
    reg = Regularizer([t1, tv]; weight=0.25)
    @test sprint(show, reg) == "Regularizer(2 terms, weight = 0.25)"
    @test sprint((io, x) -> show(io, MIME("text/plain"), x), reg) ==
          "Regularizer\n  weight (Lambda) = 0.25\n  terms (2), 34 rows:\n" *
          "    g1: :D2, :unit, weight = 2.0, rows 1:2\n" *
          "    p1 … p1 (11 entries, -1.643e7 to 1.673e7 m/s): :D3, :unit, weight = 2.5, rows 3:34\n"
end
