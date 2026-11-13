# Step G.1.3: the projection onto a grid (Phase G.1 spec D14, D10; OQ-68 item 6; the plan's M1 and M7).
# Gate "grid projection": GP1 to GP7. The oracles are the sources themselves (each shape's own f and
# cauchy_dfdv, which share nothing with the comb) and the errors recomputed here on the evaluation set the
# spec defines. Every error pinned here is the grid's representation error, deliberately loose; the
# identities (the reported errors, determinism, the grid kept) are tight (spec D10).

using Test
using TRUST
using TRUST: GRID_PROJECTION_MAXITER, GRID_PROJECTION_G_BOUNDS
import TRUST: cauchy_dfdv
using Random

const _GP_T = 700.0
const _GP_VREF = sqrt(2 * _GP_T * EV / ELECTRON_MASS)
_gp_grid(; kw...) = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = _GP_T, kw...)

# The six registered shapes at M1's parameters, and HermiteExpansion at one coefficient set inside its
# positive region, each with its pinned error in 𝒞f and in f on the 49-node grid out to ±6 thermal speeds
# (spec D10; Level 2's measurements in the comments).
const _GP_SHAPES = (
    ("Maxwellian", Maxwellian(), 1.0e-4, 5.1e-6),                          # 9.796e-5, 4.833e-6
    ("SuperGaussian(3)", SuperGaussian(3.0), 1.0e-4, 4.4e-6),             # 9.441e-5, 4.106e-6
    ("SuperGaussian(5)", SuperGaussian(5.0), 2.7e-3, 2.4e-4),             # 2.572e-3, 2.206e-4
    ("Kappa(3)", Kappa(3.0), 6.7e-4, 1.4e-4),                             # 6.385e-4, 1.303e-4
    ("SkewNormal(2)", SkewNormal(2.0), 1.8e-4, 1.1e-5),                   # 1.651e-4, 9.660e-6
    ("GeneralizedNormal(4)", GeneralizedNormal(4.0), 3.3e-3, 4.7e-4),     # 3.161e-3, 4.384e-4
    ("HermiteExpansion(0.02, 0.01, 0, 0)", HermiteExpansion(0.02, 0.01, 0.0, 0.0), 1.5e-4, 9.2e-6),  # 1.437e-4, 8.691e-6
)

# M7's narrow beam: 95 % a Maxwellian of thermal speed vth, 5 % a Gaussian of width 0.1 vth at 2.5 vth, as a
# test-local lab-frame distribution in closed form (its call and Cauchy integral are all the projection reads).
struct _GPBeam <: TRUST.AbstractLabDistribution
    vth::Float64
end
_gp_gauss(v, c, s) = exp(-((v - c) / s)^2) / (s * sqrt(pi))
(b::_GPBeam)(v::Number) = 0.95 * _gp_gauss(v, 0.0, b.vth) + 0.05 * _gp_gauss(v, 2.5 * b.vth, 0.1 * b.vth)
cauchy_dfdv(b::_GPBeam, v::Number) =
    0.95 * Zprime(v / b.vth) / b.vth^2 + 0.05 * Zprime((v - 2.5 * b.vth) / (0.1 * b.vth)) / (0.1 * b.vth)^2

# The errors of spec D14 recomputed independently, on the evaluation set v_1 to v_Nv in steps of dv/8.
function _gp_errors(f::GaussianGrid, fsrc, csrc)
    Nv = length(f.w)
    ve = range(f.u0 - (Nv - 1) / 2 * f.dv, f.u0 + (Nv - 1) / 2 * f.dv; step = f.dv / 8)
    ef = maximum(abs(f(v) - fsrc(v)) for v in ve) / maximum(abs(fsrc(v)) for v in ve)
    ec = maximum(abs(cauchy_dfdv(f, v) - csrc(v)) for v in ve) / maximum(abs(csrc(v)) for v in ve)
    return ef, ec
end
_gp_shape_f(sp) = v -> sp.f((v - sp.u) / thermal_speed(sp)) / thermal_speed(sp)
_gp_shape_c(sp) = v -> cauchy_dfdv(sp.f, (v - sp.u) / thermal_speed(sp)) / thermal_speed(sp)^2

@testset "GP1: each registered shape on the default spacing and width, out to ±6 thermal speeds (M1)" begin
    target = _gp_grid(; nodes = 49)
    for (name, shape, tol_c, tol_f) in _GP_SHAPES
        r = project_to_grid(electrons(T_eV = _GP_T, f = shape), target)
        @info "GP1 (spec D10): $name" r.error_cauchy r.error_f r.iterations
        @test r.converged
        @test r.error_cauchy <= tol_c
        @test r.error_f <= tol_f
    end
end

@testset "GP2: Kappa(3), limited by the grid's extent: the variance it cannot hold" begin
    sp = electrons(T_eV = _GP_T, f = Kappa(3.0))
    for (nodes, lo, hi) in ((49, 0.985, 0.992), (81, 0.996, 0.999))      # 0.988773 on ±6, 0.997275 on ±10
        r = project_to_grid(sp, _gp_grid(; nodes = nodes))
        ratio = electrons(f = r.f).T / (_GP_T * EV)
        @info "GP2 (spec D10): Kappa(3), $nodes nodes" ratio r.error_cauchy
        @test lo <= ratio <= hi
        @test r.error_cauchy <= 6.7e-4
    end
end

@testset "GP3: M7's narrow beam is not representable on the default grid" begin
    for (kw, ec_min, ef_min) in (((), 0.5, 0.2), ((nodes = 81, spacing = 0.125), 0.4, 0.1))   # 0.868/0.292; 0.488/0.116
        r = project_to_grid(_GPBeam(_GP_VREF), _gp_grid(; kw...))
        @info "GP3 (spec D10): the beam, $(kw)" r.error_cauchy r.error_f
        @test r.converged
        @test r.error_cauchy >= ec_min
        @test r.error_f >= ef_min
    end
end

@testset "GP4: onto a grid twice as fine and back" begin
    coarse = project_to_grid(electrons(T_eV = _GP_T, f = SkewNormal(2.0)), _gp_grid())
    fine = project_to_grid(coarse.f, _gp_grid(; nodes = 81, spacing = 0.125))
    back = project_to_grid(fine.f, _gp_grid())
    @info "GP4 (spec D10)" fine.error_cauchy back.error_cauchy
    @test fine.converged && back.converged
    @test fine.error_cauchy <= 1.1e-4
    @test back.error_cauchy <= 1.1e-4
    _, ec = _gp_errors(back.f, v -> coarse.f(v), v -> cauchy_dfdv(coarse.f, v))
    @info "GP4: back against the coarse original" ec
    @test ec <= 1e-7      # measured 5.3e-8: the coarse comb is recovered, each step's ripple undone by the next
end

@testset "GP5: the reported errors are the errors" begin
    sp = electrons(T_eV = _GP_T, u_kms = 400, f = SkewNormal(2.0))
    r = project_to_grid(sp, _gp_grid(; u0_kms = 400))
    ef, ec = _gp_errors(r.f, _gp_shape_f(sp), _gp_shape_c(sp))
    @test isapprox(r.error_f, ef; rtol = 1e-12)
    @test isapprox(r.error_cauchy, ec; rtol = 1e-12)
    src = r.f
    r2 = project_to_grid(src, _gp_grid(; nodes = 31, spacing = 0.3, u0_kms = 400))
    ef2, ec2 = _gp_errors(r2.f, v -> src(v), v -> cauchy_dfdv(src, v))
    @test isapprox(r2.error_f, ef2; rtol = 1e-12)
    @test isapprox(r2.error_cauchy, ec2; rtol = 1e-12)
    r3 = project_to_grid(electrons(f = src), _gp_grid(; nodes = 31, spacing = 0.3, u0_kms = 400))
    @test r3.f === r2.f
end

@testset "GP6: the result: the target's grid, deterministic, positive, symmetric on request" begin
    target = _gp_grid(; u0_kms = -250, nodes = 45, spacing = 0.22, width = 1.3)
    sp = ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250, u_kms = -250, f = SuperGaussian(3.0))
    tgt_i = GaussianGrid(mass_amu = 2.014, T_ref_eV = 250, u0_kms = -250, nodes = 45, spacing = 0.22, width = 1.3)
    r = project_to_grid(sp, tgt_i)
    @test (r.f.u0, r.f.dv, r.f.sigma, r.f.w0) === (tgt_i.u0, tgt_i.dv, tgt_i.sigma, tgt_i.w0)
    @test typeof(r.f) === typeof(tgt_i)
    @test project_to_grid(sp, tgt_i).f === r.f
    @test project_to_grid(sp, reconstruct(tgt_i, NamedTuple{keys(parameters(tgt_i))}(Tuple(randn(Xoshiro(31), 45))))).f === r.f
    lo, hi = GRID_PROJECTION_G_BOUNDS
    @test all(lo <= g <= hi for g in values(parameters(r.f)))
    @test all(>(0), r.f.w)
    @test r.converged && 1 <= r.iterations < GRID_PROJECTION_MAXITER
    sym = project_to_grid(electrons(T_eV = _GP_T, f = SkewNormal(2.0)), _gp_grid(; symmetric = true))
    @test length(parameters(sym.f)) == 21
    @test sym.converged
    @test all(isapprox(sym.f(y), sym.f(-y); rtol = 1e-13) for y in (0.0:0.25:4.0) .* _GP_VREF)
    @test_throws MethodError project_to_grid(electrons(T_eV = _GP_T), reconstruct(target,
        NamedTuple{keys(parameters(target))}(ntuple(_ -> 1.0f0, 45))))
end

# D14's least-squares problem written out again (spec D10): the collocation set v_1 - 5σ to v_Nv + 5σ in steps of
# dv/4, the residuals r = dv (Σ_n c_n G(v; v_n, σ) - f(v)) with c_n = w0_n e^{g_π(n)}, and their Jacobian in g,
# J[i, k] = Σ_{π(n) = k} dv c_n G(v_i; v_n, σ). At a minimum the residual is orthogonal to every column; the largest
# cosine between them, over the parameters whose weight is above 1e-8 of the total, measures how far the projection
# stopped from it.
function _gp_stationarity(f::GaussianGrid, fsrc)
    Nv, NP = length(f.w), length(f.g)
    nodes = [f.u0 + (n - (Nv + 1) / 2) * f.dv for n in 1:Nv]
    par = [NP == Nv ? n : min(n, Nv + 1 - n) for n in 1:Nv]
    vc = collect(range(nodes[1] - 5 * f.sigma, nodes[end] + 5 * f.sigma; step = f.dv / 4))
    c = [f.w0[n] * exp(f.g[par[n]]) for n in 1:Nv]
    B = [f.dv * exp(-((v - nodes[n]) / f.sigma)^2) / (f.sigma * sqrt(pi)) for v in vc, n in 1:Nv]
    r = B * c .- f.dv .* fsrc.(vc)
    J = zeros(length(vc), NP)
    for n in 1:Nv
        J[:, par[n]] .+= B[:, n] .* c[n]
    end
    w = f.w ./ f.m0
    return maximum(abs(sum(J[:, k] .* r)) / (sqrt(sum(abs2, J[:, k])) * sqrt(sum(abs2, r)))
                   for k in 1:NP if maximum(w[n] for n in 1:Nv if par[n] == k) > 1e-8)
end

@testset "GP7: the projection stops at the least-squares minimum" begin
    for shape in (Maxwellian(), SkewNormal(2.0), Kappa(3.0))
        sp = electrons(T_eV = _GP_T, f = shape)
        r = project_to_grid(sp, _gp_grid())
        c = _gp_stationarity(r.f, _gp_shape_f(sp))
        @info "GP7 (spec D10): $(shape)" c
        @test c <= 1e-6       # measured at most 5.7e-9; GRID_PROJECTION_RTOL = 1e-3 reads 9e-5
    end
end
