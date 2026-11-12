# Step G.1.1: GaussianGrid, the first lab-frame distribution (Phase G.1 spec D3-D7, D10; OQ-68 items 1-4,
# 7, 8). Gate "GaussianGrid": GG1 to GG13. Every oracle independent of the code under test (conventions.md
# 8.2): a quadrature (QuadGK, and the generic Cauchy fallbacks applied to a test-local shape that writes the
# comb out again in a reduced variable), ForwardDiff, a finite difference, or the Maxwellian's own Zprime.
# Deliberately loose rows are the grid's sampling error, never round-off; deliberately tight ones are
# identities (spec D10).

using Test
using TRUST
using TRUST: _cauchy_real, _cauchy_upper, _cauchy_lower, _grid_node, _shape_eltype
import TRUST: dfdx
using QuadGK: quadgk
using Random
import ForwardDiff

# The electron grid every row uses unless it says otherwise: T_ref = 700 eV, u0 = 300 km/s, the defaults
# (41 nodes, spacing 0.25, width 1.2). _GG_VREF is its reference thermal speed, written out here.
const _GG_VREF = sqrt(2 * 700 * EV / ELECTRON_MASS)
const _GG_U0 = 3.0e5
_gg_grid(; kw...) = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 700, u0_kms = 300, kw...)
_gg_with(f, g) = reconstruct(f, NamedTuple{keys(parameters(f))}(Tuple(g)))

# Random log-weights, seeded per row (spec D10).
_gg_random(f, seed) = _gg_with(f, 0.3 .* randn(Xoshiro(seed), length(parameters(f))))

# ------------------------------------------------------------------------------------------
# The independent oracle for the Cauchy integral (spec D10): the comb written out again as a test-local
# AbstractVelocityDistribution in x = (v - u0)/_GG_VREF, with its own call and its own real and complex dfdx,
# handed to the generic quadrature fallbacks _cauchy_real, _cauchy_upper and _cauchy_lower directly. It
# shares the weights and nothing else; 𝒞f(v) = 𝒞φ((v - u0)/_GG_VREF)/_GG_VREF².
# ------------------------------------------------------------------------------------------

struct _GGComb{N} <: TRUST.AbstractVelocityDistribution
    a::NTuple{N, Float64}
    x::NTuple{N, Float64}
    s::Float64
end

function _gg_comb(f::GaussianGrid{Float64, Nv}) where {Nv}
    m = sum(f.w)
    return _GGComb{Nv}(ntuple(n -> f.w[n] / m, Nv), ntuple(n -> (n - (Nv + 1) / 2) * f.dv / _GG_VREF, Nv),
                       f.sigma / _GG_VREF)
end

(c::_GGComb)(x::Number) = sum(c.a[n] * exp(-((x - c.x[n]) / c.s)^2) for n in eachindex(c.a)) / (c.s * sqrt(pi))
_gg_comb_dfdx(c::_GGComb, x) =
    sum(c.a[n] * (-2 * (x - c.x[n]) / c.s^2) * exp(-((x - c.x[n]) / c.s)^2) for n in eachindex(c.a)) / (c.s * sqrt(pi))
dfdx(c::_GGComb, x::Real) = _gg_comb_dfdx(c, x)
dfdx(c::_GGComb, x::Complex) = _gg_comb_dfdx(c, x)

# The Maxwellian of T_ref and u0 in lab velocity (the g = 0 reference), from the reduced form.
_gg_maxwellian_f(v) = exp(-((v - _GG_U0) / _GG_VREF)^2) / (_GG_VREF * sqrt(pi))
_gg_maxwellian_dfdv(v) = -2 * ((v - _GG_U0) / _GG_VREF) * exp(-((v - _GG_U0) / _GG_VREF)^2) / (_GG_VREF^2 * sqrt(pi))

# A finite window for every quadrature over v (conventions.md 8.8): x = (v - u0)/_GG_VREF over [-9, 9] with a
# breakpoint at every integer; the comb beyond |x| = 9 is below e^{-(4/0.3)^2}, nothing in Float64.
_gg_integral(h) = quadgk(x -> h(_GG_U0 + x * _GG_VREF) * _GG_VREF, (-9.0:1.0:9.0)...; rtol = 1e-14, atol = 0.0)[1]

# ------------------------------------------------------------------------------------------
# GG1-GG6: the distribution and its closed forms (OQ-68 items 2-4)
# ------------------------------------------------------------------------------------------

@testset "GG1: unit integral for random g" begin
    for f in (_gg_random(_gg_grid(), 1), _gg_random(_gg_grid(), 2), _gg_random(_gg_grid(; symmetric = true), 3),
              _gg_random(_gg_grid(; nodes = 9, spacing = 0.5, width = 1.5), 4))
        @test isapprox(_gg_integral(f), 1.0; atol = 1e-12)
    end
end

@testset "GG2: g = 0 against the Maxwellian of T_ref and u0" begin
    xs = -6.0:0.01:6.0
    for (kw, tol_c, tol_f, tol_d) in (((), 1.0e-4, 5.0e-6, 1.3e-4), ((width = 1.5,), 2.0e-7, 1.1e-8, 3.0e-7))
        f = _gg_grid(; kw...)
        vs = _GG_U0 .+ xs .* _GG_VREF
        err_c = maximum(abs(cauchy_dfdv(f, v) * _GG_VREF^2 - Zprime((v - _GG_U0) / _GG_VREF)) for v in vs) / 2
        err_f = maximum(abs(f(v) - _gg_maxwellian_f(v)) for v in vs) / _gg_maxwellian_f(_GG_U0)
        err_d = maximum(abs(dfdv(f, v) - _gg_maxwellian_dfdv(v)) for v in vs) / maximum(abs, _gg_maxwellian_dfdv.(vs))
        @info "GG2 (spec D10): g = 0 against the Maxwellian, $(kw)" err_c err_f err_d
        @test err_c <= tol_c
        @test err_f <= tol_f
        @test err_d <= tol_d
        @test abs(mean_velocity(f) - _GG_U0) <= 1e-12 * _GG_VREF
        @test isapprox(ELECTRON_MASS * velocity_variance(f), 700 * EV; rtol = 1e-11)
    end
end

@testset "GG3: dfdv against ForwardDiff" begin
    f = _gg_random(_gg_grid(), 5)
    vs = _GG_U0 .+ (-6.0:0.05:6.0) .* _GG_VREF
    ad = [ForwardDiff.derivative(f, v) for v in vs]
    floor = 1e-12 * maximum(abs, ad)
    for (v, d) in zip(vs, ad)
        @test isapprox(dfdv(f, v), d; rtol = 1e-12, atol = floor)
    end
    for x in -4.0:1.0:4.0, y in (-0.3, 0.2, 0.5)
        v = _GG_U0 + complex(x, y) * _GG_VREF
        d = complex(ForwardDiff.derivative(t -> real(f(v + t)), 0.0), ForwardDiff.derivative(t -> imag(f(v + t)), 0.0))
        @test isapprox(dfdv(f, v), d; rtol = 1e-12, atol = floor)
    end
end

@testset "GG4: cauchy_dfdv against the quadrature fallbacks on a test-local comb" begin
    for f in (_gg_grid(), _gg_random(_gg_grid(), 6))
        c = _gg_comb(f)
        for x in -6.0:0.5:6.0
            @test isapprox(cauchy_dfdv(f, _GG_U0 + x * _GG_VREF) * _GG_VREF^2, _cauchy_real(c, x); rtol = 1e-9)
        end
        for x in -6.0:1.0:6.0, y in (0.1, 0.5, 2.0)
            zu = complex(x, y)
            zl = complex(x, -y)
            @test isapprox(cauchy_dfdv(f, _GG_U0 + zu * _GG_VREF) * _GG_VREF^2, _cauchy_upper(c, zu); rtol = 1e-9)
            @test isapprox(cauchy_dfdv(f, _GG_U0 + zl * _GG_VREF) * _GG_VREF^2, _cauchy_lower(c, zl); rtol = 1e-9)
        end
    end
end

@testset "GG5: Im 𝒞f = π f′ on the real axis" begin
    for f in (_gg_grid(), _gg_random(_gg_grid(), 7))
        vs = _GG_U0 .+ (-7.0:0.01:7.0) .* _GG_VREF
        floor = pi * maximum(v -> abs(dfdv(f, v)), vs)
        @test maximum(v -> abs(imag(cauchy_dfdv(f, v)) - pi * dfdv(f, v)), vs) <= 1e-14 * floor
    end
end

@testset "GG6: the moments, the mean velocity and the variance against quadrature" begin
    f = _gg_random(_gg_grid(), 8)
    for n in 0:8
        q = quadgk(x -> (_GG_U0 + x * _GG_VREF)^n * f(_GG_U0 + x * _GG_VREF) * _GG_VREF, (-9.0:1.0:9.0)...;
                   rtol = 1e-14, atol = 1e-14 * _GG_VREF^n)[1]
        @test isapprox(moment(f, n), q; rtol = 1e-12, atol = 1e-12 * _GG_VREF^n)
    end
    ubar = _gg_integral(v -> v * f(v))
    @test isapprox(mean_velocity(f), ubar; rtol = 1e-12)
    @test isapprox(velocity_variance(f), _gg_integral(v -> (v - ubar)^2 * f(v)); rtol = 1e-12)
    @test isapprox(moment(f, 0), 1.0; atol = 1e-14)
    @test isapprox(moment(f, 1), mean_velocity(f); rtol = 1e-14)
    @test isapprox(moment(f, 2) - moment(f, 1)^2, velocity_variance(f); rtol = 1e-12)
    @test_throws "moment is defined for n >= 0; got n = -1" moment(f, -1)
end

# ------------------------------------------------------------------------------------------
# GG7-GG10: the parameters (OQ-68 items 3, 4, 7)
# ------------------------------------------------------------------------------------------

# The one-term closed form of OQ-68 item 4 for the partial of 𝒞f in the weight of node n.
_gg_one_term(f, v, n) = (f.w[n] / f.m0) * (Zprime((v - _grid_node(f, n)) / f.sigma) / f.sigma^2 - cauchy_dfdv(f, v))

# f with every parameter dual, N partials, parameter i seeded in partial mod1(i, N) (spec D7).
function _gg_dual(f, ::Val{N}) where {N}
    p = parameters(f)
    NP = length(p)
    return reconstruct(f, NamedTuple{keys(p)}(ntuple(i -> ForwardDiff.Dual{:GG}(p[i],
        ntuple(j -> j == mod1(i, N) ? 1.0 : 0.0, Val(N))), Val(NP))))
end

# The parameter node n reads (OQ-68 item 3): n, or the nearer of n and its mirror with the symmetric option.
_gg_param(n, Nv, NP) = NP == Nv ? n : min(n, Nv + 1 - n)

@testset "GG7: the partials of 𝒞f in each g" begin
    for f in (_gg_random(_gg_grid(), 9), _gg_random(_gg_grid(; symmetric = true), 10))
        NP = length(parameters(f))
        Nv = length(f.w)
        fd = _gg_dual(f, Val(NP))
        for v in _GG_U0 .+ (-3.3, -0.4, 0.9, 2.6) .* _GG_VREF
            c = cauchy_dfdv(fd, v)
            pr = ForwardDiff.partials(real(c))
            pi_ = ForwardDiff.partials(imag(c))
            closed = [sum(_gg_one_term(f, v, n) for n in 1:Nv if _gg_param(n, Nv, NP) == k) for k in 1:NP]
            fl = 1e-12 * maximum(abs, closed)
            for k in 1:NP
                @test isapprox(complex(pr[k], pi_[k]), closed[k]; rtol = 1e-12, atol = fl)
            end
            h = 1e-6
            fdiff = [(cauchy_dfdv(_gg_with(f, ntuple(i -> parameters(f)[i] + (i == k ? h : 0.0), NP)), v) -
                      cauchy_dfdv(_gg_with(f, ntuple(i -> parameters(f)[i] - (i == k ? h : 0.0), NP)), v)) / (2h)
                     for k in 1:NP]
            flfd = 1e-6 * maximum(abs, fdiff)
            for k in 1:NP
                @test isapprox(complex(pr[k], pi_[k]), fdiff[k]; rtol = 1e-6, atol = flfd)
            end
        end
    end
end

@testset "GG8: g + c changes nothing" begin
    f = _gg_random(_gg_grid(), 11)
    fc = _gg_with(f, values(parameters(f)) .+ 2.5)
    vs = _GG_U0 .+ (-6.0:0.1:6.0) .* _GG_VREF
    fmax = maximum(f, vs)
    dmax = maximum(v -> abs(dfdv(f, v)), vs)
    cmax = maximum(v -> abs(cauchy_dfdv(f, v)), vs)
    for v in vs
        @test isapprox(fc(v), f(v); rtol = 1e-13, atol = 1e-13 * fmax)
        @test isapprox(dfdv(fc, v), dfdv(f, v); rtol = 1e-13, atol = 1e-13 * dmax)
        @test isapprox(cauchy_dfdv(fc, v), cauchy_dfdv(f, v); rtol = 1e-13, atol = 1e-13 * cmax)
    end
    @test isapprox(mean_velocity(fc), mean_velocity(f); rtol = 1e-13)
    @test isapprox(velocity_variance(fc), velocity_variance(f); rtol = 1e-13)
    for n in 0:4
        @test isapprox(moment(fc, n), moment(f, n); rtol = 1e-13, atol = 1e-13 * _GG_VREF^n)
    end
end

@testset "GG9: an argument 150 reference thermal speeds outside the grid" begin
    f = _gg_grid()
    for x in (150.0, -150.0)
        c = cauchy_dfdv(f, _GG_U0 + x * _GG_VREF) * _GG_VREF^2
        @test isapprox(c, Zprime(x); rtol = 1e-9)
    end
end

@testset "GG10: the symmetric option" begin
    for Nv in (41, 40)
        full = _gg_grid(; nodes = Nv)
        sym = _gg_grid(; nodes = Nv, symmetric = true)
        NP = cld(Nv, 2)
        @test length(parameters(sym)) == NP
        @test keys(parameters(sym)) == Tuple(Symbol("g", i) for i in 1:NP)
        fs = _gg_random(sym, 12)
        gs = values(parameters(fs))
        ff = _gg_with(full, ntuple(n -> gs[min(n, Nv + 1 - n)], Nv))
        vs = _GG_U0 .+ (-6.0:0.1:6.0) .* _GG_VREF
        @test all(fs(v) === ff(v) for v in vs)
        @test all(dfdv(fs, v) === dfdv(ff, v) for v in vs)
        @test all(cauchy_dfdv(fs, v) === cauchy_dfdv(ff, v) for v in vs)
        @test mean_velocity(fs) === mean_velocity(ff)
        @test velocity_variance(fs) === velocity_variance(ff)
        fmax = maximum(fs, vs)
        for y in (0.0:0.1:6.0) .* _GG_VREF
            @test isapprox(fs(_GG_U0 + y), fs(_GG_U0 - y); rtol = 1e-14, atol = 1e-14 * fmax)
        end
        @test isapprox(mean_velocity(fs), _GG_U0; rtol = 1e-14)
    end
    # spec D4: the default is written from the offset (n - (Nv + 1)/2) dv, so it is exactly symmetric even where a
    # node v_n would carry u0's rounding (a deuteron grid with a flow larger than its spacing, as here)
    w0 = GaussianGrid(mass_amu = 2.014, T_ref_eV = 250, u0_kms = 300.123, nodes = 9, spacing = 0.5).w0
    @test all(w0[n] === w0[10 - n] for n in 1:9)
    e = GaussianGrid(; dv_ms = 1.0e6, default_weights = collect(w0), symmetric = true)
    @test length(parameters(e)) == 5
end

# ------------------------------------------------------------------------------------------
# GG11-GG13: the refusals, allocations and the API (OQ-68 items 2, 3, 7; spec D4, D7)
# ------------------------------------------------------------------------------------------

@testset "GG11: the refusals" begin
    me = ELECTRON_MASS
    @test_throws "GaussianGrid: nodes must be an integer of at least 3; got 2" GaussianGrid(mass_kg = me, T_ref_eV = 700, nodes = 2)
    @test_throws "GaussianGrid: nodes must be an integer of at least 3; got 41.0" GaussianGrid(mass_kg = me, T_ref_eV = 700, nodes = 41.0)
    @test_throws "GaussianGrid: spacing must be finite and positive; got 0.0" GaussianGrid(mass_kg = me, T_ref_eV = 700, spacing = 0)
    @test_throws "GaussianGrid: spacing must be finite and positive; got Inf" GaussianGrid(mass_kg = me, T_ref_eV = 700, spacing = Inf)
    @test_throws "GaussianGrid: width must be finite and positive; got -1.2" GaussianGrid(mass_kg = me, T_ref_eV = 700, width = -1.2)
    @test_throws "GaussianGrid: width must be finite and positive; got NaN" GaussianGrid(mass_kg = me, T_ref_eV = 700, width = NaN)
    @test_throws "GaussianGrid: T_ref must be finite and positive; got -1.602176634e-19" GaussianGrid(mass_kg = me, T_ref_eV = -1)
    @test_throws "GaussianGrid: T_ref must be finite and positive; got NaN" GaussianGrid(mass_kg = me, T_ref_keV = NaN)
    @test_throws "GaussianGrid: mass must be finite and positive; got 0.0" GaussianGrid(mass_kg = 0.0, T_ref_eV = 700)
    @test_throws "GaussianGrid: spacing · width must be below 1 for the Maxwellian default (its width √(v_ref² − σ²) needs σ < v_ref); got spacing = 0.5, width = 2.0" GaussianGrid(mass_kg = me, T_ref_eV = 700, spacing = 0.5, width = 2)
    @test_throws "GaussianGrid: u0 must be finite; got Inf m/s" GaussianGrid(mass_kg = me, T_ref_eV = 700, u0_kms = Inf)
    @test_throws "GaussianGrid: u0 must be finite; got NaN m/s" GaussianGrid(dv_kms = 100, default_weights = [1, 2, 1], u0_ms = NaN)
    @test_throws "GaussianGrid: dv must be finite and positive; got 0.0" GaussianGrid(dv_kms = 0, default_weights = [1, 2, 1])
    @test_throws "GaussianGrid: dv must be finite and positive; got -1000.0" GaussianGrid(dv_kms = -1, default_weights = [1, 2, 1])
    @test_throws "GaussianGrid: default_weights[2] must be finite and positive; got 0.0" GaussianGrid(dv_kms = 1, default_weights = [1, 0, 1])
    @test_throws "GaussianGrid: default_weights[3] must be finite and positive; got Inf" GaussianGrid(dv_kms = 1, default_weights = [1, 2, Inf])
    @test_throws "GaussianGrid: nodes must be an integer of at least 3; got 2" GaussianGrid(dv_kms = 1, default_weights = [1, 1])
    @test_throws "GaussianGrid: symmetric = true needs default_weights symmetric about the centre (default_weights[n] == default_weights[Nv + 1 - n]); they differ at n = 1" GaussianGrid(dv_kms = 1, default_weights = [1, 2, 3], symmetric = true)
    @test_throws "GaussianGrid: the explicit form (dv_kms or dv_ms) needs default_weights" GaussianGrid(dv_kms = 1)
    @test_throws "GaussianGrid: give a reference temperature (T_ref_eV or T_ref_keV) or a velocity spacing (dv_kms or dv_ms), not both" GaussianGrid(mass_kg = me, T_ref_eV = 700, dv_kms = 1)
    @test_throws "GaussianGrid: no grid given; expected T_ref_eV or T_ref_keV with mass_amu or mass_kg (the temperature form), or dv_kms or dv_ms with default_weights (the explicit form)" GaussianGrid(mass_kg = me)
    @test_throws "no unit keyword given for mass" GaussianGrid(T_ref_eV = 700)
    @test_throws "unrecognised keyword(s) (:spacing,)" GaussianGrid(dv_kms = 1, default_weights = [1, 2, 1], spacing = 0.25)
    @test_throws "unrecognised keyword(s) (:nodez,)" GaussianGrid(mass_kg = me, T_ref_eV = 700, nodez = 41)
    f = _gg_grid(; nodes = 5)
    @test_throws "GaussianGrid: reconstruct needs the parameters g1 to g5 in order; got (:g1, :g2, :g3, :g5, :g4)" reconstruct(f, (g1 = 0, g2 = 0, g3 = 0, g5 = 0, g4 = 0))
    @test_throws "GaussianGrid: g3 must be finite; got NaN" reconstruct(f, (g1 = 0, g2 = 0, g3 = NaN, g4 = 0, g5 = 0))
    @test_throws "GaussianGrid: g1 must be finite; got Inf" reconstruct(f, (g1 = Inf, g2 = 0, g3 = 0, g4 = 0, g5 = 0))
    @test_throws "GaussianGrid: the weights w0 exp(g) sum to Inf; the sum must be finite and positive" reconstruct(f, (g1 = 0, g2 = 0, g3 = 800, g4 = 0, g5 = 0))
    @test_throws "GaussianGrid: the weights w0 exp(g) sum to 0.0; the sum must be finite and positive" reconstruct(f, ntuple(_ -> -800.0, 5) |> t -> NamedTuple{keys(parameters(f))}(t))
end

# A number that depends on every part of the result, so that nothing is optimized away (spec D7).
_gg_consume(z::Real) = z
_gg_consume(z::Complex) = real(z) + imag(z)
_gg_consume(z::ForwardDiff.Dual) = ForwardDiff.value(z) + sum(ForwardDiff.partials(z))
_gg_consume(z::Complex{<:ForwardDiff.Dual}) = _gg_consume(real(z)) + _gg_consume(imag(z))
_gg_consume(nt::NamedTuple) = _gg_consume(nt[1]) + _gg_consume(nt[end])
_gg_consume(f::GaussianGrid) = _gg_consume(f.m0)
_gg_allocated(op, f, x) = (_gg_consume(op(f, x)); @allocated _gg_consume(op(f, x)))

@testset "GG12: allocation-free, as a value and under duals of 8 and of Nv + 3 partials" begin
    ops = ((f, v) -> f(v), dfdv, cauchy_dfdv, (f, v) -> cauchy_dfdv(f, v + 0.2im * _GG_VREF), (f, n) -> moment(f, 3),
           (f, v) -> mean_velocity(f), (f, v) -> velocity_variance(f), (f, v) -> parameters(f),
           (f, p) -> reconstruct(f, p))
    f = _gg_random(_gg_grid(), 13)
    v = _GG_U0 + 0.7 * _GG_VREF
    for fx in (f, _gg_dual(f, Val(8)), _gg_dual(f, Val(44)))
        p = parameters(fx)
        for op in ops
            arg = op === ops[end] ? p : v
            @test _gg_allocated(op, fx, arg) == 0
        end
    end
    vd = ForwardDiff.Dual{:GGv}(v, ntuple(i -> Float64(i), 8))
    for op in ops[1:4]
        @test _gg_allocated(op, f, vd) == 0
    end
end

@testset "GG13: parameters, reconstruct, the two forms, promotion, show" begin
    f = _gg_grid()
    @test isbitstype(typeof(f))
    @test typeof(f) === GaussianGrid{Float64, 41, 41}
    @test keys(parameters(f)) == Tuple(Symbol("g", i) for i in 1:41)
    @test all(iszero, values(parameters(f)))
    @test f.dv === 0.25 * _GG_VREF
    @test f.sigma === 1.2 * f.dv
    fr = _gg_random(f, 15)
    @test reconstruct(fr, parameters(fr)) === fr
    @test reconstruct(f, NamedTuple{keys(parameters(f))}(ntuple(_ -> 1, 41))) isa GaussianGrid{Float64, 41, 41}
    @test reconstruct(f, NamedTuple{keys(parameters(f))}(ntuple(_ -> 1.0f0, 41))) isa GaussianGrid{Float32, 41, 41}
    d = ForwardDiff.Dual{:GG13}(0.1, 1.0)
    fd = reconstruct(f, NamedTuple{keys(parameters(f))}(ntuple(i -> i == 7 ? d : 0.0, 41)))
    @test fd isa GaussianGrid{typeof(d), 41, 41}
    @test isbitstype(typeof(fd))
    @test (@inferred _shape_eltype(fd)) === typeof(d)     # spec D8: GaussianGrid's own method infers;
    @test (@inferred _shape_eltype(f)) === Float64         # a promote_type over 41 values does not
    # the explicit form on the temperature form's own default is the same distribution, bitwise
    @test GaussianGrid(dv_ms = f.dv, default_weights = f.w0, u0_ms = f.u0) === f
    @test GaussianGrid(mass_kg = ELECTRON_MASS, T_ref_keV = 1, u0_ms = 3.0e5) === GaussianGrid(mass_kg = ELECTRON_MASS, T_ref_eV = 1000, u0_kms = 300)
    @test GaussianGrid(mass_amu = 2.014, T_ref_eV = 200) === GaussianGrid(mass_kg = 2.014 * ATOMIC_MASS_UNIT, T_ref_eV = 200)
    @test GaussianGrid(dv_kms = f.dv / 1e3, default_weights = f.w0, u0_kms = 300).dv === f.dv
    @test sprint(show, f) == "GaussianGrid(41 nodes, u0 = 300.0 km/s, dv = 3923.0 km/s, width = 1.2)"
    @test sprint(show, _gg_grid(; nodes = 9, symmetric = true, width = 1.5)) == "GaussianGrid(9 nodes, u0 = 300.0 km/s, dv = 3923.0 km/s, width = 1.5, symmetric)"
    @test sprint(show, fd) == sprint(show, f)
end
