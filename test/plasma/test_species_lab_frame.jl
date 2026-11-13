# Step G.1.2: a species that carries a lab-frame distribution (Phase G.1 spec D8, D9, D11-D13; OQ-68 item 5).
# Gate "lab-frame family": GC1 to GC12. The spectra's oracle is the package's own Maxwellian spectrum, which
# shares no code with the comb (the shape route: alpha, xi, Zprime of the reduced phase velocity); the
# susceptibility's identity is checked exactly on a test-local lab-frame Maxwellian; derivatives against
# central differences (conventions.md 8.2, 8.4).

using Test
using TRUST
using TRUST: _shape_eltype, _plasma_frequency_sq, _distribution_u_T
import TRUST: cauchy_dfdv, parameters, reconstruct, mean_velocity, velocity_variance
using Random
import ForwardDiff

const _GC_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _GC_COLLECTION = Collection(theta_deg = 60, k_model = :exact)
const _GC_EPW = collect(range(220e-9, 260e-9; length = 512))     # M4's window, blue of the probe
const _GC_IAW = collect(range(262e-9, 264.5e-9; length = 512))   # M5's window
_gc_deuterons(; kw...) = ion(name = "D", mass_amu = 2.014, charge = 1; kw...)
_gc_plasma(e, i) = Plasma(n_e_cm3 = 5e19, electrons = (e,), ions = (i,))
_gc_peak_error(p, q, lam) = (P = spectrum(p, _GC_PROBE, _GC_COLLECTION, lam);
                             Q = spectrum(q, _GC_PROBE, _GC_COLLECTION, lam);
                             maximum(abs.(P .- Q)) / maximum(Q))
_gc_with(f, g) = reconstruct(f, NamedTuple{keys(parameters(f))}(Tuple(g)))

# The weights of a Maxwellian of temperature T (eV) on a grid of T_ref = 700 eV (M8), written out: w_n =
# dv G(v_n; u0, sqrt(v_T^2 - sigma^2)) at the offsets (n - 21) dv of the 41-node default grid.
function _gc_maxwellian_on_grid(f, T_eV)
    vT = sqrt(2 * T_eV * EV / ELECTRON_MASS)
    s0 = sqrt(vT^2 - f.sigma^2)
    w = ntuple(n -> f.dv * exp(-(((n - 21) * f.dv) / s0)^2) / (s0 * sqrt(pi)), 41)
    return _gc_with(f, ntuple(n -> log(w[n] / f.w0[n]), 41))
end

# A test-local lab-frame distribution that is a Maxwellian exactly, f(v) = e^{-((v-u)/vth)^2}/(vth sqrt(pi)),
# 𝒞f = Z'((v - u)/vth)/vth^2 (spec D10): carried by a species it must reproduce the shape route to round-off,
# which is rule 1.6's equivalence with no sampling error in it.
struct _GCLabMaxwellian <: TRUST.AbstractLabDistribution
    u::Float64
    vth::Float64
end
(f::_GCLabMaxwellian)(v::Number) = exp(-((v - f.u) / f.vth)^2) / (f.vth * sqrt(pi))
cauchy_dfdv(f::_GCLabMaxwellian, v::Number) = Zprime((v - f.u) / f.vth) / f.vth^2
mean_velocity(f::_GCLabMaxwellian) = f.u
velocity_variance(f::_GCLabMaxwellian) = f.vth^2 / 2
parameters(::_GCLabMaxwellian) = NamedTuple()
reconstruct(f::_GCLabMaxwellian, ::NamedTuple) = f

# ------------------------------------------------------------------------------------------
# GC1-GC2: the type bound, the builders, the derived u and T (OQ-68 item 5; spec D8, D9)
# ------------------------------------------------------------------------------------------

@testset "GC1: either family in the distribution slot; the builders" begin
    @test typeof(electrons(T_eV = 700)) === SpeciesSpec{Float64, Maxwellian}
    @test typeof(_gc_deuterons(T_eV = 250, f = SkewNormal(1.0))) <: SpeciesSpec{Float64, <:SkewNormal}
    g = GaussianGrid(mass_kg = ELECTRON_MASS, T_ref_eV = 700)
    eg = electrons(grid = (T_ref_eV = 700,))
    @test typeof(eg) === SpeciesSpec{Float64, GaussianGrid{Float64, 41, 41}}
    @test eg.f === g
    ef = electrons(f = g)
    @test (ef.name, ef.mass, ef.charge, ef.u, ef.T, ef.f, ef.fraction) ===
          (eg.name, eg.mass, eg.charge, eg.u, eg.T, eg.f, eg.fraction)
    di = _gc_deuterons(grid = (T_ref_eV = 250, nodes = 31, symmetric = true))
    @test di.f === GaussianGrid(mass_amu = 2.014, T_ref_eV = 250, nodes = 31, symmetric = true)
    p = _gc_plasma(eg, di)
    @test typeof(p.electrons[1]) === Species{Float64, GaussianGrid{Float64, 41, 41}}
    @test typeof(p.ions[1]) === Species{Float64, GaussianGrid{Float64, 31, 16}}
    s = spec(p.electrons[1])
    @test (s.u, s.T, s.f) === (eg.u, eg.T, eg.f)
    @test _shape_eltype(_GCLabMaxwellian(0.0, 1.0)) === Bool
end

@testset "GC2: u and T are the moments, exactly; at g = 0 they are u0 and T_ref" begin
    for (sp0, m, T_ref, u0) in ((electrons(grid = (T_ref_eV = 700, u0_kms = 300)), ELECTRON_MASS, 700, 3e5),
                                (_gc_deuterons(grid = (T_ref_eV = 250, u0_kms = -150)), 2.014 * ATOMIC_MASS_UNIT, 250, -1.5e5))
        @test abs(sp0.u - u0) <= 1e-12 * sqrt(2 * T_ref * EV / m)
        @test isapprox(sp0.T, T_ref * EV; rtol = 1e-11)
        f = _gc_with(sp0.f, 0.3 .* randn(Xoshiro(21), 41))
        sp = sp0.charge < 0 ? electrons(f = f) : _gc_deuterons(f = f)
        @test sp.u == mean_velocity(f)
        @test sp.T == m * velocity_variance(f)
        q = _gc_plasma(sp0.charge < 0 ? sp : electrons(T_eV = 700), sp0.charge < 0 ? _gc_deuterons(T_eV = 250) : sp)
        r = sp0.charge < 0 ? q.electrons[1] : q.ions[1]
        @test r.u == mean_velocity(f)
        @test r.T == r.mass * velocity_variance(f)
        @test thermal_speed(r) == sqrt(2 * r.T / r.mass)
        @test debye_length(r) == sqrt(VACUUM_PERMITTIVITY * r.T / (r.n * (r.charge * ELEMENTARY_CHARGE)^2))
        @test xi(r, 2.0e7, 3.0e14) == (3.0e14 / 2.0e7 - r.u) / thermal_speed(r)
    end
end

# ------------------------------------------------------------------------------------------
# GC3-GC6: the susceptibility and the spectra (OQ-68 items 5, 8; M4, M5, M8; spec D10, D12)
# ------------------------------------------------------------------------------------------

@testset "GC3: rule 1.6's equivalence, exactly, on a lab-frame Maxwellian" begin
    T = 700 * EV
    vth = sqrt(2 * T / ELECTRON_MASS)
    lab = electrons(f = _GCLabMaxwellian(2.0e5, vth))
    shape = electrons(T_eV = lab.T / EV, u_ms = lab.u)
    @test isapprox(lab.T, T; rtol = 1e-15)
    pl = _gc_plasma(lab, _gc_deuterons(T_eV = 250))
    ps = _gc_plasma(shape, _gc_deuterons(T_eV = 250))
    k = 2.4e7
    for x in -4.0:0.25:4.0
        om = k * (2.0e5 + x * vth)
        @test isapprox(susceptibility(pl.electrons[1], k, om), susceptibility(ps.electrons[1], k, om); rtol = 1e-13)
        @test isapprox(spectral_density(pl, k, om), spectral_density(ps, k, om); rtol = 1e-13)
        @test susceptibility(pl.electrons[1], k, om) ==
              -(_plasma_frequency_sq(pl.electrons[1]) / k^2) * cauchy_dfdv(pl.electrons[1].f, om / k)
    end
end

@testset "GC4: a gridded Maxwellian's spectrum against the package's, electrons on the EPW window" begin
    err = _gc_peak_error(_gc_plasma(electrons(grid = (T_ref_eV = 700,)), _gc_deuterons(T_eV = 250)),
                         _gc_plasma(electrons(T_eV = 700), _gc_deuterons(T_eV = 250)), _GC_EPW)
    @info "GC4 (spec D10): gridded electrons, EPW window, max |dP| / peak" err
    @test err <= 6.5e-5
end

@testset "GC5: a gridded ion on the IAW window" begin
    err = _gc_peak_error(_gc_plasma(electrons(T_eV = 700), _gc_deuterons(grid = (T_ref_eV = 250,))),
                         _gc_plasma(electrons(T_eV = 700), _gc_deuterons(T_eV = 250)), _GC_IAW)
    @info "GC5 (spec D10): gridded deuterons, IAW window, max |dP| / peak" err
    @test err <= 6.5e-5
end

@testset "GC6: one grid at T_ref = 700 eV holding Maxwellians at 500, 700 and 1000 eV (M8)" begin
    g = GaussianGrid(mass_kg = ELECTRON_MASS, T_ref_eV = 700)
    for (T, tol, Ttol) in ((500.0, 8.0e-5, 1e-12), (700.0, 6.5e-5, 1e-11), (1000.0, 6.0e-5, 2e-8))
        f = _gc_maxwellian_on_grid(g, T)
        e = electrons(f = f)
        @test isapprox(e.T, T * EV; rtol = Ttol)
        err = _gc_peak_error(_gc_plasma(e, _gc_deuterons(T_eV = 250)), _gc_plasma(electrons(T_eV = T), _gc_deuterons(T_eV = 250)), _GC_EPW)
        @info "GC6 (spec D10): T = $T eV on the 700 eV grid" err e.T / EV
        @test err <= tol
    end
end

# ------------------------------------------------------------------------------------------
# GC7-GC9: derivatives, the trace, rule 5.15 (OQ-68 items 5, 7; spec D8, D12)
# ------------------------------------------------------------------------------------------

@testset "GC7: spectral_density under a dual in a g, against central differences" begin
    g0 = _gc_with(GaussianGrid(mass_kg = ELECTRON_MASS, T_ref_eV = 700), 0.2 .* randn(Xoshiro(22), 41))
    p0 = values(parameters(g0))
    k = 2.4e7
    oms = k .* (-2.5:0.5:3.0) .* sqrt(2 * 700 * EV / ELECTRON_MASS)
    S(g, om) = spectral_density(_gc_plasma(electrons(f = _gc_with(g0, g)), _gc_deuterons(T_eV = 250)), k, om)
    for idx in (7, 21, 25)
        ad = [ForwardDiff.derivative(t -> S(ntuple(i -> p0[i] + (i == idx ? t : 0.0), 41), om), 0.0) for om in oms]
        h = 1e-3    # spec D10: near ω = 0 the ion feature makes S large, and at h = 1e-4 the difference's
                    # round-off reaches 4e-6 of the largest partial; at 1e-3 the worst point reads 1.7e-7
        fd = [(S(ntuple(i -> p0[i] + (i == idx ? h : 0.0), 41), om) - S(ntuple(i -> p0[i] - (i == idx ? h : 0.0), 41), om)) / (2h)
              for om in oms]
        floor = 1e-6 * maximum(abs, fd)
        for (a, b) in zip(ad, fd)
            @test isapprox(a, b; rtol = 1e-6, atol = floor)
        end
    end
end

_gc_consume(p) = _gc_value(p.electrons[1].T) + _gc_value(p.electrons[1].u) + _gc_value(p.n_e)
_gc_value(x::Real) = x
_gc_value(x::ForwardDiff.Dual) = ForwardDiff.value(x) + sum(ForwardDiff.partials(x))
_gc_alloc(tr, j) = (_gc_consume(plasma(tr, j)); @allocated _gc_consume(plasma(tr, j)))

@testset "GC8: plasma(trace, j): T and u follow g, allocation-free, plain and dual" begin
    p = _gc_plasma(electrons(grid = (T_ref_eV = 700,)), _gc_deuterons(T_eV = 250))
    tr = PlasmaTrace(p; t_ns = [0.0, 1.0, 2.0],
                     leaves = ("electrons.e.f.g20" => [0.0, 0.5, -0.5], "electrons.e.f.g23" => [0.0, 0.1, 0.4]))
    Ts = [plasma(tr, j).electrons[1].T for j in 1:3]
    @test Ts[1] == p.electrons[1].T
    @test Ts[2] != Ts[1] && Ts[3] != Ts[2]
    for j in 1:3
        e = plasma(tr, j).electrons[1]
        @test e.T == ELECTRON_MASS * velocity_variance(e.f)
        @test e.u == mean_velocity(e.f)
        @test (e.u, e.T) == _distribution_u_T(e.f, ELECTRON_MASS, NaN, NaN)
    end
    @test _gc_alloc(tr, 2) == 0
    for N in (8, 44)
        D(x, i) = ForwardDiff.Dual{:GC8}(x, ntuple(j -> j == i ? 1.0 : 0.0, N))
        trd = PlasmaTrace(p; t_ns = [0.0, 1.0, 2.0],
                          leaves = ("electrons.e.f.g20" => [D(0.0, 1), D(0.5, 1), D(-0.5, 1)],
                                    "electrons.e.f.g23" => [D(0.0, 2), D(0.1, 2), D(0.4, 2)],
                                    "n_e_cm3" => [D(5e19, 3), D(5e19, 3), D(5e19, 3)]))
        @test _gc_alloc(trd, 2) == 0
        e = plasma(trd, 2).electrons[1]
        @test ForwardDiff.value(e.T) == Ts[2]
        dT = ForwardDiff.derivative(t -> ELECTRON_MASS * velocity_variance(_gc_with(p.electrons[1].f,
                 ntuple(i -> i == 20 ? 0.5 + t : i == 23 ? 0.1 : 0.0, 41))), 0.0)
        @test isapprox(ForwardDiff.partials(e.T)[1], dT; rtol = 1e-12)
    end
end

@testset "GC9: rule 5.15: the derivative in a gridded ion's fraction at exactly zero is finite" begin
    k = 2.4e7
    om = k * 3.0e5
    function S(x)
        e = electrons(T_eV = 700)
        a = ion(name = "D", mass_amu = 2.014, charge = 1, grid = (T_ref_eV = 250,), fraction = x)
        b = ion(name = "H", mass_amu = 1.008, charge = 1, T_eV = 250, fraction = 1 - x)
        return spectral_density(Plasma(n_e_cm3 = 5e19, electrons = (e,), ions = (a, b)), k, om)
    end
    d0 = ForwardDiff.derivative(S, 0.0)
    @test isfinite(d0)
    @test isapprox(d0, ForwardDiff.derivative(S, 1e-10); rtol = 1e-6)
end

# ------------------------------------------------------------------------------------------
# GC10-GC12: the refusals, show, the file writer (OQ-68 item 5; spec D9, D11, D13)
# ------------------------------------------------------------------------------------------

@testset "GC10: the refusals" begin
    g = GaussianGrid(mass_kg = ELECTRON_MASS, T_ref_eV = 700)
    @test_throws "species \"e\" carries a lab-frame distribution (GaussianGrid): its T and u are computed from the distribution and cannot be given; got (:T_eV,)" electrons(f = g, T_eV = 700)
    @test_throws "species \"e\" carries a lab-frame distribution (GaussianGrid): its T and u are computed from the distribution and cannot be given; got (:u_kms,)" electrons(grid = (T_ref_eV = 700,), u_kms = 10)
    @test_throws "species \"D\" carries a lab-frame distribution (GaussianGrid): its T and u are computed from the distribution and cannot be given; got (:T_keV, :u_ms)" _gc_deuterons(grid = (T_ref_eV = 250,), T_keV = 0.25, u_ms = 0)
    @test_throws "unrecognised keyword(s) (:Te,); accepted keywords are (:name, :f, :fraction, :grid)" electrons(f = g, Te = 1)
    @test_throws "species \"e\": give f or grid, not both" electrons(f = g, grid = (T_ref_eV = 700,))
    @test_throws "species \"e\": f must be a distribution (an AbstractVelocityDistribution or an AbstractLabDistribution); got Int64" electrons(f = 3, T_eV = 700)
    @test_throws "species \"e\": grid must be a NamedTuple of GaussianGrid's temperature-form keywords (T_ref_eV or T_ref_keV, u0_kms or u0_ms, nodes, spacing, width, symmetric); got Float64" electrons(grid = 700.0)
    @test_throws "species \"e\": grid is GaussianGrid's temperature form and needs T_ref_eV or T_ref_keV; for an explicit spacing pass f = GaussianGrid(dv_kms = ..., default_weights = ...)" electrons(grid = (nodes = 41,))
    @test_throws "species \"D\": the grid takes the species' own mass; mass_amu or mass_kg is not a grid keyword here" _gc_deuterons(grid = (T_ref_eV = 250, mass_amu = 2.014))
    @test_throws "mass must be positive; got 0.0 kg for species \"D\"" ion(name = "D", mass_kg = 0.0, charge = 1, grid = (T_ref_eV = 250,))
    @test_throws "GaussianGrid: spacing · width must be below 1" electrons(grid = (T_ref_eV = 700, spacing = 0.9, width = 1.2))
    p = _gc_plasma(electrons(grid = (T_ref_eV = 700,)), _gc_deuterons(grid = (T_ref_eV = 250,)))
    @test_throws "\"electrons.e.T\": species \"e\" carries a lab-frame distribution (GaussianGrid); its T and u are computed from the distribution and are not leaves" PlasmaTrace(p; t_ns = [0.0, 1.0], leaves = ("electrons.e.T_eV" => [700, 710],))
    @test_throws "\"ions.D.u\": species \"D\" carries a lab-frame distribution (GaussianGrid); its T and u are computed from the distribution and are not leaves" PlasmaTrace(p; t_ns = [0.0, 1.0], leaves = ("ions.D.u_kms" => [0, 10],))
    @test_throws "\"ions.D.T\": species \"D\" carries a lab-frame distribution (GaussianGrid)" GradientSmearing(p; leaf = "ions.D.T_eV", width = 10.0)
    tr = PlasmaTrace(p; t_ns = [0.0, 1.0], leaves = ("electrons.e.fraction" => [1.0, 1.0], "ions.D.f.g7" => [0.0, 0.3]))
    @test plasma(tr, 2).ions[1].f.g[7] == 0.3
end

@testset "GC11: show" begin
    p = _gc_plasma(electrons(grid = (T_ref_eV = 700,)), _gc_deuterons(T_eV = 250))
    se = sprint(show, MIME"text/plain"(), p.electrons[1])
    @test occursin("  T        = 7.000000e+02 eV (computed from f)\n", se)
    @test occursin("km/s (computed from f)\n", se)
    @test occursin("  f        = GaussianGrid(41 nodes, u0 = 0.0 km/s, dv = 3923.0 km/s, width = 1.2)\n", se)
    @test !occursin("computed", sprint(show, MIME"text/plain"(), p.ions[1]))
end

@testset "GC12: write_trace refuses a lab-frame distribution before opening the file" begin
    p = _gc_plasma(electrons(T_eV = 700), _gc_deuterons(grid = (T_ref_eV = 250,)))
    tr = PlasmaTrace(p; t_ns = [0.0, 1.0])
    path = joinpath(mktempdir(), "gc12.jld2")
    @test_throws "write_trace: species \"D\" carries a lab-frame distribution (GaussianGrid), which the trace layout \"trust-trace-2\" cannot hold; nothing was written" write_trace(path, tr)
    @test !isfile(path)
end
