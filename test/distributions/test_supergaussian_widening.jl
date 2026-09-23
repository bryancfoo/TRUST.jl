# Step 3.1a (chunk 2): D14's carried Phase 2 item (OQ-31 item 6). Copy-widening the
# super-Gaussian table slice, instead of _widen_shape's generic re-derivation via reconstruct.
# conventions.md 8.1: this adds tests to a component Phase 1.SG and Phase 2 both already cover,
# so it gets its own file.

using Test
using OTIS
using OTIS: _widen_shape, _sg_uncached, SG_TABLE
import ForwardDiff
using ForwardDiff: Dual

include(joinpath(@__DIR__, "..", "fixtures", "rf1_reference.jl"))

# --- WD1: the copy is a copy ---

@testset "WD1: the copy is a copy" begin
    f = SuperGaussian(3.0)
    g = _widen_shape(f, Dual{:WD1,Float64,1})

    @test ForwardDiff.value(g.m) === f.m
    @test ForwardDiff.value(g.x0) === f.x0
    @test ForwardDiff.value(g.C) === f.C
    @test all(ForwardDiff.value.(g.slice) .=== f.slice)

    @test iszero(ForwardDiff.partials(g.m)[1])
    @test iszero(ForwardDiff.partials(g.x0)[1])
    @test iszero(ForwardDiff.partials(g.C)[1])
    @test all(iszero(ForwardDiff.partials(s)[1]) for s in g.slice)

    # Trap 1: L = 0 (the slice-free form the table's generator builds) must widen without
    # special-casing -- NTuple{0,FT}(()) is the empty tuple.
    f0 = _sg_uncached(3.0)
    @test f0.slice === ()
    g0 = _widen_shape(f0, Dual{:WD1L0,Float64,1})
    @test g0.slice === ()
    @test ForwardDiff.value(g0.m) === f0.m
    @test ForwardDiff.value(g0.x0) === f0.x0
    @test ForwardDiff.value(g0.C) === f0.C
end

# --- WD2: a derivative through the copy-widened path agrees with central differences ---
# This is the property that matters (D14): if the copy silently dropped or corrupted a value,
# the *value* the derivative is taken of would be wrong and this is the row that would catch it.
# The path exercised is exactly GA2's: GradientSmearing on a super-Gaussian template, samples(),
# which calls _with_leaf -> _widen_plasma -> _widen_shape on the smeared species.

function _wd_gs_functional(template, probe, collection, lambda, e0, w)
    e = reconstruct(e0, (width=w,))
    total = zero(w)
    for (weight, p2, probe2, collection2, lambdaq) in samples(e, template, probe, collection, lambda)
        total += weight * sum(spectrum(p2, probe2, collection2, lambdaq))
    end
    return total
end

@testset "WD2: a derivative through the copy-widened path vs central differences" begin
    template = Plasma(n_e_cm3=5e19,
        electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=64))

    for width_ev in (30.0, 60.0)
        e0 = GradientSmearing(template; leaf="electrons.e.T_eV", width=width_ev, nodes=5)
        w0 = e0.width   # SI: reconstruct's `width` is parameters(e0).width, already SI
        f(w) = _wd_gs_functional(template, probe, collection, lambda, e0, w)

        ad = ForwardDiff.derivative(f, w0)
        h = 1e-6 * w0
        fd = (f(w0 + h) - f(w0 - h)) / (2h)
        floor = 1e-6 * abs(fd)
        @test isapprox(ad, fd; rtol=1e-6, atol=floor)
    end
end

# --- WD3: cauchy_dfdv is unchanged through the copy-widened path ---

@testset "WD3: cauchy_dfdv is unchanged through the copy-widened path" begin
    f = SuperGaussian(3.0)
    g = _widen_shape(f, Dual{:WD3,Float64,1})

    edges = SG_TABLE.edges
    npanels = length(edges) - 1
    idxs = unique(round.(Int, range(1, npanels; length=5)))
    xis = [(edges[i] + edges[i+1]) / 2 for i in idxs]

    for xi in xis
        plain = cauchy_dfdv(f, xi)
        widened = cauchy_dfdv(g, xi)
        @test isapprox(ForwardDiff.value(real(widened)), real(plain); rtol=1e-15, atol=1e-15)
        @test isapprox(ForwardDiff.value(imag(widened)), imag(plain); rtol=1e-15, atol=1e-15)
    end
end

# --- WD4: the regression net still holds, bitwise ---
# RF1a (test/scattering/test_refactor.jl) and GA2 (test/instrument/test_effects.jl), re-run here
# against the same pinned test/fixtures/rf1_reference.jl arrays: nothing on the value path
# changed, so anything but bitwise agreement is a finding, not a tolerance to relax.

@testset "WD4a: RF1a's D22 regression spectrum is still bitwise unchanged" begin
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=512))
    refs = Dict(2.0 => D22_SPECTRUM_M2_0, 3.0 => D22_SPECTRUM_M3_0, 5.0 => D22_SPECTRUM_M5_0)
    for m in (2.0, 3.0, 5.0)
        p = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500, f=SuperGaussian(m)),),
            ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
        P = spectrum(p, probe, collection, lambda)
        @test all(P .=== refs[m])
    end
end

@testset "WD4b: GA2's rewrite check is still bitwise unchanged" begin
    template = Plasma(n_e_cm3=5e19,
        electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=512))
    refs = Dict(1 => GA2_COLUMN_N1, 5 => GA2_COLUMN_N5, 7 => GA2_COLUMN_N7)
    for nodes in (1, 5, 7)
        e = GradientSmearing(template; leaf="electrons.e.T_eV", width=60.0, nodes=nodes)
        col = zeros(512)
        for (w, p2, probe2, collection2, lambdaq) in samples(e, template, probe, collection, lambda)
            col .+= w .* spectrum(p2, probe2, collection2, lambdaq)
        end
        @test all(col .=== refs[nodes])
    end
end
