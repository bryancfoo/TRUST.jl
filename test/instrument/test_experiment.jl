using Test
using OTIS
using OTIS: _bin_kernel!, _make_cache, BinCache, _streak_eltype, _convolution_matrix
import ForwardDiff

_xp_plasma(; f_electron=Maxwellian()) = Plasma(
    n_e_cm3=5e19,
    electrons=(electrons(T_eV=500, f=f_electron),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))
_xp_probe() = ProbeBeam(lambda_probe_nm=263.25)
_xp_collection() = Collection(theta_deg=60, k_model=:exact)

const _XP_LAMBDA = collect(range(235e-9, 255e-9; length=64))

@testset "XP1: no effects, one bin, equals spectrum" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA)
    M = measure(expt)
    Pref = spectrum(template, probe, collection, _XP_LAMBDA)
    @test M[:, 1] == Pref
end

@testset "XP2: no effects, many bins" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    nt = 16
    t_ns = collect(range(0.0, 3.0; length=nt))
    Te_vec = collect(range(400.0, 600.0; length=nt))
    n_e_vec = collect(range(5e19, 6e19; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T_eV" => Te_vec, "n_e_cm3" => n_e_vec))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA)
    M = measure(expt)
    for j in 1:nt
        Pref = spectrum(plasma(trace, j), probe, collection, _XP_LAMBDA)
        @test M[:, j] == Pref
    end
end

@testset "XP3: the time axis" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    t_ns = collect(range(0.0, 3.0; length=8))
    trace = PlasmaTrace(template; t_ns=t_ns)
    expt1 = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA)
    @test expt1.t === trace.axis

    expt2 = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, t_ns=t_ns)
    @test expt2.t === trace.axis

    bad_t = copy(t_ns)
    bad_t[3] += 0.5
    @test_throws "disagrees with the trace's axis at bin" Experiment(trace=trace, probe=probe, collection=collection,
        lambda_m=_XP_LAMBDA, t_ns=bad_t)
end

@testset "XP4: axes(expt)" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=8)))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA)
    lam, t = axes(expt)
    @test lam === expt.lambda
    @test t === expt.t
    @test length.(axes(expt)) == size(measure(expt))
end

_chain_order_wrap(e) = OTIS._chain_order(e)

@testset "XP5: canonical order" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=4)))
    effs = (Sensitivity(scale=2.0), ConstantBackground(level=1.0), KSmearing(half_angle_deg=1.0),
        TemporalIRF(fwhm_ns=0.05), SpectralIRF(fwhm_nm=0.5))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=effs)
    orders = map(_chain_order_wrap, expt.effects)
    @test issorted(orders)
    n1 = count(==(1), orders)
    @test all(==(1), orders[1:n1])
end

@testset "XP6: order independence" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=4)))
    e1, e2, e3 = KSmearing(half_angle_deg=1.0, nodes=3), SpectralIRF(fwhm_nm=0.5), Sensitivity(scale=2.0)
    e4 = TemporalIRF(fwhm_ns=0.05)
    e5, e6 = ConstantBackground(level=1.0), PolynomialBackground(coefficients=(1.0,))
    orderings = ((e1, e2, e3, e4, e5, e6), (e6, e5, e4, e3, e2, e1), (e3, e1, e5, e2, e6, e4))
    streaks = [measure(Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=ord)) for ord in orderings]
    @test streaks[1] == streaks[2]
    @test streaks[1] == streaks[3]
end

@testset "XP7: gain before background" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA,
        effects=(Sensitivity(scale=3.0), ConstantBackground(level=25.0)))
    M = measure(expt)
    P = spectrum(template, probe, collection, _XP_LAMBDA)
    @test isapprox(M[:, 1], 3 .* P .+ 25; rtol=1e-14)
    @test !isapprox(M[:, 1], 3 .* (P .+ 25); rtol=1e-14)
end

@testset "XP8: the whole chain against its formula" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))
    lambda64 = collect(range(235e-9, 255e-9; length=64))
    trace = PlasmaTrace(template; t_ns=t_ns)
    e_k = KSmearing(half_angle_deg=1.5, nodes=3)
    e_spec = SpectralIRF(fwhm_nm=0.5)
    e_temp = TemporalIRF(fwhm_ns=0.1)
    e_gain = Sensitivity(scale=2.0)
    e_poly = PolynomialBackground(coefficients=(1.0, 0.5))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda64,
        effects=(e_k, e_spec, e_temp, e_gain, e_poly))
    M = measure(expt)

    Pkernel = zeros(64, nt)
    for j in 1:nt
        pj = plasma(trace, j)
        for (w, pq, probeq, collectionq, lambdaq) in samples(e_k, pj, probe, collection, lambda64)
            Pkernel[:, j] .+= w .* spectrum(pq, probeq, collectionq, lambdaq)
        end
    end
    Klam = _convolution_matrix(e_spec.kernel, lambda64)
    Kt = _convolution_matrix(e_temp.kernel, t_ns .* 1e-9)
    B = OTIS._prepare(e_poly, lambda64, t_ns .* 1e-9)
    oracle = B .+ e_gain.scale .* (Klam * Pkernel * transpose(Kt))
    @test isapprox(M, oracle; rtol=1e-12)
end

@testset "XP9: duplicate unique effects" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    @test_throws "only one SpectralIRF is allowed" Experiment(trace=trace, probe=probe, collection=collection,
        lambda_m=_XP_LAMBDA, effects=(SpectralIRF(fwhm_nm=0.5), SpectralIRF(fwhm_nm=0.3)))
    @test_throws "only one TemporalIRF is allowed" Experiment(trace=trace, probe=probe, collection=collection,
        lambda_m=_XP_LAMBDA, effects=(TemporalIRF(fwhm_ns=0.05), TemporalIRF(fwhm_ns=0.03)))
    @test_throws "only one Sensitivity is allowed" Experiment(trace=trace, probe=probe, collection=collection,
        lambda_m=_XP_LAMBDA, effects=(Sensitivity(scale=1.0), Sensitivity(scale=2.0)))
end

@testset "XP10: the element type" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0, 1.0])
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(Sensitivity(scale=1.0),))
    @test _streak_eltype(expt) == Float64

    dual_gain = ForwardDiff.Dual(1.0, 1.0)
    expt_g = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(Sensitivity(scale=dual_gain),))
    @test _streak_eltype(expt_g) <: ForwardDiff.Dual
    @test eltype(measure(expt_g)) <: ForwardDiff.Dual

    n_e_vec = [5e19 + 1e18 * ForwardDiff.Dual(0.0, 1.0), 5e19]
    trace_dual = PlasmaTrace(template; t_ns=[0.0, 1.0], leaves=("n_e_cm3" => n_e_vec,))
    expt_t = Experiment(trace=trace_dual, probe=probe, collection=collection, lambda_m=_XP_LAMBDA)
    @test eltype(measure(expt_t)) <: ForwardDiff.Dual

    probe_dual = ProbeBeam(lambda_probe_m=probe.lambda + ForwardDiff.Dual(0.0, 1.0) * 1e-9)
    expt_pw = Experiment(trace=trace, probe=probe_dual, collection=collection, lambda_m=_XP_LAMBDA)
    @test eltype(measure(expt_pw)) <: ForwardDiff.Dual
end

@testset "XP11: AD in the gain" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    h = 1e-6
    f(x) = sum(measure(Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(Sensitivity(scale=x),))))
    ad = ForwardDiff.derivative(f, 2.0)
    fd = (f(2.0 + h) - f(2.0 - h)) / (2h)
    @test isapprox(ad, fd; rtol=1e-6, atol=1e-6 * abs(fd))
end

@testset "XP12: AD in a spectral FWHM" begin
    # sum(measure(...)) is nearly invariant to the IRF width (the convolution conserves total
    # power to ~1e-14 relative, EF7), so a derivative test on the sum is dominated by round-off
    # noise at any reasonable step. A single pixel near the peak is genuinely sensitive to the
    # width and gives a well-posed finite difference.
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    h = 1e-4
    P0 = spectrum(template, probe, collection, _XP_LAMBDA)
    peak_idx = argmax(P0)
    function f(x)
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(SpectralIRF(fwhm_m=x),))
        return measure(expt)[peak_idx, 1]
    end
    x0 = 0.5e-9
    ad = ForwardDiff.derivative(f, x0)
    fd = (f(x0 * (1 + h)) - f(x0 * (1 - h))) / (2 * h * x0)
    @test isapprox(ad, fd; rtol=1e-4, atol=1e-4 * abs(fd))
end

@testset "XP13: allocations in the per-bin kernel" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=[0.0, 1.0])
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA)
    P = Matrix{Float64}(undef, length(_XP_LAMBDA), 2)
    cache = _make_cache(Float64, length(_XP_LAMBDA))
    _bk!() = _bin_kernel!(P, expt, 1, cache)
    _bk!()
    @test (@allocated _bk!()) == 0

    dual_gain = ForwardDiff.Dual(1.0, 1.0)
    expt_dual = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(Sensitivity(scale=dual_gain),))
    DFT = _streak_eltype(expt_dual)
    Pd = Matrix{DFT}(undef, length(_XP_LAMBDA), 2)
    cached = _make_cache(DFT, length(_XP_LAMBDA))
    _bkd!() = _bin_kernel!(Pd, expt_dual, 1, cached)
    _bkd!()
    @test (@allocated _bkd!()) == 0

    expt_k = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(KSmearing(half_angle_deg=1.0, nodes=5),))
    cachek = _make_cache(Float64, length(_XP_LAMBDA))
    _bkk!() = _bin_kernel!(P, expt_k, 1, cachek)
    _bkk!()
    @test (@allocated _bkk!()) == 0
end

@testset "XP14: show" begin
    template = _xp_plasma()
    probe = _xp_probe(); collection = _xp_collection()
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=4)))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_XP_LAMBDA, effects=(Sensitivity(scale=2.0),))
    s1 = sprint(show, expt)
    @test occursin("Experiment", s1)
    s2 = sprint(io -> show(io, MIME("text/plain"), expt))
    @test occursin("Experiment", s2)
    @test occursin("Sensitivity", s2)
end
