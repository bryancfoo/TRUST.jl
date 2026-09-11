using Test
using OTIS
using OTIS: _resolve_leaf, _leaf_value, _with_leaf, _sigma_from_fwhm, _fwhm_from_sigma,
    _convolution_matrix, GAUSS_LEGENDRE_NODES, GAUSS_LEGENDRE_WEIGHTS,
    GAUSS_HERMITE_NODES, GAUSS_HERMITE_WEIGHTS, _bin_kernel!, _make_cache, _streak_eltype
import ForwardDiff
using Random

include(joinpath(@__DIR__, "..", "fixtures", "rf1_reference.jl"))

# TR's reference plasma (test/plasma/test_trace.jl / the D22 template).
_ef_plasma(; f_electron=Maxwellian()) = Plasma(
    n_e_cm3=5e19,
    electrons=(electrons(T_eV=500, f=f_electron),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))

_ef_probe() = ProbeBeam(lambda_probe_nm=263.25)
_ef_collection() = Collection(theta_deg=60, k_model=:exact)

const _EF_LAMBDA = collect(range(235e-9, 255e-9; length=128))
const _EF_T = collect(range(2.0e-9, 3.0e-9; length=16))

@testset "EF1: the Gaussian matrix against a hand-written double loop" begin
    sigma = _sigma_from_fwhm(0.5e-9)
    K = _convolution_matrix(GaussianKernel(sigma), _EF_LAMBDA)
    N = length(_EF_LAMBDA)
    Kref = zeros(N, N)
    for lp in 1:N
        for l in 1:N
            Kref[l, lp] = exp(-(_EF_LAMBDA[l] - _EF_LAMBDA[lp])^2 / (2 * sigma^2))
        end
        Kref[:, lp] ./= sum(Kref[:, lp])
    end
    @test isapprox(K, Kref; rtol=1e-12)
    for lp in 1:N
        @test isapprox(sum(K[:, lp]), 1.0; atol=1e-14)
    end
end

@testset "EF2: a delta in the interior returns the kernel" begin
    K = _convolution_matrix(GaussianKernel(_sigma_from_fwhm(0.5e-9)), _EF_LAMBDA)
    P = zeros(128, 16)
    P[64, 8] = 1.0
    M = K * P
    @test M[:, 8] == K[:, 64]
end

@testset "EF3: the quadrature tables" begin
    # n = 1 is the trivial identity (a single node at xi = 0, weight 1, spec D11): it is not a
    # genuine quadrature rule, so the variance identity sum(w*xi^2) = 1/3 or 1/2 -- which no
    # single point at the origin can satisfy -- is checked only for the real rules, n = 3, 5, 7.
    for n in (1, 3, 5, 7)
        key = Symbol("n$n")
        xiL, wL = getfield(GAUSS_LEGENDRE_NODES, key), getfield(GAUSS_LEGENDRE_WEIGHTS, key)
        xiH, wH = getfield(GAUSS_HERMITE_NODES, key), getfield(GAUSS_HERMITE_WEIGHTS, key)
        mid = (n + 1) ÷ 2
        @test xiL[mid] == 0.0
        @test xiH[mid] == 0.0
        @test issorted(xiL)
        @test issorted(xiH)
        @test isapprox(sum(wL), 1.0; atol=1e-15)
        @test isapprox(sum(wH), 1.0; atol=1e-15)
        if n > 1
            @test isapprox(sum(wL[k] * xiL[k]^2 for k in 1:n), 1 / 3; atol=1e-14)
            @test isapprox(sum(wH[k] * xiH[k]^2 for k in 1:n), 1 / 2; atol=1e-14)
        end
    end
end

@testset "EF4: a one-sample KSmearing is the identity" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    e = KSmearing(half_angle_deg=3.0, nodes=1)
    P_ref = spectrum(template, probe, collection, _EF_LAMBDA)
    samp = samples(e, template, probe, collection)
    @test length(samp) == 1
    w, p2, probe2, collection2 = samp[1]
    @test w == 1.0
    @test spectrum(p2, probe2, collection2, _EF_LAMBDA) == P_ref
end

@testset "EF5: a five-sample KSmearing equals its own definition" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    e = KSmearing(half_angle_deg=2.0, nodes=5)
    xi, w = GAUSS_LEGENDRE_NODES.n5, GAUSS_LEGENDRE_WEIGHTS.n5
    half_angle = theta_to_si(2.0, :theta_deg)
    wsum = sum(w)
    oracle = zeros(length(_EF_LAMBDA))
    for q in 1:5
        cq = Collection(theta_rad=collection.theta + half_angle * xi[q], k_model=collection.k_model)
        oracle .+= (w[q] / wsum) .* spectrum(template, probe, cq, _EF_LAMBDA)
    end
    P = zeros(length(_EF_LAMBDA))
    for (wq, p2, probe2, collection2) in samples(e, template, probe, collection)
        P .+= wq .* spectrum(p2, probe2, collection2, _EF_LAMBDA)
    end
    @test isapprox(P, oracle; rtol=1e-13)
end

@testset "EF6: the flat field, interior and edge" begin
    # sigma/dlam is only ~1.35 bins here, so the truncated tail at exactly 8 sigma (~11 bins) is
    # still ~1.7e-11, not yet at round-off; it falls by ~40x per further bin (measured), so the
    # margin below is 12 sigma, comfortably past round-off while still well inside "farther than
    # 8 sigma from either end".
    sigma = _sigma_from_fwhm(0.5e-9)
    K = _convolution_matrix(GaussianKernel(sigma), _EF_LAMBDA)
    result = [sum(view(K, l, :)) for l in 1:128]
    dlam = _EF_LAMBDA[2] - _EF_LAMBDA[1]
    margin = ceil(Int, 12 * sigma / dlam)
    for l in (margin+1):(128-margin)
        @test isapprox(result[l], 1.0; atol=1e-14)
    end
    @test result[1] < 1.0
end

@testset "EF7: the IRFs conserve the total" begin
    Klam = _convolution_matrix(GaussianKernel(_sigma_from_fwhm(0.5e-9)), _EF_LAMBDA)
    Kt = _convolution_matrix(GaussianKernel(_sigma_from_fwhm(0.05e-9)), _EF_T)
    rng = Random.Xoshiro(20260910)
    P = rand(rng, 128, 16) .+ 0.1
    @test isapprox(sum(Klam * P), sum(P); rtol=1e-14)
    @test isapprox(sum(P * transpose(Kt)), sum(P); rtol=1e-14)
end

@testset "EF8: a zero-width GradientSmearing is the identity" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    e1 = GradientSmearing(template; leaf="electrons.e.T", width=0.0, nodes=5)
    for (w, p2, probe2, collection2) in samples(e1, template, probe, collection)
        @test p2.electrons[1].T === template.electrons[1].T
        @test probe2 === probe
        @test collection2 === collection
    end
    e2 = GradientSmearing(template; leaf="n_e", width=0.0, nodes=5)
    for (w, p2, probe2, collection2) in samples(e2, template, probe, collection)
        @test p2.n_e === template.n_e
    end
end

@testset "EF9: allocations in the sample path" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    ref = _resolve_leaf(template, LeafPath("electrons.e.T"))
    v = _leaf_value(template, ref)
    _wl() = _with_leaf(template, ref, v)
    _wl()
    @test (@allocated _wl()) == 0

    eK = KSmearing(half_angle_deg=2.0, nodes=5)
    _sk() = samples(eK, template, probe, collection)
    _sk()
    @test (@allocated _sk()) == 0

    eG = GradientSmearing(template; leaf="electrons.e.T", width=60.0 * EV, nodes=5)
    _sg() = samples(eG, template, probe, collection)
    _sg()
    @test (@allocated _sg()) == 0

    eP = ProbeOffset(probe_offset_nm=0.05)
    _sp() = samples(eP, template, probe, collection)
    _sp()
    @test (@allocated _sp()) == 0
end

@testset "EF10: AD through the sample construction" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    h = 1e-6

    function sum_with_half_angle(x)
        e = KSmearing(half_angle_rad=x, nodes=5)
        trace = PlasmaTrace(template; t_ns=[0.0])
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_EF_LAMBDA, effects=(e,))
        return sum(ideal_streak(expt))
    end
    x0 = theta_to_si(2.0, :theta_deg)
    ad = ForwardDiff.derivative(sum_with_half_angle, x0)
    fd = (sum_with_half_angle(x0 + h) - sum_with_half_angle(x0 - h)) / (2h)
    @test isapprox(ad, fd; rtol=1e-6, atol=1e-6 * abs(fd))

    function sum_with_width(x)
        e = GradientSmearing(template; leaf="electrons.e.T", width=x, nodes=5)
        trace = PlasmaTrace(template; t_ns=[0.0])
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_EF_LAMBDA, effects=(e,))
        return sum(ideal_streak(expt))
    end
    w0 = 60.0 * EV
    ad2 = ForwardDiff.derivative(sum_with_width, w0)
    fd2 = (sum_with_width(w0 * (1 + h)) - sum_with_width(w0 * (1 - h))) / (2 * h * w0)
    @test isapprox(ad2, fd2; rtol=1e-6, atol=1e-6 * abs(fd2))

    function sum_with_offset(x)
        e = ProbeOffset(probe_offset_m=x)
        trace = PlasmaTrace(template; t_ns=[0.0])
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_EF_LAMBDA, effects=(e,))
        return sum(ideal_streak(expt))
    end
    o0 = 0.05e-9
    ad3 = ForwardDiff.derivative(sum_with_offset, o0)
    fd3 = (sum_with_offset(o0 + h * 1e-9) - sum_with_offset(o0 - h * 1e-9)) / (2 * h * 1e-9)
    @test isapprox(ad3, fd3; rtol=1e-6, atol=1e-6 * abs(fd3))
end

@testset "EF11: a smeared density stays positive" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    e = GradientSmearing(template; leaf="n_e", width=0.5, nodes=7)
    for (w, p2, probe2, collection2) in samples(e, template, probe, collection)
        @test p2.n_e > 0
        @test p2.electrons[1].n > 0
        @test p2.ions[1].n > 0
    end
end

@testset "EF12: ProbeOffset moves the probe, not the axis" begin
    template = _ef_plasma()
    probe = _ef_probe(); collection = _ef_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    # A 2 nm probe offset, well over one wavelength bin (~0.157 nm here), so the shifted
    # feature is guaranteed to land in a different bin regardless of the spectrum's shape.
    e = ProbeOffset(probe_offset_nm=2.0)
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_EF_LAMBDA, effects=(e,))
    @test expt.lambda === _EF_LAMBDA
    P0 = spectrum(template, probe, collection, _EF_LAMBDA)
    P1 = ideal_streak(expt)[:, 1]
    i0, i1 = argmax(P0), argmax(P1)
    @test abs(_EF_LAMBDA[i1] - _EF_LAMBDA[i0]) > (_EF_LAMBDA[2] - _EF_LAMBDA[1])
end

@testset "EF13: the tabulated kernel" begin
    axis = collect(range(0.0, 15.0; length=16))
    tk = TabulatedKernel((-1, 0, 1), (1.0, 2.0, 1.0))
    K = _convolution_matrix(tk, axis)
    N = 16
    Kref = zeros(N, N)
    for lp in 1:N
        for (o, wt) in zip((-1, 0, 1), (1.0, 2.0, 1.0))
            l = lp + o
            1 <= l <= N && (Kref[l, lp] += wt)
        end
        Kref[:, lp] ./= sum(Kref[:, lp])
    end
    @test isapprox(K, Kref; rtol=1e-14)

    nonuniform = [0.0, 1.0, 2.5, 4.0]
    @test_throws "a tabulated kernel needs a uniform axis" _convolution_matrix(tk, nonuniform)
end

@testset "EF14: islinear is honoured" begin
    rng = Random.Xoshiro(20260911)
    P = rand(rng, 64, 8) .+ 0.1
    Q = rand(rng, 64, 8) .+ 0.1
    a, b = 0.37, -1.9
    lambda64 = collect(range(235e-9, 255e-9; length=64))
    t8 = collect(range(2.0e-9, 3.0e-9; length=8))

    e_spec = SpectralIRF(fwhm_nm=0.5)
    op_spec = _convolution_matrix(e_spec.kernel, lambda64)
    lhs = apply!(similar(P), a .* P .+ b .* Q, e_spec, op_spec)
    rhs = a .* apply!(similar(P), P, e_spec, op_spec) .+ b .* apply!(similar(P), Q, e_spec, op_spec)
    @test isapprox(lhs, rhs; rtol=1e-14)

    e_temp = TemporalIRF(fwhm_ns=0.05)
    op_temp = _convolution_matrix(e_temp.kernel, t8)
    lhs2 = apply!(similar(P), a .* P .+ b .* Q, e_temp, op_temp)
    rhs2 = a .* apply!(similar(P), P, e_temp, op_temp) .+ b .* apply!(similar(P), Q, e_temp, op_temp)
    @test isapprox(lhs2, rhs2; rtol=1e-14)

    e_gain = Sensitivity(scale=3.0)
    lhs3 = apply!(copy(a .* P .+ b .* Q), a .* P .+ b .* Q, e_gain, nothing)
    rhs3 = a .* apply!(copy(P), P, e_gain, nothing) .+ b .* apply!(copy(Q), Q, e_gain, nothing)
    @test isapprox(lhs3, rhs3; rtol=1e-14)

    for e in (e_spec, e_temp, e_gain)
        @test islinear(e)
    end
end

@testset "EF15: islinear is honest where it is false" begin
    rng = Random.Xoshiro(20260912)
    P = rand(rng, 64, 8) .+ 0.1
    Q = rand(rng, 64, 8) .+ 0.1
    a, b = 0.37, -1.9

    e_const = ConstantBackground(level=25.0)
    lhs = apply!(copy(a .* P .+ b .* Q), a .* P .+ b .* Q, e_const, nothing)
    rhs = a .* apply!(copy(P), P, e_const, nothing) .+ b .* apply!(copy(Q), Q, e_const, nothing)
    @test isapprox(lhs .- rhs, fill((1 - a - b) * e_const.level, size(P)); rtol=1e-14)

    template = _ef_plasma()
    for e in (KSmearing(half_angle_deg=1.0), GradientSmearing(template; leaf="electrons.e.T", width=10.0),
        ProbeOffset(probe_offset_nm=0.1))
        @test stage(e) == :kernel
        @test islinear(e) == false
    end
end

@testset "EF16: FWHM to sigma, once" begin
    @test isapprox(_sigma_from_fwhm(1.0) * 2 * sqrt(2 * log(2)), 1.0; rtol=1e-15)
    @test isapprox(_fwhm_from_sigma(_sigma_from_fwhm(3.0)), 3.0; rtol=1e-15)

    src_dir = joinpath(@__DIR__, "..", "..", "src")
    found_in = String[]
    for (root, _, files) in walkdir(src_dir)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            if occursin("2.354", read(path, String))
                push!(found_in, path)
            end
        end
    end
    @test isempty(found_in)
end

@testset "EF17: every effect error of D28" begin
    template = _ef_plasma()
    @test_throws "nodes must be 1, 3, 5 or 7" KSmearing(half_angle_deg=1.0, nodes=2)
    @test_throws "half angle must be finite and non-negative" KSmearing(half_angle_deg=-1.0)
    @test_throws "nodes must be 1, 3, 5 or 7" GradientSmearing(template; leaf="electrons.e.T", width=1.0, nodes=2)
    @test_throws "a fraction cannot be gradient-smeared" GradientSmearing(template; leaf="electrons.e.fraction", width=1.0)
    @test_throws "a shape parameter cannot be gradient-smeared" GradientSmearing(
        _ef_plasma(f_electron=SuperGaussian(3.0)); leaf="electrons.e.f.m", width=1.0)
    @test_throws "smear the density as \"n_e\"" GradientSmearing(template; leaf="n_e_cm3", width=1.0)
    @test_throws "width must be finite and non-negative" GradientSmearing(template; leaf="electrons.e.T", width=-1.0)
    @test_throws "FWHM must be strictly positive" SpectralIRF(fwhm_nm=-0.5)
    @test_throws "FWHM must be strictly positive" TemporalIRF(fwhm_ns=0.0)
    @test_throws "offsets must be strictly increasing" TabulatedKernel((1, 0), (1.0, 2.0))
    @test_throws "offsets and weights must have the same length" TabulatedKernel((0, 1), (1.0,))
    tk = TabulatedKernel((0,), (1.0,))
    @test_throws "a tabulated kernel needs a uniform axis" _convolution_matrix(tk, [0.0, 1.0, 2.5])
end

@testset "EF18: parameters / reconstruct" begin
    template = _ef_plasma()

    e1 = KSmearing(half_angle_deg=2.0, nodes=5)
    r1 = reconstruct(e1, parameters(e1))
    @test r1.half_angle == e1.half_angle
    @test r1.offsets == e1.offsets
    e1b = reconstruct(e1, (half_angle=e1.half_angle * 2,))
    @test e1b.offsets != e1.offsets

    e2 = GradientSmearing(template; leaf="electrons.e.T", width=60.0 * EV, nodes=5)
    r2 = reconstruct(e2, parameters(e2))
    @test r2.width == e2.width
    @test r2.offsets == e2.offsets
    e2b = reconstruct(e2, (width=e2.width * 2,))
    @test e2b.offsets != e2.offsets
    @test e2b.ref === e2.ref

    e3 = ProbeOffset(probe_offset_nm=0.05)
    @test reconstruct(e3, parameters(e3)).offset == e3.offset

    e4 = SpectralIRF(fwhm_nm=0.5)
    @test reconstruct(e4, parameters(e4)).kernel.sigma == e4.kernel.sigma

    e5 = Sensitivity(scale=2.0)
    @test reconstruct(e5, parameters(e5)).scale == e5.scale
end

@testset "EF19: a dual FWHM gives a dual matrix" begin
    lambda32 = collect(range(235e-9, 255e-9; length=32))
    fwhm_dual = ForwardDiff.Dual(0.5e-9, 1.0)
    e_dual = SpectralIRF(fwhm_m=fwhm_dual)
    e_plain = SpectralIRF(fwhm_m=0.5e-9)
    K_dual = _convolution_matrix(e_dual.kernel, lambda32)
    K_plain = _convolution_matrix(e_plain.kernel, lambda32)
    @test eltype(K_dual) <: ForwardDiff.Dual
    @test isapprox(ForwardDiff.value.(K_dual), K_plain; rtol=1e-15)
end

@testset "EF20: show" begin
    template = _ef_plasma()
    @test occursin("SpectralIRF", sprint(show, SpectralIRF(fwhm_nm=0.5)))
    @test occursin("TemporalIRF", sprint(show, TemporalIRF(fwhm_ns=0.08)))
    @test occursin("KSmearing", sprint(show, KSmearing(half_angle_deg=2.0)))
    @test occursin("GradientSmearing", sprint(show, GradientSmearing(template; leaf="electrons.e.T", width=60.0)))
    @test occursin("ProbeOffset", sprint(show, ProbeOffset(probe_offset_nm=0.05)))
    @test occursin("Sensitivity", sprint(show, Sensitivity(scale=2.0e-4)))
    @test occursin("ConstantBackground", sprint(show, ConstantBackground(level=25.0)))
end

# --- Step 2.3a: the A15 allocation rewrite and the A17 leaf-key fix (spec Section 6a) ---

_ga_template() = Plasma(n_e_cm3=5e19,
    electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
    ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
_ga_probe() = ProbeBeam(lambda_probe_nm=263.25)
_ga_collection() = Collection(theta_deg=60, k_model=:exact)
const _GA_LAMBDA = collect(range(235e-9, 255e-9; length=512))

@testset "GA1: the A15 allocation is bounded by one plasma, whatever the node count" begin
    template = _ga_template()
    probe = _ga_probe(); collection = _ga_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    allocs = Dict{Int,Int}()
    for nodes in (1, 5, 7)
        e = GradientSmearing(template; leaf="electrons.e.T_eV", width=60.0, nodes=nodes)
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_GA_LAMBDA, effects=(e,))
        FT = _streak_eltype(expt)
        P = Matrix{FT}(undef, length(_GA_LAMBDA), 1)
        cache = _make_cache(FT, length(_GA_LAMBDA))
        _bk!() = _bin_kernel!(P, expt, 1, cache)
        _bk!()
        allocs[nodes] = @allocated _bk!()
    end
    @info "GA1 allocations by node count" allocs
    @test allocs[1] == allocs[5] == allocs[7]
    @test allocs[1] <= 2048
end

@testset "GA2: the A15 rewrite changes no number" begin
    template = _ga_template()
    probe = _ga_probe(); collection = _ga_collection()
    refs = Dict(1 => GA2_COLUMN_N1, 5 => GA2_COLUMN_N5, 7 => GA2_COLUMN_N7)
    for nodes in (1, 5, 7)
        e = GradientSmearing(template; leaf="electrons.e.T_eV", width=60.0, nodes=nodes)
        col = zeros(512)
        for (w, p2, probe2, collection2) in samples(e, template, probe, collection)
            col .+= w .* spectrum(p2, probe2, collection2, _GA_LAMBDA)
        end
        @test all(col .=== refs[nodes])
    end
end

@testset "GA3: the A17 correction: leaf takes a boundary key and width its unit" begin
    template = _ef_plasma()
    e1 = GradientSmearing(template; leaf="electrons.e.T_eV", width=60.0)
    e2 = GradientSmearing(template; leaf="electrons.e.T", width=60.0 * EV)
    @test e1.offsets === e2.offsets
    @test e1.weights === e2.weights
    @test e1.width === e2.width

    e3 = GradientSmearing(template; leaf="ions.D.u_kms", width=10.0)
    e4 = GradientSmearing(template; leaf="ions.D.u", width=10.0 * 1e3)
    @test e3.offsets === e4.offsets
    @test e3.width === e4.width
end

@testset "GA4: the density is still the exception" begin
    template = _ef_plasma()
    e = GradientSmearing(template; leaf="n_e", width=0.5)
    @test e.logarithmic
    @test_throws "smear the density as \"n_e\"" GradientSmearing(template; leaf="n_e_cm3", width=0.5)
end

@testset "SN1: Sensitivity(scale = g) is today's Gain(gain = g), bitwise" begin
    # The 128 x 32 EPW chain of D24, built with the new API: Sensitivity(scale = g) with no
    # curve must give exactly the streak the old Gain(gain = g) gave, pinned at 5e7b059.
    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    t_ns = collect(range(2.0, 5.6; length=32))
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    m_of_t(t) = 2.0 + 2.9 * exp(-((t - 4.4) / 1.15)^2)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t, "electrons.e.f.m" => m_of_t))
    lambda = collect(range(220e-9, 260e-9; length=128))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
               ConstantBackground(level=25.0))
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
    # measure() applies the IRFs via mul! (BLAS gemm!); pin BLAS to one thread for a
    # bitwise-reproducible comparison, as TH3 does and as the pinned reference was generated.
    prev = pin_blas_threads(1)
    M = try
        measure(expt)
    finally
        pin_blas_threads(prev)
    end
    @test all(M .=== EPW_MEASURE)
end
