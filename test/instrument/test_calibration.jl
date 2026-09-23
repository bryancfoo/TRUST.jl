using Test
using OTIS
using OTIS: _CalibratedAxis, _chain_class, _convolution_matrix
import ForwardDiff

# WC1 to WC12 against dev/phase4A_instructions.md Section 5's Test table (spec D3-D5, D8), gated
# "wavelength calibration". WavelengthCalibration replaces the Phase 2/3 probe-wavelength-shift
# effect at Step 4.A.1a (2026-09-21, OQ-24): a kernel-stage effect perturbing the axis the kernel
# evaluates the spectrum at, never the probe and never the detector's own operators.

_wc_plasma(; f_electron=Maxwellian()) = Plasma(
    n_e_cm3=5e19,
    electrons=(electrons(T_eV=500, f=f_electron),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))

_wc_probe() = ProbeBeam(lambda_probe_nm=263.25)
_wc_collection() = Collection(theta_deg=60, k_model=:exact)

const _WC_LAMBDA = collect(range(235e-9, 255e-9; length=128))

@testset "WC1: construction and validation" begin
    e1 = WavelengthCalibration(lambda_offset_nm=0.02)
    e2 = WavelengthCalibration(lambda_offset_m=0.02 * 1e-9)
    @test e1.offset == e2.offset
    @test e1.offset == 0.02 * 1e-9

    @test_throws "more than one unit keyword given for lambda_offset" WavelengthCalibration(
        lambda_offset_nm=0.02, lambda_offset_m=0.02 * 1e-9)
    @test_throws "no unit keyword given for lambda_offset" WavelengthCalibration()

    e3 = WavelengthCalibration(lambda_offset_nm=0.0)
    @test e3.dispersion_scale == 0.0

    @test_throws "offset must be finite" WavelengthCalibration(lambda_offset_nm=Inf)
    @test_throws "offset must be finite" WavelengthCalibration(lambda_offset_nm=NaN)
    @test_throws "dispersion_scale must be finite" WavelengthCalibration(
        lambda_offset_nm=0.0, dispersion_scale=NaN)
    @test_throws "dispersion_scale must satisfy" WavelengthCalibration(
        lambda_offset_nm=0.0, dispersion_scale=1.0)
    @test_throws "dispersion_scale must satisfy" WavelengthCalibration(
        lambda_offset_nm=0.0, dispersion_scale=-1.0)
end

@testset "WC2: the traits" begin
    e = WavelengthCalibration(lambda_offset_nm=0.02, dispersion_scale=0.01)
    @test stage(e) == :kernel
    @test OTIS._chain_order(e) == 1
    @test islinear(e) == false
    @test _chain_class(e) == :WavelengthCalibration
end

@testset "WC3: parameters / reconstruct" begin
    e = WavelengthCalibration(lambda_offset_nm=0.02, dispersion_scale=0.01)
    @test keys(parameters(e)) == (:offset, :dispersion_scale)
    r = reconstruct(e, parameters(e))
    @test r.offset == e.offset
    @test r.dispersion_scale == e.dispersion_scale

    d = ForwardDiff.Dual(e.offset, 1.0)
    rd = reconstruct(e, (offset=d, dispersion_scale=e.dispersion_scale))
    @test rd isa WavelengthCalibration{<:ForwardDiff.Dual}
end

@testset "WC4: the mirror of EF12" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    delta_nm = 0.05
    e = WavelengthCalibration(lambda_offset_nm=delta_nm)
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA, effects=(e,))
    P1 = ideal_streak(expt)[:, 1]
    Pref = spectrum(template, probe, collection, _WC_LAMBDA .+ delta_nm * 1e-9)
    maxreldiff = maximum(abs.(P1 .- Pref) ./ abs.(Pref))
    @info "WC4: bitwise, max relative difference" maxreldiff
    @test all(P1 .=== Pref)
end

@testset "WC5: the dispersion scale" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    eps = 0.01
    e = WavelengthCalibration(lambda_offset_nm=0.0, dispersion_scale=eps)
    expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA, effects=(e,))
    P1 = ideal_streak(expt)[:, 1]
    lref = (first(_WC_LAMBDA) + last(_WC_LAMBDA)) / 2
    lambda_eval = lref .+ (1 + eps) .* (_WC_LAMBDA .- lref)
    Pref = spectrum(template, probe, collection, lambda_eval)
    maxreldiff = maximum(abs.(P1 .- Pref) ./ abs.(Pref))
    @info "WC5: max relative difference (not bitwise; the two forms sequence the same " *
          "arithmetic in a different order)" maxreldiff
    # Not bitwise: `lref .+ (1+eps).*(lambda.-lref)` and `lambda .+ eps.*(lambda.-lref)` are the
    # same real number but different floating-point operation sequences. The spec's stated
    # rtol=4e-16 (WC4's tolerance) does not hold here -- measured maxreldiff is ~1.76e-14 on this
    # fixture, about 44x looser than 4e-16 but three orders tighter than the 1e-12 the spec says
    # not to fall back to. Reported as a finding (chunk 3 report, "WC4/WC5's bitwise-or-not
    # answer"); not silently loosened to 1e-12.
    @test isapprox(P1, Pref; rtol=3e-14)
end

@testset "WC6: lambda_ref is the axis midpoint and nothing else" begin
    template = _wc_plasma()
    probe = _wc_probe()
    trace = PlasmaTrace(template; t_ns=[0.0])
    # A Collection with a real dispersion whose reference pixel (1) is not the window's middle
    # pixel (32.5 of 64), so lambda_center sits far from the axis midpoint (spec D5's point 3).
    collection = Collection(theta_deg=60, k_model=:exact, lambda_center_nm=263.0,
        dispersion_nm_px=0.02, npixels=64, reference_pixel=1)
    axis = detector_axis(collection)
    lref = (first(axis) + last(axis)) / 2
    @test abs(lref - collection.lambda_center) / lref > 1e-3   # genuinely different anchors

    eps = 0.01
    e = WavelengthCalibration(lambda_offset_nm=0.0, dispersion_scale=eps)
    expt = Experiment(trace=trace, probe=probe, collection=collection, effects=(e,))
    P1 = ideal_streak(expt)[:, 1]

    lambda_eval_ref = lref .+ (1 + eps) .* (axis .- lref)
    Pref_ref = spectrum(template, probe, collection, lambda_eval_ref)
    @test isapprox(P1, Pref_ref; rtol=3e-14)

    lc = collection.lambda_center
    lambda_eval_lc = lc .+ (1 + eps) .* (axis .- lc)
    Pref_lc = spectrum(template, probe, collection, lambda_eval_lc)
    @test maximum(abs.(P1 .- Pref_lc) ./ abs.(Pref_lc)) > 1e-6
end

@testset "WC7: a uniform shift commutes with a column-normalized IRF" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    delta_nm = 0.05
    fwhm_nm = 0.5
    e = WavelengthCalibration(lambda_offset_nm=delta_nm)
    expt_a = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA,
        effects=(e, SpectralIRF(fwhm_nm=fwhm_nm)))
    Ma = measure(expt_a)[:, 1]
    expt_b = Experiment(trace=trace, probe=probe, collection=collection,
        lambda_m=_WC_LAMBDA .+ delta_nm * 1e-9, effects=(SpectralIRF(fwhm_nm=fwhm_nm),))
    Mb = measure(expt_b)[:, 1]

    # Excludes the 4 pixels nearest each edge (8 of 128), where the column renormalization of
    # _convolution_matrix (spec D13) truncates the Gaussian tail against a different window on
    # the two axes. Measured on this fixture the whole array (edges included) agreed to
    # round-off (maxreldiff 0.0), so rtol=1e-12 is deliberately loose here by round-off
    # standards, exactly as the spec anticipates.
    n = length(Ma)
    interior = 5:(n-4)
    reldiff = abs.(Ma[interior] .- Mb[interior]) ./ abs.(Mb[interior])
    @info "WC7: max relative difference on the interior (4 pixels excluded per edge)" maxreldiff=maximum(reldiff)
    @test isapprox(Ma[interior], Mb[interior]; rtol=1e-12)
end

@testset "WC8: the operators do not move" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    e = WavelengthCalibration(lambda_offset_nm=0.05)
    expt_cal = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA,
        effects=(e, SpectralIRF(fwhm_nm=0.5)))
    expt_nocal = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA,
        effects=(SpectralIRF(fwhm_nm=0.5),))
    @test expt_cal.lambda === _WC_LAMBDA
    @test expt_nocal.lambda === _WC_LAMBDA

    op_cal = expt_cal.operators[findfirst(x -> x isa SpectralIRF, expt_cal.effects)]
    op_nocal = expt_nocal.operators[findfirst(x -> x isa SpectralIRF, expt_nocal.effects)]
    @test op_cal == op_nocal
end

@testset "WC9: AD through the sample construction" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    h = 1e-6

    function sum_with_offset(x)
        e = WavelengthCalibration(lambda_offset_m=x)
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA, effects=(e,))
        return sum(ideal_streak(expt))
    end
    x0 = 0.05e-9
    hx = h * 1e-9
    ad = ForwardDiff.derivative(sum_with_offset, x0)
    fd = (sum_with_offset(x0 + hx) - sum_with_offset(x0 - hx)) / (2hx)
    @test isapprox(ad, fd; rtol=1e-6, atol=1e-6 * abs(fd))

    function sum_with_dispersion(x)
        e = WavelengthCalibration(lambda_offset_nm=0.05, dispersion_scale=x)
        expt = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=_WC_LAMBDA, effects=(e,))
        return sum(ideal_streak(expt))
    end
    x02 = 0.01
    ad2 = ForwardDiff.derivative(sum_with_dispersion, x02)
    fd2 = (sum_with_dispersion(x02 + h) - sum_with_dispersion(x02 - h)) / (2h)
    @test isapprox(ad2, fd2; rtol=1e-6, atol=1e-6 * abs(fd2))
end

@testset "WC10: show" begin
    e = WavelengthCalibration(lambda_offset_nm=0.02)
    s = sprint(show, e)
    @test occursin("WavelengthCalibration", s)
    @test occursin("0.020", s)
    @test occursin("dispersion_scale", s)
end

@testset "WC11: the sample path allocates zero" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    e = WavelengthCalibration(lambda_offset_nm=0.05)
    _sp() = samples(e, template, probe, collection, _WC_LAMBDA)
    _sp()
    @test (@allocated _sp()) == 0
end

@testset "WC12: two WavelengthCalibrations in one chain is a loud error" begin
    template = _wc_plasma()
    probe = _wc_probe(); collection = _wc_collection()
    trace = PlasmaTrace(template; t_ns=[0.0])
    @test_throws "only one WavelengthCalibration is allowed" Experiment(trace=trace, probe=probe,
        collection=collection, lambda_m=_WC_LAMBDA,
        effects=(WavelengthCalibration(lambda_offset_nm=0.02), WavelengthCalibration(lambda_offset_nm=0.03)))
end

@testset "WC13: reconstruct is concretely typed under a dual, and still validates (D25)" begin
    e = WavelengthCalibration(lambda_offset_nm=0.02, dispersion_scale=0.01)
    d1 = ForwardDiff.Dual(e.offset, 1.0)
    d2 = ForwardDiff.Dual(e.dispersion_scale, 1.0)
    @inferred reconstruct(e, (offset=d1, dispersion_scale=d2))

    @test_throws "offset must be finite" reconstruct(e, (offset=Inf, dispersion_scale=0.0))
    @test_throws "offset must be finite" reconstruct(e, (offset=NaN, dispersion_scale=0.0))
    @test_throws "dispersion_scale must be finite" reconstruct(e, (offset=0.0, dispersion_scale=NaN))
    @test_throws "dispersion_scale must satisfy" reconstruct(e, (offset=0.0, dispersion_scale=1.0))
    @test_throws "dispersion_scale must satisfy" reconstruct(e, (offset=0.0, dispersion_scale=-1.0))
end
