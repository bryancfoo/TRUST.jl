using Test
using OTIS
using OTIS: _convective_gain, GAIN_SERIES_CUTOFF, SIN_THETA_MIN
import ForwardDiff

include(joinpath(@__DIR__, "..", "fixtures", "rf1_reference.jl"))

# The D22 regression plasma and geometry (m = 3), reused from RF1a.
_cg_probe(; intensity_wcm2=0.0, diameter_um=0.0) =
    ProbeBeam(lambda_probe_nm=263.25, intensity_wcm2=intensity_wcm2, diameter_um=diameter_um)
_cg_collection() = Collection(theta_deg=60, k_model=:exact)
_cg_plasma() = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
    ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
const CG_LAMBDA = collect(range(235e-9, 255e-9; length=512))

@testset "CG1: gain = :none is bitwise today" begin
    p = _cg_plasma()
    probe = _cg_probe()
    collection = _cg_collection()
    P = spectrum(p, probe, collection, CG_LAMBDA; gain=:none)
    @test all(P .=== D22_SPECTRUM_M3_0)
end

@testset "CG2: |a-bar_0|^2 = a0_sq * polarization_factor" begin
    for psi_deg in (0.0, 20.0, 45.0, 90.0), theta_deg in (30.0, 60.0, 154.2)
        probe = ProbeBeam(lambda_probe_nm=263.25, polarization_deg=psi_deg, intensity_wcm2=1e15,
            diameter_um=100.0)
        collection = Collection(theta_deg=theta_deg)
        psi = theta_to_si(psi_deg, :theta_deg)
        theta = theta_to_si(theta_deg, :theta_deg)
        oracle = sin(psi)^2 * cos(theta)^2 + cos(psi)^2
        @test isapprox(polarization_factor(probe, collection), oracle; rtol=1e-15)
        @test isapprox(probe.a0_sq * polarization_factor(probe, collection), probe.a0_sq * oracle; rtol=1e-15)
    end
end

@testset "CG3: the expm1 crossover is continuous" begin
    # Directly exercise the two branches of the formula (series vs expm1) at the crossover,
    # independent of any particular plasma: g(gl) as _convective_gain would compute it.
    g_of_gl(gl) = abs(gl) < GAIN_SERIES_CUTOFF ? 1 + gl / 2 : expm1(gl) / gl
    for gl in (GAIN_SERIES_CUTOFF * 0.999, GAIN_SERIES_CUTOFF * 1.001,
        -GAIN_SERIES_CUTOFF * 0.999, -GAIN_SERIES_CUTOFF * 1.001)
        series = 1 + gl / 2
        exact = expm1(gl) / gl
        @test isapprox(series, exact; rtol=1e-14)
    end
    # g -> 1 as gl -> 0: the exact mathematical limit, at gl = 0 itself (not a small but
    # nonzero gl, which legitimately gives g = 1 + gl/2 != 1).
    @test isapprox(g_of_gl(0.0), 1.0; rtol=1e-15)
end

# The gain rate per unit a0_sq at one (k, omega), for calibrating a target |gamma*L| in CG4/CG6.
function _gamma_per_a0sq(p, probe, collection, lambda_s)
    k, omega, k_s = wavevector(probe, collection, lambda_s, p)
    d = dielectric(p, k, omega)
    F_chi = d.chi_e * (d.epsilon - d.chi_e) / d.epsilon
    # The minus sign matches _convective_gain's (spec D43, corrected 2026-09-11; gain.md
    # Section 6): Turnbull's Eq. 6 translated into this package's omega convention.
    return -k^2 / (4k_s) * imag(F_chi) * polarization_factor(probe, collection)
end

const CG_LAMBDA_BLUE = collect(range(256e-9, 262.5e-9; length=64))
const CG_LAMBDA_RED = collect(range(264.0e-9, 270e-9; length=64))

@testset "CG4: the sign -- amplification red of the probe, depletion blue" begin
    p = _cg_plasma()
    collection = _cg_collection()
    lam_i = lambda_to_si(263.25, :lambda_nm)
    diameter = 100e-6
    theta = collection.theta

    # Calibrate an intensity giving |gamma*L| ~ 0.1 at a wavelength red of the probe --
    # deliberately above the user's own OMEGA-probe number (which gives |gamma*L| ~ 0.03,
    # docs/derivations/gain.md Section 5), so the sign is measured cleanly.
    lam_red = CG_LAMBDA_RED[1]
    gpa_red = _gamma_per_a0sq(p, _cg_probe(diameter_um=100.0), collection, lam_red)
    L = diameter / sin(theta)
    target_gl = 0.1
    a0_sq_needed = target_gl / abs(gpa_red * L)
    intensity_needed = a0_sq_needed / (OTIS.A0_SQ_COEFFICIENT * lam_i^2)
    intensity_wcm2 = intensity_needed / 1e4

    probe = _cg_probe(intensity_wcm2=intensity_wcm2, diameter_um=100.0)
    @info "CG4 calibrated probe" intensity_wcm2 a0_sq = probe.a0_sq

    P_none_red = spectrum(p, probe, collection, CG_LAMBDA_RED; gain=:none)
    P_gain_red = spectrum(p, probe, collection, CG_LAMBDA_RED; gain=:convective)
    P_none_blue = spectrum(p, probe, collection, CG_LAMBDA_BLUE; gain=:none)
    P_gain_blue = spectrum(p, probe, collection, CG_LAMBDA_BLUE; gain=:convective)
    g_red = P_gain_red ./ P_none_red
    g_blue = P_gain_blue ./ P_none_blue
    @info "CG4 measured gain" g_red_range = extrema(g_red) g_blue_range = extrema(g_blue)
    @test all(>(1), g_red)
    @test all(<(1), g_blue)
end

@testset "CG5: :weak agrees with :convective at small |gamma L|" begin
    # A narrow window around one wavelength, so |gamma*L| stays close to the calibrated target
    # across it (Im(F_chi) varies noticeably over CG_LAMBDA_RED's wider span).
    p = _cg_plasma()
    collection = _cg_collection()
    lam_i = lambda_to_si(263.25, :lambda_nm)
    lam_red = 264.0e-9
    lambdas = collect(range(lam_red - 0.05e-9, lam_red + 0.05e-9; length=16))
    diameter = 100e-6
    gpa = _gamma_per_a0sq(p, _cg_probe(diameter_um=100.0), collection, lam_red)
    L = diameter / sin(collection.theta)
    target_gl = 0.05
    a0_sq_needed = target_gl / abs(gpa * L)
    intensity_needed = a0_sq_needed / (OTIS.A0_SQ_COEFFICIENT * lam_i^2)
    probe = _cg_probe(intensity_wcm2=intensity_needed / 1e4, diameter_um=100.0)

    P_conv = spectrum(p, probe, collection, lambdas; gain=:convective)
    P_weak = spectrum(p, probe, collection, lambdas; gain=:weak)
    worst = maximum(abs.(P_conv .- P_weak) ./ abs.(P_conv))
    @info "CG5 :weak vs :convective, worst relative diff" worst
    @test worst < 2e-3
end

@testset "CG6: the gain grows with probe intensity, monotonically" begin
    p = _cg_plasma()
    collection = _cg_collection()
    peaks = Float64[]
    for I_wcm2 in (2.1e14, 4.5e14, 7.0e14, 9.2e14)
        probe = _cg_probe(intensity_wcm2=I_wcm2, diameter_um=100.0)
        P = spectrum(p, probe, collection, CG_LAMBDA_RED; gain=:convective)
        push!(peaks, maximum(P))
    end
    @info "CG6 Stokes peak vs intensity" peaks
    @test issorted(peaks)
    @test allunique(peaks)
end

@testset "CG7: errors" begin
    p = _cg_plasma()
    collection = _cg_collection()
    lam = collect(range(235e-9, 255e-9; length=8))

    probe_no_a0 = _cg_probe(diameter_um=100.0)
    @test_throws "needs the probe's intensity" spectrum(p, probe_no_a0, collection, lam; gain=:convective)

    probe_no_d = _cg_probe(intensity_wcm2=1e15)
    @test_throws "needs the probe's beam diameter" spectrum(p, probe_no_d, collection, lam; gain=:convective)

    probe_ok = _cg_probe(intensity_wcm2=1e15, diameter_um=100.0)
    collection_0 = Collection(theta_deg=0, k_model=:approximate)
    @test_throws "sin(theta) is below SIN_THETA_MIN" OTIS._convective_gain(
        :convective, (epsilon=1.0 + 0im, chi_e=0.1 + 0.01im), 1e7, 1e7, probe_ok, collection_0)

    @test_throws "spectrum: gain must be :none, :convective or :weak" spectrum(
        p, probe_ok, collection, lam; gain=:nonsense)
end

@testset "CG8: allocations and AD" begin
    p = _cg_plasma()
    collection = _cg_collection()
    probe = _cg_probe(intensity_wcm2=1e15, diameter_um=100.0)
    lam = lambda_to_si(250.0, :lambda_nm)
    _sa() = OTIS._spectrum_at(p, probe, collection, lam, true, :convective)
    _sa()
    @test (@allocated _sa()) == 0

    lam_i = lambda_to_si(263.25, :lambda_nm)
    lam_red = lam_i * 1.02
    h = 1e-5
    function stokes_peak_of_Te(Te)
        pp = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=Te, f=SuperGaussian(3.0)),),
            ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
        return OTIS._spectrum_at(pp, probe, collection, lam_red, true, :convective)
    end
    ad = ForwardDiff.derivative(stokes_peak_of_Te, 500.0)
    fd = (stokes_peak_of_Te(500.0 * (1 + h)) - stokes_peak_of_Te(500.0 * (1 - h))) / (2h * 500.0)
    floor_ = 1e-6 * abs(fd)
    @info "CG8 AD check" ad fd floor_
    @test isapprox(ad, fd; rtol=1e-6, atol=floor_)
end

# ------------------------------------------------------------------------------------------
# CG9 (Phase 4.D spec D15): the gain's missing-input checks compare primal values
# ------------------------------------------------------------------------------------------

@testset "CG9: the gain's missing-input checks compare primal values (Phase 4.D spec D15)" begin
    # Both checks come before _convective_gain reads its dielectric argument, so none is built.
    collection = Collection(theta_deg=60, k_model=:exact)
    for s in (1.0, -1.0)
        probe_i = ProbeBeam(lambda_probe_nm=263.25, intensity_wm2=ForwardDiff.Dual(0.0, s), diameter_um=100.0)
        @test_throws "a0_sq == 0" OTIS._convective_gain(:convective, nothing, 1.0, 1.0, probe_i, collection)
        probe_d = ProbeBeam(lambda_probe_nm=263.25, intensity_wcm2=1e15, diameter_m=ForwardDiff.Dual(0.0, s))
        @test_throws "diameter == 0" OTIS._convective_gain(:convective, nothing, 1.0, 1.0, probe_d, collection)
    end
end
