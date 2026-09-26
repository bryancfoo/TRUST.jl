# Step 2.3a: the acceptance test for the geometry split and the dielectric hoist (spec D39,
# Section 6a). RF1's whole claim is that the refactor changes nothing; RF2 checks the hoist
# actually halves the dielectric cost it claims to; RF3 checks the three-argument
# spectral_density wrapper survives bitwise. RF4 (allocations unchanged) is TR14, EF9, XP13,
# TH5 re-run in their own files, not repeated here.

using Test
using OTIS

include(joinpath(@__DIR__, "..", "fixtures", "rf1_reference.jl"))

# --- RF1: the D22 regression case, 512 bins ---

@testset "RF1a: the D22 regression spectrum is bitwise unchanged" begin
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

# --- RF1: the three D24 cut-down (128 x 32) problems, ideal_streak and measure ---
# Built directly from D24's profiles (examples/problems.jl does not exist until Step 2.5).

function _rf_epw_ablation(; nlambda=128, nt=32)
    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    t_ns = collect(range(2.0, 5.6; length=nt))
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    m_of_t(t) = 2.0 + 2.9 * exp(-((t - 4.4) / 1.15)^2)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t, "electrons.e.f.m" => m_of_t))
    lambda = collect(range(220e-9, 260e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
               ConstantBackground(level=25.0))
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
end

function _rf_iaw_multiflow(; nlambda=128, nt=32)
    template = Plasma(n_e_cm3=3e19, electrons=(electrons(T_eV=400.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=200, fraction=0.95),
              ion(name="C", mass_amu=12.011, charge=6, T_eV=200, fraction=0.05)))
    t_ns = collect(range(0.0, 3.0; length=nt))
    n_e_of_t(t) = 3e19 * (10 / 3)^(t / 3.0)
    fD_of_t(t) = 0.95 - 0.45 * (t / 3.0)^2
    fC_of_t(t) = 1 - fD_of_t(t)
    uC_of_t(t) = 500 * exp(-t / 0.90)
    TD_of_t(t) = 200 + 400 * t
    TC_of_t(t) = 200 + 800 * (t / 3.0)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "ions.D.fraction" => fD_of_t, "ions.C.fraction" => fC_of_t,
                "ions.C.u_kms" => uC_of_t, "ions.D.T_eV" => TD_of_t, "ions.C.T_eV" => TC_of_t))
    lambda = collect(range(262.0e-9, 264.5e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6))
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
end

function _rf_iaw_shock(; nlambda=128, nt=32)
    template = Plasma(n_e_cm3=6e19, electrons=(electrons(T_eV=500.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=60, fraction=0.78),
              ion(name="C", mass_amu=12.011, charge=6, T_eV=100, fraction=0.22)))
    t_ns = collect(range(0.0, 4.0; length=nt))
    ts = 1.15
    TD_of_t(t) = t < ts ? 60.0 : 60 + 380 * exp(-(t - ts) / 1.60)
    TC_of_t(t) = t < ts ? 100.0 : 100 + 1450 * exp(-(t - ts) / 0.85)
    fD_of_t(t) = t < ts ? 0.78 : 0.63
    fC_of_t(t) = 1 - fD_of_t(t)
    u_of_t(t) = t < ts ? 250.0 : 40.0
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("ions.D.T_eV" => TD_of_t, "ions.C.T_eV" => TC_of_t,
                "ions.D.fraction" => fD_of_t, "ions.C.fraction" => fC_of_t,
                "ions.D.u_kms" => u_of_t, "ions.C.u_kms" => u_of_t))
    lambda = collect(range(262.0e-9, 264.5e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6))
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
end

@testset "RF1b: the three D24 cut-down problems are bitwise unchanged" begin
    # measure() applies SpectralIRF/TemporalIRF via mul! (BLAS gemm!); its reassociation under
    # more than one BLAS thread is not bitwise-reproducible run to run, which is exactly why
    # TH3 pins BLAS to one thread before comparing measure results. The pinned reference was
    # generated the same way.
    # The *_MEASURE arrays match only under Pkg.test()'s --check-bounds=yes (Phase 4.A chunk 10,
    # Level 3's finding, reproduced by Level 2): under the default flag they differ in up to 1 557
    # entries at up to 8.1e-16, so re-pin them under --check-bounds=yes.
    prev = pin_blas_threads(1)
    try
        cases = (("EPW", _rf_epw_ablation, EPW_IDEAL, EPW_MEASURE),
                 ("MULTIFLOW", _rf_iaw_multiflow, MULTIFLOW_IDEAL, MULTIFLOW_MEASURE),
                 ("SHOCK", _rf_iaw_shock, SHOCK_IDEAL, SHOCK_MEASURE))
        for (name, builder, ideal_ref, measure_ref) in cases
            expt = builder()
            P = ideal_streak(expt)
            M = measure(expt)
            @test all(P .=== ideal_ref)
            @test all(M .=== measure_ref)
        end
    finally
        pin_blas_threads(prev)
    end
end

# --- RF2: dielectric is called exactly once per (bin, wavelength) ---
# Counting `cauchy_dfdv` calls is equivalent: `susceptibility` (which `dielectric` calls once
# per species) calls `cauchy_dfdv` exactly once, so a 1:1 correspondence holds without touching
# src/ (rule 8.2: the reference never shares machinery with the code under test).

mutable struct _CountingShape{F} <: OTIS.AbstractVelocityDistribution
    inner::F
    counter::Base.RefValue{Int}
end
(w::_CountingShape)(x) = w.inner(x)
OTIS.dfdx(w::_CountingShape, x) = OTIS.dfdx(w.inner, x)
OTIS.moment(w::_CountingShape, n::Integer) = OTIS.moment(w.inner, n)
function OTIS.cauchy_dfdv(w::_CountingShape, xi::Number)
    w.counter[] += 1
    return OTIS.cauchy_dfdv(w.inner, xi)
end
OTIS.parameters(w::_CountingShape) = OTIS.parameters(w.inner)
OTIS.reconstruct(w::_CountingShape, v::NamedTuple) = _CountingShape(OTIS.reconstruct(w.inner, v), w.counter)

@testset "RF2: dielectric is called exactly once per (bin, wavelength)" begin
    counter = Ref(0)
    p = Plasma(n_e_cm3=5e19,
        electrons=(electrons(T_eV=500, f=_CountingShape(Maxwellian(), counter)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250, f=_CountingShape(Maxwellian(), counter)),))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=64))
    spectrum(p, probe, collection, lambda)
    @test counter[] == 2 * 64
end

# --- RF3: the three-argument spectral_density is bitwise the four-argument one ---

@testset "RF3: the 3-argument spectral_density agrees with the 4-argument form" begin
    p = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    lambda = collect(range(235e-9, 255e-9; length=64))
    for lam in lambda
        k, omega, k_s = wavevector(probe, collection, lam, p)
        d = dielectric(p, k, omega)
        S3 = spectral_density(p, k, omega)
        S4 = spectral_density(p, k, omega, d)
        @test S3 === S4
    end
end
