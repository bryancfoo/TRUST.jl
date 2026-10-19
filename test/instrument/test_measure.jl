using Test
using TRUST
using TRUST: _bin_kernel!, _make_cache, _streak_eltype
import ForwardDiff
using LinearAlgebra.BLAS: get_num_threads

_th_probe() = ProbeBeam(lambda_probe_nm=263.25)
_th_collection() = Collection(theta_deg=60, k_model=:exact)

function _th_epw_problem(; nlambda=128, nt=32)
    template = Plasma(n_e_cm3=5e19,
        electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))
    t_ns = collect(range(2.0, 5.0; length=nt))
    n_e_vec = collect(range(1.5e19, 6e19; length=nt))
    Te_vec = collect(range(1500.0, 430.0; length=nt))
    m_vec = collect(range(2.0, 5.0; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_vec, "electrons.e.T_eV" => Te_vec, "electrons.e.f.m" => m_vec))
    lambda = collect(range(220e-9, 260e-9; length=nlambda))
    probe = _th_probe(); collection = _th_collection()
    effects = (SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
        ConstantBackground(level=25.0))
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
end

function _th_maxwellian_problem(; nlambda=128, nt=32)
    template = Plasma(n_e_cm3=5e19,
        electrons=(electrons(T_eV=500),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250, fraction=0.6),
            ion(name="C", mass_amu=12.011, charge=6, T_eV=250, fraction=0.4)))
    t_ns = collect(range(0.0, 3.0; length=nt))
    n_e_vec = collect(range(3e19, 1e20; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("n_e_cm3" => n_e_vec,))
    lambda = collect(range(262e-9, 264.5e-9; length=nlambda))
    probe = _th_probe(); collection = _th_collection()
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6)))
end

@testset "TH1: serial and threaded ideal_streak agree (EPW-like)" begin
    expt = _th_epw_problem()
    Pserial = ideal_streak(expt; scheduler=:serial)
    Pauto = ideal_streak(expt; scheduler=:auto)
    @test all(Pserial .=== Pauto)
end

@testset "TH2: serial and threaded ideal_streak agree (Maxwellian)" begin
    expt = _th_maxwellian_problem()
    Pserial = ideal_streak(expt; scheduler=:serial)
    Pauto = ideal_streak(expt; scheduler=:auto)
    @test all(Pserial .=== Pauto)
end

@testset "TH3: serial and threaded measure agree" begin
    prev = pin_blas_threads(1)
    try
        expt = _th_epw_problem()
        Mserial = measure(expt; scheduler=:serial)
        Mauto = measure(expt; scheduler=:auto)
        @test all(Mserial .=== Mauto)
    finally
        pin_blas_threads(prev)
    end
end

@testset "TH4: nothing indexes by thread id" begin
    src_dir = joinpath(@__DIR__, "..", "..", "src")
    found = String[]
    for (root, _, files) in walkdir(src_dir)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            occursin("threadid", read(path, String)) && push!(found, path)
        end
    end
    @test isempty(found)
end

@testset "TH5: the kernel still allocates nothing under the threaded path" begin
    expt = _th_epw_problem()
    FT = _streak_eltype(expt)
    P = Matrix{FT}(undef, length(expt.lambda), length(expt.t))
    cache = _make_cache(FT, length(expt.lambda))
    _bk!() = _bin_kernel!(P, expt, 1, cache)
    _bk!()
    @test (@allocated _bk!()) == 0
end

@testset "TH6: every bin is visited exactly once" begin
    counts_serial = zeros(Int, 10)
    TRUST.foreach_bin!(10, () -> nothing; scheduler=:serial) do j, _
        counts_serial[j] += 1
    end
    @test all(==(1), counts_serial)

    counts_auto = zeros(Int, 10)
    lk = Threads.SpinLock()
    TRUST.foreach_bin!(10, () -> nothing; scheduler=:auto) do j, _
        Threads.lock(lk) do
            counts_auto[j] += 1
        end
    end
    @test all(==(1), counts_auto)
end

@testset "TH7: the scheduler keyword" begin
    @test_throws "scheduler must be :auto or :serial" TRUST.foreach_bin!(4, () -> nothing; scheduler=:nonsense) do j, c
    end
end

@testset "TH8: pin_blas_threads" begin
    before = get_num_threads()
    prev = pin_blas_threads(1)
    @test prev == before
    @test get_num_threads() == 1
    pin_blas_threads(before)
    @test get_num_threads() == before
end
