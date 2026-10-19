# The progress lines' fixture (Phase 4.K spec D12), shared by test/inversion/test_progress.jl (PG1 to
# PG8, the fast tier) and test/slow/test_progress.jl (PG9, the slow tier): the bitwise comparison of two
# FitResults, the capture of what a call writes to stdout, the two regular expressions the lines are
# parsed with (spec D14), and the fast tier's problem, test_optimizers.jl's P2 (ten bins, 32
# wavelengths, T_D and u_D per bin, one D2 term on T_D) with a flat start from which LevenbergMarquardt
# rejects its first four trial steps.

# Printf's @sprintf through TRUST's own import (src/inversion/optimizers.jl): neither test environment
# declares Printf, so `using Printf` fails in Pkg.test()'s sandbox, and the environments are not edited.
using TRUST: @sprintf

# Every FitResult field OQ-61 (a) names, compared bitwise (=== on every Float64); `elapsed` excepted.
function _pg_test_bitwise(a::FitResult, b::FitResult)
    @test length(a.theta) == length(b.theta) && all(a.theta .=== b.theta)
    @test length(a.trajectory) == length(b.trajectory) &&
          all(all(values(x) .=== values(y)) for (x, y) in zip(a.trajectory, b.trajectory))
    @test a.objective === b.objective
    @test a.chi2 === b.chi2
    @test a.R === b.R
    @test length(a.term_values) == length(b.term_values) && all(a.term_values .=== b.term_values)
    @test a.status === b.status
    @test a.termination === b.termination
    @test a.converged === b.converged
    @test a.iterations === b.iterations
    @test a.f_calls === b.f_calls
    @test a.g_calls === b.g_calls
    return nothing
end

# What `f()` writes to stdout, through a temporary file (an IOBuffer cannot stand in for stdout), and
# what it returns.
function _pg_capture(f)
    return mktemp() do path, io
        value = redirect_stdout(f, io)
        close(io)
        (read(path, String), value)
    end
end

# The lines of a capture; every line, the last included, ends in "\n".
_pg_lines(text::AbstractString) = split(chomp(text), '\n')

# The two regular expressions (spec D14): a progress line in OQ-61 item 3's field order, with the
# LevenbergMarquardt and the explorer tails as alternatives, and the final line.
const _PG_LINE = r"^progress (?<name>levenberg_marquardt|lbfgs|noisy_descent): it=(?<it>\d+) chi2=(?<chi2>\S+) chi2/n=(?<chi2n>\S+) R=(?<R>\S+) obj=(?<obj>\S+) dchi2=(?<dchi2>\S+) rate=(?<rate>\S+) gmax=(?<gmax>\S+) nb=(?<nb>\d+)/(?<ntheta>\d+) t=(?<t>\d+\.\d)s(?: s/it=(?<sit>\d+\.\d\d))?(?: acc=(?<acc>\d+) rej=(?<rej>\d+) mu=(?<mu>\S+)| best=(?<best>\S+) noise=(?<noise>\S+))?$"
const _PG_DONE = r"^progress (?<name>levenberg_marquardt|lbfgs|noisy_descent): done status=(?<status>\w+) converged=(?<converged>true|false) it=(?<it>\d+) chi2=(?<chi2>\S+) chi2/n=(?<chi2n>\S+) obj=(?<obj>\S+) t=(?<t>\d+\.\d)s$"

# A capture with its two clock fields removed, so that two runs of one fit compare equal.
_pg_strip(text::AbstractString) = replace(text, r" t=\d+\.\ds( s/it=\d+\.\d\d)?" => "")

# P2 (test_optimizers.jl's, D46): ten bins, 32 wavelengths, a linear ramp in T_D and u_D, noise
# sigma, one D2 term on T_D at weight 1e-3.
function _pg_problem()
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=10)))
    expt = Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact),
        lambda_m=collect(range(262.0, 264.5; length=32)) .* 1e-9,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
    pspec = ParameterSpec(expt;
        variables=[FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true),
            FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
    nt = nbins(pspec.trace)
    theta_truth = flatten(pspec, ["T_D" => collect(range(220.0, 260.0; length=nt)),
        "u_D" => collect(range(-20.0, 20.0; length=nt))])
    image = measure(unflatten(working, pspec, theta_truth))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image,
        sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    term = RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)
    return FitProblem(working, pspec, data; regularizer=Regularizer([term]; weight=1e-3))
end

# The start: T_D = 80 eV and u_D = -250 km/s in every bin. From it LevenbergMarquardt rejects trial steps
# 1 to 4 and stops on :x_tolerance after an accepted step (10 rows, 13 attempted), LBFGS stops on
# :f_tolerance (37 rows) and a 40-step explorer runs its 41 rows, none of them warning (spec D12).
_pg_start(problem) = flatten(problem.pspec, ["T_D" => fill(80.0, 10), "u_D" => fill(-250.0, 10)])
