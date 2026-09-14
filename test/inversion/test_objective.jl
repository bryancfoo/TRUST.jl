using Test
using OTIS
using OTIS: _full_residuals, _objective_and_gradient!, _regularizer_value, _value_table, _near_bound
using REPL: REPL  # gives Base.Docs.doc its methods (empty otherwise); see OB7 below

# OB1 to OB8 against dev/phase3_instructions.md Section 7's OB table (spec D26), gated
# "objective". D46's P1, P2 and P3 are built here, not imported across the @safetestset boundary
# (P2 here also carries a dependent, "T_D_K", not part of D46's own recipe, so that OB5 has
# something to check "values holds every dependent too" against -- flagged under rule 11 zero).
#
# JT10 (chunk 6) was withdrawn 2026-09-13 as unsound on P2 (a dense J is only 2x its bound there)
# and replaced by OB8 below, stated instead on an EPW-scale problem where a dense J actually
# dominates (npixels >= 4000, ntheta >= 20) -- see the carried-item note in this chunk's prompt.

# ------------------------------------------------------------------------------------------
# shared fixtures -- D46's P1 and P3, plus a parameterized EPW-shaped problem used for P2 (OB1,
# OB2, OB3, OB4, OB5) and, at a larger size, for OB8.
# ------------------------------------------------------------------------------------------

_ob_perturb(theta0::AbstractVector, scale::Real=0.05) = theta0 .+ scale .* (1:length(theta0))

function _ob_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

function _ob_data(working::Experiment, pspec::ParameterSpec, theta::AbstractVector)
    truth = unflatten(working, pspec, theta)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    return StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig)
end

# --- P1 (D46): the one-bin problem, noise-free ---

function _ob_p1_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _ob_p1_expt()
    trace = PlasmaTrace(_ob_p1_template(); t_ns=[0.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=24)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
end

function _ob_p1_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0)
    v_u = FittedVariable("u_D"; bounds=(-200.0, 200.0), init=0.0)
    return ParameterSpec(expt; variables=[v_T, v_u],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

function _ob_p1_problem()
    expt = _ob_p1_expt()
    pspec = _ob_p1_pspec(expt)
    working = _ob_working(expt, pspec)
    theta0 = flatten(pspec)
    truth = unflatten(working, pspec, theta0)
    image = measure(truth)
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=ones(size(image)))
    problem = FitProblem(working, pspec, data)
    theta = _ob_perturb(theta0)
    return problem, theta
end

# --- P3 (D46): the oracle problem, every effect on ---

function _ob_p3_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0, f=SuperGaussian(2.5))],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=0.7),
              ion(name="C", mass_amu=12.0, charge=6, T_eV=200.0, u_kms=0.0, fraction=0.3)])
end

const _OB_P3_LAMBDA_NM = collect(range(262.9, 263.6; length=8))

function _ob_p3_expt()
    trace = PlasmaTrace(_ob_p3_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = _OB_P3_LAMBDA_NM .* 1e-9
    effects = (
        KSmearing(half_angle_deg=2.0, nodes=3),
        GradientSmearing(_ob_p3_template(); leaf="electrons.e.T_eV", width=30.0, nodes=3),
        ProbeOffset(probe_offset_nm=0.02),
        SpectralIRF(fwhm_nm=0.08),
        TemporalIRF(TabulatedKernel((-1, 0, 2), (0.2, 0.5, 0.3))),
        Sensitivity(scale=1.0e-6, lambda_nm=[262.0, 263.25, 264.5], response=[0.8, 1.0, 0.9]),
        RationalBackground(numerator=(20.0, 2.0), denominator=(0.3,)),
    )
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m, effects=effects)
end

function _ob_p3_pspec(expt::Experiment)
    ksmearing_idx = findfirst(e -> e isa KSmearing, expt.effects)
    spectral_idx = findfirst(e -> e isa SpectralIRF, expt.effects)
    sens_idx = findfirst(e -> e isa Sensitivity, expt.effects)
    rational_idx = findfirst(e -> e isa RationalBackground, expt.effects)

    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0, time_varying=true)
    v_fD = FittedVariable("f_D"; bounds=(0.3, 0.9), init=0.7)
    v_half = FittedVariable("half_angle"; bounds=(0.001, 0.2), init=2.0 * pi / 180)
    v_sig = FittedVariable("sig_spec"; bounds=(1e-11, 2e-10), init=3.5e-11)
    v_scale = FittedVariable("scale"; bounds=(1e-7, 1e-5), init=1e-6)
    v_numer = FittedVariable("numer0"; bounds=(5.0, 50.0), init=20.0)

    return ParameterSpec(expt;
        variables=[v_Te, v_fD, v_half, v_sig, v_scale, v_numer],
        bindings=["T_e" => "electrons.e.T_eV", "f_D" => "ions.D.fraction",
            "half_angle" => (ksmearing_idx, :half_angle),
            "sig_spec" => (spectral_idx, :sigma),
            "scale" => (sens_idx, :scale),
            "numer0" => (rational_idx, :numerator, 1)])
end

function _ob_p3_problem()
    expt = _ob_p3_expt()
    pspec = _ob_p3_pspec(expt)
    working = _ob_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _ob_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)
    theta = _ob_perturb(theta0, 0.02)
    return problem, theta
end

# --- the EPW-shaped problem: P2 at its D46 size (OB1-OB5), and larger for OB8 ---

function _ob_epw_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _ob_epw_expt(; nlambda::Int=32, nt::Int=10)
    trace = PlasmaTrace(_ob_epw_template(); t_ns=collect(range(0.0, 3.0; length=nt)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=nlambda)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
end

# `with_dependent` adds "T_D_K" = 11604.5 * T_D (a smooth, harmless dependent -- drawing none of
# conventions.md 5.9's one sanctioned warning) so OB5 has a dependent to find in `values`.
function _ob_epw_pspec(expt::Experiment; with_dependent::Bool=true)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    deps = with_dependent ? ["T_D_K" => "11604.5 * T_D"] : ()
    return ParameterSpec(expt; variables=[v_T, v_u], dependents=deps,
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

function _ob_p2_problem_plain()
    expt = _ob_epw_expt()
    pspec = _ob_epw_pspec(expt)
    working = _ob_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _ob_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)
    theta = _ob_perturb(theta0)
    return problem, theta
end

# P2 with a two-term L2 regularizer at Lambda = 0.25 (OB1's own conditions).
function _ob_p2_problem_with_reg()
    expt = _ob_epw_expt()
    pspec = _ob_epw_pspec(expt)
    working = _ob_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _ob_data(working, pspec, theta0)

    term_T = RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)
    term_u = RegularizerTerm(pspec; select=["u_D"], operator=:D1, normalization=:unit, weight=2.0)
    reg = Regularizer([term_T, term_u]; weight=0.25)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    theta = _ob_perturb(theta0)
    return problem, theta, reg
end

# ------------------------------------------------------------------------------------------
# OB1, OB2: objective's two identities (P2, two-term regularizer, Lambda = 0.25)
# ------------------------------------------------------------------------------------------

@testset "OB1: objective == chi2 + Lambda * R (P2, regularized)" begin
    problem, theta, reg = _ob_p2_problem_with_reg()
    o = objective(problem, theta)
    c2 = chi2(problem, theta)
    Rval = _regularizer_value(reg, problem.pspec, theta)
    expected = c2 + reg.weight * Rval
    @info "OB1" o expected c2 Rval reg.weight
    @test isapprox(o, expected; rtol=1e-14)
end

@testset "OB2: objective == norm(r_full)^2 (P2, regularized)" begin
    problem, theta, reg = _ob_p2_problem_with_reg()
    o = objective(problem, theta)
    r_full = _full_residuals(problem, theta)
    @test isapprox(o, sum(abs2, r_full); rtol=1e-14)
end

# ------------------------------------------------------------------------------------------
# OB3: the combined evaluator agrees with the separate calls, bitwise (P2, regularized)
# ------------------------------------------------------------------------------------------

@testset "OB3: _objective_and_gradient! agrees with objective and gradient! (P2, regularized)" begin
    problem, theta, reg = _ob_p2_problem_with_reg()
    ntheta = problem.pspec.ntheta

    o_sep = objective(problem, theta)
    g_sep = zeros(ntheta)
    gradient!(g_sep, problem, theta)

    g_comb = zeros(ntheta)
    o_comb = _objective_and_gradient!(g_comb, problem, theta)

    @test o_comb === o_sep
    @test all(g_comb .=== g_sep)
end

# ------------------------------------------------------------------------------------------
# OB4: the gradient against central finite differences (P1, P2, P3) -- D42's loose class
# ------------------------------------------------------------------------------------------

@testset "OB4: the gradient against finite differences (P1, P2, P3)" begin
    for (problem, theta) in (_ob_p1_problem(), _ob_p2_problem_plain(), _ob_p3_problem())
        ntheta = problem.pspec.ntheta
        g = zeros(ntheta)
        _objective_and_gradient!(g, problem, theta)
        h = 1e-6
        fd = zeros(ntheta)
        for k in 1:ntheta
            thp = copy(theta)
            thp[k] += h * max(abs(theta[k]), 1.0)
            thm = copy(theta)
            thm[k] -= h * max(abs(theta[k]), 1.0)
            step = thp[k] - thm[k]
            fd[k] = (objective(problem, thp) - objective(problem, thm)) / step
        end
        floor_k = 1e-6 * maximum(abs, fd)
        for k in 1:ntheta
            @test isapprox(g[k], fd[k]; rtol=1e-6, atol=max(1e-6, floor_k))
        end
    end
end

# ------------------------------------------------------------------------------------------
# OB5: FitResult's fields (P2, regularized: values, near_bound, term_values)
# ------------------------------------------------------------------------------------------

@testset "OB5: FitResult's fields (values, near_bound, term_values sum to R)" begin
    problem, theta, reg = _ob_p2_problem_with_reg()
    pspec = problem.pspec
    nvar = length(pspec.variables)

    vtable = _value_table(pspec, theta)
    values = Dict{String,Vector{Float64}}()
    for (i, v) in enumerate(pspec.variables)
        values[v.name] = vtable[i, :]
    end
    for (k, d) in enumerate(pspec.dependents)
        values[d.name] = vtable[nvar+k, :]
    end
    near_bound = Dict{String,BitVector}()
    for (i, v) in enumerate(pspec.variables)
        near_bound[v.name] = BitVector(_near_bound.(vtable[i, :], v.lower, v.upper))
    end

    term_values = [_regularizer_value(Regularizer([t]), pspec, theta) for t in reg.terms]
    Rval = _regularizer_value(reg, pspec, theta)
    c2 = chi2(problem, theta)
    obj = c2 + reg.weight * Rval
    fitted = unflatten(problem.experiment, pspec, theta)

    result = FitResult(theta, fitted, pspec, values, near_bound, c2, Rval, obj,
        reg.weight, term_values, problem.npixels,
        NamedTuple{(:iteration, :objective, :chi2, :R),NTuple{4,Float64}}[],
        true, :gradient_norm, 0, 0, 0, 0.0, "test")

    expected_names = Set(vcat([v.name for v in pspec.variables], [d.name for d in pspec.dependents]))
    @test Set(keys(result.values)) == expected_names
    @test "T_D_K" in keys(result.values)   # the dependent, explicitly (D26)
    for v in pspec.variables
        @test length(result.near_bound[v.name]) == length(result.values[v.name])
    end
    @test isapprox(sum(result.term_values), result.R; rtol=1e-14)
    @test isapprox(result.objective, result.chi2 + result.weight * result.R; rtol=1e-14)
end

# ------------------------------------------------------------------------------------------
# OB6: show names a flagged variable
# ------------------------------------------------------------------------------------------

@testset "OB6: show names a flagged variable" begin
    problem, theta = _ob_p1_problem()
    pspec = problem.pspec
    a, b = 50.0, 800.0
    values = Dict("T_D" => [a + 5e-4 * (b - a)], "u_D" => [0.0])
    near_bound = Dict("T_D" => BitVector([true]), "u_D" => BitVector([false]))
    fitted = unflatten(problem.experiment, pspec, theta)
    result = FitResult(theta, fitted, pspec, values, near_bound, 1.0, 0.0, 1.0,
        0.0, Float64[], problem.npixels,
        NamedTuple{(:iteration, :objective, :chi2, :R),NTuple{4,Float64}}[],
        true, :gradient_norm, 1, 1, 1, 0.1, "test")
    str = sprint(show, MIME("text/plain"), result)
    @test occursin("T_D", str)
end

# ------------------------------------------------------------------------------------------
# OB7: the log-posterior docstring
# ------------------------------------------------------------------------------------------

@testset "OB7: the log-posterior docstring" begin
    doc_str = string(Base.Docs.doc(objective))
    @test occursin("log posterior", doc_str)
    @test occursin("log-Jacobian", doc_str)
end

# ------------------------------------------------------------------------------------------
# OB8 (carried item 2, JT10's replacement): _objective_and_gradient! does not form a dense J,
# on an EPW-scale problem where a dense J actually dominates (npixels >= 4000, ntheta >= 20).
# ------------------------------------------------------------------------------------------

function _ob8_problem()
    expt = _ob_epw_expt(; nlambda=128, nt=32)
    pspec = _ob_epw_pspec(expt; with_dependent=false)
    working = _ob_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _ob_data(working, pspec, theta0)
    problem = FitProblem(working, pspec, data)
    theta = _ob_perturb(theta0)
    return problem, theta
end

@testset "OB8: _objective_and_gradient! does not form a dense J (EPW-scale)" begin
    problem, theta = _ob8_problem()
    npixels = problem.npixels
    ntheta = problem.pspec.ntheta
    @test npixels >= 4000
    @test ntheta >= 20

    g = zeros(ntheta)
    _objective_and_gradient!(g, problem, theta)   # warm
    bytes = @allocated _objective_and_gradient!(g, problem, theta)
    bound = 0.25 * npixels * ntheta * 8
    dense_J_bytes = npixels * ntheta * 8
    @info "OB8 allocation" bytes bound npixels ntheta dense_J_bytes
    @test bytes < bound
end
