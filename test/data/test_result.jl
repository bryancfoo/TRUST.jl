using Test
using OTIS
import JLD2
using JLD2: jldopen
using Random: Xoshiro
using Dates: Dates

# RE1 to RE11 against dev/phase3_instructions.md Section 9 (spec D34), gated "result file". This
# file also runs "data io" and "data streak" (RE10's other half, RE5): read_trace's default must
# stay :arrays, and the streak/trace layouts must stay untouched.

function _walk_leaf_values(f, prefix="")
    vals = Pair{String,Any}[]
    container = isempty(prefix) ? f : f[prefix]
    for k in keys(container)
        full = isempty(prefix) ? k : "$prefix/$k"
        v = f[full]
        if v isa JLD2.Group
            append!(vals, _walk_leaf_values(f, full))
        else
            push!(vals, full => v)
        end
    end
    return vals
end

# ------------------------------------------------------------------------------------------
# The fixture: a 24 x 6 problem (RE9's non-square shape, reused everywhere so a nl/nt mixup
# cannot hide behind a square array anywhere in this file), two trace variables, one dependent,
# a two-term D2 regularizer, and a genuine (noisy) fit.
# ------------------------------------------------------------------------------------------

const _RE_SEED = 20260915

function _re_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _re_expt()
    trace = PlasmaTrace(_re_template(); t_ns=collect(range(0.0, 3.0; length=6)))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=24)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
end

function _re_pspec(expt::Experiment)
    v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true)
    return ParameterSpec(expt; variables=[v_T, v_u], dependents=["ne_cm3" => "3.0e19 + 0.0*T_D"],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
end

function _re_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

# Builds the fixture problem, solves it a handful of LBFGS steps (just enough for a genuine,
# non-degenerate FitResult -- convergence quality is not this file's concern), and runs a small
# weight scan against a known truth so RE7's "present" half has real rss/rss_normalized content.
function _re_fixture()
    expt = _re_expt()
    pspec = _re_pspec(expt)
    working = _re_working(expt, pspec)
    nt = nbins(pspec.trace)
    t_ramp = range(220.0, 260.0; length=nt)
    u_ramp = range(-20.0, 20.0; length=nt)
    theta_truth = flatten(pspec, ["T_D" => collect(t_ramp), "u_D" => collect(u_ramp)])
    truth_expt = unflatten(working, pspec, theta_truth)
    data = synthesize(truth_expt, GaussianNoise(a=1.0, b=0.02); rng=Xoshiro(_RE_SEED))

    term_T = RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:absolute, weight=1.0)
    term_u = RegularizerTerm(pspec; select=["u_D"], operator=:D2, normalization=:absolute, weight=1.0)
    reg = Regularizer([term_T, term_u]; weight=1e-2)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    theta_start = theta_truth .+ 0.02 .* (1:length(theta_truth))
    result = solve(problem, LBFGS(iterations=25); start=theta_start)

    truth = Dict("T_D" => collect(t_ramp), "u_D" => collect(u_ramp))
    scan = weight_scan(problem, [1e-3, 1e-2], LBFGS(iterations=25); start=theta_start, truth=truth)

    return result, scan
end

const _RE_RESULT, _RE_SCAN = _re_fixture()

# ------------------------------------------------------------------------------------------
# RE1: the layout as written
# ------------------------------------------------------------------------------------------

@testset "RE1: the layout as written" begin
    result = _RE_RESULT
    pspec = result.pspec
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, result; deck="dummy deck text", source="test")
        jldopen(path, "r") do f
            got = Set(first.(_walk_leaf_values(f)))
            expected = Set{String}([
                "layout_version", "meta/source", "meta/created", "meta/created_unix",
                "meta/otis_version", "deck", "axis_name", "axis", "axis_unit",
                "axis_boundary_unit",
                "fit/theta", "fit/chi2", "fit/R", "fit/objective", "fit/weight",
                "fit/term_values", "fit/npixels", "fit/iterations", "fit/f_calls",
                "fit/g_calls", "fit/converged", "fit/status", "fit/termination",
                "fit/near_bound_fraction", "fit/optimizer", "fit/elapsed",
                "trajectory/iteration", "trajectory/objective", "trajectory/chi2", "trajectory/R",
            ])
            for v in pspec.variables
                for k in ("values", "unit", "bounds", "bins", "time_varying", "vary", "near_bound")
                    push!(expected, "variables/$(v.name)/$k")
                end
            end
            for dep in pspec.dependents
                push!(expected, "dependents/$(dep.name)/values")
                push!(expected, "dependents/$(dep.name)/expression")
            end
            for p in result.experiment.trace.paths
                push!(expected, "leaves/$(string(p))/values")
            end
            # channels/ (D40, chunk 7): every otis-result-3 file has it, "main" alone for a
            # single-channel fit; write_result is given no problem here, so no residual/mask.
            push!(expected, "channels/names")
            for summ in result.channels
                cprefix = "channels/$(summ.name)"
                for k in ("weight", "chi2", "npixels", "nparams", "reduced_chi2",
                          "reduced_chi2_simple", "lambda", "model")
                    push!(expected, "$cprefix/$k")
                end
            end
            for (c, summ) in enumerate(result.channels)
                cprefix = "channels/$(summ.name)"
                for (k, e) in enumerate(result.experiments[c].effects)
                    push!(expected, "$cprefix/effects/$k/type")
                    for field in keys(parameters(e))
                        push!(expected, "$cprefix/effects/$k/$field")
                    end
                end
            end
            @test got == expected
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE2: plain data only
# ------------------------------------------------------------------------------------------

@testset "RE2: plain data only" begin
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE_RESULT; deck="d", scan=_RE_SCAN)
        jldopen(path, "r") do f
            for (name, v) in _walk_leaf_values(f)
                @test v isa Union{AbstractArray{<:Real},AbstractArray{<:AbstractString},AbstractString,Real}
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE3: a bitwise round trip
# ------------------------------------------------------------------------------------------

@testset "RE3: a bitwise round trip" begin
    result = _RE_RESULT
    pspec = result.pspec
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, result; deck="the effective deck", source="fit", scan=_RE_SCAN)
        r = read_result(path)

        @test r.layout_version == "otis-result-3"
        @test r.deck == "the effective deck"
        @test r.axis_name == :t
        @test r.axis == pspec.axis
        @test r.axis_unit == "s"
        @test r.axis_boundary_unit == pspec.axis_unit

        for v in pspec.variables
            @test r.variables[v.name].values == result.values[v.name]
            @test r.variables[v.name].bounds == [v.lower, v.upper]
            @test r.variables[v.name].time_varying == UInt8(v.time_varying)
            @test r.variables[v.name].vary == UInt8(v.vary)
            @test r.variables[v.name].near_bound == UInt8.(result.near_bound[v.name])
        end
        for dep in pspec.dependents
            @test r.dependents[dep.name].values == result.values[dep.name]
            @test r.dependents[dep.name].expression == dep.text
        end
        for (i, p) in enumerate(result.experiment.trace.paths)
            @test r.leaves[string(p)] == result.experiment.trace.values[i, :]
        end

        @test r.fit.theta == result.theta
        @test r.fit.chi2 == result.chi2
        @test r.fit.R == result.R
        @test r.fit.objective == result.objective
        @test r.fit.weight == result.weight
        @test r.fit.term_values == result.term_values
        @test r.fit.npixels == result.npixels
        @test r.fit.iterations == result.iterations
        @test r.fit.f_calls == result.f_calls
        @test r.fit.g_calls == result.g_calls
        @test r.fit.converged == UInt8(result.converged)
        @test r.fit.status == string(result.status)
        @test r.fit.termination == string(result.termination)
        @test r.fit.near_bound_fraction == OTIS._near_bound_fraction(result)
        @test r.fit.optimizer == result.optimizer
        @test r.fit.elapsed == result.elapsed

        @test r.trajectory.objective == [row.objective for row in result.trajectory]
        @test r.trajectory.chi2 == [row.chi2 for row in result.trajectory]
        @test r.trajectory.R == [row.R for row in result.trajectory]

        @test r.scan.status == string.(getfield.(_RE_SCAN.results, :status))
        @test r.scan.termination == string.(getfield.(_RE_SCAN.results, :termination))
        @test r.scan.weights == _RE_SCAN.weights
        @test r.scan.chi2 == _RE_SCAN.chi2
        @test r.scan.R == _RE_SCAN.R
        @test r.scan.chi2_normalized == _RE_SCAN.chi2_normalized
        @test r.scan.R_normalized == _RE_SCAN.R_normalized
        @test r.scan.chi2_unreg == _RE_SCAN.chi2_unreg
        @test r.scan.R_unreg == _RE_SCAN.R_unreg
        for (name, v) in _RE_SCAN.rss
            @test r.scan.rss[name] == v
        end
        for (name, v) in _RE_SCAN.rss_normalized
            @test r.scan.rss_normalized[name] == v
        end
        for (name, v) in _RE_SCAN.rss_unreg
            @test r.scan.rss_unreg[name] == v
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE4: layout_version
# ------------------------------------------------------------------------------------------

@testset "RE4: layout_version" begin
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE_RESULT)
        @test read_result(path).layout_version == "otis-result-3"

        badpath = joinpath(dir, "bad.jld2")
        jldopen(badpath, "w") do f
            f["layout_version"] = "otis-result-0"
        end
        @test_throws "has layout_version" read_result(badpath)
    end
end

# ------------------------------------------------------------------------------------------
# RE5: the streak and trace layouts are untouched
# ------------------------------------------------------------------------------------------

@testset "RE5: the streak and trace layouts are untouched" begin
    @test OTIS.STREAK_LAYOUT_VERSION == "otis-streak-1"
    @test OTIS.TRACE_LAYOUT_VERSION == "otis-trace-2"   # amended 2026-10-10: Phase 4.J's one bump (spec D4, D9)

    nl, nt = 16, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt))
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd; source="test")
        sd2 = read_streak(path)
        @test all(sd2.image .=== sd.image)
    end
end

# ------------------------------------------------------------------------------------------
# RE6: meta/created
# ------------------------------------------------------------------------------------------

@testset "RE6: meta/created" begin
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE_RESULT)
        jldopen(path, "r") do f
            created = f["meta/created"]
            @test endswith(created, "Z")
            parsed = Dates.DateTime(created[1:end-1], Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss")
            @test parsed isa Dates.DateTime
            @test f["meta/created_unix"] isa Float64
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE7: the scan half
# ------------------------------------------------------------------------------------------

@testset "RE7: the scan half" begin
    mktempdir() do dir
        with_scan = joinpath(dir, "with_scan.jld2")
        write_result(with_scan, _RE_RESULT; scan=_RE_SCAN)
        jldopen(with_scan, "r") do f
            @test haskey(f, "scan")
            @test f["scan/status"] == string.(getfield.(_RE_SCAN.results, :status))
            @test f["scan/termination"] == string.(getfield.(_RE_SCAN.results, :termination))
            @test f["scan/weights"] == _RE_SCAN.weights
            @test f["scan/chi2"] == _RE_SCAN.chi2
            @test f["scan/R"] == _RE_SCAN.R
            @test f["scan/chi2_normalized"] == _RE_SCAN.chi2_normalized
            @test f["scan/R_normalized"] == _RE_SCAN.R_normalized
            @test f["scan/chi2_unreg"] == _RE_SCAN.chi2_unreg
            @test f["scan/R_unreg"] == _RE_SCAN.R_unreg
            for name in ("T_D", "u_D")
                @test f["scan/rss/$name"] == _RE_SCAN.rss[name]
                @test f["scan/rss_normalized/$name"] == _RE_SCAN.rss_normalized[name]
                @test f["scan/rss_unreg/$name"] == _RE_SCAN.rss_unreg[name]
            end
        end

        without_scan = joinpath(dir, "without_scan.jld2")
        write_result(without_scan, _RE_RESULT)
        jldopen(without_scan, "r") do f
            @test !haskey(f, "scan")
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE8: dependents are written
# ------------------------------------------------------------------------------------------

@testset "RE8: dependents are written" begin
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE_RESULT)
        jldopen(path, "r") do f
            @test f["dependents/ne_cm3/values"] == _RE_RESULT.values["ne_cm3"]
            @test f["dependents/ne_cm3/expression"] == "3.0e19 + 0.0*T_D"
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE9: a non-square result
# ------------------------------------------------------------------------------------------

@testset "RE9: a non-square result" begin
    result = _RE_RESULT
    nt = nbins(result.pspec.trace)
    @test length(result.experiment.lambda) == 24
    @test nt == 6
    @test length(result.experiment.lambda) != nt

    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE_RESULT)
        jldopen(path, "r") do f
            @test length(f["axis"]) == nt
            @test length(f["variables/T_D/values"]) == nt
            @test length(f["variables/T_D/near_bound"]) == nt
            @test length(f["dependents/ne_cm3/values"]) == nt
            for p in result.experiment.trace.paths
                @test length(f["leaves/$(string(p))/values"]) == nt
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE10: read_trace(path; as = :trace)
# ------------------------------------------------------------------------------------------

@testset "RE10: read_trace(path; as = :trace)" begin
    template = Plasma(; n_e_cm3=2e19,
        electrons=[electrons(T_eV=350.0, f=SuperGaussian(2.6))],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=180.0, u_kms=10.0)])
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 2.0; length=5)),
        leaves=["ions.D.T_eV" => fill(180.0, 5), "ions.D.u_kms" => collect(range(0.0, 40.0; length=5))])

    mktempdir() do dir
        path = joinpath(dir, "trace.jld2")
        write_trace(path, trace; source="test")

        arrays = read_trace(path)
        @test arrays.layout_version == "otis-trace-2"   # amended 2026-10-10 (Phase 4.J spec D4, D9)
        @test arrays.values == trace.values

        rebuilt = read_trace(path; as=:trace)
        @test rebuilt isa PlasmaTrace
        @test all(rebuilt.values[i, j] === trace.values[i, j] for i in axes(trace.values, 1), j in axes(trace.values, 2))
        @test rebuilt.axis == trace.axis
        @test string.(rebuilt.paths) == string.(trace.paths)

        rt = rebuilt.template
        tt = trace.template
        @test isapprox(rt.n_e, tt.n_e; rtol=1e-14)
        for i in eachindex(tt.electrons.species)
            @test isapprox(rt.electrons[i].T, tt.electrons[i].T; rtol=1e-14)
            @test isapprox(rt.electrons[i].u, tt.electrons[i].u; rtol=1e-14)
            @test isapprox(rt.electrons[i].fraction, tt.electrons[i].fraction; rtol=1e-14)
            @test typeof(rt.electrons[i].f) == typeof(tt.electrons[i].f)
            @test isapprox(parameters(rt.electrons[i].f).m, parameters(tt.electrons[i].f).m; rtol=1e-14)
        end
        for i in eachindex(tt.ions.species)
            @test isapprox(rt.ions[i].T, tt.ions[i].T; rtol=1e-14)
            @test isapprox(rt.ions[i].u, tt.ions[i].u; rtol=1e-14)
            @test isapprox(rt.ions[i].fraction, tt.ions[i].fraction; rtol=1e-14)
            @test rt.ions[i].mass == tt.ions[i].mass
            @test rt.ions[i].charge == tt.ions[i].charge
        end

        for j in 1:nbins(trace)
            p1 = plasma(rebuilt, j)
            p2 = plasma(trace, j)
            @test isapprox(p1.ions[1].T, p2.ions[1].T; rtol=1e-14)
            @test isapprox(p1.ions[1].u, p2.ions[1].u; rtol=1e-14)
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE11: an unknown as
# ------------------------------------------------------------------------------------------

@testset "RE11: an unknown as" begin
    template = Plasma(; n_e_cm3=1e19, electrons=[electrons(T_eV=300.0)])
    trace = PlasmaTrace(template; t_ns=[0.0, 1.0])
    mktempdir() do dir
        path = joinpath(dir, "trace.jld2")
        write_trace(path, trace)
        @test_throws "as must be :arrays or :trace" read_trace(path; as=:bogus)
    end
end

# ------------------------------------------------------------------------------------------
# RE12, RE13 (A19, D47): laplace/ and calibration_scan/ inside otis-result-3. A joint,
# two-channel fixture, built here (not shared across the @safetestset boundary): channel b
# carries an unbound WavelengthCalibration, chunk 9's own uncertainty fixture (D35's MC fixture,
# copied, dlam_b removed).
# ------------------------------------------------------------------------------------------

function _re12_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _re12_experiments()
    trace = PlasmaTrace(_re12_template(); t_ns=[0.0, 1.0, 2.0, 3.0])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_a = collect(range(262.9, 263.6; length=16)) .* 1e-9
    lambda_b = collect(range(262.2, 262.9; length=12)) .* 1e-9
    expt_a = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_a,
        effects=(SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6), ConstantBackground(level=20.0)))
    expt_b = Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_b,
        effects=(WavelengthCalibration(lambda_offset_nm=0.01), SpectralIRF(fwhm_nm=0.12),
            Sensitivity(scale=2.0e-6), ConstantBackground(level=15.0)))
    return expt_a, expt_b
end

function _re12_pspec(expt_a::Experiment, expt_b::Experiment)
    sens_a = findfirst(e -> e isa Sensitivity, expt_a.effects)
    bg_a = findfirst(e -> e isa ConstantBackground, expt_a.effects)
    sens_b = findfirst(e -> e isa Sensitivity, expt_b.effects)
    bg_b = findfirst(e -> e isa ConstantBackground, expt_b.effects)

    v_TD = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
    v_Te = FittedVariable("T_e"; bounds=(100.0, 800.0), init=400.0)
    v_scale_a = FittedVariable("scale_a"; bounds=(1e-7, 1e-5), init=1e-6)
    v_scale_b = FittedVariable("scale_b"; bounds=(1e-7, 1e-5), init=1e-6)
    v_bg = FittedVariable("bg"; bounds=(1.0, 60.0), init=20.0)

    return ParameterSpec((expt_a, expt_b);
        variables=[v_TD, v_Te, v_scale_a, v_scale_b, v_bg],
        bindings=["T_D" => "ions.D.T_eV", "T_e" => "electrons.e.T_eV",
            "scale_a" => (channel=1, effect=sens_a, field=:scale),
            "scale_b" => (channel=2, effect=sens_b, field=:scale),
            "bg" => (channel=1, effect=bg_a, field=:level),
            "bg" => (channel=2, effect=bg_b, field=:level)])
end

function _re12_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

_re12_regularizer(pspec::ParameterSpec) = Regularizer(
    [RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:unit, weight=1.0)];
    weight=1e-3)

function _re12_joint_and_star()
    expt_a, expt_b = _re12_experiments()
    pspec = _re12_pspec(expt_a, expt_b)
    reg = _re12_regularizer(pspec)
    working_a = _re12_working(expt_a, pspec)
    working_b = _re12_working(expt_b, pspec)
    theta0 = flatten(pspec)
    rng = Xoshiro(20260923)
    truth_a = unflatten(working_a, pspec, theta0; channel=1)
    data_a = synthesize(truth_a, GaussianNoise(a=1.0, b=0.02); rng=rng)
    truth_b = unflatten(working_b, pspec, theta0; channel=2)
    data_b = synthesize(truth_b, GaussianNoise(a=1.0, b=0.02); rng=rng)
    ch_a = FitProblem(working_a, pspec, data_a; channel=1, name="a")
    ch_b = FitProblem(working_b, pspec, data_b; channel=2, name="b")
    jp = JointFitProblem((ch_a, ch_b); weights=[0.3, 2.0], regularizer=reg)
    theta_start = theta0 .+ 0.02 .* (1:length(theta0))
    r = solve(jp, LevenbergMarquardt(); start=theta_start)
    pol = polish(jp, r)
    return jp, pol.result
end

const _RE12_JP, _RE12_RESULT = _re12_joint_and_star()

@testset "RE12: write_result with laplace (D47)" begin
    lap = laplace(_RE12_JP, _RE12_RESULT.theta)
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE12_RESULT; deck="d", laplace=lap)

        jldopen(path, "r") do f
            for (name, v) in _walk_leaf_values(f, "laplace")
                @test v isa Union{AbstractArray{<:Real},AbstractArray{<:AbstractString},AbstractString,Real}
            end
        end

        r = read_result(path)
        @test r.laplace !== nothing
        @test r.laplace.decrease == lap.decrease
        for name in keys(lap.sigma_post)
            @test isequal(r.laplace.sigma_post[name], lap.sigma_post[name])
            @test isequal(r.laplace.sigma_samp[name], lap.sigma_samp[name])
            @test r.laplace.near_bound[name] == UInt8.(lap.near_bound[name])
        end

        r2 = solve(_RE12_JP, LevenbergMarquardt(); start=_RE12_RESULT.theta .+ 0.01)
        wrong_lap = laplace(_RE12_JP, polish(_RE12_JP, r2).result.theta)
        @test_throws r"different θ than result.theta" write_result(path, _RE12_RESULT; laplace=wrong_lap)
    end
end

@testset "RE13: write_result with calibration_scan (D47)" begin
    scan = calibration_scan(_RE12_JP, _RE12_RESULT.theta, [-0.01, 0.0, 0.01]; channel=2)
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE12_RESULT; deck="d", calibration_scan=scan)

        jldopen(path, "r") do f
            for (name, v) in _walk_leaf_values(f, "calibration_scan")
                @test v isa Union{AbstractArray{<:Real},AbstractArray{<:AbstractString},AbstractString,Real}
            end
        end

        r = read_result(path)
        cs = r.calibration_scan
        @test cs !== nothing
        @test cs.shifts_nm == scan.shifts_nm
        @test cs.channel == scan.channel
        @test cs.base_offset_nm == scan.base_offset_nm
        @test isequal(cs.chi2, scan.chi2)
        @test isequal(cs.objective, scan.objective)
        @test isequal(cs.decrease, scan.decrease)
        @test cs.status == string.(scan.status)
        for name in keys(scan.delta)
            @test isequal(cs.delta[name], scan.delta[name])
            @test isequal(cs.s_sys[name], scan.s_sys[name])
        end
    end
end

# ------------------------------------------------------------------------------------------
# RE14 (A19, D52): scatter/ inside otis-result-3. A local copy of test_uncertainty.jl's SXA
# fixture (not shared across the @safetestset boundary, chunk 6's own rule): a two-channel deck,
# T_D and u_D free and time-varying, T_e fixed, noise-free data.
# ------------------------------------------------------------------------------------------

_re14_template() = Plasma(; n_e_cm3=3e19,
    electrons=[electrons(T_eV=400.0)],
    ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])

_re14_TD(t_ns) = 180.0 + 60.0 * t_ns
_re14_uD(t_ns) = 20.0 + 15.0 * t_ns
_re14_path(p::AbstractString) = replace(p, "\\" => "/")

function _re14_write(path::AbstractString, t_ns_recorded::AbstractVector, t_shift_ns::Real,
    lambda_lo::Real, lambda_hi::Real, nlambda::Integer, scale::Real, background::Bool)
    t_true_ns = t_ns_recorded .+ t_shift_ns
    trace = PlasmaTrace(_re14_template(); t_ns=collect(t_true_ns),
        leaves=("ions.D.T_eV" => _re14_TD, "ions.D.u_kms" => _re14_uD))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda = collect(range(lambda_lo, lambda_hi; length=nlambda)) .* 1e-9
    effects = background ? (Sensitivity(scale=scale), ConstantBackground(level=25.0)) :
              (Sensitivity(scale=scale),)
    expt = Experiment(; trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
    image = measure(expt)
    sig = sigma(GaussianNoise(; a=121.0, b=141.75), image)
    sd = StreakData(; lambda_m=lambda, t_s=t_ns_recorded .* 1e-9, image=image, sigma=sig)
    write_streak(path, sd; source="re14_test")
    return sd
end

_re14_write_a(path::AbstractString) = _re14_write(path, collect(range(0.05, 1.15; length=12)), 0.0,
    262.0, 264.5, 24, 1.0e-6, true)
_re14_write_b(path::AbstractString) = _re14_write(path, collect(range(0.03, 1.13; length=12)), 0.02,
    262.5, 264.0, 16, 2.0e-6, false)

function _re14_deck_text(path_a::AbstractString, path_b::AbstractString, outdir::AbstractString)
    return """
    deck_version = "otis-deck-2"
    channels = ["a", "b"]

    [[fit.regions]]
    t_range_ns = [0.0, 1.2]
    nbins = 3

    [channel.a.data]
    file = "$(_re14_path(path_a))"
    [channel.a.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"
    [channel.a.instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [channel.b.data]
    file = "$(_re14_path(path_b))"
    t_shift_ns = 0.02
    [channel.b.geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"
    [channel.b.instrument]
    sensitivity = { scale = 2.0e-6 }

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = "u_D"

    [regularizer]
    weight = 1.0
    terms = [
      { select = ["T_D", "u_D"], operator = "D2", normalization = "relative" },
    ]

    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_re14_path(outdir))"
    """
end

function _re14_deck_and_scatter()
    return mktempdir() do dir
        path_a = joinpath(dir, "sxa.jld2")
        path_b = joinpath(dir, "sxb.jld2")
        _re14_write_a(path_a)
        _re14_write_b(path_b)
        text = _re14_deck_text(path_a, path_b, joinpath(dir, "out"))
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, text)
        deck = read_deck(deckpath)
        problem, opt, start = OTIS.build_problem(deck)
        r = solve(problem, opt; start=start)
        pol = polish(problem, r)
        sc = slice_scatter(deck, pol.result)
        (; deck, problem, result=pol.result, sc)
    end
end

const _RE14_FIXTURE = _re14_deck_and_scatter()

@testset "RE14: write_result with scatter (D52)" begin
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE14_FIXTURE.result; deck="d", scatter=_RE14_FIXTURE.sc)

        jldopen(path, "r") do f
            for (name, v) in _walk_leaf_values(f, "scatter")
                @test v isa Union{AbstractArray{<:Real},AbstractArray{<:AbstractString},AbstractString,Real}
            end
        end

        r = read_result(path)
        sc_read = r.scatter
        @test sc_read !== nothing
        @test isequal(sc_read.m, _RE14_FIXTURE.sc.m)
        @test isequal(sc_read.t_fine, _RE14_FIXTURE.sc.t_fine)
        @test isequal(sc_read.coarse_bin, _RE14_FIXTURE.sc.coarse_bin)
        @test isequal(sc_read.counts, _RE14_FIXTURE.sc.counts)
        @test sc_read.status == string.(_RE14_FIXTURE.sc.status)
        @test isequal(sc_read.decrease, _RE14_FIXTURE.sc.decrease)
        for name in keys(_RE14_FIXTURE.sc.values)
            @test isequal(sc_read.values[name], _RE14_FIXTURE.sc.values[name])
            @test sc_read.near_bound[name] == UInt8.(_RE14_FIXTURE.sc.near_bound[name])
            @test isequal(sc_read.s_stat[name], _RE14_FIXTURE.sc.s_stat[name])
            @test isequal(sc_read.n_near_bound[name], _RE14_FIXTURE.sc.n_near_bound[name])
        end

        wrong_theta = copy(_RE14_FIXTURE.sc.theta)
        wrong_theta[1] += 0.01
        wrong_sc = ScatterResult(wrong_theta, _RE14_FIXTURE.sc.m, _RE14_FIXTURE.sc.t_fine,
            _RE14_FIXTURE.sc.coarse_bin, _RE14_FIXTURE.sc.counts, _RE14_FIXTURE.sc.status,
            _RE14_FIXTURE.sc.decrease, _RE14_FIXTURE.sc.values, _RE14_FIXTURE.sc.near_bound,
            _RE14_FIXTURE.sc.s_stat, _RE14_FIXTURE.sc.n_near_bound)
        @test_throws r"different θ than result.theta" write_result(path, _RE14_FIXTURE.result; scatter=wrong_sc)
    end
end

# ------------------------------------------------------------------------------------------
# RE15 (Phase 4.B spec D5, D21): fit/gn_decrease, optional inside otis-result-3
# ------------------------------------------------------------------------------------------

@testset "RE15: write_result with gn_decrease (Phase 4.B spec D5)" begin
    mktempdir() do dir
        path = joinpath(dir, "result.jld2")
        write_result(path, _RE12_RESULT; deck="d", gn_decrease=0.25)
        jldopen(path, "r") do f
            @test f["fit/gn_decrease"] == 0.25
        end
        r = read_result(path)
        @test r.fit.gn_decrease == 0.25

        path_nan = joinpath(dir, "result_nan.jld2")
        write_result(path_nan, _RE12_RESULT; deck="d", gn_decrease=NaN)
        r_nan = read_result(path_nan)
        @test isnan(r_nan.fit.gn_decrease)

        path_none = joinpath(dir, "result_none.jld2")
        write_result(path_none, _RE12_RESULT; deck="d")
        jldopen(path_none, "r") do f
            @test !haskey(f, "fit/gn_decrease")
        end
        r_none = read_result(path_none)
        @test r_none.fit.gn_decrease === nothing
    end
end
