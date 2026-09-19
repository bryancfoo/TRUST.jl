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

        @test r.layout_version == "otis-result-2"
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
        @test read_result(path).layout_version == "otis-result-2"

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
    @test OTIS.TRACE_LAYOUT_VERSION == "otis-trace-1"

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
        @test arrays.layout_version == "otis-trace-1"
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
