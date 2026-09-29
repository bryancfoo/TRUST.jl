# Step 4.D.1: the saturation warning (Phase 4.D spec D4-D8, OQ-49 option (a)). NB1-NB6, gate
# "near-bound warning". fit(deck) warns when any fitted entry is within BOUND_PROXIMITY_TOL of a
# bound, naming each variable and its bins, and the result file lists the entries beside
# fit/near_bound_fraction; the fit, its converged and status, and every other dataset are exactly
# as they would be without the warning (OQ-36, OQ-41).

using Test
using OTIS
using OTIS: _near_bound, _near_bound_entries, _near_bound_fraction, _warn_near_bound
using JLD2: jldopen

# ------------------------------------------------------------------------------------------
# The fixture (spec D8): four bins, 24 wavelengths, noise-free, sigma = 1. T_D's truth is 250 eV
# except bin 3 (400 eV, above the deck's upper bound of 300) and bin 4 (40 eV, below its lower
# bound of 50), so the data pull exactly those two entries onto their bounds; u_D is a scalar,
# truth 40 km/s. Two entries of five: a fraction of 0.4, under SATURATION_FRACTION, the case
# the fraction warning cannot see (Phase 4.B's A14).
# ------------------------------------------------------------------------------------------

const _NB_T_NS = [0.0, 1.0, 2.0, 3.0]
_nb_T_truth(t) = t == 2.0 ? 400.0 : t == 3.0 ? 40.0 : 250.0

function _nb_expt(t_ns=_NB_T_NS)
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=250.0, u_kms=40.0)])
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("ions.D.T_eV" => _nb_T_truth,))
    return Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact),
        lambda_m=collect(range(262.0, 264.5; length=24)) .* 1e-9,
        effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
end

function _nb_streak(path::AbstractString; t_ns=_NB_T_NS)
    expt = _nb_expt(t_ns)
    image = measure(expt)
    sd = StreakData(lambda_m=expt.lambda, t_s=expt.t, image=image, sigma=ones(size(image)))
    write_streak(path, sd; source="test")
    return sd
end

function _nb_deck(datafile, outdir; t_bounds="[50.0, 300.0]", t_init=200.0, time_varying="true")
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    T_D = { bounds = $t_bounds, init = $t_init, time_varying = $time_varying }
    u_D = { bounds = [-200.0, 200.0], init = 10.0 }

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

    [instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "lbfgs"

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

const _NB_HEAD2 = "fit: 2 fitted entries are within BOUND_PROXIMITY_TOL of a bound, where dx/dy -> 0 and an " *
    "entry may have stalled short of a minimum (the result is as fitted; `converged` does not see this):"
const _NB_MSG = _NB_HEAD2 * "\n  T_D: bin 4 (lower bound 50.0); bin 3 (upper bound 300.0)"

# ------------------------------------------------------------------------------------------
# NB1: the entries, from the flags
# ------------------------------------------------------------------------------------------

@testset "NB1: the near-bound entries of a converged fit with two bins on their bounds" begin
    mktempdir() do dir
        datafile = joinpath(dir, "nb.jld2")
        _nb_streak(datafile)
        deckpath = joinpath(dir, "pinned.toml")
        write(deckpath, _nb_deck(datafile, joinpath(dir, "pinned")))
        problem, optimizer, start = OTIS.build_problem(read_deck(deckpath))
        r = solve(problem, optimizer; start=start)
        @info "NB1 the pinned fit" status = r.status converged = r.converged iterations = r.iterations T_D = string(r.values["T_D"]) u_D = r.values["u_D"][1]
        @test r.converged
        @test r.near_bound["T_D"] == [false, false, true, true]
        @test r.near_bound["u_D"] == [false]
        @test _near_bound_entries(r) == [(variable="T_D", bin=3, bound=:upper), (variable="T_D", bin=4, bound=:lower)]
        @test _near_bound_fraction(r) == 0.4 < OTIS.SATURATION_FRACTION
    end
end

# ------------------------------------------------------------------------------------------
# NB2: fit(deck) warns once, naming the variable and both bins, and changes nothing
# ------------------------------------------------------------------------------------------

@testset "NB2: fit(deck) warns once, and the fit is exactly solve's" begin
    mktempdir() do dir
        datafile = joinpath(dir, "nb.jld2")
        _nb_streak(datafile)
        deckpath = joinpath(dir, "pinned.toml")
        write(deckpath, _nb_deck(datafile, joinpath(dir, "pinned")))
        r = @test_logs (:warn, _NB_MSG) min_level=Base.CoreLogging.Warn fit(deckpath)
        problem, optimizer, start = OTIS.build_problem(read_deck(deckpath))
        s = solve(problem, optimizer; start=start)
        @test all(r.theta .=== s.theta)
        @test r.converged === s.converged && r.status === s.status && r.iterations == s.iterations
        @test r.objective === s.objective
        logger = Test.TestLogger(min_level=Base.CoreLogging.Warn)
        Base.CoreLogging.with_logger(() -> fit(deckpath), logger)
        record = only(logger.logs)
        @test record.kwargs[:near_bound_fraction] == 0.4
        @test record.kwargs[:status] === r.status
        @test record.kwargs[:converged] === r.converged
    end
end

# ------------------------------------------------------------------------------------------
# NB3: no entry, no warning, no dataset
# ------------------------------------------------------------------------------------------

@testset "NB3: a fit with no near-bound entry is silent and writes no list" begin
    mktempdir() do dir
        datafile = joinpath(dir, "nb.jld2")
        _nb_streak(datafile)
        outdir = joinpath(dir, "free")
        deckpath = joinpath(dir, "free.toml")
        write(deckpath, _nb_deck(datafile, outdir; t_bounds="[10.0, 800.0]"))
        r = @test_logs min_level=Base.CoreLogging.Warn fit(deckpath)
        @test isempty(_near_bound_entries(r))
        @test isempty(_warn_near_bound(r))
        jldopen(joinpath(outdir, "result.jld2"), "r") do f
            @test !haskey(f, "fit/near_bound_entries")
        end
        @test read_result(joinpath(outdir, "result.jld2")).fit.near_bound_entries === nothing
    end
end

# ------------------------------------------------------------------------------------------
# NB4: the result file lists the entries beside fit/near_bound_fraction
# ------------------------------------------------------------------------------------------

@testset "NB4: fit/near_bound_entries, written and read back" begin
    mktempdir() do dir
        datafile = joinpath(dir, "nb.jld2")
        _nb_streak(datafile)
        outdir = joinpath(dir, "pinned")
        deckpath = joinpath(dir, "pinned.toml")
        write(deckpath, _nb_deck(datafile, outdir))
        r = Base.CoreLogging.with_logger(() -> fit(deckpath), Test.TestLogger(min_level=Base.CoreLogging.Warn))
        path = joinpath(outdir, "result.jld2")
        jldopen(path, "r") do f
            @test f["fit/near_bound_entries/variable"] == ["T_D", "T_D"]
            @test f["fit/near_bound_entries/bin"] == Int64[3, 4]
            @test f["fit/near_bound_entries/bound"] == ["upper", "lower"]
            @test f["fit/near_bound_fraction"] == 0.4
        end
        rr = read_result(path)
        e = rr.fit.near_bound_entries
        @test e.variable == ["T_D", "T_D"] && e.bin == [3, 4] && e.bound == ["upper", "lower"]
        @test findall(!iszero, rr.variables["T_D"].near_bound) == e.bin
        @test rr.fit.converged == UInt8(r.converged)
        @test rr.fit.theta == r.theta
    end
end

# ------------------------------------------------------------------------------------------
# NB5: a scalar variable is named as such; the fraction warning fires beside it at 0.5
# ------------------------------------------------------------------------------------------

@testset "NB5: a scalar entry, and both warnings at a fraction of 0.5" begin
    mktempdir() do dir
        datafile = joinpath(dir, "nb1.jld2")
        _nb_streak(datafile; t_ns=[0.0])                  # one bin, T_D truth 250 eV (DK21's fixture B)
        outdir = joinpath(dir, "scalar")
        deckpath = joinpath(dir, "scalar.toml")
        write(deckpath, _nb_deck(datafile, outdir; t_bounds="[50.0, 150.0]", t_init=100.0, time_varying="false"))
        msg = "fit: 1 fitted entry is within BOUND_PROXIMITY_TOL of a bound, where dx/dy -> 0 and an " *
            "entry may have stalled short of a minimum (the result is as fitted; `converged` does not see this):" *
            "\n  T_D: scalar (upper bound 150.0)"
        r = @test_logs (:warn, r"near-bound fraction reaches SATURATION_FRACTION") (:warn, msg) min_level=Base.CoreLogging.Warn fit(deckpath)
        @test _near_bound_entries(r) == [(variable="T_D", bin=0, bound=:upper)]
        rr = read_result(joinpath(outdir, "result.jld2"))
        @test rr.fit.near_bound_entries.bin == [0]
        @test rr.fit.near_bound_fraction == 0.5
    end
end

# ------------------------------------------------------------------------------------------
# NB6: a frozen variable near its bound, and a variable's NaN outside its interval, are never
# listed (built in code: an interval and a frozen variable on the same four bins)
# ------------------------------------------------------------------------------------------

@testset "NB6: frozen variables and NaN outside an interval are never listed" begin
    @test _near_bound(NaN, -200.0, 200.0) === false
    expt = _nb_expt()
    image = measure(expt)
    data = StreakData(lambda_m=expt.lambda, t_s=expt.t, image=image, sigma=ones(size(image)))
    early = FitInterval("early", expt.trace.axis; t_range_ns=[0.0, 1.5])     # bins 1:2
    v_T = FittedVariable("T_D"; bounds=(50.0, 300.0), init=200.0, time_varying=true)
    v_u = FittedVariable("u_early"; bounds=(-200.0, 200.0), init=10.0, time_varying=true, interval=1)
    v_e = FittedVariable("T_e"; bounds=(399.9, 800.0), init=400.0, vary=false)
    pspec = ParameterSpec(expt; variables=[v_T, v_u, v_e], intervals=[early],
        dependents=["u_D" => "t < 1.5 ? u_early : 40.0"],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms", "T_e" => "electrons.e.T_eV"])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
    r = solve(FitProblem(working, pspec, data), LBFGS(); start=flatten(pspec))
    @test r.near_bound["T_e"] == [true]                       # frozen 0.1 above its lower bound: flagged
    @test isnan.(r.values["u_early"]) == [false, false, true, true]
    @test r.near_bound["u_early"] == [false, false, false, false]
    @test _near_bound_entries(r) == [(variable="T_D", bin=3, bound=:upper), (variable="T_D", bin=4, bound=:lower)]
    @test_logs (:warn, _NB_MSG) min_level=Base.CoreLogging.Warn _warn_near_bound(r)
end
