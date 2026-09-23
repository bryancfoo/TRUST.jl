using Test
using OTIS
using OTIS: _logistic, _from_unconstrained, _to_unconstrained, _dx_dy, _near_bound
import ForwardDiff

# PM1 to PM26 against dev/phase3_instructions.md Section 4's Step 3.1 table (spec D3-D13),
# gated "fit parameters" (not "parameters": test/plasma/test_parameters.jl is "plasma
# parameters", Pitfall 22).
#
# PM3/PM4 note (2026-09-13, approved by the user after Level 3 verified it against a real Julia
# session): D3's overflow-safe logistic cannot make `_from_unconstrained` land *strictly* inside
# (a,b) for every finite y -- Float64 has no representable value between 1-eps(1.0)/2 and 1.0, so
# `_logistic(y)` is exactly 1.0 for any y >~ 37 (and exactly 0.0 for y <~ -745), regardless of
# formula. PM3/PM4 below test the achievable property instead: `a <= x <= b` and finiteness, never
# strict interiority at the extremes. See docs/derivations/parameters.md for the numbers.

# ------------------------------------------------------------------------------------------
# shared fixtures
# ------------------------------------------------------------------------------------------

function _pm_template(; f_electron=Maxwellian())
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0, f=f_electron)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _pm_template_two_ion()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=0.5),
              ion(name="C", mass_amu=12.0, charge=6, T_eV=200.0, u_kms=0.0, fraction=0.5)])
end

function _pm_wrap(trace::PlasmaTrace; effects=())
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=24)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m, effects=effects)
end

# P1 (D46): one bin, with Sensitivity and ConstantBackground (chain order already canonical).
function _pm_p1()
    trace = PlasmaTrace(_pm_template(); t_ns=[0.0])
    return _pm_wrap(trace; effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
end

# A small multi-bin problem for the parameter-layer bookkeeping tests (D46's P2 is a specific
# physical setup for the Jacobian oracle, Step 3.2; this chunk only needs the same *shape*: several
# bins, an IRF chain present).
function _pm_p2()
    trace = PlasmaTrace(_pm_template(); t_ns=collect(range(0.0, 3.0; length=10)))
    return _pm_wrap(trace; effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4),
        Sensitivity(scale=1.0e-6)))
end

function _pm_p2_two_ion()
    trace = PlasmaTrace(_pm_template_two_ion(); t_ns=collect(range(0.0, 3.0; length=10)))
    return _pm_wrap(trace)
end

# A rebuilt (or unrebuilt) experiment/pspec pair, wired the way unflatten requires: expt built
# from pspec's own (rebuilt) trace.
function _pm_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

# ------------------------------------------------------------------------------------------
# D3: the transform
# ------------------------------------------------------------------------------------------

@testset "PM1: the transform round-trips" begin
    bound_pairs = ((0.0, 1.0), (50.0, 800.0), (1e-7, 1e-5))
    for (a, b) in bound_pairs
        for frac in range(0.02, 0.98; length=20)
            x = a + frac * (b - a)
            y = _to_unconstrained(x, a, b)
            x2 = _from_unconstrained(y, a, b)
            @test isapprox(x2, x; rtol=1e-14)
        end
    end
end

@testset "PM2: the transform's derivative" begin
    a, b = 50.0, 800.0
    for y in (-3.0, -0.5, 0.0, 0.7, 4.0)
        x = _from_unconstrained(y, a, b)
        ad = ForwardDiff.derivative(yy -> _from_unconstrained(yy, a, b), y)
        closed = _dx_dy(x, a, b)
        @test isapprox(ad, closed; rtol=1e-12)
        h = 1e-6
        fd = (_from_unconstrained(y + h, a, b) - _from_unconstrained(y - h, a, b)) / (2h)
        floor = 1e-6 * max(abs(fd), 1e-8)
        @test isapprox(ad, fd; rtol=1e-6, atol=floor)
    end
end

@testset "PM3: the logistic does not overflow" begin
    @test isfinite(_logistic(-800.0)) && 0.0 <= _logistic(-800.0) <= 1.0
    @test isfinite(_logistic(800.0)) && 0.0 <= _logistic(800.0) <= 1.0
    a, b = 50.0, 800.0
    x = _from_unconstrained(-800.0, a, b)
    @test isfinite(x) && a <= x <= b
end

@testset "PM4: every finite theta lands in the bounds" begin
    a, b = 50.0, 800.0
    for y in (-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0)
        x = _from_unconstrained(y, a, b)
        @test isfinite(x)
        @test a <= x <= b
    end
end

# ------------------------------------------------------------------------------------------
# D4, D5, D8-D12: ParameterSpec, flatten, unflatten
# ------------------------------------------------------------------------------------------

@testset "PM5: flatten circ unflatten is the identity" begin
    expt = _pm_p2()
    axis = expt.trace.axis
    late = FitInterval("late", axis; t_range_ns=[1.5, 3.0])
    v_T = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0, time_varying=true)
    v_u = FittedVariable("u_late"; bounds=(-300.0, 300.0), init=50.0, time_varying=true, interval=1)
    v_n = FittedVariable("logne"; bounds=(18.0, 21.0), init=19.5)
    pspec = ParameterSpec(expt; variables=[v_T, v_u, v_n], intervals=[late],
        bindings=["T_e" => "electrons.e.T_eV", "u_late" => "ions.D.u_kms", "logne" => "n_e_cm3"])
    working = _pm_working(expt, pspec)
    theta = flatten(pspec)
    result = unflatten(working, pspec, theta)

    row_T = findfirst(==(LeafPath("electrons.e.T")), result.trace.paths)
    row_u = findfirst(==(LeafPath("ions.D.u")), result.trace.paths)
    row_n = findfirst(==(LeafPath("n_e")), result.trace.paths)
    @test all(isapprox.(result.trace.values[row_T, :], T_to_si(400.0, :T_eV); rtol=1e-14))
    for j in setdiff(1:nbins(expt.trace), late.bins)
        @test isapprox(result.trace.values[row_u, j], 0.0; atol=1e-30)
    end
    for j in late.bins
        @test isapprox(result.trace.values[row_u, j], u_to_si(50.0, :u_kms); rtol=1e-14)
    end
    @test all(isapprox.(result.trace.values[row_n, :], n_to_si(19.5, :n_cm3); rtol=1e-14))
end

@testset "PM6: unflatten against a hand-built Experiment" begin
    expt = _pm_p2()
    axis = expt.trace.axis
    late = FitInterval("late", axis; t_range_ns=[1.5, 3.0])
    v_T = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0, time_varying=true)
    v_u = FittedVariable("u_late"; bounds=(-300.0, 300.0), init=50.0, time_varying=true, interval=1)
    pspec = ParameterSpec(expt; variables=[v_T, v_u], intervals=[late],
        bindings=["T_e" => "electrons.e.T_eV", "u_late" => "ions.D.u_kms"])
    working = _pm_working(expt, pspec)
    result = unflatten(working, pspec, flatten(pspec))

    nt = nbins(expt.trace)
    T_vec = fill(400.0, nt)
    u_vec = [j in late.bins ? 50.0 : 0.0 for j in 1:nt]
    hand = PlasmaTrace(_pm_template(); t_ns=axis ./ 1e-9,
        leaves=("electrons.e.T_eV" => T_vec, "ions.D.u_kms" => u_vec))

    row_T = findfirst(==(LeafPath("electrons.e.T")), result.trace.paths)
    row_u = findfirst(==(LeafPath("ions.D.u")), result.trace.paths)
    hrow_T = findfirst(==(LeafPath("electrons.e.T")), hand.paths)
    hrow_u = findfirst(==(LeafPath("ions.D.u")), hand.paths)
    @test isapprox(result.trace.values[row_T, :], hand.values[hrow_T, :]; rtol=1e-12)
    @test isapprox(result.trace.values[row_u, :], hand.values[hrow_u, :]; rtol=1e-12)
end

@testset "PM7: a dual theta gives a dual Experiment" begin
    expt = _pm_p1()
    v = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0)
    pspec = ParameterSpec(expt; variables=[v], bindings=["T_e" => "electrons.e.T_eV"])
    working = _pm_working(expt, pspec)
    theta_plain = flatten(pspec)
    theta_dual = ForwardDiff.Dual.(theta_plain, 1.0)

    result_plain = unflatten(working, pspec, theta_plain)
    result_dual = unflatten(working, pspec, theta_dual)
    @test eltype(result_dual.trace.values) <: ForwardDiff.Dual
    row = findfirst(==(LeafPath("electrons.e.T")), result_dual.trace.paths)
    row2 = findfirst(==(LeafPath("electrons.e.T")), result_plain.trace.paths)
    @test isapprox(ForwardDiff.value.(result_dual.trace.values[row, :]),
        result_plain.trace.values[row2, :]; rtol=1e-15)
end

@testset "PM8: the dual reaches measure" begin
    expt = _pm_p1()
    v = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0)
    pspec = ParameterSpec(expt; variables=[v], bindings=["T_e" => "electrons.e.T_eV"])
    working = _pm_working(expt, pspec)
    theta_dual = ForwardDiff.Dual.(flatten(pspec), 1.0)
    result = unflatten(working, pspec, theta_dual)
    @test eltype(measure(result)) <: ForwardDiff.Dual
end

@testset "PM9: a bound effect parameter rebuilds the Experiment" begin
    expt = _pm_p1()
    v = FittedVariable("scale"; bounds=(1e-7, 1e-5), init=1e-6)
    pspec1 = ParameterSpec(expt; variables=[v], bindings=["scale" => (1, :scale)])
    working1 = _pm_working(expt, pspec1)
    result1 = unflatten(working1, pspec1, flatten(pspec1))
    @test result1 !== working1
    @test result1.effects[1] !== working1.effects[1]

    v2 = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0)
    pspec2 = ParameterSpec(expt; variables=[v2], bindings=["T_e" => "electrons.e.T_eV"])
    working2 = _pm_working(expt, pspec2)
    result2 = unflatten(working2, pspec2, flatten(pspec2))
    @test result2 === working2
end

@testset "PM10: an untouched effect is passed through ===" begin
    expt = _pm_p1()
    v = FittedVariable("scale"; bounds=(1e-7, 1e-5), init=1e-6)
    pspec = ParameterSpec(expt; variables=[v], bindings=["scale" => (1, :scale)])
    working = _pm_working(expt, pspec)
    result = unflatten(working, pspec, flatten(pspec))
    @test result.effects[2] === working.effects[2]
end

@testset "PM11: a frozen variable" begin
    expt = _pm_p2()
    v_frozen = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=350.0, time_varying=true, vary=false)
    v_other = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=20.0, time_varying=true)
    pspec = ParameterSpec(expt; variables=[v_frozen, v_other],
        bindings=["T_e" => "electrons.e.T_eV", "u_D" => "ions.D.u_kms"])
    @test pspec.offsets[1] == 0 && pspec.widths[1] == 0
    @test pspec.ntheta == nbins(expt.trace)

    working = _pm_working(expt, pspec)
    theta = flatten(pspec)
    @test length(theta) == pspec.ntheta
    result = unflatten(working, pspec, theta)
    row = findfirst(==(LeafPath("electrons.e.T")), result.trace.paths)
    @test all(isapprox.(result.trace.values[row, :], T_to_si(350.0, :T_eV); rtol=1e-14))
end

@testset "PM12: an interval variable is absent outside its window" begin
    expt = _pm_p2()
    axis = expt.trace.axis
    pre = FitInterval("pre", axis; t_range_ns=[0.0, 1.4])
    post = FitInterval("post", axis; t_range_ns=[1.4, 3.0])
    @test length(pre.bins) + length(post.bins) == nbins(expt.trace)

    u_pre = FittedVariable("u_pre"; bounds=(-300.0, 300.0), init=50.0, time_varying=true, interval=1)
    u_post = FittedVariable("u_post"; bounds=(-300.0, 300.0), init=-50.0, time_varying=true, interval=2)
    pspec = ParameterSpec(expt; variables=[u_pre, u_post], intervals=[pre, post],
        dependents=["u_ion" => "t < 1.4 ? u_pre : u_post"],
        bindings=["u_ion" => "ions.D.u_kms"])

    @test pspec.ntheta == length(pre.bins) + length(post.bins)
    for j in pre.bins
        @test 1 in pspec.bin_slots[j]
        @test !(2 in pspec.bin_slots[j])
    end
    for j in post.bins
        @test 2 in pspec.bin_slots[j]
        @test !(1 in pspec.bin_slots[j])
    end

    working = _pm_working(expt, pspec)
    result = unflatten(working, pspec, flatten(pspec))
    row = findfirst(==(LeafPath("ions.D.u")), result.trace.paths)
    for j in pre.bins
        @test isapprox(result.trace.values[row, j], u_to_si(50.0, :u_kms); rtol=1e-14)
    end
    for j in post.bins
        @test isapprox(result.trace.values[row, j], u_to_si(-50.0, :u_kms); rtol=1e-14)
    end
end

@testset "PM13: stick-breaking fractions sum to one" begin
    expt = _pm_p2_two_ion()
    f_D = FittedVariable("f_D"; bounds=(0.0, 1.0), init=0.5)
    pspec = ParameterSpec(expt; variables=[f_D], dependents=["f_C" => "1 - f_D"],
        bindings=["f_D" => "ions.D.fraction", "f_C" => "ions.C.fraction"])
    working = _pm_working(expt, pspec)
    result = unflatten(working, pspec, flatten(pspec))
    for j in 1:nbins(result.trace)
        p = plasma(result.trace, j)
        @test isapprox(neutrality_residual(p), 0.0; atol=1e-14)
    end
end

@testset "PM14: a variable in two unit keywords" begin
    expt = _pm_p2_two_ion()
    v = FittedVariable("T"; bounds=(50.0, 1500.0), init=400.0)
    @test_throws r"different units" ParameterSpec(expt; variables=[v],
        bindings=["T" => "electrons.e.T_eV", "T" => "ions.D.T_keV"])
end

@testset "PM15: one variable, two keys of the same factor, is fine" begin
    expt = _pm_p2_two_ion()
    v = FittedVariable("u_shared"; bounds=(-300.0, 300.0), init=100.0)
    pspec = ParameterSpec(expt; variables=[v],
        bindings=["u_shared" => "ions.D.u_kms", "u_shared" => "ions.C.u_kms"])
    working = _pm_working(expt, pspec)
    result = unflatten(working, pspec, flatten(pspec))
    rowD = findfirst(==(LeafPath("ions.D.u")), result.trace.paths)
    rowC = findfirst(==(LeafPath("ions.C.u")), result.trace.paths)
    @test isapprox(result.trace.values[rowD, :], result.trace.values[rowC, :]; rtol=1e-14)
end

@testset "PM16: a trace variable on an instrument key" begin
    expt = _pm_p1()
    v = FittedVariable("scale_t"; bounds=(1e-7, 1e-5), init=1e-6, time_varying=true)
    @test_throws r"time-varying" ParameterSpec(expt; variables=[v], bindings=["scale_t" => (1, :scale)])

    v2 = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0, time_varying=true)
    @test_throws r"time-varying" ParameterSpec(expt; variables=[v2],
        dependents=["scale_dep" => "T_e / 400"],
        bindings=["T_e" => "electrons.e.T_eV", "scale_dep" => (1, :scale)])
end

@testset "PM17: every other error of D39 that belongs to this step" begin
    expt = _pm_p1()

    # FittedVariable: lower >= upper
    @test_throws r"is not below the upper bound" FittedVariable("x"; bounds=(10.0, 5.0), init=7.0)

    # FittedVariable: bad name
    @test_throws r"not a valid variable name" FittedVariable("1x"; bounds=(0.0, 1.0), init=0.5)

    # ParameterSpec: a name in both [fit.variables] and [fit.dependent]
    vdup = FittedVariable("dup"; bounds=(0.0, 1.0), init=0.5)
    @test_throws r"declared both" ParameterSpec(expt; variables=[vdup],
        dependents=["dup" => "1 - 0"], bindings=[])

    # FitInterval: empty window
    @test_throws r"contains no bin" FitInterval("empty", expt.trace.axis; t_range_ns=[100.0, 200.0])

    # ParameterSpec: a dependent reads a variable outside its interval, unconditionally
    axis = expt.trace.axis  # single bin at t=0
    trace10 = PlasmaTrace(_pm_template(); t_ns=collect(range(0.0, 3.0; length=10)))
    expt10 = _pm_wrap(trace10)
    late = FitInterval("late", expt10.trace.axis; t_range_ns=[1.5, 3.0])
    u_late = FittedVariable("u_late"; bounds=(-300.0, 300.0), init=50.0, time_varying=true, interval=1)
    @test_throws r"not defined on" ParameterSpec(expt10; variables=[u_late], intervals=[late],
        dependents=["u_ion" => "u_late * 2"], bindings=["u_ion" => "ions.D.u_kms"])

    # Binding: two variables on one leaf
    va = FittedVariable("a_var"; bounds=(-300.0, 300.0), init=10.0)
    vb = FittedVariable("b_var"; bounds=(-300.0, 300.0), init=10.0)
    @test_throws r"are bound to" ParameterSpec(expt; variables=[va, vb],
        bindings=["a_var" => "ions.D.u_kms", "b_var" => "ions.D.u_kms"])

    # Binding: a probe or collection key
    vc = FittedVariable("th"; bounds=(0.0, 180.0), init=60.0)
    @test_throws r"WavelengthCalibration" ParameterSpec(expt; variables=[vc],
        bindings=["th" => "collection.theta_deg"])

    # Binding: a TabulatedKernel parameter (no fittable parameters)
    trace_tk = PlasmaTrace(_pm_template(); t_ns=[0.0, 1.0])
    expt_tk = _pm_wrap(trace_tk; effects=(TemporalIRF(TabulatedKernel((-1, 0, 1), (0.2, 0.6, 0.2))),))
    vd = FittedVariable("bad"; bounds=(0.0, 1.0), init=0.5)
    @test_throws r"no fittable parameters" ParameterSpec(expt_tk; variables=[vd],
        bindings=["bad" => (1, :sigma)])
end

@testset "PM18: the bound-proximity flag" begin
    a, b = 0.0, 100.0
    @test _near_bound(a + 5e-4 * (b - a), a, b)
    @test !_near_bound(a + 0.5 * (b - a), a, b)
end

@testset "PM19: a start on a bound is an error" begin
    @test_throws r"strictly inside" FittedVariable("x"; bounds=(0.0, 10.0), init=0.0)
    @test_throws r"strictly inside" FittedVariable("x"; bounds=(0.0, 10.0), init=10.0)
end

@testset "PM20: an infinite bound is an error" begin
    @test_throws r"must be finite" FittedVariable("x"; bounds=(0.0, Inf), init=1.0)
end

@testset "PM21: a wrong-length theta" begin
    expt = _pm_p1()
    v = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0)
    pspec = ParameterSpec(expt; variables=[v], bindings=["T_e" => "electrons.e.T_eV"])
    working = _pm_working(expt, pspec)
    @test_throws r"length" unflatten(working, pspec, ones(pspec.ntheta + 1))
end

@testset "PM22: theta's index map" begin
    trace6 = PlasmaTrace(_pm_template(); t_ns=collect(range(0.0, 3.0; length=6)))
    expt6 = _pm_wrap(trace6)
    axis = expt6.trace.axis

    interval = FitInterval("mid", axis; t_range_ns=[0.6, 1.8])
    @test interval.bins == 2:4

    f_D = FittedVariable("f_D"; bounds=(0.0, 1.0), init=1.0 - 1e-3, time_varying=false)
    T_e = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0, time_varying=true)
    u_pre = FittedVariable("u_pre"; bounds=(-300.0, 300.0), init=50.0, time_varying=true, interval=1)

    pspec = ParameterSpec(expt6; variables=[f_D, T_e, u_pre], intervals=[interval],
        bindings=["f_D" => "ions.D.fraction", "T_e" => "electrons.e.T_eV", "u_pre" => "ions.D.u_kms"])

    @test pspec.offsets == [1, 2, 8]
    @test pspec.widths == [1, 6, 3]
    @test pspec.ntheta == 10

    expected_entries = [[2], [3, 8], [4, 9], [5, 10], [6], [7]]
    expected_slots = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    @test pspec.bin_entries == expected_entries
    @test pspec.bin_slots == expected_slots
    @test pspec.global_entries == [1]
    @test pspec.global_slots == [1]
    @test sum(pspec.widths) == pspec.ntheta
end

@testset "PM23: a fitted leaf the trace did not tabulate gains a row" begin
    trace = PlasmaTrace(_pm_template(); t_ns=[0.0, 1.0])
    @test isempty(trace.paths)
    expt = _pm_wrap(trace)
    v = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=50.0)
    pspec = ParameterSpec(expt; variables=[v], bindings=["u_D" => "ions.D.u_kms"])
    @test length(pspec.trace.paths) == 1
    @test pspec.trace.paths[1] == LeafPath("ions.D.u")
    @test all(isapprox.(pspec.trace.values[1, :], 0.0; atol=0.0))
end

@testset "PM24: show" begin
    expt = _pm_p1()
    v = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0)
    pspec = ParameterSpec(expt; variables=[v], bindings=["T_e" => "electrons.e.T_eV"])
    s = sprint(show, pspec)
    @test occursin("ParameterSpec", s)
    s2 = sprint(show, MIME("text/plain"), pspec)
    @test occursin("T_e", s2)
    @test occursin("ntheta", s2)
end

@testset "PM25: a cycle among dependents is refused" begin
    expt = _pm_p1()
    @test_throws r"is cyclic" ParameterSpec(expt; variables=FittedVariable[],
        dependents=["a" => "b + 1", "b" => "a + 1"], bindings=[])
    @test_throws r"is cyclic" ParameterSpec(expt; variables=FittedVariable[],
        dependents=["p" => "q + 1", "q" => "r + 1", "r" => "p + 1"], bindings=[])
end

@testset "PM26: the real read-time probe" begin
    expt = _pm_p1()

    x_refused = FittedVariable("x"; bounds=(0.5, 3.0), init=1.0)
    @test_throws r"could not be evaluated at its initial values" ParameterSpec(expt;
        variables=[x_refused], dependents=["y" => "1/(x - 1)"], bindings=["y" => "electrons.e.T_eV"])

    x_ok = FittedVariable("x"; bounds=(0.5, 3.0), init=2.0)
    pspec_ok = ParameterSpec(expt; variables=[x_ok], dependents=["y" => "1/(x - 1)"],
        bindings=["y" => "electrons.e.T_eV"])
    @test pspec_ok isa ParameterSpec

    trace_pos = PlasmaTrace(_pm_template(); t_ns=[1.0, 2.0])
    expt_pos = _pm_wrap(trace_pos)
    v_dummy = FittedVariable("dummy"; bounds=(0.0, 1.0), init=0.5)
    pspec_logt = ParameterSpec(expt_pos; variables=[v_dummy], dependents=["logt" => "log(t)"],
        bindings=["logt" => "electrons.e.T_eV"])
    @test pspec_logt isa ParameterSpec
end

# PM27 (Step 3.1c remediation, 2026-09-13): ParameterSpec's `trace` field was declared
# `::PlasmaTrace{FT}`, a parametric type with only one of six parameters given, hence not
# concrete (`isconcretetype(PlasmaTrace{Float64}) == false`) -- an abstract field, boxing every
# write through it. Fixed by giving ParameterSpec a third type parameter `TR<:PlasmaTrace{FT}`
# and storing `trace::TR`, inferred at the construction site.
_pm_unflatten_allocated(expt, pspec, theta) = @allocated unflatten(expt, pspec, theta)

@testset "PM27: ParameterSpec is concretely typed" begin
    expt = _pm_p1()
    v = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0)
    pspec = ParameterSpec(expt; variables=[v], bindings=["T_e" => "electrons.e.T_eV"])
    for f in fieldnames(typeof(pspec))
        @test isconcretetype(fieldtype(typeof(pspec), f))
    end

    trace32 = PlasmaTrace(_pm_template(); t_ns=collect(range(0.0, 3.0; length=32)))
    expt32 = _pm_wrap(trace32)
    v1 = FittedVariable("T_e"; bounds=(50.0, 1500.0), init=400.0, time_varying=true)
    v2 = FittedVariable("u_D"; bounds=(-300.0, 300.0), init=20.0, time_varying=true)
    pspec32 = ParameterSpec(expt32; variables=[v1, v2],
        bindings=["T_e" => "electrons.e.T_eV", "u_D" => "ions.D.u_kms"])
    working32 = _pm_working(expt32, pspec32)
    theta32 = flatten(pspec32)

    _pm_unflatten_allocated(working32, pspec32, theta32)  # warm up
    bytes = _pm_unflatten_allocated(working32, pspec32, theta32)
    @test bytes < 4096
end
