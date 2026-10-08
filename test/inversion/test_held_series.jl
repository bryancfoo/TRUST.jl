using Test
using OTIS
using OTIS: _frozen_value, _slot_defined_at, _value_table, _regularizer_raw_rows, _regularizer_closed_form

# Step 4.I.1: held series (Phase 4.I spec D7, D10, D11, D13; OQ-59 item 3), gated "held series". HD1,
# HD2 and HD5: the mechanism, on test_jacobian.jl's P2 chain (test/fixtures/held_series_fixture.jl);
# HD6: a deck's [fit.hold]. HD3 and HD4, which compile the brute-force Jacobian and the per-bin dual
# pass of a chain nothing earlier builds with a held variable, are in the slow tier
# (test/slow/test_held_series.jl).

include(joinpath(@__DIR__, "..", "fixtures", "held_series_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "start_hold_fixture.jl"))

# ------------------------------------------------------------------------------------------
# HD1: FittedVariable's held field and its rules
# ------------------------------------------------------------------------------------------

@testset "HD1: held implies vary = false, and its refusals" begin
    v = FittedVariable("x"; bounds=(0.0, 1.0), init=0.5, time_varying=true, held=[0.2, 0.3])
    @test v.vary == false
    @test v.held == [0.2, 0.3] && v.held isa Vector{Float64}
    @test FittedVariable("x"; bounds=(0.0, 1.0), init=0.5, vary=false, held=[0.0]).held == [0.0]   # a bound is a value it may take
    w = FittedVariable("x"; bounds=(0.0, 1.0), init=0.5)
    @test w.held === nothing && w.vary == true
    @test FittedVariable("x"; bounds=(0.0, 1.0), init=0.5, vary=false).held === nothing
    @test_throws "FittedVariable \"x\": held is given, so the variable is not fitted; vary = true contradicts it" FittedVariable("x";
        bounds=(0.0, 1.0), init=0.5, vary=true, held=[0.2])
    @test_throws "FittedVariable \"x\": a scalar variable is held at one value; got 2" FittedVariable("x";
        bounds=(0.0, 1.0), init=0.5, held=[0.2, 0.3])
    @test_throws "FittedVariable \"x\": held value 1.5 (entry 2) must be finite and within the bounds [0.0, 1.0]" FittedVariable("x";
        bounds=(0.0, 1.0), init=0.5, time_varying=true, held=[0.2, 1.5])
    @test_throws "FittedVariable \"x\": held is empty" FittedVariable("x"; bounds=(0.0, 1.0), init=0.5, held=Float64[])
    @test_throws "ParameterSpec: \"T_D\" is held at 5 values, but its interval has 6 bins" _hd_problem(; T_held=_HD_HELD[1:5])
end

# ------------------------------------------------------------------------------------------
# HD2: the held series is the variable's value at its own bins, bitwise, and is not in theta
# ------------------------------------------------------------------------------------------

@testset "HD2: unflatten writes the held series at the variable's own bins" begin
    problem, theta = _hd_problem()
    pspec = problem.pspec
    @test pspec.ntheta == 10                       # u_D alone: T_D's six bins and s are not in theta
    T, s = pspec.variables[1], pspec.variables[3]
    @test [_frozen_value(T, j, pspec.intervals) for j in 3:8] == _HD_HELD
    @test isnan(_frozen_value(T, 2, pspec.intervals)) && isnan(_frozen_value(T, 9, pspec.intervals))
    @test all(j -> _frozen_value(s, j, pspec.intervals) === 2.0e-6, 1:10)
    @test [_slot_defined_at(1, j, pspec.variables, 3, pspec.intervals) for j in 1:10] == [falses(2); trues(6); falses(2)]

    expt = unflatten(problem.experiment, pspec, theta)
    row = findfirst(p -> string(p) == "ions.D.T", expt.trace.paths)
    @test all(expt.trace.values[row, 3:8] .=== _HD_HELD .* OTIS.EV)
    @test all(expt.trace.values[row, [1, 2, 9, 10]] .=== 200.0 * OTIS.EV)   # outside its interval: not written
    @test expt.effects[3].scale === 2.0e-6
    vt = _value_table(pspec, theta)
    @test all(vt[1, 3:8] .=== _HD_HELD) && all(isnan, vt[1, [1, 2, 9, 10]])

    # on the whole axis, every bin's value is the series'
    whole = collect(range(150.0, 330.0; length=10))
    pw, tw = _hd_problem(; T_held=whole, T_interval=0)
    ew = unflatten(pw.experiment, pw.pspec, tw)
    @test all(ew.trace.values[row, :] .=== whole .* OTIS.EV)
    @test pw.pspec.ntheta == 10
end

# ------------------------------------------------------------------------------------------
# HD5: the regularizer reads the held series as a frozen variable: constant rows, no column
# ------------------------------------------------------------------------------------------

@testset "HD5: a term selecting a held variable" begin
    problem, theta = _hd_problem()
    pspec = problem.pspec
    reg = problem.regularizer
    rows_T = reg.terms[2].rows
    raw = _regularizer_raw_rows(reg, pspec, theta)
    t = pspec.axis_boundary[3:8]
    h = _HD_HELD
    expected = [2 * ((h[k+1] - h[k]) / (t[k+1] - t[k]) - (h[k] - h[k-1]) / (t[k] - t[k-1])) / (t[k+1] - t[k-1]) / 100.0 for k in 2:5]
    @test raw[rows_T] ≈ expected rtol = 1e-12
    @test maximum(abs, raw[rows_T]) > 0
    @test raw[rows_T] == _regularizer_raw_rows(reg, pspec, theta .+ 0.3)[rows_T]
    _, rr, cc, _ = _regularizer_closed_form(reg, pspec, theta)
    @test !any(r -> r in rows_T, rr)               # no theta column reaches T_D's rows
    @test_throws "a term's interval must lie inside every selected variable's own interval" RegularizerTerm(pspec;
        select=["T_D"], operator=:D2, normalization=:unit)
end

# ------------------------------------------------------------------------------------------
# HD6: a deck's [fit.hold]
# ------------------------------------------------------------------------------------------

@testset "HD6: [fit.hold] in a deck" begin
    mktempdir() do dir
        _st_streaks(dir)
        text = _sh_epw_text(dir)
        p0, _, _ = OTIS.build_problem(read_deck(_st_write(dir, "plain.toml", text)))
        sp = _sh_start_file(dir)
        deck = read_deck(_st_write(dir, "hold.toml", _sh_tables(text; hold=(sp, ["T_e", "m_e"]))))
        problem, _, start = OTIS.build_problem(deck)
        pspec = problem.pspec
        @test pspec.ntheta == p0.pspec.ntheta - 8      # T_e's and m_e's four bins each
        @test length(start) == pspec.ntheta
        ax = read_start(sp).axis
        T = only(v for v in pspec.variables if v.name == "T_e")
        @test !T.vary && T.held == [_sh_interp(ax, _SH_SERIES["T_e"], t) for t in pspec.axis]
        expt = unflatten(problem.experiment, pspec, start)
        row = findfirst(p -> string(p) == "electrons.e.T", expt.trace.paths)
        @test all(expt.trace.values[row, :] .=== T.held .* OTIS.EV)
        mrow = findfirst(p -> string(p) == "electrons.e.f.m", expt.trace.paths)
        @test all(expt.trace.values[mrow, :] .=== 3.5)   # a one-value entry, held at every bin
        eff = effective_deck(deck, problem, LBFGS())
        @test occursin("[fit.hold]", eff)
        @test read_deck(_st_write(dir, "eff.toml", eff)) == deck
    end
end
