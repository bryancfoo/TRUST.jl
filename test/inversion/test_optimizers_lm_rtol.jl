using Test
using TRUST
using TRUST: _eval_all!, _warn_iteration_cap
using Random: Xoshiro, randn
import TOML

# Step 4.L.1: the stall test for LevenbergMarquardt (Phase 4.L spec D4 to D8, D11; OQ-62, decided
# 2026-10-13). FT1 to FT10, gate "lm_rtol"; the housekeeping row is test_readable.jl's RD14. With
# lm_rtol > 0, solve ends a fit whose objective has moved by at most lm_rtol of itself over the last
# lm_rtol_window accepted steps, status = :lm_rtol, converged = true, termination untouched; with
# lm_rtol = 0.0, the default, every result is bitwise what it was. A fit the test ends is, up to that
# row, the fit without it, bitwise.

include(joinpath(@__DIR__, "..", "fixtures", "progress_fixture.jl"))

# Every fit in this file runs with BLAS on one thread, the count fit(deck) pins (conventions.md 6.6):
# LevenbergMarquardt's dot and norm sum in another order on more threads, so a trajectory depends on
# the count, and the fixture's below were measured on one. The count found is restored at the end.
const _FT_NBLAS = pin_blas_threads(1)

# The stall fixture (spec D7): the Phase 4.K fixture's streak (P2's noise-free image of the T_D ramp
# 220 to 260 eV and the u_D ramp -20 to 20 km/s, sigma from GaussianNoise(a = 1, b = 0.02)) fitted from
# a deck that bounds T_D above at 258 eV, from T_D = 80 eV and u_D = -250 km/s in every bin. From there
# LevenbergMarquardt rejects ten of its first 33 trial steps, settles far from the generating values
# (objective near 3.2556e6) and crawls: over four accepted steps its objective moves by 3.1e-6 of itself
# at row 22, 6.5e-7 at row 23 (attempt 33) and less after, and neither D44 nor any other test ends it
# before the 60-attempt cap. Measured through Pkg.test()'s sandbox, whose test/Manifest.toml the
# trajectories depend on to the last bit (docs/derivations/uncertainty.md Section 8).
const _FT_DIR = mktempdir()
const _FT_DATA = joinpath(_FT_DIR, "ft.jld2")
write_streak(_FT_DATA, _pg_problem().data; source="test")

_ft_path(p::AbstractString) = replace(p, "\\" => "/")

function _ft_deck_text(outdir::AbstractString, optimizer::AbstractString)
    return """
    deck_version = "trust-deck-2"

    [data]
    file = "$(_ft_path(_FT_DATA))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    k_model = "exact"

    [instrument]
    spectral_irf = { fwhm_nm = 0.08 }
    temporal_irf = { fwhm_ns = 0.4 }
    sensitivity = { scale = 1.0e-6 }

    [fit.variables]
    T_D = { bounds = [50.0, 258.0], init = 80.0, time_varying = true }
    u_D = { bounds = [-300.0, 300.0], init = -250.0, time_varying = true }

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

    [fit]
    sigma = "supplied"

    [regularizer]
    weight = 1.0e-3
    terms = [
      { select = ["T_D"], operator = "D2", normalization = "unit" },
    ]

    [optimizer]
    $optimizer

    [output]
    dir = "$(_ft_path(outdir))"
    """
end

function _ft_deck(name::AbstractString, optimizer::AbstractString)
    path = joinpath(_FT_DIR, name)
    write(path, _ft_deck_text(joinpath(_FT_DIR, "out_" * first(splitext(name))), optimizer))
    return path
end

const _FT_LM = "procedure = \"levenberg_marquardt\""
const _FT_BUILT = TRUST.build_problem(read_deck(_ft_deck("base.toml", _FT_LM)))
const _FT_P = _FT_BUILT[1]
const _FT_START = _FT_BUILT[3]

# (b)'s values (spec D7): the cap N, the tolerance rho and the window w
const _FT_N = 60
const _FT_RHO = 1.5e-6
const _FT_W = 4

# LevenbergMarquardt with only D44, the cap and (when lm_rtol > 0) the window able to end the fit
_ft_lm(; kwargs...) = LevenbergMarquardt(; iterations=_FT_N, g_abstol=0.0, x_reltol=0.0, kwargs...)
_ft_on(; kwargs...) = _ft_lm(; lm_rtol=_FT_RHO, lm_rtol_window=_FT_W, kwargs...)

const _FT_REF = solve(_FT_P, _ft_lm(); start=_FT_START)
const _FT_ON = solve(_FT_P, _ft_on(); start=_FT_START)

# OQ-62 item 1's inequality at row i of a trajectory's objectives O (O[i + 1] the row of index i)
_ft_holds(O, i, w, rho) = abs(O[i+1-w] - O[i+1]) <= rho * abs(O[i+1])
_ft_ratio(O, i, w) = abs(O[i+1-w] - O[i+1]) / abs(O[i+1])

# Every field _pg_test_bitwise compares but status, termination and converged: the iterate and its record
function _ft_test_same_fit(a::FitResult, b::FitResult)
    @test length(a.theta) == length(b.theta) && all(a.theta .=== b.theta)
    @test length(a.trajectory) == length(b.trajectory) &&
          all(all(values(x) .=== values(y)) for (x, y) in zip(a.trajectory, b.trajectory))
    @test a.objective === b.objective && a.chi2 === b.chi2 && a.R === b.R
    @test a.iterations === b.iterations && a.f_calls === b.f_calls && a.g_calls === b.g_calls
    return nothing
end

# (c)'s problem (U1): the Phase 4.K fixture's, with noise sigma .* randn(Xoshiro(1)) drawn into its image
function _ft_noisy_problem()
    p = _pg_problem()
    d = p.data
    image = d.image .+ d.sigma .* randn(Xoshiro(1), size(d.image)...)
    data = StreakData(lambda_m=d.lambda, t_s=d.t, image=image, sigma=d.sigma)
    return FitProblem(p.experiment, p.pspec, data; regularizer=p.regularizer)
end

# ------------------------------------------------------------------------------------------
# FT1: the fields, their defaults and the constructor's refusals (OQ-62 (a), (d))
# ------------------------------------------------------------------------------------------

@testset "FT1: the fields, their defaults and the constructor's refusals" begin
    @test fieldnames(LevenbergMarquardt)[end-2:end] == (:progress, :lm_rtol, :lm_rtol_window)
    @test LevenbergMarquardt().lm_rtol === 0.0
    @test LevenbergMarquardt().lm_rtol_window === 50
    @test LevenbergMarquardt() === LevenbergMarquardt(lm_rtol=0.0, lm_rtol_window=50)
    o = LevenbergMarquardt(lm_rtol=1e-6, lm_rtol_window=20)
    @test o.lm_rtol === 1e-6 && o.lm_rtol_window === 20
    @test LevenbergMarquardt(lm_rtol=0).lm_rtol === 0.0
    @test_throws "LevenbergMarquardt: lm_rtol must be a finite number ≥ 0; got -1.0" LevenbergMarquardt(lm_rtol=-1.0)
    @test_throws "LevenbergMarquardt: lm_rtol must be a finite number ≥ 0; got NaN" LevenbergMarquardt(lm_rtol=NaN)
    @test_throws "LevenbergMarquardt: lm_rtol must be a finite number ≥ 0; got Inf" LevenbergMarquardt(lm_rtol=Inf)
    @test_throws "LevenbergMarquardt: lm_rtol_window must be an integer ≥ 1; got 0" LevenbergMarquardt(lm_rtol_window=0)
end

# ------------------------------------------------------------------------------------------
# FT2: the key off is bitwise today's fit (OQ-62 (a))
# ------------------------------------------------------------------------------------------

@testset "FT2: lm_rtol = 0.0 is bitwise the default, on the Phase 4.K fixture" begin
    p = _pg_problem()
    r0 = solve(p, LevenbergMarquardt(); start=_pg_start(p))
    r1 = solve(p, LevenbergMarquardt(lm_rtol=0.0, lm_rtol_window=1); start=_pg_start(p))
    _pg_test_bitwise(r1, r0)
    @test r0.status === :x_tolerance && length(r0.trajectory) == 10 && r0.iterations == 13
end

# ------------------------------------------------------------------------------------------
# FT3: the stop, and where it stops (OQ-62 (b))
# ------------------------------------------------------------------------------------------

@testset "FT3: the window stops the fit at the first row the inequality holds" begin
    O = [row.objective for row in _FT_REF.trajectory]
    # the reference: only the cap ended it, so the fixture's stop lies inside it with rows to spare
    @test _FT_REF.status === :iterations && _FT_REF.iterations == _FT_N
    i = findfirst(j -> j >= _FT_W && _ft_holds(O, j, _FT_W, _FT_RHO), 0:length(O)-1) - 1
    @test i < length(O) - 1
    # the first row is unambiguous: the ratio at it is under rho / 2, at every earlier row over 2 rho
    @test _ft_ratio(O, i, _FT_W) <= _FT_RHO / 2
    @test minimum(_ft_ratio(O, j, _FT_W) for j in _FT_W:i-1) >= 2 * _FT_RHO
    r = _FT_ON
    @test r.status === :lm_rtol && r.converged === true && r.termination === :none
    @test length(r.trajectory) == i + 1
    @test all(all(values(a) .=== values(b)) for (a, b) in zip(r.trajectory, _FT_REF.trajectory[1:i+1]))
    @test Int(r.trajectory[end].iteration) > i
    @test r.iterations == Int(r.trajectory[end].iteration)
    @test r.f_calls == r.iterations + 1 && r.g_calls == i + 1
end

# ------------------------------------------------------------------------------------------
# FT4: the capped reruns: the key-off fit capped there is the same fit, and the cap fires first
# ------------------------------------------------------------------------------------------

@testset "FT4: capped at the stop's attempt count, the key-off fit is the same and the cap precedes the window" begin
    K = _FT_ON.iterations
    off = solve(_FT_P, _ft_lm(iterations=K); start=_FT_START)
    @test off.status === :iterations && off.converged === false
    _ft_test_same_fit(off, _FT_ON)
    # the key on at the same cap: the cap ends the loop before the next loop top's window test
    on = solve(_FT_P, _ft_on(iterations=K); start=_FT_START)
    _pg_test_bitwise(on, off)
end

# ------------------------------------------------------------------------------------------
# FT5: the gradient test precedes the window at the same loop top (OQ-62 item 4)
# ------------------------------------------------------------------------------------------

@testset "FT5: the gradient stop fires first where both hold" begin
    # the gradient stop's own quantity, 2 maximum(abs, J'r), at the stop row's iterate: a running
    # minimum over the rows, so g_abstol set to it fires at that row's loop top and no earlier
    g = zeros(_FT_P.pspec.ntheta)
    _eval_all!(g, _FT_P, _FT_ON.theta)
    gstop = maximum(abs, g)
    off = solve(_FT_P, _ft_lm(g_abstol=gstop); start=_FT_START)
    on = solve(_FT_P, _ft_on(g_abstol=gstop); start=_FT_START)
    @test on.status === :gradient_norm
    @test length(on.trajectory) == length(_FT_ON.trajectory)
    _pg_test_bitwise(on, off)
end

# ------------------------------------------------------------------------------------------
# FT6: the progress lines' final line names the stop, and the fit is unchanged (OQ-62 (b), OQ-61 (a))
# ------------------------------------------------------------------------------------------

@testset "FT6: the final progress line reads status=lm_rtol converged=true" begin
    buf = IOBuffer()
    r = solve(_FT_P, _ft_on(progress=1); start=_FT_START, progress_io=buf)
    _pg_test_bitwise(r, _FT_ON)
    lines = _pg_lines(String(take!(buf)))
    d = match(_PG_DONE, lines[end])
    @test d !== nothing && d[:status] == "lm_rtol" && d[:converged] == "true"
    @test parse(Int, d[:it]) == Int(_FT_ON.trajectory[end].iteration)
    @test length(lines) == length(_FT_ON.trajectory) + 2
end

# ------------------------------------------------------------------------------------------
# FT7: D44 is untouched (OQ-62 (c), U1)
# ------------------------------------------------------------------------------------------

@testset "FT7: a restart that D44 ends at its first attempt ends there with the key on" begin
    p = _ft_noisy_problem()
    first_fit = solve(p, LevenbergMarquardt(); start=_pg_start(p))
    @test first_fit.status === :f_tolerance
    off = solve(p, LevenbergMarquardt(); start=first_fit.theta)
    on = solve(p, LevenbergMarquardt(lm_rtol=1.0, lm_rtol_window=1); start=first_fit.theta)
    @test off.status === :f_tolerance && off.iterations == 1 && length(off.trajectory) == 1
    @test on.status === :f_tolerance && on.termination === :none
    _pg_test_bitwise(on, off)
end

# ------------------------------------------------------------------------------------------
# FT8: the deck keys, the effective deck and read_deck's refusals (OQ-62 (d))
# ------------------------------------------------------------------------------------------

@testset "FT8: the deck keys" begin
    deck = read_deck(_ft_deck("keys.toml",
        "$_FT_LM\nlevenberg_marquardt = { lm_rtol = 1e-6, lm_rtol_window = 20 }"))
    lm = deck.sections["optimizer"]["levenberg_marquardt"]
    @test lm["lm_rtol"] === 1e-6 && lm["lm_rtol_window"] === 20
    problem, opt, _ = TRUST.build_problem(deck)
    @test opt isa LevenbergMarquardt && opt.lm_rtol === 1e-6 && opt.lm_rtol_window === 20
    eff = effective_deck(deck, problem, opt)
    efflm = TOML.parse(eff)["optimizer"]["levenberg_marquardt"]
    @test efflm["lm_rtol"] === 1e-6 && efflm["lm_rtol_window"] === 20
    effpath = joinpath(_FT_DIR, "keys_eff.toml")
    write(effpath, eff)
    @test read_deck(effpath) == deck

    # a table without the keys: the defaults in the validated table and the effective deck
    deck0 = read_deck(_ft_deck("nokeys.toml", "$_FT_LM\nlevenberg_marquardt = { iterations = 3 }"))
    p0, o0, _ = TRUST.build_problem(deck0)
    @test deck0.sections["optimizer"]["levenberg_marquardt"]["lm_rtol"] === 0.0
    @test deck0.sections["optimizer"]["levenberg_marquardt"]["lm_rtol_window"] === 50
    eff0 = TOML.parse(effective_deck(deck0, p0, o0))["optimizer"]["levenberg_marquardt"]
    @test eff0["lm_rtol"] === 0.0 && eff0["lm_rtol_window"] === 50

    # a TOML integer reads by the Float64 rule, and the effective deck writes it as a float
    deckz = read_deck(_ft_deck("zero.toml", "$_FT_LM\nlevenberg_marquardt = { lm_rtol = 0 }"))
    @test deckz.sections["optimizer"]["levenberg_marquardt"]["lm_rtol"] === 0.0
    pz, oz, _ = TRUST.build_problem(deckz)
    @test occursin("lm_rtol = 0.0\n", effective_deck(deckz, pz, oz))

    # refused at read time, with the section, the key and the value
    for (text, shown) in (("-1.0", "-1.0"), ("nan", "NaN"), ("inf", "Inf"))
        path = _ft_deck("bad.toml", "$_FT_LM\nlevenberg_marquardt = { lm_rtol = $text }")
        @test_throws "read_deck: key \"lm_rtol\" in section \"optimizer.levenberg_marquardt\" must be a finite number ≥ 0; got $shown" read_deck(path)
    end
    @test_throws "read_deck: key \"lm_rtol_window\" in section \"optimizer.levenberg_marquardt\" must be an integer ≥ 1; got 0" read_deck(
        _ft_deck("bad.toml", "$_FT_LM\nlevenberg_marquardt = { lm_rtol_window = 0 }"))
    # a value of the wrong type, by the existing checks
    @test_throws "read_deck: key \"lm_rtol\" in section \"optimizer.levenberg_marquardt\" takes a string, not a variable reference" read_deck(
        _ft_deck("bad.toml", "$_FT_LM\nlevenberg_marquardt = { lm_rtol = \"1e-6\" }"))
    @test_throws "read_deck: key \"lm_rtol_window\" in section \"optimizer.levenberg_marquardt\" must be an integer; got 2.5" read_deck(
        _ft_deck("bad.toml", "$_FT_LM\nlevenberg_marquardt = { lm_rtol_window = 2.5 }"))
    # LevenbergMarquardt's keys only
    @test_throws "read_deck: unknown key \"lm_rtol\" in section \"optimizer.lbfgs\"" read_deck(
        _ft_deck("bad.toml", "procedure = \"lbfgs\"\nlbfgs = { lm_rtol = 1e-6 }"))
end

# ------------------------------------------------------------------------------------------
# FT9: the sequence passes the status through (OQ-62 (e))
# ------------------------------------------------------------------------------------------

@testset "FT9: an explorer then LevenbergMarquardt ends on the window" begin
    # five steps of plain gradient descent (noise 0), then (b)'s key-on LevenbergMarquardt, which the
    # window stops at its own row 16 (spec D7)
    r = solve(_FT_P, OptimizerSequence(NoisyGradientDescent(steps=5, noise=0.0), _ft_on()); start=_FT_START)
    @test r.status === :lm_rtol && r.converged === true && r.termination === :none
end

# ------------------------------------------------------------------------------------------
# FT10: fit(deck) (OQ-62 (f))
# ------------------------------------------------------------------------------------------

@testset "FT10: fit(deck) ends on the window and the result file says so" begin
    path = _ft_deck("fit.toml", "$_FT_LM\nlevenberg_marquardt = { iterations = $_FT_N, g_abstol = 0.0, " *
                                "x_reltol = 0.0, lm_rtol = $_FT_RHO, lm_rtol_window = $_FT_W }")
    deck = read_deck(path)
    result = Base.CoreLogging.with_logger(() -> fit(deck), Base.CoreLogging.NullLogger())
    @test result.status === :lm_rtol && result.converged === true
    _pg_test_bitwise(result, _FT_ON)
    @test _warn_iteration_cap(result, "the final fit") === false
    dir = deck.sections["output"]["dir"]
    rf = read_result(joinpath(dir, "result.jld2"))
    @test rf.fit.status == "lm_rtol" && rf.fit.termination == "none" && rf.fit.converged == 1
    efflm = TOML.parse(read(joinpath(dir, "effective_deck.toml"), String))["optimizer"]["levenberg_marquardt"]
    @test efflm["lm_rtol"] === _FT_RHO && efflm["lm_rtol_window"] === _FT_W
end

pin_blas_threads(_FT_NBLAS)
