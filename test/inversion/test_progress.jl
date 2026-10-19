using Test
using TRUST
using TRUST: PROGRESS_WINDOW, _progress_on, _count_near_bound, _eval_all!
import TOML

# Step 4.K.1: the progress lines during a fit (Phase 4.K spec D4 to D10, D13; OQ-61, decided
# 2026-10-11). PG1 to PG8, gate "progress"; PG9, the staged deck's banners, fits a two-channel deck and
# is in the slow tier (test/slow/test_progress.jl). With `progress = n > 0` a `solve` prints a header, a
# line at every recorded row whose index is a multiple of n and a final line, each one print and one
# flush; with the lines on or off, every FitResult field but `elapsed` is the same, bitwise.

include(joinpath(@__DIR__, "..", "fixtures", "progress_fixture.jl"))

const _PG_P = _pg_problem()
const _PG_START = _pg_start(_PG_P)
const _PG_TYPES = (LevenbergMarquardt, LBFGS, NoisyGradientDescent)
const _PG_NAMES = Dict(LevenbergMarquardt => "levenberg_marquardt", LBFGS => "lbfgs",
    NoisyGradientDescent => "noisy_descent")

# The three optimizers at their defaults but `progress`, the explorer at 40 steps so that its rows
# reach past PROGRESS_WINDOW (OQ-61 (c) asks for at least 30).
_pg_opt(::Type{NoisyGradientDescent}, n) = NoisyGradientDescent(steps=40, progress=n)
_pg_opt(T::Type, n) = T(progress=n)

# One fit of _PG_P from _PG_START, its lines into an IOBuffer: (result, what it wrote).
function _pg_run(opt; kwargs...)
    buf = IOBuffer()
    result = solve(_PG_P, opt; start=_PG_START, progress_io=buf, kwargs...)
    return result, String(take!(buf))
end

const _PG_RUNS = Dict((T, n) => _pg_run(_pg_opt(T, n)) for T in _PG_TYPES for n in (0, 1, 7))

# The header each optimizer prints on _PG_P (320 pixels, 20 entries) at `every = n`.
_pg_header(::Type{LevenbergMarquardt}, n) =
    "progress levenberg_marquardt: start npixels=320 ntheta=20 cap=2000 every=$n g_abstol=1.00e-08"
_pg_header(::Type{LBFGS}, n) = "progress lbfgs: start npixels=320 ntheta=20 cap=20000 every=$n g_abstol=3.20e-04"
_pg_header(::Type{NoisyGradientDescent}, n) = "progress noisy_descent: start npixels=320 ntheta=20 cap=40 every=$n"

# The progress lines of a capture, parsed: every line but the header and the final one.
_pg_rows(text::AbstractString) = [match(_PG_LINE, l) for l in _pg_lines(text)[2:end-1]]

# ------------------------------------------------------------------------------------------
# PG1: the field, its default and the constructors' refusal
# ------------------------------------------------------------------------------------------

@testset "PG1: the field, its default and the constructors' refusal" begin
    @test PROGRESS_WINDOW == 20
    refusal = Dict(LevenbergMarquardt => "LevenbergMarquardt: progress must be an integer ≥ 0; got -1",
        LBFGS => "LBFGS: progress must be an integer ≥ 0; got -1",
        NoisyGradientDescent => "NoisyGradientDescent: progress must be an integer ≥ 0; got -1")
    for T in _PG_TYPES
        @test T().progress == 0
        @test T() === T(progress=0)
        @test T(progress=7).progress == 7
        @test_throws refusal[T] T(progress=-1)
    end
end

# ------------------------------------------------------------------------------------------
# PG2: bitwise, the lines off, on at every row and on at every seventh (OQ-61 (a))
# ------------------------------------------------------------------------------------------

@testset "PG2: the fit is bitwise the same with the lines on ($(_PG_NAMES[T]))" for T in _PG_TYPES
    r0, out0 = _PG_RUNS[(T, 0)]
    @test isempty(out0)
    for n in (1, 7)
        _pg_test_bitwise(_PG_RUNS[(T, n)][1], r0)
    end
end

# ------------------------------------------------------------------------------------------
# PG3: the lines, their count, their fields in order, and the final line (OQ-61 (b))
# ------------------------------------------------------------------------------------------

@testset "PG3: the lines ($(_PG_NAMES[T]), every $n)" for T in _PG_TYPES, n in (1, 7)
    r, out = _PG_RUNS[(T, n)]
    traj = r.trajectory
    N = length(traj)
    rows = 0:n:N-1
    lines = _pg_lines(out)
    @test endswith(out, "\n")
    @test length(lines) == fld(N - 1, n) + 3
    @test lines[1] == _pg_header(T, n)
    ms = _pg_rows(out)
    @test all(m -> m !== nothing && m[:name] == _PG_NAMES[T], ms)
    @test [parse(Int, m[:it]) for m in ms] == [Int(traj[i+1].iteration) for i in rows]
    @test [m[:chi2] for m in ms] == [@sprintf("%.6g", traj[i+1].chi2) for i in rows]
    @test [m[:chi2n] for m in ms] == [@sprintf("%.4f", traj[i+1].chi2 / _PG_P.npixels) for i in rows]
    @test [m[:R] for m in ms] == [@sprintf("%.4g", traj[i+1].R) for i in rows]
    @test [m[:obj] for m in ms] == [@sprintf("%.6g", traj[i+1].objective) for i in rows]
    @test all(m -> parse(Int, m[:nb]) <= 20 && m[:ntheta] == "20", ms)
    @test all(m -> (m[:sit] === nothing) == (m[:it] == "0"), ms)
    @test issorted([parse(Float64, m[:t]) for m in ms])
    @test all(m -> (m[:acc] !== nothing) == (T === LevenbergMarquardt) &&
                   (m[:best] !== nothing) == (T === NoisyGradientDescent), ms)
    d = match(_PG_DONE, lines[end])
    @test d !== nothing && d[:name] == _PG_NAMES[T]
    @test d[:status] == string(r.status) && d[:converged] == string(r.converged)
    @test parse(Int, d[:it]) == Int(traj[end].iteration)
    @test d[:chi2] == @sprintf("%.6g", r.chi2) && d[:chi2n] == @sprintf("%.4f", r.chi2 / r.npixels) &&
          d[:obj] == @sprintf("%.6g", r.objective)
end

# ------------------------------------------------------------------------------------------
# PG4: dchi2 and rate over the last 20 recorded rows (OQ-61 (c))
# ------------------------------------------------------------------------------------------

@testset "PG4: dchi2 and rate over the last 20 rows ($(_PG_NAMES[T]))" for T in _PG_TYPES
    r, out = _PG_RUNS[(T, 1)]
    c = [row.chi2 for row in r.trajectory]
    ms = _pg_rows(out)
    @test length(ms) == length(c)
    want_d = String[]
    want_r = String[]
    for i in 0:length(c)-1
        j = i >= 20 ? i - 20 : 0
        dchi = i == 0 ? 0.0 : c[i+1] - c[j+1]
        push!(want_d, @sprintf("%+.3e", dchi))
        push!(want_r, @sprintf("%+.3e", i == 0 ? 0.0 : dchi / min(i, 20)))
    end
    @test [m[:dchi2] for m in ms] == want_d
    @test [m[:rate] for m in ms] == want_r
    @test ms[1][:dchi2] == "+0.000e+00" && ms[1][:rate] == "+0.000e+00"
end

# ------------------------------------------------------------------------------------------
# PG5: each optimizer's own fields, against its own hook
# ------------------------------------------------------------------------------------------

@testset "PG5: each optimizer's own fields" begin
    ntheta = _PG_P.pspec.ntheta
    gmax_at(y) = (g = zeros(ntheta); _eval_all!(g, _PG_P, y); @sprintf("%.2e", maximum(abs, g)))

    # LevenbergMarquardt: acc the row's index and acc + rej its it; mu the damping going into the next
    # trial step (mu_trace's entry for it); gmax the stop's own quantity at the row's iterate
    mt = Float64[]
    buf = IOBuffer()
    r = solve(_PG_P, LevenbergMarquardt(progress=1); start=_PG_START, progress_io=buf, mu_trace=mt)
    ms = _pg_rows(String(take!(buf)))
    N = length(r.trajectory)
    @test length(ms) == N
    @test [parse(Int, m[:acc]) for m in ms] == collect(0:N-1)
    @test all(m -> parse(Int, m[:acc]) + parse(Int, m[:rej]) == parse(Int, m[:it]), ms)
    @test any(m -> parse(Int, m[:rej]) > 0, ms)
    @test r.status === :x_tolerance && parse(Int, ms[end][:acc]) + parse(Int, ms[end][:rej]) == r.iterations
    @test [m[:mu] for m in ms[1:end-1]] == [@sprintf("%.2e", mt[parse(Int, m[:it])+1]) for m in ms[1:end-1]]
    @test ms[1][:gmax] == gmax_at(_PG_START) && ms[end][:gmax] == gmax_at(r.theta)

    # LBFGS: gmax and nb are g_trace's, row for row
    gt = []
    buf = IOBuffer()
    r = solve(_PG_P, LBFGS(progress=1); start=_PG_START, progress_io=buf, g_trace=gt)
    ms = _pg_rows(String(take!(buf)))
    @test length(ms) == length(gt) == length(r.trajectory)
    @test [m[:gmax] for m in ms] == [@sprintf("%.2e", e.gnorm_inf) for e in gt]
    @test [parse(Int, m[:nb]) for m in ms] == [e.near_bound for e in gt]

    # NoisyGradientDescent: best the least objective so far; noise the step's s_k, 0 at the start;
    # gmax and nb at the step's own iterate (y_trace's), not the best one
    yt = []
    buf = IOBuffer()
    r = solve(_PG_P, NoisyGradientDescent(steps=40, progress=1); start=_PG_START, progress_io=buf, y_trace=yt)
    ms = _pg_rows(String(take!(buf)))
    ys = [[_PG_START]; yt]
    objs = [row.objective for row in r.trajectory]
    @test length(ms) == length(ys) == 41
    @test [m[:best] for m in ms] == [@sprintf("%.6g", minimum(objs[1:i+1])) for i in 0:40]
    @test [m[:noise] for m in ms] == ["0.00e+00"; [@sprintf("%.2e", 0.1 * exp(-(k - 1) / 500.0)) for k in 1:40]]
    @test [m[:gmax] for m in ms] == [gmax_at(y) for y in ys]
    @test [parse(Int, m[:nb]) for m in ms] == [_count_near_bound(_PG_P.pspec, y) for y in ys]
end

# ------------------------------------------------------------------------------------------
# PG6: one write and one flush per line; stdout looked up at the call (OQ-61 (d))
# ------------------------------------------------------------------------------------------

# An IO that counts its writes (a string's unsafe_write, or one byte's write) and its flushes, and keeps
# what was written in `buf`.
mutable struct _PGCountingIO <: IO
    buf::IOBuffer
    writes::Int
    flushes::Int
end
_PGCountingIO() = _PGCountingIO(IOBuffer(), 0, 0)
Base.unsafe_write(c::_PGCountingIO, p::Ptr{UInt8}, n::UInt) = (c.writes += 1; unsafe_write(c.buf, p, n))
Base.write(c::_PGCountingIO, x::UInt8) = (c.writes += 1; write(c.buf, x))
Base.flush(c::_PGCountingIO) = (c.flushes += 1; nothing)

@testset "PG6: one write and one flush per line, stdout by default ($(_PG_NAMES[T]))" for T in _PG_TYPES
    want = _pg_strip(_PG_RUNS[(T, 1)][2])
    c = _PGCountingIO()
    solve(_PG_P, _pg_opt(T, 1); start=_PG_START, progress_io=c)
    text = String(take!(c.buf))
    nlines = length(_pg_lines(text))
    @test c.writes == nlines
    @test c.flushes == nlines
    @test _pg_strip(text) == want
    out, _ = _pg_capture(() -> solve(_PG_P, _pg_opt(T, 1); start=_PG_START))
    @test _pg_strip(out) == want
    out0, _ = _pg_capture(() -> solve(_PG_P, _pg_opt(T, 0); start=_PG_START))
    @test isempty(out0)
end

# ------------------------------------------------------------------------------------------
# PG7: the sequence, each member's block by name, and _progress_on (OQ-61 (f))
# ------------------------------------------------------------------------------------------

@testset "PG7: the sequence prints each member's block and changes nothing" begin
    seq(n) = OptimizerSequence(NoisyGradientDescent(steps=5, progress=n), LBFGS(progress=n))
    buf0 = IOBuffer()
    r0 = solve(_PG_P, seq(0); start=_PG_START, progress_io=buf0)
    buf1 = IOBuffer()
    r1 = solve(_PG_P, seq(1); start=_PG_START, progress_io=buf1)
    @test isempty(take!(buf0))
    _pg_test_bitwise(r1, r0)

    lines = _pg_lines(String(take!(buf1)))
    n2 = length(r1.trajectory) - 6        # LBFGS's own rows, after the explorer's six
    @test length(lines) == (1 + 6 + 1) + (1 + n2 + 1)
    @test lines[1] == "progress noisy_descent: start npixels=320 ntheta=20 cap=5 every=1"
    @test all(l -> startswith(l, "progress noisy_descent: it="), lines[2:7])
    @test startswith(lines[8], "progress noisy_descent: done status=steps converged=false it=5 ")
    @test lines[9] == "progress lbfgs: start npixels=320 ntheta=20 cap=20000 every=1 g_abstol=3.20e-04"
    ms = [match(_PG_LINE, l) for l in lines[10:end-1]]
    @test all(m -> m !== nothing && m[:name] == "lbfgs", ms)
    @test [parse(Int, m[:it]) for m in ms] == collect(0:n2-1)
    @test [m[:chi2] for m in ms] == [@sprintf("%.6g", row.chi2) for row in r1.trajectory[7:end]]
    @test startswith(lines[end], "progress lbfgs: done status=$(r1.status) ")

    @test _progress_on(seq(1)) && !_progress_on(seq(0))
    @test _progress_on(OptimizerSequence(NoisyGradientDescent(), LBFGS(progress=3)))
    @test _progress_on(LevenbergMarquardt(progress=1)) && !_progress_on(LevenbergMarquardt())
end

# ------------------------------------------------------------------------------------------
# PG8: the deck key, its effective deck and its refusals (OQ-61 (e))
# ------------------------------------------------------------------------------------------

# test_deck.jl's minimal deck (a flat 8 x 4 streak, T_D and u_D scalar), its [optimizer] body given.
function _pg_deck(dir::AbstractString, name::AbstractString, optimizer::AbstractString)
    datafile = joinpath(dir, "data.jld2")
    isfile(datafile) || write_streak(datafile, StreakData(;
        lambda_m=collect(range(262.0, 264.5; length=8)) .* 1e-9, t_s=collect(range(0.0, 3.0; length=4)) .* 1e-9,
        image=fill(30.0, 8, 4), sigma=fill(1.0, 8, 4)); source="test")
    path = joinpath(dir, name)
    write(path, """
    deck_version = "trust-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0 }
    u_D = { bounds = [-200.0, 200.0], init = 0.0 }

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

    [fit]
    sigma = "supplied"

    [optimizer]
    $optimizer

    [output]
    dir = "$(replace(joinpath(dir, "out"), "\\" => "/"))"
    """)
    return path
end

@testset "PG8: the deck key" begin
    mktempdir() do dir
        deck = read_deck(_pg_deck(dir, "on.toml",
            "procedure = \"levenberg_marquardt\"\nlevenberg_marquardt = { progress = 10 }"))
        @test deck.sections["optimizer"]["levenberg_marquardt"]["progress"] == 10
        problem, opt, _ = TRUST.build_problem(deck)
        @test opt isa LevenbergMarquardt && opt.progress == 10
        eff = effective_deck(deck, problem, opt)
        @test TOML.parse(eff)["optimizer"]["levenberg_marquardt"]["progress"] == 10
        effpath = joinpath(dir, "eff.toml")
        write(effpath, eff)
        @test read_deck(effpath) == deck

        # a table without the key: the default, 0, filled into the validated table and the effective deck
        deck0 = read_deck(_pg_deck(dir, "off.toml", "procedure = \"lbfgs\"\nlbfgs = { iterations = 3 }"))
        p0, o0, _ = TRUST.build_problem(deck0)
        @test deck0.sections["optimizer"]["lbfgs"]["progress"] == 0 && o0.progress == 0
        @test TOML.parse(effective_deck(deck0, p0, o0))["optimizer"]["lbfgs"]["progress"] == 0

        # below zero, refused by read_deck in each table; a non-integer, by the integer check
        for key in ("levenberg_marquardt", "lbfgs", "noisy_descent")
            path = _pg_deck(dir, "neg_$key.toml", "procedure = \"lbfgs\"\n$key = { progress = -1 }")
            @test_throws "read_deck: key \"progress\" in section \"optimizer.$key\" must be an integer ≥ 0; got -1" read_deck(path)
        end
        @test_throws "read_deck: key \"progress\" in section \"optimizer.lbfgs\" must be an integer; got 2.5" read_deck(
            _pg_deck(dir, "real.toml", "procedure = \"lbfgs\"\nlbfgs = { progress = 2.5 }"))
        @test_throws "read_deck: key \"progress\" in section \"optimizer.lbfgs\" must be an integer; got \"10\"" read_deck(
            _pg_deck(dir, "string.toml", "procedure = \"lbfgs\"\nlbfgs = { progress = \"10\" }"))
    end
end
