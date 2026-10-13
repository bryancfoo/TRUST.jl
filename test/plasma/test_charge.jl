using Test
using OTIS
using OTIS: _divide_by_charge, _widen_trace, LeafRef, _resolve_leaf, _leaf_value, _leaf_si_path,
    _leaf_display_unit, _enumerate_leaves, _with_leaf
import ForwardDiff
import TOML
using JLD2: jldopen

# Phase 4.J (OQ-60): the continuous ion charge, gated "continuous charge". Step 4.J.1, CZ1 to CZ8 (Phase
# 4.J spec D12): ion's real charge and its floor, today's model bitwise, the guarded division by a
# charge, the identity with the C4/C6 pair and its contrast at the number average, the trace layout
# otis-trace-2, the deck's real literal. Step 4.J.2, CZ9 to CZ13 (spec D18): the leaf, a trace whose
# charge differs per bin, _with_leaf and GradientSmearing, the binding and its floor, the deck's
# charge = "ZC". The derivative and the recovery are in test/slow/test_charge.jl.

include(joinpath(@__DIR__, "..", "fixtures", "charge_fixture.jl"))

const _CZ_MSG = "an ion's charge number must be a finite real ≥ 1: for a mixture of charge states it is ⟨Z²⟩/⟨Z⟩, never below 1"

# ------------------------------------------------------------------------------------------
# CZ1: ion takes a real charge >= 1 and refuses the rest (OQ-60 (b); spec D7)
# ------------------------------------------------------------------------------------------

@testset "CZ1: ion's real charge and its floor" begin
    for c in (1, 1.0, 4.6, 6, 6.0, 9 // 2)
        sp = ion(name="C", mass_amu=12.011, charge=c, T_eV=1000.0)
        @test sp.charge === Float64(c)
        @test sp isa SpeciesSpec{Float64,Maxwellian}
    end
    @test electrons(T_eV=500.0).charge === -1.0
    d = ForwardDiff.Dual(4.6, 1.0)
    spd = ion(name="C", mass_amu=12.011, charge=d, T_eV=1000.0)
    @test spd isa SpeciesSpec{typeof(d),Maxwellian}
    @test spd.charge === d
    for (c, shown) in ((0.5, "0.5"), (0, "0"), (-1, "-1"), (NaN, "NaN"), (Inf, "Inf"))
        @test_throws "ion \"C\" has charge $shown; $_CZ_MSG" ion(name="C", mass_amu=12.011, charge=c, T_eV=1000.0)
    end
    @test_throws _CZ_MSG ion(name="C", mass_amu=12.011, charge=ForwardDiff.Dual(0.5, 1.0), T_eV=1000.0)
end

# ------------------------------------------------------------------------------------------
# CZ2: charge = 6 and charge = 6.0 are one model, bitwise (OQ-60 (a))
# ------------------------------------------------------------------------------------------

@testset "CZ2: charge = 6 and charge = 6.0 are one model, bitwise" begin
    a = ion(name="C", mass_amu=12.011, charge=6, T_eV=_CZ_TC_EV, fraction=0.7)
    b = ion(name="C", mass_amu=12.011, charge=6.0, T_eV=_CZ_TC_EV, fraction=0.7)
    @test all(getfield(a, k) === getfield(b, k) for k in (:mass, :charge, :u, :T, :f, :fraction))
    pa, pb = _cz_single(6, 0.7), _cz_single(6.0, 0.7)
    @test pa.ions["C"].n === pb.ions["C"].n
    sweep = _cz_sweep()
    @test all(spectral_density(pa, k, w) === spectral_density(pb, k, w) for (k, w) in sweep)
    @test all(dielectric(pa, k, w).epsilon === dielectric(pb, k, w).epsilon for (k, w) in sweep)
    mktempdir() do dir
        _cz_streak(dir)
        models = map(("6", "6.0")) do c
            deck = read_deck(_cz_write(dir, "c$(c).toml", _cz_deck_text(dir; charge=c)))
            problem, _, start = OTIS.build_problem(deck)
            measure(unflatten(problem.experiment, problem.pspec, start))
        end
        @test all(models[1] .=== models[2])
    end
end

# ------------------------------------------------------------------------------------------
# CZ3: _divide_by_charge divides by a charge without partials as the Int did (spec D5, U1)
# ------------------------------------------------------------------------------------------

@testset "CZ3: _divide_by_charge, the guarded division" begin
    @test _divide_by_charge(3.5, 6.0) === 3.5 / 6.0
    dual(v, p...) = ForwardDiff.Dual{:CZ3}(v, p...)
    ps = 1.0:1.0:50.0
    for Z in (6.0, 4.0)
        guarded = [_divide_by_charge(dual(3.0, p, 2p), dual(Z, 0.0, 0.0)) for p in ps]
        as_int = [dual(3.0, p, 2p) / Int(Z) for p in ps]          # today's arithmetic: the charge an Int
        @test all(guarded .=== as_int)
        plain = [dual(3.0, p, 2p) / dual(Z, 0.0, 0.0) for p in ps]  # the dual division the guard avoids
        # it differs in the last place for some p at Z = 6 and for none at Z = 4, a power of two
        @test any(plain .!== as_int) == (Z == 6.0)
    end
    x, c = dual(3.0, 2.0, 1.0), dual(6.0, 1.0, 0.0)              # a fitted charge: the dual division
    q = _divide_by_charge(x, c)
    @test q === x / c
    @test isapprox(ForwardDiff.partials(q)[1], (2.0 * 6.0 - 3.0 * 1.0) / 36; rtol=1e-15)
    @test isapprox(ForwardDiff.partials(q)[2], (1.0 * 6.0 - 3.0 * 0.0) / 36; rtol=1e-15)
end

# ------------------------------------------------------------------------------------------
# CZ4: the two callers keep today's dual arithmetic for a charge without partials (spec D5)
# ------------------------------------------------------------------------------------------

@testset "CZ4: an unfitted charge's density keeps today's dual arithmetic" begin
    dual(v, p) = ForwardDiff.Dual{:CZ4}(v, p)
    D = typeof(dual(1.0, 1.0))
    # Plasma's keyword constructor on a dual spec: the charge is a dual without partials
    p = Plasma(n_e_m3=dual(5.0e25, 1.0e25), electrons=(electrons(T_eV=_CZ_TE_EV),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=dual(_CZ_TD_EV, 0.0), fraction=0.3),
            ion(name="C", mass_amu=12.011, charge=6, T_eV=dual(_CZ_TC_EV, 0.0), fraction=0.7)))
    @test p.ions["C"].charge isa D && iszero(ForwardDiff.partials(p.ions["C"].charge))
    @test p.ions["C"].n === D(0.7) * p.n_e / 6
    @test p.ions["D"].n === D(0.3) * p.n_e / 1
    # plasma(trace, j) on a widened trace whose n_e row is dual
    trace = PlasmaTrace(_cz_single(6, 0.7); t_ns=[0.0, 1.0], leaves=["n_e_cm3" => [5.0e19, 6.0e19]])
    work = _widen_trace(trace, D)
    work.values[1, 2] = dual(6.0e25, 1.0e25)
    q = plasma(work, 2)
    @test q.ions["C"].n === q.ions["C"].fraction * q.n_e / 6
    @test q.ions["D"].n === q.ions["D"].fraction * q.n_e / 1
end

# ------------------------------------------------------------------------------------------
# CZ5: one carbon at Z* = <Z^2>/<Z> is the C4/C6 pair, to round-off (OQ-60 (c))
# ------------------------------------------------------------------------------------------

@testset "CZ5: one carbon at Z* is the C4/C6 pair" begin
    worst = 0.0
    sweep = _cz_sweep()
    for shape in (:maxwellian, :kappa), (f4, f6) in ((0.35, 0.35), (0.1, 0.5))
        pair = _cz_pair(f4, f6; shape=shape)
        single = _cz_single(_cz_zstar(f4, f6), f4 + f6; shape=shape)
        @test isapprox(zeff(single), zeff(pair); rtol=1e-14)
        for (k, w) in sweep
            dp, ds = dielectric(pair, k, w), dielectric(single, k, w)
            @test dp.chi_e === ds.chi_e
            @test isapprox(ds.epsilon, dp.epsilon; rtol=1e-12)
            Sp, Ss = spectral_density(pair, k, w), spectral_density(single, k, w)
            @test isapprox(Ss, Sp; rtol=1e-12)
            worst = max(worst, abs(Ss - Sp) / abs(Sp))
        end
        for window in (:epw, :iaw)
            Mp, Ms = _cz_measure(pair, window), _cz_measure(single, window)
            floor_m = 1e-12 * maximum(abs, Mp)
            @test all(isapprox.(Ms, Mp; rtol=1e-12, atol=floor_m))
            worst = max(worst, maximum(abs.(Ms .- Mp) ./ max.(abs.(Mp), floor_m)))
        end
    end
    @info "CZ5: the largest relative difference, the pair against one carbon at Z*" worst
end

# ------------------------------------------------------------------------------------------
# CZ6: one carbon at the number average Zbar is not the pair (OQ-60 (c), the contrast)
# ------------------------------------------------------------------------------------------

@testset "CZ6: one carbon at Zbar is a different spectrum" begin
    f4, f6 = 0.35, 0.35
    pair = _cz_pair(f4, f6)
    zstar, zbar = _cz_zstar(f4, f6), _cz_zbar(f4, f6)
    @test isapprox(zstar, 5.0; rtol=1e-15) && isapprox(zbar, 4.8; rtol=1e-15)
    at_zbar = _cz_single(zbar, f4 + f6)
    # the carbon's ion weight (zeff less the deuterium's, which is its fraction) scales by Zbar/Z*
    wC_pair = zeff(pair) - (1 - (f4 + f6))
    wC_zbar = zeff(at_zbar) - (1 - (f4 + f6))
    @test isapprox(wC_zbar / wC_pair, zbar / zstar; rtol=1e-12)
    sweep = _cz_sweep()
    S_pair = [spectral_density(pair, k, w) for (k, w) in sweep]
    S_zbar = [spectral_density(at_zbar, k, w) for (k, w) in sweep]
    diff = maximum(abs.(S_zbar .- S_pair)) / maximum(abs, S_pair)
    @info "CZ6: one carbon at Zbar against the pair: the largest difference over the pair's peak; Zbar/Z*" diff ratio = zbar / zstar
    @test diff > 1e-3
end

# ------------------------------------------------------------------------------------------
# CZ7: the trace layout otis-trace-2 (OQ-60 (b), (f), item 6; spec D9)
# ------------------------------------------------------------------------------------------

@testset "CZ7: otis-trace-2 carries a real charge bitwise; otis-trace-1 is refused" begin
    @test OTIS.TRACE_LAYOUT_VERSION == "otis-trace-2"
    trace = PlasmaTrace(_cz_single(4.6, 0.7); t_ns=[0.0, 1.0, 2.0], leaves=["n_e_cm3" => [5.0e19, 5.5e19, 6.0e19]])
    mktempdir() do dir
        path = joinpath(dir, "trace.jld2")
        write_trace(path, trace; source="CZ7")
        r = read_trace(path)
        @test r.layout_version == "otis-trace-2"
        @test r.template["ions"]["species"]["C"]["charge"] === 4.6
        @test r.template["ions"]["species"]["D"]["charge"] === 1.0
        @test r.template["electrons"]["species"]["e"]["charge"] === -1.0
        rebuilt = read_trace(path; as=:trace)
        @test rebuilt.template.ions["C"].charge === 4.6
        @test all(plasma(rebuilt, j).ions["C"].n === plasma(trace, j).ions["C"].n for j in 1:3)
        old = joinpath(dir, "old.jld2")
        jldopen(old, "w") do f
            f["layout_version"] = "otis-trace-1"
        end
        @test_throws "has layout_version \"otis-trace-1\"; expected \"otis-trace-2\"" read_trace(old)
    end
end

# ------------------------------------------------------------------------------------------
# CZ8: the deck's charge is an integer, a real or a reference (OQ-60 (b), (g), item 2; spec D8)
# ------------------------------------------------------------------------------------------

@testset "CZ8: the deck's real charge" begin
    mktempdir() do dir
        _cz_streak(dir)
        deck = read_deck(_cz_write(dir, "real.toml", _cz_deck_text(dir; charge="4.6")))
        @test deck.sections["plasma"]["ions"]["C"]["charge"] === 4.6
        problem, optimizer, _ = OTIS.build_problem(deck)
        @test problem.pspec.trace.template.ions["C"].charge === 4.6
        @test read_deck(_cz_write(dir, "eff.toml", effective_deck(deck, problem, optimizer))) == deck
        six = read_deck(_cz_write(dir, "six.toml", _cz_deck_text(dir; charge="6")))
        @test six.sections["plasma"]["ions"]["C"]["charge"] === 6.0
        low = read_deck(_cz_write(dir, "low.toml", _cz_deck_text(dir; charge="0.5")))
        @test_throws "ion \"C\" has charge 0.5; $_CZ_MSG" OTIS.build_problem(low)
        @test_throws "read_deck: key \"charge\" in section \"plasma.ions.C\" refers to \"nope\", which is not a declared variable or dependent" read_deck(
            _cz_write(dir, "nope.toml", _cz_deck_text(dir; charge="\"nope\"")))
    end
end

# ------------------------------------------------------------------------------------------
# CZ9: ions.<name>.charge is a leaf; an electron's charge and every mass are not (OQ-60 (b), item 3)
# ------------------------------------------------------------------------------------------

@testset "CZ9: the charge leaf" begin
    p = _cz_single(5.0, 0.7)
    ref = _resolve_leaf(p, LeafPath("ions.C.charge"))
    @test ref == LeafRef(:ions, 2, :charge, false)
    @test _leaf_value(p, ref) === 5.0
    @test _leaf_si_path("ions.C.charge") == ("ions.C.charge", 1.0)
    @test _leaf_display_unit(LeafPath("ions.C.charge")) == ("", 1.0)
    @test_throws "\"electrons.e.charge\": an electron species' charge is -1 by definition and is not a fittable leaf (only an ion's charge is)" _resolve_leaf(
        p, LeafPath("electrons.e.charge"))
    @test_throws "\"ions.D.mass\": mass is not a fittable leaf" _resolve_leaf(p, LeafPath("ions.D.mass"))
    @test_throws "\"electrons.e.mass\": mass is not a fittable leaf" _resolve_leaf(p, LeafPath("electrons.e.mass"))
    @test_throws "\"ions.D.bogus\" is not a leaf; expected T, u, fraction, charge (ions) or f.<parameter>" _resolve_leaf(
        p, LeafPath("ions.D.bogus"))
    paths = [string(path) for (_, path) in _enumerate_leaves(p)]
    @test "ions.D.charge" in paths && "ions.C.charge" in paths && !("electrons.e.charge" in paths)
end

# ------------------------------------------------------------------------------------------
# CZ10: a trace whose carbon charge differs per bin (OQ-60 (f))
# ------------------------------------------------------------------------------------------

const _CZ_ZS = [4.2, 4.6, 5.1, 5.6]

@testset "CZ10: a trace whose carbon charge differs per bin" begin
    t_ns = [0.0, 1.0, 2.0, 3.0]
    template = _cz_single(5.0, 0.7)
    snaps = [_cz_single(z, 0.7) for z in _CZ_ZS]
    tv = PlasmaTrace(template; t_ns=t_ns, leaves=["ions.C.charge" => _CZ_ZS])
    ts = PlasmaTrace(snaps; t_ns=t_ns, leaves=["ions.C.charge"])
    for trace in (tv, ts), j in 1:4
        q = plasma(trace, j)
        @test q.ions["C"].charge === _CZ_ZS[j]
        @test q.ions["C"].n === snaps[j].ions["C"].n
        @test q.ions["D"].n === template.ions["D"].n
    end
    @test_throws "\"ions.C.charge\" varies at bin 2 but is not in the leaves list" PlasmaTrace(snaps; t_ns=t_ns)
    _p() = plasma(tv, 2)
    _p()
    @test (@allocated _p()) == 0
    mktempdir() do dir
        path = joinpath(dir, "trace.jld2")
        write_trace(path, tv; source="CZ10")
        rebuilt = read_trace(path; as=:trace)
        @test string.(rebuilt.paths) == ["ions.C.charge"]
        @test all(rebuilt.values .=== tv.values)
        @test all(plasma(rebuilt, j).ions["C"].charge === _CZ_ZS[j] for j in 1:4)
    end
end

# ------------------------------------------------------------------------------------------
# CZ11: _with_leaf serves a charge; GradientSmearing refuses one (spec D14)
# ------------------------------------------------------------------------------------------

@testset "CZ11: _with_leaf on a charge, and GradientSmearing's refusal" begin
    p = _cz_single(5.0, 0.7)
    ref = _resolve_leaf(p, LeafPath("ions.C.charge"))
    q = _with_leaf(p, ref, 4.5)
    @test q.ions["C"].charge === 4.5
    @test q.ions["C"].n === p.ions["C"].fraction * p.n_e / 4.5
    @test q.ions["C"].T === p.ions["C"].T
    @test q.ions["D"].n === p.ions["D"].n
    @test_throws "GradientSmearing: a charge cannot be gradient-smeared" GradientSmearing(p; leaf="ions.C.charge", width=0.1)
end

# ------------------------------------------------------------------------------------------
# CZ12: a variable bound to ions.C.charge, and the floor on its bounds (OQ-60 (b), item 4; spec D15)
# ------------------------------------------------------------------------------------------

@testset "CZ12: the charge binding and its floor" begin
    expt = Experiment(; trace=PlasmaTrace(_cz_single(5.0, 0.7); t_ns=[0.0, 1.0, 2.0, 3.0]),
        probe=ProbeBeam(lambda_probe_nm=263.25), collection=_CZ_COLL, lambda_m=_CZ_IAW,
        effects=(SpectralIRF(fwhm_nm=0.03), Sensitivity(scale=1.0e-6)))
    zc = FittedVariable("ZC"; bounds=(3.0, 7.0), init=5.0, time_varying=true)
    pspec = ParameterSpec(expt; variables=[zc], bindings=["ZC" => "ions.C.charge"])
    b = only(pspec.bindings)
    @test b.kind === :leaf && string(b.path) == "ions.C.charge" && b.factor == 1.0 && b.unit_key == ""
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects)
    out = unflatten(working, pspec, flatten(pspec, ["ZC" => _CZ_ZS]))
    for j in 1:4
        q = plasma(out.trace, j)
        @test isapprox(q.ions["C"].charge, _CZ_ZS[j]; rtol=1e-13)
        @test q.ions["C"].n === q.ions["C"].fraction * q.n_e / q.ions["C"].charge
    end
    msg = "is bound to \"ions.C.charge\" with lower bound 0.9, below 1: an ion's charge number is a finite real ≥ 1"
    @test_throws "\"ZC\" $msg" ParameterSpec(expt;
        variables=[FittedVariable("ZC"; bounds=(0.9, 7.0), init=5.0, time_varying=true)], bindings=["ZC" => "ions.C.charge"])
    @test_throws "\"ZF\" $msg" ParameterSpec(expt;
        variables=[FittedVariable("ZF"; bounds=(0.9, 7.0), init=5.0, vary=false)], bindings=["ZF" => "ions.C.charge"])
    # a dependent bound to a charge is not range-checked (no dependent is): ZC - 2.5 reaches 0.5 at ZC's lower bound
    dep = ParameterSpec(expt; variables=[zc], dependents=["Zd" => "ZC - 2.5"], bindings=["Zd" => "ions.C.charge"])
    @test only(dep.bindings).name == "Zd"
end

# ------------------------------------------------------------------------------------------
# CZ13: charge = "ZC" in a deck; [fit.start] and [fit.hold] on it (OQ-60 (g))
# ------------------------------------------------------------------------------------------

@testset "CZ13: charge = \"ZC\" in a deck" begin
    mktempdir() do dir
        _cz_streak(dir)
        zc = "ZC = { bounds = [3.0, 7.0], init = 5.0, time_varying = true }"
        deck = read_deck(_cz_write(dir, "zc.toml", _cz_deck_text(dir; charge="\"ZC\"", variables=zc)))
        problem, optimizer, _ = OTIS.build_problem(deck)
        @test any(b -> b.name == "ZC" && b.kind === :leaf && string(b.path) == "ions.C.charge", problem.pspec.bindings)
        eff = effective_deck(deck, problem, optimizer)
        @test TOML.parse(eff)["fit"]["variables"]["ZC"]["bindings"] == ["ions.C.charge"]
        @test read_deck(_cz_write(dir, "zc_eff.toml", eff)) == deck
        sp = write_start(joinpath(dir, "start.jld2"); axis_ns=[4.0, 5.0], values=Dict("ZC" => [4.4, 4.8]),
            source="CZ13")
        table(key) = "[fit.$key]\nfile = \"$(replace(sp, "\\" => "/"))\"\nvariables = [\"ZC\"]\n"
        ps, _, start = OTIS.build_problem(read_deck(_cz_write(dir, "zs.toml",
            _cz_deck_text(dir; charge="\"ZC\"", variables=zc, tables=table("start")))))
        z_start = plasma(unflatten(ps.experiment, ps.pspec, start).trace, 1).ions["C"].charge
        @test isapprox(z_start, 4.6; rtol=1e-13)
        ph, _, start_h = OTIS.build_problem(read_deck(_cz_write(dir, "zh.toml",
            _cz_deck_text(dir; charge="\"ZC\"", variables=zc, tables=table("hold")))))
        held = only(v for v in ph.pspec.variables if v.name == "ZC").held
        @test length(held) == 1 && isapprox(held[1], 4.6; rtol=1e-14)
        @test plasma(unflatten(ph.experiment, ph.pspec, start_h).trace, 1).ions["C"].charge === held[1]
        low = "ZC = { bounds = [0.9, 7.0], init = 5.0, time_varying = true }"
        @test_throws "\"ZC\" is bound to \"ions.C.charge\" with lower bound 0.9, below 1" OTIS.build_problem(
            read_deck(_cz_write(dir, "zlow.toml", _cz_deck_text(dir; charge="\"ZC\"", variables=low))))
    end
end
