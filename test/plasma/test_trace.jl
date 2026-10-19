using Test
using TRUST
using TRUST: LeafRef, _resolve_leaf, _leaf_row, _leaf_value, _with_leaf
import ForwardDiff

# The D22 reference plasma (test/scattering/test_spectrum_supergaussian.jl): n_e_cm3 = 5e19, one
# electron species at 500 eV, deuterium (2.014 amu, charge 1) at 250 eV, named "D" here so the
# leaf paths below read naturally; a second, carbon species (12.011 amu, charge 6) is added
# wherever two ions are wanted. Built locally per spec Section 6 (no cross-@safetestset import).

_tr_plasma(; f_electron=Maxwellian()) = Plasma(
    n_e_cm3=5e19,
    electrons=(electrons(T_eV=500, f=f_electron),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))

_tr_plasma2ion(; fD=0.6, fC=0.4) = Plasma(
    n_e_cm3=5e19,
    electrons=(electrons(T_eV=500),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250, fraction=fD),
        ion(name="C", mass_amu=12.011, charge=6, T_eV=250, fraction=fC)))

@testset "TR1: round trip from vectors" begin
    template = _tr_plasma()
    nt = 16
    n_e_vec = collect(range(5e19, 6e19; length=nt))
    Te_vec = collect(range(400.0, 600.0; length=nt))
    u_vec = collect(range(0.0, 50.0; length=nt))
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=nt)),
        leaves=("n_e_cm3" => n_e_vec, "electrons.e.T_eV" => Te_vec, "ions.D.u_kms" => u_vec))
    @test nbins(trace) == nt
    @test string.(trace.paths) == ["n_e", "electrons.e.T", "ions.D.u"]
    @test all(trace.values[1, :] .=== n_e_vec .* 1e6)
    @test all(trace.values[2, :] .=== Te_vec .* EV)
    @test all(trace.values[3, :] .=== u_vec .* 1e3)
end

@testset "TR2: round trip from functions of the axis" begin
    template = _tr_plasma()
    t_ns = collect(range(0.0, 3.0; length=16))
    g_ne(t) = 5e19 + 1e18 * t
    g_Te(t) = 500.0 + 10 * t
    g_u(t) = 2.0 * t
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => g_ne, "electrons.e.T_eV" => g_Te, "ions.D.u_kms" => g_u))
    @test all(trace.values[1, j] === g_ne(t_ns[j]) * 1e6 for j in eachindex(t_ns))
    @test all(trace.values[2, j] === g_Te(t_ns[j]) * EV for j in eachindex(t_ns))
    @test all(trace.values[3, j] === g_u(t_ns[j]) * 1e3 for j in eachindex(t_ns))
end

@testset "TR3: the callable's argument is in the axis keyword's unit" begin
    template = _tr_plasma()
    t_ns = collect(range(0.0, 3.0; length=16))
    t_s = t_ns .* 1e-9
    g_ns(t) = 500.0 + 0.1 * t                  # t expected in ns
    g_s(t) = 500.0 + 0.1 * (t * 1e9)           # same physics, t expected in s
    trace_ns = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T_eV" => g_ns,))
    trace_s = PlasmaTrace(template; t_s=t_s, leaves=("electrons.e.T_eV" => g_s,))
    @test isapprox(trace_ns.values[1, :], trace_s.values[1, :]; rtol=1e-15)
end

@testset "TR4: the unit keywords agree" begin
    template = _tr_plasma()
    t_ns = collect(range(0.0, 3.0; length=8))
    tr1 = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T_eV" => fill(500.0, 8),))
    tr2 = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T_keV" => fill(0.5, 8),))
    tr3 = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T" => fill(500.0 * EV, 8),))
    @test isapprox(tr1.values[1, :], tr2.values[1, :]; rtol=1e-15)
    @test isapprox(tr1.values[1, :], tr3.values[1, :]; rtol=1e-15)
end

@testset "TR5: the snapshot equals a hand-built Plasma" begin
    template = _tr_plasma()
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))
    n_e_vec = collect(range(5e19, 7e19; length=nt))
    Te_vec = collect(range(400.0, 600.0; length=nt))
    u_vec = collect(range(0.0, 50.0; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_vec, "electrons.e.T_eV" => Te_vec, "ions.D.u_kms" => u_vec))
    for j in 1:nt
        p = plasma(trace, j)
        p_ref = Plasma(n_e_cm3=n_e_vec[j],
            electrons=(electrons(T_eV=Te_vec[j]),),
            ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250, u_kms=u_vec[j]),))
        @test isapprox(p.n_e, p_ref.n_e; rtol=1e-12)
        @test isapprox(p.electrons[1].T, p_ref.electrons[1].T; rtol=1e-12)
        @test isapprox(p.electrons[1].n, p_ref.electrons[1].n; rtol=1e-12)
        @test isapprox(p.ions[1].u, p_ref.ions[1].u; rtol=1e-12)
        @test isapprox(p.ions[1].n, p_ref.ions[1].n; rtol=1e-12)
    end
end

@testset "TR6: constant leaves stay constant" begin
    template = _tr_plasma()
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))
    Te_vec = collect(range(400.0, 600.0; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T_eV" => Te_vec,))
    for j in 1:nt
        p = plasma(trace, j)
        @test p.n_e === template.n_e
        @test p.electrons[1].u === template.electrons[1].u
        @test p.electrons[1].fraction === template.electrons[1].fraction
        @test p.ions[1].T === template.ions[1].T
        @test p.ions[1].u === template.ions[1].u
        @test p.ions[1].fraction === template.ions[1].fraction
        @test p.ions[1].n === template.ions[1].n
    end
end

@testset "TR7: densities are re-resolved" begin
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))

    template = _tr_plasma()
    n_e_vec = collect(range(5e19, 8e19; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("n_e_cm3" => n_e_vec,))
    for j in 1:nt
        p = plasma(trace, j)
        @test isapprox(p.electrons[1].n, p.electrons[1].fraction * p.n_e; rtol=1e-14)
        @test isapprox(p.ions[1].n, p.ions[1].fraction * p.n_e / p.ions[1].charge; rtol=1e-14)
        @test isapprox(neutrality_residual(p), 0.0; atol=1e-14)
    end

    template2 = _tr_plasma2ion(fD=0.6, fC=0.4)
    fD_vec = collect(range(0.6, 0.3; length=nt))
    fC_vec = 1.0 .- fD_vec
    trace2 = PlasmaTrace(template2; t_ns=t_ns,
        leaves=("ions.D.fraction" => fD_vec, "ions.C.fraction" => fC_vec))
    for j in 1:nt
        p = plasma(trace2, j)
        nD, nC = p.ions["D"].n, p.ions["C"].n
        @test isapprox(nD, fD_vec[j] * p.n_e / 1; rtol=1e-14)
        @test isapprox(nC, fC_vec[j] * p.n_e / 6; rtol=1e-14)
        @test isapprox(neutrality_residual(p), 0.0; atol=1e-14)
    end
end

@testset "TR8: a shape parameter varies" begin
    template = _tr_plasma(f_electron=SuperGaussian(3.0))
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))
    m_vec = collect(range(2.0, 5.0; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.f.m" => m_vec,))
    for j in 1:nt
        p = plasma(trace, j)
        f = p.electrons[1].f
        @test f.m === m_vec[j]
        ref = SuperGaussian(m_vec[j])
        @test f.x0 === ref.x0
        @test f.C === ref.C
        @test f.slice === ref.slice
    end
end

@testset "TR9: the shape survives untouched when nothing in it varies" begin
    template = _tr_plasma(f_electron=SuperGaussian(3.0))
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))
    Te_vec = collect(range(400.0, 600.0; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=("electrons.e.T_eV" => Te_vec,))
    for j in 1:nt
        p = plasma(trace, j)
        @test p.electrons[1].f === template.electrons[1].f
    end
end

@testset "TR10: the snapshot input form" begin
    nt = 8
    Te_vals = collect(range(400.0, 600.0; length=nt))
    snapshots = [Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=Te),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),)) for Te in Te_vals]
    t_ns = collect(range(0.0, 3.0; length=nt))
    trace = PlasmaTrace(snapshots; t_ns=t_ns, leaves=("electrons.e.T",))
    @test trace.template === snapshots[1]
    row = _leaf_row(trace, LeafPath("electrons.e.T"))
    @test row != 0
    @test all(trace.values[row, j] === Te_vals[j] * EV for j in 1:nt)
end

@testset "TR11: the snapshot form refuses a silently varying leaf" begin
    nt = 8
    Te_vals = collect(range(400.0, 600.0; length=nt))
    n_e_vals = collect(range(5e19, 6e19; length=nt))
    snapshots = [Plasma(n_e_cm3=n_e_vals[i], electrons=(electrons(T_eV=Te_vals[i]),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),)) for i in 1:nt]
    t_ns = collect(range(0.0, 3.0; length=nt))
    @test_throws "varies at bin" PlasmaTrace(snapshots; t_ns=t_ns, leaves=("electrons.e.T",))
end

@testset "TR12: the constant trace" begin
    template = _tr_plasma()
    t_ns = collect(range(0.0, 3.0; length=4))
    trace = PlasmaTrace(template; t_ns=t_ns)
    @test size(trace.values) == (0, 4)
    for j in 1:4
        p = plasma(trace, j)
        @test p.n_e === template.n_e
        @test p.electrons[1].T === template.electrons[1].T
        @test p.electrons[1].u === template.electrons[1].u
        @test p.electrons[1].fraction === template.electrons[1].fraction
        @test p.electrons[1].n === template.electrons[1].n
        @test p.electrons[1].f === template.electrons[1].f
        @test p.ions[1].T === template.ions[1].T
        @test p.ions[1].n === template.ions[1].n
    end
end

@testset "TR13: every trace error of D28" begin
    template = _tr_plasma()
    @test_throws "is not a valid leaf path" LeafPath("")
    @test_throws "is not a valid leaf path" LeafPath("a..b")
    @test_throws "must begin with" _resolve_leaf(template, LeafPath("bogus.e.T"))
    @test_throws "has no species named" _resolve_leaf(template, LeafPath("electrons.x.T"))
    # amended 2026-10-10 (Phase 4.J spec D4, D13): an ion's charge is a leaf, the shared message names it,
    # and mass is refused on its own
    @test_throws "is not a leaf; expected T, u, fraction, charge (ions) or f.<parameter>" _resolve_leaf(template, LeafPath("electrons.e.bogus"))
    @test_throws "mass is not a fittable leaf" _resolve_leaf(template, LeafPath("electrons.e.mass"))
    @test_throws "is not a parameter of" _resolve_leaf(template, LeafPath("electrons.e.f.bogus"))

    t_ns8 = collect(range(0.0, 3.0; length=8))
    @test_throws "axis must be strictly increasing" PlasmaTrace(template; t_ns=[1.0, 0.5, 2.0])
    @test_throws "axis must have at least one bin" PlasmaTrace(template; t_ns=Float64[])
    @test_throws "has length" PlasmaTrace(template; t_ns=t_ns8,
        leaves=("electrons.e.T_eV" => [1.0, 2.0],))
    @test_throws "is given twice" PlasmaTrace(template; t_ns=t_ns8,
        leaves=("electrons.e.T_eV" => fill(500.0, 8), "electrons.e.T" => fill(500.0 * EV, 8)))
    @test_throws "does not convert to the template's element type" PlasmaTrace(template; t_ns=t_ns8,
        leaves=("electrons.e.T_eV" => (t -> "not a number"),))
    @test_throws "does not convert to the template's element type" PlasmaTrace(template; t_ns=t_ns8,
        leaves=("electrons.e.T_eV" => (t -> 500.0 + 1.0im),))

    bad_snapshots = [_tr_plasma(),
        Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
            ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),)),
        _tr_plasma(), _tr_plasma()]
    @test_throws "differs from the template's species" PlasmaTrace(bad_snapshots;
        t_ns=collect(range(0.0, 3.0; length=4)))

    trace8 = PlasmaTrace(template; t_ns=t_ns8)
    @test_throws "is out of range" plasma(trace8, 0)
    @test_throws "is out of range" plasma(trace8, 9)
end

@testset "TR14: allocations" begin
    nt = 1000
    t_ns = collect(range(0.0, 10.0; length=nt))

    template_a = _tr_plasma2ion()
    n_e_vec = collect(range(5e19, 6e19; length=nt))
    fD_vec = collect(range(0.6, 0.3; length=nt))
    fC_vec = 1.0 .- fD_vec
    trace_a = PlasmaTrace(template_a; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_vec, "ions.D.fraction" => fD_vec, "ions.C.fraction" => fC_vec))
    _p_a() = plasma(trace_a, 1)
    _p_a()
    @test (@allocated _p_a()) == 0

    template_b = _tr_plasma(f_electron=SuperGaussian(3.0))
    m_vec = collect(range(2.0, 5.0; length=nt))
    trace_b = PlasmaTrace(template_b; t_ns=t_ns, leaves=("electrons.e.f.m" => m_vec,))
    _p_b() = plasma(trace_b, 1)
    _p_b()
    @test (@allocated _p_b()) == 0

    trace_c = PlasmaTrace(template_b; t_ns=t_ns, leaves=("electrons.e.T_eV" => fill(500.0, nt),))
    _p_c() = plasma(trace_c, 1)
    _p_c()
    @test (@allocated _p_c()) == 0
end

@testset "TR15: the dual path" begin
    nt = 8
    t_ns = collect(range(0.0, 3.0; length=nt))
    j = 4
    h = 1e-6

    template1 = _tr_plasma()
    Te_vec = collect(range(400.0, 600.0; length=nt))
    trace_of1(x) = PlasmaTrace(template1; t_ns=t_ns, leaves=("electrons.e.T_eV" => Te_vec .* x,))
    f1(x) = plasma(trace_of1(x), j).electrons[1].T
    ad1 = ForwardDiff.derivative(f1, 1.0)
    fd1 = (f1(1.0 + h) - f1(1.0 - h)) / (2h)
    @test isapprox(ad1, fd1; rtol=1e-6, atol=1e-6 * abs(fd1))

    template2 = _tr_plasma2ion(fD=0.6, fC=0.4)
    n_e_vec = collect(range(5e19, 6e19; length=nt))
    fD_vec = collect(range(0.6, 0.3; length=nt))
    fC_vec = 1.0 .- fD_vec
    trace_of2(x) = PlasmaTrace(template2; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_vec .* x, "ions.D.fraction" => fD_vec .* x, "ions.C.fraction" => fC_vec))
    g2(x) = (p = plasma(trace_of2(x), j); p.n_e + p.ions["D"].n)
    ad2 = ForwardDiff.derivative(g2, 1.0)
    fd2 = (g2(1.0 + h) - g2(1.0 - h)) / (2h)
    @test isapprox(ad2, fd2; rtol=1e-6, atol=1e-6 * abs(fd2))

    template3 = _tr_plasma(f_electron=SuperGaussian(3.0))
    m_vec = collect(range(2.0, 5.0; length=nt))
    trace_of3(x) = PlasmaTrace(template3; t_ns=t_ns, leaves=("electrons.e.f.m" => m_vec .* x,))
    g3(x) = plasma(trace_of3(x), j).electrons[1].f(0.5)
    ad3 = ForwardDiff.derivative(g3, 1.0)
    fd3 = (g3(1.0 + h) - g3(1.0 - h)) / (2h)
    floor3 = 1e-6 * abs(fd3)
    @test isapprox(ad3, fd3; rtol=1e-6, atol=max(floor3, 1e-12))
end

@testset "TR16: LeafPath and show" begin
    p = LeafPath("electrons.e.f.m")
    @test string(p) == "electrons.e.f.m"
    p2 = LeafPath("electrons.e.f.m")
    @test p == p2
    @test hash(p) == hash(p2)

    template = _tr_plasma()
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=5)),
        leaves=("electrons.e.T_eV" => fill(500.0, 5),))
    s = sprint(show, trace)
    @test occursin("1 leaves", s)
    @test occursin("5 bins", s)
    @test occursin(":t", s)
end
