# Step G.2.1: gridded traces (Phase G.2 spec D3 to D7, D10; OQ-69 items 1 to 3). Gate "gridded trace":
# GT1 to GT8. The whole-distribution leaf against the pairs it stands for (bitwise: the same trace); the
# snapshot route with a gridded species; project_to_grid on a trace, one source against Phase G.1's
# projection of each bin's species (bitwise) and a merged source against a test-local mixture and the
# mixture's moments written out (conventions.md 8.2); size_grid against its rule written out. Deliberately
# tight rows are identities (===, ==, exact messages); the loose ones say why.

using Test
using TRUST
using TRUST: _derives_u_T, ELECTRON_MASS, ATOMIC_MASS_UNIT, EV, _lab_cauchy
using Random: Xoshiro, randn

const _GT_T = [0.0, 1.0, 2.0, 3.0]                  # ns, four bins

# The electron grid every row uses unless it says otherwise: T_ref = 700 eV, the defaults (41 nodes).
_gt_egrid() = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 700.0)
_gt_deuterons(; kw...) = ion(; name = "D", mass_amu = 2.014, charge = 1, kw...)
_gt_plasma(e, i) = Plasma(n_e_cm3 = 5e19, electrons = (e,), ions = (i,))

# Per-bin log-weights for the 41-node grid, seeded (spec D10).
_gt_g(seed) = 0.3 .* randn(Xoshiro(seed), 41, length(_GT_T))

# ------------------------------------------------------------------------------------------
# GT1-GT2: a whole distribution as one leaf (OQ-69 item 1; spec D3)
# ------------------------------------------------------------------------------------------

@testset "GT1: the whole-distribution leaf is the same trace as its pairs, bitwise" begin
    tmpl = _gt_plasma(electrons(f = _gt_egrid()), _gt_deuterons(T_eV = 250.0))
    G = _gt_g(1)
    ne = [5.0e19, 5.5e19, 6.0e19, 6.5e19]
    TD = [250.0, 260.0, 270.0, 280.0]
    pnames = keys(parameters(_gt_egrid()))
    pairs_form = ["n_e_cm3" => ne, ("electrons.e.f.$(pnames[i])" => G[i, :] for i in 1:41)..., "ions.D.T_eV" => TD]
    a = PlasmaTrace(tmpl; t_ns = _GT_T, leaves = pairs_form)
    b = PlasmaTrace(tmpl; t_ns = _GT_T, leaves = ["n_e_cm3" => ne, "electrons.e.f" => G, "ions.D.T_eV" => TD])
    @test b.paths == a.paths
    @test all(b.values .=== a.values)
    @test typeof(b) === typeof(a)
    @test all(plasma(b, j).electrons[1].T === plasma(a, j).electrons[1].T for j in 1:4)
    # a function of the axis value returning one value per parameter, against the pairs' functions
    gfun(t) = [0.1 * i * sin(t + i) for i in 1:41]
    c = PlasmaTrace(tmpl; t_ns = _GT_T, leaves = ["electrons.e.f" => gfun])
    d = PlasmaTrace(tmpl; t_ns = _GT_T, leaves = ["electrons.e.f.$(pnames[i])" => (t -> gfun(t)[i]) for i in 1:41])
    @test c.paths == d.paths
    @test all(c.values .=== d.values)
    # any distribution with a parameter: a one-parameter shape and a four-parameter one
    tsg = _gt_plasma(electrons(T_eV = 500.0, f = SuperGaussian(3.0)), _gt_deuterons(T_eV = 250.0))
    m = [2.0 3.0 4.0 5.0]
    @test all(PlasmaTrace(tsg; t_ns = _GT_T, leaves = ["electrons.e.f" => m]).values .===
              PlasmaTrace(tsg; t_ns = _GT_T, leaves = ["electrons.e.f.m" => vec(m)]).values)
    the = _gt_plasma(electrons(T_eV = 500.0), _gt_deuterons(T_eV = 250.0, f = HermiteExpansion(c3 = 0.01)))
    H = [0.01 0.02 0.03 0.04; 0.0 0.01 0.0 0.01; 0.0 0.0 0.0 0.0; 0.0 0.0 0.001 0.0]
    e = PlasmaTrace(the; t_ns = _GT_T, leaves = ["ions.D.f" => H])
    f = PlasmaTrace(the; t_ns = _GT_T, leaves = ["ions.D.f.$k" => H[i, :] for (i, k) in enumerate((:c3, :c4, :c5, :c6))])
    @test e.paths == f.paths
    @test all(e.values .=== f.values)
    @test string.(e.paths) == ["ions.D.f.c3", "ions.D.f.c4", "ions.D.f.c5", "ions.D.f.c6"]
end

@testset "GT2: the whole-distribution leaf's refusals, by name" begin
    tmpl = _gt_plasma(electrons(f = _gt_egrid()), _gt_deuterons(T_eV = 250.0))
    G = _gt_g(2)
    pt(leaves) = PlasmaTrace(tmpl; t_ns = _GT_T, leaves = leaves)
    @test_throws "\"ions.D.f\": Maxwellian has no parameter, so its distribution cannot be a leaf" pt(["ions.D.f" => zeros(1, 4)])
    @test_throws "\"electrons.e.f\" has 40 rows but GaussianGrid has 41 parameters, one row each" pt(["electrons.e.f" => G[1:40, :]])
    @test_throws "\"electrons.e.f\" has 3 columns but the axis has 4 bins" pt(["electrons.e.f" => G[:, 1:3]])
    @test_throws "\"electrons.e.f\" takes a matrix with one row per parameter and one column per bin, or a function of the axis returning one value per parameter; got a Vector{Float64}" pt(["electrons.e.f" => zeros(41)])
    @test_throws "\"electrons.e.f\" returned 40 values at bin 1 but GaussianGrid has 41 parameters" pt(["electrons.e.f" => (t -> zeros(40))])
    @test_throws "\"electrons.e.f\" is given with \"electrons.e.f.g7\", one of its own parameters; give the whole distribution or its parameters, not both" pt(["electrons.e.f" => G, "electrons.e.f.g7" => G[7, :]])
    @test_throws "\"electrons.e.f\" is given with \"electrons.e.f.g7\", one of its own parameters; give the whole distribution or its parameters, not both" pt(["electrons.e.f.g7" => G[7, :], "electrons.e.f" => G])
    @test_throws "\"electrons.e.f\" is given twice" pt(["electrons.e.f" => G, "electrons.e.f" => G])
    @test_throws "\"electrons.x.f\" has no species named \"x\"" pt(["electrons.x.f" => G])
end

# ------------------------------------------------------------------------------------------
# GT3: the snapshot route with a gridded species (OQ-69 item 1; G.1's A7; spec D4)
# ------------------------------------------------------------------------------------------

@testset "GT3: snapshots of a gridded species: derived T and u skipped, the grid compared" begin
    G = _gt_g(3)
    grid = _gt_egrid()
    names = keys(parameters(grid))
    snap(j) = _gt_plasma(electrons(f = reconstruct(grid, NamedTuple{names}(Tuple(G[:, j])))), _gt_deuterons(T_eV = 250.0))
    snaps = [snap(j) for j in 1:4]
    gpaths = ["electrons.e.f.$n" for n in names]
    tr = PlasmaTrace(snaps; t_ns = _GT_T, leaves = gpaths)
    @test string.(tr.paths) == gpaths                  # no T, no u among the rows
    for j in 1:4
        p, s = plasma(tr, j), snaps[j]
        @test p.electrons[1].f.g === s.electrons[1].f.g
        @test p.electrons[1].T === s.electrons[1].T
        @test p.electrons[1].u === s.electrons[1].u
    end
    # without the g leaves the message names a log-weight, never the derived T
    @test_throws "\"electrons.e.f.g1\" varies at bin 2 but is not in the leaves list" PlasmaTrace(snaps; t_ns = _GT_T)
    # a snapshot on another grid (the same node count, another centre) is refused, naming the bin
    other = _gt_plasma(electrons(f = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 700.0, u0_kms = 10.0)), _gt_deuterons(T_eV = 250.0))
    @test_throws "snapshot at bin 3: species \"e\" carries its GaussianGrid on another grid than the template's; a trace's grid is the same in every bin" PlasmaTrace([snaps[1], snaps[2], other, snaps[4]]; t_ns = _GT_T, leaves = gpaths)
    ion_other = _gt_plasma(electrons(T_eV = 500.0), ion(name = "D", mass_amu = 2.014, charge = 1, grid = (T_ref_eV = 250.0, width = 1.1)))
    ion_ref = _gt_plasma(electrons(T_eV = 500.0), ion(name = "D", mass_amu = 2.014, charge = 1, grid = (T_ref_eV = 250.0,)))
    @test_throws "snapshot at bin 2: species \"D\" carries its GaussianGrid on another grid than the template's" PlasmaTrace([ion_ref, ion_other]; t_ns = [0.0, 1.0])
    # identical grids pass the check (a shape species is never compared)
    @test nbins(PlasmaTrace([ion_ref, ion_ref]; t_ns = [0.0, 1.0])) == 2
end

# ------------------------------------------------------------------------------------------
# GT4-GT6: a parametric trace projected onto grids (OQ-69 item 2; spec D5)
# ------------------------------------------------------------------------------------------

# A parametric trace: super-Gaussian electrons cooling as m rises, deuterons heating, n_e and the
# deuteron flow varying; the electrons projected onto a 41-node grid at T_ref = 900 eV.
function _gt_parametric()
    tmpl = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 600.0, f = SuperGaussian(2.5)),),
                  ions = (_gt_deuterons(T_eV = 250.0, u_kms = 20.0),))
    return PlasmaTrace(tmpl; t_ns = _GT_T,
        leaves = ["n_e_cm3" => [5.0e19, 5.2e19, 5.4e19, 5.6e19], "electrons.e.T_eV" => [900.0, 800.0, 700.0, 600.0],
                  "electrons.e.f.m" => [2.0, 2.5, 3.0, 3.5], "ions.D.u_kms" => [20.0, 30.0, 40.0, 50.0]])
end
_gt_grid900() = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 900.0)

@testset "GT4: one source is Phase G.1's projection of each bin's species, bitwise; the rest untouched" begin
    tr = _gt_parametric()
    grid = _gt_grid900()
    r = project_to_grid(tr, "electrons.e" => grid)
    nt = r.trace
    @test string.(nt.paths) == ["n_e", "ions.D.u", ("electrons.e.f.g$i" for i in 1:41)...]
    @test all(nt.values[1, :] .=== tr.values[1, :])               # n_e, untouched
    @test all(nt.values[2, :] .=== tr.values[4, :])               # the deuterons' flow, untouched
    @test nt.template.electrons[1].f === grid                     # the template carries the grid itself
    @test nt.template.electrons[1].fraction === 1.0
    @test nt.template.ions[1] === tr.template.ions[1]             # every other species bitwise
    @test nt.template.n_e === tr.template.n_e
    @test nt.axis == tr.axis && nt.axis_name === tr.axis_name
    for j in 1:4
        g1 = project_to_grid(plasma(tr, j).electrons[1], grid)
        @test all(nt.values[3:end, j] .=== collect(g1.f.g))
        @test r.error_f[1][j] === g1.error_f
        @test r.error_cauchy[1][j] === g1.error_cauchy
        @test r.iterations[1][j] === g1.iterations
        @test r.converged[1][j] === g1.converged
        @test plasma(nt, j).electrons[1].f.g === g1.f.g
        @test plasma(nt, j).ions[1].u === plasma(tr, j).ions[1].u
    end
    @test length(r.error_f) == 1 && length(r.error_f[1]) == 4
    @info "GT4: one source, the reported errors per bin" error_cauchy = r.error_cauchy[1] iterations = r.iterations[1]
    # a fraction leaf stays a leaf, in its place, the values bitwise
    tf = PlasmaTrace(Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 600.0),),
                            ions = (_gt_deuterons(T_eV = 250.0, fraction = 0.9), ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300.0, fraction = 0.1))),
                     t_ns = _GT_T, leaves = ["ions.D.fraction" => [0.9, 0.8, 0.7, 0.6], "ions.D.T_eV" => [250.0, 260.0, 270.0, 280.0],
                                            "ions.C.fraction" => [0.1, 0.2, 0.3, 0.4]])
    gD = GaussianGrid(; mass_amu = 2.014, T_ref_eV = 300.0)
    rf = project_to_grid(tf, "ions.D" => gD)
    @test string.(rf.trace.paths) == ["ions.D.fraction", "ions.C.fraction", ("ions.D.f.g$i" for i in 1:41)...]
    @test all(rf.trace.values[1:2, :] .=== tf.values[[1, 3], :])
    # a source whose fraction is not 1 is still Phase G.1's projection of it, bitwise (never f · f_lab / f)
    for j in 1:4
        @test plasma(rf.trace, j).ions[1].f.g === project_to_grid(plasma(tf, j).ions[1], gD).f.g
    end
end

# The mixture written out, a test-local lab-frame distribution (spec D10's oracle for a merged source):
# Σ fᵢ f_lab,ᵢ / Σ fᵢ and the same sum of the species' lab-frame Cauchy integrals, from the species alone.
struct _GTMixture{S} <: AbstractLabDistribution
    species::S
    weights::Vector{Float64}
end
(m::_GTMixture)(v) = sum(m.weights[i] * f_lab(m.species[i], v) for i in eachindex(m.weights)) / sum(m.weights)
TRUST.cauchy_dfdv(m::_GTMixture, v) = sum(m.weights[i] * _lab_cauchy(m.species[i].f, m.species[i], v) for i in eachindex(m.weights)) / sum(m.weights)

# A two-population electron trace and a two-population ion trace with flows.
function _gt_bimodal()
    tmpl = Plasma(n_e_cm3 = 5e19,
        electrons = (electrons(name = "cold", T_eV = 300.0, fraction = 0.9), electrons(name = "hot", T_eV = 1200.0, fraction = 0.1)),
        ions = (ion(name = "H", mass_amu = 1.008, charge = 1, T_eV = 200.0, fraction = 0.7),
                ion(name = "Hb", mass_amu = 1.008, charge = 1, T_eV = 100.0, u_kms = 200.0, fraction = 0.3)))
    return PlasmaTrace(tmpl; t_ns = _GT_T,
        leaves = ["electrons.cold.fraction" => [0.9, 0.8, 0.7, 0.75], "electrons.hot.fraction" => [0.1, 0.2, 0.3, 0.25],
                  "electrons.hot.T_eV" => [1200.0, 1100.0, 1000.0, 900.0], "ions.Hb.u_kms" => [200.0, 150.0, 100.0, 50.0]])
end

@testset "GT5: a merged source is the projection of the fraction-weighted mixture" begin
    tr = _gt_bimodal()
    ge = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 1200.0, nodes = 61, spacing = 0.125)
    gi = GaussianGrid(; mass_amu = 1.008, T_ref_eV = 200.0, u0_kms = 100.0, nodes = 61, spacing = 0.2)
    r = project_to_grid(tr, ("electrons.cold", "electrons.hot") => ge, ["ions.H", "ions.Hb"] => gi; names = ("e", "H"))
    nt = r.trace
    @test nt.template.electrons.names == ("e",) && nt.template.ions.names == ("H",)
    @test string.(nt.paths) == ["electrons.e.fraction", ("electrons.e.f.g$i" for i in 1:61)..., ("ions.H.f.g$i" for i in 1:61)...]
    @test nt.values[1, :] == tr.values[1, :] .+ tr.values[2, :]     # the fraction, Σ in the order given
    @test nt.template.ions[1].fraction === 0.7 + 0.3
    for j in 1:4
        p = plasma(tr, j)
        mixe = _GTMixture((p.electrons[1], p.electrons[2]), [p.electrons[1].fraction, p.electrons[2].fraction])
        mixi = _GTMixture((p.ions[1], p.ions[2]), [p.ions[1].fraction, p.ions[2].fraction])
        oe = project_to_grid(mixe, ge)
        oi = project_to_grid(mixi, gi)
        @test plasma(nt, j).electrons[1].f.g === oe.f.g       # the same fit on the same source values: bitwise
        @test plasma(nt, j).ions[1].f.g === oi.f.g
        # the mixture's moments written out: u = Σ wᵢ uᵢ, T = Σ wᵢ (Tᵢ + m (uᵢ − u)²), w the fractions normalized
        for (set, gset) in ((p.electrons, plasma(nt, j).electrons), (p.ions, plasma(nt, j).ions))
            w = [sp.fraction for sp in set] ./ sum(sp.fraction for sp in set)
            u = sum(w[i] * set[i].u for i in 1:2)
            T = sum(w[i] * (set[i].T + set[i].mass * (set[i].u - u)^2) for i in 1:2)
            @test isapprox(gset[1].T, T; rtol = 1e-6)            # the projection's accuracy (measured 2.4e-7)
            @test abs(gset[1].u - u) <= 1e-10 * sqrt(2 * T / set[1].mass)   # (measured 1.9e-12)
        end
    end
    @info "GT5: merged sources, the reported errors" error_cauchy_e = r.error_cauchy[1] error_cauchy_i = r.error_cauchy[2]
    @test all(all(c) for c in r.converged)
    # a bin where every fraction is zero takes equal weights
    tz = PlasmaTrace(tr.template; t_ns = [0.0, 1.0], leaves = ["electrons.cold.fraction" => [0.9, 0.0], "electrons.hot.fraction" => [0.1, 0.0]])
    rz = project_to_grid(tz, ("electrons.cold", "electrons.hot") => ge)
    pz = plasma(tz, 2)
    oz = project_to_grid(_GTMixture((pz.electrons[1], pz.electrons[2]), [1.0, 1.0]), ge)
    @test plasma(rz.trace, 2).electrons[1].f.g === oz.f.g
    @test rz.trace.values[1, 2] === 0.0
end

@testset "GT6: project_to_grid's refusals" begin
    tr = _gt_bimodal()
    ge = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 1200.0)
    gi = GaussianGrid(; mass_amu = 1.008, T_ref_eV = 200.0)
    @test_throws "project_to_grid: no source => grid pair given" project_to_grid(tr)
    @test_throws "project_to_grid: \"electrons\" is not a species key; expected \"electrons.<name>\" or \"ions.<name>\"" project_to_grid(tr, "electrons" => ge)
    @test_throws "project_to_grid: \"electrons.cold.f\" is not a species key" project_to_grid(tr, "electrons.cold.f" => ge)
    @test_throws "project_to_grid: \"ions.X\" names no species of the trace" project_to_grid(tr, "ions.X" => gi)
    @test_throws "project_to_grid: a source's species are all electrons or all ions; got \"electrons.cold\" and \"ions.H\"" project_to_grid(tr, ("electrons.cold", "ions.H") => ge)
    @test_throws "project_to_grid: \"electrons.cold\" is given twice" project_to_grid(tr, ("electrons.cold", "electrons.cold") => ge)
    @test_throws "project_to_grid: \"electrons.hot\" is in two sources" project_to_grid(tr, ("electrons.cold", "electrons.hot") => ge, "electrons.hot" => ge)
    @test_throws "project_to_grid: a source is a species key (\"electrons.<name>\" or \"ions.<name>\") or a tuple or vector of such keys; got Int64" project_to_grid(tr, 3 => ge)
    @test_throws "project_to_grid: a source needs at least one species key" project_to_grid(tr, String[] => ge)
    @test_throws "project_to_grid: each source's grid must be a GaussianGrid{Float64}; got " project_to_grid(tr, "electrons.cold" => SuperGaussian(3.0))
    @test_throws "project_to_grid: names has 1 entries for 2 sources" project_to_grid(tr, "electrons.cold" => ge, "ions.H" => gi; names = ("e",))
    @test_throws "duplicate species name \"hot\" in electrons" project_to_grid(tr, "electrons.cold" => ge; names = ("hot",))
    td = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500.0),),
                ions = (_gt_deuterons(T_eV = 250.0, fraction = 0.5), ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300.0, fraction = 0.3),
                        ion(name = "D2", mass_amu = 2.014, charge = 2, T_eV = 300.0, fraction = 0.2)))
    tdt = PlasmaTrace(td; t_ns = [0.0, 1.0])
    gD = GaussianGrid(; mass_amu = 2.014, T_ref_eV = 300.0)
    @test_throws "project_to_grid: \"ions.C\" and \"ions.D\" differ in mass or charge; one source is one mass and one charge" project_to_grid(tdt, ("ions.D", "ions.C") => gD)
    @test_throws "project_to_grid: \"ions.D2\" and \"ions.D\" differ in mass or charge" project_to_grid(tdt, ("ions.D", "ions.D2") => gD)
    tq = PlasmaTrace(td; t_ns = [0.0, 1.0], leaves = ["ions.C.charge" => [6.0, 5.5]])
    @test_throws "project_to_grid: \"ions.C\"'s charge is a leaf of the trace; a source's charge must be constant" project_to_grid(tq, "ions.C" => GaussianGrid(; mass_amu = 12.011, T_ref_eV = 300.0))
end

# ------------------------------------------------------------------------------------------
# GT7: a grid sized for a trace (OQ-69 item 3; spec D6)
# ------------------------------------------------------------------------------------------

@testset "GT7: size_grid follows its rule to the node" begin
    tr = _gt_bimodal()
    vth(T_eV, m) = sqrt(2 * (T_eV * EV) / m)
    # the electrons merged: T_hot 1200 eV (hot, bin 1), T_cold 300 eV (cold), every flow 0
    g = size_grid(tr, ("electrons.cold", "electrons.hot"))
    vref, vcold = vth(1200.0, ELECTRON_MASS), vth(300.0, ELECTRON_MASS)
    Nv = length(g.w0)
    @test isodd(Nv) && length(g.g) == Nv && all(iszero, g.g)
    @test g.u0 === 0.0
    @test isapprox(g.dv, 0.2 * vcold; rtol = 1e-14)            # the defaults: spacing 0.2, extent 4 (the user, U1)
    @test g.sigma === 1.2 * g.dv
    half = (Nv - 1) ÷ 2
    @test half * g.dv >= 4 * vref * (1 - 1e-12)                 # the end nodes reach u ± 4 v_ref
    @test (half - 1) * g.dv < 4 * vref                          # and no fewer would
    @test Nv == 2 * ceil(Int, 4 * vref / (0.2 * vcold)) + 1     # 2 ⌈20 √(1200/300)⌉ + 1 = 81
    @test Nv == 81
    # the default weights: the Maxwellian of T_hot at u0, GaussianGrid's temperature form
    ref = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = (1200.0 * EV) / EV, u0_ms = 0.0, nodes = Nv,
                       spacing = 0.2 * vcold / vref, width = 1.2)
    @test isapprox(collect(g.w0), collect(ref.w0); rtol = 1e-13)
    # the protons merged, with flows from 0 to 200 km/s: u0 the middle, the ends u_lo − 4 v_ref and u_hi + 4 v_ref
    gi = size_grid(tr, ["ions.H", "ions.Hb"])
    vref_i, vcold_i = vth(200.0, 1.008 * ATOMIC_MASS_UNIT), vth(100.0, 1.008 * ATOMIC_MASS_UNIT)
    @test gi.u0 === 100e3
    @test isapprox(gi.dv, 0.2 * vcold_i; rtol = 1e-14)
    hi = (length(gi.w0) - 1) ÷ 2
    @test gi.u0 + hi * gi.dv >= (200e3 + 4 * vref_i) * (1 - 1e-12)
    @test gi.u0 - hi * gi.dv <= (0.0 - 4 * vref_i) + 1e-12 * vref_i
    @test (hi - 1) * gi.dv < 100e3 + 4 * vref_i
    # the keywords, and the symmetric option
    gk = size_grid(tr, "electrons.cold"; spacing = 0.5, extent = 3, width = 1.1)
    @test length(gk.w0) == 2 * ceil(Int, 3 / 0.5) + 1 && gk.sigma === 1.1 * gk.dv
    gs = size_grid(tr, ("electrons.cold", "electrons.hot"); symmetric = true)
    @test length(gs.g) == cld(length(gs.w0), 2)
    @test_throws "size_grid: symmetric = true needs one flow for the whole source; its flows run from 0.0 to 200.0 km/s" size_grid(tr, ["ions.H", "ions.Hb"]; symmetric = true)
    @test_throws "size_grid: spacing must be finite and positive; got 0.0" size_grid(tr, "electrons.cold"; spacing = 0.0)
    @test_throws "size_grid: extent must be finite and positive; got NaN" size_grid(tr, "electrons.cold"; extent = NaN)
    @test_throws "size_grid: width must be finite and positive; got -1.2" size_grid(tr, "electrons.cold"; width = -1.2)
    @test_throws "GaussianGrid: spacing · width must be below 1" size_grid(tr, "electrons.cold"; spacing = 1.0)
    @test_throws "size_grid: \"ions.X\" names no species of the trace" size_grid(tr, "ions.X")
    # a source that is already gridded is sized from its computed moments
    r = project_to_grid(tr, ("electrons.cold", "electrons.hot") => g; names = ("e",))
    gg = size_grid(r.trace, "electrons.e")
    Ts = [plasma(r.trace, j).electrons[1].T for j in 1:4]
    @test isapprox(gg.dv, 0.2 * sqrt(2 * minimum(Ts) / ELECTRON_MASS); rtol = 1e-14)
end

# ------------------------------------------------------------------------------------------
# GT8: the builders' rejection lists name `grid` (the G.1 summary's A6; spec D9)
# ------------------------------------------------------------------------------------------

@testset "GT8: with a shape, ion's rejection message lists grid" begin
    @test_throws "unrecognised keyword(s) (:T_ev,); accepted keywords are (:name, :mass_amu, :mass_kg, :charge, :T_eV, :T_keV, :u_kms, :u_ms, :f, :fraction, :grid)" ion(mass_amu = 2.014, charge = 1, T_eV = 250, T_ev = 1)
    @test_throws "unrecognised keyword(s) (:Te,); accepted keywords are (:name, :mass_amu, :mass_kg, :charge, :f, :fraction, :grid)" ion(mass_amu = 2.014, charge = 1, grid = (T_ref_eV = 250,), Te = 1)
end
