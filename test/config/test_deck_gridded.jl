# Step G.2.2: one registry, and a deck that may name a grid (Phase G.2 spec D13; OQ-69 item 5, the user's decision
# of one registry). Gate "gridded registry": GR1 to GR4. A species table's `shape = { type = "GaussianGrid", … }`
# builds the species through the builders' `grid` keyword, with the species' own mass; every key is a fixed
# number; T and u beside it, a reference in any key and a missing T_ref_eV are refused at read_deck; the effective
# deck writes all six keys and reads back; the deck forward-models. The oracle is the builders' own grid
# (conventions.md 8.2's "the generating parameters"); every row is exact.

using Test
using TRUST
using TRUST: SHAPE_REGISTRY, _SHAPE_OPTIONS, GaussianGridOptions, build_problem, effective_deck

function _gr_streak(path)
    sd = StreakData(; lambda_m = collect(range(262.0, 264.5; length = 8)) .* 1e-9,
                    t_s = collect(range(0.0, 3.0; length = 4)) .* 1e-9, image = fill(30.0, 8, 4), sigma = fill(1.0, 8, 4))
    write_streak(path, sd; source = "GR")
end

# A minimal deck (test_deck.jl's DK fixture, its electrons on a grid): the deuterons' T and u fitted.
function _gr_deck(datafile, outdir; eshape = "{ type = \"GaussianGrid\", T_ref_eV = 400.0, nodes = 41 }",
                  eextra = "", dshape = "")
    return """
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
    shape = $eshape
    $eextra

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = "u_D"
    $dshape

    [instrument]
    sensitivity = { scale = 1.0e-6 }

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "lbfgs"
    lbfgs = { iterations = 3 }

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

function _gr_read(dir, text; name = "deck.toml")
    path = joinpath(dir, name)
    write(path, text)
    return read_deck(path)
end

@testset "GR1: GaussianGrid in the one registry and the deck's table of option structs" begin
    @test SHAPE_REGISTRY["GaussianGrid"] === GaussianGrid
    @test _SHAPE_OPTIONS["GaussianGrid"] === GaussianGridOptions
    o = GaussianGridOptions()
    @test isnan(o.T_ref_eV) && o.u0_kms === 0.0 && o.nodes === 41 && o.spacing === 0.25 && o.width === 1.2 && o.symmetric === false
    @test fieldnames(GaussianGridOptions) == (:T_ref_eV, :u0_kms, :nodes, :spacing, :width, :symmetric)
end

@testset "GR2: a deck that names a grid builds it with the species' own mass, and the effective deck writes all six keys" begin
    mktempdir() do dir
        data = joinpath(dir, "s.jld2"); _gr_streak(data)
        deck = _gr_read(dir, _gr_deck(data, joinpath(dir, "out")))
        sh = deck.sections["plasma"]["electrons"]["e"]["shape"]
        @test sh == Dict{String,Any}("type" => "GaussianGrid", "T_ref_eV" => 400.0, "u0_kms" => 0.0, "nodes" => 41,
                                      "spacing" => 0.25, "width" => 1.2, "symmetric" => false)
        problem, optimizer, _ = build_problem(deck)
        e = problem.experiment.trace.template.electrons[1]
        @test e.f === electrons(grid = (T_ref_eV = 400.0,)).f
        @test e.T === electrons(grid = (T_ref_eV = 400.0,)).T
        text = effective_deck(deck, problem, optimizer)
        @test all(occursin("$k = ", text) for k in ("T_ref_eV", "u0_kms", "nodes", "spacing", "width", "symmetric"))
        again = _gr_read(dir, text; name = "effective.toml")
        @test again == deck
        # an ion's grid takes the ion's mass; every key given
        d2 = _gr_read(dir, _gr_deck(data, joinpath(dir, "out2"); eshape = "{ type = \"Maxwellian\" }", eextra = "T_eV = 400.0",
            dshape = ""); name = "d2.toml")
        @test d2.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "Maxwellian"
        txt3 = replace(_gr_deck(data, joinpath(dir, "out3")),
            "shape = { type = \"GaussianGrid\", T_ref_eV = 400.0, nodes = 41 }" =>
            "shape = { type = \"GaussianGrid\", T_ref_eV = 400.0, u0_kms = 10.0, nodes = 31, spacing = 0.3, width = 1.1, symmetric = true }")
        p3, _, _ = build_problem(_gr_read(dir, txt3; name = "d3.toml"))
        @test p3.experiment.trace.template.electrons[1].f ===
              electrons(grid = (T_ref_eV = 400.0, u0_kms = 10.0, nodes = 31, spacing = 0.3, width = 1.1, symmetric = true)).f
    end
end

@testset "GR3: the deck's refusals of a grid's keys, at read_deck" begin
    mktempdir() do dir
        data = joinpath(dir, "s.jld2"); _gr_streak(data)
        out = joinpath(dir, "out")
        for (k, v) in (("T_ref_eV", "\"T_D\""), ("u0_kms", "\"u_D\""), ("nodes", "\"T_D\""), ("spacing", "\"T_D\""),
                       ("width", "\"T_D\""), ("symmetric", "\"T_D\""))
            extra = k == "T_ref_eV" ? "" : ", T_ref_eV = 400.0"
            sh = "{ type = \"GaussianGrid\", $k = $v$extra }"
            @test_throws "read_deck: key \"$k\" in section \"plasma.electrons.e.shape\" cannot be a variable reference: a GaussianGrid's keys take numbers only, its grid being fixed for a whole fit" _gr_read(dir, _gr_deck(data, out; eshape = sh))
        end
        @test_throws "read_deck: section \"plasma.electrons.e.shape\" needs \"T_ref_eV\" for a GaussianGrid" _gr_read(dir, _gr_deck(data, out; eshape = "{ type = \"GaussianGrid\", nodes = 41 }"))
        for extra in ("T_eV = 400.0", "T_keV = 0.4", "u_kms = 10.0", "u_ms = 1.0e4")
            key = first(split(extra, " "))
            @test_throws "read_deck: species \"plasma.electrons.e\" carries a lab-frame distribution (GaussianGrid): its T and u are computed from the distribution and cannot be given; got (\"$key\",)" _gr_read(dir, _gr_deck(data, out; eextra = extra))
        end
        @test_throws "read_deck: unknown key \"bogus\" in section \"plasma.electrons.e.shape\"" _gr_read(dir, _gr_deck(data, out; eshape = "{ type = \"GaussianGrid\", T_ref_eV = 400.0, bogus = 1 }"))
        @test_throws "read_deck: key \"nodes\" in section \"plasma.electrons.e.shape\" must be an integer; got 41.5" _gr_read(dir, _gr_deck(data, out; eshape = "{ type = \"GaussianGrid\", T_ref_eV = 400.0, nodes = 41.5 }"))
    end
end

@testset "GR4: a deck with a gridded species forward-models and fits everything but the distribution" begin
    mktempdir() do dir
        data = joinpath(dir, "s.jld2"); _gr_streak(data)
        problem, _, start = build_problem(_gr_read(dir, _gr_deck(data, joinpath(dir, "out"))))
        r = residuals(problem, start)
        @test length(r) == 32 && all(isfinite, r)
        @test string.(problem.experiment.trace.paths) == ["ions.D.T", "ions.D.u"]
    end
end
