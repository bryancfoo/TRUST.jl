using Test
using OTIS
using LinearAlgebra: BLAS

# DK1 to DK16 against dev/phase3_instructions.md Section 9 (spec D33), gated "deck". DK14
# ("a script deck's callable dependent") is not implemented here: it needs a `CallableNode`
# fifth node type and `name => callable` support in `ParameterSpec`'s constructor
# (src/inversion/parameters.jl, src/inversion/expressions.jl), neither of which exists anywhere
# in the tree at this chunk's start (grep confirms no `CallableNode` in either file) and neither
# of which is in chunk 12's own file list (src/config/deck.jl, scripts/fit.jl, Project.toml,
# examples/*.toml, this file, one include+export line). Reported in the chunk 12 hand-off as a
# finding, not fixed here.

# ------------------------------------------------------------------------------------------
# A small synthetic StreakData file, written once per fixture and reused across tests.
# ------------------------------------------------------------------------------------------

function _dk_write_streak(path::AbstractString; nl=8, nt=4, lambda_lo=262.0, lambda_hi=264.5,
                           t_lo=0.0, t_hi=3.0)
    lambda_m = collect(range(lambda_lo, lambda_hi; length=nl)) .* 1e-9
    t_s = collect(range(t_lo, t_hi; length=nt)) .* 1e-9
    image = fill(30.0, nl, nt)
    sigma = fill(1.0, nl, nt)
    sd = StreakData(; lambda_m=lambda_m, t_s=t_s, image=image, sigma=sigma)
    write_streak(path, sd; source="test")
    return sd
end

# A minimal, valid deck: one fixed ion, one scalar fitted ion temperature and flow, no
# regularizer, no interval, LBFGS with a tiny iteration cap. `datafile` and `outdir` are
# substituted in.
function _dk_minimal_deck(datafile::AbstractString, outdir::AbstractString; extra_geometry="")
    return """
    deck_version = "otis-deck-1"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0
    $extra_geometry

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
    procedure = "lbfgs"
    lbfgs = { iterations = 3 }

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

# ------------------------------------------------------------------------------------------
# DK1: a deck round-trips
# ------------------------------------------------------------------------------------------

@testset "DK1: a deck round-trips" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _dk_minimal_deck(datafile, outdir))

        deck1 = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck1)
        eff = effective_deck(deck1, problem, optimizer)

        effpath = joinpath(dir, "effective.toml")
        write(effpath, eff)
        deck2 = read_deck(effpath)

        @test deck2 == deck1
    end
end

# ------------------------------------------------------------------------------------------
# DK2: a deck reproduces a hand-built problem (P2, D46)
# ------------------------------------------------------------------------------------------

@testset "DK2: a deck reproduces a hand-built problem" begin
    mktempdir() do dir
        nl, nt = 32, 10
        datafile = joinpath(dir, "p2.jld2")
        _dk_write_streak(datafile; nl=nl, nt=nt, lambda_lo=262.0, lambda_hi=264.5, t_lo=0.0, t_hi=3.0)

        # -- hand-built (D46's P2) --
        template = Plasma(; n_e_cm3=3.0e19, electrons=[electrons(T_eV=400.0)],
            ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
        t_ns = collect(range(0.0, 3.0; length=nt))
        trace = PlasmaTrace(template; t_ns=t_ns)
        probe = ProbeBeam(lambda_probe_nm=263.25)
        collection = Collection(theta_deg=60.0, k_model=:exact)
        lambda_m = collect(range(262.0, 264.5; length=nl)) .* 1e-9
        effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6))
        hand_expt0 = Experiment(; trace=trace, probe=probe, collection=collection, lambda_m=lambda_m,
            effects=effects)

        v_T = FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true)
        v_u = FittedVariable("u_D"; bounds=(-200.0, 200.0), init=0.0, time_varying=true)
        hand_pspec = ParameterSpec(hand_expt0; variables=[v_T, v_u],
            bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms"])
        hand_expt = Experiment(; trace=hand_pspec.trace, probe=hand_expt0.probe,
            collection=hand_expt0.collection, lambda_m=hand_expt0.lambda, effects=hand_expt0.effects)
        hand_theta = flatten(hand_pspec)
        hand_M = measure(unflatten(hand_expt, hand_pspec, hand_theta))

        # -- the deck --
        deck_text = """
        deck_version = "otis-deck-1"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0

        [fit.variables]
        T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
        u_D = { bounds = [-200.0, 200.0], init = 0.0, time_varying = true }

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
        spectral_irf = { fwhm_nm = 0.08 }
        temporal_irf = { fwhm_ns = 0.4 }
        sensitivity = { scale = 1.0e-6 }

        [fit]
        sigma = "supplied"
        """
        deckpath = joinpath(dir, "p2.toml")
        write(deckpath, deck_text)
        deck = read_deck(deckpath)
        deck_problem, _, deck_start = OTIS.build_problem(deck)
        deck_M = measure(unflatten(deck_problem.experiment, deck_problem.pspec, deck_start))

        @test deck_start == hand_theta
        @test deck_M == hand_M
    end
end

# ------------------------------------------------------------------------------------------
# DK3: the effective deck records what the input did not
# ------------------------------------------------------------------------------------------

@testset "DK3: the effective deck records what the input did not" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _dk_minimal_deck(datafile, outdir))

        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)

        @test occursin("ions.D.T (unit_key=T_eV)", eff)
        @test occursin("ions.D.u (unit_key=u_kms)", eff)
        @test occursin("axis_unit", eff)
        @test occursin("\"ns\"", eff)
    end
end

@testset "DK3: a dependent's text is recorded verbatim" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir),
            "[plasma]" => "[fit.dependent]\nf_C = \"1 - 0.0*T_D\"\n\n[plasma]")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)

        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)
        @test occursin("1 - 0.0*T_D", eff)
    end
end

# ------------------------------------------------------------------------------------------
# DK4: variable references resolve
# ------------------------------------------------------------------------------------------

@testset "DK4: variable references resolve" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = """
        deck_version = "otis-deck-1"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0

        [fit.variables]
        u1     = { bounds = [-200.0, 200.0], init = 0.0 }
        f_D    = { bounds = [0.3, 1.0], init = 0.9 }
        scale  = { bounds = [1.0e-7, 1.0e-5], init = 1.0e-6 }
        bg     = { bounds = [0.0, 200.0], init = 25.0 }
        c1     = { bounds = [-10.0, 10.0], init = 0.0 }

        [fit.dependent]
        f_C = "1 - f_D"

        [plasma]
        n_e_cm3 = 3.0e19

        [plasma.electrons.e]
        fraction = 1.0
        T_eV = 400.0

        [plasma.ions.D]
        mass_amu = 2.014
        charge = 1
        fraction = "f_D"
        T_eV = 200.0
        u_kms = "u1"

        [plasma.ions.C]
        mass_amu = 12.011
        charge = 6
        fraction = "f_C"
        T_eV = 200.0
        u_kms = "u1"

        [instrument]
        sensitivity = { scale = "scale" }
        background = [{ type = "Polynomial", coefficients = ["bg", "c1", 0.0] }]

        [fit]
        sigma = "supplied"
        """
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)

        bindings = Dict{String,Vector}()
        for b in problem.pspec.bindings
            push!(get!(bindings, b.name, []), b)
        end

        @test length(bindings["u1"]) == 2
        @test all(b -> b.kind === :leaf && string(b.path) in ("ions.D.u", "ions.C.u"), bindings["u1"])
        @test only(bindings["f_D"]).kind === :leaf
        @test string(only(bindings["f_D"]).path) == "ions.D.fraction"
        @test string(only(bindings["f_C"]).path) == "ions.C.fraction"
        @test only(bindings["scale"]).kind === :effect
        @test only(bindings["scale"]).field === :scale
        @test length(bindings["bg"]) == 1 && only(bindings["bg"]).element == 1
        @test length(bindings["c1"]) == 1 && only(bindings["c1"]).element == 2
    end
end

# ------------------------------------------------------------------------------------------
# DK5: a string in a non-numeric key stays a string
# ------------------------------------------------------------------------------------------

@testset "DK5: a string in a non-numeric key stays a string" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = """
        deck_version = "otis-deck-1"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0
        k_model = "exact"

        [fit.variables]
        exact = { bounds = [-1.0, 1.0], init = 0.0 }

        [plasma]
        n_e_cm3 = 3.0e19

        [plasma.electrons.e]
        fraction = 1.0
        T_eV = 400.0
        shape = { type = "SuperGaussian", m = 2.2 }

        [plasma.ions.D]
        mass_amu = 2.014
        charge = 1
        fraction = 1.0
        T_eV = 200.0
        u_kms = 0.0

        [instrument]
        gradient_smearing = { leaf = "plasma.electrons.e.T_eV", width = 10.0, nodes = 3 }

        [fit]
        sigma = "supplied"
        """
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        deck = read_deck(deckpath)

        @test deck.sections["geometry"]["k_model"] == "exact"
        @test deck.sections["plasma"]["electrons"]["e"]["shape"]["type"] == "SuperGaussian"
        @test deck.sections["instrument"]["gradient_smearing"]["leaf"] ==
              "plasma.electrons.e.T_eV"

        problem, optimizer, start = OTIS.build_problem(deck)
        @test problem.experiment.collection.k_model === :exact
    end
end

# ------------------------------------------------------------------------------------------
# DK6: a variable reference naming nothing
# ------------------------------------------------------------------------------------------

@testset "DK6: a variable reference naming nothing" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = """
        deck_version = "otis-deck-1"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0

        [plasma]
        n_e_cm3 = 3.0e19

        [plasma.electrons.e]
        fraction = 1.0
        T_eV = "nosuchvar"

        [fit]
        sigma = "supplied"
        """
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "which is not a declared variable or dependent" read_deck(deckpath)
    end
end

# ------------------------------------------------------------------------------------------
# DK7: unknown keys and unknown names
# ------------------------------------------------------------------------------------------

@testset "DK7: an unknown key in [instrument]" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        deck_text = """
        deck_version = "otis-deck-1"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0

        [plasma]
        n_e_cm3 = 3.0e19
        [plasma.electrons.e]
        fraction = 1.0
        T_eV = 400.0

        [instrument]
        bogus_key = { fwhm_nm = 0.5 }

        [fit]
        sigma = "supplied"
        """
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "unknown key" read_deck(deckpath)
    end
end

@testset "DK7: an unknown shape type" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        deck_text = """
        deck_version = "otis-deck-1"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0

        [plasma]
        n_e_cm3 = 3.0e19
        [plasma.electrons.e]
        fraction = 1.0
        T_eV = 400.0
        shape = { type = "Bogus" }

        [fit]
        sigma = "supplied"
        """
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "unknown shape \"Bogus\"; valid names are" read_deck(deckpath)
    end
end

@testset "DK7: an unknown operator" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir),
            "[instrument]" => """
            [regularizer]
            weight = 1.0
            terms = [{ select = ["T_D"], operator = "Bogus", normalization = "unit" }]

            [instrument]""")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "unknown operator \"Bogus\"; valid names are" read_deck(deckpath)
    end
end

@testset "DK7: an unknown normalization" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir),
            "[instrument]" => """
            [regularizer]
            weight = 1.0
            terms = [{ select = ["T_D"], operator = "D2", normalization = "Bogus" }]

            [instrument]""")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "unknown normalization \"Bogus\"; valid names are" read_deck(deckpath)
    end
end

# ------------------------------------------------------------------------------------------
# DK8: sigma must be "supplied"
# ------------------------------------------------------------------------------------------

@testset "DK8: sigma must be supplied" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir),
            "sigma = \"supplied\"" => "sigma = \"StreakCamera\"")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "sigma is frozen and supplied with the data" read_deck(deckpath)
    end
end

# ------------------------------------------------------------------------------------------
# DK9: the geometry is fixed
# ------------------------------------------------------------------------------------------

@testset "DK9: the geometry is fixed" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir), "theta_deg = 60.0" => "theta_deg = \"th\"")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "the probe and the collection are fixed in Phase 3; use ProbeOffset" read_deck(deckpath)
    end
end

# ------------------------------------------------------------------------------------------
# DK10: the registry has no if chain
# ------------------------------------------------------------------------------------------

@testset "DK10: the registry has no if name == chain" begin
    configdir = joinpath(@__DIR__, "..", "..", "src", "config")
    found = String[]
    for file in readdir(configdir)
        endswith(file, ".jl") || continue
        path = joinpath(configdir, file)
        for (lineno, line) in enumerate(eachline(path))
            code = split(line, '#'; limit=2)[1]
            if occursin("if name ==", code) || occursin("elseif name ==", code)
                push!(found, "$path:$lineno: $(strip(line))")
            end
        end
    end
    @test isempty(found)
end

# ------------------------------------------------------------------------------------------
# DK11: every default is filled
# ------------------------------------------------------------------------------------------

@testset "DK11: every default is filled" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir),
            "[fit]" => "[instrument.k_smearing]\nhalf_angle_deg = 3.0\n\n[fit]")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)

        effpath = joinpath(dir, "eff.toml")
        write(effpath, eff)
        deck2 = read_deck(effpath)

        ks = deck2.sections["instrument"]["k_smearing"]
        @test Set(keys(ks)) == Set(String.(fieldnames(OTIS.KSmearingOptions)))
        @test ks["nodes"] == 5
        @test ks["half_angle_deg"] == 3.0

        lbfgs = deck2.sections["optimizer"]["lbfgs"]
        @test Set(keys(lbfgs)) == Set(String.(fieldnames(LBFGS)))
    end
end

# ------------------------------------------------------------------------------------------
# DK12: a crop selects bins
# ------------------------------------------------------------------------------------------

@testset "DK12: a crop selects bins" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        sd = _dk_write_streak(datafile; nl=16, nt=8, lambda_lo=260.0, lambda_hi=267.0, t_lo=0.0, t_hi=7.0)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir),
            "[fit.variables]" => "[data]\nfile = \"$(replace(datafile, "\\" => "/"))\"\n" *
                                  "lambda_range_nm = [262.0, 264.0]\nt_range_ns = [2.0, 5.0]\n\n[fit.variables]")
        # remove the original bare [data] block (the substitution above adds a second one)
        deck_text = replace(deck_text, r"\[data\]\nfile = \"[^\"]*\"\n\n" => "", count=1)
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)

        l_idx = findall(l -> 262.0e-9 <= l <= 264.0e-9, sd.lambda)
        t_idx = findall(t -> 2.0e-9 <= t <= 5.0e-9, sd.t)
        @test length(l_idx) < length(sd.lambda)
        @test length(t_idx) < length(sd.t)
        @test problem.experiment.lambda == sd.lambda[l_idx]
        @test problem.experiment.t == sd.t[t_idx]
        @test problem.data.image == sd.image[l_idx, t_idx]
    end
end

# ------------------------------------------------------------------------------------------
# DK13: fit pins BLAS
# ------------------------------------------------------------------------------------------

@testset "DK13: fit pins BLAS" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _dk_minimal_deck(datafile, outdir))

        previous = BLAS.get_num_threads()
        BLAS.set_num_threads(4)
        @test BLAS.get_num_threads() == 4
        try
            fit(deckpath)
            @test BLAS.get_num_threads() == 1
        finally
            BLAS.set_num_threads(previous)
        end
    end
end

# ------------------------------------------------------------------------------------------
# DK14: a script deck's callable dependent -- NOT IMPLEMENTED, see the file header.
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# DK15: fit end to end
# ------------------------------------------------------------------------------------------

@testset "DK15: fit end to end" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _dk_minimal_deck(datafile, outdir))

        result = fit(deckpath)
        @test result isa FitResult
        @test isfile(joinpath(outdir, "result.jld2"))
        @test isfile(joinpath(outdir, "effective_deck.toml"))
        r = read_result(joinpath(outdir, "result.jld2"))
        @test r.layout_version == "otis-result-1"
    end
end

# ------------------------------------------------------------------------------------------
# DK16: the deck is versioned and round-trips
# ------------------------------------------------------------------------------------------

@testset "DK16: the deck is versioned and round-trips" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")

        no_version_path = joinpath(dir, "no_version.toml")
        write(no_version_path, replace(_dk_minimal_deck(datafile, outdir),
            "deck_version = \"otis-deck-1\"\n" => ""))
        @test_throws "otis-deck-1" read_deck(no_version_path)

        wrong_version_path = joinpath(dir, "wrong_version.toml")
        write(wrong_version_path, replace(_dk_minimal_deck(datafile, outdir),
            "deck_version = \"otis-deck-1\"" => "deck_version = \"otis-deck-2\""))
        @test_throws "otis-deck-1" read_deck(wrong_version_path)

        good_path = joinpath(dir, "good.toml")
        write(good_path, _dk_minimal_deck(datafile, outdir))
        deck = read_deck(good_path)
        problem, optimizer, start = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)
        @test occursin("deck_version = \"otis-deck-1\"", eff)

        effpath = joinpath(dir, "eff.toml")
        write(effpath, eff)
        @test read_deck(effpath) == deck
    end
end

# ------------------------------------------------------------------------------------------
# DK17: the five unsafe instrument keys refuse a variable reference (carried from chunk 12,
# dev/phase3_prompt_13.md's carried item)
# ------------------------------------------------------------------------------------------
#
# Chunk 12's `_resolve_bindings` (src/inversion/parameters.jl, D9) hardcodes `factor = 1.0` for
# every `:effect` binding -- correct for the four instrument keys whose deck name already equals
# `parameters(effect)`'s own key, and silently wrong for five others (deck.jl's box above
# `_fill_option_fields`). Chunk 12's `_INSTRUMENT_REF_OK` therefore fills those five with
# `refs_allowed = false`, so a deck naming a variable there is refused at `read_deck` time with
# D39's "takes a string, not a variable reference" fragment rather than silently mishandled.
# Nothing tested that containment before this chunk (a grep of this file's own history found no
# row putting a string in any of the five); DK17 is that test. Each string below names a
# *declared* fitted variable ("x"), not a nonexistent name, so a passing row is the "takes a
# string" error and not DK6's "which is not a declared variable or dependent" -- the row must
# fail for the right reason, or it would pass a version of `_fill_option_fields` that checked
# `refs_allowed` after the name lookup instead of before it.

function _dk17_deck(datafile::AbstractString, instrument_line::AbstractString)
    return """
    deck_version = "otis-deck-1"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    x = { bounds = [0.0, 1.0], init = 0.5 }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = 200.0
    u_kms = 0.0

    [instrument]
    $instrument_line

    [fit]
    sigma = "supplied"
    """
end

@testset "DK17: the five unsafe instrument keys refuse a variable reference" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)

        rows = (
            "spectral_irf" => "spectral_irf = { fwhm_nm = \"x\" }",
            "temporal_irf" => "temporal_irf = { fwhm_ns = \"x\" }",
            "probe_offset" => "probe_offset = { probe_offset_nm = \"x\" }",
            "k_smearing" => "k_smearing = { half_angle_deg = \"x\" }",
            "gradient_smearing" => "gradient_smearing = { leaf = \"plasma.electrons.e.T_eV\", width = \"x\" }",
        )
        for (label, instrument_line) in rows
            deckpath = joinpath(dir, "dk17_$label.toml")
            write(deckpath, _dk17_deck(datafile, instrument_line))
            @test_throws "takes a string, not a variable reference" read_deck(deckpath)
        end
    end
end
