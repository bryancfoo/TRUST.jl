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
    deck_version = "otis-deck-2"

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
        deck_version = "otis-deck-2"

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
        deck_version = "otis-deck-2"

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
        deck_version = "otis-deck-2"

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
        deck_version = "otis-deck-2"

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
        deck_version = "otis-deck-2"

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
        deck_version = "otis-deck-2"

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

@testset "DK7: an unknown key in [data]" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deck_text = replace(_dk_minimal_deck(datafile, outdir), "[data]" => "[data]\nbogus_key = 1.0")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, deck_text)
        @test_throws "unknown key" read_deck(deckpath)
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
        @test_throws "wavelength_calibration" read_deck(deckpath)
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
        @test r.layout_version == "otis-result-3"
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
            "deck_version = \"otis-deck-2\"\n" => ""))
        @test_throws "otis-deck-2" read_deck(no_version_path)

        # otis-deck-1 is refused by name, not just as an unknown string: it names the version
        # this deck actually has.
        wrong_version_path = joinpath(dir, "wrong_version.toml")
        write(wrong_version_path, replace(_dk_minimal_deck(datafile, outdir),
            "deck_version = \"otis-deck-2\"" => "deck_version = \"otis-deck-1\""))
        @test_throws "otis-deck-1" read_deck(wrong_version_path)

        good_path = joinpath(dir, "good.toml")
        write(good_path, _dk_minimal_deck(datafile, outdir))
        deck = read_deck(good_path)
        problem, optimizer, start = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)
        @test occursin("deck_version = \"otis-deck-2\"", eff)

        effpath = joinpath(dir, "eff.toml")
        write(effpath, eff)
        @test read_deck(effpath) == deck
    end
end

# ------------------------------------------------------------------------------------------
# DK17: the one instrument key that still refuses a variable reference; the calibration offset
# now binds (Phase 4.A, Step 4.A.1b, D9-D12, chunk 4, 2026-09-23; carried from chunk 12,
# dev/phase3_prompt_13.md's carried item -- shape changed, [propagate] to roadmap.md Step
# 4.A.1's Test sentence)
# ------------------------------------------------------------------------------------------
#
# Through chunk 3, `_resolve_bindings` (src/inversion/parameters.jl, D9) hardcoded `factor = 1.0`
# for every `:effect` binding -- correct for the four instrument keys whose deck name already
# equalled `parameters(effect)`'s own key, silently wrong for five others (deck.jl's box above
# `_fill_option_fields`). Chunk 4's `_INSTRUMENT_BINDING` table (D9) closes that: every key but
# one now converts correctly and binds. **`gradient_smearing.width` is the one exception** (D12):
# its conversion factor is the smeared leaf's own, not a constant of the key, so it cannot live in
# the table, and it is refused with a message that says why rather than the generic "takes a
# string" fragment (this is the one row DK17 keeps). The other four rows this @testset used to
# hold (spectral_irf, temporal_irf, wavelength_calibration, k_smearing) are deleted; BF4 and BF6
# (test/inversion/test_binding_factors.jl) cover them as positive rows. **DK17 keeps one positive
# row of its own** (A19, amended 2026-09-23): the calibration offset deck must *build*
# (`OTIS.build_problem`), not just `read_deck`, because a revert of the table lookup in
# `_build_effect_and_pending!` fails at build time, not at read time -- without this row,
# reverting D9's table lookup in `deck.jl` leaves `deck`'s own pass count unchanged.

function _dk17_deck(datafile::AbstractString, instrument_line::AbstractString)
    return """
    deck_version = "otis-deck-2"

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

@testset "DK17: gradient_smearing.width still refuses; the calibration offset now binds" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)

        refused_path = joinpath(dir, "dk17_gradient_smearing.toml")
        write(refused_path, _dk17_deck(datafile,
            "gradient_smearing = { leaf = \"plasma.electrons.e.T_eV\", width = \"x\" }"))
        @test_throws "its conversion factor is the smeared leaf's own" read_deck(refused_path)

        builds_path = joinpath(dir, "dk17_wavelength_calibration.toml")
        write(builds_path, _dk17_deck(datafile, "wavelength_calibration = { lambda_offset_nm = \"x\" }"))
        deck = read_deck(builds_path)
        problem, optimizer, start = OTIS.build_problem(deck)
        b = only(b for b in problem.pspec.bindings if b.name == "x")
        @test b.factor == 1e-9
        @test b.unit_key == "lambda_offset_nm"
    end
end

# ------------------------------------------------------------------------------------------
# DK18: `precondition` round-trips as a Bool, and a non-Bool value is refused cleanly (D41)
# ------------------------------------------------------------------------------------------
#
# `_fill_option_field` classified every `Bool` field (`precondition` is the first one on any
# optimizer) through its `FT <: Integer` branch, since `Bool <: Integer` in Julia: a deck's
# `precondition = false` built and round-tripped, but the effective deck wrote `precondition = 0`,
# and `precondition = 2` reached `LBFGS`'s own constructor and raised a raw `InexactError: Bool(2)`
# rather than a `read_deck` error. D41 gives `Bool` its own branch, checked *before* `Integer`'s.

@testset "DK18: precondition round-trips as Bool, and a non-Bool value is refused cleanly" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")
        base = _dk_minimal_deck(datafile, outdir)

        deck_false = replace(base, "lbfgs = { iterations = 3 }" =>
            "lbfgs = { iterations = 3, precondition = false }")
        deckpath = joinpath(dir, "dk18_false.toml")
        write(deckpath, deck_false)

        deck1 = read_deck(deckpath)
        @test deck1.sections["optimizer"]["lbfgs"]["precondition"] === false

        problem, optimizer, start = OTIS.build_problem(deck1)
        @test optimizer.precondition === false

        eff = effective_deck(deck1, problem, optimizer)
        @test occursin("precondition = false", eff)
        @test !occursin("precondition = 0", eff)
        @test !occursin("precondition = 1", eff)

        effpath = joinpath(dir, "dk18_effective.toml")
        write(effpath, eff)
        deck2 = read_deck(effpath)
        @test deck2.sections["optimizer"]["lbfgs"]["precondition"] === false
        @test deck2 == deck1

        deck_bad = replace(base, "lbfgs = { iterations = 3 }" =>
            "lbfgs = { iterations = 3, precondition = 2 }")
        badpath = joinpath(dir, "dk18_bad.toml")
        write(badpath, deck_bad)
        @test_throws "key \"precondition\" in section \"optimizer.lbfgs\" must be true or false" read_deck(badpath)
    end
end

# ------------------------------------------------------------------------------------------
# DK19: the top-level known-key check (D40 item 1, chunk 7, Pitfall 9)
# ------------------------------------------------------------------------------------------
#
# Before chunk 7, read_deck read the sections it knew by name and silently ignored everything
# else, so a [channel.epw] table in an otis-deck-2 deck would be silently ignored and the fit
# would run single-channel. The channel deck's own CD1 covers a [channel.x] table specifically;
# this row covers a plain unknown top-level key on an otherwise-shorthand deck.

@testset "DK19: an unknown top-level key is refused" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _dk_write_streak(datafile)
        outdir = joinpath(dir, "out")

        bad = _dk_minimal_deck(datafile, outdir) * "\n[bogus_section]\nx = 1.0\n"
        badpath = joinpath(dir, "dk19_bad.toml")
        write(badpath, bad)
        @test_throws "unknown top-level key \"bogus_section\"" read_deck(badpath)
    end
end

# ------------------------------------------------------------------------------------------
# Fixtures for DK20-DK22 (Phase 4.B spec D5, D21): fit(deck)'s minimum report.
# ------------------------------------------------------------------------------------------

# Phase 4.B spec D5: P1's noise-free streak (one bin, 24 wavelengths, sigma = 1), truth
# T_D = 250 eV, u_D = 40 km/s, for the minimum-report rows DK20-DK22.
function _dk_p1_streak(path::AbstractString)
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=250.0, u_kms=40.0)])
    trace = PlasmaTrace(template; t_ns=[0.0])
    expt = Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact),
        lambda_m=collect(range(262.0, 264.5; length=24)) .* 1e-9,
        effects=(Sensitivity(scale=1.0e-6), ConstantBackground(level=25.0)))
    image = measure(expt)
    sd = StreakData(lambda_m=expt.lambda, t_s=expt.t, image=image, sigma=ones(size(image)))
    write_streak(path, sd; source="test")
    return sd
end

function _dk_p1_deck(datafile, outdir; t_bounds="[50.0, 800.0]", t_init=200.0,
                     optimizer="procedure = \"levenberg_marquardt\"", extra_variables="")
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    T_D = { bounds = $t_bounds, init = $t_init }
    u_D = { bounds = [-200.0, 200.0], init = 10.0 }
    $extra_variables

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
    $optimizer

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

# Phase 4.B spec D5: the warnings are captured with Test.TestLogger and
# Base.CoreLogging.with_logger, so no Logging entry is added to test/Project.toml.
function _dk_fit_logged(deckpath)
    logger = Test.TestLogger(min_level=Base.CoreLogging.Warn)
    result = Base.CoreLogging.with_logger(() -> fit(deckpath), logger)
    return result, logger.logs
end
_dk_count(logs, needle) = count(l -> occursin(needle, string(l.message)), logs)

# ------------------------------------------------------------------------------------------
# DK20: fit(deck) writes fit/gn_decrease (Phase 4.B spec D5, D21)
# ------------------------------------------------------------------------------------------

@testset "DK20: fit(deck) writes fit/gn_decrease" begin
    mktempdir() do dir
        datafile = joinpath(dir, "p1.jld2")
        _dk_p1_streak(datafile)

        # Fixture A2: lbfgs, one iteration, far from a minimum.
        outdir_a2 = joinpath(dir, "a2")
        deckpath_a2 = joinpath(dir, "a2.toml")
        write(deckpath_a2, _dk_p1_deck(datafile, outdir_a2;
            optimizer="procedure = \"lbfgs\"\n    lbfgs = { f_abstol_per_pixel = 1.0e6, patience = 1 }"))
        result_a2 = fit(deckpath_a2)
        @test result_a2.converged == true
        @test result_a2.status == :f_tolerance
        @test result_a2.iterations == 1

        r_a2 = read_result(joinpath(outdir_a2, "result.jld2"))
        @test r_a2.fit.gn_decrease > OTIS.GN_DECREASE_WARN
        problem2, _, _ = OTIS.build_problem(read_deck(deckpath_a2))
        @test r_a2.fit.gn_decrease ≈ gn_decrease(problem2, result_a2.theta) rtol = 1e-12

        # Fixture C: levenberg_marquardt, converges to a genuine minimum.
        outdir_c = joinpath(dir, "c")
        deckpath_c = joinpath(dir, "c.toml")
        write(deckpath_c, _dk_p1_deck(datafile, outdir_c))
        result_c = fit(deckpath_c)
        r_c = read_result(joinpath(outdir_c, "result.jld2"))
        @test 0 <= r_c.fit.gn_decrease < 1e-10
    end
end

# ------------------------------------------------------------------------------------------
# DK21: the two warnings, exactly once, on the row that earns them (Phase 4.B spec D5, D21)
# ------------------------------------------------------------------------------------------

@testset "DK21: fit(deck) warns exactly on the row that earns it" begin
    mktempdir() do dir
        datafile = joinpath(dir, "p1.jld2")
        _dk_p1_streak(datafile)

        # Fixture A2: the decrease warning, once; not the fraction one.
        outdir_a2 = joinpath(dir, "a2")
        deckpath_a2 = joinpath(dir, "a2.toml")
        write(deckpath_a2, _dk_p1_deck(datafile, outdir_a2;
            optimizer="procedure = \"lbfgs\"\n    lbfgs = { f_abstol_per_pixel = 1.0e6, patience = 1 }"))
        result_a2, logs_a2 = _dk_fit_logged(deckpath_a2)
        @test _dk_count(logs_a2, "gn_decrease exceeds GN_DECREASE_WARN") == 1
        @test _dk_count(logs_a2, "near-bound fraction reaches") == 0
        decrease_log = only(filter(l -> occursin("gn_decrease exceeds GN_DECREASE_WARN", string(l.message)), logs_a2))
        r_a2 = read_result(joinpath(outdir_a2, "result.jld2"))
        @test decrease_log.kwargs[:gn_decrease] == r_a2.fit.gn_decrease
        @test decrease_log.kwargs[:threshold] == 1e-2

        # Fixture B: T_D pinned at its upper bound, fraction exactly 0.5.
        outdir_b = joinpath(dir, "b")
        deckpath_b = joinpath(dir, "b.toml")
        write(deckpath_b, _dk_p1_deck(datafile, outdir_b; t_bounds="[50.0, 150.0]", t_init=100.0))
        result_b, logs_b = _dk_fit_logged(deckpath_b)
        @test _dk_count(logs_b, "near-bound fraction reaches SATURATION_FRACTION") == 1
        @test _dk_count(logs_b, "gn_decrease exceeds") == 0
        fraction_log = only(filter(l -> occursin("near-bound fraction reaches", string(l.message)), logs_b))
        @test fraction_log.kwargs[:near_bound_fraction] >= 0.5

        # Fixture C: no warning at all.
        outdir_c = joinpath(dir, "c")
        deckpath_c = joinpath(dir, "c.toml")
        write(deckpath_c, _dk_p1_deck(datafile, outdir_c))
        result_c, logs_c = _dk_fit_logged(deckpath_c)
        @test isempty(logs_c)
    end
end

# ------------------------------------------------------------------------------------------
# DK22: a PosDefException becomes NaN with a warning, never a lost fit (Phase 4.B spec D5)
# ------------------------------------------------------------------------------------------

@testset "DK22: gn_decrease could not be computed" begin
    mktempdir() do dir
        datafile = joinpath(dir, "p1.jld2")
        _dk_p1_streak(datafile)

        # Fixture D: C plus a variable no residual depends on.
        outdir_d = joinpath(dir, "d")
        deckpath_d = joinpath(dir, "d.toml")
        write(deckpath_d, _dk_p1_deck(datafile, outdir_d;
            extra_variables="junk = { bounds = [0.0, 1.0], init = 0.5 }"))
        result_d, logs_d = _dk_fit_logged(deckpath_d)
        @test result_d isa FitResult
        @test _dk_count(logs_d, "gn_decrease could not be computed") == 1
        @test _dk_count(logs_d, "gn_decrease exceeds") == 0
        @test isfile(joinpath(outdir_d, "result.jld2"))
        @test isfile(joinpath(outdir_d, "effective_deck.toml"))

        r_d = read_result(joinpath(outdir_d, "result.jld2"))
        @test isnan(r_d.fit.gn_decrease)
    end
end
