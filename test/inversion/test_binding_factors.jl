using Test
using OTIS
using OTIS: jvp!, jtvp!
import ForwardDiff

# BF1 to BF9 against dev/phase4A_instructions.md Section 6 (spec D9-D12), gated "binding
# factors". Step 4.A.1b, chunk 4: closes Phase 3's B5, the box at src/config/deck.jl above
# `_fill_option_fields` -- every effect-parameter binding now reads its conversion factor from
# `_INSTRUMENT_BINDING` instead of silently taking `factor = 1.0`.

# ------------------------------------------------------------------------------------------
# shared fixtures
# ------------------------------------------------------------------------------------------

function _bf_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _bf_write_streak(path::AbstractString; nl=8, nt=4, lambda_lo=262.0, lambda_hi=264.5,
                           t_lo=0.0, t_hi=3.0)
    lambda_m = collect(range(lambda_lo, lambda_hi; length=nl)) .* 1e-9
    t_s = collect(range(t_lo, t_hi; length=nt)) .* 1e-9
    image = fill(30.0, nl, nt)
    sigma = fill(1.0, nl, nt)
    sd = StreakData(; lambda_m=lambda_m, t_s=t_s, image=image, sigma=sigma)
    write_streak(path, sd; source="test")
    return sd
end

# A minimal, valid deck: one fixed ion and electron, no regularizer, no interval. `datafile` and
# `instrument_line`/`variable_line` are substituted in.
function _bf_deck(datafile::AbstractString, instrument_line::AbstractString, variable_line::AbstractString)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(replace(datafile, "\\" => "/"))"

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    $variable_line

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

# A validly-constructed, representative effect of each registered type in _INSTRUMENT_BINDING.
# Several option structs' own field defaults are not independently constructible (SpectralIRF's
# and TemporalIRF's fwhm = 0 is refused, "FWHM must be strictly positive"), so BF1 reads "an
# effect built from that option struct's defaults" as "a validly-constructed representative
# effect of that type" -- parameters(e)'s key set does not depend on the field values, only on
# the type, so any valid instance gives the same answer BF1 checks.
const _BF_SAMPLE_EFFECT = Dict{String,Function}(
    "WavelengthCalibration" => () -> WavelengthCalibration(lambda_offset_nm=0.02),
    "KSmearing" => () -> KSmearing(half_angle_deg=2.0),
    "SpectralIRF" => () -> SpectralIRF(fwhm_nm=0.08),
    "TemporalIRF" => () -> TemporalIRF(fwhm_ns=0.08),
    "Sensitivity" => () -> Sensitivity(scale=1.0e-6),
    "Constant" => () -> ConstantBackground(level=25.0),
    "Polynomial" => () -> PolynomialBackground(coefficients=(1.0, 2.0)),
    "Rational" => () -> RationalBackground(numerator=(20.0, 2.0), denominator=(0.3,)),
)

# ------------------------------------------------------------------------------------------
# BF1: the table is consistent with the code
# ------------------------------------------------------------------------------------------

@testset "BF1: the table is consistent with the code" begin
    for (regname, table) in OTIS._INSTRUMENT_BINDING
        OptT = OTIS._INSTRUMENT_OPTIONS[regname]
        optfields = Set(String.(fieldnames(OptT)))
        effect = _BF_SAMPLE_EFFECT[regname]()
        pnt = parameters(effect)
        for (key, (field, factor, unit_key)) in table
            @testset "($regname, \"$key\") => (:$field, $factor, \"$unit_key\")" begin
                @test key in optfields
                @test field in keys(pnt)
            end
        end
    end
end

# ------------------------------------------------------------------------------------------
# BF2: the three non-unit factors
# ------------------------------------------------------------------------------------------

@testset "BF2: the three non-unit factors are the right numbers" begin
    @test OTIS._INSTRUMENT_BINDING["WavelengthCalibration"]["lambda_offset_nm"][2] == 1e-9
    @test OTIS._INSTRUMENT_BINDING["KSmearing"]["half_angle_deg"][2] == pi / 180
    @test OTIS._INSTRUMENT_BINDING["SpectralIRF"]["fwhm_nm"][2] == 1e-9 / (2 * sqrt(2 * log(2)))
    @test OTIS._INSTRUMENT_BINDING["TemporalIRF"]["fwhm_ns"][2] == 1e-9 / (2 * sqrt(2 * log(2)))
    @test OTIS._INSTRUMENT_BINDING["SpectralIRF"]["fwhm_nm"][2] == 1e-9 * OTIS._SIGMA_PER_FWHM
end

# ------------------------------------------------------------------------------------------
# BF3: the value path
# ------------------------------------------------------------------------------------------

@testset "BF3: the value path" begin
    # "A θ whose x-value is 0.02": vals is the per-bin value vector _rebuild_effect reads
    # (post-_from_unconstrained), not theta itself, so the binding's factor is isolated from the
    # logit round-trip's own few-ULP roundoff (PM3/PM4's own documented slack) -- rtol = 1e-15 is
    # a statement about the multiplication, not about flatten/unflatten's round trip.
    calib = WavelengthCalibration(lambda_offset_nm=0.5)   # any valid effect; only :offset is read
    b = OTIS.Binding("dlam", 1, :effect, OTIS._DUMMY_LEAF_PATH, 0, 1, :offset, 0, 1e-9, "lambda_offset_nm")
    vals = [0.02]
    rebuilt = OTIS._rebuild_effect(calib, [b], vals)
    @test isapprox(rebuilt.offset, 0.02e-9; rtol=1e-15)
end

# ------------------------------------------------------------------------------------------
# BF4: a deck binds the offset by name
# ------------------------------------------------------------------------------------------

@testset "BF4: a deck binds the offset by name" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _bf_write_streak(datafile)
        deckpath = joinpath(dir, "bf4.toml")
        write(deckpath, _bf_deck(datafile, "wavelength_calibration = { lambda_offset_nm = \"dlam\" }",
            "dlam = { bounds = [-0.2, 0.2], init = 0.02 }"))
        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)
        b = only(b for b in problem.pspec.bindings if b.name == "dlam")
        @test b.factor == 1e-9
        @test b.unit_key == "lambda_offset_nm"
    end
end

# ------------------------------------------------------------------------------------------
# BF5: the round-trip -- the B5 row that did not exist before this chunk
# ------------------------------------------------------------------------------------------

@testset "BF5: the round-trip" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _bf_write_streak(datafile)
        deckpath = joinpath(dir, "bf5.toml")
        write(deckpath, _bf_deck(datafile, "wavelength_calibration = { lambda_offset_nm = \"dlam\" }",
            "dlam = { bounds = [-0.2, 0.2], init = 0.02 }"))
        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)
        eff = effective_deck(deck, problem, optimizer)
        @test occursin("unit_key=lambda_offset_nm", eff)

        effpath = joinpath(dir, "bf5_eff.toml")
        write(effpath, eff)
        deck2 = read_deck(effpath)
        problem2, optimizer2, start2 = OTIS.build_problem(deck2)
        b2 = only(b for b in problem2.pspec.bindings if b.name == "dlam")
        @test b2.factor == 1e-9
        @test b2.unit_key == "lambda_offset_nm"
    end
end

# ------------------------------------------------------------------------------------------
# BF6: k_smearing, spectral_irf and temporal_irf bind by name
# ------------------------------------------------------------------------------------------

@testset "BF6: k_smearing, spectral_irf and temporal_irf bind by name" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _bf_write_streak(datafile)

        rows = (
            (instrument_line="k_smearing = { half_angle_deg = \"x\" }", varname="x",
             expected_factor=pi / 180, expected_unit_key="half_angle_deg", init=2.0,
             bounds="[0.001, 10.0]", field=:half_angle, regname="KSmearing"),
            (instrument_line="spectral_irf = { fwhm_nm = \"x\" }", varname="x",
             expected_factor=1e-9 * OTIS._SIGMA_PER_FWHM, expected_unit_key="fwhm_nm", init=0.08,
             bounds="[0.02, 0.5]", field=:sigma, regname="SpectralIRF"),
            (instrument_line="temporal_irf = { fwhm_ns = \"x\" }", varname="x",
             expected_factor=1e-9 * OTIS._SIGMA_PER_FWHM, expected_unit_key="fwhm_ns", init=0.08,
             bounds="[0.02, 0.5]", field=:sigma, regname="TemporalIRF"),
        )
        for (i, row) in enumerate(rows)
            deckpath = joinpath(dir, "bf6_$i.toml")
            write(deckpath, _bf_deck(datafile, row.instrument_line,
                "$(row.varname) = { bounds = $(row.bounds), init = $(row.init) }"))
            deck = read_deck(deckpath)
            problem, optimizer, start = OTIS.build_problem(deck)
            b = only(b for b in problem.pspec.bindings if b.name == row.varname)
            @test b.factor == row.expected_factor
            @test b.unit_key == row.expected_unit_key

            theta = flatten(problem.pspec)
            result = unflatten(problem.experiment, problem.pspec, theta)
            eidx = findfirst(e -> e isa OTIS._EFFECT_CTOR[row.regname], result.effects)
            @test eidx !== nothing
            got = parameters(result.effects[eidx])[row.field]
            reldiff = abs(got - row.init * row.expected_factor) / abs(row.init * row.expected_factor)
            @info "BF6 unflatten relative difference" row.regname reldiff
            @test isapprox(got, row.init * row.expected_factor; rtol=1e-15)
        end
    end
end

# ------------------------------------------------------------------------------------------
# BF7: the one-factor rule now covers effect bindings
# ------------------------------------------------------------------------------------------

@testset "BF7: the one-factor rule now covers effect bindings" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _bf_write_streak(datafile)
        deckpath = joinpath(dir, "bf7.toml")
        write(deckpath, """
        deck_version = "otis-deck-2"

        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [geometry]
        lambda_probe_nm = 263.25
        theta_deg = 60.0

        [fit.variables]
        shared = { bounds = [0.0, 1.0], init = 0.5 }

        [plasma]
        n_e_cm3 = 3.0e19

        [plasma.electrons.e]
        fraction = 1.0
        T_eV = "shared"

        [plasma.ions.D]
        mass_amu = 2.014
        charge = 1
        fraction = 1.0
        T_eV = 200.0
        u_kms = 0.0

        [instrument]
        spectral_irf = { fwhm_nm = "shared" }

        [fit]
        sigma = "supplied"
        """)
        deck = read_deck(deckpath)
        err = nothing
        try
            OTIS.build_problem(deck)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("shared", err.msg)
        @test occursin("T_eV", err.msg)
        @test occursin("fwhm_nm", err.msg)
    end
end

# ------------------------------------------------------------------------------------------
# BF8: the chain-global column carries the factor
# ------------------------------------------------------------------------------------------

@testset "BF8: the chain-global column carries the factor" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _bf_write_streak(datafile)
        deckpath = joinpath(dir, "bf8.toml")
        write(deckpath, _bf_deck(datafile, "spectral_irf = { fwhm_nm = \"sig\" }",
            "sig = { bounds = [0.02, 0.5], init = 0.08 }"))
        deck = read_deck(deckpath)
        problem, optimizer, start = OTIS.build_problem(deck)
        theta = start .+ 0.01 .* (1:length(start))

        Jac = jacobian(problem, theta)
        sig_idx = findfirst(v -> v.name == "sig", problem.pspec.variables)
        theta_idx = problem.pspec.offsets[sig_idx]
        @test theta_idx in Jac.chain_global_idx

        J_bruteforce = ForwardDiff.jacobian(th -> residuals(problem, th), theta)
        v = zeros(length(theta)); v[theta_idx] = 1.0
        out = zeros(problem.npixels)
        jvp!(out, problem, Jac, v)
        oracle = J_bruteforce[:, theta_idx]
        floor_ = 1e-10 * maximum(abs, oracle)
        relerr = maximum(abs.(out .- oracle) ./ max.(abs.(oracle), floor_))
        @info "BF8 measured relative difference" relerr
        @test relerr <= 1e-10
    end
end

# ------------------------------------------------------------------------------------------
# BF9: gradient_smearing.width still refuses
# ------------------------------------------------------------------------------------------

@testset "BF9: gradient_smearing.width still refuses" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _bf_write_streak(datafile)
        deckpath = joinpath(dir, "bf9.toml")
        write(deckpath, _bf_deck(datafile,
            "gradient_smearing = { leaf = \"plasma.electrons.e.T_eV\", width = \"x\" }",
            "x = { bounds = [0.0, 1.0], init = 0.5 }"))
        @test_throws "its conversion factor is the smeared leaf's own" read_deck(deckpath)
    end
end
