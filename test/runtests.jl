using SafeTestsets
using Test

# D47 (the user, 2026-09-11): a comma-separated, case-insensitive OTIS_TESTS filter, matched
# against each @safetestset's own name, with a plain `if` around each. Unset or empty runs
# everything, so every existing invocation (and Level 1's own) keeps working unchanged. D48
# (Level 2, 2026-09-11): the gates below live inside one outer `@testset "OTIS"` because a
# top-level `@safetestset` throws the instant it records a failure, which would otherwise kill
# this script and silently skip every gate written after it; a nested testset records the
# failure and lets its siblings run, so the summary always names every gate that was selected.
const _OTIS_TESTS_RAW = get(ENV, "OTIS_TESTS", "")
const _OTIS_TESTS_TOKENS = [lowercase(strip(t)) for t in split(_OTIS_TESTS_RAW, ",") if !isempty(strip(t))]
_wanted(name::AbstractString) = isempty(_OTIS_TESTS_TOKENS) ||
    any(occursin(tok, lowercase(name)) for tok in _OTIS_TESTS_TOKENS)

@testset "OTIS" verbose = true begin
    if _wanted("smoke")
        @safetestset "smoke" begin include("test_smoke.jl") end
    end
    if _wanted("constants")
        @safetestset "constants" begin include("test_constants.jl") end
    end
    if _wanted("plasma dispersion")
        @safetestset "plasma dispersion" begin include("test_plasma_dispersion.jl") end
    end
    if _wanted("distribution interface")
        @safetestset "distribution interface" begin include("distributions/test_distribution.jl") end
    end
    if _wanted("Maxwellian")
        @safetestset "Maxwellian" begin include("distributions/test_maxwellian.jl") end
    end
    if _wanted("SuperGaussian")
        @safetestset "SuperGaussian" begin include("distributions/test_supergaussian.jl") end
    end
    if _wanted("SuperGaussian table")
        @safetestset "SuperGaussian table" begin include("distributions/test_supergaussian_table.jl") end
    end
    if _wanted("SuperGaussian widening")
        @safetestset "SuperGaussian widening" begin include("distributions/test_supergaussian_widening.jl") end
    end
    if _wanted("SuperGaussian m-derivative")
        @safetestset "SuperGaussian m-derivative" begin include("distributions/test_supergaussian_dm.jl") end
    end
    if _wanted("Chebyshev table")
        @safetestset "Chebyshev table" begin include("distributions/test_chebyshev_table.jl") end
    end
    if _wanted("Kappa")
        @safetestset "Kappa" begin include("distributions/test_kappa.jl") end
    end
    if _wanted("GeneralizedNormal")
        @safetestset "GeneralizedNormal" begin include("distributions/test_generalized_normal.jl") end
    end
    if _wanted("SkewNormal")
        @safetestset "SkewNormal" begin include("distributions/test_skew_normal.jl") end
    end
    if _wanted("shape cost")
        @safetestset "shape cost" begin include("distributions/test_shape_cost.jl") end
    end
    if _wanted("species")
        @safetestset "species" begin include("plasma/test_species.jl") end
    end
    if _wanted("plasma parameters")
        @safetestset "plasma parameters" begin include("plasma/test_parameters.jl") end
    end
    if _wanted("plasma")
        @safetestset "plasma" begin include("plasma/test_plasma.jl") end
    end
    if _wanted("plasma trace")
        @safetestset "plasma trace" begin include("plasma/test_trace.jl") end
    end
    if _wanted("expressions")
        @safetestset "expressions" begin include("inversion/test_expressions.jl") end
    end
    if _wanted("fit parameters")
        @safetestset "fit parameters" begin include("inversion/test_parameters.jl") end
    end
    if _wanted("binding factors")
        @safetestset "binding factors" begin include("inversion/test_binding_factors.jl") end
    end
    if _wanted("fit residuals")
        @safetestset "fit residuals" begin include("inversion/test_residuals.jl") end
    end
    if _wanted("fit jacobian")
        @safetestset "fit jacobian" begin include("inversion/test_jacobian.jl") end
    end
    if _wanted("joint fit")
        @safetestset "joint fit" begin include("inversion/test_joint.jl") end
    end
    if _wanted("two-channel problem")
        @safetestset "two-channel problem" begin include("inversion/test_two_channel.jl") end
    end
    if _wanted("regularizer")
        @safetestset "regularizer" begin include("inversion/test_regularizer.jl") end
    end
    if _wanted("objective")
        @safetestset "objective" begin include("inversion/test_objective.jl") end
    end
    if _wanted("optimizers")
        @safetestset "optimizers" begin include("inversion/test_optimizers.jl") end
    end
    if _wanted("optimizer diagnostics")
        @safetestset "optimizer diagnostics" begin include("inversion/test_optimizers_diagnostics.jl") end
    end
    if _wanted("weight scan")
        @safetestset "weight scan" begin include("inversion/test_weight_scan.jl") end
    end
    if _wanted("L-curve corner")
        @safetestset "L-curve corner" begin include("inversion/test_lcurve_corner.jl") end
    end
    if _wanted("uncertainty")
        @safetestset "uncertainty" begin include("inversion/test_uncertainty.jl") end
    end
    if _wanted("susceptibility")
        @safetestset "susceptibility" begin include("scattering/test_susceptibility.jl") end
    end
    if _wanted("spectral density")
        @safetestset "spectral density" begin include("scattering/test_spectral_density.jl") end
    end
    if _wanted("scattering geometry")
        @safetestset "scattering geometry" begin include("scattering/test_geometry.jl") end
    end
    if _wanted("spectrum")
        @safetestset "spectrum" begin include("scattering/test_spectrum.jl") end
    end
    if _wanted("super-Gaussian spectrum")
        @safetestset "super-Gaussian spectrum" begin include("scattering/test_spectrum_supergaussian.jl") end
    end
    if _wanted("refactor (Step 2.3a)")
        @safetestset "refactor (Step 2.3a)" begin include("scattering/test_refactor.jl") end
    end
    if _wanted("convective gain")
        @safetestset "convective gain" begin include("scattering/test_gain.jl") end
    end
    if _wanted("instrument effects")
        @safetestset "instrument effects" begin include("instrument/test_effects.jl") end
    end
    if _wanted("wavelength calibration")
        @safetestset "wavelength calibration" begin include("instrument/test_calibration.jl") end
    end
    if _wanted("sensitivity")
        @safetestset "sensitivity" begin include("instrument/test_sensitivity.jl") end
    end
    if _wanted("instrument background")
        @safetestset "instrument background" begin include("instrument/test_background.jl") end
    end
    if _wanted("instrument experiment")
        @safetestset "instrument experiment" begin include("instrument/test_experiment.jl") end
    end
    if _wanted("instrument measure")
        @safetestset "instrument measure" begin include("instrument/test_measure.jl") end
    end
    if _wanted("data streak")
        @safetestset "data streak" begin include("data/test_streak.jl") end
    end
    if _wanted("streak averaging")
        @safetestset "streak averaging" begin include("data/test_average.jl") end
    end
    if _wanted("data io")
        @safetestset "data io" begin include("data/test_io.jl") end
    end
    if _wanted("result file")
        @safetestset "result file" begin include("data/test_result.jl") end
    end
    if _wanted("warp")
        @safetestset "warp" begin include("data/test_warp.jl") end
    end
    if _wanted("deck")
        @safetestset "deck" begin include("config/test_deck.jl") end
    end
    if _wanted("near-bound warning")
        @safetestset "near-bound warning" begin include("config/test_deck_near_bound.jl") end
    end
    if _wanted("channel deck")
        @safetestset "channel deck" begin include("config/test_channel_deck.jl") end
    end
    if _wanted("staged fit")
        @safetestset "staged fit" begin include("config/test_deck_staged.jl") end
    end
    if _wanted("pixel integration")
        @safetestset "pixel integration" begin include("instrument/test_pixel_integration.jl") end
    end
    if _wanted("resonance width")
        @safetestset "resonance width" begin include("config/test_deck_resonance.jl") end
    end
    if _wanted("tabulated k-smearing")
        @safetestset "tabulated k-smearing" begin include("instrument/test_tabulated_k_smearing.jl") end
    end
    if _wanted("gain in fit")
        @safetestset "gain in fit" begin include("instrument/test_gain_in_fit.jl") end
    end
    if _wanted("start file")
        @safetestset "start file" begin include("data/test_start.jl") end
    end
    if _wanted("held series")
        @safetestset "held series" begin include("inversion/test_held_series.jl") end
    end
    if _wanted("per-bin amplitude")
        @safetestset "per-bin amplitude" begin include("instrument/test_per_bin_amplitude.jl") end
    end
    if _wanted("sequential staging")
        @safetestset "sequential staging" begin include("config/test_deck_sequence.jl") end
    end
    if _wanted("noise")
        @safetestset "noise" begin include("instrument/test_noise.jl") end
    end
    if _wanted("examples")
        @safetestset "examples" begin include("instrument/test_examples.jl") end
    end
    if _wanted("reproduce")
        @safetestset "reproduce" begin include("inversion/test_reproduce.jl") end
    end
    if _wanted("regularizer piecewise")
        @safetestset "regularizer piecewise" begin include("inversion/test_regularizer_piecewise.jl") end
    end
end

# The slow tier (conventions.md 8.6). test/slow/ carries its OWN environment, because its files
# need CairoMakie, whose precompile has no place in the fast suite's (OQ-15). Nothing in the tier
# runs in this environment, so the gate prints how to run it rather than including it.
if get(ENV, "OTIS_SLOW_TESTS", "") != ""
    @info "OTIS_SLOW_TESTS is set. The slow tier has its own environment; run it with: " *
          "julia --project=test/slow test/slow/runslow.jl"
end
