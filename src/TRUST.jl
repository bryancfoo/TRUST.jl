module TRUST

include("constants.jl")
include("plasma_dispersion.jl")
include("distributions/distribution.jl")
include("distributions/maxwellian.jl")
include("distributions/chebyshev_table.jl")       # new (4.B.2a): the general (xi, p) Chebyshev table machinery
include("distributions/supergaussian.jl")         # new (1.SG.1)
include("distributions/supergaussian_table.jl")   # new (1.SG.2): builds SG_TABLE at include time
include("distributions/kappa.jl")                 # new (4.B.2b): Kappa and KAPPA_TABLE, built at include time
include("distributions/generalized_normal.jl")     # new (4.B.3): GeneralizedNormal and GN_TABLE, built at include time
include("distributions/skew_normal.jl")            # new (4.B.4): SkewNormal and SN_TABLE, built at include time
include("distributions/hermite.jl")                # new (5.A.1): HermiteExpansion, closed form through Z's derivatives
include("plasma/species.jl")
include("plasma/parameters.jl")     # methods on Species and SpeciesSpec
include("plasma/plasma.jl")         # Plasma constructor uses Species and the unit table
include("plasma/trace.jl")          # new (2.1): needs Plasma, Species, reconstruct
include("scattering/susceptibility.jl")   # new: needs alpha, xi, cauchy_dfdv
include("scattering/spectral_density.jl") # new: needs susceptibility, dielectric, f_lab
include("scattering/geometry.jl")         # new: needs the unit table and Plasma
include("scattering/spectrum.jl")         # new: needs wavevector and spectral_density
include("instrument/effects.jl")          # new (2.2): needs ProbeBeam, Collection, Plasma, LeafRef
include("instrument/background.jl")       # new (2.2)
include("instrument/experiment.jl")       # new (2.2): needs the effects and PlasmaTrace
include("instrument/measure.jl")          # new (2.2/2.3): needs Experiment and spectrum!
include("instrument/resonance.jl")        # new (4.F.2): the EPW resonance's width; needs PixelIntegration, dielectric
include("data/streak.jl")                 # new (2.4)
include("data/io.jl")                     # new (2.4): needs StreakData and PlasmaTrace
include("data/warp.jl")                   # new (3.8): WarpMap, dewarp, read_warpmap; needs StreakData
include("data/average.jl")                # new (4.A.2): average_regions, AveragingRegion; needs StreakData
include("instrument/noise.jl")            # new (2.5): needs Experiment and StreakData
include("inversion/expressions.jl")       # new (3.1b): the dependent-expression layer; no exports yet
include("inversion/parameters.jl")        # new (3.1c): ParameterSpec, the transform, the index map
include("inversion/residuals.jl")         # new (3.2a): FitProblem, residuals, residuals!, chi2
include("inversion/jacobian.jl")          # new (3.2b): FitJacobian, jacobian, jvp!, jtvp!, gradient!
include("inversion/regularizer.jl")       # new (3.3): AbstractRegularizerTerm, RegularizerTerm, Regularizer
include("inversion/objective.jl")         # new (3.4a): objective, FitResult
include("inversion/optimizers.jl")        # new (3.4b): AbstractOptimizer, the four optimizers, solve
include("inversion/weight_scan.jl")       # new (3.5): weight_scan, WeightScanResult
include("inversion/lcurve_corner.jl")     # new (4.C.1): lcurve_corner, LCurveCorner; needs WeightScanResult
include("inversion/uncertainty.jl")       # new (4.A.4a): normal_matrices, gn_decrease, polish, laplace, calibration_scan
include("config/options.jl")              # new (3.6a): option structs, spec D32
include("config/registry.jl")             # new (3.6a): component registries, spec D32
include("data/result.jl")                 # new (3.6a): write_result/read_result, spec D34
include("data/start.jl")                  # new (4.I.1): write_start/read_start, the trust-start-1 layout
include("data/readable.jl")               # new (4.M.1): write_readable, the trust-readable-1 layout
include("config/deck.jl")                 # new (3.6b): read_deck, build_problem, effective_deck, fit
include("config/scatter.jl")              # new (4.A.4b): slice_scatter; needs Deck and D50's stages

# types
export AbstractVelocityDistribution, Maxwellian, SuperGaussian, Kappa, GeneralizedNormal, SkewNormal, HermiteExpansion,
       SpeciesSpec, Species, SpeciesSet, Plasma

# shape interface
export dfdx, cauchy_dfdv, moment, skewness, excess_kurtosis, parameters, reconstruct

# plasma dispersion
export Z, Zprime

# species builders and accessors
export electrons, ion, spec
export thermal_speed, plasma_frequency, debye_length, alpha, xi, f_lab

# plasma quantities
export mean_charge_state, mean_ion_mass, neutrality_residual, zeff

# trace (2.1)
export PlasmaTrace, LeafPath, plasma, nbins

# scattering
export susceptibility, dielectric, spectral_density, spectrum, spectrum!

# geometry (the single-struct geometry type retired at Step 2.3a, spec D40)
export ProbeBeam, Collection, wavevector, polarization_factor, kdirection, detector_axis

# instrument (2.2, 2.3, 2.3a, 2.3b)
export AbstractInstrumentEffect, SpectralIRF, TemporalIRF, KSmearing, GradientSmearing,
       WavelengthCalibration, Sensitivity, ConstantBackground, PolynomialBackground,
       ImageBackground, RationalBackground, GaussianKernel, TabulatedKernel, PixelIntegration,
       TabulatedKSmearing
export Experiment, measure, ideal_streak, ideal_streak!, islinear, stage, samples, apply!
export foreach_bin!, pin_blas_threads

# data (2.4)
export StreakData, read_streak, write_streak, read_trace, write_trace

# data (3.8)
export WarpMap, dewarp, read_warpmap

# data (4.A.2)
export average_regions, AveragingRegion

# data (3.6a)
export write_result, read_result

# data (4.I.1): the start file
export write_start, read_start

# data (4.M.1): the readable companion of a result file
export write_readable

# config (3.6b): build_problem, Deck and DECK_VERSION are deliberately not exported (the chunk 12
# prompt names exactly these three); tests reach the others through TRUST.build_problem etc.
export read_deck, effective_deck, fit

# noise (2.5)
export AbstractNoiseModel, StreakCameraNoise, GaussianNoise, sigma, synthesize

# inversion (3.1)
export FittedVariable, FitInterval, ParameterSpec, flatten, unflatten

# inversion (3.2a)
export FitProblem, residuals, residuals!, chi2

# inversion (3.2b)
export jacobian, gradient!

# inversion (3.3)
export AbstractRegularizerTerm, RegularizerTerm, Regularizer

# inversion (3.4a)
export objective, FitResult

# inversion (4.A.3a): the multi-channel problem
export AbstractFitProblem, JointFitProblem, ChannelSummary

# inversion (3.4b)
export AbstractOptimizer, LBFGS, LevenbergMarquardt, NoisyGradientDescent, OptimizerSequence, solve

# inversion (3.5)
export weight_scan, WeightScanResult

# inversion (4.C.1)
export lcurve_corner, LCurveCorner

# inversion (4.A.4a): the uncertainty reports
export polish, gn_decrease, normal_matrices, laplace, LaplaceResult, calibration_scan,
       CalibrationScanResult

# config (4.A.4b): the per-slice scatter
export slice_scatter, ScatterResult

# plotting stubs (2.4 declares them, 2.6 gives Makie methods)
export streakplot, streakplot!, spectrumplot, spectrumplot!,
       traceplot, traceplot!, lcurveplot, lcurveplot!

# constants
export ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY, SPEED_OF_LIGHT,
       ATOMIC_MASS_UNIT, EV

# boundary conversions
export T_to_si, T_from_si, n_to_si, n_from_si, u_to_si, u_from_si,
       mass_to_si, mass_from_si, lambda_to_si, lambda_from_si,
       t_to_si, t_from_si, theta_to_si, theta_from_si

end
