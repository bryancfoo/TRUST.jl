# The component registries, D32 (conventions.md 7.2): one Dict{String,Any} per pluggable
# category, mapping a deck name to the constructor (or, for the two selector registries, the
# Symbol) that name means. Step 3.6a, chunk 11 of Phase 3.
#
# The deck layer (Step 3.6b) looks a name up here and never contains a hand-written
# `if name == ...`/`elseif name == ...` chain (conventions.md 7.2, DK10); an unknown name is the
# deck layer's own loud error naming the section, the key and the valid names (D39), not this
# file's concern. This file only says what each name means.

"""
    SHAPE_REGISTRY

Deck name -> `AbstractVelocityDistribution` constructor (spec D32). Each entry's option struct
is in `options.jl` (`MaxwellianOptions`, `SuperGaussianOptions`, `KappaOptions`,
`GeneralizedNormalOptions`, `SkewNormalOptions`).
"""
const SHAPE_REGISTRY = Dict{String,Any}(
    "Maxwellian" => Maxwellian,
    "SuperGaussian" => SuperGaussian,
    "Kappa" => Kappa,
    "GeneralizedNormal" => GeneralizedNormal,
    "SkewNormal" => SkewNormal,
)

"""
    EFFECT_REGISTRY

Deck name -> `AbstractInstrumentEffect` constructor (spec D32). `"Constant"`, `"Polynomial"`,
`"Image"` and `"Rational"` are the four backgrounds (spec D14, D46); the rest are the effects of
spec D9-D13. Each entry's option struct is in `options.jl`.
"""
const EFFECT_REGISTRY = Dict{String,Any}(
    "SpectralIRF" => SpectralIRF,
    "TemporalIRF" => TemporalIRF,
    "KSmearing" => KSmearing,
    "GradientSmearing" => GradientSmearing,
    "WavelengthCalibration" => WavelengthCalibration,
    "Sensitivity" => Sensitivity,
    "Constant" => ConstantBackground,
    "Polynomial" => PolynomialBackground,
    "Image" => ImageBackground,
    "Rational" => RationalBackground,
)

"""
    KERNEL_REGISTRY

Deck name -> `AbstractIRFKernel` constructor (spec D12, D32). Each entry's option struct is in
`options.jl` (`GaussianKernelOptions`, `TabulatedKernelOptions`).
"""
const KERNEL_REGISTRY = Dict{String,Any}(
    "Gaussian" => GaussianKernel,
    "Tabulated" => TabulatedKernel,
)

"""
    OPERATOR_REGISTRY

Deck name -> the `RegularizerTerm` `operator` Symbol it selects (spec D23, D32). No option
struct: an operator is a fixed choice, not a constructed component.
"""
const OPERATOR_REGISTRY = Dict{String,Symbol}(
    "D2" => :D2,
    "D1" => :D1,
    "I" => :I,
)

"""
    NORMALIZATION_REGISTRY

Deck name -> the `RegularizerTerm` `normalization` Symbol it selects (spec D24, D32). No option
struct, for the same reason as `OPERATOR_REGISTRY`.
"""
const NORMALIZATION_REGISTRY = Dict{String,Symbol}(
    "absolute" => :absolute,
    "initial" => :initial,
    "relative" => :relative,
    "unit" => :unit,
)

"""
    OPTIMIZER_REGISTRY

Deck name -> `AbstractOptimizer` constructor (spec D27-D30, D32). Each entry is already its own
`Base.@kwdef` option struct (`src/inversion/optimizers.jl`); see `options.jl`'s front matter for
why none is duplicated here.
"""
const OPTIMIZER_REGISTRY = Dict{String,Any}(
    "lbfgs" => LBFGS,
    "levenberg_marquardt" => LevenbergMarquardt,
    "noisy_descent" => NoisyGradientDescent,
)

"""
    NOISE_REGISTRY

Deck name -> `AbstractNoiseModel` constructor (spec D21, D32). `"StreakCamera"` is already its
own `Base.@kwdef` option struct (`src/instrument/noise.jl`); `"Gaussian"`'s is
`GaussianNoiseOptions` (`options.jl`).
"""
const NOISE_REGISTRY = Dict{String,Any}(
    "StreakCamera" => StreakCameraNoise,
    "Gaussian" => GaussianNoise,
)
