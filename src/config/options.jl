# Config option structs, D32 (conventions.md 7.1): one Base.@kwdef option struct per
# deck-nameable component, field names the deck's own keys, defaults the single source of truth
# for that component's deck-facing configuration. Step 3.6a, chunk 11 of Phase 3.
#
# LBFGS, LevenbergMarquardt and NoisyGradientDescent (src/inversion/optimizers.jl) and
# StreakCameraNoise (src/instrument/noise.jl) are each ALREADY a Base.@kwdef struct with plain,
# concrete field types and deck-sensible defaults pinned by their own spec items (D27-D29, D21) --
# duplicating them here would create a second place their defaults could drift out of sync, so
# registry.jl's OPTIMIZER_REGISTRY and NOISE_REGISTRY point at them directly instead of at a
# wrapper defined in this file. Every other deck-nameable component either takes unit-suffixed
# keyword arguments (spec D4's boundary convention), takes a positional argument, or carries
# cached/derived fields that are not deck keys, so each gets its own option struct below.
#
# D9's numeric-key rule reads a key's target type with `fieldtype` on these structs: a `<:Real`
# field, or an element of one typed `NTuple{N,<:Real}`/`AbstractVector{<:Real}`, accepts a
# variable reference; `Symbol`, `String` and `Bool` never do. `PolynomialBackgroundOptions` and
# `RationalBackgroundOptions` therefore hold `Vector{Float64}`, not `NTuple`, so a deck list
# literal (`coefficients = ["bg", "c1", 0.0]`) has one fixed field type to check element by
# element, each entry independently a variable reference or a fixed number.

# --- SHAPE_REGISTRY's two option structs ---

"""
    MaxwellianOptions

No fields: `Maxwellian` is a non-parametric singleton (`conventions.md` Section 4). Exists so
`SHAPE_REGISTRY`'s `"Maxwellian"` entry has an option struct on the same footing as every other
shape (D32).
"""
Base.@kwdef struct MaxwellianOptions end

"""
    SuperGaussianOptions

`m`, the super-Gaussian exponent (spec D3, D4). `SuperGaussian(m::Real)`'s own constructor has no
default (`m` is a required positional argument); `2.0` is the deck's default, the isotropic 3-D
super-Gaussian's Maxwellian-equivalent exponent (`docs/derivations/supergaussian.md`).
"""
Base.@kwdef struct SuperGaussianOptions
    m::Float64 = 2.0
end

# --- EFFECT_REGISTRY's ten option structs ---

"""
    KSmearingOptions

Deck keys for `KSmearing` (spec D10, D11). `KSmearing`'s own constructor requires `half_angle`;
`half_angle_deg = 0.0` is the deck's default, giving the identity to five decimal places absent
an explicit width. `nodes` must be one of `1, 3, 5, 7` (checked by `KSmearing` itself, not here).
"""
Base.@kwdef struct KSmearingOptions
    half_angle_deg::Float64 = 0.0
    nodes::Int = 5
end

"""
    GradientSmearingOptions

Deck keys for `GradientSmearing` (spec D10, D11, A17). `leaf` is a boundary key (`""` is not a
valid leaf and must be overridden by the deck); `width` is in that leaf's own unit. The template
`Plasma` `GradientSmearing`'s real constructor additionally needs is not a deck key: it comes
from the deck's already-parsed `[plasma]` section (chunk 12), not from this option struct.
"""
Base.@kwdef struct GradientSmearingOptions
    leaf::String = ""
    width::Float64 = 0.0
    nodes::Int = 5
end

"""
    WavelengthCalibrationOptions

The deck keys for `WavelengthCalibration` (spec D3), replacing the removed probe-wavelength-shift
effect's own option struct at Step 4.A.1a (2026-09-22).
"""
Base.@kwdef struct WavelengthCalibrationOptions
    lambda_offset_nm::Float64 = 0.0
    dispersion_scale::Float64 = 0.0
end

"""
    SpectralIRFOptions
    TemporalIRFOptions

The deck key for the Gaussian-kernel form of each post-kernel IRF (spec D12, D13). The
positional, arbitrary-`AbstractIRFKernel` form (a `TabulatedKernel`) is a script-deck construct
(`conventions.md` 7.4), not this option struct's concern.
"""
Base.@kwdef struct SpectralIRFOptions
    fwhm_nm::Float64 = 0.0
end
Base.@kwdef struct TemporalIRFOptions
    fwhm_ns::Float64 = 0.0
end

"""
    SensitivityOptions

The deck key for the scalar half of `Sensitivity` (spec D40, D42). The measured response curve
(`lambda`/`response`) is data, not a scalar deck key, and is a script-deck construct.
"""
Base.@kwdef struct SensitivityOptions
    scale::Float64 = 1.0
end

"""
    ConstantBackgroundOptions

The deck key for `ConstantBackground` (spec D14).
"""
Base.@kwdef struct ConstantBackgroundOptions
    level::Float64 = 0.0
end

"""
    PolynomialBackgroundOptions

The deck key for `PolynomialBackground` (spec D14). `coefficients` is `Vector{Float64}`, not
`NTuple`, per this file's front matter.
"""
Base.@kwdef struct PolynomialBackgroundOptions
    coefficients::Vector{Float64} = Float64[]
end

"""
    ImageBackgroundOptions

The deck key for the scalar half of `ImageBackground` (spec D14). `image` is data, not a scalar
deck key, and is a script-deck construct.
"""
Base.@kwdef struct ImageBackgroundOptions
    scale::Float64 = 1.0
end

"""
    RationalBackgroundOptions

The deck keys for `RationalBackground` (spec D46). `numerator`/`denominator` are `Vector{Float64}`,
not `NTuple`, per this file's front matter; Pitfall 17's bound-guarded deck examples
(`rat11`/`rat21`) set these through the deck's `[fit.variables]`, not through the defaults here.
"""
Base.@kwdef struct RationalBackgroundOptions
    numerator::Vector{Float64} = Float64[]
    denominator::Vector{Float64} = Float64[]
end

# --- KERNEL_REGISTRY's two option structs ---

"""
    GaussianKernelOptions

The deck key for `GaussianKernel` (spec D12), used where an IRF's kernel type is named
explicitly rather than built through `SpectralIRF`/`TemporalIRF`'s own `fwhm` keyword.
"""
Base.@kwdef struct GaussianKernelOptions
    sigma::Float64 = 0.0
end

"""
    TabulatedKernelOptions

The deck keys for `TabulatedKernel` (spec D12). `offsets` is `Vector{Int}`, never a variable
reference (an offset is a bin index, not a fittable physical quantity); `weights` is
`Vector{Float64}`, each entry independently numeric (D9).
"""
Base.@kwdef struct TabulatedKernelOptions
    offsets::Vector{Int} = Int[]
    weights::Vector{Float64} = Float64[]
end

# --- NOISE_REGISTRY's one option struct (StreakCameraNoise needs none, see front matter) ---

"""
    GaussianNoiseOptions

The deck keys for `GaussianNoise` (spec D21), mirroring its own keyword constructor's defaults
(`GaussianNoise(; a = 1.0, b = 0.0)`) — kept here, rather than reused directly, because
`GaussianNoise` is not itself a `Base.@kwdef` struct (its inner constructor validates `a > 0`).
Change one default, change the other.
"""
Base.@kwdef struct GaussianNoiseOptions
    a::Float64 = 1.0
    b::Float64 = 0.0
end
