# AbstractInstrumentEffect and its concrete effects: the three in-kernel effects (KSmearing,
# GradientSmearing, ProbeOffset), the two post-kernel IRF convolutions and Gain, and the
# convolution-matrix builder they and the noise model's point spread share.
# docs/derivations/instrument.md; conventions.md Section 4; spec D9-D13.

using Printf: @printf

"""
    AbstractInstrumentEffect

The abstract type of one link in the instrument chain. Every concrete effect implements
`stage`, `islinear`, `parameters`, `reconstruct` and `_chain_order`, and, according to its
stage, either `samples` (`:kernel`) or `apply!` (`:detector`). Nothing in the package dispatches
on a concrete effect type with `isa` (`conventions.md` Section 4).

Spec D9.
"""
abstract type AbstractInstrumentEffect end

"""
    stage(effect::AbstractInstrumentEffect) -> Symbol

`:kernel` for an effect evaluated inside the per-bin kernel (it changes what physics is
evaluated), `:detector` for one applied to the streak after the per-bin loop.

Spec D9.
"""
function stage end

"""
    islinear(effect::AbstractInstrumentEffect) -> Bool

`true` only when `effect`'s action on the streak is a fixed linear operator
(`effect(aP+bQ) = a·effect(P) + b·effect(Q)` to round-off): the two IRFs and `Gain`. The three
`:kernel` effects act on the plasma, not the streak, so linearity is not even a meaningful
question for them, and `islinear` is `false`; the backgrounds are affine, not linear, and are
also `false`. `islinear` answers exactly the question Step 3.2 asks: may this effect be pushed
through the Jacobian as a fixed operator?

Spec D9.
"""
function islinear end

"""
    _chain_order(effect::AbstractInstrumentEffect) -> Int

The effect's canonical position in the chain, `1` (kernel) to `5` (backgrounds); `Experiment`'s
constructor stably sorts effects by this so the chain always runs kernel -> `SpectralIRF` ->
`TemporalIRF` -> `Gain` -> backgrounds ([OQ-13](../../dev/open_questions.md#oq-13)).

Spec D9, D15.
"""
function _chain_order end

"""
    samples(effect::AbstractInstrumentEffect, plasma::Plasma, probe::ProbeBeam, collection::Collection) -> NTuple

For a `:kernel` effect: the weighted list of `(weight, plasma, probe, collection)` samples the
per-bin kernel folds over; each in-kernel effect rebuilds only the one of `plasma`, `probe` or
`collection` it modifies (`KSmearing` a `Collection`, `ProbeOffset` a `ProbeBeam`,
`GradientSmearing` a `Plasma`). Spec D9, D10, D40.
"""
function samples end

"""
    apply!(dest::AbstractMatrix, src::AbstractMatrix, effect::AbstractInstrumentEffect, operator)

For a `:detector` effect: writes `effect`'s action on `src` into `dest` (`dest === src` is
allowed for every effect but the two IRFs, which need a separate destination). `operator` is
the value `_prepare` built for this effect at `Experiment` construction, or `nothing`.

Spec D9, D16.
"""
function apply! end

"""
    _prepare(effect::AbstractInstrumentEffect, lambda::AbstractVector, t::AbstractVector)

Built once per effect by the `Experiment` constructor: the convolution matrix for an IRF, the
background vector for `PolynomialBackground`, `nothing` for everything else.

Spec D15.
"""
_prepare(::AbstractInstrumentEffect, lambda, t) = nothing

"""
    _check_nodes(nodes::Integer) -> Integer

Validates that `nodes` is one of `1, 3, 5, 7` (spec D11, D28), returning it unchanged.
"""
function _check_nodes(nodes::Integer)
    nodes in (1, 3, 5, 7) || throw(ArgumentError("nodes must be 1, 3, 5 or 7; got $nodes"))
    return nodes
end

# --- D11: the Gauss-Legendre and Gauss-Hermite node/weight tables, computed by Golub-Welsch
# and checked (sum(w) == 1, sum(w*xi^2) == 1/3 or 1/2) to better than 1.3e-15 (Level 2,
# 2026-09-09). Node order is increasing in xi, negative first, so the fold's summation order is
# fixed and reproducible bit for bit.

const GAUSS_LEGENDRE_NODES = (
    n1=(0.0,),
    n3=(-0.7745966692414834, 0.0, 0.7745966692414834),
    n5=(-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640),
    n7=(-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
        0.4058451513773972, 0.7415311855993945, 0.9491079123427585),
)
const GAUSS_LEGENDRE_WEIGHTS = (
    n1=(1.0,),
    n3=(0.2777777777777778, 0.4444444444444444, 0.2777777777777778),
    n5=(0.1184634425280945, 0.2393143352496832, 0.2844444444444444, 0.2393143352496832, 0.1184634425280945),
    n7=(0.0647424830844348, 0.1398526957446383, 0.1909150252525595, 0.2089795918367347,
        0.1909150252525595, 0.1398526957446383, 0.0647424830844348),
)
const GAUSS_HERMITE_NODES = (
    n1=(0.0,),
    n3=(-1.2247448713915890, 0.0, 1.2247448713915890),
    n5=(-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856),
    n7=(-2.6519613568352334, -1.6735516287674714, -0.8162878828589647, 0.0,
        0.8162878828589647, 1.6735516287674714, 2.6519613568352334),
)
const GAUSS_HERMITE_WEIGHTS = (
    n1=(1.0,),
    n3=(0.1666666666666667, 0.6666666666666666, 0.1666666666666667),
    n5=(0.0112574113277207, 0.2220759220056127, 0.5333333333333333, 0.2220759220056127, 0.0112574113277207),
    n7=(0.0005482688559722, 0.0307571239675864, 0.2401231786050128, 0.4571428571428571,
        0.2401231786050128, 0.0307571239675864, 0.0005482688559722),
)

function _gauss_legendre(nodes::Integer)
    _check_nodes(nodes)
    nodes == 1 && return GAUSS_LEGENDRE_NODES.n1, GAUSS_LEGENDRE_WEIGHTS.n1
    nodes == 3 && return GAUSS_LEGENDRE_NODES.n3, GAUSS_LEGENDRE_WEIGHTS.n3
    nodes == 5 && return GAUSS_LEGENDRE_NODES.n5, GAUSS_LEGENDRE_WEIGHTS.n5
    return GAUSS_LEGENDRE_NODES.n7, GAUSS_LEGENDRE_WEIGHTS.n7
end

function _gauss_hermite(nodes::Integer)
    _check_nodes(nodes)
    nodes == 1 && return GAUSS_HERMITE_NODES.n1, GAUSS_HERMITE_WEIGHTS.n1
    nodes == 3 && return GAUSS_HERMITE_NODES.n3, GAUSS_HERMITE_WEIGHTS.n3
    nodes == 5 && return GAUSS_HERMITE_NODES.n5, GAUSS_HERMITE_WEIGHTS.n5
    return GAUSS_HERMITE_NODES.n7, GAUSS_HERMITE_WEIGHTS.n7
end

function _legendre_offsets_weights(half_angle::Real, nodes::Integer)
    xi, w = _gauss_legendre(nodes)
    FT = float(typeof(half_angle))
    offsets = ntuple(q -> FT(half_angle) * FT(xi[q]), Val(length(xi)))
    wsum = FT(sum(w))
    weights = ntuple(q -> FT(w[q]) / wsum, Val(length(w)))
    return offsets, weights
end

function _hermite_offsets_weights(width::Real, nodes::Integer)
    xi, w = _gauss_hermite(nodes)
    FT = float(typeof(width))
    sqrt2 = sqrt(FT(2))
    offsets = ntuple(q -> FT(width) * sqrt2 * FT(xi[q]), Val(length(xi)))
    wsum = FT(sum(w))
    weights = ntuple(q -> FT(w[q]) / wsum, Val(length(w)))
    return offsets, weights
end

# --- D10: the three in-kernel effects ---

"""
    KSmearing{NQ,FT<:Real} <: AbstractInstrumentEffect

A weighted average of the ideal spectrum over a uniform spread of collection angles of
half-width `half_angle` (rad): `θ_q = θ + half_angle·ξ_q` on Gauss-Legendre nodes `ξ_q` on
`[-1, 1]`, weights normalized to sum to one.

Built with `KSmearing(; half_angle_deg | half_angle_rad, nodes::Integer = 5)`; `nodes` is one
of `1, 3, 5, 7` and `nodes = 1` is the identity, exactly.

Spec D10, D11.
"""
struct KSmearing{NQ,FT<:Real} <: AbstractInstrumentEffect
    half_angle::FT
    offsets::NTuple{NQ,FT}
    weights::NTuple{NQ,FT}
end

function KSmearing(; nodes::Integer=5, kwargs...)
    half_angle_raw = _si_value(:half_angle, kwargs)
    _reject_extra_kwargs(kwargs, (:half_angle_deg, :half_angle_rad))
    (isfinite(half_angle_raw) && half_angle_raw >= 0) || throw(ArgumentError(
        "KSmearing: half angle must be finite and non-negative; got $half_angle_raw rad"))
    _check_nodes(nodes)
    offsets, weights = _legendre_offsets_weights(half_angle_raw, nodes)
    FT = eltype(offsets)
    return KSmearing{length(offsets),FT}(FT(half_angle_raw), offsets, weights)
end

stage(::KSmearing) = :kernel
_chain_order(::KSmearing) = 1
islinear(::KSmearing) = false
parameters(e::KSmearing) = (half_angle=e.half_angle,)

function reconstruct(::KSmearing{NQ}, values::NamedTuple) where {NQ}
    half_angle = values.half_angle
    (isfinite(half_angle) && half_angle >= 0) || throw(ArgumentError(
        "KSmearing: half angle must be finite and non-negative; got $half_angle rad"))
    offsets, weights = _legendre_offsets_weights(half_angle, NQ)
    FT = eltype(offsets)
    return KSmearing{NQ,FT}(FT(half_angle), offsets, weights)
end

"""
    samples(e::KSmearing, p::Plasma, probe::ProbeBeam, collection::Collection) -> NTuple

`θ -> θ + e.offsets[q]`; the plasma and the probe are untouched. Only `Collection` is rebuilt,
through its **inner** constructor `Collection{FT}`, never the keyword constructor, so this is
allocation-free (EF9; Level 2 measured 0 bytes over 1000 samples for the whole geometry before
the split, and the split makes the rebuilt struct smaller still, spec D40).

Spec D10, D40, Pitfall 10.
"""
function samples(e::KSmearing{NQ,FTe}, p::Plasma, probe::ProbeBeam, collection::Collection{FTc}) where {NQ,FTe,FTc}
    FTg = promote_type(FTc, FTe)
    return ntuple(Val(NQ)) do q
        (e.weights[q], p, probe,
            Collection{FTg}(FTg(collection.theta) + FTg(e.offsets[q]), collection.k_model,
                FTg(collection.lambda_center), FTg(collection.dispersion), collection.npixels,
                FTg(collection.reference_pixel)))
    end
end

"""
    GradientSmearing{NQ,FT<:Real} <: AbstractInstrumentEffect

A weighted average of the ideal spectrum over a normal distribution of one plasma leaf inside
the scattering volume: additive for a species' `T` or `u` (`x_q = x + width·√2·ξ_q`), or
multiplicative for `n_e` (`x_q = x·exp(width·√2·ξ_q)`, so the density stays positive for any
width), on Gauss-Hermite nodes `ξ_q`, weights normalized to sum to one.

Built with `GradientSmearing(template::Plasma; leaf::AbstractString, width::Real, nodes::Integer
= 5)`; `leaf` is a **boundary key** (`"electrons.e.T_eV"`, `"ions.D.u_kms"`, or the bare SI path
`"electrons.e.T"`), resolved through `_leaf_si_path` exactly as `PlasmaTrace`'s `leaves` are, and
`width` is in that key's own unit (eV for `T_eV`, km/s for `u_kms`, joules for a bare `T`) —
**corrected 2026-09-11, item A17**: `leaf` previously had to be an SI path and `width` had to be
SI (joules), the one public keyword in Phase 2 carrying a temperature in a unit no other keyword
uses. The density is the one exception: `leaf = "n_e"` (never `"n_e_cm3"`/`"n_e_m3"`, refused
loudly) takes a **dimensionless** (log) `width`, since it is the σ of the multiplicative
lognormal offset below, not a unit-bearing quantity. `fraction` and a shape parameter are refused
(offsetting a fraction breaks the sum-to-one closure with no repair, and a shape parameter's
`reconstruct` per sample per bin is a cost this phase does not pay).

Spec D10, D11, D28, A17.
"""
struct GradientSmearing{NQ,FT<:Real} <: AbstractInstrumentEffect
    ref::LeafRef
    path::LeafPath
    width::FT
    logarithmic::Bool
    offsets::NTuple{NQ,FT}
    weights::NTuple{NQ,FT}
end

function GradientSmearing(template::Plasma; leaf::AbstractString, width::Real, nodes::Integer=5)
    if leaf in ("n_e_cm3", "n_e_m3")
        throw(ArgumentError(
            "GradientSmearing: smear the density as \"n_e\" with a dimensionless (log) width, not \"$leaf\""))
    end
    (isfinite(width) && width >= 0) || throw(ArgumentError(
        "GradientSmearing: width must be finite and non-negative; got $width"))
    _check_nodes(nodes)
    si_path, factor = _leaf_si_path(leaf)
    path = LeafPath(si_path)
    ref = _resolve_leaf(template, path)
    ref.field === :fraction && throw(ArgumentError(
        "GradientSmearing: a fraction cannot be gradient-smeared"))
    ref.shape && throw(ArgumentError(
        "GradientSmearing: a shape parameter cannot be gradient-smeared"))
    logarithmic = ref.set === :n_e
    width_si = width * factor
    offsets, weights = _hermite_offsets_weights(width_si, nodes)
    FT = eltype(offsets)
    return GradientSmearing{length(offsets),FT}(ref, path, FT(width_si), logarithmic, offsets, weights)
end

stage(::GradientSmearing) = :kernel
_chain_order(::GradientSmearing) = 1
islinear(::GradientSmearing) = false
parameters(e::GradientSmearing) = (width=e.width,)

function reconstruct(e::GradientSmearing{NQ}, values::NamedTuple) where {NQ}
    width = values.width
    (isfinite(width) && width >= 0) || throw(ArgumentError(
        "GradientSmearing: width must be finite and non-negative; got $width"))
    offsets, weights = _hermite_offsets_weights(width, NQ)
    FT = eltype(offsets)
    return GradientSmearing{NQ,FT}(e.ref, e.path, FT(width), e.logarithmic, offsets, weights)
end

_offset_value(e::GradientSmearing, x, q) = e.logarithmic ? x * exp(e.offsets[q]) : x + e.offsets[q]

"""
    samples(e::GradientSmearing, p::Plasma, probe::ProbeBeam, collection::Collection) -> NTuple

One leaf of `p` offset per sample (`_with_leaf`, allocation-free by the same argument as
`plasma(trace, j)`); the probe and the collection are untouched.

Spec D10, D40, Pitfall 12.
"""
function samples(e::GradientSmearing{NQ}, p::Plasma, probe::ProbeBeam, collection::Collection) where {NQ}
    return ntuple(q -> (e.weights[q],
            _with_leaf(p, e.ref, _offset_value(e, _leaf_value(p, e.ref), q)), probe, collection),
        Val(NQ))
end

"""
    ProbeOffset{FT<:Real} <: AbstractInstrumentEffect

A shift `lambda_probe -> lambda_probe + offset` of the probe wavelength — **not** of the
detector wavelength axis. It enters `wavevector` and `_omega_probe` through `probe.lambda`;
`expt.lambda` is untouched (EF12, Pitfall 11).

Built with `ProbeOffset(; probe_offset_nm | probe_offset_m)`.

Spec D10, D40.
"""
struct ProbeOffset{FT<:Real} <: AbstractInstrumentEffect
    offset::FT
end

function ProbeOffset(; kwargs...)
    offset_raw = _si_value(:probe_offset, kwargs)
    _reject_extra_kwargs(kwargs, (:probe_offset_nm, :probe_offset_m))
    FT = float(typeof(offset_raw))
    return ProbeOffset{FT}(FT(offset_raw))
end

stage(::ProbeOffset) = :kernel
_chain_order(::ProbeOffset) = 1
islinear(::ProbeOffset) = false
parameters(e::ProbeOffset) = (offset=e.offset,)
reconstruct(::ProbeOffset, values::NamedTuple) = ProbeOffset(; probe_offset_m=values.offset)

"""
    samples(e::ProbeOffset, p::Plasma, probe::ProbeBeam, collection::Collection) -> NTuple

`lambda_probe -> lambda_probe + e.offset`; the plasma and the collection are untouched. One
sample, weight one. Only `ProbeBeam` is rebuilt, through its **inner** constructor.

Spec D10, D40.
"""
function samples(e::ProbeOffset, p::Plasma, probe::ProbeBeam{FTp0}, collection::Collection) where {FTp0}
    FTp = promote_type(FTp0, typeof(e.offset))
    return ((one(FTp), p,
        ProbeBeam{FTp}(FTp(probe.lambda) + FTp(e.offset), FTp(probe.polarization), FTp(probe.a0_sq),
            FTp(probe.diameter)),
        collection),)
end

# --- D12: the post-kernel IRFs and Gain ---

"""
    AbstractIRFKernel

The shape of a `SpectralIRF`/`TemporalIRF`'s point-spread: `GaussianKernel` (built from a FWHM)
or `TabulatedKernel` (arbitrary tap offsets and weights).

Spec D12.
"""
abstract type AbstractIRFKernel end

"""
    GaussianKernel{FT<:Real} <: AbstractIRFKernel

A Gaussian of standard deviation `sigma`, in the axis's own SI unit (m on `lambda`, s on `t`).
`sigma` must be strictly positive: **an experiment with no IRF omits the effect rather than
using `sigma = 0` as the identity.**
"""
struct GaussianKernel{FT<:Real} <: AbstractIRFKernel
    sigma::FT
    function GaussianKernel{FT}(sigma::FT) where {FT<:Real}
        (isfinite(sigma) && sigma > 0) || throw(ArgumentError(
            "GaussianKernel: sigma must be strictly positive; got $sigma"))
        return new{FT}(sigma)
    end
end
GaussianKernel(sigma::Real) = GaussianKernel{typeof(sigma)}(sigma)

"""
    TabulatedKernel{N,FT<:Real} <: AbstractIRFKernel

An arbitrary point-spread given as `N` taps: `offsets` in bins relative to the source bin,
strictly increasing; `weights` as given (the convolution matrix renormalizes by column anyway).

Spec D12.
"""
struct TabulatedKernel{N,FT<:Real} <: AbstractIRFKernel
    offsets::NTuple{N,Int}
    weights::NTuple{N,FT}
    function TabulatedKernel{N,FT}(offsets::NTuple{N,Int}, weights::NTuple{N,FT}) where {N,FT<:Real}
        for k in 2:N
            offsets[k] > offsets[k-1] || throw(ArgumentError(
                "TabulatedKernel: offsets must be strictly increasing"))
        end
        return new{N,FT}(offsets, weights)
    end
end

function TabulatedKernel(offsets, weights)
    length(offsets) == length(weights) || throw(ArgumentError(
        "TabulatedKernel: offsets and weights must have the same length; got $(length(offsets)) and $(length(weights))"))
    N = length(offsets)
    FT = float(promote_type(typeof.(weights)...))
    return TabulatedKernel{N,FT}(NTuple{N,Int}(Tuple(offsets)), NTuple{N,FT}(FT.(Tuple(weights))))
end

"""
    _sigma_from_fwhm(w) -> Real

`σ = FWHM / (2√(2 ln 2))`. The one place this conversion is computed; every other use goes
through this function or its inverse, `_fwhm_from_sigma` (Pitfall 5). **Never write the decimal
expansion of `2√(2 ln 2)` as a literal anywhere else in `src/`** — EF16 greps for it.
"""
_sigma_from_fwhm(w) = w / (2 * sqrt(2 * log(oftype(float(w), 2))))

"""
    _fwhm_from_sigma(s) -> Real

The inverse of `_sigma_from_fwhm`, used by `show`.
"""
_fwhm_from_sigma(s) = s * (2 * sqrt(2 * log(oftype(float(s), 2))))

"""
    SpectralIRF{K<:AbstractIRFKernel} <: AbstractInstrumentEffect
    TemporalIRF{K<:AbstractIRFKernel} <: AbstractInstrumentEffect

A convolution of the streak with `kernel`, in wavelength (`SpectralIRF`, `M = K_λ·P`) or in time
(`TemporalIRF`, `M = P·K_tᵀ`) — applied **after** the per-bin loop, never inside the kernel, so
the per-bin Jacobian blocks of Step 3.2 stay independent ([OQ-16](../../dev/open_questions.md#oq-16),
Pitfall 2).

Built with `SpectralIRF(; fwhm_nm | fwhm_m)` / `TemporalIRF(; fwhm_ns | fwhm_s)` (a
`GaussianKernel`) or the positional `SpectralIRF(kernel::AbstractIRFKernel)` for a tabulated
kernel.

Spec D12, D13.
"""
struct SpectralIRF{K<:AbstractIRFKernel} <: AbstractInstrumentEffect
    kernel::K
end
struct TemporalIRF{K<:AbstractIRFKernel} <: AbstractInstrumentEffect
    kernel::K
end

function SpectralIRF(; kwargs...)
    fwhm_raw = _si_value(:fwhm_lambda, kwargs)
    _reject_extra_kwargs(kwargs, (:fwhm_nm, :fwhm_m))
    fwhm_raw > 0 || throw(ArgumentError("SpectralIRF: FWHM must be strictly positive; got $fwhm_raw m"))
    return SpectralIRF(GaussianKernel(_sigma_from_fwhm(fwhm_raw)))
end

function TemporalIRF(; kwargs...)
    fwhm_raw = _si_value(:fwhm_t, kwargs)
    _reject_extra_kwargs(kwargs, (:fwhm_ns, :fwhm_s))
    fwhm_raw > 0 || throw(ArgumentError("TemporalIRF: FWHM must be strictly positive; got $fwhm_raw s"))
    return TemporalIRF(GaussianKernel(_sigma_from_fwhm(fwhm_raw)))
end

stage(::SpectralIRF) = :detector
stage(::TemporalIRF) = :detector
_chain_order(::SpectralIRF) = 2
_chain_order(::TemporalIRF) = 3
islinear(::SpectralIRF) = true
islinear(::TemporalIRF) = true

parameters(e::SpectralIRF{<:GaussianKernel}) = (sigma=e.kernel.sigma,)
parameters(e::TemporalIRF{<:GaussianKernel}) = (sigma=e.kernel.sigma,)
parameters(::SpectralIRF{<:TabulatedKernel}) = NamedTuple()
parameters(::TemporalIRF{<:TabulatedKernel}) = NamedTuple()

reconstruct(::SpectralIRF{<:GaussianKernel}, values::NamedTuple) = SpectralIRF(GaussianKernel(values.sigma))
reconstruct(e::SpectralIRF{<:TabulatedKernel}, ::NamedTuple) = e
reconstruct(::TemporalIRF{<:GaussianKernel}, values::NamedTuple) = TemporalIRF(GaussianKernel(values.sigma))
reconstruct(e::TemporalIRF{<:TabulatedKernel}, ::NamedTuple) = e

_prepare(e::SpectralIRF, lambda, t) = _convolution_matrix(e.kernel, lambda)
_prepare(e::TemporalIRF, lambda, t) = _convolution_matrix(e.kernel, t)

"""
    Sensitivity{FT<:Real,L,R} <: AbstractInstrumentEffect

A fitted scalar `scale` turning `spectrum`'s m⁻¹ into detector counts, and, optionally, a fixed
measured wavelength response curve `lambda`/`response`, normalized to unit maximum at
construction (`response_norm` the divisor kept, so the supplied curve is recoverable) so that
`scale` reads as counts per m⁻¹ at the most sensitive wavelength. **The curve is fixed data and
the scalar is the fitted leaf** — the same division `ImageBackground` already uses — so the
effect still exposes exactly one leaf, `scale`. `Gain` until 2026-09-10 (`conventions.md` 1.8);
`Sensitivity(; scale)` with no curve is bitwise `Gain(gain = scale)` (SN1).

Built with `Sensitivity(; scale = 1.0)` (no curve, `lambda`/`response` both `nothing`) or
`Sensitivity(; scale = 1.0, lambda_nm | lambda_m, response)` (a measured curve, added Step
2.3b). `_prepare` interpolates the curve onto `expt.lambda` **once**, linearly; a detector
window not spanned by the calibration curve is a loud error (spec D28), never clamped or
extrapolated, since silently extrapolating a sensitivity biases exactly the feature wings being
fitted.

Spec D40, D42.
"""
struct Sensitivity{FT<:Real,L,R} <: AbstractInstrumentEffect
    scale::FT
    lambda::L
    response::R
    response_norm::FT
    function Sensitivity{FT,L,R}(scale::FT, lambda::L, response::R, response_norm::FT) where {FT,L,R}
        isfinite(scale) || throw(ArgumentError("Sensitivity: scale must be finite; got $scale"))
        return new{FT,L,R}(scale, lambda, response, response_norm)
    end
end

function Sensitivity(scale::Real)
    FT = typeof(float(scale))
    return Sensitivity{FT,Nothing,Nothing}(FT(scale), nothing, nothing, one(FT))
end

function Sensitivity(; scale::Real=1.0, response=nothing, kwargs...)
    if response === nothing
        _reject_extra_kwargs(kwargs, ())
        return Sensitivity(float(scale))
    end
    lambda_raw = _si_axis(:lambda, kwargs)
    _reject_extra_kwargs(kwargs, (:lambda_nm, :lambda_m))
    length(lambda_raw) == length(response) || throw(ArgumentError(
        "Sensitivity: lambda has length $(length(lambda_raw)) but response has length $(length(response))"))
    isempty(lambda_raw) && throw(ArgumentError("Sensitivity: the response curve must have at least one point"))
    for l in 2:length(lambda_raw)
        lambda_raw[l] > lambda_raw[l-1] || throw(ArgumentError(
            "Sensitivity: the calibration wavelengths must be strictly increasing"))
    end
    norm = maximum(response)
    FT = float(promote_type(typeof(scale), eltype(lambda_raw), eltype(response)))
    lam = FT.(lambda_raw)
    resp = FT.(response) ./ FT(norm)
    return Sensitivity{FT,typeof(lam),typeof(resp)}(FT(scale), lam, resp, FT(norm))
end

stage(::Sensitivity) = :detector
_chain_order(::Sensitivity) = 4
islinear(::Sensitivity) = true
parameters(e::Sensitivity) = (scale=e.scale,)

function reconstruct(e::Sensitivity{FT,L,R}, values::NamedTuple) where {FT,L,R}
    FT2 = typeof(float(values.scale))
    return Sensitivity{FT2,L,R}(FT2(values.scale), e.lambda, e.response, FT2(e.response_norm))
end

"""
    _interpolate_response(lam_cal::AbstractVector, resp_cal::AbstractVector, lambda::AbstractVector) -> Vector

Linear interpolation of `resp_cal(lam_cal)` onto `lambda`. `lambda` must lie within
`[first(lam_cal), last(lam_cal)]` (checked by the caller, `_prepare`); the bracket for each
`lambda[i]` is found by `searchsortedlast`.

Spec D42, D45.
"""
function _interpolate_response(lam_cal::AbstractVector, resp_cal::AbstractVector, lambda::AbstractVector)
    FT = promote_type(eltype(lam_cal), eltype(resp_cal), eltype(lambda))
    n = length(lam_cal)
    out = Vector{FT}(undef, length(lambda))
    @inbounds for i in eachindex(lambda)
        l = lambda[i]
        j = searchsortedlast(lam_cal, l)
        j = clamp(j, 1, n - 1)
        x0, x1 = lam_cal[j], lam_cal[j+1]
        y0, y1 = resp_cal[j], resp_cal[j+1]
        out[i] = y0 + (l - x0) / (x1 - x0) * (y1 - y0)
    end
    return out
end

"""
    _prepare(e::Sensitivity, lambda, t)

`nothing` when `e` carries no response curve; otherwise `_interpolate_response(e.lambda,
e.response, lambda)`, once, at `Experiment` construction. Throws if `lambda` (the detector
window) is not spanned by `e.lambda` (the calibration curve's own range) — never clamped or
extrapolated.

Spec D42, D28.
"""
_prepare(::Sensitivity{FT,Nothing,Nothing}, lambda::AbstractVector, t::AbstractVector) where {FT} = nothing
function _prepare(e::Sensitivity, lambda::AbstractVector, t::AbstractVector)
    (first(lambda) >= first(e.lambda) && last(lambda) <= last(e.lambda)) || throw(ArgumentError(
        "Sensitivity: the detector window $(first(lambda)) to $(last(lambda)) m is not spanned " *
        "by the calibration curve's $(first(e.lambda)) to $(last(e.lambda)) m range"))
    return _interpolate_response(e.lambda, e.response, lambda)
end

"""
    _convolution_matrix(kernel::AbstractIRFKernel, axis::AbstractVector) -> Matrix

The convolution matrix for `kernel` on `axis`: `K[l, l'] = exp(-(a[l]-a[l'])^2/(2σ^2))` for a
`GaussianKernel`, or `K[l'+offset_k, l'] = weight_k` for a `TabulatedKernel`, **then every
column is renormalized to sum to one.** The source bin `l'` (the column) distributes all of its
power over whichever detector bins exist; at the edges the truncated kernel is renormalized over
exactly those bins — **no zero padding and no wraparound**
([OQ-16](../../dev/open_questions.md#oq-16)). This conserves the streak's total exactly
(`sum(K*P) == sum(P)`, EF7) but does **not** leave a flat field flat at the edges (EF6): the row
sums are one only in the interior. That trade — power conservation over flat-field preservation
— is deliberate.

**Orientation** (Sheffield Eq. 7.2.1, the transmitted power at output `p` as an integral of the
input over the source wavelength — output is the row, source is the column): the spectral IRF
is therefore a **left** multiplication, `M = K_λ·P`, and the temporal one a **right**
multiplication by the transpose, `M = P·K_tᵀ` (`apply!`, spec D16).

The element type is `promote_type` of the kernel's own type and the axis's, never `Float64`
(Pitfall 3): a dual `σ` or a dual set of weights gives a dual matrix.

Sheffield Eq. 7.2.1; `conventions.md` 1.9; spec D13.
"""
function _convolution_matrix(kernel::GaussianKernel, axis::AbstractVector)
    N = length(axis)
    FT = promote_type(typeof(kernel.sigma), eltype(axis))
    K = Matrix{FT}(undef, N, N)
    sigma = kernel.sigma
    two_sigma_sq = 2 * sigma^2
    @inbounds for lp in 1:N
        for l in 1:N
            d = axis[l] - axis[lp]
            K[l, lp] = exp(-(d^2) / two_sigma_sq)
        end
        s = sum(view(K, :, lp))
        for l in 1:N
            K[l, lp] /= s
        end
    end
    return K
end

function _convolution_matrix(kernel::TabulatedKernel{N0,FTk}, axis::AbstractVector) where {N0,FTk}
    Naxis = length(axis)
    if Naxis >= 3
        d0 = axis[2] - axis[1]
        for l in 3:Naxis
            isapprox(axis[l] - axis[l-1], d0; rtol=1e-10) || throw(ArgumentError(
                "_convolution_matrix: a tabulated kernel needs a uniform axis"))
        end
    end
    FT = promote_type(FTk, eltype(axis))
    K = zeros(FT, Naxis, Naxis)
    for lp in 1:Naxis
        for k in 1:N0
            l = lp + kernel.offsets[k]
            if 1 <= l <= Naxis
                K[l, lp] += kernel.weights[k]
            end
        end
        s = sum(@view K[:, lp])
        iszero(s) && throw(ArgumentError("_convolution_matrix: column $lp receives no weight"))
        for l in 1:Naxis
            K[l, lp] /= s
        end
    end
    return K
end

# --- show (spec D29) ---

Base.show(io::IO, e::SpectralIRF{<:GaussianKernel}) =
    @printf(io, "SpectralIRF(FWHM %.3f nm)", lambda_from_si(_fwhm_from_sigma(e.kernel.sigma), :lambda_nm))
Base.show(io::IO, e::SpectralIRF{<:TabulatedKernel}) =
    print(io, "SpectralIRF(tabulated, $(length(e.kernel.offsets)) taps)")
Base.show(io::IO, e::TemporalIRF{<:GaussianKernel}) =
    @printf(io, "TemporalIRF(FWHM %.3f ns)", t_from_si(_fwhm_from_sigma(e.kernel.sigma), :t_ns))
Base.show(io::IO, e::TemporalIRF{<:TabulatedKernel}) =
    print(io, "TemporalIRF(tabulated, $(length(e.kernel.offsets)) taps)")
Base.show(io::IO, e::KSmearing) =
    @printf(io, "KSmearing(\xb1%.2f deg, %d nodes)", theta_from_si(e.half_angle, :theta_deg), length(e.offsets))
function Base.show(io::IO, e::GradientSmearing)
    if e.logarithmic
        @printf(io, "GradientSmearing(%s, %.3f (log), %d nodes)", string(e.path), e.width, length(e.offsets))
    else
        unit, factor = _leaf_display_unit(e.path)
        @printf(io, "GradientSmearing(%s, %.1f %s, %d nodes)", string(e.path), e.width / factor, unit, length(e.offsets))
    end
end
Base.show(io::IO, e::ProbeOffset) = @printf(io, "ProbeOffset(%+.3f nm)", lambda_from_si(e.offset, :lambda_nm))
Base.show(io::IO, e::Sensitivity{FT,Nothing,Nothing}) where {FT} = @printf(io, "Sensitivity(%.1e)", e.scale)
Base.show(io::IO, e::Sensitivity) = @printf(io, "Sensitivity(%.1e, response curve)", e.scale)
