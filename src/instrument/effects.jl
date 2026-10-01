# AbstractInstrumentEffect and its concrete effects: the three in-kernel effects (KSmearing,
# GradientSmearing, WavelengthCalibration), the two post-kernel IRF convolutions and Sensitivity,
# and the convolution-matrix builder they and the noise model's point spread share.
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
(`effect(aP+bQ) = a·effect(P) + b·effect(Q)` to round-off): the two IRFs and `Sensitivity`. The three
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
`TemporalIRF` -> `Sensitivity` -> backgrounds ([OQ-13](../../dev/open_questions.md#oq-13)).

Spec D9, D15.
"""
function _chain_order end

"""
    _chain_class(effect::AbstractInstrumentEffect) -> Symbol

The effect's uniqueness class: `:none` for an effect any number of which may appear in a chain,
or a per-type Symbol for one of which `Experiment` allows at most one
(`_check_unique_chain_classes`). Unlike `_chain_order`, which is keyed on the chain's numeric
position and cannot tell apart the three different classes that share position 1
(`KSmearing`, `GradientSmearing`, `WavelengthCalibration`), this is a trait with one method per
type — not `isa` branching (`conventions.md` Section 4) — so a kernel-stage effect can be made
unique-per-chain without the position number being asked to mean two things.

Spec D8 (Phase 4.A, Step 4.A.1a, 2026-09-22).
"""
_chain_class(::AbstractInstrumentEffect) = :none

"""
    samples(effect::AbstractInstrumentEffect, plasma::Plasma, probe::ProbeBeam, collection::Collection,
            lambda::AbstractVector) -> NTuple

For a `:kernel` effect: the weighted list of `(weight, plasma, probe, collection, lambda)` samples
the per-bin kernel folds over; each in-kernel effect rebuilds only the one of `plasma`, `probe`,
`collection` or `lambda` it modifies (`KSmearing` a `Collection`, `GradientSmearing` a `Plasma`,
`WavelengthCalibration` the wavelength axis), passing the other three through unchanged, as the
same objects. Spec D9, D10, D40; the fifth element (the evaluation axis) added at Step 4.A.1a
(spec D4, 2026-09-22).
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

# D25 (chunk 4, 2026-09-23): Val-dispatched, so reconstruct(::KSmearing{NQ})/reconstruct(::
# GradientSmearing{NQ}) infer a concrete offsets/weights type under a dual instead of Tuple{Any,
# Any} -- the Integer form's runtime `if` over four NTuple lengths was invisible in Float64 (a
# reconstruct called alone with constant arguments reads 0 B, constant propagation rescues it) but
# made the per-bin dual pass dispatch dynamically once a kernel-stage effect is bound (JT16).
_gauss_legendre(::Val{1}) = (GAUSS_LEGENDRE_NODES.n1, GAUSS_LEGENDRE_WEIGHTS.n1)
_gauss_legendre(::Val{3}) = (GAUSS_LEGENDRE_NODES.n3, GAUSS_LEGENDRE_WEIGHTS.n3)
_gauss_legendre(::Val{5}) = (GAUSS_LEGENDRE_NODES.n5, GAUSS_LEGENDRE_WEIGHTS.n5)
_gauss_legendre(::Val{7}) = (GAUSS_LEGENDRE_NODES.n7, GAUSS_LEGENDRE_WEIGHTS.n7)

_gauss_hermite(::Val{1}) = (GAUSS_HERMITE_NODES.n1, GAUSS_HERMITE_WEIGHTS.n1)
_gauss_hermite(::Val{3}) = (GAUSS_HERMITE_NODES.n3, GAUSS_HERMITE_WEIGHTS.n3)
_gauss_hermite(::Val{5}) = (GAUSS_HERMITE_NODES.n5, GAUSS_HERMITE_WEIGHTS.n5)
_gauss_hermite(::Val{7}) = (GAUSS_HERMITE_NODES.n7, GAUSS_HERMITE_WEIGHTS.n7)

function _legendre_offsets_weights(half_angle::Real, nodes::Integer)
    _check_nodes(nodes)
    return _legendre_offsets_weights(half_angle, Val(nodes))
end
function _legendre_offsets_weights(half_angle::Real, v::Val{N}) where {N}
    xi, w = _gauss_legendre(v)
    FT = float(typeof(half_angle))
    offsets = ntuple(q -> FT(half_angle) * FT(xi[q]), Val(N))
    wsum = FT(sum(w))
    weights = ntuple(q -> FT(w[q]) / wsum, Val(N))
    return offsets, weights
end

function _hermite_offsets_weights(width::Real, nodes::Integer)
    _check_nodes(nodes)
    return _hermite_offsets_weights(width, Val(nodes))
end
function _hermite_offsets_weights(width::Real, v::Val{N}) where {N}
    xi, w = _gauss_hermite(v)
    FT = float(typeof(width))
    sqrt2 = sqrt(FT(2))
    offsets = ntuple(q -> FT(width) * sqrt2 * FT(xi[q]), Val(N))
    wsum = FT(sum(w))
    weights = ntuple(q -> FT(w[q]) / wsum, Val(N))
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
    (isfinite(half_angle_raw) && _primal_value(half_angle_raw) >= 0) || throw(ArgumentError(
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
    (isfinite(half_angle) && _primal_value(half_angle) >= 0) || throw(ArgumentError(
        "KSmearing: half angle must be finite and non-negative; got $half_angle rad"))
    offsets, weights = _legendre_offsets_weights(half_angle, Val(NQ))
    FT = eltype(offsets)
    return KSmearing{NQ,FT}(FT(half_angle), offsets, weights)
end

"""
    samples(e::KSmearing, p::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector) -> NTuple

`θ -> θ + e.offsets[q]`; the plasma, the probe and the wavelength axis are untouched. Only
`Collection` is rebuilt, through its **inner** constructor `Collection{FT}`, never the keyword
constructor, so this is allocation-free (EF9; Level 2 measured 0 bytes over 1000 samples for the
whole geometry before the split, and the split makes the rebuilt struct smaller still, spec D40).

Spec D10, D40, Pitfall 10.
"""
function samples(e::KSmearing{NQ,FTe}, p::Plasma, probe::ProbeBeam, collection::Collection{FTc},
                  lambda::AbstractVector) where {NQ,FTe,FTc}
    FTg = promote_type(FTc, FTe)
    return ntuple(Val(NQ)) do q
        (e.weights[q], p, probe,
            Collection{FTg}(FTg(collection.theta) + FTg(e.offsets[q]), collection.k_model,
                FTg(collection.lambda_center), FTg(collection.dispersion), collection.npixels,
                FTg(collection.reference_pixel)), lambda)
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
    (isfinite(width) && _primal_value(width) >= 0) || throw(ArgumentError(
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
    (isfinite(width) && _primal_value(width) >= 0) || throw(ArgumentError(
        "GradientSmearing: width must be finite and non-negative; got $width"))
    offsets, weights = _hermite_offsets_weights(width, Val(NQ))
    FT = eltype(offsets)
    return GradientSmearing{NQ,FT}(e.ref, e.path, FT(width), e.logarithmic, offsets, weights)
end

_offset_value(e::GradientSmearing, x, q) = e.logarithmic ? x * exp(e.offsets[q]) : x + e.offsets[q]

"""
    samples(e::GradientSmearing, p::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector) -> NTuple

One leaf of `p` offset per sample (`_with_leaf`, allocation-free by the same argument as
`plasma(trace, j)`); the probe, the collection and the wavelength axis are untouched.

Spec D10, D40, Pitfall 12.
"""
function samples(e::GradientSmearing{NQ}, p::Plasma, probe::ProbeBeam, collection::Collection,
                  lambda::AbstractVector) where {NQ}
    return ntuple(q -> (e.weights[q],
            _with_leaf(p, e.ref, _offset_value(e, _leaf_value(p, e.ref), q)), probe, collection, lambda),
        Val(NQ))
end

"""
    WavelengthCalibration{FT<:Real} <: AbstractInstrumentEffect

A calibration of the **measured wavelength axis**, replacing the Phase 2/3 probe-wavelength-shift
effect at Step 4.A.1a (the user, 2026-09-21, [OQ-24](../../dev/open_questions.md#oq-24)): the collaborator judges the
dominant uncertainty to be the calibration of the measured axis, not the probe beam, whose offset
is small experimentally, and the two are nearly degenerate (both shift the spectrum against the
pixels), so this is a **replacement**, not an addition.

A scalar `offset` (m) and a dimensionless `dispersion_scale` perturb **the axis at which the
kernel evaluates the spectrum for each pixel**: the pixel labelled `λ_l` is modelled as seeing

    λ_eval,l = λ_l + offset + dispersion_scale · (λ_l − λ_ref),   λ_ref = (first(λ) + last(λ)) / 2

with `λ` the evaluation axis the effect receives and `λ_ref` its midpoint, **always** — not a
field, a keyword or a deck key (spec D5): λ_ref is a pure reparameterization of `(offset,
dispersion_scale)`, so the only thing any choice of it can change is the conditioning of the two
parameters against each other, and the window's own midpoint is what decorrelates them. **The IRF,
the sensitivity and every background stay on the detector's pixel axis** `expt.lambda`, which never
moves (WC8) — a spectrometer's response belongs there, not on a perturbed evaluation point. A
probe wavelength is never touched: `ProbeBeam.lambda` is fixed.

Built with `WavelengthCalibration(; lambda_offset_nm | lambda_offset_m, dispersion_scale = 0.0)`.
The offset is **required** (no default), so that `WavelengthCalibration()` is a loud error rather
than a silent identity; `dispersion_scale = 0.0` is the deliberate identity and needs no keyword.

**Composes like every other order-1 effect** (D4): the order-1 effects in a chain are stably
sorted, and each passes through what it does not modify (`KSmearing`/`GradientSmearing` the
wavelength axis, `WavelengthCalibration` the plasma, probe and collection), so the fold gives the
same result whatever relative order the user declares them in.

Spec D3, D4, D5 (Phase 4.A, Step 4.A.1a, 2026-09-22).
"""
struct WavelengthCalibration{FT<:Real} <: AbstractInstrumentEffect
    offset::FT
    dispersion_scale::FT
end

"""
    _check_calibration(offset, dispersion_scale)

The three validations `WavelengthCalibration`'s keyword constructor and `reconstruct` both apply
(D25, chunk 4): `offset` finite, `dispersion_scale` finite, `abs(dispersion_scale) < 1`. Shared so
that WC1's messages do not move whichever path throws them.
"""
function _check_calibration(offset, dispersion_scale)
    isfinite(offset) || throw(ArgumentError(
        "WavelengthCalibration: offset must be finite; got $offset m"))
    isfinite(dispersion_scale) || throw(ArgumentError(
        "WavelengthCalibration: dispersion_scale must be finite; got $dispersion_scale"))
    abs(_primal_value(dispersion_scale)) < 1 || throw(ArgumentError(
        "WavelengthCalibration: dispersion_scale must satisfy abs(dispersion_scale) < 1; got $dispersion_scale"))
    return nothing
end

function WavelengthCalibration(; dispersion_scale::Real=0.0, kwargs...)
    offset_raw = _si_value(:lambda_offset, kwargs)
    _reject_extra_kwargs(kwargs, (:lambda_offset_nm, :lambda_offset_m))
    _check_calibration(offset_raw, dispersion_scale)
    FT = promote_type(float(typeof(offset_raw)), float(typeof(dispersion_scale)))
    return WavelengthCalibration{FT}(FT(offset_raw), FT(dispersion_scale))
end

stage(::WavelengthCalibration) = :kernel
_chain_order(::WavelengthCalibration) = 1
islinear(::WavelengthCalibration) = false
_chain_class(::WavelengthCalibration) = :WavelengthCalibration
parameters(e::WavelengthCalibration) = (offset=e.offset, dispersion_scale=e.dispersion_scale)

# D25 (chunk 4): through the inner constructor, not the keyword one -- the keyword path's
# _si_value has a data-dependent filter length and so infers only WavelengthCalibration, no type
# parameter, which is invisible in Float64 but dispatches dynamically in the per-bin dual pass
# once the calibration is bound (JT16). _check_calibration keeps WC1's messages verbatim.
function reconstruct(::WavelengthCalibration, values::NamedTuple)
    offset = values.offset
    dispersion_scale = values.dispersion_scale
    _check_calibration(offset, dispersion_scale)
    FT = promote_type(float(typeof(offset)), float(typeof(dispersion_scale)))
    return WavelengthCalibration{FT}(FT(offset), FT(dispersion_scale))
end

"""
    _CalibratedAxis{FT<:Real,V<:AbstractVector} <: AbstractVector{FT}

The lazy, never-materialized perturbed evaluation axis `samples(::WavelengthCalibration, ...)`
returns: `a[l] = base[l] + offset + dispersion_scale * (base[l] - lambda_ref)`, computed on
indexing rather than stored, so that a bin's evaluation axis costs no `N_λ`-element allocation in
the value pass or the dual pass (`conventions.md` 5.2). Not `isbits` (it holds a reference to
`base`); it is the one new object per bin this phase's mechanism adds (Section 12 row 2).

Spec D4.
"""
struct _CalibratedAxis{FT<:Real,V<:AbstractVector} <: AbstractVector{FT}
    base::V
    offset::FT
    dispersion_scale::FT
    lambda_ref::FT
end

Base.size(a::_CalibratedAxis) = size(a.base)
Base.IndexStyle(::Type{<:_CalibratedAxis}) = IndexLinear()
@inline Base.getindex(a::_CalibratedAxis, l::Int) =
    a.base[l] + a.offset + a.dispersion_scale * (a.base[l] - a.lambda_ref)

"""
    samples(e::WavelengthCalibration, p::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector) -> NTuple

The plasma, the probe and the collection are untouched; only the wavelength axis is rebuilt, as a
lazy `_CalibratedAxis` (never materialized). `λ_ref` is computed from `first(lambda)`/`last(lambda)`
of the axis this call receives, **always the plain-float base axis** even under a dual pass (the
base is `expt.lambda`, plain `Float64`), so `λ_ref` carries no partials: it is a constant of the
window, not a parameter. One sample, weight one.

Spec D4, D5.
"""
function samples(e::WavelengthCalibration, p::Plasma, probe::ProbeBeam, collection::Collection,
                  lambda::AbstractVector)
    FT = promote_type(eltype(lambda), typeof(e.offset), typeof(e.dispersion_scale))
    lref = (FT(first(lambda)) + FT(last(lambda))) / 2
    axis = _CalibratedAxis{FT,typeof(lambda)}(lambda, FT(e.offset), FT(e.dispersion_scale), lref)
    return ((one(FT), p, probe, collection, axis),)
end

"""
    PixelIntegration <: AbstractInstrumentEffect

The detector pixel's integration over its own width, sampled ([OQ-53](../../dev/open_questions.md#oq-53),
the user's decision of 2026-09-30; Phase 4.F spec D10): the spectrum the kernel evaluates for the
pixel at `λ_l` is the mean of its values at the `N = subsamples` points

    λ_l + (i − (N+1)/2) · Δ_l / N,    i = 1, …, N,    Δ_l = (λ_{l+1} − λ_{l−1}) / 2

(one-sided at the axis's two ends: `Δ_1 = λ_2 − λ_1`, `Δ_n = λ_n − λ_{n−1}`), equal weights `1/N`,
so that the spectral IRF convolves pixel-averaged samples rather than pixel-centre ones. `N = 1`
is the pixel centre, exactly today's model: weight `1.0` and an offset of `0.0`.

**The innermost in-kernel effect, always**: `Experiment` sorts it after every other effect of
chain order 1 (`_kernel_rank`), so it receives the axis `WavelengthCalibration` has already
perturbed and its `Δ_l` is that axis's own spacing, and every node of `KSmearing` and
`GradientSmearing` is sub-sampled. At most one per chain (`_chain_class`). No fittable parameter.
The count is a field, not a type parameter, so every count shares one compiled kernel.

Built with `PixelIntegration(; subsamples::Integer = 1)`; `subsamples` a positive integer.

Spec D10, D11 (Phase 4.F, Step 4.F.2, 2026-09-30).
"""
struct PixelIntegration <: AbstractInstrumentEffect
    subsamples::Int
    function PixelIntegration(; subsamples::Integer=1)
        subsamples >= 1 || throw(ArgumentError(
            "PixelIntegration: subsamples must be a positive integer; got $subsamples"))
        return new(Int(subsamples))
    end
end

stage(::PixelIntegration) = :kernel
_chain_order(::PixelIntegration) = 1
islinear(::PixelIntegration) = false
_chain_class(::PixelIntegration) = :PixelIntegration
parameters(::PixelIntegration) = NamedTuple()
reconstruct(e::PixelIntegration, ::NamedTuple) = e

"""
    _kernel_rank(effect::AbstractInstrumentEffect) -> Int

The order of an effect among those of one `_chain_order` (spec D11): `0` for every effect but
`PixelIntegration`, `1` for it, so `Experiment`'s stable sort puts it after every other in-kernel
effect, whatever order the chain was given in, and leaves every other chain exactly as before.
"""
_kernel_rank(::AbstractInstrumentEffect) = 0
_kernel_rank(::PixelIntegration) = 1

"""
    _prepare(e::PixelIntegration, lambda, t)

`nothing`, after checking that `lambda` has at least two pixels: one pixel has no width to
integrate over.
"""
function _prepare(::PixelIntegration, lambda::AbstractVector, t::AbstractVector)
    length(lambda) >= 2 || throw(ArgumentError(
        "PixelIntegration: the wavelength axis needs at least two pixels to give a pixel a width; got $(length(lambda))"))
    return nothing
end

"""
    _SubsampledAxis{FT<:Real,V<:AbstractVector} <: AbstractVector{FT}

The lazy, never-materialized axis of one `PixelIntegration` sample: `a[l] = base[l] + c · Δ_l`,
`Δ_l` the centred spacing of `base` at `l` (one-sided at the ends), `c = (i − (N+1)/2)/N`.
Computed on indexing, as `_CalibratedAxis` is, so a sample costs no `N_λ`-element allocation in
the value pass or the dual pass.

Spec D10.
"""
struct _SubsampledAxis{FT<:Real,V<:AbstractVector} <: AbstractVector{FT}
    base::V
    c::FT
end

"""
    _subsample_lambda(lambda::AbstractVector, l::Int, c) -> Real

`lambda[l] + c · Δ_l`, `Δ_l` the centred spacing of `lambda` at `l`, one-sided at the two ends
(spec D10): the one formula for a sub-sample's wavelength, read by `_SubsampledAxis` and by the
kernel's own fold over `PixelIntegration` (`measure.jl`).
"""
@inline function _subsample_lambda(lambda::AbstractVector, l::Int, c)
    n = length(lambda)
    lo = l == 1 ? 1 : l - 1
    hi = l == n ? n : l + 1
    return lambda[l] + c * ((lambda[hi] - lambda[lo]) / (hi - lo))
end

Base.size(a::_SubsampledAxis) = size(a.base)
Base.IndexStyle(::Type{<:_SubsampledAxis}) = IndexLinear()
@inline Base.getindex(a::_SubsampledAxis, l::Int) = _subsample_lambda(a.base, l, a.c)

"""
    samples(e::PixelIntegration, p::Plasma, probe::ProbeBeam, collection::Collection, lambda::AbstractVector) -> Vector

`N = e.subsamples` samples, each the plasma, the probe and the collection untouched and the axis
a lazy `_SubsampledAxis` of `lambda`, each weighted `1/N`: the in-kernel effect's contract, as a
`Vector` rather than a tuple since `N` is a runtime value. The kernel never calls it: it folds the
same samples, in the same order, through its own method of `_accumulate!` (`measure.jl`, spec
D11), since `PixelIntegration` is always the last in-kernel effect.

Spec D10.
"""
function samples(e::PixelIntegration, p::Plasma, probe::ProbeBeam, collection::Collection,
                  lambda::AbstractVector)
    N = e.subsamples
    FT = promote_type(eltype(lambda), Float64)
    w = 1.0 / N
    return [(w, p, probe, collection, _SubsampledAxis{FT,typeof(lambda)}(lambda, FT((i - (N + 1) / 2) / N)))
            for i in 1:N]
end

# --- D12: the post-kernel IRFs and Sensitivity ---

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
        (isfinite(sigma) && _primal_value(sigma) > 0) || throw(ArgumentError(
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
@doc (@doc SpectralIRF) TemporalIRF

function SpectralIRF(; kwargs...)
    fwhm_raw = _si_value(:fwhm_lambda, kwargs)
    _reject_extra_kwargs(kwargs, (:fwhm_nm, :fwhm_m))
    _primal_value(fwhm_raw) > 0 || throw(ArgumentError("SpectralIRF: FWHM must be strictly positive; got $fwhm_raw m"))
    return SpectralIRF(GaussianKernel(_sigma_from_fwhm(fwhm_raw)))
end

function TemporalIRF(; kwargs...)
    fwhm_raw = _si_value(:fwhm_t, kwargs)
    _reject_extra_kwargs(kwargs, (:fwhm_ns, :fwhm_s))
    _primal_value(fwhm_raw) > 0 || throw(ArgumentError("TemporalIRF: FWHM must be strictly positive; got $fwhm_raw s"))
    return TemporalIRF(GaussianKernel(_sigma_from_fwhm(fwhm_raw)))
end

stage(::SpectralIRF) = :detector
stage(::TemporalIRF) = :detector
_chain_order(::SpectralIRF) = 2
_chain_order(::TemporalIRF) = 3
islinear(::SpectralIRF) = true
islinear(::TemporalIRF) = true
_chain_class(::SpectralIRF) = :SpectralIRF
_chain_class(::TemporalIRF) = :TemporalIRF

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
_chain_class(::Sensitivity) = :Sensitivity
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
Base.show(io::IO, e::WavelengthCalibration) =
    @printf(io, "WavelengthCalibration(%+.3f nm, dispersion_scale = %.3e)",
        lambda_from_si(e.offset, :lambda_nm), e.dispersion_scale)
Base.show(io::IO, e::PixelIntegration) = print(io, "PixelIntegration($(e.subsamples) subsamples)")
Base.show(io::IO, e::Sensitivity{FT,Nothing,Nothing}) where {FT} = @printf(io, "Sensitivity(%.1e)", e.scale)
Base.show(io::IO, e::Sensitivity) = @printf(io, "Sensitivity(%.1e, response curve)", e.scale)
