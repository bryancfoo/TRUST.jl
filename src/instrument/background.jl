# The three background models: ConstantBackground, PolynomialBackground, ImageBackground.
# All :detector, chain order 5, affine (islinear false). Spec D14.

"""
    ConstantBackground{FT<:Real} <: AbstractInstrumentEffect

Adds `level` (counts) to every pixel. Built with `ConstantBackground(; level = 0.0)`.

Spec D14.
"""
struct ConstantBackground{FT<:Real} <: AbstractInstrumentEffect
    level::FT
end
ConstantBackground(; level::Real=0.0) = ConstantBackground(float(level))

"""
    PolynomialBackground{N,FT<:Real} <: AbstractInstrumentEffect

Adds, to every time bin, `B[l] = Σ_{k=0}^{N-1} c_k s_l^k` with `s_l = (λ_l - λ_0)/Δλ`,
`λ_0 = (λ_first+λ_last)/2`, `Δλ = (λ_last-λ_first)/2`, so `s` runs over `[-1, 1]` and the
coefficients are O(1) whatever the axis is. `λ_0` and `Δλ` are fixed at `Experiment`
construction (`_prepare`), not stored in the effect, so the same `PolynomialBackground` is
reusable on a different axis.

Built with `PolynomialBackground(; coefficients)`.

Spec D14.
"""
struct PolynomialBackground{N,FT<:Real} <: AbstractInstrumentEffect
    coefficients::NTuple{N,FT}
end
function PolynomialBackground(; coefficients)
    N = length(coefficients)
    FT = float(promote_type(typeof.(coefficients)...))
    return PolynomialBackground{N,FT}(NTuple{N,FT}(FT.(Tuple(coefficients))))
end

"""
    ImageBackground{A<:AbstractMatrix,FT<:Real} <: AbstractInstrumentEffect

Adds `scale .* image` to the streak; `image` (`N_λ x N_t`, wavelength first) is shared and never
mutated.

Built with `ImageBackground(; image, scale = 1.0)`.

Spec D14.
"""
struct ImageBackground{A<:AbstractMatrix,FT<:Real} <: AbstractInstrumentEffect
    image::A
    scale::FT
end
ImageBackground(; image, scale::Real=1.0) = ImageBackground(image, float(scale))

for T in (:ConstantBackground, :PolynomialBackground, :ImageBackground)
    @eval stage(::$T) = :detector
    @eval _chain_order(::$T) = 5
    @eval islinear(::$T) = false
end

parameters(e::ConstantBackground) = (level=e.level,)
reconstruct(::ConstantBackground, values::NamedTuple) = ConstantBackground(; level=values.level)

parameters(e::PolynomialBackground) = (coefficients=e.coefficients,)
reconstruct(::PolynomialBackground, values::NamedTuple) = PolynomialBackground(; coefficients=values.coefficients)

parameters(e::ImageBackground) = (scale=e.scale,)
reconstruct(e::ImageBackground, values::NamedTuple) = ImageBackground(; image=e.image, scale=values.scale)

"""
    _prepare(e::PolynomialBackground, lambda, t) -> Vector

The length-`N_λ` vector `B[l]` of the docstring above, evaluated once from `lambda`.
"""
function _prepare(e::PolynomialBackground{N}, lambda::AbstractVector, t::AbstractVector) where {N}
    lam0 = (first(lambda) + last(lambda)) / 2
    dlam = (last(lambda) - first(lambda)) / 2
    FT = promote_type(eltype(e.coefficients), eltype(lambda))
    B = Vector{FT}(undef, length(lambda))
    @inbounds for l in eachindex(lambda)
        s = (lambda[l] - lam0) / dlam
        acc = zero(FT)
        spow = one(FT)
        for k in 1:N
            acc += e.coefficients[k] * spow
            spow *= s
        end
        B[l] = acc
    end
    return B
end

"""
    RationalBackground{P,Q,FT<:Real} <: AbstractInstrumentEffect

A rational-function background on the scaled wavelength `s = (λ - λ₀)/Δλ` `PolynomialBackground`
uses (`λ₀ = (λ_first+λ_last)/2`, `Δλ = (λ_last-λ_first)/2`, so `s ∈ [-1, 1]`):

    B[l] = (Σ_{k=0}^{P-1} numerator[k+1]·s_l^k) / (1 + Σ_{k=1}^{Q} denominator[k]·s_l^k)

The denominator's **constant** term is fixed at `1` (not its leading coefficient, unlike
MATLAB's `rat11`/`rat21`, which are monic: `(p1·s+p2)/(s+q1) ≡ (a0+a1·s)/(1+b1·s)` with `a0 =
p2/q1`, `a1 = p1/q1`, `b1 = 1/q1`, valid whenever `q1 ≠ 0`). Fixing the constant term at `1`
nests `PolynomialBackground` exactly: `denominator = ()` is bitwise `PolynomialBackground(;
coefficients = numerator)` (RB2), and it is well conditioned at `s = 0`, the middle of the
window, which is where the feature usually is.

`rat11` (linear/linear): `numerator = (a0, a1)`, `denominator = (b1,)`. `rat21`
(quadratic/linear): `numerator = (a0, a1, a2)`, `denominator = (b1,)`.

**The pole is the one failure mode the other three backgrounds do not have.** `_prepare` guards
it: the denominator `1 + Σ b_k s^k`, evaluated over the detector window, must not change sign and
its minimum modulus must not fall below `1e-3` — deliberately loose, since the guard refuses a
background that is already ill-conditioned over the data rather than trying to locate the pole
exactly.

Built with `RationalBackground(; numerator, denominator)`.

Spec D46.
"""
struct RationalBackground{P,Q,FT<:Real} <: AbstractInstrumentEffect
    numerator::NTuple{P,FT}
    denominator::NTuple{Q,FT}
end

function RationalBackground(; numerator, denominator)
    P = length(numerator)
    Q = length(denominator)
    FT = float(promote_type(typeof.(numerator)..., typeof.(denominator)...))
    return RationalBackground{P,Q,FT}(NTuple{P,FT}(FT.(Tuple(numerator))),
        NTuple{Q,FT}(FT.(Tuple(denominator))))
end

stage(::RationalBackground) = :detector
_chain_order(::RationalBackground) = 5
islinear(::RationalBackground) = false

parameters(e::RationalBackground) = (numerator=e.numerator, denominator=e.denominator)
reconstruct(::RationalBackground, values::NamedTuple) =
    RationalBackground(; numerator=values.numerator, denominator=values.denominator)

"""
    _prepare(e::RationalBackground, lambda, t) -> Vector

The length-`N_λ` vector `B[l]` of the docstring above, evaluated once from `lambda`, after
guarding the denominator against a pole inside the detector window (spec D28, D46): throws if
`1 + Σ b_k s^k` changes sign over `lambda`, or if its minimum modulus over `lambda` is below
`1e-3`, naming the window, the coefficients and the wavelength where it is smallest.
"""
function _prepare(e::RationalBackground{P,Q}, lambda::AbstractVector, t::AbstractVector) where {P,Q}
    lam0 = (first(lambda) + last(lambda)) / 2
    dlam = (last(lambda) - first(lambda)) / 2
    FT = promote_type(eltype(e.numerator), eltype(e.denominator), eltype(lambda))

    denom = Vector{FT}(undef, length(lambda))
    @inbounds for l in eachindex(lambda)
        s = (lambda[l] - lam0) / dlam
        acc = one(FT)
        spow = s
        for k in 1:Q
            acc += e.denominator[k] * spow
            spow *= s
        end
        denom[l] = acc
    end

    minabs, minidx = findmin(abs, denom)
    signchange = any(l -> denom[l] * denom[l-1] < 0, 2:length(denom))
    if signchange || minabs < 1e-3
        throw(ArgumentError("RationalBackground: the denominator $(e.denominator) has a pole " *
            "(or is nearly singular) over the window $(first(lambda)) to $(last(lambda)) m; " *
            "smallest |denominator| = $minabs at lambda = $(lambda[minidx]) m"))
    end

    B = Vector{FT}(undef, length(lambda))
    @inbounds for l in eachindex(lambda)
        s = (lambda[l] - lam0) / dlam
        acc = zero(FT)
        spow = one(FT)
        for k in 1:P
            acc += e.numerator[k] * spow
            spow *= s
        end
        B[l] = acc / denom[l]
    end
    return B
end

# --- show (spec D29) ---

Base.show(io::IO, e::ConstantBackground) = print(io, "ConstantBackground($(e.level))")
Base.show(io::IO, e::PolynomialBackground) = print(io, "PolynomialBackground($(e.coefficients))")
Base.show(io::IO, e::ImageBackground) = print(io, "ImageBackground($(size(e.image)), scale=$(e.scale))")
Base.show(io::IO, e::RationalBackground) =
    print(io, "RationalBackground($(e.numerator) / (1, $(e.denominator)...))")
