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

# --- show (spec D29) ---

Base.show(io::IO, e::ConstantBackground) = print(io, "ConstantBackground($(e.level))")
Base.show(io::IO, e::PolynomialBackground) = print(io, "PolynomialBackground($(e.coefficients))")
Base.show(io::IO, e::ImageBackground) = print(io, "ImageBackground($(size(e.image)), scale=$(e.scale))")
