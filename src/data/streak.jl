# StreakData: the measured-streak container. conventions.md Section 4; spec D18.

using Printf: @printf

"""
    StreakData{FT<:Real,I<:AbstractMatrix{FT},S<:Union{Nothing,AbstractMatrix{FT}},
               M<:Union{Nothing,AbstractMatrix{Bool}}}

A measured streak: axes, image, uncertainties, mask, metadata. `lambda` (m) and `t` (s) are both
strictly increasing; `image` is `N_λ × N_t`, **wavelength first**; `sigma` is `nothing` or a
same-shape matrix of strictly positive uncertainties; `mask` is `nothing` or a same-shape
`Bool` matrix, **`true` meaning the pixel is used** — the opposite convention is the likeliest
misreading, and Phase 3's χ² depends on it; `meta` is a free-form `Dict{String,Any}` for
metadata (every key a `String`, every value a `Real`, a `String`, or an `AbstractArray` of
those, so `write_streak` can write it without a type map). **A meta key must not contain `/`**
(D26, Step 4.A.2, 2026-09-23): `write_streak` stores each meta key as the JLD2 path
`"meta/\$k"`, where `/` is a group separator, so a `/`-bearing key is accepted, written and then
unreadable by `read_streak` -- a loud error at construction catches it instead.

Built with `StreakData(; lambda_nm | lambda_m, t_ns | t_s, image, sigma = nothing, mask =
nothing, meta = Dict{String,Any}())`.

`conventions.md` Section 4; spec D18, D26.
"""
struct StreakData{FT<:Real,I<:AbstractMatrix{FT},S<:Union{Nothing,AbstractMatrix{FT}},
    M<:Union{Nothing,AbstractMatrix{Bool}}}
    lambda::Vector{FT}
    t::Vector{FT}
    image::I
    sigma::S
    mask::M
    meta::Dict{String,Any}
end

function _check_streak_axis(axis::AbstractVector, name::AbstractString)
    isempty(axis) && throw(ArgumentError("StreakData: $name axis must be strictly increasing"))
    for i in 2:length(axis)
        axis[i] > axis[i-1] || throw(ArgumentError("StreakData: $name axis must be strictly increasing"))
    end
    return nothing
end

_check_meta_value(v::Real) = true
_check_meta_value(v::AbstractString) = true
_check_meta_value(v::AbstractArray) = all(x -> x isa Real || x isa AbstractString, v)
_check_meta_value(v) = false

function StreakData(; image, sigma=nothing, mask=nothing, meta=Dict{String,Any}(), kwargs...)
    lambda_raw = _si_axis(:lambda, kwargs)
    t_raw = _si_axis(:t, kwargs)
    _reject_extra_kwargs(kwargs, (:lambda_nm, :lambda_m, :t_ns, :t_s))

    _check_streak_axis(lambda_raw, "wavelength")
    _check_streak_axis(t_raw, "time")

    nl, nt = length(lambda_raw), length(t_raw)
    size(image) == (nl, nt) || throw(ArgumentError(
        "StreakData: image is $(size(image)) but does not match the axes $nl x $nt"))
    if sigma !== nothing
        size(sigma) == (nl, nt) || throw(ArgumentError(
            "StreakData: sigma is $(size(sigma)) but does not match the axes $nl x $nt"))
        all(>(0), sigma) || throw(ArgumentError("StreakData: sigma must be strictly positive"))
    end
    if mask !== nothing
        size(mask) == (nl, nt) || throw(ArgumentError(
            "StreakData: mask is $(size(mask)) but does not match the axes $nl x $nt"))
    end
    for (k, v) in meta
        k isa String || throw(ArgumentError("StreakData: meta key $k must be a String"))
        occursin('/', k) && throw(ArgumentError(
            "StreakData: meta key \"$k\" must not contain \"/\": write_streak stores each meta key as a " *
            "JLD2 dataset name, where \"/\" separates groups, and the key would not read back"))
        _check_meta_value(v) || throw(ArgumentError(
            "StreakData: meta[\"$k\"] must be a Real, a String, or an AbstractArray of those"))
    end

    FT = float(promote_type(eltype(lambda_raw), eltype(t_raw), eltype(image)))
    lambda = FT.(lambda_raw)
    t = FT.(t_raw)
    image_ft = eltype(image) === FT ? image : FT.(image)
    sigma_ft = sigma === nothing || eltype(sigma) === FT ? sigma : FT.(sigma)
    return StreakData(lambda, t, image_ft, sigma_ft, mask, meta)
end

"""
    Base.size(sd::StreakData) -> (N_λ, N_t)

The image's size.
"""
Base.size(sd::StreakData) = size(sd.image)

"""
    nbins(sd::StreakData) -> Int

The number of time bins, `length(sd.t)`.
"""
nbins(sd::StreakData) = length(sd.t)

# --- plotting stubs (spec D26; methods exist only when Makie is loaded, Step 2.6) ---

"""
    streakplot(sd::StreakData; masked = true, kwargs...)
    streakplot!(ax, sd::StreakData; masked = true, kwargs...)
    streakplot(lambda, t, image; kwargs...)

A heatmap of a measured streak, λ up, t across. A method exists only when Makie is loaded. With
`masked = true`, the default, a `StreakData` with a mask is drawn with its masked-out pixels blank
(`NaN`, Makie's `nan_color`; `_masked_image`, Phase 4.N spec D18); `masked = false` draws every pixel.

Spec D26; Phase 4.N spec D18.
"""
function streakplot end
function streakplot! end
@doc (@doc streakplot) streakplot!

"""
    _masked_image(sd::StreakData) -> Matrix

What `streakplot` draws of `sd` (Phase 4.N spec D18, [OQ-65](../../dev/open_questions.md#oq-65) item 1):
`sd.image` itself, not a copy, when `sd` has no mask; otherwise a `Matrix{Float64}` copy of the image
with `NaN` at every pixel whose mask is `false` (the pixels the fit does not use), every other pixel
unchanged. Pure; `sd` is not modified.
"""
function _masked_image(sd::StreakData)
    sd.mask === nothing && return sd.image
    img = Matrix{Float64}(sd.image)
    img[.!sd.mask] .= NaN
    return img
end

"""
    spectrumplot(lambda, P; kwargs...)
    spectrumplot!(ax, lambda, P; kwargs...)

One spectrum (or one streak column) against wavelength. A method exists only when Makie is
loaded.

Spec D26.
"""
function spectrumplot end
function spectrumplot! end
@doc (@doc spectrumplot) spectrumplot!

"""
    traceplot(trace::PlasmaTrace, path::AbstractString; sigma = nothing, kwargs...)
    traceplot!(ax, trace, path; sigma = nothing, kwargs...)

The row of `trace.values` named by `path`, against `trace.axis`, converted to the leaf's
canonical boundary unit, with an optional ±σ band. A method exists only when Makie is loaded.

Spec D26.
"""
function traceplot end
function traceplot! end
@doc (@doc traceplot) traceplot!

"""
    lcurveplot(chi2::AbstractVector, R::AbstractVector; weights = nothing, kwargs...)
    lcurveplot!(ax, chi2, R; weights = nothing, kwargs...)

An L-curve from plain vectors (Phase 2 has no L-curve object). A method exists only when Makie
is loaded.

Spec D26.
"""
function lcurveplot end
function lcurveplot! end
@doc (@doc lcurveplot) lcurveplot!

# --- show (spec D29) ---

function Base.show(io::IO, sd::StreakData)
    nl, nt = size(sd)
    s = sd.sigma === nothing ? "no sigma" : "sigma"
    m = sd.mask === nothing ? "no mask" : "mask"
    print(io, "StreakData($nl x $nt, $s, $m)")
end

function Base.show(io::IO, ::MIME"text/plain", sd::StreakData)
    println(io, "StreakData")
    @printf(io, "  lambda: %e to %e nm (%d bins)\n", lambda_from_si(sd.lambda[1], :lambda_nm),
        lambda_from_si(sd.lambda[end], :lambda_nm), length(sd.lambda))
    @printf(io, "  t:      %e to %e ns (%d bins)\n", t_from_si(sd.t[1], :t_ns),
        t_from_si(sd.t[end], :t_ns), length(sd.t))
    println(io, "  sigma: ", sd.sigma === nothing ? "none" : "present")
    println(io, "  mask:  ", sd.mask === nothing ? "none" : "present")
    if !isempty(sd.meta)
        println(io, "  meta keys: ", join(sort(collect(keys(sd.meta))), ", "))
    end
end
