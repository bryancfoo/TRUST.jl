# AbstractNoiseModel, StreakCameraNoise, GaussianNoise, sigma, the Poisson sampler, and
# synthesize. docs/derivations/instrument.md Part 5; spec D21, D22, D23.

using Random: AbstractRNG, default_rng, randn

"""
    AbstractNoiseModel

The abstract type of a streak-camera noise model: `sigma(model, streak)` for the χ² weights,
`synthesize(experiment, model; rng)` for synthetic data.

Spec D21.
"""
abstract type AbstractNoiseModel end

"""
    StreakCameraNoise{FT<:Real} <: AbstractNoiseModel

Swadling et al. (2022) streak-camera noise: `σ[l,j] = sqrt(S̄[l,j]·camera_gain·noise_factor +
read_noise²)` (Swadling Eqs. 37, 40 with n = 1). `camera_gain` is Swadling's `G`, CCD counts per
photoelectron — a different number from `Sensitivity`'s `scale` (counts per m⁻¹), hence the
different name. `sigma_px` is the single-photoelectron point spread, in pixels.

`sigma(model, expected)` is the **realized** per-pixel σ: Swadling's pre-spread variance
`S̄·G·F²` propagated through the squared weights of the point spread, `σ² = (K_λ∘K_λ)(S̄GF²)
(K_t∘K_t)ᵀ + σ_RN²`, the read noise added after the spread (OQ-40 option A, Phase 4.B spec D8).
It is **diagonal-approximate**: the spread also correlates neighbouring pixels (Swadling Eq. 38),
which is not modelled, so an average of many pixels has a larger variance than the averaged σ
says (Phase 4.B's summary A1). At `sigma_px = 0` it is the pre-spread formula bitwise. The data's
mean `K_λ S̄ K_tᵀ` is blurred by a kernel the forward model does not have; that is not modelled
either (the user, 2026-09-26).

Built with `StreakCameraNoise(; camera_gain = 135.0, noise_factor = 1.05, sigma_px = 1.0,
read_noise = 11.0)`.

Spec D21, D8.
"""
Base.@kwdef struct StreakCameraNoise{FT<:Real} <: AbstractNoiseModel
    camera_gain::FT = 135.0
    noise_factor::FT = 1.05
    sigma_px::FT = 1.0
    read_noise::FT = 11.0
end

"""
    GaussianNoise{FT<:Real} <: AbstractNoiseModel

`σ[l,j] = sqrt(a + b·S̄[l,j])`. `a` must be strictly positive, so σ is never zero.

Built with `GaussianNoise(; a = 1.0, b = 0.0)`.

Spec D21.
"""
struct GaussianNoise{FT<:Real} <: AbstractNoiseModel
    a::FT
    b::FT
    function GaussianNoise{FT}(a::FT, b::FT) where {FT<:Real}
        a > 0 || throw(ArgumentError("GaussianNoise: a must be strictly positive so that sigma is never zero; got $a"))
        return new{FT}(a, b)
    end
end
function GaussianNoise(; a::Real=1.0, b::Real=0.0)
    FT = float(promote_type(typeof(a), typeof(b)))
    return GaussianNoise{FT}(FT(a), FT(b))
end

function _check_nonneg_signal(expected::AbstractMatrix)
    for j in axes(expected, 2), l in axes(expected, 1)
        expected[l, j] >= 0 || throw(ArgumentError(
            "expected signal is negative at ($l, $j): $(expected[l,j])"))
    end
    return nothing
end

"""
    sigma(model::AbstractNoiseModel, expected::AbstractMatrix) -> Matrix
    sigma(model::AbstractNoiseModel, sd::StreakData) -> Matrix

σ(λ, t) from the **expected** signal `expected` (or `sd.image`, which is only correct as a χ²
weight when `sd.image` already *is* the expected signal — the docstring warns that passing a
*noisy* streak here gives that realisation's own counts' σ, not the χ² weight, which is a
supplied `sd.sigma` or the model applied to the model's own prediction). Throws naming the pixel
and the value for a negative entry.

For `StreakCameraNoise` with `sigma_px > 0`, this is the **realized** per-pixel σ: the pre-spread
variance `V = expected·camera_gain·noise_factor` propagated through the squared weights of the
point spread, `σ² = (K_λ∘K_λ)·V·(K_t∘K_t)ᵀ + read_noise²` (`_spread_variance`; OQ-40 option A,
Phase 4.B spec D8, `conventions.md` Section 4's `AbstractNoiseModel` entry). At `sigma_px <= 0` it
is today's pre-spread formula, bitwise.

Swadling Eqs. 37, 40; spec D21, D8.
"""
function sigma(model::StreakCameraNoise, expected::AbstractMatrix)
    _check_nonneg_signal(expected)
    model.sigma_px > 0 ||
        return sqrt.(expected .* model.camera_gain .* model.noise_factor .+ model.read_noise^2)
    V = expected .* model.camera_gain .* model.noise_factor
    return sqrt.(_spread_variance(V, model.sigma_px) .+ model.read_noise^2)
end
function sigma(model::GaussianNoise, expected::AbstractMatrix)
    _check_nonneg_signal(expected)
    return sqrt.(model.a .+ model.b .* expected)
end
sigma(model::AbstractNoiseModel, sd::StreakData) = sigma(model, sd.image)

"""
    POISSON_KNUTH_MAX

Above this mean, `_rand_poisson` splits into independent Knuth draws (spec D22) rather than
running Knuth's method directly, since its cost is `O(λ)` and `exp(-λ)` underflows for large λ.

Spec D22, D45.
"""
const POISSON_KNUTH_MAX = 20

"""
    _rand_poisson_knuth(rng::AbstractRNG, lambda::Real) -> Int

Knuth's product-of-uniforms method: `L = exp(-λ)`; multiply uniforms until the product drops
below `L`; return the count of uniforms drawn minus one. `O(λ)` uniforms per call. Valid for `0 <
λ <= POISSON_KNUTH_MAX`, where `exp(-λ)` does not underflow.
"""
function _rand_poisson_knuth(rng::AbstractRNG, lambda::Real)
    L = exp(-lambda)
    k = 0
    p = 1.0
    while true
        k += 1
        p *= rand(rng)
        p > L || return k - 1
    end
end

"""
    _rand_poisson(rng::AbstractRNG, lambda::Real) -> Int

A Poisson deviate of mean `lambda`, without `Distributions.jl` ([OQ-1](../../dev/open_questions.md#oq-1)).
`lambda < 0` throws. `lambda == 0` returns `0`. `lambda <= POISSON_KNUTH_MAX` uses
`_rand_poisson_knuth` directly; above it, splits into `n = ceil(Int, lambda/POISSON_KNUTH_MAX)`
independent Knuth draws at mean `lambda/n` and sums them — exact, since a sum of `n` independent
Poissons of mean `λ/n` is Poisson of mean `λ`, and it is why `exp(-λ)` never underflows and no
rejection method is needed.

The cost is `O(λ)` uniforms per call, acceptable here: the package's expected photoelectrons per
pixel are tens, and the whole-streak cost scales with the *total* photoelectron count (about
`3×10⁶` for a full-size example). If a profile ever shows this dominating, the upgrade is
Hörmann's transformed-rejection method — not written here.

Spec D22.
"""
function _rand_poisson(rng::AbstractRNG, lambda::Real)
    lambda < 0 && throw(ArgumentError("lambda must be non-negative; got $lambda"))
    lambda == 0 && return 0
    lambda <= POISSON_KNUTH_MAX && return _rand_poisson_knuth(rng, lambda)
    n = ceil(Int, lambda / POISSON_KNUTH_MAX)
    lam_n = lambda / n
    total = 0
    for _ in 1:n
        total += _rand_poisson_knuth(rng, lam_n)
    end
    return total
end

"""
    _point_spread_matrices(nl::Integer, nt::Integer, sigma_px::Real) -> (Kl, Kt)

The point spread's two column-normalized convolution matrices, `_convolution_matrix(GaussianKernel(
sigma_px), 1:n)` on each axis (spec D13 of Phase 2): `Kl` is `nl × nl` along λ, `Kt` is `nt × nt`
along t. Shared by `_point_spread` (the image) and `_spread_variance` (the realized σ), so the two
can never use different kernels (Phase 4.B spec D8). Requires `sigma_px > 0` (`GaussianKernel`
throws otherwise).
"""
function _point_spread_matrices(nl::Integer, nt::Integer, sigma_px::Real)
    Kl = _convolution_matrix(GaussianKernel(sigma_px), 1:nl)
    Kt = _convolution_matrix(GaussianKernel(sigma_px), 1:nt)
    return Kl, Kt
end

"""
    _point_spread(image::AbstractMatrix, sigma_px::Real) -> Matrix

The single-photoelectron point spread of `StreakCameraNoise`, a separable Gaussian of standard
deviation `sigma_px` **pixels** applied to both axes of `image` (Swadling Eqs. 25, 26), using the
same column-normalized `_convolution_matrix` the IRFs use (spec D13), built by
`_point_spread_matrices` — so, unlike Swadling's own kernel, which loses flux at the edges, this
one **conserves counts exactly** (NZ5), differing from his form only in the outermost `8σ_px`
pixels.
"""
function _point_spread(image::AbstractMatrix, sigma_px::Real)
    Kl, Kt = _point_spread_matrices(size(image, 1), size(image, 2), sigma_px)
    return Kl * image * transpose(Kt)
end

"""
    _spread_variance(V::AbstractMatrix, sigma_px::Real) -> Matrix

The variance of a pixel after the point spread, given the pre-spread per-pixel variances `V` of
independent pixels: `(Kl ∘ Kl) · V · (Kt ∘ Kt)ᵀ`, `∘` the elementwise square (Phase 4.B spec D8;
`conventions.md` Section 4's `AbstractNoiseModel` entry, [OQ-40](../../dev/open_questions.md#oq-40)).
Exact for the diagonal; the covariance between neighbours (Swadling Eq. 38) is not computed.
"""
function _spread_variance(V::AbstractMatrix, sigma_px::Real)
    Kl, Kt = _point_spread_matrices(size(V, 1), size(V, 2), sigma_px)
    return (Kl .^ 2) * V * transpose(Kt .^ 2)
end

function _synthesize_meta(expt::Experiment, source::AbstractString)
    meta = Dict{String,Any}(
        "source" => String(source),
        "lambda_probe_nm" => lambda_from_si(expt.probe.lambda, :lambda_nm),
        "theta_deg" => theta_from_si(expt.collection.theta, :theta_deg),
        "polarization_deg" => theta_from_si(expt.probe.polarization, :theta_deg),
        "k_model" => string(expt.collection.k_model),
    )
    return meta
end

"""
    synthesize(expt::Experiment, model::AbstractNoiseModel;
               rng::AbstractRNG = Random.default_rng(), source = "synthesize") -> StreakData

A synthetic noisy `StreakData` from `measure(expt)` (spec D23), in this order:

1. `Sbar = measure(expt)`, the expected signal in counts.
2. Throw (naming the pixel and value) if any `Sbar[l,j] < 0`.
3. **`StreakCameraNoise`**: draw `N_e[l,j] = _rand_poisson(rng, Sbar[l,j]/camera_gain)` for every
   pixel, in **column-major order** (`l` fastest, then `j`), so a seed reproduces the same image
   whatever `Threads.nthreads()` is (`synthesize` is single-threaded, NZ7).
4. `N_C[l,j] = camera_gain·N_e[l,j] + camera_gain·sqrt((noise_factor-1)·N_e[l,j])·randn(rng)`
   (Swadling Eqs. 23, 24 — the `camera_gain²` form; see `docs/derivations/instrument.md` Part 5
   for why, against Eq. 24 as printed).
5. If `sigma_px > 0`: `N_C` is convolved with the point spread (`_point_spread`), which
   conserves counts exactly.
6. `image[l,j] = N_C[l,j] + read_noise·randn(rng)`.
7. Returns `StreakData(lambda = expt.lambda, t = expt.t, image, sigma = sigma(model, Sbar),
   meta = source and the probe/collection geometry keys)`: `sigma = sigma(model, Sbar)`, the
   realized per-pixel σ (Phase 4.B spec D8). **`σ` is from `Sbar`, never from the noisy `image`.**

**`GaussianNoise`**: steps 3-6 become `image = Sbar .+ sigma(model, Sbar) .* randn(rng)`, drawn
in the same column-major order.

**The counts are not rounded to integers** — a continuous image is what χ² and AD want.

Spec D23.
"""
function synthesize(expt::Experiment, model::StreakCameraNoise;
    rng::AbstractRNG=default_rng(), source="synthesize")
    Sbar = measure(expt)
    _check_nonneg_signal(Sbar)
    nl, nt = size(Sbar)
    G = model.camera_gain
    F2 = model.noise_factor
    N_C = Matrix{Float64}(undef, nl, nt)
    for j in 1:nt, l in 1:nl
        N_e = _rand_poisson(rng, Sbar[l, j] / G)
        N_C[l, j] = G * N_e + G * sqrt((F2 - 1) * N_e) * randn(rng)
    end
    if model.sigma_px > 0
        N_C = _point_spread(N_C, model.sigma_px)
    end
    image = Matrix{Float64}(undef, nl, nt)
    for j in 1:nt, l in 1:nl
        image[l, j] = N_C[l, j] + model.read_noise * randn(rng)
    end
    sd_sigma = sigma(model, Sbar)
    return StreakData(lambda_m=expt.lambda, t_s=expt.t, image=image, sigma=sd_sigma,
        meta=_synthesize_meta(expt, source))
end

function synthesize(expt::Experiment, model::GaussianNoise;
    rng::AbstractRNG=default_rng(), source="synthesize")
    Sbar = measure(expt)
    _check_nonneg_signal(Sbar)
    nl, nt = size(Sbar)
    sd_sigma = sigma(model, Sbar)
    image = Matrix{Float64}(undef, nl, nt)
    for j in 1:nt, l in 1:nl
        image[l, j] = Sbar[l, j] + sd_sigma[l, j] * randn(rng)
    end
    return StreakData(lambda_m=expt.lambda, t_s=expt.t, image=image, sigma=sd_sigma,
        meta=_synthesize_meta(expt, source))
end

# --- show (spec D29) ---

function Base.show(io::IO, m::StreakCameraNoise)
    print(io, "StreakCameraNoise(G=$(m.camera_gain), F2=$(m.noise_factor), sigma_px=$(m.sigma_px), read_noise=$(m.read_noise))")
end
Base.show(io::IO, m::GaussianNoise) = print(io, "GaussianNoise(a=$(m.a), b=$(m.b))")
