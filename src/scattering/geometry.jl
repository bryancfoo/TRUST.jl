# The probe beam, the collection, k(lambda_s) from the light-wave dispersion relation, the mean
# k-hat convention, the polarization factor and the detector axis from a dispersion.
# conventions.md 1.2, 1.3, 3.2, 3.3, 3.4, 1.8; docs/derivations/geometry.md; spec D40, D41, D44.
#
# The former single geometry struct is retired (spec D40, 2026-09-10): it mixed the probe's own properties
# (wavelength, polarization, energetics) with the collection's (angle, k model, dispersion).
# Split into ProbeBeam and Collection so the convective gain of Step 2.3b can reach the probe's
# energetics without a second, overlapping type.

using Printf: @sprintf, @printf

"""
    ProbeBeam{FT<:Real}

The probe: its wavelength, its incident polarization angle, its normalized vector potential
squared `a0_sq` (derived from `intensity_wcm2 | intensity_wm2` with `A0_SQ_COEFFICIENT`,
`conventions.md` 1.3), and its beam diameter. All fields SI (`lambda`, `diameter` in m,
`polarization` in rad, `a0_sq` dimensionless). `a0_sq` and `diameter` are `0` when not given
(`intensity_wcm2 | intensity_wm2` and `diameter_um | diameter_m` both default to `0`); anything
that reads either (the convective gain, spec D43) errors loudly on `0` rather than treating it as
a value (`conventions.md` 5.9).

Built with `ProbeBeam(; lambda_probe_nm | lambda_probe_m, polarization_deg | polarization_rad =
0, intensity_wcm2 | intensity_wm2 = 0, diameter_um | diameter_m = 0)`.

`conventions.md` 1.2, 1.3, Section 4; `docs/derivations/geometry.md`; spec D40.
"""
struct ProbeBeam{FT<:Real}
    lambda::FT
    polarization::FT
    a0_sq::FT
    diameter::FT
end

function ProbeBeam(; kwargs...)
    lambda_raw = _si_value(:lambda_probe, kwargs)
    polarization_raw = _si_value(:polarization, kwargs, 0.0)
    intensity_raw = _si_value(:intensity, kwargs, 0.0)
    diameter_raw = _si_value(:diameter, kwargs, 0.0)
    _reject_extra_kwargs(kwargs, (:lambda_probe_nm, :lambda_probe_m, :polarization_deg,
        :polarization_rad, :intensity_wcm2, :intensity_wm2, :diameter_um, :diameter_m))

    lambda_raw > 0 || throw(ArgumentError(
        "ProbeBeam: lambda_probe must be positive; got $lambda_raw m"))
    intensity_raw >= 0 || throw(ArgumentError(
        "ProbeBeam: intensity must be non-negative; got $intensity_raw W/m^2"))
    diameter_raw >= 0 || throw(ArgumentError(
        "ProbeBeam: diameter must be non-negative; got $diameter_raw m"))

    FT = float(promote_type(typeof(lambda_raw), typeof(polarization_raw), typeof(intensity_raw),
        typeof(diameter_raw)))
    a0_sq = FT(A0_SQ_COEFFICIENT) * FT(lambda_raw)^2 * FT(intensity_raw)
    return ProbeBeam{FT}(FT(lambda_raw), FT(polarization_raw), a0_sq, FT(diameter_raw))
end

"""
    Collection{FT<:Real}

The collection optics: the scattering angle `theta` between the probe and the scattered
direction (rad), the `k` model (`conventions.md` 3.3), and, optionally, the detector dispersion
(`lambda_center`, `dispersion`, `npixels`, `reference_pixel`) that `detector_axis` reads.
`lambda_center`, `dispersion`, `reference_pixel` are `0` and `npixels` is `0` when no dispersion
is given.

Built with `Collection(; theta_deg | theta_rad, k_model::Symbol = :exact, lambda_center_nm |
lambda_center_m = nothing, dispersion_nm_px | dispersion_m_px = nothing, npixels = nothing,
reference_pixel = nothing)`, or with a direction pair in place of `theta_*`:
`Collection(; k_probe, k_scatter, k_model = :exact, ...)`, two non-degenerate 3-vectors that are
normalized and reduced to `theta = acos(k̂_probe · k̂_scatter)` and then dropped
(`conventions.md` 3.2 fixes **k̂** as one direction for the whole spectrum, so nothing downstream
consumes a direction). Exactly one of `theta_*` or the direction pair must be given. The four
dispersion keywords are all-or-nothing: giving one without the rest is an error.

`conventions.md` 1.2, 3.2-3.4, Section 4; `docs/derivations/geometry.md`; spec D40, D44.
"""
struct Collection{FT<:Real}
    theta::FT
    k_model::Symbol
    lambda_center::FT
    dispersion::FT
    npixels::Int
    reference_pixel::FT
end

"""
    _theta_from_directions(k_probe, k_scatter) -> Real

The angle between two non-degenerate 3-vectors, `acos(k̂_probe · k̂_scatter)`. Throws for a zero
vector, or for a pair too close to parallel or antiparallel (within `1e-9` of the cosine's
domain edge) to define a scattering plane reliably.

Spec D40.
"""
function _theta_from_directions(k_probe, k_scatter)
    length(k_probe) == 3 && length(k_scatter) == 3 || throw(ArgumentError(
        "Collection: k_probe and k_scatter must be 3-vectors"))
    np = sqrt(sum(abs2, k_probe))
    ns = sqrt(sum(abs2, k_scatter))
    (np > 0 && ns > 0) || throw(ArgumentError(
        "Collection: k_probe and k_scatter must be non-zero vectors"))
    khat_p = k_probe ./ np
    khat_s = k_scatter ./ ns
    c = sum(khat_p .* khat_s)
    (-1 + 1e-9 <= c <= 1 - 1e-9) || throw(ArgumentError(
        "Collection: k_probe and k_scatter are too close to parallel or antiparallel to define " *
        "a scattering angle"))
    return acos(c)
end

function _check_dispersion_kwargs(kwargs)
    names = (:lambda_center_nm, :lambda_center_m, :dispersion_nm_px, :dispersion_m_px,
        :npixels, :reference_pixel)
    given = filter(k -> haskey(kwargs, k), names)
    has_lambda_center = :lambda_center_nm in given || :lambda_center_m in given
    has_dispersion = :dispersion_nm_px in given || :dispersion_m_px in given
    has_npixels = :npixels in given
    has_reference = :reference_pixel in given
    all_given = has_lambda_center && has_dispersion && has_npixels && has_reference
    none_given = isempty(given)
    (all_given || none_given) || throw(ArgumentError(
        "Collection: lambda_center, dispersion, npixels and reference_pixel must be given " *
        "together or not at all; got $given"))
    return none_given
end

function Collection(; k_model::Symbol=:exact, k_probe=nothing, k_scatter=nothing, kwargs...)
    k_model in (:exact, :approximate) || throw(ArgumentError(
        "Collection: k_model must be :exact or :approximate; got :$k_model"))

    have_theta = haskey(kwargs, :theta_deg) || haskey(kwargs, :theta_rad)
    have_dirs = (k_probe !== nothing) || (k_scatter !== nothing)
    have_theta && have_dirs && throw(ArgumentError(
        "Collection: give theta_deg/theta_rad or k_probe/k_scatter, not both"))

    if have_dirs
        (k_probe !== nothing && k_scatter !== nothing) || throw(ArgumentError(
            "Collection: both k_probe and k_scatter must be given"))
        theta_raw = _theta_from_directions(k_probe, k_scatter)
    elseif have_theta
        theta_raw = _si_value(:theta, kwargs)
    else
        throw(ArgumentError(
            "Collection: give theta_deg/theta_rad or k_probe/k_scatter"))
    end

    no_dispersion = _check_dispersion_kwargs(kwargs)
    lambda_center_raw = no_dispersion ? 0.0 : _si_value(:lambda_center, kwargs)
    dispersion_raw = no_dispersion ? 0.0 : _si_value(:dispersion, kwargs)
    npixels_raw = no_dispersion ? 0 : kwargs[:npixels]
    reference_pixel_raw = no_dispersion ? 0.0 : kwargs[:reference_pixel]

    _reject_extra_kwargs(kwargs, (:theta_deg, :theta_rad, :lambda_center_nm, :lambda_center_m,
        :dispersion_nm_px, :dispersion_m_px, :npixels, :reference_pixel))

    0 <= theta_raw <= π || throw(ArgumentError(
        "Collection: theta must be between 0 and pi radians; got $theta_raw rad"))

    FT = float(promote_type(typeof(theta_raw), typeof(lambda_center_raw), typeof(dispersion_raw),
        typeof(float(reference_pixel_raw))))
    return Collection{FT}(FT(theta_raw), k_model, FT(lambda_center_raw), FT(dispersion_raw),
        Int(npixels_raw), FT(reference_pixel_raw))
end

"""
    wavevector(probe::ProbeBeam, collection::Collection, lambda_s, plasma::Plasma) -> (k, omega, k_s)

The probed wavevector magnitude, frequency shift and scattered wavenumber for one scattered
wavelength `lambda_s` (m), from the light-wave dispersion relation:

    ω_i   = 2πc/λ_i,             λ_i = probe.lambda
    ω_s   = 2πc/λ_s,             ω = ω_s − ω_i               (ω > 0 is a blue shift)
    ω_pe² = n_e e²/(ε₀ m_e),     n_e = plasma.n_e             (the TOTAL electron density)
    k_i   = √(ω_i² − ω_pe²)/c,   k_s = √(ω_s² − ω_pe²)/c
    k     = √(k_i² + k_s² − 2 k_i k_s cos θ)                  (`k_model = :exact`, default)
    k     = (4π/λ_i) sin(θ/2)                                 (`k_model = :approximate`, the
                                                                **vacuum** k_i = 2π/λ_i)

`k`, `k_s` are in m⁻¹, `omega` in rad/s. `k_s` is the plasma-corrected scattered wavenumber
regardless of `k_model` (it is already computed for the exact form; Step 2.3b's convective gain
reuses it, spec D41). Throws if `λ_i` or `λ_s` is at or beyond the plasma cutoff (`ω ≤ ω_pe`), or
if the resulting `k` is zero (only at `θ = 0` with `λ_s = λ_i`).

Sheffield Eqs. 5.4.1-5.4.3; `conventions.md` 3.3 ([OQ-8](../../dev/open_questions.md#oq-8));
`docs/derivations/geometry.md`; spec D41.
"""
function wavevector(probe::ProbeBeam, collection::Collection, lambda_s, plasma::Plasma)
    lambda_i = probe.lambda
    omega_probe = 2π * SPEED_OF_LIGHT / lambda_i
    omega_s = 2π * SPEED_OF_LIGHT / lambda_s
    omega = omega_s - omega_probe

    omega_pe_sq = plasma.n_e * ELEMENTARY_CHARGE^2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS)
    omega_pe = sqrt(omega_pe_sq)
    cutoff = 2π * SPEED_OF_LIGHT / omega_pe

    if !(omega_probe > omega_pe)
        throw(ArgumentError("wavevector: the probe wavelength $(@sprintf("%e", lambda_i)) m is at " *
            "or beyond the plasma cutoff $(@sprintf("%e", cutoff)) m (omega_probe <= omega_pe) " *
            "for n_e = $(@sprintf("%e", plasma.n_e)) m^-3"))
    end
    if !(omega_s > omega_pe)
        throw(ArgumentError("wavevector: the scattered wavelength $(@sprintf("%e", lambda_s)) m is at " *
            "or beyond the plasma cutoff $(@sprintf("%e", cutoff)) m (omega_s <= omega_pe) " *
            "for n_e = $(@sprintf("%e", plasma.n_e)) m^-3"))
    end

    k_i = sqrt(omega_probe^2 - omega_pe_sq) / SPEED_OF_LIGHT
    k_s = sqrt(omega_s^2 - omega_pe_sq) / SPEED_OF_LIGHT

    if collection.k_model === :exact
        k = sqrt(k_i^2 + k_s^2 - 2 * k_i * k_s * cos(collection.theta))
        if iszero(k)
            throw(ArgumentError("wavevector: k = 0 at theta = $(collection.theta) rad with lambda_s " *
                "equal to the probe wavelength; this geometry probes no wavevector"))
        end
    else
        s = sin(collection.theta / 2)
        if iszero(s)
            throw(ArgumentError("wavevector: k = 0 at theta = $(collection.theta) rad with lambda_s " *
                "equal to the probe wavelength; this geometry probes no wavevector"))
        end
        # oftype(k_i, ...), not a bare 4π*s/lambda_i: k_i already carries whatever (possibly dual)
        # type plasma.n_e promoted omega_pe_sq to, even when n_e's own partials are all zero (every
        # field of a widened Plasma is promoted together, spec D12). Without this, the two branches
        # of this if/else return different concrete types for k depending on the runtime value of
        # collection.k_model, which Julia cannot discharge statically -- k's inferred type becomes a
        # non-bits Union and every call boxes its return. Found chasing Step 3.2b's JT10/JT11 (the
        # per-bin dual pass was allocating ~224 B per wavelength bin with k_model = :exact, which
        # both P2 and P3 use); fixed here since it is invisible in the plain pass (n_e is always
        # Float64 there) and would otherwise recur for every future AD path through this function.
        k = oftype(k_i, 4π * s / lambda_i)
    end

    return k, omega, k_s
end

"""
    polarization_factor(probe::ProbeBeam, collection::Collection) -> FT

The polarization factor `|ŝ × (ŝ × Ê_i0)|² = 1 − sin²θ sin²ψ`, with `ψ = probe.polarization` the
angle of the incident polarization from the normal to the scattering plane. `ψ = 0` (the usual
arrangement) gives 1; `ψ = π/2` gives `cos²θ`. Never multiplied into `spectrum`'s own result,
since the sensitivity absorbs it (`conventions.md` 1.8) — but it is **load-bearing** whenever
`spectrum`'s `gain` is `:convective` or `:weak` (Step 2.3b, spec D43), since Turnbull's projected
vector potential is `a0_sq * polarization_factor(probe, collection)` exactly.

Sheffield Eq. 2.3.13; `conventions.md` 1.8; `docs/derivations/geometry.md`; spec D41.
"""
function polarization_factor(probe::ProbeBeam, collection::Collection)
    return 1 - sin(collection.theta)^2 * sin(probe.polarization)^2
end

"""
    kdirection(probe::ProbeBeam, collection::Collection) -> NTuple{3,FT}

The mean **k̂** of `conventions.md` 3.2 (one direction for the whole spectrum), as a unit vector
in the canonical frame of `docs/derivations/geometry.md`: the probe propagates along **+ẑ**, the
scattered direction lies in the x–z plane at angle `θ` from **+ẑ**, and

    k̂ = (k_s ŝ − k_i ẑ) / |k_s ŝ − k_i ẑ|      evaluated at λ_s = λ_probe (so k_s = k_i, and the
                                                 magnitudes cancel: kdirection needs no plasma)

so a user can print the direction their flows are quoted along instead of reading a figure.

`conventions.md` 3.2; `docs/derivations/geometry.md`; spec D41.
"""
function kdirection(probe::ProbeBeam, collection::Collection)
    theta = collection.theta
    FT = promote_type(typeof(probe.lambda), typeof(theta))
    dx = sin(theta)
    dz = cos(theta) - one(FT)
    n = sqrt(dx^2 + dz^2)
    return (dx / n, zero(FT), dz / n)
end

"""
    detector_axis(collection::Collection) -> Vector

The detector's wavelength axis from `collection`'s dispersion, `λ[p] = lambda_center +
dispersion * (p - reference_pixel)` for pixel `p` in `1:npixels`, **always returned increasing**:
since `dispersion` is signed (OMEGA's IAW axis runs backwards, `-0.00444` nm/px), a negative
dispersion gives an axis reversed relative to pixel order, and `read_streak`/the dewarp step
(Step 2.4) reverse the image's wavelength dimension the same way whenever the dispersion that
generated the axis is negative — so a feature lands at the same wavelength whichever sign the
dispersion has (`conventions.md` 3.4). Throws if `collection` carries no dispersion.

`conventions.md` 3.4; `docs/derivations/geometry.md`; spec D44.
"""
function detector_axis(collection::Collection)
    iszero(collection.dispersion) && throw(ArgumentError(
        "detector_axis: collection has no dispersion (dispersion == 0)"))
    collection.npixels > 0 || throw(ArgumentError(
        "detector_axis: collection has no pixel count (npixels == 0)"))
    n = collection.npixels
    lam = [collection.lambda_center + collection.dispersion * (p - collection.reference_pixel)
           for p in 1:n]
    return collection.dispersion > 0 ? lam : reverse(lam)
end

# --- show methods (spec D29) ---

function Base.show(io::IO, p::ProbeBeam)
    lam = round(lambda_from_si(p.lambda, :lambda_nm); sigdigits=12)
    print(io, "ProbeBeam($lam nm)")
end

function Base.show(io::IO, ::MIME"text/plain", p::ProbeBeam)
    println(io, "ProbeBeam")
    @printf(io, "  lambda_probe = %e nm\n", lambda_from_si(p.lambda, :lambda_nm))
    @printf(io, "  polarization = %e deg\n", theta_from_si(p.polarization, :theta_deg))
    @printf(io, "  a0_sq        = %e\n", p.a0_sq)
    @printf(io, "  diameter     = %e um\n", p.diameter / 1e-6)
end

function Base.show(io::IO, c::Collection)
    th = round(theta_from_si(c.theta, :theta_deg); sigdigits=12)
    print(io, "Collection($th deg, :$(c.k_model))")
end

function Base.show(io::IO, ::MIME"text/plain", c::Collection)
    println(io, "Collection")
    @printf(io, "  theta   = %e deg\n", theta_from_si(c.theta, :theta_deg))
    println(io, "  k_model = :$(c.k_model)")
    if !iszero(c.dispersion)
        @printf(io, "  lambda_center   = %e nm\n", c.lambda_center / 1e-9)
        @printf(io, "  dispersion      = %e nm/px\n", c.dispersion / 1e-9)
        println(io, "  npixels         = $(c.npixels)")
    end
end
