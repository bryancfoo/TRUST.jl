# The EPW resonance's width against the wavelength pixel: the post-fit check of Phase 4.F
# (Step 4.F.2; OQ-53, the user's decision of 2026-09-30). Spec D13 to D16.
#
# A report, never a model: nothing here is called by measure, the Jacobian or an optimizer.
# fit(deck) calls it once, after its other warnings, on the fitted plasma of every bin.

"""
    RESONANCE_WIDTH_WARN

`1.0`: the width, in wavelength pixels, below which `fit(deck)` warns that a fitted EPW resonance
is narrower than the pixels sampling it (Phase 4.F spec D14; [OQ-53](../../dev/open_questions.md#oq-53),
"about one pixel"). Phase 4.E's B6 threshold: the two aliased fits read 0.20 and 0.27 pixels, every
healthy bin 1.2 or more.
"""
const RESONANCE_WIDTH_WARN = 1.0

"""
    RESONANCE_HALF_WINDOW_PX

`8`: the fine grid around a located resonance spans this many pixels on each side of the pixel
pair where `Re ε` changes sign (spec D13).
"""
const RESONANCE_HALF_WINDOW_PX = 8

"""
    RESONANCE_STEPS_PER_PX

`64`: the fine grid's points per wavelength pixel (spec D13): a step of 1/64 pixel, so a width
of 1 pixel is measured to about 1/64 of itself and the narrowest line Phase 4.E found, 0.20
pixels, spans 13 steps.
"""
const RESONANCE_STEPS_PER_PX = 64

"""
    _epw_resonance_widths(p::Plasma, probe::ProbeBeam, collection::Collection,
                          lambda::AbstractVector, first_order::Bool, gain::Symbol = :none) -> Vector{Float64}

The full width at half maximum, in pixels, of every electron-plasma-wave resonance of `p` inside
the window `lambda` (m, increasing), one entry per resonance found (empty when there is none).
Spec D13. `gain` is the fitted experiment's (Phase 4.H spec D6; `_resonance_widths` passes it), so
the width is measured on the model that was fitted; `:none`, the default, keeps the calls written
before Phase 4.H (RW1, RW2) as they were.

**Where a resonance is.** At each pixel `λ_l`, `(k_l, ω_l) = wavevector(probe, collection, λ_l, p)`
and `ε_l = dielectric(p, k_l, ω_l).epsilon`; an EPW resonance lies between `λ_l` and `λ_{l+1}`
when `Re ε` changes sign there and `|ω| > ω_pe` at both pixels (`ω_pe² = n_e e²/(ε₀ m_e)`). The
second condition keeps the Bohm–Gross branch (`ω² ≈ ω_pe² + 3k²v²`, always above `ω_pe`) and
drops the strongly damped crossing nearer the probe and every ion-acoustic one.

**Its width.** The spectrum `_spectrum_at` on `RESONANCE_STEPS_PER_PX` points per pixel over
`RESONANCE_HALF_WINDOW_PX` pixels either side of the pair (clipped to the window), `Δ = λ_{l+1} −
λ_l`; the maximum `S*` of that grid; on each side the first grid point at or below `S*/2`, the
half-maximum crossing interpolated linearly between it and its neighbour; the width the distance
between the two crossings over `Δ`. A side that never falls to `S*/2` inside the grid gives `Inf`:
the line is at least as wide as the grid, and not narrow.
"""
function _epw_resonance_widths(p::Plasma, probe::ProbeBeam, collection::Collection,
                               lambda::AbstractVector, first_order::Bool, gain::Symbol=:none)
    n = length(lambda)
    widths = Float64[]
    n >= 2 || return widths
    omega_pe = sqrt(p.n_e * ELEMENTARY_CHARGE^2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS))
    re = Vector{Float64}(undef, n)
    om = Vector{Float64}(undef, n)
    for l in 1:n
        k, omega, _ = wavevector(probe, collection, lambda[l], p)
        re[l] = real(dielectric(p, k, omega).epsilon)
        om[l] = omega
    end
    for l in 1:(n-1)
        ((re[l] < 0) != (re[l+1] < 0)) || continue
        (abs(om[l]) > omega_pe && abs(om[l+1]) > omega_pe) || continue
        dl = lambda[l+1] - lambda[l]
        lo = max(lambda[1], lambda[l] - RESONANCE_HALF_WINDOW_PX * dl)
        hi = min(lambda[n], lambda[l+1] + RESONANCE_HALF_WINDOW_PX * dl)
        npts = round(Int, (hi - lo) / dl * RESONANCE_STEPS_PER_PX) + 1
        grid = range(lo, hi; length=npts)
        S = [_spectrum_at(p, probe, collection, x, first_order, gain) for x in grid]
        kmax = argmax(S)
        half = S[kmax] / 2
        a = kmax
        while a > 1 && S[a] > half
            a -= 1
        end
        b = kmax
        while b < npts && S[b] > half
            b += 1
        end
        if S[a] > half || S[b] > half
            push!(widths, Inf)
            continue
        end
        x_left = grid[a] + (half - S[a]) / (S[a+1] - S[a]) * (grid[a+1] - grid[a])
        x_right = grid[b-1] + (S[b-1] - half) / (S[b-1] - S[b]) * (grid[b] - grid[b-1])
        push!(widths, (x_right - x_left) / dl)
    end
    return widths
end

"""
    _subsamples(effect::AbstractInstrumentEffect) -> Int

The sub-sample count an effect gives each pixel: `subsamples` for a `PixelIntegration`, `1` for
every other effect (spec D15). A trait, not an `isa` (`conventions.md` Section 4).
"""
_subsamples(::AbstractInstrumentEffect) = 1
_subsamples(e::PixelIntegration) = e.subsamples

"""
    _recommended_subsamples(width_px::Real) -> Int

The count the warning recommends for a resonance `width_px` pixels wide (spec D15): `1` at or
above `RESONANCE_WIDTH_WARN`, otherwise the smallest power of two at or above `2 / width_px`, so
that the sub-sample spacing `Δ/N` is at most half the line's width.
"""
_recommended_subsamples(width_px::Real) =
    width_px >= RESONANCE_WIDTH_WARN ? 1 : Int(nextpow(2, 2 / width_px))
