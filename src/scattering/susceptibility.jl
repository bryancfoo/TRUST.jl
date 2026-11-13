# The per-species susceptibility and the plasma's longitudinal dielectric function.
# conventions.md 1.6; docs/derivations/susceptibility.md; spec D8, D9.

"""
    susceptibility(sp::Species, k, omega) -> Complex

The species susceptibility

    χ_s(k, ω) = −(α_s² / 2) · 𝒞φ_s(ξ_s),

with `α_s = alpha(sp, k)`, `ξ_s = xi(sp, k, omega)` and `𝒞φ_s` the species' shape's Cauchy
integral, `cauchy_dfdv(sp.f, ξ_s)`. `k` is in m⁻¹, `omega` in rad/s; the result is dimensionless.
For a Maxwellian species this reduces to Sheffield Eq. 5.2.5 (electrons) or Eq. 5.2.6 (ions).

Sheffield Eq. 3.3.9 (Gaussian units) in SI and shape form; `conventions.md` 1.6;
`docs/derivations/susceptibility.md`.
"""
function susceptibility(sp::Species, k, omega)
    alpha_sq = alpha(sp, k)^2   # alpha(sp, k) returns α, not α² (conventions.md 1.6); squared once, here
    xi_s = xi(sp, k, omega)
    return -(alpha_sq / 2) * cauchy_dfdv(sp.f, xi_s)
end

"""
    susceptibility(sp::Species{<:Real, <:AbstractLabDistribution}, k, omega) -> Complex

The susceptibility of a species carrying a lab-frame distribution,

    χ_s(k, ω) = −(ω_ps²/k²) · 𝒞f(ω/k),    ω_ps² = n_s q_s²/(ε₀ m_s),

with `𝒞f` the distribution's Cauchy integral in lab velocity, `cauchy_dfdv(sp.f, ω/k)`. This is the
method above with the thermal speed divided out (for `f(v) = φ((v − u)/v_th)/v_th`, `𝒞f = 𝒞φ/v_th²` and
`α_s²/2 = ω_ps²/(k² v_th²)`), so no temperature and no flow appear. The minus sign, and the absence of a
`/2`, are written once, here (`conventions.md` 1.12; `docs/derivations/gaussian_grid.md` Section 4).
`ω_ps²` is `_plasma_frequency_sq`, linear in the density (`conventions.md` 5.15). `k` in m⁻¹, `omega`
in rad/s; the result is dimensionless.

`conventions.md` 1.12; [OQ-68](../../dev/open_questions.md#oq-68) item 5; Phase G.1 spec D12.
"""
function susceptibility(sp::Species{<:Real, <:AbstractLabDistribution}, k, omega)
    return -(_plasma_frequency_sq(sp) / k^2) * cauchy_dfdv(sp.f, omega / k)
end

"""
    dielectric(plasma::Plasma, k, omega) -> (epsilon = ..., chi_e = ...)

The longitudinal dielectric function `ε = 1 + Σ_s χ_s`, summed over every species, electron and
ion alike, together with the total electron susceptibility `χ_e = Σ_j χ_ej`, summed over
electron species only. `k` is in m⁻¹, `omega` in rad/s; both returned fields are dimensionless
`Complex`.

Sheffield Eq. 3.3.10; `conventions.md` 1.6, Section 4; `docs/derivations/susceptibility.md`.
"""
function dielectric(plasma::Plasma, k, omega)
    chi_e = sum(sp -> susceptibility(sp, k, omega), plasma.electrons)
    chi_i = sum(sp -> susceptibility(sp, k, omega), plasma.ions; init = zero(chi_e))
    epsilon = one(chi_e) + chi_e + chi_i
    return (epsilon = epsilon, chi_e = chi_e)
end
