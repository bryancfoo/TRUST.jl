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
