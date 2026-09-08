# The spectral density function S(k, omega). conventions.md 1.7; docs/derivations/spectral_density.md;
# spec D10.

"""
    spectral_density(plasma::Plasma, k, omega) -> FT

The spectral density function

    S(k, ω) = (2π/k) [ |1 − χ_e/ε|² Σ_j (n_ej/n_e) f_lab(sp_ej, ω/k)
                     + |χ_e/ε|²    Σ_i (Z_i² n_i/n_e) f_lab(sp_i, ω/k) ],

summed over every electron species j and every ion species i, with `ε` and `χ_e` from one
`dielectric` call. `k` is in m⁻¹, `omega` in rad/s; the result is in **seconds**.

Sheffield Eq. 5.1.2 (one electron, one ion species), generalized to any number of species by
`conventions.md` 1.7 ([OQ-6](../../dev/open_questions.md#oq-6));
`docs/derivations/spectral_density.md`.
"""
function spectral_density(plasma::Plasma, k, omega)
    d = dielectric(plasma, k, omega)
    v = omega / k
    n_e = plasma.n_e

    electron_term = sum(sp -> (sp.n / n_e) * f_lab(sp, v), plasma.electrons)
    ion_term = sum(sp -> (sp.charge^2 * sp.n / n_e) * f_lab(sp, v), plasma.ions;
                   init = zero(electron_term))

    screened_electron = abs2(one(d.epsilon) - d.chi_e / d.epsilon)
    screened_ion = abs2(d.chi_e / d.epsilon)

    return (2π / k) * (screened_electron * electron_term + screened_ion * ion_term)
end
