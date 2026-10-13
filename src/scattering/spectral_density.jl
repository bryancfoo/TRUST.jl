# The spectral density function S(k, omega). conventions.md 1.7; docs/derivations/spectral_density.md;
# spec D10, D39.

"""
    spectral_density(plasma::Plasma, k, omega) -> FT
    spectral_density(plasma::Plasma, k, omega, d) -> FT

The spectral density function

    S(k, ω) = (2π/k) [ |1 − χ_e/ε|² Σ_j (n_ej/n_e) f_lab(sp_ej, ω/k)
                     + |χ_e/ε|²    Σ_i (Z_i² n_i/n_e) f_lab(sp_i, ω/k) ],

summed over every electron species j and every ion species i, with `ε` and `χ_e` the fields of
`d = dielectric(plasma, k, omega)`. `k` is in m⁻¹, `omega` in rad/s; the result is in
**seconds**. The three-argument form computes `d` itself and calls the four-argument one; the
four-argument form is what `_spectrum_at` calls, with `d` computed once per (bin, wavelength) and
threaded down so the convective gain of Step 2.3b can reuse it rather than calling `dielectric` a
second time (spec D39). `spectral_density` stays the pure structure factor of rule 1.7 either
way — the sum-rule test of `conventions.md` 8.5 depends on it.

`Z_i` is the ion species' `charge`, a real number since Phase 4.J. With charge fractions its weight
`Z_i² n_i / n_e` is `Z_i · fraction_i`, as is its share of `ε`, so charge states of one element
sharing T, u, mass and shape are reproduced exactly by one species of charge ⟨Z²⟩/⟨Z⟩ and their summed
fraction; that is what a non-integer charge is (`conventions.md` 1.7,
[OQ-60](../../dev/open_questions.md#oq-60)).

Sheffield Eq. 5.1.2 (one electron, one ion species), generalized to any number of species by
`conventions.md` 1.7 ([OQ-6](../../dev/open_questions.md#oq-6));
`docs/derivations/spectral_density.md`.
"""
spectral_density(plasma::Plasma, k, omega) = spectral_density(plasma, k, omega, dielectric(plasma, k, omega))

function spectral_density(plasma::Plasma, k, omega, d)
    v = omega / k
    n_e = plasma.n_e

    electron_term = sum(sp -> (sp.n / n_e) * f_lab(sp, v), plasma.electrons)
    ion_term = sum(sp -> (sp.charge^2 * sp.n / n_e) * f_lab(sp, v), plasma.ions;
                   init = zero(electron_term))

    screened_electron = abs2(one(d.epsilon) - d.chi_e / d.epsilon)
    screened_ion = abs2(d.chi_e / d.epsilon)

    return (2π / k) * (screened_electron * electron_term + screened_ion * ion_term)
end
