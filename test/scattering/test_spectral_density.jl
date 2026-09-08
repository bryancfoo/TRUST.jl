using Test
using OTIS
using SpecialFunctions: dawson
using QuadGK: quadgk

# Independent oracle, never through Z or Zprime or susceptibility (conventions.md 8.2 / OQ-23).
Rw(x) = 1 - 2x * dawson(x)                 # Sheffield 5.2.7
Iw(x) = sqrt(pi) * x * exp(-x^2)           # Sheffield 5.2.8

reference_plasma() = Plasma(
    n_e_cm3 = 1e19,
    electrons = (electrons(T_eV = 500),),
    ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))

# Sheffield Eqs. 5.2.2-5.2.4, transcribed independently from spec Section 4.
function S_ref(plasma, k, omega)
    e = plasma.electrons["e"]
    ionsp = plasma.ions["i"]
    a = thermal_speed(e)
    b = thermal_speed(ionsp)
    alpha_e = 1 / (k * debye_length(e))
    x_e = omega / (k * a)
    x_i = omega / (k * b)
    Zion = ionsp.charge
    r = Zion * e.T / ionsp.T

    Ae = exp(-x_e^2) * ((1 + alpha_e^2 * r * Rw(x_i))^2 + (alpha_e^2 * r * Iw(x_i))^2)
    Ai = Zion * sqrt(ionsp.mass * e.T / (e.mass * ionsp.T)) * exp(-x_i^2) *
         ((alpha_e^2 * Rw(x_e))^2 + (alpha_e^2 * Iw(x_e))^2)
    eps2 = (1 + alpha_e^2 * (Rw(x_e) + r * Rw(x_i)))^2 +
           (alpha_e^2 * Iw(x_e) + alpha_e^2 * r * Iw(x_i))^2

    return (2 * sqrt(pi) / (k * a)) * (Ae + Ai) / eps2
end

function plasma_for(alpha_target, r_target; T_eV = 500.0, n_e_cm3 = 1e19, mass_amu = 2.014, Zion = 1)
    Ti_eV = Zion * T_eV / r_target
    p0 = Plasma(n_e_cm3 = n_e_cm3, electrons = (electrons(T_eV = T_eV),),
                ions = (ion(mass_amu = mass_amu, charge = Zion, T_eV = Ti_eV),))
    e = p0.electrons["e"]
    k = 1 / (alpha_target * debye_length(e))
    return p0, k
end

# Sheffield Eqs. 5.3.5 (Bohm-Gross) and 5.3.9, in SI, for the sum-rule breakpoints (spec D20).
function omega_resonances(p, k)
    e = p.electrons["e"]
    ionsp = p.ions["i"]
    wpe = plasma_frequency(e)
    lamDe = debye_length(e)
    w_epw = sqrt(wpe^2 + 3 * k^2 * e.T / e.mass)
    w_ia = k * sqrt(ionsp.charge * e.T / (ionsp.mass * (1 + k^2 * lamDe^2)) + 3 * ionsp.T / ionsp.mass)
    return w_epw, w_ia
end

function sum_rule_breakpoints(p, k)
    e = p.electrons["e"]
    wmax = 8 * k * thermal_speed(e)
    w_epw, w_ia = omega_resonances(p, k)
    bp = Float64[-wmax, wmax]
    for w in (-w_epw, -w_ia, 0.0, w_ia, w_epw)
        -wmax < w < wmax && push!(bp, w)
    end
    return sort!(unique!(bp))
end

function contour_integrand(p, k, omega, alpha_e)
    ionsp = p.ions["i"]
    d = dielectric(p, k, omega)
    chi_i = susceptibility(ionsp, k, omega)
    return (2 / (alpha_e^2 * omega)) * imag((1 + chi_i) * d.chi_e / d.epsilon)
end

# The peak method of spec D23: parabolic offset from the grid maximum.
function peak_position(x, y)
    i = argmax(y)
    (i == 1 || i == length(y)) && return x[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    h = x[i] - x[i - 1]
    d = (y0 - y2) / (2 * (y0 - 2y1 + y2))
    return x[i] + d * h
end

@testset "SD1: S vs Sheffield 5.2.2-5.2.4" begin
    for alpha_target in (0.1, 0.3, 1.0, 2.0, 3.0, 5.0), r in (0.1, 0.5, 1.0, 3.0, 10.0)
        p, k = plasma_for(alpha_target, r)
        a = thermal_speed(p.electrons["e"])
        for x_e in (0.0, 0.3, -0.3, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0)
            omega = x_e * k * a
            @test isapprox(spectral_density(p, k, omega), S_ref(p, k, omega); rtol = 1e-10)
        end
    end
end

@testset "SD2: sum rule 1, Salpeter total at Te=Ti" begin
    for (mass_amu, Zion) in ((2.014, 1), (12.011, 6)), alpha_target in (0.05, 0.3, 1.0, 2.0, 3.0)
        T_eV = 500.0
        p0 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = T_eV),),
                    ions = (ion(mass_amu = mass_amu, charge = Zion, T_eV = T_eV),))
        k = 1 / (alpha_target * debye_length(p0.electrons["e"]))
        bp = sum_rule_breakpoints(p0, k)
        val, _ = quadgk(omega -> spectral_density(p0, k, omega), bp...; rtol = 1e-10, atol = 1e-14)
        expected = 2π * (1 + alpha_target^2 * Zion) / (1 + alpha_target^2 + alpha_target^2 * Zion)
        @test isapprox(val, expected; rtol = 1e-6)
    end
end

@testset "SD3: sum rule 2, the contour identity" begin
    configs = ((1, 2.014, 1.0), (1, 2.014, 4.0), (6, 12.011, 2.0))
    for alpha_e in (0.3, 1.0, 2.0, 3.0), (Zion, mass_amu, r) in configs
        T_eV = 500.0
        Ti_eV = Zion * T_eV / r
        p0 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = T_eV),),
                    ions = (ion(mass_amu = mass_amu, charge = Zion, T_eV = Ti_eV),))
        k = 1 / (alpha_e * debye_length(p0.electrons["e"]))
        bp = sum_rule_breakpoints(p0, k)
        val, _ = quadgk(omega -> contour_integrand(p0, k, omega, alpha_e), bp...; rtol = 1e-10, atol = 1e-14)
        alpha_i2 = alpha_e^2 * r
        expected = 2π * (1 + alpha_i2) / (1 + alpha_e^2 + alpha_i2)
        @test isapprox(val, expected; rtol = 1e-6)
    end
end

@testset "SD4: noncollective limit" begin
    T_eV = 500.0
    p0 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = T_eV),),
                ions = (ion(mass_amu = 2.014, charge = 1, T_eV = T_eV),))
    e = p0.electrons["e"]
    k = 1 / (0.01 * debye_length(e))
    a = thermal_speed(e)
    for x_e in (0.0, 0.3, -0.3, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0)
        omega = x_e * k * a
        S = spectral_density(p0, k, omega)
        ref = (2π / k) * f_lab(e, omega / k)
        @test isapprox(S, ref; rtol = 3e-4)
    end
end

@testset "SD5: species-splitting invariance, electrons" begin
    ionspec = ion(mass_amu = 2.014, charge = 1, T_eV = 250)
    p1 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),), ions = (ionspec,))
    p2 = Plasma(n_e_cm3 = 1e19,
                electrons = (electrons(name = "e1", T_eV = 500, fraction = 0.4),
                             electrons(name = "e2", T_eV = 500, fraction = 0.6)),
                ions = (ionspec,))
    k, omega = 1e6, 1e11
    @test isapprox(spectral_density(p1, k, omega), spectral_density(p2, k, omega); rtol = 1e-14)
end

@testset "SD6: species-splitting invariance, ions" begin
    espec = electrons(T_eV = 500)
    p1 = Plasma(n_e_cm3 = 1e19, electrons = (espec,),
                ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
    p2 = Plasma(n_e_cm3 = 1e19, electrons = (espec,),
                ions = (ion(name = "D1", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = 0.25),
                        ion(name = "D2", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = 0.75)))
    k, omega = 1e6, 1e11
    @test isapprox(spectral_density(p1, k, omega), spectral_density(p2, k, omega); rtol = 1e-14)
end

@testset "SD7: Salpeter resonances" begin
    p0 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),),
                ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 50),))    # Z Te/Ti = 10
    k = 1 / (3.0 * debye_length(p0.electrons["e"]))                        # alpha = 3
    w_epw, w_ia = omega_resonances(p0, k)

    epw_grid = collect(range(0.6 * w_epw, 1.4 * w_epw; length = 4001))
    S_epw = [spectral_density(p0, k, w) for w in epw_grid]
    @test isapprox(peak_position(epw_grid, S_epw), w_epw; rtol = 0.05)

    ia_grid = collect(range(0.5 * w_ia, 1.8 * w_ia; length = 4001))
    S_ia = [spectral_density(p0, k, w) for w in ia_grid]
    @test isapprox(peak_position(ia_grid, S_ia), w_ia; rtol = 0.05)
end

@testset "SD8: S is strictly positive" begin
    p0 = reference_plasma()
    k = 1e6
    a = thermal_speed(p0.electrons["e"])
    for omega in range(-5 * k * a, 5 * k * a; length = 21)
        @test spectral_density(p0, k, omega) > 0
    end
end

@testset "SD9: allocations" begin
    p1 = reference_plasma()
    p2 = Plasma(n_e_cm3 = 1e19,
                electrons = (electrons(name = "e1", T_eV = 500, fraction = 0.5),
                             electrons(name = "e2", T_eV = 800, fraction = 0.5)),
                ions = (ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = 0.5),
                        ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300, fraction = 0.5)))
    k, omega = 1e6, 1e11
    _sd1() = spectral_density(p1, k, omega)
    _sd2() = spectral_density(p2, k, omega)
    _sd1()
    _sd2()
    @test (@allocated _sd1()) == 0
    @test (@allocated _sd2()) == 0
end
