using Test
using OTIS
using SpecialFunctions: dawson
using ForwardDiff

# Independent oracle, never through Z or Zprime (conventions.md 8.2 / OQ-23 rule 1).
Rw(x) = 1 - 2x * dawson(x)                 # Sheffield 5.2.7
Iw(x) = sqrt(pi) * x * exp(-x^2)           # Sheffield 5.2.8

reference_plasma() = Plasma(
    n_e_cm3 = 1e19,
    electrons = (electrons(T_eV = 500),),
    ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))

const X_GRID = (-4.0, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, 4.0)
const K_GRID = (1e5, 1e6, 5e6, 2e7)

@testset "CH1: chi_e vs Sheffield 5.2.5" begin
    p = reference_plasma()
    e = p.electrons["e"]
    for k in K_GRID, x_e in X_GRID
        omega = x_e * k * thermal_speed(e)
        chi = susceptibility(e, k, omega)
        ref = alpha(e, k)^2 * (Rw(x_e) + im * Iw(x_e))
        @test isapprox(chi, ref; rtol = 1e-12)
    end
end

@testset "CH2: chi_i vs Sheffield 5.2.6" begin
    p = reference_plasma()
    e = p.electrons["e"]
    ionsp = p.ions["i"]
    for k in K_GRID, x_i in X_GRID
        omega = x_i * k * thermal_speed(ionsp)
        chi = susceptibility(ionsp, k, omega)
        r = ionsp.charge * e.T / ionsp.T                # Z * T_e/T_i
        ref = alpha(e, k)^2 * r * (Rw(x_i) + im * Iw(x_i))
        @test isapprox(chi, ref; rtol = 1e-12)
    end
end

@testset "CH3: static limit chi_s(0) = alpha^2" begin
    p = reference_plasma()
    for k in (1e5, 1e6, 2e7), sp in (p.electrons["e"], p.ions["i"])
        @test isapprox(susceptibility(sp, k, 0.0), alpha(sp, k)^2; rtol = 1e-14)
    end
end

@testset "CH4: Landau sign" begin
    p = reference_plasma()
    e = p.electrons["e"]
    k = 1e6
    for x_e in 0.1:0.1:5.0
        omega = x_e * k * thermal_speed(e)
        @test imag(susceptibility(e, k, omega)) > 0
    end
end

@testset "CH5: dielectric field identity" begin
    p = reference_plasma()
    e = p.electrons["e"]
    ionsp = p.ions["i"]
    for (k, omega) in ((1e5, 1e11), (1e6, -2e12), (5e6, 3e13), (2e7, 0.0))
        d = dielectric(p, k, omega)
        chi_e_expected = susceptibility(e, k, omega)
        chi_i_expected = susceptibility(ionsp, k, omega)
        @test d.chi_e == chi_e_expected
        @test d.epsilon == 1 + chi_e_expected + chi_i_expected
    end
end

@testset "CH6: dielectric sums a two-electron, two-ion plasma" begin
    p = Plasma(n_e_cm3 = 1e19,
        electrons = (electrons(name = "e1", T_eV = 500, fraction = 0.5),
                     electrons(name = "e2", T_eV = 800, fraction = 0.5)),
        ions = (ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = 0.5),
                ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300, fraction = 0.5)))
    k, omega = 1e6, 5e12
    d = dielectric(p, k, omega)
    chi_e_sum = sum(susceptibility(sp, k, omega) for sp in p.electrons)
    chi_i_sum = sum(susceptibility(sp, k, omega) for sp in p.ions)
    @test isapprox(d.chi_e, chi_e_sum; rtol = 1e-14)
    @test isapprox(d.epsilon, 1 + chi_e_sum + chi_i_sum; rtol = 1e-14)
end

@testset "CH7: dielectric with no ions" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    k, omega = 1e6, 1e11
    d = dielectric(p, k, omega)
    chi_e_expected = susceptibility(p.electrons["e"], k, omega)
    @test d.chi_e == chi_e_expected
    @test d.epsilon == 1 + chi_e_expected
end

@testset "CH8: type stability and allocations" begin
    p = reference_plasma()
    e = p.electrons["e"]
    k, omega = 1e6, 1e11
    _sus() = susceptibility(e, k, omega)
    _die() = dielectric(p, k, omega)
    _sus()
    _die()
    @test (@allocated _sus()) == 0
    @test (@allocated _die()) == 0

    # PF2 (Step 1.6): the same, on a two-electron/two-ion plasma.
    p2 = Plasma(n_e_cm3 = 1e19,
        electrons = (electrons(name = "e1", T_eV = 500, fraction = 0.5),
                     electrons(name = "e2", T_eV = 800, fraction = 0.5)),
        ions = (ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = 0.5),
                ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300, fraction = 0.5)))
    e2 = p2.electrons["e1"]
    _sus2() = susceptibility(e2, k, omega)
    _die2() = dielectric(p2, k, omega)
    _sus2()
    _die2()
    @test (@allocated _sus2()) == 0
    @test (@allocated _die2()) == 0
end

@testset "CH9: AD in omega" begin
    p = reference_plasma()
    e = p.electrons["e"]
    k = 1e6
    for x_e in (0.5, 1.0, 2.0)
        omega0 = x_e * k * thermal_speed(e)
        g(omega) = real(susceptibility(e, k, omega))
        ad = ForwardDiff.derivative(g, omega0)
        h = 1e-5 * omega0
        fd = (g(omega0 + h) - g(omega0 - h)) / (2h)
        @test isapprox(ad, fd; rtol = 1e-8)
    end
end

@testset "CH10: AD in a plasma parameter (T_e)" begin
    k, omega = 1e6, 1e11
    function chi_e_of_Te(Te_eV)
        p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = Te_eV),),
                   ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
        return real(susceptibility(p.electrons["e"], k, omega))
    end
    Te0 = 500.0
    ad = ForwardDiff.derivative(chi_e_of_Te, Te0)
    h = 1e-5 * Te0
    fd = (chi_e_of_Te(Te0 + h) - chi_e_of_Te(Te0 - h)) / (2h)
    @test isapprox(ad, fd; rtol = 1e-8)
end
