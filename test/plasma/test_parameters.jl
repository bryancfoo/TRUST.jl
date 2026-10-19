using Test
using TRUST
using QuadGK: quadgk

const P_PLASMA = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)])
const P_E = P_PLASMA.electrons["e"]

@testset "P1: reference case (n_e = 1e19 cm^-3, T_e = 500 eV)" begin
    # Deliberately loose (1e-6 relative): catches a missing factor of 2, a sqrt(2), an
    # eV/keV or cm^-3/m^-3 slip, while surviving a future CODATA revision.
    @test isapprox(P_E.T, 8.010883170e-17; rtol = 1e-6)
    @test isapprox(thermal_speed(P_E), 1.326205115e7; rtol = 1e-6)
    @test isapprox(plasma_frequency(P_E), 1.783986364e14; rtol = 1e-6)
    @test isapprox(debye_length(P_E), 5.256590797e-8; rtol = 1e-6)
end

@testset "P2: deuteron thermal speed at 300 eV" begin
    d = ion(mass_amu = 2.013553212745, charge = 1, T_eV = 300)
    @test isapprox(d.mass, 3.343583777e-27; rtol = 1e-6)
    @test isapprox(thermal_speed(d), 1.695605113e5; rtol = 1e-6)
end

@testset "P3: thermal_speed == sqrt(2) * plasma_frequency * debye_length" begin
    # Tight: pure algebra (omega_pe * lambda_D = sqrt(T/m)); catches the factor of two in v_th.
    @test isapprox(thermal_speed(P_E), sqrt(2) * plasma_frequency(P_E) * debye_length(P_E); rtol = 1e-15)
end

@testset "P4: alpha is 1/(k*debye_length) and dimensionless" begin
    k = 1e5
    @test isapprox(alpha(P_E, k), 1 / (k * debye_length(P_E)); rtol = 1e-15)
end

@testset "P5: alpha_i^2 = alpha_e^2 * Z * Te/Ti" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
               ions = [ion(mass_amu = 2.013553212745, charge = 1, T_eV = 300)])
    e = p.electrons["e"]
    i = p.ions["i"]
    k = 1e5
    lhs = alpha(i, k)^2
    rhs = alpha(e, k)^2 * i.charge * e.T / i.T
    @test isapprox(lhs, rhs; rtol = 1e-14)
end

@testset "P6: xi" begin
    sp = electrons(T_eV = 500, u_kms = 100)
    k = 1e5
    omega = 3e10
    @test isapprox(xi(sp, k, omega), (omega / k - sp.u) / thermal_speed(sp); rtol = 1e-15)
    @test isapprox(xi(sp, k, 0.0), -sp.u / thermal_speed(sp); rtol = 1e-15)
end

@testset "P7-P9: f_lab normalizations, dimensional form" begin
    sp = electrons(T_eV = 500, u_kms = 100)
    u = sp.u
    m0, _ = quadgk(v -> f_lab(sp, v), -Inf, u, Inf; rtol = 1e-12)
    m1, _ = quadgk(v -> v * f_lab(sp, v), -Inf, u, Inf; rtol = 1e-12)
    m2, _ = quadgk(v -> (v - u)^2 * f_lab(sp, v), -Inf, u, Inf; rtol = 1e-12)
    @test isapprox(m0, 1; atol = 1e-10)
    @test isapprox(m1, u; rtol = 1e-10)
    @test isapprox(m2, sp.T / sp.mass; rtol = 1e-10)
end

@testset "P10: SpeciesSpec accepted by thermal_speed, xi, f_lab; not by the others" begin
    sp = electrons(T_eV = 500)
    @test thermal_speed(sp) isa Real
    @test xi(sp, 1e5, 1e10) isa Real
    @test f_lab(sp, 1e5) isa Real
    @test_throws MethodError plasma_frequency(sp)
    @test_throws MethodError debye_length(sp)
    @test_throws MethodError alpha(sp, 1e5)
end

@testset "P11: allocation-free thermal_speed and f_lab" begin
    _bench_ts(sp) = thermal_speed(sp)
    _bench_flab(sp, v) = f_lab(sp, v)
    _bench_ts(P_E)
    _bench_flab(P_E, 1e5)
    @test (@allocated _bench_ts(P_E)) == 0
    @test (@allocated _bench_flab(P_E, 1e5)) == 0
end
