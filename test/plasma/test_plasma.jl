using Test
using OTIS
using ForwardDiff

reference_plasma() = Plasma(
    n_e_cm3 = 1e19,
    electrons = [electrons(T_eV = 500)],
    ions = [ion(name = "D", mass_amu = 2.013553212745, charge = 1, T_eV = 300, fraction = 0.6),
            ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300, fraction = 0.4)])

@testset "Q1: charge-fraction ion densities" begin
    p = reference_plasma()
    @test isapprox(p.ions["D"].n, 0.6 * p.n_e / 1; rtol = 1e-15)
    @test isapprox(p.ions["C"].n, 0.4 * p.n_e / 6; rtol = 1e-15)
end

@testset "Q2: number-fraction electron density" begin
    p = reference_plasma()
    @test isapprox(p.electrons["e"].n, p.n_e; rtol = 1e-15)
end

@testset "Q3: neutrality residual is zero" begin
    p = reference_plasma()
    @test isapprox(neutrality_residual(p), 0; atol = 1e-14)
end

@testset "Q4: n_e is the constructor value, not recomputed" begin
    p = reference_plasma()
    @test p.n_e == n_to_si(1e19, :n_cm3)
end

@testset "Q5: lookup by string, position and property agree" begin
    p = reference_plasma()
    @test p.ions["D"] === p.ions[1] === p.ions.D
end

@testset "Q6: spec round-trips" begin
    p = reference_plasma()
    d = p.ions["D"]
    s = spec(d)
    @test s.name == d.name
    @test s.mass == d.mass
    @test s.charge == d.charge
    @test s.u == d.u
    @test s.T == d.T
    @test s.f == d.f
    @test s.fraction == d.fraction
end

@testset "Q7: validation errors" begin
    @test_throws "duplicate species name \"D\" in ions" Plasma(n_e_cm3 = 1e19,
        electrons = [electrons(T_eV = 500)],
        ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5),
                ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5)])

    @test_throws "not 1, within 1e-8" Plasma(n_e_cm3 = 1e19,
        electrons = [electrons(T_eV = 500)],
        ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5),
                ion(name = "C", mass_amu = 12, charge = 6, T_eV = 300, fraction = 0.45)])

    @test_throws "is outside [0, 1]" Plasma(n_e_cm3 = 1e19,
        electrons = [electrons(T_eV = 500)],
        ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = -0.1)])

    @test_throws "a Plasma needs at least one electron species" Plasma(n_e_cm3 = 1e19, electrons = SpeciesSpec[])
end

@testset "Q8: fraction sum tolerance boundary" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
               ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5 + 2.5e-9),
                       ion(name = "C", mass_amu = 12, charge = 6, T_eV = 300, fraction = 0.5 + 2.5e-9)])
    @test p isa Plasma

    @test_throws "not 1, within 1e-8" Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
        ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5 + 2.5e-8),
                ion(name = "C", mass_amu = 12, charge = 6, T_eV = 300, fraction = 0.5 + 2.5e-8)])
end

struct _Q9Shape <: OTIS.AbstractVelocityDistribution end
(::_Q9Shape)(x) = exp(-x^2) / sqrt(pi)

@testset "Q9: mixed shape types give a concrete Plasma type" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
               ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5, f = Maxwellian()),
                       ion(name = "C", mass_amu = 12, charge = 6, T_eV = 300, fraction = 0.5, f = _Q9Shape())])
    @test isconcretetype(typeof(p))
end

@testset "Q10: empty ions" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)])
    @test length(p.ions) == 0
end

@testset "Q11: mean_charge_state and mean_ion_mass" begin
    p = reference_plasma()
    nD, nC = p.ions["D"].n, p.ions["C"].n
    mD, mC = p.ions["D"].mass, p.ions["C"].mass
    zD, zC = p.ions["D"].charge, p.ions["C"].charge
    @test isapprox(mean_charge_state(p), (nD * zD + nC * zC) / (nD + nC); rtol = 1e-14)
    @test isapprox(mean_ion_mass(p), (nD * mD + nC * mC) / (nD + nC); rtol = 1e-14)
end

@testset "Q12: mean_charge_state / mean_ion_mass with no ions throws" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)])
    @test_throws "mean_charge_state is undefined for a plasma with no ion species" mean_charge_state(p)
    @test_throws "mean_charge_state is undefined for a plasma with no ion species" mean_ion_mass(p)
end

@testset "Q13: vectors vs tuples give equal plasmas" begin
    e = electrons(T_eV = 500)
    d = ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300)
    p1 = Plasma(n_e_cm3 = 1e19, electrons = [e], ions = [d])
    p2 = Plasma(n_e_cm3 = 1e19, electrons = (e,), ions = (d,))
    @test p1.n_e == p2.n_e
    @test p1.electrons.species == p2.electrons.species
    @test p1.ions.species == p2.ions.species
end

@testset "Q14: species-splitting invariance" begin
    p1 = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
                ions = [ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 1.0)])
    p2 = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
                ions = [ion(name = "D1", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5),
                        ion(name = "D2", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.5)])
    @test isapprox(p2.ions["D1"].n + p2.ions["D2"].n, p1.ions["D"].n; rtol = 1e-15)
end

@testset "Q15: allocation of Plasma construction (reported, no assertion)" begin
    _bench() = reference_plasma()
    _bench()
    bytes = @allocated _bench()
    @info "Q15 allocated bytes for Plasma construction" bytes
    @test true
end

@testset "Q16: Plasma show" begin
    p = reference_plasma()
    @test sprint(show, p) == "Plasma(1 electron species, 2 ion species)"
end

@testset "Q17: promotion across species (Float64 electron, Int-built ion)" begin
    e = electrons(T_eV = 500.0)
    i = ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300)
    p = Plasma(n_e_cm3 = 1e19, electrons = [e], ions = [i])
    FT = typeof(p.n_e)
    @test typeof(p.electrons["e"].T) === FT
    @test typeof(p.ions["D"].T) === FT
    @test p.electrons["e"].charge isa Int
    @test p.ions["D"].charge isa Int
end

@testset "Q18: dual promotion across species" begin
    d = ForwardDiff.Dual(300.0, 1.0)
    e = electrons(T_eV = 500.0)
    ion_d = ion(name = "D", mass_amu = 2.0, charge = 1, T_eV = d, fraction = 0.5)
    c = ion(name = "C", mass_amu = 12.0, charge = 6, T_eV = 300.0, fraction = 0.5)
    p = Plasma(n_e_cm3 = 1e19, electrons = [e], ions = [ion_d, c])
    FT = typeof(p.n_e)
    @test FT === typeof(d)
    @test typeof(p.electrons["e"].T) === FT
    @test typeof(p.ions["D"].T) === FT
    @test typeof(p.ions["C"].T) === FT
    @test isconcretetype(typeof(p))
end
