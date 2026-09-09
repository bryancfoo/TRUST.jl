using Test
using OTIS
using ForwardDiff

@testset "S1: T boundary round trip" begin
    sp1 = electrons(T_eV = 500)
    sp2 = electrons(T_keV = 0.5)
    @test isapprox(sp1.T, 500 * EV; rtol = 1e-15)
    @test isapprox(sp1.T, sp2.T; rtol = 1e-15)
end

@testset "S2: u boundary round trip and default" begin
    sp1 = electrons(T_eV = 500, u_kms = 100)
    sp2 = electrons(T_eV = 500, u_ms = 1e5)
    @test isapprox(sp1.u, sp2.u; rtol = 1e-15)
    sp3 = electrons(T_eV = 500)
    @test sp3.u == 0
end

@testset "S3: ion mass table value" begin
    sp = ion(mass_amu = 2.013553212745, charge = 1, T_eV = 300)
    @test isapprox(sp.mass, 2.013553212745 * ATOMIC_MASS_UNIT; rtol = 1e-15)
end

@testset "S4: electrons() defaults" begin
    sp = electrons(T_eV = 500)
    @test sp.charge == -1
    @test sp.mass === ELECTRON_MASS
end

@testset "S5: T keyword errors" begin
    @test_throws "no unit keyword given for T; expected exactly one of (:T_eV, :T_keV)" electrons()
    @test_throws "more than one unit keyword given for T: (:T_eV, :T_keV); expected exactly one" electrons(T_eV = 500, T_keV = 0.5)
    @test_throws "unrecognised keyword(s) (:T_ev,); accepted keywords are (:name, :T_eV, :T_keV, :u_kms, :u_ms, :f, :fraction)" electrons(T_eV = 500, T_ev = 500)
end

@testset "S6: Int/Float64 promotion" begin
    sp = ion(mass_amu = 2, charge = 1, T_eV = 300, fraction = 1)
    @test sp isa SpeciesSpec{Float64, Maxwellian}
end

@testset "S7: Dual promotion, charge stays Int" begin
    d = ForwardDiff.Dual(300.0, 1.0)
    sp = ion(mass_amu = 2.0, charge = 1, T_eV = d, fraction = 1.0)
    @test sp isa SpeciesSpec{typeof(d), Maxwellian}
    @test sp.charge isa Int
end

@testset "S8: species name validation" begin
    for bad in ("", "D 2", "a.b", "a/b", "C6+", "names", "species")
        @test_throws "invalid species name" ion(name = bad, mass_amu = 2, charge = 1, T_eV = 300)
    end
    for good in ("D", "C6", "e_cold", "e-hot")
        sp = ion(name = good, mass_amu = 2, charge = 1, T_eV = 300)
        @test sp.name == good
    end
end

@testset "S9: SpeciesSet accessors" begin
    d = ion(name = "D", mass_amu = 2, charge = 1, T_eV = 300, fraction = 0.6)
    c = ion(name = "C", mass_amu = 12, charge = 6, T_eV = 300, fraction = 0.4)
    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)], ions = [d, c])
    ions = p.ions

    @test ions[1] === ions["D"]
    @test ions[2] === ions["C"]
    @test ions.D === ions["D"]
    @test keys(ions) == ("D", "C")
    @test length(ions) == 2
    @test collect(ions) == [ions[1], ions[2]]
    @test Set(propertynames(ions)) == Set((:names, :species, :D, :C))
    @test ions.names == ("D", "C")
end

@testset "S10: show" begin
    sp = ion(name = "D", mass_amu = 2.013553212745, charge = 1, T_eV = 300)
    @test sprint(show, sp) == "SpeciesSpec(\"D\")"

    p = Plasma(n_e_cm3 = 1e19, electrons = [electrons(T_eV = 500)],
               ions = [ion(name = "D", mass_amu = 2.013553212745, charge = 1, T_eV = 300, fraction = 0.6),
                       ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300, fraction = 0.4)])
    d = p.ions["D"]
    @test sprint(show, d) == "Species(\"D\")"
    @test sprint(show, p.ions) == "SpeciesSet(\"D\", \"C\")"
    @test sprint(show, p) == "Plasma(1 electron species, 2 ion species)"

    text = sprint(io -> show(io, MIME("text/plain"), d))
    @test occursin("eV", text)
    @test occursin("cm^-3", text)
    @test occursin("km/s", text)
    @test occursin("amu", text)
    @test !occursin(string(d.T), text)
end

@testset "HK3: a dual shape parameter makes the plasma dual with zero-partial scalars (Step 2.0)" begin
    # OQ-9 trap 6, decided by Level 1 2026-09-09 (option (b), at the constructors): electrons()
    # and ion() now promote FT over values(parameters(f)) as well as the scalar fields, so a
    # SuperGaussian with a dual m makes the whole plasma dual, with the scalar fields carrying
    # an exact zero partial, with no companion seeding needed on any scalar keyword.
    m_dual = ForwardDiff.Dual{:HK}(3.5, 1.0)
    p = Plasma(n_e_cm3 = 5e19,
               electrons = (electrons(T_eV = 500.0, f = SuperGaussian(m_dual)),),
               ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))
    FT = typeof(p.n_e)
    @test FT <: ForwardDiff.Dual
    @test p.n_e isa FT
    for sp in (p.electrons[1], p.ions[1])
        for field in (sp.mass, sp.u, sp.T, sp.fraction, sp.n)
            @test field isa FT
            @test iszero(ForwardDiff.partials(field))
        end
    end
end

@testset "S11: non-positive T, mass, and zero ion charge" begin
    @test_throws "temperature must be positive; got" electrons(T_eV = -1.0)
    @test_throws "for species \"e\"" electrons(T_eV = -1.0)
    @test_throws "mass must be positive; got 0.0 kg for species \"D\"" ion(name = "D", mass_kg = 0.0, charge = 1, T_eV = 300)
    @test_throws "temperature must be positive" ion(name = "D", mass_amu = 2, charge = 1, T_eV = -1)
    @test_throws "ion \"D\" has charge 0" ion(name = "D", mass_amu = 2, charge = 0, T_eV = 300)
    @test_throws "ion \"D\" has charge -1; an ion's charge number must be a positive integer" ion(name = "D", mass_amu = 2, charge = -1, T_eV = 300)
end
