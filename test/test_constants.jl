using Test
using TRUST
using TRUST: ACCEPTED_UNITS, _basefloat
using ForwardDiff: Dual

@testset "boundary units: round trip" begin
    # Tight (1e-14 relative): two floating-point multiplications; anything worse is a table
    # error (spec Section 4, Step 0.2).
    quantities = (:T, :n, :u, :mass, :lambda, :t, :theta)
    # lambda_probe and polarization (Step 1.3, spec D4) have no _to_si/_from_si pair of their
    # own -- they reuse lambda_to_si/theta_to_si -- so they are round-tripped separately below.
    for q in quantities
        row = getfield(ACCEPTED_UNITS, q)
        to_si = getfield(TRUST, Symbol(q, "_to_si"))
        from_si = getfield(TRUST, Symbol(q, "_from_si"))
        for key in keys(row)
            for mag in (1e-6, 1.0, 1e6)
                @test isapprox(from_si(to_si(mag, key), key), mag; rtol = 1e-14)
            end
        end
    end
end

@testset "boundary units: table values" begin
    # Tight: these define the table.
    @test T_to_si(1, :T_eV) == EV
    @test T_to_si(1, :T_keV) == 1000 * EV
    @test n_to_si(1, :n_cm3) == 1e6
    @test u_to_si(1, :u_kms) == 1e3
    @test mass_to_si(1, :mass_amu) == ATOMIC_MASS_UNIT
    @test isapprox(theta_to_si(180, :theta_deg), π; rtol = 1e-15)
    @test EV == ELEMENTARY_CHARGE
end

@testset "boundary units: canonical key is first in each row" begin
    # Tight: guards the `show` contract (conventions.md 1.2).
    @test first(keys(ACCEPTED_UNITS.T)) == :T_eV
    @test first(keys(ACCEPTED_UNITS.n)) == :n_cm3
    @test first(keys(ACCEPTED_UNITS.n_e)) == :n_e_cm3
    @test first(keys(ACCEPTED_UNITS.u)) == :u_kms
    @test first(keys(ACCEPTED_UNITS.mass)) == :mass_amu
    @test first(keys(ACCEPTED_UNITS.lambda)) == :lambda_nm
    @test first(keys(ACCEPTED_UNITS.t)) == :t_ns
    @test first(keys(ACCEPTED_UNITS.theta)) == :theta_deg
end

@testset "GE14: lambda_probe and polarization rows (spec D4)" begin
    # Tight: guards the show/deck contract (conventions.md 1.2), same as the other rows above.
    @test first(keys(ACCEPTED_UNITS.lambda_probe)) == :lambda_probe_nm
    @test first(keys(ACCEPTED_UNITS.polarization)) == :polarization_deg
    @test values(ACCEPTED_UNITS.lambda_probe) == values(ACCEPTED_UNITS.lambda)
    @test values(ACCEPTED_UNITS.polarization) == values(ACCEPTED_UNITS.theta)
end

@testset "boundary units: unknown key throws (D10)" begin
    @test_throws "T_to_si: :T_erg is not an accepted unit key for T; expected one of (:T_eV, :T_keV)" T_to_si(1, :T_erg)
    @test_throws "n_from_si: :bogus is not an accepted unit key for n; expected one of (:n_cm3, :n_m3)" n_from_si(1.0, :bogus)
end

@testset "_basefloat" begin
    # Tight: exact type equality, measured Float64, Float32, Float64, Float64, Float64.
    @test _basefloat(1.0) === Float64
    @test _basefloat(1.0f0) === Float32
    @test _basefloat(1) === Float64
    d = Dual(1.0, 1.0)
    @test _basefloat(d) === Float64
    dd = Dual(Dual(1.0, 1.0), Dual(1.0, 1.0))
    @test _basefloat(dd) === Float64
    @test_throws "no primal float type for ComplexF64" _basefloat(1.0 + 2.0im)
    @test_throws ArgumentError _basefloat(Complex{Float64})
end
