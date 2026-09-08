using Test
using OTIS
using ForwardDiff

@testset "GE1: constructor round trip across unit keywords" begin
    g1 = ScatteringGeometry(lambda_probe_nm = 263.25, theta_deg = 60, polarization_deg = 90)
    g2 = ScatteringGeometry(lambda_probe_m = 263.25e-9, theta_rad = pi / 3, polarization_rad = pi / 2)
    @test isapprox(g1.lambda_probe, g2.lambda_probe; rtol = 1e-15)
    @test isapprox(g1.theta, g2.theta; rtol = 1e-15)
    @test isapprox(g1.polarization, g2.polarization; rtol = 1e-15)
end

@testset "GE2: defaults" begin
    g = ScatteringGeometry(lambda_probe_nm = 263.25, theta_deg = 60)
    @test g.polarization == 0
    @test g.k_model === :exact
end

@testset "GE3: constructor errors" begin
    @test_throws "no unit keyword given for lambda_probe" ScatteringGeometry(theta_deg = 60)
    @test_throws "more than one unit keyword given for lambda_probe" ScatteringGeometry(
        lambda_probe_nm = 263.25, lambda_probe_m = 263.25e-9, theta_deg = 60)
    @test_throws "unrecognised keyword" ScatteringGeometry(
        lambda_probe_nm = 263.25, theta_deg = 60, thetaa_deg = 60)
    @test_throws "ScatteringGeometry: k_model must be :exact or :approximate; got :foo" ScatteringGeometry(
        lambda_probe_nm = 263.25, theta_deg = 60, k_model = :foo)
    @test_throws "ScatteringGeometry: lambda_probe must be positive; got 0.0 m" ScatteringGeometry(
        lambda_probe_nm = 0, theta_deg = 60)
    @test_throws "ScatteringGeometry: theta must be between 0 and pi radians" ScatteringGeometry(
        lambda_probe_nm = 263.25, theta_deg = 200)
end

@testset "GE4: wavevector against D21a" begin
    g = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90)
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))

    k500, w500 = wavevector(g, lambda_to_si(500.0, :lambda_nm), p)
    @test isapprox(k500, 1.73096164807239920e+07; rtol = 1e-12)
    @test isapprox(w500, 1.89617346756637500e+14; rtol = 1e-12)

    # Built from geometry.lambda_probe itself (D21a's note): the same Float64, so omega is
    # exactly zero, not one ulp away as an independently-written literal would give.
    kmid, wmid = wavevector(g, g.lambda_probe, p)
    @test isapprox(kmid, 1.68560532223105170e+07; rtol = 1e-12)
    @test wmid == 0.0

    k560, w560 = wavevector(g, lambda_to_si(560.0, :lambda_nm), p)
    @test isapprox(k560, 1.63583912268507201e+07; rtol = 1e-12)
    @test isapprox(w560, -2.14022274809545500e+14; rtol = 1e-12)
end

@testset "GE5: :approximate is plasma-independent" begin
    g = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90, k_model = :approximate)
    p1 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    p2 = Plasma(n_e_cm3 = 1e17, electrons = (electrons(T_eV = 500),))
    lambdas = (lambda_to_si(500.0, :lambda_nm), g.lambda_probe, lambda_to_si(560.0, :lambda_nm))
    expected = 4pi * sin(g.theta / 2) / g.lambda_probe

    ks = [first(wavevector(g, lam, p1)) for lam in lambdas]
    for k in ks
        @test isapprox(k, expected; rtol = 1e-15)
    end
    @test ks[1] == ks[2] == ks[3]
    for lam in lambdas
        @test first(wavevector(g, lam, p1)) == first(wavevector(g, lam, p2))
    end
end

@testset "GE6: k/k_approx - 1 -> -n_e/(2 n_cr)" begin
    g_exact = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90, k_model = :exact)
    g_approx = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90, k_model = :approximate)
    omega_i = 2pi * SPEED_OF_LIGHT / g_exact.lambda_probe
    n_cr = VACUUM_PERMITTIVITY * ELECTRON_MASS * omega_i^2 / ELEMENTARY_CHARGE^2

    for n_e_cm3 in (1e19, 1e17, 1e15, 1e13)
        p = Plasma(n_e_cm3 = n_e_cm3, electrons = (electrons(T_eV = 500),))
        k_exact, _ = wavevector(g_exact, g_exact.lambda_probe, p)
        k_approx, _ = wavevector(g_approx, g_approx.lambda_probe, p)
        n_e = n_to_si(n_e_cm3, :n_cm3)
        @test isapprox(k_exact / k_approx - 1, -n_e / (2 * n_cr); rtol = 1e-3)
    end
end

@testset "GE7: blue is a larger k" begin
    g = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90)
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    k_blue, _ = wavevector(g, lambda_to_si(500.0, :lambda_nm), p)
    k_mid, _ = wavevector(g, g.lambda_probe, p)
    k_red, _ = wavevector(g, lambda_to_si(560.0, :lambda_nm), p)
    @test k_blue > k_mid > k_red
end

@testset "GE8: cutoff errors" begin
    # cutoff at n_e = 1e19 cm^-3 is ~1.055867e-05 m (D21a); 1.1e-5 m is beyond it, matching
    # D15's literal example messages exactly.
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    g = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90)
    @test_throws "wavevector: the scattered wavelength 1.100000e-05 m is at or beyond the plasma cutoff " *
                 "1.055867e-05 m (omega_s <= omega_pe) for n_e = 1.000000e+25 m^-3" wavevector(g, 1.1e-5, p)

    g_bad_probe = ScatteringGeometry(lambda_probe_m = 1.1e-5, theta_deg = 90)
    @test_throws "wavevector: the probe wavelength 1.100000e-05 m is at or beyond the plasma cutoff " *
                 "1.055867e-05 m (omega_probe <= omega_pe) for n_e = 1.000000e+25 m^-3" wavevector(
        g_bad_probe, lambda_to_si(500.0, :lambda_nm), p)
end

@testset "GE9: k = 0 throws" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    msg = "wavevector: k = 0 at theta = 0.0 rad with lambda_s equal to the probe wavelength; " *
          "this geometry probes no wavevector"

    g_exact = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 0)
    @test_throws msg wavevector(g_exact, g_exact.lambda_probe, p)

    g_approx = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 0, k_model = :approximate)
    @test_throws msg wavevector(g_approx, g_approx.lambda_probe, p)
end

@testset "GE10: polarization_factor" begin
    # atol = 1e-15 alongside rtol: at theta = 90 deg, cos(theta_rad)^2 evaluates to ~3.7e-33
    # (not exactly 0) from floating-point argument reduction near pi/2, while
    # polarization_factor's own 1 - sin(theta)^2 gives exactly 0.0 there -- a pure relative
    # tolerance against a near-zero reference is unsatisfiable (conventions.md 8.8's principle,
    # applied here to a plain trig identity rather than a quadrature oracle).
    for theta_deg in (60.0, 90.0)
        theta_rad = theta_to_si(theta_deg, :theta_deg)

        g0 = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = theta_deg, polarization_deg = 0)
        @test isapprox(polarization_factor(g0), 1; rtol = 1e-15, atol = 1e-15)

        g90 = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = theta_deg, polarization_deg = 90)
        @test isapprox(polarization_factor(g90), cos(theta_rad)^2; rtol = 1e-15, atol = 1e-15)

        g30 = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = theta_deg, polarization_deg = 30)
        psi = theta_to_si(30.0, :theta_deg)
        @test isapprox(polarization_factor(g30), 1 - sin(theta_rad)^2 * sin(psi)^2; rtol = 1e-15, atol = 1e-15)
    end
end

@testset "GE11: show" begin
    g = ScatteringGeometry(lambda_probe_nm = 263.25, theta_deg = 60)
    @test sprint(show, g) == "ScatteringGeometry(263.25 nm, 60.0 deg, :exact)"

    ml = sprint((io, x) -> show(io, MIME("text/plain"), x), g)
    @test occursin("nm", ml)
    @test occursin("deg", ml)
    @test !occursin(" m\n", ml)
    @test !occursin(" rad\n", ml)
end

@testset "GE12: allocations" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    g_exact = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90)
    g_approx = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90, k_model = :approximate)
    lam = lambda_to_si(500.0, :lambda_nm)
    _wv_exact() = wavevector(g_exact, lam, p)
    _wv_approx() = wavevector(g_approx, lam, p)
    _wv_exact()
    _wv_approx()
    @test (@allocated _wv_exact()) == 0
    @test (@allocated _wv_approx()) == 0
end

@testset "GE13: AD" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    g = ScatteringGeometry(lambda_probe_nm = 526.5, theta_deg = 90)
    lam0 = lambda_to_si(500.0, :lambda_nm)

    f_lam(lam) = first(wavevector(g, lam, p))
    ad1 = ForwardDiff.derivative(f_lam, lam0)
    h1 = 1e-5 * lam0
    fd1 = (f_lam(lam0 + h1) - f_lam(lam0 - h1)) / (2h1)
    @test isapprox(ad1, fd1; rtol = 1e-8)

    function k_of_ne(n_e_cm3)
        pp = Plasma(n_e_cm3 = n_e_cm3, electrons = (electrons(T_eV = 500),))
        return first(wavevector(g, lam0, pp))
    end
    ne0 = 1e19
    ad2 = ForwardDiff.derivative(k_of_ne, ne0)
    h2 = 1e-5 * ne0
    fd2 = (k_of_ne(ne0 + h2) - k_of_ne(ne0 - h2)) / (2h2)
    @test isapprox(ad2, fd2; rtol = 1e-8)
end
