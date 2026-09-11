using Test
using OTIS
using ForwardDiff

@testset "GE1: constructor round trip across unit keywords" begin
    p1 = ProbeBeam(lambda_probe_nm = 263.25, polarization_deg = 90)
    p2 = ProbeBeam(lambda_probe_m = 263.25e-9, polarization_rad = pi / 2)
    @test isapprox(p1.lambda, p2.lambda; rtol = 1e-15)
    @test isapprox(p1.polarization, p2.polarization; rtol = 1e-15)

    c1 = Collection(theta_deg = 60)
    c2 = Collection(theta_rad = pi / 3)
    @test isapprox(c1.theta, c2.theta; rtol = 1e-15)
end

@testset "GE2: defaults" begin
    p = ProbeBeam(lambda_probe_nm = 263.25)
    @test p.polarization == 0
    @test p.a0_sq == 0
    @test p.diameter == 0

    c = Collection(theta_deg = 60)
    @test c.k_model === :exact
    @test c.dispersion == 0
end

@testset "GE3: constructor errors" begin
    @test_throws "no unit keyword given for lambda_probe" ProbeBeam()
    @test_throws "more than one unit keyword given for lambda_probe" ProbeBeam(
        lambda_probe_nm = 263.25, lambda_probe_m = 263.25e-9)
    @test_throws "unrecognised keyword" ProbeBeam(lambda_probe_nm = 263.25, thetaa_deg = 60)
    @test_throws "ProbeBeam: lambda_probe must be positive; got 0.0 m" ProbeBeam(lambda_probe_nm = 0)

    @test_throws "Collection: k_model must be :exact or :approximate; got :foo" Collection(
        theta_deg = 60, k_model = :foo)
    @test_throws "Collection: theta must be between 0 and pi radians" Collection(theta_deg = 200)
    @test_throws "Collection: give theta_deg/theta_rad or k_probe/k_scatter" Collection()
end

@testset "GE4: wavevector against D21a" begin
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))

    k500, w500, ks500 = wavevector(probe, collection, lambda_to_si(500.0, :lambda_nm), p)
    @test isapprox(k500, 1.73096164807239920e+07; rtol = 1e-12)
    @test isapprox(w500, 1.89617346756637500e+14; rtol = 1e-12)

    kmid, wmid, ksmid = wavevector(probe, collection, probe.lambda, p)
    @test isapprox(kmid, 1.68560532223105170e+07; rtol = 1e-12)
    @test wmid == 0.0
    # k_s = k_i at lambda_s = lambda_probe (omega = 0); GS6 checks k_s against its own
    # independent oracle in general, this just pins the value here too.
    @test isapprox(ksmid, 1.19190295375371200e+07; rtol = 1e-12)

    k560, w560, ks560 = wavevector(probe, collection, lambda_to_si(560.0, :lambda_nm), p)
    @test isapprox(k560, 1.63583912268507201e+07; rtol = 1e-12)
    @test isapprox(w560, -2.14022274809545500e+14; rtol = 1e-12)
end

@testset "GE5: :approximate is plasma-independent" begin
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90, k_model = :approximate)
    p1 = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    p2 = Plasma(n_e_cm3 = 1e17, electrons = (electrons(T_eV = 500),))
    lambdas = (lambda_to_si(500.0, :lambda_nm), probe.lambda, lambda_to_si(560.0, :lambda_nm))
    expected = 4pi * sin(collection.theta / 2) / probe.lambda

    ks = [first(wavevector(probe, collection, lam, p1)) for lam in lambdas]
    for k in ks
        @test isapprox(k, expected; rtol = 1e-15)
    end
    @test ks[1] == ks[2] == ks[3]
    for lam in lambdas
        @test first(wavevector(probe, collection, lam, p1)) == first(wavevector(probe, collection, lam, p2))
    end
end

@testset "GE6: k/k_approx - 1 -> -n_e/(2 n_cr)" begin
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    c_exact = Collection(theta_deg = 90, k_model = :exact)
    c_approx = Collection(theta_deg = 90, k_model = :approximate)
    omega_i = 2pi * SPEED_OF_LIGHT / probe.lambda
    n_cr = VACUUM_PERMITTIVITY * ELECTRON_MASS * omega_i^2 / ELEMENTARY_CHARGE^2

    for n_e_cm3 in (1e19, 1e17, 1e15, 1e13)
        p = Plasma(n_e_cm3 = n_e_cm3, electrons = (electrons(T_eV = 500),))
        k_exact, _, _ = wavevector(probe, c_exact, probe.lambda, p)
        k_approx, _, _ = wavevector(probe, c_approx, probe.lambda, p)
        n_e = n_to_si(n_e_cm3, :n_cm3)
        @test isapprox(k_exact / k_approx - 1, -n_e / (2 * n_cr); rtol = 1e-3)
    end
end

@testset "GE7: blue is a larger k" begin
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    k_blue, _, _ = wavevector(probe, collection, lambda_to_si(500.0, :lambda_nm), p)
    k_mid, _, _ = wavevector(probe, collection, probe.lambda, p)
    k_red, _, _ = wavevector(probe, collection, lambda_to_si(560.0, :lambda_nm), p)
    @test k_blue > k_mid > k_red
end

@testset "GE8: cutoff errors" begin
    # cutoff at n_e = 1e19 cm^-3 is ~1.055867e-05 m (D21a); 1.1e-5 m is beyond it, matching
    # D15's literal example messages exactly.
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    @test_throws "wavevector: the scattered wavelength 1.100000e-05 m is at or beyond the plasma cutoff " *
                 "1.055867e-05 m (omega_s <= omega_pe) for n_e = 1.000000e+25 m^-3" wavevector(
        probe, collection, 1.1e-5, p)

    probe_bad = ProbeBeam(lambda_probe_m = 1.1e-5)
    @test_throws "wavevector: the probe wavelength 1.100000e-05 m is at or beyond the plasma cutoff " *
                 "1.055867e-05 m (omega_probe <= omega_pe) for n_e = 1.000000e+25 m^-3" wavevector(
        probe_bad, collection, lambda_to_si(500.0, :lambda_nm), p)
end

@testset "GE9: k = 0 throws" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    msg = "wavevector: k = 0 at theta = 0.0 rad with lambda_s equal to the probe wavelength; " *
          "this geometry probes no wavevector"
    probe = ProbeBeam(lambda_probe_nm = 526.5)

    c_exact = Collection(theta_deg = 0)
    @test_throws msg wavevector(probe, c_exact, probe.lambda, p)

    c_approx = Collection(theta_deg = 0, k_model = :approximate)
    @test_throws msg wavevector(probe, c_approx, probe.lambda, p)
end

@testset "GE10: polarization_factor" begin
    # atol = 1e-15 alongside rtol: at theta = 90 deg, cos(theta_rad)^2 evaluates to ~3.7e-33
    # (not exactly 0) from floating-point argument reduction near pi/2, while
    # polarization_factor's own 1 - sin(theta)^2 gives exactly 0.0 there -- a pure relative
    # tolerance against a near-zero reference is unsatisfiable (conventions.md 8.8's principle,
    # applied here to a plain trig identity rather than a quadrature oracle).
    for theta_deg in (60.0, 90.0)
        theta_rad = theta_to_si(theta_deg, :theta_deg)
        collection = Collection(theta_deg = theta_deg)

        p0 = ProbeBeam(lambda_probe_nm = 526.5, polarization_deg = 0)
        @test isapprox(polarization_factor(p0, collection), 1; rtol = 1e-15, atol = 1e-15)

        p90 = ProbeBeam(lambda_probe_nm = 526.5, polarization_deg = 90)
        @test isapprox(polarization_factor(p90, collection), cos(theta_rad)^2; rtol = 1e-15, atol = 1e-15)

        p30 = ProbeBeam(lambda_probe_nm = 526.5, polarization_deg = 30)
        psi = theta_to_si(30.0, :theta_deg)
        @test isapprox(polarization_factor(p30, collection), 1 - sin(theta_rad)^2 * sin(psi)^2;
            rtol = 1e-15, atol = 1e-15)
    end
end

@testset "GE11: show" begin
    p = ProbeBeam(lambda_probe_nm = 263.25)
    @test sprint(show, p) == "ProbeBeam(263.25 nm)"
    c = Collection(theta_deg = 60)
    @test sprint(show, c) == "Collection(60.0 deg, :exact)"

    ml = sprint((io, x) -> show(io, MIME("text/plain"), x), p)
    @test occursin("nm", ml)
    @test !occursin(" m\n", ml)
end

@testset "GE12: allocations" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    c_exact = Collection(theta_deg = 90)
    c_approx = Collection(theta_deg = 90, k_model = :approximate)
    lam = lambda_to_si(500.0, :lambda_nm)
    _wv_exact() = wavevector(probe, c_exact, lam, p)
    _wv_approx() = wavevector(probe, c_approx, lam, p)
    _wv_exact()
    _wv_approx()
    @test (@allocated _wv_exact()) == 0
    @test (@allocated _wv_approx()) == 0

    # PF2 (Step 1.6): the same, on a two-electron/two-ion plasma.
    p2 = Plasma(n_e_cm3 = 1e19,
        electrons = (electrons(name = "e1", T_eV = 500, fraction = 0.5),
                     electrons(name = "e2", T_eV = 800, fraction = 0.5)),
        ions = (ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = 0.5),
                ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300, fraction = 0.5)))
    _wv_exact2() = wavevector(probe, c_exact, lam, p2)
    _wv_exact2()
    @test (@allocated _wv_exact2()) == 0
end

@testset "GE13: AD" begin
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    lam0 = lambda_to_si(500.0, :lambda_nm)

    f_lam(lam) = first(wavevector(probe, collection, lam, p))
    ad1 = ForwardDiff.derivative(f_lam, lam0)
    h1 = 1e-5 * lam0
    fd1 = (f_lam(lam0 + h1) - f_lam(lam0 - h1)) / (2h1)
    @test isapprox(ad1, fd1; rtol = 1e-8)

    function k_of_ne(n_e_cm3)
        pp = Plasma(n_e_cm3 = n_e_cm3, electrons = (electrons(T_eV = 500),))
        return first(wavevector(probe, collection, lam0, pp))
    end
    ne0 = 1e19
    ad2 = ForwardDiff.derivative(k_of_ne, ne0)
    h2 = 1e-5 * ne0
    fd2 = (k_of_ne(ne0 + h2) - k_of_ne(ne0 - h2)) / (2h2)
    @test isapprox(ad2, fd2; rtol = 1e-8)
end

# --- Step 2.3a: the geometry split (spec Section 6a) ---

@testset "GS1: Collection from directions equals Collection from theta" begin
    theta0 = theta_to_si(60.0, :theta_deg)
    c_theta = Collection(theta_rad = theta0)
    k_probe = [0.0, 0.0, 1.0]
    k_scatter = [sin(theta0), 0.0, cos(theta0)]
    c_dir = Collection(k_probe = k_probe, k_scatter = k_scatter)
    @test isapprox(c_dir.theta, c_theta.theta; rtol = 1e-14)
end

@testset "GS2: degenerate directions throw" begin
    @test_throws "too close to parallel or antiparallel" Collection(
        k_probe = [0.0, 0.0, 1.0], k_scatter = [0.0, 0.0, 1.0])
    @test_throws "too close to parallel or antiparallel" Collection(
        k_probe = [0.0, 0.0, 1.0], k_scatter = [0.0, 0.0, -1.0 + 1e-12])
    @test_throws "non-zero vectors" Collection(
        k_probe = [0.0, 0.0, 0.0], k_scatter = [0.0, 0.0, 1.0])
end

@testset "GS3: kdirection is a unit vector in the x-z plane" begin
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    for theta_deg in (30.0, 60.0, 90.0, 154.2)
        collection = Collection(theta_deg = theta_deg)
        khat = kdirection(probe, collection)
        @test isapprox(sqrt(sum(abs2, khat)), 1.0; atol = 1e-15)
        @test khat[2] == 0.0
    end
end

@testset "GS4: polarization_factor is unchanged by the split" begin
    for psi_deg in (0.0, 30.0, 90.0), theta_deg in (30.0, 60.0, 154.2)
        probe = ProbeBeam(lambda_probe_nm = 526.5, polarization_deg = psi_deg)
        collection = Collection(theta_deg = theta_deg)
        theta_rad = theta_to_si(theta_deg, :theta_deg)
        psi_rad = theta_to_si(psi_deg, :theta_deg)
        expected = 1 - sin(theta_rad)^2 * sin(psi_rad)^2
        @test polarization_factor(probe, collection) === expected
    end
end

@testset "GS5: a0_sq against the standard formula" begin
    # a0 = 0.85 sqrt(I18 * lambda_um^2); at I = 1e18 W/cm^2, lambda = 1 um both factors are 1,
    # giving 0.85^2 = 0.7225. The package's formula is exact and gives 0.731 (spec D40); the two
    # agree to the rounding of the quoted 0.85 (about 1%), not to 1e-3 -- the 1e-3 tolerance is
    # against Level 2's own precomputed 0.731, a regression pin on the formula itself.
    probe = ProbeBeam(lambda_probe_m = 1e-6, intensity_wcm2 = 1e18)
    standard = 0.85^2
    @test isapprox(probe.a0_sq, 0.731; rtol = 1e-3)
    reldiff = abs(probe.a0_sq - standard) / standard
    @info "GS5 a0_sq against the standard 0.85 formula" a0_sq = probe.a0_sq standard reldiff
    @test reldiff < 0.02
end

@testset "GS6: wavevector returns k_s consistent with its own k" begin
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    p = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = 500),))
    lam_s = lambda_to_si(500.0, :lambda_nm)
    k, omega, k_s = wavevector(probe, collection, lam_s, p)
    omega_s = omega + OTIS._omega_probe(probe)
    omega_pe_sq = p.n_e * ELEMENTARY_CHARGE^2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS)
    k_s_oracle = sqrt(omega_s^2 - omega_pe_sq) / SPEED_OF_LIGHT
    @test isapprox(k_s, k_s_oracle; rtol = 1e-15)
end

@testset "AX1: detector_axis round-trips a positive dispersion" begin
    collection = Collection(theta_deg = 60, lambda_center_nm = 230.0, dispersion_nm_px = 0.0494,
        npixels = 1024, reference_pixel = 512)
    axis = detector_axis(collection)
    @test issorted(axis)
    ref = [lambda_to_si(230.0 + 0.0494 * (p - 512), :lambda_nm) for p in 1:1024]
    @test isapprox(axis, ref; rtol = 1e-15)
end

@testset "AX2: a negative dispersion gives an increasing axis" begin
    collection = Collection(theta_deg = 60, lambda_center_nm = 263.25, dispersion_nm_px = -0.00444,
        npixels = 1024, reference_pixel = 512)
    axis = detector_axis(collection)
    @test issorted(axis)
    ref_reversed = sort([lambda_to_si(263.25 + (-0.00444) * (p - 512), :lambda_nm) for p in 1:1024])
    @test isapprox(axis[1], ref_reversed[1]; rtol = 1e-15)
    @test isapprox(axis[end], ref_reversed[end]; rtol = 1e-15)
end

@testset "AX3: the sign trap - a feature lands at the same wavelength either sign" begin
    # reference_pixel = 1 for the positive-dispersion collection and reference_pixel = npixels
    # for the negative one (same lambda_center, same |dispersion|) makes the two detector_axis
    # results identical element-wise: detector_axis reverses the negative-dispersion pixel order,
    # which exactly undoes the reference_pixel offset. This is the row that stops a silently
    # mirrored spectrum: with the sign handled correctly the two axes -- and hence the two
    # measured spectra -- are bitwise the same; get the sign wrong and they would be mirrored.
    probe = ProbeBeam(lambda_probe_nm = 263.25)
    n = 256
    c_pos = Collection(theta_deg = 60, lambda_center_nm = 263.25, dispersion_nm_px = 0.005,
        npixels = n, reference_pixel = 1)
    c_neg = Collection(theta_deg = 60, lambda_center_nm = 263.25, dispersion_nm_px = -0.005,
        npixels = n, reference_pixel = n)
    axis_pos = detector_axis(c_pos)
    axis_neg = detector_axis(c_neg)
    @test axis_pos == axis_neg

    p = Plasma(n_e_cm3 = 3e19, electrons = (electrons(T_eV = 400),),
               ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 200, u_kms = 300),))
    P_pos = spectrum(p, probe, c_pos, axis_pos)
    P_neg = spectrum(p, probe, c_neg, axis_neg)
    peak_pos = axis_pos[argmax(P_pos)]
    peak_neg = axis_neg[argmax(P_neg)]
    @test isapprox(peak_pos, peak_neg; rtol = 1e-12)
end

@testset "AX4: Experiment falls back to detector_axis, and disagreement throws" begin
    template = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500),),
        ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
    trace = PlasmaTrace(template; t_ns = [0.0])
    probe = ProbeBeam(lambda_probe_nm = 263.25)
    collection = Collection(theta_deg = 60, lambda_center_nm = 250.0, dispersion_nm_px = 0.05,
        npixels = 64, reference_pixel = 32)

    expt1 = Experiment(trace = trace, probe = probe, collection = collection)
    @test expt1.lambda == detector_axis(collection)

    axis_ok = detector_axis(collection)
    expt2 = Experiment(trace = trace, probe = probe, collection = collection, lambda_m = axis_ok)
    @test expt2.lambda === axis_ok

    axis_bad = copy(axis_ok)
    axis_bad[5] += 1e-9
    @test_throws "disagrees with the collection's dispersion at bin" Experiment(
        trace = trace, probe = probe, collection = collection, lambda_m = axis_bad)
end
