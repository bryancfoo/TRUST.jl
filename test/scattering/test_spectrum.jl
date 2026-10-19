using Test
using TRUST
using ForwardDiff

const H_MASS_AMU = 1.007276466879   # CODATA proton mass, in amu

reference_plasma() = Plasma(
    n_e_cm3 = 1e19,
    electrons = (electrons(T_eV = 500),),
    ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))

# The peak method of spec D23: parabolic offset from the grid maximum, position and value.
function peak_position_and_value(x, y)
    i = argmax(y)
    (i == 1 || i == length(y)) && return x[i], y[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    h = x[i] - x[i - 1]
    d = (y0 - y2) / (2 * (y0 - 2y1 + y2))
    return x[i] + d * h, y1 - (y0 - y2) * d / 4
end
peak_position(x, y) = first(peak_position_and_value(x, y))

@testset "SP1: spectrum, spectrum! and _spectrum_at agree" begin
    p = reference_plasma()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    lambdas = range(lambda_to_si(500.0, :lambda_nm), lambda_to_si(560.0, :lambda_nm); length = 32)
    P = spectrum(p, probe, collection, lambdas)
    for (i, lam) in enumerate(lambdas)
        @test P[i] == TRUST._spectrum_at(p, probe, collection, lam)
    end
    P2 = similar(P)
    spectrum!(P2, p, probe, collection, lambdas)
    @test P2 == P
end

@testset "SP2: the factorization" begin
    p = reference_plasma()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    lambdas = range(lambda_to_si(500.0, :lambda_nm), lambda_to_si(560.0, :lambda_nm); length = 8)
    for lam in lambdas
        k, omega, k_s = wavevector(probe, collection, lam, p)
        val = TRUST._spectrum_at(p, probe, collection, lam)
        omega_probe = TRUST._omega_probe(probe)
        factored = val / TRUST._wavelength_jacobian(lam) / TRUST._prefactor(omega, omega_probe, true)
        @test isapprox(factored, spectral_density(p, k, omega); rtol = 1e-14)
    end
end

@testset "SP3: the prefactor option" begin
    p = reference_plasma()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    lam = lambda_to_si(500.0, :lambda_nm)
    k, omega, k_s = wavevector(probe, collection, lam, p)
    S = spectral_density(p, k, omega)
    @test TRUST._spectrum_at(p, probe, collection, lam, false) == S * TRUST._wavelength_jacobian(lam)

    @test_throws "spectrum: prefactor must be :first_order or :none; got :full" spectrum(
        p, probe, collection, [lam]; prefactor = :full)
end

@testset "SP4: Sheffield Eq. 4.5.3" begin
    p = Plasma(n_e_cm3 = 5e12, electrons = (electrons(T_eV = 10),),
               ions = (ion(mass_amu = H_MASS_AMU, charge = 1, T_eV = 10),))
    probe = ProbeBeam(lambda_probe_nm = 694.3)
    collection = Collection(theta_deg = 90, k_model = :exact)
    lam_i = probe.lambda
    theta = collection.theta
    a = thermal_speed(p.electrons["e"])

    dlam_max = 2 * (a / SPEED_OF_LIGHT) * sin(theta / 2) * lam_i
    dlams = collect(range(-dlam_max, dlam_max; length = 401))
    lambdas = lam_i .+ dlams

    P = spectrum(p, probe, collection, lambdas; prefactor = :first_order)
    Pn = P ./ maximum(P)

    s2 = sin(theta / 2)^2
    ref = [(1 - 3.5 * dl / lam_i + (SPEED_OF_LIGHT^2 * dl^3) / (4 * a^2 * lam_i^3 * s2)) *
           exp(-(SPEED_OF_LIGHT^2 * dl^2) / (4 * a^2 * lam_i^2 * s2)) for dl in dlams]
    refn = ref ./ maximum(ref)

    err = maximum(abs.(Pn .- refn))
    @info "SP4 max normalized-curve error" err
    @test err < 1e-3
end

@testset "SP5: centre shift vs Eq. 4.5.4" begin
    p = Plasma(n_e_cm3 = 1e14, electrons = (electrons(T_eV = 117),))
    probe = ProbeBeam(lambda_probe_nm = 694.3)
    collection = Collection(theta_deg = 83)
    lam_i = probe.lambda
    theta = collection.theta
    a = thermal_speed(p.electrons["e"])

    dlam_max = 3 * (a / SPEED_OF_LIGHT) * sin(theta / 2) * lam_i
    lambdas = collect(range(lam_i - dlam_max, lam_i + dlam_max; length = 20001))
    P = spectrum(p, probe, collection, lambdas)

    dlam_peak = peak_position(lambdas, P) - lam_i
    predicted = -7 * (a / SPEED_OF_LIGHT)^2 * sin(theta / 2)^2 * lam_i
    @info "SP5 centre shift" measured_A = dlam_peak * 1e10 predicted_A = predicted * 1e10
    @test isapprox(dlam_peak, predicted; rtol = 0.05)
end

@testset "SP6: +/- omega symmetry" begin
    p = reference_plasma()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    c_approx = Collection(theta_deg = 90, k_model = :approximate)
    omega_i = TRUST._omega_probe(probe)
    for omega in (1e12, 1e13, 5e13, 2e14, 5e14)
        lam_plus = 2pi * SPEED_OF_LIGHT / (omega_i + omega)
        lam_minus = 2pi * SPEED_OF_LIGHT / (omega_i - omega)
        Pp = TRUST._spectrum_at(p, probe, c_approx, lam_plus, false) / TRUST._wavelength_jacobian(lam_plus)
        Pm = TRUST._spectrum_at(p, probe, c_approx, lam_minus, false) / TRUST._wavelength_jacobian(lam_minus)
        @test isapprox(Pp, Pm; rtol = 1e-12)
    end

    # The exact k breaks this symmetry by design.
    c_exact = Collection(theta_deg = 90, k_model = :exact)
    lam_i = probe.lambda
    dlam = lambda_to_si(5.0, :lambda_nm)
    Pp = TRUST._spectrum_at(p, probe, c_exact, lam_i + dlam, false) / TRUST._wavelength_jacobian(lam_i + dlam)
    Pm = TRUST._spectrum_at(p, probe, c_exact, lam_i - dlam, false) / TRUST._wavelength_jacobian(lam_i - dlam)
    @test abs(Pp - Pm) / max(abs(Pp), abs(Pm)) > 1e-3
end

function _sp78_setup()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    omega_i = TRUST._omega_probe(probe)
    n_cr = VACUUM_PERMITTIVITY * ELECTRON_MASS * omega_i^2 / ELEMENTARY_CHARGE^2
    n_e = 0.005 * n_cr
    p = Plasma(n_e_cm3 = n_from_si(n_e, :n_cm3), electrons = (electrons(T_eV = 300),),
               ions = (ion(mass_amu = H_MASS_AMU, charge = 1, T_eV = 300),))
    a = thermal_speed(p.electrons["e"])
    lam_i = probe.lambda

    dlam_pred = 2 * sqrt(0.005 + 3 * (a / SPEED_OF_LIGHT)^2) * (1 + 1.5 * 0.005) * lam_i
    du = 0.9 * dlam_pred / (2 * lam_i)
    blue_grid = collect(range(lam_i * (1 - 1.4du), lam_i * (1 - 0.25du); length = 4001))
    red_grid = collect(range(lam_i * (1 + 0.25du), lam_i * (1 + 1.4du); length = 4001))

    return p, probe, collection, lam_i, a, dlam_pred, blue_grid, red_grid
end

@testset "SP7: EPW separation vs Eq. 5.4.6" begin
    p, probe, collection, lam_i, a, dlam_pred, blue_grid, red_grid = _sp78_setup()
    Pblue = spectrum(p, probe, collection, blue_grid)
    Pred = spectrum(p, probe, collection, red_grid)
    blue_peak = peak_position(blue_grid, Pblue)
    red_peak = peak_position(red_grid, Pred)
    separation = red_peak - blue_peak
    @info "SP7 EPW separation" measured_nm = separation * 1e9 predicted_nm = dlam_pred * 1e9
    @test isapprox(separation, dlam_pred; rtol = 0.06)
end

@testset "SP8: blue/red EPW peak ratio vs Eq. 5.4.10" begin
    p, probe, collection, lam_i, a, dlam_pred, blue_grid, red_grid = _sp78_setup()
    Pblue_over_J = spectrum(p, probe, collection, blue_grid) ./ TRUST._wavelength_jacobian.(blue_grid)
    Pred_over_J = spectrum(p, probe, collection, red_grid) ./ TRUST._wavelength_jacobian.(red_grid)
    _, blue_val = peak_position_and_value(blue_grid, Pblue_over_J)
    _, red_val = peak_position_and_value(red_grid, Pred_over_J)
    ratio = blue_val / red_val
    @test ratio > 1

    k_i, _, _ = wavevector(probe, collection, lam_i, p)
    omega_i = TRUST._omega_probe(probe)
    omega_pe = plasma_frequency(p.electrons["e"])
    predicted = 1 + 4 * sqrt(omega_pe^2 + a^2 * k_i^2) / omega_i
    @info "SP8 blue/red ratio" measured = ratio predicted = predicted
    @test isapprox(ratio, predicted; rtol = 0.10)
end

@testset "SP9: gradient wrt (n_e, T_e, T_i, u)" begin
    probe = ProbeBeam(lambda_probe_nm = 263.25)
    collection = Collection(theta_deg = 60)
    lam_i = probe.lambda
    lambdas = collect(range(0.9 * lam_i, 1.1 * lam_i; length = 9))

    function spectrum_of_theta(theta)
        n_e_cm3, T_e_eV, T_i_eV, u_ms = theta
        p = Plasma(n_e_cm3 = n_e_cm3,
                   electrons = (electrons(T_eV = T_e_eV, u_ms = u_ms),),
                   ions = (ion(mass_amu = 2.014, charge = 1, T_eV = T_i_eV),))
        return spectrum(p, probe, collection, lambdas)
    end

    theta0 = [5e19, 500.0, 250.0, 0.0]
    steps = [1e-5 * theta0[1], 1e-5 * theta0[2], 1e-5 * theta0[3], 1e-2]
    names = ("n_e", "T_e", "T_i", "u")

    J_ad = ForwardDiff.jacobian(spectrum_of_theta, theta0)
    for j in 1:4
        hvec = zeros(4)
        hvec[j] = steps[j]
        fd_col = (spectrum_of_theta(theta0 .+ hvec) .- spectrum_of_theta(theta0 .- hvec)) ./ (2 * steps[j])
        ad_col = @view J_ad[:, j]
        floor_j = 1e-6 * maximum(abs, fd_col)
        @info "SP9 gradient check" param = names[j] worst_abs_diff = maximum(abs.(ad_col .- fd_col)) floor = floor_j
        @test isapprox(ad_col, fd_col; rtol = 1e-6, atol = floor_j)
    end
end

@testset "SP10: gradient wrt an ion fraction" begin
    probe = ProbeBeam(lambda_probe_nm = 263.25)
    collection = Collection(theta_deg = 60)
    lam_i = probe.lambda
    lambdas = collect(range(0.999 * lam_i, 1.001 * lam_i; length = 9))

    function spectrum_of_f(f)
        p = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500),),
                   ions = (ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250, fraction = f),
                           ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 250, fraction = 1 - f)))
        return spectrum(p, probe, collection, lambdas)
    end

    f0 = 0.6
    ad = ForwardDiff.derivative(spectrum_of_f, f0)
    h = 1e-6
    fd = (spectrum_of_f(f0 + h) .- spectrum_of_f(f0 - h)) ./ (2h)
    floor_f = 1e-6 * maximum(abs, fd)
    @info "SP10 gradient check" worst_abs_diff = maximum(abs.(ad .- fd)) floor = floor_f
    @test isapprox(ad, fd; rtol = 1e-6, atol = floor_f)
end

@testset "SP11: allocations and length mismatch" begin
    p = reference_plasma()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    lambdas = collect(range(lambda_to_si(500.0, :lambda_nm), lambda_to_si(560.0, :lambda_nm); length = 512))
    P = similar(lambdas)
    _spec!() = spectrum!(P, p, probe, collection, lambdas)
    _spec!()
    @test (@allocated _spec!()) == 0

    P_bad = zeros(100)
    @test_throws "spectrum!: P has length 100 but lambda has length 512" spectrum!(
        P_bad, p, probe, collection, lambdas)
end

@testset "SP12: Float32 promotes to Float64" begin
    # Values chosen to round-trip exactly through Float32 (small integers and dyadic fractions):
    # 1e19 or 2.014 as a Float32 literal already differs from the Float64 literal by ~1e-7,
    # long before ACCEPTED_UNITS's Float64 factors ever touch it, so that discrepancy would
    # swamp a 1e-14 comparison -- it is Float32's own precision, not a promotion defect. With
    # exactly-representable inputs the promoted Float64 values are bit-identical, isolating the
    # promotion behavior the test is actually about.
    p32 = Plasma(n_e_cm3 = 1f6, electrons = (electrons(T_eV = 500f0),),
                 ions = (ion(mass_amu = 2f0, charge = 1, T_eV = 250f0),))
    probe32 = ProbeBeam(lambda_probe_nm = 526.5f0)
    collection32 = Collection(theta_deg = 90f0)
    p64 = Plasma(n_e_cm3 = 1e6, electrons = (electrons(T_eV = 500.0),),
                 ions = (ion(mass_amu = 2.0, charge = 1, T_eV = 250.0),))
    probe64 = ProbeBeam(lambda_probe_nm = 526.5)
    collection64 = Collection(theta_deg = 90.0)

    @test typeof(p32.n_e) === Float64
    @test typeof(probe32.lambda) === Float64

    lam = lambda_to_si(500.0, :lambda_nm)
    P32 = spectrum(p32, probe32, collection32, [lam])
    P64 = spectrum(p64, probe64, collection64, [lam])
    @test isapprox(P32[1], P64[1]; rtol = 1e-14)
end

@testset "PF3: a Dual plasma's spectrum matches the Float64 value" begin
    p64 = reference_plasma()
    probe = ProbeBeam(lambda_probe_nm = 526.5)
    collection = Collection(theta_deg = 90)
    lambdas = collect(range(lambda_to_si(500.0, :lambda_nm), lambda_to_si(560.0, :lambda_nm); length = 16))
    P64 = spectrum(p64, probe, collection, lambdas)

    Te_dual = ForwardDiff.Dual(500.0, 1.0)
    p_dual = Plasma(n_e_cm3 = 1e19, electrons = (electrons(T_eV = Te_dual),),
                     ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
    P_dual = spectrum(p_dual, probe, collection, lambdas)
    @test isapprox(ForwardDiff.value.(P_dual), P64; rtol = 1e-14)
end
