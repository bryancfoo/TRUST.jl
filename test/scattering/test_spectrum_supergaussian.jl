using Test
using OTIS
using OTIS: _sg_cauchy_reference
import ForwardDiff

# D22's regression geometry and plasma (foo2026 Section III.A, the EPW problem).

_ss_geometry() = ScatteringGeometry(lambda_probe_nm = 263.25, theta_deg = 60, k_model = :exact)

_ss_plasma(m; f_electron = SuperGaussian(m)) = Plasma(
    n_e_cm3 = 5e19,
    electrons = (electrons(T_eV = 500, f = f_electron),),
    ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))

const _SS_LAMBDA = collect(range(235e-9, 255e-9; length = 512))

# The peak method of Phase 1 spec D23, reused verbatim by spec D22: the parabolic offset from
# the grid maximum, its position and value. Copied rather than imported across @safetestset
# boundaries (spec Section 6).
function peak_position_and_value(x, y)
    i = argmax(y)
    (i == 1 || i == length(y)) && return x[i], y[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    h = x[i] - x[i - 1]
    d = (y0 - y2) / (2 * (y0 - 2y1 + y2))
    return x[i] + d * h, y1 - (y0 - y2) * d / 4
end
peak_position(x, y) = first(peak_position_and_value(x, y))

# D22's FWHM: the difference of the two half-maximum crossings against the parabolic peak value
# y_pk, each found by linear interpolation at the first pair of adjacent grid points that
# bracket y_pk/2, walking outward from the discrete argmax i. No golden section, no refinement,
# no re-gridding (spec D22).
function fwhm(x, y)
    i = argmax(y)
    _, y_pk = peak_position_and_value(x, y)
    half = y_pk / 2
    n = length(y)

    j = i
    while j > 1 && y[j] > half
        j -= 1
    end
    x_lo = x[j] + (half - y[j]) / (y[j + 1] - y[j]) * (x[j + 1] - x[j])

    k = i
    while k < n && y[k] > half
        k += 1
    end
    x_hi = x[k - 1] + (half - y[k - 1]) / (y[k] - y[k - 1]) * (x[k] - x[k - 1])

    return x_hi - x_lo
end

# A test-only adapter, not new package source: it forces cauchy_dfdv through the reference
# quadrature path unconditionally, so SS4 can check the shape and the geometry are right
# *before* SS5 blames the table for anything (spec Section 9's pitfall 5). Every other part of
# the shape interface delegates to the wrapped SuperGaussian unchanged.
struct _ReferencePathSG{FT,L} <: OTIS.AbstractVelocityDistribution
    inner::SuperGaussian{FT,L}
end
(w::_ReferencePathSG)(x) = w.inner(x)
OTIS.dfdx(w::_ReferencePathSG, x) = OTIS.dfdx(w.inner, x)
OTIS.moment(w::_ReferencePathSG, n::Integer) = OTIS.moment(w.inner, n)
OTIS.cauchy_dfdv(w::_ReferencePathSG, xi::Number) = _sg_cauchy_reference(w.inner, xi)
OTIS.parameters(w::_ReferencePathSG) = OTIS.parameters(w.inner)
OTIS.reconstruct(w::_ReferencePathSG, v::NamedTuple) = _ReferencePathSG(OTIS.reconstruct(w.inner, v))

# D22's nine printed numbers, through the reference (quadrature) path, this document's ground
# truth for the regression case.
const _SS_PRINTED = Dict(
    2 => (position = 243.6392118419, fwhm = 8.0142134446, peak = 7.7194597121e7),
    3 => (position = 242.9148147324, fwhm = 6.1240102688, peak = 6.9782948589e7),
    5 => (position = 242.4828506348, fwhm = 2.6954666553, peak = 6.7149235477e7),
)

@testset "SS1: reduction to Maxwellian at m=2" begin
    g = _ss_geometry()
    p_ref = _ss_plasma(2.0; f_electron = _ReferencePathSG(SuperGaussian(2.0)))
    p_tab = _ss_plasma(2.0)
    p_mw = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500),),
                  ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
    P_ref = spectrum(p_ref, g, _SS_LAMBDA)
    P_tab = spectrum(p_tab, g, _SS_LAMBDA)
    P_mw = spectrum(p_mw, g, _SS_LAMBDA)
    maxP = maximum(abs, P_mw)
    worst_ref = maximum(abs.(P_ref .- P_mw)) / maxP
    worst_tab = maximum(abs.(P_tab .- P_mw)) / maxP
    @info "SS1 reduction to Maxwellian, worst normalized error" worst_ref worst_tab
    @test all(isapprox.(P_ref, P_mw; rtol = 1e-12, atol = 1e-12 * maxP))
    @test all(isapprox.(P_tab, P_mw; rtol = 1e-8, atol = 1e-8 * maxP))
end

@testset "SS2: d(spectrum)/dm vs central differences" begin
    g = _ss_geometry()
    h = 1e-5
    for m in (2.0, 3.5)
        # Plain T_eV = 500.0: Step 2.0 made the companion-dual seeding unnecessary. electrons()
        # now promotes FT over values(parameters(f)) as well as the scalar fields, so a dual m
        # already makes the species (and the plasma) dual with a zero partial on T_e, with no
        # need to seed one by hand (spec Step 2.0 item 2, OQ-9 trap 6).
        function spec_of_m(mm)
            p = Plasma(n_e_cm3 = 5e19,
                       electrons = (electrons(T_eV = 500.0, f = SuperGaussian(mm)),),
                       ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
            return spectrum(p, g, _SS_LAMBDA)
        end
        ad = ForwardDiff.derivative(spec_of_m, m)
        fd = (spec_of_m(m + h) .- spec_of_m(m - h)) ./ (2h)
        floor_ = 1e-6 * maximum(abs, fd)
        @info "SS2 dP/dm check" m = m worst_abs_diff = maximum(abs.(ad .- fd)) floor = floor_
        @test all(isapprox.(ad, fd; rtol = 1e-6, atol = floor_))
    end
end

@testset "SS3: joint (T_e, m) gradient" begin
    g = _ss_geometry()
    lambdas8 = collect(range(235e-9, 255e-9; length = 8))
    function spec_of_theta(theta)
        Te, mm = theta
        p = Plasma(n_e_cm3 = 5e19,
                   electrons = (electrons(T_eV = Te, f = SuperGaussian(mm)),),
                   ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
        return spectrum(p, g, lambdas8)
    end
    theta0 = [500.0, 3.5]
    steps = [1e-5 * theta0[1], 1e-5 * theta0[2]]
    names = ("T_e", "m")
    J_ad = ForwardDiff.jacobian(spec_of_theta, theta0)
    for j in 1:2
        hvec = zeros(2)
        hvec[j] = steps[j]
        fd_col = (spec_of_theta(theta0 .+ hvec) .- spec_of_theta(theta0 .- hvec)) ./ (2 * steps[j])
        ad_col = @view J_ad[:, j]
        floor_j = 1e-6 * maximum(abs, fd_col)
        @info "SS3 gradient check" param = names[j] worst_abs_diff = maximum(abs.(ad_col .- fd_col)) floor = floor_j
        @test isapprox(ad_col, fd_col; rtol = 1e-6, atol = floor_j)
    end
end

@testset "SS4: the regression case through the reference path" begin
    g = _ss_geometry()
    for m in (2, 3, 5)
        p = _ss_plasma(Float64(m); f_electron = _ReferencePathSG(SuperGaussian(Float64(m))))
        P = spectrum(p, g, _SS_LAMBDA)
        lam_nm = _SS_LAMBDA .* 1e9
        pos, val = peak_position_and_value(lam_nm, P)
        width = fwhm(lam_nm, P)
        printed = _SS_PRINTED[m]
        @info "SS4 reference vs printed" m = m dpos = pos - printed.position dfwhm = width - printed.fwhm dval = val - printed.peak
        @test isapprox(pos, printed.position; rtol = 1e-9)
        @test isapprox(width, printed.fwhm; rtol = 1e-9)
        @test isapprox(val, printed.peak; rtol = 1e-9)
    end
end

@testset "SS5: the regression case through the table" begin
    g = _ss_geometry()
    lam_nm = _SS_LAMBDA .* 1e9
    peaks = Dict{Int,Float64}()
    widths = Dict{Int,Float64}()
    for m in (2, 3, 5)
        p = _ss_plasma(Float64(m))
        P = spectrum(p, g, _SS_LAMBDA)
        pos, val = peak_position_and_value(lam_nm, P)
        width = fwhm(lam_nm, P)
        printed = _SS_PRINTED[m]
        @info "SS5 table vs printed" m = m dpos = pos - printed.position dfwhm = width - printed.fwhm dval = val - printed.peak
        @test isapprox(pos, printed.position; rtol = 1e-6)
        @test isapprox(width, printed.fwhm; rtol = 1e-6)
        @test isapprox(val, printed.peak; rtol = 1e-6)
        peaks[m] = pos
        widths[m] = width
    end
    @test widths[5] < widths[3] < widths[2]
    blue_shift = peaks[2] - peaks[5]
    @test isapprox(blue_shift, 1.16; rtol = 0.01)
end

@testset "SS6: allocations" begin
    g = _ss_geometry()
    p = _ss_plasma(3.0)
    P = similar(_SS_LAMBDA)
    _spec!() = spectrum!(P, p, g, _SS_LAMBDA)
    _spec!()
    @test (@allocated _spec!()) == 0
end

@testset "SS7: the documented slow path" begin
    g = _ss_geometry()
    lam32 = collect(range(235e-9, 255e-9; length = 32))
    p = _ss_plasma(8.5)
    P = spectrum(p, g, lam32)
    @test all(isfinite, P)
    @test all(>(0), P)

    p_ref = _ss_plasma(8.5; f_electron = _ReferencePathSG(SuperGaussian(8.5)))
    P_ref = spectrum(p_ref, g, lam32)
    worst = maximum(abs.(P .- P_ref)) / maximum(abs, P_ref)
    @info "SS7 slow path vs forced reference, worst normalized error" worst
    @test isapprox(P, P_ref; rtol = 1e-12)
end

@testset "SS8: the dual path" begin
    g = _ss_geometry()
    p64 = _ss_plasma(3.0)
    P64 = spectrum(p64, g, _SS_LAMBDA)

    @testset "dual on T_e" begin
        Te_dual = ForwardDiff.Dual(500.0, 1.0)
        p_dual_Te = Plasma(n_e_cm3 = 5e19,
                            electrons = (electrons(T_eV = Te_dual, f = SuperGaussian(3.0)),),
                            ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
        P_dual_Te = spectrum(p_dual_Te, g, _SS_LAMBDA)
        worst_Te = maximum(abs.(ForwardDiff.value.(P_dual_Te) .- P64)) / maximum(abs, P64)
        @info "SS8 dual T_e path vs Float64, worst normalized diff" worst_Te
        @test isapprox(ForwardDiff.value.(P_dual_Te), P64; rtol = 1e-14)
    end

    @testset "dual on m" begin
        # Plain T_eV = 500.0: Step 2.0 made the companion-dual seeding unnecessary (see SS2's
        # comment). electrons() now promotes FT over values(parameters(f)), so a dual m already
        # makes the species and the plasma dual with a zero partial on T_e.
        m_dual = ForwardDiff.Dual(3.0, 1.0)
        p_dual_m = Plasma(n_e_cm3 = 5e19,
                           electrons = (electrons(T_eV = 500.0, f = SuperGaussian(m_dual)),),
                           ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250),))
        P_dual_m = spectrum(p_dual_m, g, _SS_LAMBDA)
        worst_m = maximum(abs.(ForwardDiff.value.(P_dual_m) .- P64)) / maximum(abs, P64)
        @info "SS8 dual m path vs Float64, worst normalized diff" worst_m
        @test isapprox(ForwardDiff.value.(P_dual_m), P64; rtol = 1e-14)
    end
end

@testset "HK4: the D22 regression plasma runs spectrum with no companion seeding (Step 2.0)" begin
    # _ss_plasma(m) passes m straight through to SuperGaussian(m) and T_eV = 500 plain (no
    # `* one(m)` companion): with electrons() now promoting FT over values(parameters(f)), a
    # dual m alone is enough to make the whole plasma, and spectrum's output, dual.
    g = _ss_geometry()
    m_dual = ForwardDiff.Dual(3.0, 1.0)
    p_dual = _ss_plasma(m_dual)
    p64 = _ss_plasma(3.0)
    P_dual = spectrum(p_dual, g, _SS_LAMBDA)
    P64 = spectrum(p64, g, _SS_LAMBDA)
    @test eltype(P_dual) <: ForwardDiff.Dual
    @test isapprox(ForwardDiff.value.(P_dual), P64; rtol = 1e-14)
end
