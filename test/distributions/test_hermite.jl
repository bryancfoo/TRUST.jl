# Step 5.A.1: HermiteExpansion, the Gram-Charlier shape (Phase 5.A spec D3-D12; OQ-66). Gate "Hermite": HE1 to
# HE11, HE13 and HE14. HE1-HE13 are OQ-66 item 7's (a) to (m); HE14 is the large-|xi| branch of the user's
# decision of 2026-10-16 (Phase 5.A summary U1). HE12 (the recovery fits) and HE15 (fit(deck)'s warning) are
# test/slow/test_hermite.jl's: they compile the fit path for the new shape, about 26 s (the user, U3). Every
# oracle independent of the code under test (conventions.md 8.2): a quadrature, a BigFloat evaluation of the
# Fried-Conte series, ForwardDiff or a finite difference.

using Test
using TRUST
using TRUST: _Zderiv, _Zpp, _hermite_lobe, _hermite_lobe_entries, _warn_hermite_lobe, _cauchy_real,
            _cauchy_upper, _cauchy_lower, _shape_eltype, _widen_shape, HERMITE_NEGATIVE_MASS_WARN,
            ZDERIV_ASYMPTOTIC_XI, SHAPE_REGISTRY
using QuadGK: quadgk
using Random
import ForwardDiff

include(joinpath(@__DIR__, "..", "shape_contract.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "hermite_fixture.jl"))

# The three coefficient sets of OQ-66 item 7 (a): skew only, kurtosis only, all four.
const _HE_SETS = ((0.03, 0.0, 0.0, 0.0), (0.0, 0.01, 0.0, 0.0), (0.02, 0.01, 0.003, 0.001))

# ------------------------------------------------------------------------------------------
# The BigFloat oracle for Z^{(k)} (spec D12): the Fried-Conte power series
#   Z(xi) = i sqrt(pi) e^{-xi^2} - 2 sum_{n>=0} (-2)^n xi^{2n+1} / (2n+1)!!,
# entire, differentiated k times term by term, with (e^{-xi^2})^{(k)} = (-1)^k H_k(xi) e^{-xi^2}. Neither
# erfcx nor the recursion: the series is summed until a term falls below eps(BigFloat) of the sum, past
# its largest term (n about |xi|^2), at a precision covering that term's size.
# ------------------------------------------------------------------------------------------

function _he_zk_big(xi::Number, k::Int)
    prec = max(128, ceil(Int, 2.2 * abs(xi)^2 + 128))
    return setprecision(BigFloat, prec) do
        x = Complex{BigFloat}(big(real(xi)), big(imag(xi)))
        h0, h1 = one(x), 2x
        hk = k == 0 ? h0 : h1
        for n in 1:(k-1)
            h0, h1 = h1, 2x * h1 - 2n * h0
            hk = h1
        end
        herm = im * sqrt(big(pi)) * (-1)^k * hk * exp(-x^2)
        s = zero(x)
        a = big(1)
        n = 0
        while true
            p = 2n + 1
            if p >= k
                term = a * prod(big(p - k + 1):big(p); init = big(1)) * x^(p - k)
                s += term
                (n > 2 * abs(xi)^2 + 10 && abs(term) < abs(s) * eps(BigFloat)) && break
            end
            n += 1
            a *= -2 / big(2n + 1)
        end
        ComplexF64(herm - 2s)
    end
end

# Re Z^{(k)}(x) at real |x| >= 10 by Float64 quadrature (spec D12): (k!/sqrt(pi)) times the integral of
# e^{-t^2}/(t - x)^{k+1} over |t| <= 9, with breakpoints at -3, 0 and 3; the Gaussian beyond |t| = 9 and the
# pole's neighbourhood contribute of order e^{-81}, nothing in Float64. Independent of the series and of the
# recursion. (Over infinite limits QuadGK's transformed interval misses the Gaussian at x = 140: relative
# error 1.0, measured.)
function _he_quad_rezk(x::Float64, k::Int)
    v, _ = quadgk(t -> exp(-t^2) / (t - x)^(k + 1), -9.0, -3.0, 0.0, 3.0, 9.0; rtol = 1e-15, atol = 0)
    return factorial(k) / sqrt(pi) * v
end

# The negative mass by an independent route (spec D12): the factor's sign changes bracketed on a 1e-3 grid
# of |x| <= 6 and refined by bisection, then quadgk of -phi over each negative interval.
function _he_negative_mass(c3, c4)
    fac(x) = 1 + c3 * (8x^3 - 12x) + c4 * (16x^4 - 48x^2 + 12)
    xs = collect(-6.0:1e-3:6.0)
    s = fac.(xs)
    root(a, b) = (fa = fac(a); for _ in 1:100; m = (a + b) / 2; (fac(m) < 0) == (fa < 0) ? (a = m) : (b = m); end; (a + b) / 2)
    total = 0.0
    i = 1
    while i <= length(xs)
        if s[i] < 0
            j = i
            while j < length(xs) && s[j+1] < 0
                j += 1
            end
            lo = i == 1 ? xs[1] : root(xs[i-1], xs[i])
            hi = j == length(xs) ? xs[end] : root(xs[j], xs[j+1])
            total += quadgk(x -> -exp(-x^2) / sqrt(pi) * fac(x), lo, hi; rtol = 1e-13, atol = 0)[1]
            i = j + 1
        else
            i += 1
        end
    end
    return total
end

# The five xi of rows HE6 and HE7 (spec D12): three real, one above and one below the axis, all inside the
# recursion's range.
const _HE_XI5 = (0.4, -1.7, 2.9, 0.8 + 0.6im, -1.5 - 0.3im)

# ------------------------------------------------------------------------------------------
# HE1-HE5: the shape (OQ-66 (a) to (e))
# ------------------------------------------------------------------------------------------

@testset "HE1: the three normalizations" begin
    for cs in _HE_SETS
        check_shape_normalizations(HermiteExpansion(cs...))
    end
end

@testset "HE2: the Maxwellian bitwise at zero coefficients" begin
    # Deliberately bitwise (spec D6): a failure is a defect, not a re-pin.
    f0 = HermiteExpansion()
    M = Maxwellian()
    for x in -20.0:0.25:20.0
        @test f0(x) === M(x)
        @test dfdx(f0, x) === dfdx(M, x)
        @test cauchy_dfdv(f0, x) === Zprime(x)
        for y in (-1.5, -0.3, 0.4, 2.0)
            z = complex(x, y)
            @test f0(z) === M(z)
            @test dfdx(f0, z) === dfdx(M, z)
            @test cauchy_dfdv(f0, z) === Zprime(z)
        end
    end
    @test HermiteExpansion(0.0, 0.0, 0.0, 0.0)(0.7) === M(0.7)
end

@testset "HE3: dfdx against ForwardDiff" begin
    # rtol 1e-12 with conventions.md 8.4's floor, 1e-12 of the largest |dfdx| on the grid (the zeros of phi').
    xs = collect(-5.0:0.05:5.0)
    for cs in _HE_SETS
        f = HermiteExpansion(cs...)
        floor_ = 1e-12 * maximum(x -> abs(dfdx(f, x)), xs)
        for x in xs
            @test isapprox(dfdx(f, x), ForwardDiff.derivative(f, x); rtol = 1e-12, atol = floor_)
        end
    end
end

@testset "HE4: cauchy_dfdv against the quadrature fallback" begin
    # Deliberately loose (spec D12): CAUCHY_RTOL = 1e-10 and the Plemelj split's cancellation bound the
    # oracle; measured worst 3.6e-11 real, 1.5e-9 above the axis, 9.7e-10 below.
    for cs in ((0.03, 0.01, 0.0, 0.0), (0.02, 0.01, 0.003, 0.001))
        f = HermiteExpansion(cs...)
        for xi in -6.0:0.5:6.0
            @test isapprox(cauchy_dfdv(f, xi), _cauchy_real(f, xi); rtol = 1e-8)
        end
        for x in -6.0:1.0:6.0
            for y in (0.1, 0.5, 2.0)
                @test isapprox(cauchy_dfdv(f, complex(x, y)), _cauchy_upper(f, complex(x, y)); rtol = 1e-8)
            end
            for y in (-0.1, -0.5, -1.0)
                @test isapprox(cauchy_dfdv(f, complex(x, y)), _cauchy_lower(f, complex(x, y)); rtol = 1e-8)
            end
        end
    end
end

@testset "HE5: the moments, and skewness and excess kurtosis" begin
    for cs in _HE_SETS
        f = HermiteExpansion(cs...)
        for n in 0:8
            ref, _ = quadgk(x -> x^n * f(x), -Inf, 0.0, Inf; rtol = 1e-13, atol = 1e-14)
            @test isapprox(moment(f, n), ref; atol = 1e-10)
        end
        @test moment(f, 0) === 1.0
        @test moment(f, 1) === 0.0
        @test moment(f, 2) === 0.5
        @test isapprox(skewness(f), 12 * sqrt(2) * cs[1]; rtol = 1e-12)
        @test isapprox(excess_kurtosis(f), 96 * cs[2]; rtol = 1e-12)
    end
    @test_throws ArgumentError moment(HermiteExpansion(c3 = 0.01), -1)
end

# ------------------------------------------------------------------------------------------
# HE6-HE7: the coefficient partials and the Z recursion (OQ-66 (f), (g))
# ------------------------------------------------------------------------------------------

_he_seed(f::HermiteExpansion, n::Int, tag) = reconstruct(f, merge(parameters(f),
    NamedTuple{(Symbol("c$n"),)}((ForwardDiff.Dual{tag}(parameters(f)[Symbol("c$n")], 1.0),))))

_he_iaw_plasma(f) = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500.0, f = f),),
                           ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))
const _HE_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _HE_COLLECTION = Collection(theta_deg = 60, k_model = :exact)

@testset "HE6: the coefficient partials" begin
    # The partial of cauchy_dfdv in c_n is (-1)^n Z^{(n+1)}(xi), exactly in the code (one multiplication);
    # against the BigFloat oracle to rtol 1e-12, at the five xi and at two in the asymptotic branch (U1).
    for base in ((0.02, 0.01, 0.003, 0.001), (0.0, 0.0, 0.0, 0.0))
        f = HermiteExpansion(base...)
        for xi in (_HE_XI5..., 9.0, -20.0), n in 3:6
            c = cauchy_dfdv(_he_seed(f, n, :HE6), xi)
            partial = complex(ForwardDiff.partials(real(c))[1], ForwardDiff.partials(imag(c))[1])
            @test isapprox(partial, (-1)^n * _he_zk_big(xi, n + 1); rtol = 1e-12)
        end
    end
    # The spectral density's partial in each c_n against central differences (SK16's form, floor 1e-6 of
    # the largest |FD| per coefficient), at the four-coefficient point and at zero, where a seeded zero
    # coefficient must still carry its partial (spec D6). h = 1e-7: a coefficient's natural scale is about
    # 1e-3 (H6(2.5) = 1255), and SK16's h = 1e-4 leaves c6's difference 3.2e-2 from the derivative (measured;
    # 3.3e-8 at h = 1e-7).
    h = 1e-7
    for base in ((0.02, 0.01, 0.003, 0.001), (0.0, 0.0, 0.0, 0.0)), n in 3:6
        fd = Float64[]
        ad = Float64[]
        for lam in (240e-9, 245e-9, 250e-9, 276e-9, 284e-9)
            bp = collect(base); bp[n-2] += h
            bm = collect(base); bm[n-2] -= h
            pp = _he_iaw_plasma(HermiteExpansion(bp...))
            pm = _he_iaw_plasma(HermiteExpansion(bm...))
            (k, om, _) = wavevector(_HE_PROBE, _HE_COLLECTION, lam, pp)
            push!(fd, (spectral_density(pp, k, om) - spectral_density(pm, k, om)) / (2h))
            s = spectral_density(_he_iaw_plasma(_he_seed(HermiteExpansion(base...), n, :HE6S)), k, om)
            push!(ad, ForwardDiff.partials(s)[1])
        end
        floor_ = 1e-6 * maximum(abs, fd)
        for (a, d) in zip(ad, fd)
            @test isapprox(a, d; rtol = 1e-6, atol = floor_)
        end
    end
end

@testset "HE7: the Z recursion" begin
    for xi in _HE_XI5
        d = _Zderiv(xi, Val(7))
        @test d[1] === Z(xi)
        @test d[2] === Zprime(xi)
        @test d[3] === _Zpp(xi)
        # Z'' against ForwardDiff of Zprime along the real direction, rtol 1e-12
        zpp = ForwardDiff.derivative(t -> Zprime(xi + t), 0.0)
        @test isapprox(d[3], zpp; rtol = 1e-12)
        # Z^{(k)} for k = 2 to 7 against the BigFloat series, rtol 1e-10 (measured worst 4.6e-14)
        for k in 2:7
            @test isapprox(d[k+1], _he_zk_big(xi, k); rtol = 1e-10)
        end
    end
end

# ------------------------------------------------------------------------------------------
# A FitResult built without fitting (OB10's route, test_objective.jl; spec D12): the recovery's problem
# (test/fixtures/hermite_fixture.jl) unflattened at chosen values, for HE8's warning rows
# ------------------------------------------------------------------------------------------

function _he_result_at(c3, c4, T_e)
    problem, pspec, _ = _he_recovery_problem(SkewNormal(1.0))
    theta = flatten(pspec, ["c3_e" => c3, "c4_e" => c4, "T_e" => T_e])
    values = Dict("c3_e" => [c3], "c4_e" => [c4], "T_e" => [T_e])
    near_bound = Dict(name => BitVector([false]) for name in keys(values))
    return FitResult(theta, unflatten(problem.experiment, pspec, theta), pspec, values, near_bound, 1.0, 0.0, 1.0,
        0.0, Float64[], problem.npixels, NamedTuple{(:iteration, :objective, :chi2, :R),NTuple{4,Float64}}[],
        true, :gradient_norm, :none, 1, 1, 1, 0.1, "test")
end

# ------------------------------------------------------------------------------------------
# HE8-HE10: the lobe report, allocations, the API (OQ-66 (h) to (j))
# ------------------------------------------------------------------------------------------


@testset "HE8: the lobe report" begin
    # Level 1's figures of 2026-10-16, re-measured (spec D7); each mass against an independent quadrature.
    r = _hermite_lobe(HermiteExpansion(c3 = 0.04))
    @test r.factor_min < 0
    @test r.x_min == -6.0
    @test isapprox(r.factor_min, -65.24; rtol = 1e-12)
    @test isapprox(r.negative_mass, _he_negative_mass(0.04, 0.0); rtol = 1e-6)
    @test isapprox(r.negative_mass, 4.232e-3; rtol = 1e-3)
    r = _hermite_lobe(HermiteExpansion(c3 = 0.04, c4 = 0.01))
    @test r.factor_min > 0
    @test isapprox(r.x_min, -2.1; atol = 1e-12)
    @test r.negative_mass === 0.0
    @test _he_negative_mass(0.04, 0.01) === 0.0
    r = _hermite_lobe(HermiteExpansion(c4 = -0.01))
    @test r.factor_min < 0
    @test isapprox(r.negative_mass, _he_negative_mass(0.0, -0.01); rtol = 1e-6)
    @test isapprox(r.negative_mass, 3.750e-3; rtol = 1e-3)
    # show prints the lobe for the first and third and not the second
    @test sprint(show, HermiteExpansion(c3 = 0.04)) ==
          "HermiteExpansion(c3 = 0.04, c4 = 0.0, c5 = 0.0, c6 = 0.0; negative lobe: factor -65.2 at x = -6.0, negative mass 0.00423)"
    @test sprint(show, HermiteExpansion(c3 = 0.04, c4 = 0.01)) == "HermiteExpansion(c3 = 0.04, c4 = 0.01, c5 = 0.0, c6 = 0.0)"
    @test sprint(show, HermiteExpansion(c4 = -0.01)) ==
          "HermiteExpansion(c3 = 0.0, c4 = -0.01, c5 = 0.0, c6 = 0.0; negative lobe: factor -189.0 at x = -6.0, negative mass 0.00375)"
    # a dual coefficient reports as its primal
    @test _hermite_lobe(HermiteExpansion(c3 = ForwardDiff.Dual{:HE8}(0.04, 1.0))) === _hermite_lobe(HermiteExpansion(c3 = 0.04))
    # every other shape has no report
    @test _hermite_lobe(Maxwellian()) === nothing
    @test _hermite_lobe(SkewNormal(0.5)) === nothing
    # the warning, on FitResults built without fitting: a flattened core past the threshold warns, naming the
    # species and the bin, its mass that of the shape rebuilt from result.values; a skew inside the positive
    # region does not
    r_lobe = _he_result_at(0.0, -0.008, 380.0)
    msg = _HE_WARN_HEAD * "\n  electrons.e: bin 1 (largest negative mass 0.00241)"
    entries = @test_logs (:warn, msg) min_level = Base.CoreLogging.Warn _warn_hermite_lobe(r_lobe, "the final fit")
    @test length(entries) == 1
    @test entries[1].species == "electrons.e"
    @test entries[1].bin == 1
    @test entries[1].negative_mass === _hermite_lobe(HermiteExpansion(c3 = r_lobe.values["c3_e"][1], c4 = r_lobe.values["c4_e"][1])).negative_mass
    @test entries[1].negative_mass > HERMITE_NEGATIVE_MASS_WARN
    @test isempty(@test_logs min_level = Base.CoreLogging.Warn _warn_hermite_lobe(_he_result_at(0.01, 0.004, 400.0), "the final fit"))
    # the constructor refuses NaN and Inf, naming the coefficient, and nothing else
    for name in (:c3, :c4, :c5, :c6), bad in (NaN, Inf, -Inf)
        @test_throws "$(name) must be finite" HermiteExpansion(; NamedTuple{(name,)}((bad,))...)
    end
    @test HermiteExpansion(c3 = 5.0, c4 = -3.0, c5 = 1.0, c6 = -2.0) isa HermiteExpansion{Float64}
end

# Barriers (conventions.md 5.2's measurement rules; test_shape_cost.jl's form): the result consumed as a
# number, values and partials, inside a function with the arguments passed in.
_he_consume(z::Complex{<:ForwardDiff.Dual}) = ForwardDiff.value(real(z)) + sum(ForwardDiff.partials(real(z))) +
                                               ForwardDiff.value(imag(z)) + sum(ForwardDiff.partials(imag(z)))
_he_cdfdv(f, xi) = _he_consume(cauchy_dfdv(f, xi))
_he_allocated(f, xi) = (_he_cdfdv(f, xi); @allocated _he_cdfdv(f, xi))

@testset "HE9: four dual coefficients allocate nothing" begin
    D(v, i) = ForwardDiff.Dual{:HE9}(v, ntuple(j -> j == i ? 1.0 : 0.0, 4)...)
    fd = HermiteExpansion(D(0.02, 1), D(0.01, 2), D(0.003, 3), D(0.001, 4))
    for xi in (1.3, 1.3 + 0.2im, 1.3 - 0.2im, 9.0, -25.0)
        @test _he_allocated(fd, xi) == 0
    end
end

@testset "HE10: parameters, reconstruct, show, promotion, the dual species" begin
    f = HermiteExpansion(0.02, 0.01, 0.003, 0.001)
    @test parameters(f) == (c3 = 0.02, c4 = 0.01, c5 = 0.003, c6 = 0.001)
    @test keys(parameters(HermiteExpansion())) == (:c3, :c4, :c5, :c6)
    g = reconstruct(f, (c3 = -0.01, c4 = 0.0, c5 = 0.0, c6 = 0.002))
    @test parameters(g) == (c3 = -0.01, c4 = 0.0, c5 = 0.0, c6 = 0.002)
    @test HermiteExpansion(c4 = 0.01) === HermiteExpansion(0.0, 0.01, 0.0, 0.0)
    @test typeof(HermiteExpansion(1, 0, 0, 0)) === HermiteExpansion{Float64}
    @test typeof(HermiteExpansion(0.1f0, 0.0f0, 0.0f0, 0.0f0)) === HermiteExpansion{Float32}
    @test sprint(show, HermiteExpansion()) == "HermiteExpansion(c3 = 0.0, c4 = 0.0, c5 = 0.0, c6 = 0.0)"
    d = ForwardDiff.Dual{:HE10}(0.02, 1.0)
    fd = HermiteExpansion(c3 = d)
    @test fd isa HermiteExpansion{typeof(d)}
    @test _shape_eltype(fd) === typeof(d)
    sp = electrons(T_eV = 400.0, f = fd)
    @test sp.T isa typeof(d)
    w = _widen_shape(f, typeof(d))
    @test w isa HermiteExpansion{typeof(d)}
    @test ForwardDiff.value(w.c5) === 0.003
end

# ------------------------------------------------------------------------------------------
# HE11: registration (OQ-66 (k))
# ------------------------------------------------------------------------------------------

@testset "HE11: registration" begin
    @test SHAPE_REGISTRY["HermiteExpansion"] === HermiteExpansion
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _he11_write_streak(datafile)
        outdir = joinpath(dir, "out")
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _he11_deck(datafile, outdir))
        deck = read_deck(deckpath)
        shape = deck.sections["plasma"]["electrons"]["e"]["shape"]
        @test shape["type"] == "HermiteExpansion"
        @test shape["c5"] === 0.002
        @test shape["c6"] === 0.0

        problem, optimizer, start = TRUST.build_problem(deck)
        b3 = only(b for b in problem.pspec.bindings if b.name == "c3_e")
        b4 = only(b for b in problem.pspec.bindings if b.name == "c4_e")
        @test b3.kind === :leaf && string(b3.path) == "electrons.e.f.c3"
        @test b4.kind === :leaf && string(b4.path) == "electrons.e.f.c4"
        @test only(v for v in problem.pspec.variables if v.name == "c3_e").time_varying
        @test !only(v for v in problem.pspec.variables if v.name == "c4_e").time_varying
        ft = problem.experiment.trace.template.electrons["e"].f
        @test ft isa HermiteExpansion
        @test ForwardDiff.value(ft.c5) == 0.002 && ForwardDiff.value(ft.c6) == 0.0

        eff = effective_deck(deck, problem, optimizer)
        @test occursin("HermiteExpansion", eff)
        efftmp = joinpath(dir, "effective.toml")
        write(efftmp, eff)
        shape2 = read_deck(efftmp).sections["plasma"]["electrons"]["e"]["shape"]
        @test shape2["type"] == "HermiteExpansion"
        @test shape2["c5"] === 0.002
        @test shape2["c6"] === 0.0
        @test string(shape2["c3"]) == string(shape["c3"])

        bad = replace(_he11_deck(datafile, outdir), "c5 = 0.002 }" => "c5 = 0.002, c7 = 0.1 }")
        badpath = joinpath(dir, "bad.toml")
        write(badpath, bad)
        @test_throws "unknown key \"c7\"" read_deck(badpath)

        template = Plasma(; n_e_cm3 = 3e19, electrons = [electrons(T_eV = 400.0, f = HermiteExpansion(0.02, -0.01, 0.003, 0.001))],
            ions = [ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0)])
        trace = PlasmaTrace(template; t_ns = [0.0, 1.0])
        tracepath = joinpath(dir, "trace.jld2")
        write_trace(tracepath, trace)
        trace2 = read_trace(tracepath; as = :trace)
        @test parameters(trace2.template.electrons["e"].f) === parameters(trace.template.electrons["e"].f)
    end
end

# ------------------------------------------------------------------------------------------
# HE13-HE14: the IAW asymmetry and the large-|xi| branch (OQ-66 (m); U1). HE12 is the slow tier's.
# ------------------------------------------------------------------------------------------
@testset "HE13: the IAW asymmetry, pinned" begin
    iaw(f) = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500.0),),
                    ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0, f = f),))
    plasma_M = iaw(Maxwellian())
    (k, _, _) = wavevector(_HE_PROBE, _HE_COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    omegas = collect(range(0.2, 3.0; length = 1401)) .* (k * cs)
    idx = argmax([spectral_density(plasma_M, k, om) for om in omegas])
    @test idx == 309
    omega0 = omegas[idx]
    ratio(f) = (p = iaw(f); spectral_density(p, k, omega0) / spectral_density(p, k, -omega0))
    @test isapprox(ratio(HermiteExpansion()), 1.0; rtol = 1e-12)
    r = ratio(HermiteExpansion(c3 = 0.03))
    @info "HE13 ratio at c3 = 0.03" r
    @test isapprox(r, 1.2678644320604842; rtol = 1e-7)
    @test isapprox(r * ratio(HermiteExpansion(c3 = -0.03)), 1.0; rtol = 1e-9)
end

@testset "HE14: the large-|xi| branch" begin
    # The switch (spec D5): below ZDERIV_ASYMPTOTIC_XI the recursion, so element 3 is _Zpp's; at it the series.
    x0 = prevfloat(ZDERIV_ASYMPTOTIC_XI)
    @test _Zderiv(x0, Val(7))[3] === _Zpp(x0)
    @test _Zderiv(ZDERIV_ASYMPTOTIC_XI, Val(7))[3] !== _Zpp(ZDERIV_ASYMPTOTIC_XI)
    # Z^{(k)} for k = 2 to 7 against the BigFloat series: below the switch the recursion's own error,
    # deliberately loose (rtol 1e-6 at 6.0, measured 1.4e-7; 1e-5 just below 7, 1.1e-6), which holds the
    # switch from below; rtol 1e-10 at |xi| = 7 (1.3e-11), 1e-12 at 7.5 (1.8e-13), 1e-14 from 8 on (5.3e-16);
    # the imaginary parts to 1e-14 everywhere.
    for (x, tol) in ((6.0, 1e-6), (x0, 1e-5), (7.0, 1e-10), (-7.0, 1e-10), (7.5, 1e-12), (8.0, 1e-14), (10.0, 1e-14),
                     (-12.0, 1e-14), (20.0, 1e-14))
        d = _Zderiv(x, Val(7))
        for k in 2:7
            ref = _he_zk_big(x, k)
            @test isapprox(real(d[k+1]), real(ref); rtol = tol)
            @test isapprox(imag(d[k+1]), imag(ref); rtol = 1e-14)
        end
    end
    # far out, against the Float64 quadrature oracle, real parts
    for x in (30.0, -50.0, 140.0), k in 2:7
        @test isapprox(real(_Zderiv(x, Val(7))[k+1]), _he_quad_rezk(x, k); rtol = 1e-12)
    end
    # the dual path through the branch: d/dxi of Z^{(k)} is Z^{(k+1)}
    for x in (7.5, 9.0, -20.0, 140.0), k in 2:6
        ad = ForwardDiff.derivative(t -> real(_Zderiv(t, Val(7))[k+1]), x)
        @test isapprox(ad, real(_Zderiv(x, Val(7))[k+2]); rtol = 1e-13)
    end
    # the shape's Cauchy integral far out, the case the branch exists for
    f = HermiteExpansion(0.02, 0.01, 0.003, 0.001)
    for x in (30.0, -50.0, 140.0)
        ref = Zprime(x) - 0.02 * _he_quad_rezk(x, 4) + 0.01 * _he_quad_rezk(x, 5) - 0.003 * _he_quad_rezk(x, 6) +
              0.001 * _he_quad_rezk(x, 7)
        @test isapprox(real(cauchy_dfdv(f, x)), real(ref); rtol = 1e-12)
    end
end
