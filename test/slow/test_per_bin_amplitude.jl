using Test
using TRUST
using TRUST: jvp!, jtvp!, _gauss_newton_diagonal, _kernel_effects, _widen_trace, _make_cache, _jacobian_bin!,
    JACOBIAN_CHUNK, _collect_kernel_bindings
using LinearAlgebra: dot
import Random
const ForwardDiff = TRUST.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

# Step 4.I.2's slow rows (Phase 4.I spec D20, D21), gated "per-bin amplitude (slow)": PA5 builds a
# deck's problems on a chain nothing earlier builds; PA6 and PA7 compile the brute-force Jacobian and
# the per-bin dual pass of one; PA8 and PA9 fit. The fast rows are
# test/instrument/test_per_bin_amplitude.jl's. fit(deck) pins BLAS to one thread; this file restores
# the count it found, as TS2 does.

include(joinpath(@__DIR__, "..", "fixtures", "amplitude_ramp_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "staged_fixture.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "start_hold_fixture.jl"))

const _PAS_NBLAS = pin_blas_threads()
pin_blas_threads(_PAS_NBLAS)

# ------------------------------------------------------------------------------------------
# PA5: the deck -- sensitivity = { scale = "a_epw" } with a_epw time-varying; nothing new to write
# ------------------------------------------------------------------------------------------

@testset "PA5: a per-bin amplitude from a deck" begin
    mktempdir() do dir
        _st_streaks(dir)
        notirf(raw) = foreach(c -> delete!(raw["channel"][c]["instrument"], "temporal_irf"), ("epw", "iaw"))
        perbin(raw) = (raw["channel"]["epw"]["instrument"]["sensitivity"]["scale"] = "a_epw";
            raw["fit"]["variables"]["a_epw"] = Dict("bounds" => [1.0e-5, 1.0e-2], "init" => 2.0e-4, "time_varying" => true))
        plain = _sh_edit(notirf, _st_text(dir))
        p0, _, _ = TRUST.build_problem(read_deck(_st_write(dir, "plain.toml", plain)))
        text = _sh_edit(raw -> (notirf(raw); perbin(raw)), _st_text(dir))
        deck = read_deck(_st_write(dir, "pa.toml", text))
        problem, _, start = TRUST.build_problem(deck)
        @test problem.pspec.ntheta == p0.pspec.ntheta + 4
        epw = unflatten(problem.channels[1].experiment, problem.pspec, start; channel=1)
        s = only(e for e in epw.effects if e isa Sensitivity)
        @test s.scale isa Vector{Float64} && length(s.scale) == 4 && all(isapprox.(s.scale, 2.0e-4; rtol=1e-14))
        @test read_deck(_st_write(dir, "eff.toml", effective_deck(deck, problem, LBFGS()))) == deck

        @test_throws "\"a_epw\" is a per-bin Sensitivity scale, which cannot stand beside a TemporalIRF" TRUST.build_problem(
            read_deck(_st_write(dir, "tirf.toml", _sh_edit(perbin, _st_text(dir)))))
        width = _sh_edit(_st_text(dir)) do raw
            notirf(raw)
            raw["channel"]["iaw"]["instrument"]["spectral_irf"]["fwhm_nm"] = "w"
            raw["fit"]["variables"]["w"] = Dict("bounds" => [0.01, 0.2], "init" => 0.08, "time_varying" => true)
        end
        @test_throws "\"w\" is time-varying and cannot be bound to an instrument parameter other than Sensitivity's scale" TRUST.build_problem(
            read_deck(_st_write(dir, "w.toml", width)))
    end
end

# ------------------------------------------------------------------------------------------
# PA6: the Jacobian with a per-bin amplitude, against brute-force ForwardDiff (acceptance (c))
# ------------------------------------------------------------------------------------------

@testset "PA6: the Jacobian with a per-bin amplitude" begin
    jp, theta = _ar_joint()
    J = jacobian(jp, theta)
    for c in 1:2
        @test length(J.channels[c].chain_blocks) == 4 && size(J.channels[c].chain_blocks[1], 2) == 1
        @test size(J.channels[c].response) == (length(jp.channels[c].experiment.lambda), 4)
    end
    J_brute = ForwardDiff.jacobian(th -> residuals(jp, th), theta)
    @test maximum(abs, J_brute) > 0
    ntheta = jp.pspec.ntheta
    nres = size(J_brute, 1)
    J_ours = zeros(nres, ntheta)
    worst = 0.0
    for k in 1:ntheta
        e = zeros(ntheta); e[k] = 1.0
        c = zeros(nres)
        jvp!(c, jp, J, e)
        J_ours[:, k] = c
        ref = @view J_brute[:, k]
        worst = max(worst, maximum(abs.(c .- ref) ./ max.(abs.(ref), 1e-10 * maximum(abs, ref))))
    end
    @info "PA6: worst relative error against ForwardDiff" worst
    @test worst <= 1e-10

    # the adjoint identity, JT2's form
    rng = Random.Xoshiro(20261007)
    adj = 0.0
    for _ in 1:5
        v = randn(rng, ntheta)
        w = randn(rng, nres)
        o1 = zeros(nres); jvp!(o1, jp, J, v)
        o2 = zeros(ntheta); jtvp!(o2, jp, J, w)
        adj = max(adj, abs(dot(o1, w) - dot(v, o2)) / abs(dot(o1, w)))
        @test isapprox(dot(o1, w), dot(v, o2); rtol=1e-12)
    end
    # the Gauss-Newton diagonal, MC6's form
    d = _gauss_newton_diagonal(jp, J)
    dense = vec(sum(abs2, J_ours; dims=1))
    @test all(k -> isapprox(d[k], dense[k]; rtol=1e-12), 1:ntheta)
    @info "PA6: the adjoint identity and the diagonal" adj diag = maximum(abs.(d .- dense) ./ dense)
end

# ------------------------------------------------------------------------------------------
# PA7: the per-bin dual pass is untouched by the amplitude and allocates nothing
# ------------------------------------------------------------------------------------------

function _pa_dual_pass_bytes(c::Int)
    jp, theta = _ar_joint()
    problem = jp.channels[c]
    pspec = problem.pspec
    expt = unflatten(problem.experiment, pspec, theta; channel=c)
    nvar = length(pspec.variables)
    nlambda = length(expt.lambda)
    kernel_effects0 = _kernel_effects(expt)
    kernel_bindings = _collect_kernel_bindings(pspec, expt, length(kernel_effects0); channel=c)
    kernel_global_var = [i for i in 1:nvar if pspec.variables[i].vary && !pspec.variables[i].time_varying &&
                         any(b -> b.slot == i && b.kind === :effect && b.channel == c && b.effect <= length(kernel_effects0), pspec.bindings)]
    G = length(kernel_global_var)
    kernel_global_idx = [pspec.offsets[i] for i in kernel_global_var]
    kernel_global_of_var = zeros(Int, nvar)
    for (gk, i) in enumerate(kernel_global_var)
        kernel_global_of_var[i] = gk
    end
    DFT = ForwardDiff.Dual{TRUST._JacobianTag,Float64,JACOBIAN_CHUNK}
    work_trace = _widen_trace(pspec.trace, DFT)
    vals = Vector{DFT}(undef, pspec.nslots)
    col_cache = _make_cache(DFT, nlambda)
    block = Matrix{Float64}(undef, nlambda, length(pspec.bin_entries[2]))
    kernel_globals = Array{Float64,3}(undef, nlambda, 4, G)
    flag = TRUST._prefactor_flag(expt.prefactor)
    _jb!() = _jacobian_bin!(block, kernel_globals, 2, work_trace, vals, col_cache, pspec, nvar, G,
        kernel_global_idx, kernel_global_of_var, kernel_effects0, kernel_bindings, expt.probe,
        expt.collection, expt.lambda, flag, theta, expt.gain)
    _jb!()
    return (@allocated _jb!()), G
end

@testset "PA7: the dual pass with a per-bin amplitude allocates nothing" begin
    bytes1, G1 = _pa_dual_pass_bytes(1)
    bytes2, G2 = _pa_dual_pass_bytes(2)
    @test G1 == 1 && G2 == 0          # the calibration offset, on the EPW alone
    @test bytes1 == 0
    @test bytes2 == 0
end

# ------------------------------------------------------------------------------------------
# PA8: the ramp problem (acceptance (e)) -- the three fits, once
# ------------------------------------------------------------------------------------------

function _pa_ramp_fits()
    root = mktempdir()
    flat_dir, ramp_dir = joinpath(root, "flat"), joinpath(root, "ramp")
    mkpath(flat_dir); mkpath(ramp_dir)
    p = _ar_streaks(flat_dir; ramp=false)
    _ar_streaks(ramp_dir; ramp=true)
    nblas = pin_blas_threads()
    flat = fit(_ar_deck(flat_dir; per_bin=false))
    one_scale = fit(_ar_deck(ramp_dir; per_bin=false))
    per_bin_path = _ar_deck(ramp_dir; per_bin=true)
    per_bin = fit(per_bin_path)
    pin_blas_threads(nblas)
    return (; p, flat, one_scale, per_bin, per_bin_path)
end

const _PA_RAMP = _pa_ramp_fits()

@testset "PA8: the ramp problem -- one scale biased, one amplitude per bin recovers" begin
    r = _PA_RAMP
    e_flat = _ar_errors(r.flat, r.p)
    e_one = _ar_errors(r.one_scale, r.p)
    e_bin = _ar_errors(r.per_bin, r.p)
    t_ns = r.per_bin.pspec.axis .* 1e9
    ratio_epw = r.per_bin.values["a_epw"] ./ (2.0e-4 .* _ar_ramp_epw.(t_ns))
    ratio_iaw = r.per_bin.values["a_iaw"] ./ (1.0e-6 .* _ar_ramp_iaw.(t_ns))
    @info "PA8: median (largest) relative error, un-ramped one scale" ne = e_flat.ne ne_max = e_flat.ne_max Te = e_flat.Te Te_max = e_flat.Te_max
    @info "PA8: ramped, one scale per channel (the bias)" ne = e_one.ne ne_max = e_one.ne_max Te = e_one.Te Te_max = e_one.Te_max
    @info "PA8: ramped, one amplitude per bin" ne = e_bin.ne ne_max = e_bin.ne_max Te = e_bin.Te Te_max = e_bin.Te_max
    @info "PA8: the recovered amplitude over the imposed ramp, smallest and largest" epw = extrema(ratio_epw) iaw = extrema(ratio_iaw)
    # the bound is the un-ramped staged fit's own accuracy on this fixture (spec D21)
    @test e_flat.ne <= 0.02 && e_flat.Te <= 0.02
    @test e_bin.ne <= 0.02
    @test e_bin.Te <= 0.02
    @test e_one.ne > 0.1               # the pathology, visible: deliberately loose
    @test maximum(abs.(ratio_epw .- 1)) <= 0.2
    @test maximum(abs.(ratio_iaw .- 1)) <= 0.2
end

# ------------------------------------------------------------------------------------------
# PA9: the reports run on a problem with a per-bin amplitude (acceptance (d))
# ------------------------------------------------------------------------------------------

@testset "PA9: gn_decrease, polish, laplace, calibration_scan and slice_scatter with a per-bin amplitude" begin
    result = _PA_RAMP.per_bin
    deck = read_deck(_PA_RAMP.per_bin_path)
    problem, _, _ = TRUST.build_problem(deck)
    theta = result.theta
    @test gn_decrease(problem, theta) <= 1e-6
    pol = polish(problem, result; tol=1e-6)
    @test pol.result.theta == theta
    lap = laplace(problem, theta)
    @test length(lap.sigma_samp["a_epw"]) == 16 && length(lap.sigma_samp["a_iaw"]) == 16
    @test count(isfinite, lap.sigma_samp["a_epw"]) + count(lap.near_bound["a_epw"]) == 16
    @test all(s -> s > 0, filter(isfinite, lap.sigma_samp["a_epw"]))
    scan = calibration_scan(problem, theta, [-0.01, 0.0, 0.01]; channel=1)
    @test scan.objective[2] <= result.objective          # the zero shift refits from theta*: no worse
    @test length(scan.delta["a_epw"][:, 1]) == 16
    mktempdir() do dir
        # slice_scatter ignores [fit.start] (spec D12): this file covers the coarse bins' times alone,
        # which the fine windows reach past, so a window that read it would be refused
        sp = write_start(joinpath(dir, "start.jld2"); axis_s=result.pspec.axis,
            values=Dict("T_D" => result.values["T_D"], "T_e" => result.values["T_e"]), source="PA9")
        text = read(_PA_RAMP.per_bin_path, String)
        deck_s = read_deck(_st_write(dir, "s.toml", _sh_tables(text; start=(sp, ["T_D"]))))
        sc = slice_scatter(deck_s, result)
        @test length(sc.s_stat["a_epw"]) == 16 && length(sc.s_stat["a_iaw"]) == 16
        # T_e, not T_D: the deck's start_from names T_D, and read_deck refuses a held name there.
        # Phase 4.N spec D9: a held series is no longer refused (OQ-64 item 2), but this deck holds T_e,
        # which result fitted, so result is not this deck's fit
        deck_h = read_deck(_st_write(dir, "h.toml", _sh_tables(text; hold=(sp, ["T_e"]))))
        @test_throws "slice_scatter: result was not fitted from this deck" slice_scatter(deck_h, result)
    end
    @info "PA9: laplace's sigma for a_epw relative to the value, smallest and largest" rel = extrema(filter(isfinite, lap.sigma_samp["a_epw"] ./ result.values["a_epw"]))
end

pin_blas_threads(_PAS_NBLAS)
