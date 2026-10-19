using Test
using TRUST
using TRUST: _scale_bins

# Step 4.I.2: the per-bin amplitude (Phase 4.I spec D14 to D18, D20; OQ-59 item 1), gated "per-bin
# amplitude". PA1 to PA4: the effect, measure column by column and the refusals; PA5 to PA9, which
# build the deck's problems, compile the brute-force Jacobian and the dual pass of a chain nothing earlier
# builds, or fit, are in the slow tier (test/slow/test_per_bin_amplitude.jl).

const _PA_A = [2.0e-4, 4.0e-4, 7.0e-4, 1.2e-3]
const _PA_TRACE = PlasmaTrace(Plasma(n_e_cm3=1.5e19, electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.5)),),
    ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250.0),)); t_ns=[2.5, 3.2, 3.9, 4.6])
const _PA_CURVE = (lambda_nm=[220.0, 240.0, 260.0], response=[0.6, 1.0, 0.8])

_pa_expt(s; trace=_PA_TRACE, extra=()) = Experiment(; trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
    collection=Collection(theta_deg=60.0, k_model=:exact), lambda_m=collect(range(225e-9, 255e-9; length=20)),
    effects=(SpectralIRF(fwhm_nm=0.5), s, ConstantBackground(level=25.0), extra...))

# ------------------------------------------------------------------------------------------
# PA1: the effect's two forms
# ------------------------------------------------------------------------------------------

@testset "PA1: Sensitivity's scale, a scalar or one value per bin" begin
    v = Sensitivity(scale=_PA_A)
    @test v.scale == _PA_A && v.scale !== _PA_A && v.scale isa Vector{Float64}
    @test Sensitivity(_PA_A).scale == _PA_A
    @test parameters(v) === (scale=v.scale,)
    @test _scale_bins(v) == 4 && _scale_bins(Sensitivity(scale=2.0)) == 0
    c = Sensitivity(; scale=_PA_A, _PA_CURVE...)
    @test c.scale == _PA_A && c.response == Sensitivity(; scale=1.0, _PA_CURVE...).response
    r = reconstruct(c, (scale=[1.0, 2.0, 3.0, 4.0],))
    @test r.scale == [1.0, 2.0, 3.0, 4.0] && r.response === c.response
    @test reconstruct(c, (scale=3.0,)).scale === 3.0             # either form, either way
    @test reconstruct(Sensitivity(scale=2.0), (scale=_PA_A,)).scale == _PA_A
    @test sprint(show, v) == "Sensitivity(4 bins, 2.0e-04 to 1.2e-03)"
    @test sprint(show, c) == "Sensitivity(4 bins, 2.0e-04 to 1.2e-03, response curve)"
    @test sprint(show, Sensitivity(scale=2.0e-4)) == "Sensitivity(2.0e-04)"
    @test_throws "Sensitivity: scale must be finite" Sensitivity(scale=[1.0, NaN])
    @test_throws "Sensitivity: a per-bin scale must have at least one value" Sensitivity(scale=Float64[])
end

# ------------------------------------------------------------------------------------------
# PA2: measure's column j is the scalar model's at scale = a_j, bitwise (acceptance (b))
# ------------------------------------------------------------------------------------------

@testset "PA2: column j of a per-bin model is the scalar model at a_j" begin
    for curve in ((;), _PA_CURVE)
        M = measure(_pa_expt(Sensitivity(; scale=_PA_A, curve...)))
        for j in 1:4
            Mj = measure(_pa_expt(Sensitivity(; scale=_PA_A[j], curve...)))
            @test all(M[:, j] .=== Mj[:, j])
        end
    end
    @test eltype(measure(_pa_expt(Sensitivity(scale=Float32[1, 2, 3, 4])))) == Float64
end

# ------------------------------------------------------------------------------------------
# PA3: Experiment's two refusals
# ------------------------------------------------------------------------------------------

@testset "PA3: the length and the TemporalIRF refused at Experiment" begin
    @test_throws "Experiment: Sensitivity's per-bin scale has 3 values, but the trace has 4 bins" _pa_expt(Sensitivity(scale=_PA_A[1:3]))
    @test_throws "Experiment: a per-bin Sensitivity scale cannot stand beside a TemporalIRF: the per-bin amplitude is defined after the temporal IRF" _pa_expt(
        Sensitivity(scale=_PA_A); extra=(TemporalIRF(fwhm_ns=0.1),))
    @test _pa_expt(Sensitivity(scale=2.0e-4); extra=(TemporalIRF(fwhm_ns=0.1),)) isa Experiment   # the scalar is unchanged
end

# ------------------------------------------------------------------------------------------
# PA4: the binding rule's one exception, and unflatten gathering the series
# ------------------------------------------------------------------------------------------

@testset "PA4: a time-varying variable binds Sensitivity's scale and nothing else" begin
    expt = _pa_expt(Sensitivity(scale=2.0e-4))
    si = findfirst(e -> e isa Sensitivity, expt.effects)
    bi = findfirst(e -> e isa ConstantBackground, expt.effects)
    ii = findfirst(e -> e isa SpectralIRF, expt.effects)
    a = FittedVariable("a"; bounds=(1e-5, 1e-2), init=2e-4, time_varying=true)
    pspec = ParameterSpec(expt; variables=[a], bindings=["a" => (si, :scale)])
    @test pspec.ntheta == 4
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects)
    e = unflatten(working, pspec, flatten(pspec, ["a" => _PA_A]))
    @test all(e.effects[si].scale .=== [TRUST._from_unconstrained(TRUST._to_unconstrained(x, 1e-5, 1e-2), 1e-5, 1e-2) for x in _PA_A])

    msg = "\"a\" is time-varying and cannot be bound to an instrument parameter other than Sensitivity's scale"
    @test_throws msg ParameterSpec(expt; variables=[a], bindings=["a" => (bi, :level)])
    @test_throws msg ParameterSpec(expt; variables=[a], bindings=["a" => (ii, :sigma)])
    mid = FitInterval("mid", expt.trace.axis; t_range_ns=[3.0, 5.0])
    ai = FittedVariable("a"; bounds=(1e-5, 1e-2), init=2e-4, time_varying=true, interval=1)
    @test_throws "\"a\" lives on an interval, but a per-bin Sensitivity scale needs a value at every bin" ParameterSpec(expt;
        variables=[ai], intervals=[mid], bindings=["a" => (si, :scale)])
    tirf = _pa_expt(Sensitivity(scale=2.0e-4); extra=(TemporalIRF(fwhm_ns=0.1),))
    @test_throws "\"a\" is a per-bin Sensitivity scale, which cannot stand beside a TemporalIRF" ParameterSpec(tirf;
        variables=[a], bindings=["a" => (findfirst(e -> e isa Sensitivity, tirf.effects), :scale)])
end
