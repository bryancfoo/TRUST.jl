# The per-bin amplitude's fixture (Phase 4.I spec D20, D21), shared by
# test/instrument/test_per_bin_amplitude.jl (PA1 to PA4, fast) and its slow namesake under test/slow/
# (PA5 to PA9). Two parts.
#
# (1) _ar_joint(): the Jacobian's problem (PA6, PA7) -- four bins of the two-channel example's plasma,
# its electrons Maxwellian (a SuperGaussian template makes the dual pass allocate whatever the
# amplitude, spec D20, and every dual-pass row before this one is Maxwellian), seen by
# a cut-down EPW camera (20 wavelengths, 225 to 255 nm: a WavelengthCalibration whose offset
# is fitted, so the dual pass carries a kernel global; a SpectralIRF; a Sensitivity with a measured
# response curve and a per-bin scale; a ConstantBackground whose level is fitted, a chain global) and
# an IAW camera (16 wavelengths: a SpectralIRF and a per-bin Sensitivity). n_e, T_e and T_D fitted per
# bin, each camera's amplitude per bin.
#
# (2) The ramp problem (PA8, PA9; OQ-59 acceptance (d) and (e)): examples/problems.jl's
# two_channel_problem cut to 96 wavelengths per camera, 128 and 124 recorded slices and 16 bins, with
# no temporal IRF (a per-bin scale refuses one), each camera's streak synthesized with an amplitude
# ramp imposed on its recorded slices (a per-bin Sensitivity at synthesis, on the camera's own
# axis): x6 across the window on the EPW, 6^((t - 2.2)/3.4), and 0.5 to 2.0 linearly on the IAW. The
# decks are examples/two_channel_staged.toml's, without temporal_irf, with an unbound
# WavelengthCalibration on the EPW (for calibration_scan), and each sensitivity scale a fitted
# variable: one per channel (s_epw, s_iaw, today's form) or one per bin (a_epw, a_iaw), unregularized.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))
import TOML
using Statistics: median

# --- (1) the Jacobian's problem -------------------------------------------------------------

const _AR_CURVE = (lambda_nm=[220.0, 240.0, 260.0], response=[0.6, 1.0, 0.8])
const _AR_A_EPW = [2.0e-4, 4.0e-4, 7.0e-4, 1.2e-3]
const _AR_A_IAW = [0.5e-6, 0.9e-6, 1.4e-6, 2.0e-6]

function _ar_joint()
    template = Plasma(n_e_cm3=1.5e19, electrons=(electrons(T_eV=350.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250.0),))
    trace = PlasmaTrace(template; t_ns=[2.5, 3.2, 3.9, 4.6])
    probe = ProbeBeam(lambda_probe_nm=263.25)
    coll = Collection(theta_deg=60.0, k_model=:exact)
    eE = Experiment(; trace=trace, probe=probe, collection=coll,
        lambda_m=collect(range(225e-9, 255e-9; length=20)),
        effects=(WavelengthCalibration(lambda_offset_nm=0.0), SpectralIRF(fwhm_nm=0.5),
                 Sensitivity(; scale=2.0e-4, _AR_CURVE...), ConstantBackground(level=25.0)))
    eI = Experiment(; trace=trace, probe=probe, collection=coll,
        lambda_m=collect(range(262.5e-9, 264.0e-9; length=16)),
        effects=(SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=1.0e-6)))
    idx(e, T) = findfirst(x -> x isa T, e.effects)
    pspec = ParameterSpec((eE, eI);
        variables=[FittedVariable("ne"; bounds=(5e18, 1e20), init=1.5e19, time_varying=true),
            FittedVariable("Te"; bounds=(100.0, 2000.0), init=350.0, time_varying=true),
            FittedVariable("TD"; bounds=(50.0, 1000.0), init=250.0, time_varying=true),
            FittedVariable("a_epw"; bounds=(1e-5, 1e-2), init=2e-4, time_varying=true),
            FittedVariable("a_iaw"; bounds=(1e-8, 1e-4), init=1e-6, time_varying=true),
            FittedVariable("b"; bounds=(0.0, 100.0), init=25.0),
            FittedVariable("dl"; bounds=(-0.5, 0.5), init=0.01)],
        bindings=["ne" => "n_e_cm3", "Te" => "electrons.e.T_eV", "TD" => "ions.D.T_eV",
            "a_epw" => (channel=1, effect=idx(eE, Sensitivity), field=:scale),
            "a_iaw" => (channel=2, effect=idx(eI, Sensitivity), field=:scale),
            "b" => (channel=1, effect=idx(eE, ConstantBackground), field=:level),
            "dl" => (channel=1, effect=idx(eE, WavelengthCalibration), field=:offset, factor=1e-9,
                unit_key="lambda_offset_nm")])
    working = [Experiment(; trace=pspec.trace, probe=probe, collection=coll, lambda_m=e.lambda,
        effects=e.effects) for e in (eE, eI)]
    theta0 = flatten(pspec, ["a_epw" => _AR_A_EPW, "a_iaw" => _AR_A_IAW])
    data = map(1:2) do c
        image = measure(unflatten(working[c], pspec, theta0; channel=c))
        StreakData(lambda_m=working[c].lambda, t_s=working[c].t, image=image,
            sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    end
    jp = JointFitProblem((FitProblem(working[1], pspec, data[1]; channel=1, name="epw"),
        FitProblem(working[2], pspec, data[2]; channel=2, name="iaw")))
    return jp, theta0 .+ 0.03 .* (1:length(theta0))
end

# --- (2) the ramp problem ------------------------------------------------------------------

const _AR_CUT = (nlambda_epw=96, nlambda_iaw=96, nt_epw=128, nt_iaw=124, nbins=16)
_ar_ramp_epw(t_ns) = 6.0^((t_ns - 2.2) / 3.4)
_ar_ramp_iaw(t_ns) = 0.5 + 1.5 * (t_ns - 2.2) / 3.4

# The two streaks in `dir`, ramped or not, and the problem they were synthesized from.
function _ar_streaks(dir::AbstractString; ramp::Bool)
    p = two_channel_problem(; _AR_CUT...)
    te = p.epw.t_ns_recorded .+ p.epw.t_shift_ns
    ti = p.iaw.t_ns_recorded .+ p.iaw.t_shift_ns
    se = ramp ? 2.0e-4 .* _ar_ramp_epw.(te) : fill(2.0e-4, length(te))
    si = ramp ? 1.0e-6 .* _ar_ramp_iaw.(ti) : fill(1.0e-6, length(ti))
    epw = merge(p.epw, (; effects=(SpectralIRF(fwhm_nm=0.5), Sensitivity(scale=se), ConstantBackground(level=25.0))))
    iaw = merge(p.iaw, (; effects=(SpectralIRF(fwhm_nm=0.08), Sensitivity(scale=si))))
    q = merge(p, (; epw, iaw))
    write_streak(joinpath(dir, "epw.jld2"), synthesize_channel(q, :epw); source="ramp fixture")
    write_streak(joinpath(dir, "iaw.jld2"), synthesize_channel(q, :iaw); source="ramp fixture")
    return p
end

# The staged deck on those streaks: per_bin = false fits one scale per channel (s_epw, s_iaw),
# per_bin = true one per bin (a_epw, a_iaw); every other key is examples/two_channel_staged.toml's.
function _ar_deck(dir::AbstractString; per_bin::Bool)
    raw = TOML.parsefile(joinpath(@__DIR__, "..", "..", "examples", "two_channel_staged.toml"))
    raw["channel"]["epw"]["data"]["file"] = replace(joinpath(dir, "epw.jld2"), "\\" => "/")
    raw["channel"]["iaw"]["data"]["file"] = replace(joinpath(dir, "iaw.jld2"), "\\" => "/")
    raw["fit"]["regions"][1]["nbins"] = _AR_CUT.nbins
    raw["output"]["dir"] = replace(joinpath(dir, per_bin ? "per_bin" : "one_scale"), "\\" => "/")
    for c in ("epw", "iaw")
        delete!(raw["channel"][c]["instrument"], "temporal_irf")
    end
    raw["channel"]["epw"]["instrument"]["wavelength_calibration"] = Dict("lambda_offset_nm" => 0.0)
    ae, ai = per_bin ? ("a_epw", "a_iaw") : ("s_epw", "s_iaw")
    raw["channel"]["epw"]["instrument"]["sensitivity"]["scale"] = ae
    raw["channel"]["iaw"]["instrument"]["sensitivity"]["scale"] = ai
    raw["fit"]["variables"][ae] = Dict("bounds" => [1.0e-5, 1.0e-2], "init" => 2.0e-4, "time_varying" => per_bin)
    raw["fit"]["variables"][ai] = Dict("bounds" => [1.0e-8, 1.0e-4], "init" => 1.0e-6, "time_varying" => per_bin)
    path = joinpath(dir, per_bin ? "per_bin.toml" : "one_scale.toml")
    open(io -> TOML.print(io, raw; sorted=true), path, "w")
    return path
end

# Median and largest relative error of n_e and T_e against the problem's own profiles at the fitted
# axis (TS1's _ts_medians, for two of its four quantities).
function _ar_errors(result::FitResult, p)
    t_ns = result.pspec.axis .* 1e9
    base = Dict(p.leaves)
    ne = base["n_e_cm3"].(t_ns)
    Te = base["electrons.e.T_eV"].(t_ns)
    e_ne = abs.(result.values["ne_cm3"] .- ne) ./ ne
    e_Te = abs.(result.values["T_e"] .- Te) ./ Te
    return (; ne=median(e_ne), ne_max=maximum(e_ne), Te=median(e_Te), Te_max=maximum(e_Te))
end
