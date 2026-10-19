# The three test problems of foo2026 Section III, as functions of grid resolution. Spec D24.
# Included by `examples/*.jl` and by `test/instrument/test_examples.jl` (both by explicit
# relative path, so the profiles have one definition).
#
# two_channel_problem, synthesize_channel and single_channel_deck_text (spec D41, Step 4.A.3c,
# chunk 8): the synthetic two-channel problem, added beside the three above.

using TRUST
using Random: Xoshiro
import TOML

"""
    epw_ablation_problem(; nlambda = 512, nt = 256)

foo2026 Section III.A (Figs. 3 top, 4 top): a super-Gaussian electron species with a
time-varying exponent `m(t)`, ablating deuterium ions at fixed temperature.

Spec D24.
"""
function epw_ablation_problem(; nlambda=512, nt=256)
    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    t_ns = collect(range(2.0, 5.6; length=nt))
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    m_of_t(t) = 2.0 + 2.9 * exp(-((t - 4.4) / 1.15)^2)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t, "electrons.e.f.m" => m_of_t))
    lambda = collect(range(220e-9, 260e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
               ConstantBackground(level=25.0))
    return (; name="epw_ablation", trace, probe, collection, lambda, effects,
        noise=StreakCameraNoise(), seed=20260901)
end

"""
    iaw_multiflow_problem(; nlambda = 512, nt = 256)

foo2026 Section III.B (Figs. 3 bottom, 4 bottom): interpenetrating deuteron and carbon flows,
Maxwellian, with a fixed electron background.

Spec D24.
"""
function iaw_multiflow_problem(; nlambda=512, nt=256)
    template = Plasma(n_e_cm3=3e19, electrons=(electrons(T_eV=400.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=200, fraction=0.95),
              ion(name="C", mass_amu=12.011, charge=6, T_eV=200, fraction=0.05)))
    t_ns = collect(range(0.0, 3.0; length=nt))
    n_e_of_t(t) = 3e19 * (10 / 3)^(t / 3.0)
    fD_of_t(t) = 0.95 - 0.45 * (t / 3.0)^2
    fC_of_t(t) = 1 - fD_of_t(t)
    uC_of_t(t) = 500 * exp(-t / 0.90)
    TD_of_t(t) = 200 + 400 * t
    TC_of_t(t) = 200 + 800 * (t / 3.0)
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_of_t, "ions.D.fraction" => fD_of_t, "ions.C.fraction" => fC_of_t,
                "ions.C.u_kms" => uC_of_t, "ions.D.T_eV" => TD_of_t, "ions.C.T_eV" => TC_of_t))
    lambda = collect(range(262.0e-9, 264.5e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6))
    return (; name="iaw_multiflow", trace, probe, collection, lambda, effects,
        noise=StreakCameraNoise(), seed=20260902)
end

"""
    iaw_shock_problem(; nlambda = 512, nt = 256)

foo2026 Section III.C (Fig. 5): a shock front at `t_s = 1.15` ns, everything discontinuous
there, deuteron and carbon ions sharing one flow velocity.

Spec D24.
"""
function iaw_shock_problem(; nlambda=512, nt=256)
    template = Plasma(n_e_cm3=6e19, electrons=(electrons(T_eV=500.0),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=60, fraction=0.78),
              ion(name="C", mass_amu=12.011, charge=6, T_eV=100, fraction=0.22)))
    t_ns = collect(range(0.0, 4.0; length=nt))
    ts = 1.15
    TD_of_t(t) = t < ts ? 60.0 : 60 + 380 * exp(-(t - ts) / 1.60)
    TC_of_t(t) = t < ts ? 100.0 : 100 + 1450 * exp(-(t - ts) / 0.85)
    fD_of_t(t) = t < ts ? 0.78 : 0.63
    fC_of_t(t) = 1 - fD_of_t(t)
    u_of_t(t) = t < ts ? 250.0 : 40.0
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("ions.D.T_eV" => TD_of_t, "ions.C.T_eV" => TC_of_t,
                "ions.D.fraction" => fD_of_t, "ions.C.fraction" => fC_of_t,
                "ions.D.u_kms" => u_of_t, "ions.C.u_kms" => u_of_t))
    lambda = collect(range(262.0e-9, 264.5e-9; length=nlambda))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6))
    return (; name="iaw_shock", trace, probe, collection, lambda, effects,
        noise=StreakCameraNoise(), seed=20260903)
end

"""
    build_experiment(problem) -> Experiment

Builds the `Experiment` for one of the three problem `NamedTuple`s above.
"""
function build_experiment(problem)
    return Experiment(trace=problem.trace, probe=problem.probe, collection=problem.collection,
        lambda_m=problem.lambda, effects=problem.effects)
end

"""
    two_channel_problem(; nlambda_epw = 256, nlambda_iaw = 256, nt_epw = 1024, nt_iaw = 960, nbins = 32)

The synthetic two-channel problem of Phase 4.A Step 4.A.3c (spec D41): the EPW ablation plasma
plus a time-varying deuterium temperature, seen by an EPW and an IAW camera through one probe
and one collection. Each camera has its own window, instrument chain, noise seed and time axis;
the IAW camera's recorded axis runs 0.05 ns behind true time.
"""
function two_channel_problem(; nlambda_epw=256, nlambda_iaw=256, nt_epw=1024, nt_iaw=960, nbins=32)
    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=SuperGaussian(2.0)),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))
    n_e_of_t(t) = 1.5e19 + 6.3e19 * (1 - exp(-(t - 2.0) / 0.75)) - 0.42e19 * (t - 2.0)
    Te_of_t(t) = 350 + 1150 * exp(-(t - 2.0) / 1.35)
    m_of_t(t) = 2.0 + 2.9 * exp(-((t - 4.4) / 1.15)^2)
    TD_of_t(t) = 150 + 350 * (1 - exp(-(t - 2.0) / 1.0))
    leaves = ("n_e_cm3" => n_e_of_t, "electrons.e.T_eV" => Te_of_t,
              "electrons.e.f.m" => m_of_t, "ions.D.T_eV" => TD_of_t)
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    epw = (; t_ns_recorded=collect(range(2.0, 5.8; length=nt_epw)), t_shift_ns=0.0,
        lambda=collect(range(220e-9, 260e-9; length=nlambda_epw)),
        effects=(SpectralIRF(fwhm_nm=0.5), TemporalIRF(fwhm_ns=0.08), Sensitivity(scale=2.0e-4),
                 ConstantBackground(level=25.0)),
        noise=StreakCameraNoise(), seed=20260924)
    iaw = (; t_ns_recorded=collect(range(1.95, 5.75; length=nt_iaw)), t_shift_ns=0.05,
        lambda=collect(range(262.0e-9, 264.5e-9; length=nlambda_iaw)),
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.05), Sensitivity(scale=1.0e-6)),
        noise=StreakCameraNoise(), seed=20260925)
    regions = [AveragingRegion(; nbins=nbins, t_range_ns=[2.2, 5.6])]
    return (; name="two_channel", template, leaves, probe, collection, epw, iaw, regions)
end

"""
    shapes_problem(name::Symbol) -> NamedTuple

`two_channel_problem()`'s NamedTuple with three fields replaced, for `name ∈ (:kappa,
:generalized_normal, :skew_normal)`: the `template`'s electron shape, the shape leaf (the third
of `leaves`), and each camera's `seed` (EPW `20260927`, IAW `20260928`, the same for all three
shapes -- distinct from `two_channel_problem()`'s own `20260924`/`20260925`). Everything else --
the other three leaves (`n_e_cm3`, `electrons.e.T_eV`, `ions.D.T_eV`), both cameras' axes, windows
and instrument chains -- is `two_channel_problem()`'s own.

Spec D39.
"""
function shapes_problem(name::Symbol)
    p = two_channel_problem()
    bump(t) = exp(-((t - 4.4) / 1.15)^2)

    shape, shape_key, shape_of_t = if name === :kappa
        (Kappa(4.0), "electrons.e.f.kappa", t -> 3.0 + 4.0 * bump(t))
    elseif name === :generalized_normal
        (GeneralizedNormal(3.0), "electrons.e.f.m", t -> 2.4 + 2.5 * bump(t))
    elseif name === :skew_normal
        (SkewNormal(0.0), "electrons.e.f.alpha", t -> 2.0 * tanh((t - 3.9) / 0.8))
    else
        throw(ArgumentError("shapes_problem: name must be :kappa, :generalized_normal or :skew_normal, got $name"))
    end

    template = Plasma(n_e_cm3=1.5e19,
        electrons=(electrons(T_eV=350.0, f=shape),),
        ions=(ion(name="D", mass_amu=2.014, charge=1, T_eV=250),))
    leaves = (p.leaves[1], p.leaves[2], shape_key => shape_of_t, p.leaves[4])
    epw = merge(p.epw, (; seed=20260927))
    iaw = merge(p.iaw, (; seed=20260928))

    return (; name=String(name), template, leaves, probe=p.probe, collection=p.collection, epw, iaw, regions=p.regions)
end

"""
    synthesize_channel(problem, name::Symbol) -> StreakData

One camera's synthetic streak, as that camera records it: the trace is built on the camera's
TRUE axis (recorded + shift), synthesized with its own seed (so its temporal IRF acts on its own
axis), and the result is relabelled with the RECORDED axis.
"""
function synthesize_channel(problem, name::Symbol)
    c = getproperty(problem, name)
    trace = PlasmaTrace(problem.template; t_ns=c.t_ns_recorded .+ c.t_shift_ns, leaves=problem.leaves)
    expt = Experiment(; trace=trace, probe=problem.probe, collection=problem.collection,
        lambda_m=c.lambda, effects=c.effects)
    sd = synthesize(expt, c.noise; rng=Xoshiro(c.seed))
    return StreakData(; lambda_m=sd.lambda, t_s=c.t_ns_recorded .* 1e-9, image=sd.image,
        sigma=sd.sigma, mask=sd.mask)
end

"""
    staged_joint_fit(joint_deck_path::AbstractString; optimizer = LevenbergMarquardt(),
                     tol::Real = 1e-6, shape_variable::AbstractString = "m_e") ->
        (; deck, problem, result, decreases, epw, iaw)

The user's decision of 2026-09-25 (D48 item 1, amended by D53): from a two-channel deck's own
inits, the joint LBFGS fit reports `converged` at an objective far above a reachable one (chunk 8's
F2), so Step 4.A.4's tests and example get θ\\* by a **written-out procedure** instead:

1. The EPW-only and IAW-only decks (`single_channel_deck_text`), each fitted alone, every variable
   free, with the deck's own optimizer (D41).
2. The joint problem started from `logn`, `T_e` and `shape_variable` of the EPW-only fit and `T_D`
   of the IAW-only fit -- **by bin index**: the single-channel axes are each channel's own `t̄`, the
   joint axis is the pooled one, and they share the regions and so the bin count.
3. `LevenbergMarquardt()`, then `polish`.

Returns the joint `deck` and `problem`, the polished `result`, `polish`'s own `decreases`, and the
two single-channel `FitResult`s (`epw`, `iaw`), since `slice_scatter` needs the `Deck` and
`laplace`/`calibration_scan` need the `problem`.

Spec D48 item 1, D53. `shape_variable` (Phase 4.B spec D39) names the deck's electron shape
variable in place of the literal `"m_e"`; its default reproduces every existing caller bitwise.

**Reduced to a call at Step 4.F.1** (Phase 4.F spec D8): the procedure is the package's own
`procedure = "staged"` (`TRUST._staged_fit`), called with the deck's own `procedure` for the two
stages, `optimizer` for the joint fit, `tol` for `polish` and `T_D` taken from the IAW stage;
every other fitted variable comes from the EPW stage, the first-listed channel, which for these
decks is exactly `logn`, `T_e` and `shape_variable` (checked). Bitwise the procedure above.
"""
function staged_joint_fit(joint_deck_path::AbstractString; optimizer=LevenbergMarquardt(), tol::Real=1e-6,
    shape_variable::AbstractString="m_e")
    deck = read_deck(joint_deck_path)
    st = TRUST._staged_fit(deck; stage_procedure=deck.sections["optimizer"]["procedure"],
        joint_optimizer=optimizer, polish_tol=tol, start_from=Dict{String,Any}("T_D" => "iaw"))
    [s.name for s in st.stages] == ["epw", "iaw"] || error(
        "staged_joint_fit: the deck's channels must be [\"epw\", \"iaw\"]")
    Dict(st.sources)[shape_variable] == "epw" || error(
        "staged_joint_fit: \"$shape_variable\" did not start from the EPW stage")
    return (; deck, problem=st.problem, result=st.result, decreases=st.decreases,
        epw=st.stages[1].result, iaw=st.stages[2].result)
end

"""
    with_wavelength_calibration(text::AbstractString) -> String

Parses the channel-form deck `text`, adds `wavelength_calibration = { lambda_offset_nm = 0.0 }`
(unbound) to every `[channel.<n>.instrument]`, and re-serializes with `TOML.print(...; sorted =
true)` (D55). `examples/two_channel.toml` itself is not edited -- TC3 and TS1 read it -- so the
example that needs the calibration for `calibration_scan` builds it from the deck's own text
instead.

Spec D55.
"""
function with_wavelength_calibration(text::AbstractString)
    raw = TOML.parse(text)
    for (nm, chan) in raw["channel"]
        instrument = get!(chan, "instrument", Dict{String,Any}())
        instrument["wavelength_calibration"] = Dict{String,Any}("lambda_offset_nm" => 0.0)
    end
    buf = IOBuffer()
    TOML.print(buf, raw; sorted=true)
    return String(take!(buf))
end

"""
    with_staged_procedure(text::AbstractString; stage_procedure, start_from = Dict{String,Any}()) -> String

Parses the channel-form deck `text`, sets its `[optimizer]` `procedure = "staged"` with an
`[optimizer.staged]` table of `stage_procedure` and `start_from` (`joint_procedure` and
`polish_tol` left to their defaults, `"levenberg_marquardt"` and `1e-6`; every other key of
`[optimizer]` kept), and re-serializes with `TOML.print(...; sorted = true)`, as
`with_wavelength_calibration` does. The committed decks are not edited: the diagnostics derive
single-channel decks from their text (Phase 4.F spec D8).
"""
function with_staged_procedure(text::AbstractString; stage_procedure, start_from=Dict{String,Any}())
    raw = TOML.parse(text)
    opt = get!(raw, "optimizer", Dict{String,Any}())
    opt["procedure"] = "staged"
    opt["staged"] = Dict{String,Any}("stage_procedure" => stage_procedure,
        "start_from" => Dict{String,Any}(start_from))
    buf = IOBuffer()
    TOML.print(buf, raw; sorted=true)
    return String(take!(buf))
end

"""
    single_channel_deck_text(text::AbstractString, name::AbstractString) -> String

The single-channel deck derived, in code, from a channel-form deck's own text (spec D41, Section
9.3): parses `text`, restricts `channels` to `[name]`, drops every other `[channel.<other>]`
table, points `[output].dir` at `runs/two_channel/<name>`, and re-serializes it with
`TOML.print`. Every variable, the regions and the regularizer are left exactly as the input deck
gave them -- only the channel selection and the output directory change, so the single-channel
fit is free to move every variable the joint fit is.
"""
function single_channel_deck_text(text::AbstractString, name::AbstractString)
    raw = TOML.parse(text)
    raw["channels"] = Any[name]
    for k in collect(keys(raw["channel"]))
        k == name || delete!(raw["channel"], k)
    end
    raw["output"]["dir"] = "runs/two_channel/$name"
    buf = IOBuffer()
    TOML.print(buf, raw; sorted=true)
    return String(take!(buf))
end

# ------------------------------------------------------------------------------------------
# The shock's jump metric (Phase 4.C spec D15): one definition for the slow tier's FR9 and
# examples/diagnostics/shock_piecewise.jl. Functions only: fit_iaw_shock.jl and iaw_shock.jl both
# include this file, so a script can include it twice.
# ------------------------------------------------------------------------------------------

"""
    iaw_shock_truth(trace::PlasmaTrace) -> Dict{String,Vector{Float64}}

`iaw_shock_problem`'s ground truth at its trace's bin centres, keyed and in the units of the shock
decks' own names: `f_D` (the deuteron fraction), `T_D` and `T_C` (eV), and `u_ion` (km/s, the one
flow the D and C ions share). Spec D15.
"""
function iaw_shock_truth(trace::PlasmaTrace)
    row(path) = collect(Float64, trace.values[findfirst(p -> string(p) == path, trace.paths), :])
    return Dict("f_D" => row("ions.D.fraction"), "T_D" => T_from_si.(row("ions.D.T"), :T_eV),
        "T_C" => T_from_si.(row("ions.C.T"), :T_eV), "u_ion" => u_from_si.(row("ions.D.u"), :u_kms))
end

"""
    iaw_shock_steps() -> Dict{String,Tuple{Float64,Float64}}

`(a, b)` per quantity: `iaw_shock_problem`'s truth just before the shock and just after it, at
`t_s⁺` (the post-shock temperatures decay from there). Spec D15.
"""
iaw_shock_steps() = Dict("f_D" => (0.78, 0.63), "T_D" => (60.0, 440.0), "T_C" => (100.0, 1550.0),
    "u_ion" => (250.0, 40.0))

"""
    jump_window(t_ns; ts = 1.15, W = 0.5) -> Vector{Int}

The bins `j` with `|t_j − ts| ≤ W`, `t_ns` in ns: the jump window J of spec D12. `ts = 1.15` ns is
`iaw_shock_problem`'s shock time.
"""
jump_window(t_ns; ts=1.15, W=0.5) = findall(t -> abs(t - ts) <= W, t_ns)

"""
    jump_rss(row, truth, J) -> Float64

`Σ_{j∈J} (row_j − truth_j)²`, in the quantity's own unit squared; `row` and `truth` full `nt`-bin
rows. Spec D12.
"""
jump_rss(row, truth, J) = sum(abs2, row[J] .- truth[J])

"""
    jump_rss_ratios(values_a, values_b, truth, J) -> (mean, ratios)

`ratios[v] = jump_rss(values_a[v], truth[v], J) / jump_rss(values_b[v], truth[v], J)` for `v` in
`f_D`, `T_D`, `T_C`, and `mean` their mean: FR9's asserted quantity with `a` the piecewise fit and
`b` the whole-axis one. The flow is reported, not compared (spec D12). `values_a` and `values_b` are
`FitResult.values`-like dictionaries of full `nt`-bin rows.
"""
function jump_rss_ratios(values_a, values_b, truth, J)
    ratios = Dict(v => jump_rss(values_a[v], truth[v], J) / jump_rss(values_b[v], truth[v], J)
                  for v in ("f_D", "T_D", "T_C"))
    return sum(values(ratios)) / 3, ratios
end

"""
    jump_crossing(t_ns, u, level, J; ts = 1.15) -> Union{Nothing,Float64}

The time at which `u` crosses `level` nearest `ts`, by linear interpolation between consecutive
bins of the window `J`; `nothing` when it never does there. A segment lying exactly on the level
crosses at its first bin; a non-finite value (a variable's `NaN` outside its interval) is skipped.
Spec D12.
"""
function jump_crossing(t_ns, u, level, J; ts=1.15)
    best = nothing
    for idx in 1:(length(J)-1)
        j, j2 = J[idx], J[idx+1]
        a, b = u[j] - level, u[j2] - level
        (isfinite(a) && isfinite(b)) || continue
        if a == b
            a == 0 || continue
            tc = t_ns[j]
        elseif (a <= 0 <= b) || (b <= 0 <= a)
            tc = t_ns[j] + (t_ns[j2] - t_ns[j]) * a / (a - b)
        else
            continue
        end
        (best === nothing || abs(tc - ts) < abs(best - ts)) && (best = tc)
    end
    return best
end

"""
    jump_width(t_ns, u, a, b, J; ts = 1.15) -> Union{Nothing,Float64}

The 10 to 90 % width of a jump from `a` to `b`: the distance in ns between `u`'s crossings of
`a + 0.1 (b − a)` and `a + 0.9 (b − a)` nearest `ts` (`jump_crossing`); `nothing` when either
crossing does not exist in `J`, which is reported as undefined, never as a number. Spec D12.
"""
function jump_width(t_ns, u, a, b, J; ts=1.15)
    c10 = jump_crossing(t_ns, u, a + 0.1 * (b - a), J; ts=ts)
    c90 = jump_crossing(t_ns, u, a + 0.9 * (b - a), J; ts=ts)
    (c10 === nothing || c90 === nothing) && return nothing
    return abs(c90 - c10)
end
