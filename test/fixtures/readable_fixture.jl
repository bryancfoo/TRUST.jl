# The readable output's fixture (Phase 4.M spec D16), shared by test/data/test_readable.jl (RD1 to RD10,
# the fast tier) and test/slow/test_readable.jl (RD11 to RD13, the slow tier). Two cameras after
# test_uncertainty.jl's SX fixture (a: 24 wavelengths and a background; b: 16, recorded 0.02 ns early),
# twelve slices each, T_D and u_D ramps in time, T_e = 400 eV, n_e = 3e19 cm^-3: the fast tier's decks
# keep SX's own effect chains and variables, whose problem types the "uncertainty" gate has compiled; the
# slow tier's RD deck adds an interval, a held series, two dependents, two scalars and a frozen variable.
# Every package the files need is reached through OTIS (test/slow's environment has no JLD2, TOML or
# Random of its own).

const _RD_JLD2 = parentmodule(OTIS.jldopen)
const _RD_TOML = OTIS.TOML

# Every leaf dataset of a JLD2 file, or of one group of it: name => value (test_result.jl's
# _walk_leaf_values).
function _rd_leaves(f, prefix="")
    vals = Pair{String,Any}[]
    container = isempty(prefix) ? f : f[prefix]
    for k in keys(container)
        full = isempty(prefix) ? k : "$prefix/$k"
        v = f[full]
        if v isa _RD_JLD2.Group
            append!(vals, _rd_leaves(f, full))
        else
            push!(vals, full => v)
        end
    end
    return vals
end

# A file's leaf datasets as a Dict.
_rd_dict(path::AbstractString) = OTIS.jldopen(f -> Dict(_rd_leaves(f)), path, "r")

const _RD_PLAIN = Union{AbstractArray{<:Real},AbstractArray{<:AbstractString},AbstractString,Real}

_rd_path(p::AbstractString) = replace(p, "\\" => "/")

function _rd_write(dir::AbstractString, name::AbstractString, text::AbstractString)
    path = joinpath(dir, name)
    write(path, text)
    return path
end

# fit(deck) with its warnings discarded (the RD deck's background ends on its bound, by design).
_rd_fit(deck) = Base.CoreLogging.with_logger(() -> fit(deck), Base.CoreLogging.NullLogger())

_rd_template() = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
    ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
_rd_TD(t_ns) = 180.0 + 60.0 * t_ns
_rd_uD(t_ns) = 20.0 + 15.0 * t_ns

# One camera's streak at `path`: the truth on the recorded times `t_rec` (ns) shifted by `shift` (ns),
# `nl` wavelengths from `l0` to `l1` (nm); sigma from GaussianNoise(a = 121, b = 141.75); noise
# sigma .* randn drawn from Xoshiro(seed) when `seed` is given, none otherwise; the first `masked`
# wavelength rows unused at every time (a mask is written only when `masked > 0`).
function _rd_streak(path, t_rec, shift, l0, l1, nl, scale, background; seed=nothing, masked=0)
    trace = PlasmaTrace(_rd_template(); t_ns=collect(t_rec .+ shift),
        leaves=("ions.D.T_eV" => _rd_TD, "ions.D.u_kms" => _rd_uD))
    lambda = collect(range(l0, l1; length=nl)) .* 1e-9
    effects = background ? (Sensitivity(scale=scale), ConstantBackground(level=25.0)) :
              (Sensitivity(scale=scale),)
    expt = Experiment(; trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact), lambda_m=lambda, effects=effects)
    image = measure(expt)
    sig = sigma(GaussianNoise(; a=121.0, b=141.75), image)
    seed === nothing || (image = image .+ sig .* randn(OTIS.Xoshiro(seed), size(image)...))
    mask = nothing
    if masked > 0
        mask = fill(true, size(image))
        mask[1:masked, :] .= false
    end
    write_streak(path, StreakData(; lambda_m=lambda, t_s=t_rec .* 1e-9, image=image, sigma=sig, mask=mask);
        source="rd_test")
    return path
end

_rd_streak_a(path; kw...) = _rd_streak(path, collect(range(0.05, 1.15; length=12)), 0.0, 262.0, 264.5,
    24, 1.0e-6, true; kw...)
_rd_streak_b(path; kw...) = _rd_streak(path, collect(range(0.03, 1.13; length=12)), 0.02, 262.5, 264.0,
    16, 2.0e-6, false; kw...)

const _RD_GEOMETRY = "lambda_probe_nm = 263.25\ntheta_deg = 60.0\nk_model = \"exact\""

const _RD_IONS = """
[plasma.ions.D]
mass_amu = 2.014
charge = 1
fraction = 1.0
T_eV = "T_D"
u_kms = "u_D"
"""

const _RD_REGIONS = "[[fit.regions]]\nt_range_ns = [0.0, 1.2]\nnbins = 3"

const _RD_REGULARIZER = "[regularizer]\nweight = 1.0\nterms = [\n  { select = [\"T_D\", \"u_D\"], operator = \"D2\", normalization = \"relative\" },\n]"

# The fast tier's deck, test_uncertainty.jl's SXA (T_D and u_D free and time-varying on three averaged
# bins, T_e fixed, levenberg_marquardt): `te_free` makes T_e a free scalar (SXB), `regions` and
# `regularizer` replace their blocks (a one-window deck, RD9), `readable` adds the key to [output].
function _rd_sx_text(pa, pb, outdir; regions=_RD_REGIONS, regularizer=_RD_REGULARIZER, te_free=false,
    readable=nothing)
    te = te_free ? "\"T_e\"" : "400.0"
    te_var = te_free ? "T_e = { bounds = [100.0, 1000.0], init = 380.0 }\n" : ""
    out = readable === nothing ? "" : "readable = $readable\n"
    return """
    deck_version = "otis-deck-2"
    channels = ["a", "b"]

    $regions

    [channel.a.data]
    file = "$(_rd_path(pa))"
    [channel.a.geometry]
    $_RD_GEOMETRY
    [channel.a.instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [channel.b.data]
    file = "$(_rd_path(pb))"
    t_shift_ns = 0.02
    [channel.b.geometry]
    $_RD_GEOMETRY
    [channel.b.instrument]
    sensitivity = { scale = 2.0e-6 }

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }
    $te_var
    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = $te

    $_RD_IONS
    $regularizer

    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_rd_path(outdir))"
    $out"""
end

# Camera a alone in the shorthand form, on its own twelve recorded slices (nothing averaged), as
# test_uncertainty.jl's SX5 deck.
function _rd_single_text(pa, outdir)
    return """
    deck_version = "otis-deck-2"

    [data]
    file = "$(_rd_path(pa))"
    [geometry]
    $_RD_GEOMETRY
    [instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = 25.0 }]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true }

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0

    $_RD_IONS
    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_rd_path(outdir))"
    """
end

# The slow tier's RD deck: T_D on the whole axis, u_D on the interval "late" (bins 2 and 3), T_e held
# at a start file's series, the scalars s_b (camera b's sensitivity) and bg (camera a's background, its
# truth 25 below its lower bound 26, so the fit ends on the bound), w_b frozen (camera b's spectral
# width, fwhm_nm), and the dependents ne (bound to n_e_cm3) and ratio (bound to nothing).
function _rd_deck_text(pa, pb, hold, outdir)
    return """
    deck_version = "otis-deck-2"
    channels = ["a", "b"]

    $_RD_REGIONS

    [channel.a.data]
    file = "$(_rd_path(pa))"
    [channel.a.geometry]
    $_RD_GEOMETRY
    [channel.a.instrument]
    sensitivity = { scale = 1.0e-6 }
    background = [{ type = "Constant", level = "bg" }]

    [channel.b.data]
    file = "$(_rd_path(pb))"
    t_shift_ns = 0.02
    [channel.b.geometry]
    $_RD_GEOMETRY
    [channel.b.instrument]
    spectral_irf = { fwhm_nm = "w_b" }
    sensitivity = { scale = "s_b" }

    [fit.intervals.late]
    t_range_ns = [0.5, 1.2]

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0, time_varying = true }
    u_D = { bounds = [-200.0, 200.0], init = 10.0, time_varying = true, interval = "late" }
    T_e = { bounds = [100.0, 1000.0], init = 380.0, time_varying = true }
    s_b = { bounds = [1.0e-6, 4.0e-6], init = 2.2e-6 }
    bg = { bounds = [26.0, 40.0], init = 30.0 }
    w_b = { bounds = [0.001, 0.5], init = 0.05, vary = false }

    [fit.dependent]
    ne = "3.0e19 + 0.0 * T_D"
    ratio = "T_D / T_e"

    [fit.hold]
    file = "$(_rd_path(hold))"
    variables = ["T_e"]

    [plasma]
    n_e_cm3 = "ne"

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = "T_e"

    $_RD_IONS
    [regularizer]
    weight = 1.0
    terms = [
      { select = ["T_D"], operator = "D2", normalization = "relative" },
    ]

    [optimizer]
    procedure = "levenberg_marquardt"

    [fit]
    sigma = "supplied"

    [output]
    dir = "$(_rd_path(outdir))"
    """
end

# The leaf datasets a readable file holds for `result` (channel names `names`), with the optional
# datasets and groups as written: the layout's key set, pinned (OQ-63 (f)).
function _rd_expected_keys(result, names; deck::Bool, gn::Bool, laplace=nothing, scatter=nothing)
    want = Set(["layout_version", "README", "source", "t_ns", "bin_lo_ns", "bin_hi_ns",
        "fit/chi2", "fit/chi2_per_n", "fit/R", "fit/objective", "fit/status", "fit/converged",
        "fit/termination", "fit/iterations", "fit/npixels", "fit/ntheta"])
    deck && push!(want, "deck")
    gn && push!(want, "fit/gn_decrease")
    for nm in names, k in ("lambda_nm", "data", "sigma", "model", "mask", "chi2", "npixels", "chi2_total", "weight")
        push!(want, "channels/$nm/$k")
    end
    for nm in keys(result.values), g in ("params", "units", "flags")
        push!(want, "$g/$nm")
    end
    if laplace !== nothing
        push!(want, "laplace/decrease")
        for nm in keys(laplace.sigma_post), g in ("sigma", "sigma_samp", "near_bound")
            push!(want, "laplace/$g/$nm")
        end
    end
    if scatter !== nothing
        for k in ("t_ns", "bin", "status", "decrease", "chi2")
            push!(want, "scatter/$k")
        end
        for nm in keys(scatter.values), g in ("params", "near_bound", "sigma_stat")
            push!(want, "scatter/$g/$nm")
        end
        for nm in names, k in ("data", "sigma", "mask", "model")
            push!(want, "scatter/channels/$nm/$k")
        end
    end
    return want
end

# chi2 per bin of one channel's group in a readable file, recomputed from its data, model, sigma and
# mask: (chi2 per bin, fitted pixels per bin).
function _rd_chi2_bins(f, prefix::AbstractString)
    D, M = f["$prefix/data"], f["$prefix/model"]
    S, K = f["$prefix/sigma"], f["$prefix/mask"]
    nl, nt = size(D)
    chi2 = zeros(nt)
    n = zeros(Int, nt)
    for j in 1:nt, l in 1:nl
        K[l, j] == 1 || continue
        chi2[j] += ((M[l, j] - D[l, j]) / S[l, j])^2
        n[j] += 1
    end
    return chi2, n
end
