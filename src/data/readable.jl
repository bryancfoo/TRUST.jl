# write_readable: the otis-readable-2 layout, the readable companion fit(deck) writes beside
# result.jld2 (Phase 4.M, Step 4.M.1; OQ-63; docs/derivations/data_layout.md Section 6). One fit as a
# user reads it: the time and wavelength axes in ns and nm, per channel the data, sigma, model, mask
# and the chi2 per bin, every parameter in its deck unit, and the Laplace sigma and the scatter's
# windows when given. Plain arrays and strings only, as every OTIS layout; no reader in the package,
# whose own record of a fit stays result.jld2.

using JLD2: jldopen

"""
    READABLE_LAYOUT_VERSION

The `layout_version` string written to every readable file: `"otis-readable-2"` since Phase 4.N
(spec D7, [OQ-64](../../dev/open_questions.md#oq-64) item 4: `scatter/p16`, `median`, `p84` and
`held` joined the scatter group), `"otis-readable-1"` before (Phase 4.M spec D5). Nothing in the
package reads the file back.
"""
const READABLE_LAYOUT_VERSION = "otis-readable-2"

# The file's own README dataset (Phase 4.M spec D9; Phase 4.N spec D7), ASCII so that any HDF5
# reader decodes it.
const _READABLE_README = """
otis-readable-2: one OTIS fit, written to be read directly (h5py or any HDF5 reader).
Orientation: arrays are stored as Julia writes them, (n_lambda, n_t), so h5py reads them as
(n_t, n_lambda): data[j] is time bin j's spectrum against lambda_nm, one row per time bin.
Units: times in ns, wavelengths in nm; params/<name> in the unit units/<name> names, the deck key
the name is bound through (T_eV, n_e_cm3, u_kms, fwhm_nm, ...), "-" for a dimensionless target, a
key already in SI (u_ms, n_e_m3, mass_kg) or a name bound to nothing.
t_ns: the fitted bins' centres; bin_lo_ns, bin_hi_ns: each bin's averaging window when the data were
averaged, else equal to t_ns.
channels/<name>/: lambda_nm; data, sigma, model, every pixel; mask, 1 where the pixel was fitted;
chi2 and npixels, per bin over the fitted pixels; chi2_total, their sum; weight, the channel's.
params/<name>: every fitted variable and dependent, NaN outside a variable's interval, one value for
a scalar variable. flags/<name>, the same shape: F fitted; D dependent; H not fitted (held at a
[fit.hold] series or frozen at its init); B fitted and ended near a bound, so not a measurement;
"" where params is NaN.
fit/: chi2 (the weighted total over channels), chi2_per_n (chi2 / npixels), R, objective, status,
converged, termination, iterations, npixels, ntheta, gn_decrease when known.
laplace/ (only when written with a Laplace result): sigma/<name>, the raw Laplace sigma per fitted
variable in its unit (to scale it, multiply by sqrt(fit/chi2_per_n)); sigma_samp/<name> beside it;
near_bound/<name>, 1 where the entry was left out; decrease.
scatter/ (only when written with a scatter result): one entry per fine window: t_ns, bin (the fitted
bin it lies in), status, decrease, chi2, params/<name>, near_bound/<name>; sigma_stat/<name>, one per
fitted bin, the detrended spread of its windows' values, never divided by sqrt(m); p16/<name>,
median/<name>, p84/<name>, one per fitted bin, the 16th, 50th and 84th percentiles of its windows'
values (type 7, numpy's default), over every window whatever its status; held/<name>, one per window,
the value a [fit.hold] variable was held at; NaN in params, sigma_stat, the percentiles and held where
the variable's interval leaves the window's bin out (the variable was then frozen at its init);
channels/<name>/data, sigma, mask, model, (n_lambda, n_w), each window as its one-bin fit saw it.
deck: the effective deck's text, when given. source: "fit" or "write_readable".
No string in the file is empty but the elements of a flags array.
The package's own record of the fit is result.jld2 beside this file.
"""

"""
    write_readable(path, result::FitResult, problem::AbstractFitProblem;
                   deck = nothing, laplace = nothing, scatter = nothing) -> path

Writes the readable companion of a fit to `path` (`docs/derivations/data_layout.md` Section 6,
[OQ-63](../../dev/open_questions.md#oq-63); Phase 4.M spec D5 to D12), the layout `otis-readable-2`:
`layout_version`, `README` (what the file holds, its orientation and its units), `source`
(`"write_readable"`; `"fit"` when `fit(deck)` writes it), `deck` when given and not empty (the
effective deck's text; h5py cannot read an empty scalar string, so no string the file holds is one,
and a unit it has no name for is written `"-"`); the time axis `t_ns` and each bin's window `bin_lo_ns`, `bin_hi_ns`; per channel
`channels/<name>/` with `lambda_nm`, `data`, `sigma`, `model` and `mask` (`(n_λ, n_t)`, every pixel;
`mask` `UInt8`, 1 where fitted), `chi2` and `npixels` per bin over the fitted pixels, `chi2_total`
and `weight`; `params/<name>`, `units/<name>` and `flags/<name>` for every variable and dependent, in
the deck units `FitResult.values` already holds them in; the `fit/` summary; and, only when given,
`laplace/` (`laplace`'s raw σ) and `scatter/` (`slice_scatter`'s windows, each with its own data and
model, the per-bin quantiles and the held values, Phase 4.N spec D7). **The call rewrites the whole
file**: opened with `"w"`, no append, no partial update.

`problem` is the problem `result` was fitted from, or one rebuilt from the same deck
(`OTIS.build_problem(read_deck(path))`): the data, σ and mask come from it and the model from
`result`, so its channel names, its time axis (bitwise), its `ntheta` and each channel's image size
must match `result`'s, a loud error otherwise. `laplace` and `scatter` must have been computed at
`result.theta` exactly, and `scatter` must carry its windows' data and its quantiles (a
`ScatterResult` from `slice_scatter`, not one built with the eleven- or sixteen-field constructor).
`fit/gn_decrease` is written when
`laplace` is given (its `decrease`, the Gauss-Newton decrease at the same θ).

`result.jld2` is untouched: this file is for reading, not a round trip, and nothing in the package
reads it. Plain arrays and strings only ([OQ-15](../../dev/open_questions.md#oq-15)).
"""
function write_readable(path::AbstractString, @nospecialize(result::FitResult),
    @nospecialize(problem::AbstractFitProblem);
    deck::Union{Nothing,AbstractString}=nothing, laplace::Union{Nothing,LaplaceResult}=nothing,
    scatter::Union{Nothing,ScatterResult}=nothing)
    gn = laplace === nothing ? nothing : laplace.decrease
    return _write_readable(path, result, problem; deck=deck, laplace=laplace, scatter=scatter,
        source="write_readable", gn_decrease=gn)
end

"""
    _write_readable(path, result, problem; deck, laplace, scatter, source, gn_decrease) -> path

`write_readable`'s body, with the two keywords `fit(deck)` passes and a script does not (Phase 4.M
spec D6): `source`, `"fit"` or `"write_readable"`, and `gn_decrease`, `nothing` or the number written
as `fit/gn_decrease`.
"""
function _write_readable(path::AbstractString, @nospecialize(result::FitResult),
    @nospecialize(problem::AbstractFitProblem);
    deck::Union{Nothing,AbstractString}=nothing, laplace::Union{Nothing,LaplaceResult}=nothing,
    scatter::Union{Nothing,ScatterResult}=nothing, source::AbstractString="write_readable",
    gn_decrease::Union{Nothing,Real}=nothing)
    _check_readable_problem(result, problem)
    if laplace !== nothing
        laplace.theta == result.theta || throw(ArgumentError(
            "write_readable: laplace was computed at a different θ than result.theta"))
    end
    if scatter !== nothing
        scatter.theta == result.theta || throw(ArgumentError(
            "write_readable: scatter was computed at a different θ than result.theta"))
        length(scatter.data) == length(result.channels) || throw(ArgumentError(
            "write_readable: scatter holds the windows of $(length(scatter.data)) channel(s), but result " *
            "has $(length(result.channels)); pass the ScatterResult slice_scatter returned"))
        # Phase 4.N spec D7: a ScatterResult built with the sixteen-field constructor has no quantiles
        missing_q = sort([name for name in keys(scatter.values) if !(haskey(scatter.p16, name) &&
                                                                       haskey(scatter.median, name) && haskey(scatter.p84, name))])
        isempty(missing_q) || throw(ArgumentError(
            "write_readable: scatter holds no quantiles for $missing_q; pass the ScatterResult " *
            "slice_scatter returned"))
    end

    jldopen(path, "w") do f
        f["layout_version"] = READABLE_LAYOUT_VERSION
        f["README"] = _READABLE_README
        f["source"] = String(source)
        (deck === nothing || isempty(deck)) || (f["deck"] = String(deck))
        t_ns = t_from_si.(result.pspec.axis, :t_ns)
        bin_lo_ns, bin_hi_ns = _readable_bin_edges(problem, result.pspec.axis)
        f["t_ns"] = Vector{Float64}(t_ns)
        f["bin_lo_ns"] = bin_lo_ns
        f["bin_hi_ns"] = bin_hi_ns
        _readable_channels!(f, result, problem)
        _readable_params!(f, result)
        _readable_fit!(f, result, gn_decrease)
        laplace === nothing || _readable_laplace!(f, result, laplace)
        scatter === nothing || _readable_scatter!(f, result, scatter)
    end
    return path
end

"""
    _check_readable_problem(result::FitResult, problem::AbstractFitProblem) -> nothing

`write_readable`'s check that `problem` is the one `result` was fitted from or one rebuilt from the same
deck (Phase 4.M spec D7): the channel names in order, the time axis bitwise, `ntheta`, and each
channel's image size against `result`'s experiments. A loud error naming the first mismatch.
"""
function _check_readable_problem(@nospecialize(result::FitResult), @nospecialize(problem::AbstractFitProblem))
    nch = length(result.channels)
    pnames = problem isa JointFitProblem ? String[ch.name for ch in problem.channels] : String[problem.name]
    rnames = String[c.name for c in result.channels]
    pnames == rnames || throw(ArgumentError(
        "write_readable: problem has channels $pnames, but result has $rnames; pass the problem the " *
        "result was fitted from, or one rebuilt from the same deck"))
    problem.pspec.axis == result.pspec.axis || throw(ArgumentError(
        "write_readable: problem's time axis is not result's; pass the problem the result was fitted " *
        "from, or one rebuilt from the same deck"))
    problem.pspec.ntheta == result.pspec.ntheta || throw(ArgumentError(
        "write_readable: problem has ntheta = $(problem.pspec.ntheta), but result has " *
        "$(result.pspec.ntheta); pass the problem the result was fitted from, or one rebuilt from the same deck"))
    for c in 1:nch
        ch = _channel_problem(problem, c)
        want = (length(result.experiments[c].lambda), length(result.pspec.axis))
        size(ch.data.image) == want || throw(ArgumentError(
            "write_readable: channel \"$(rnames[c])\"'s data is $(size(ch.data.image)), but result's model " *
            "is $want; pass the problem the result was fitted from, or one rebuilt from the same deck"))
    end
    return nothing
end

"""
    _readable_bin_edges(problem::AbstractFitProblem, axis::AbstractVector) -> (bin_lo_ns, bin_hi_ns)

Each fitted bin's averaging window in ns (Phase 4.M spec D8): from channel 1's `regions_starts`,
`regions_ends` and `regions_nbins` meta keys (`average_regions`'s, `src/data/average.jl`) when the
data carry them and their windows count exactly the fitted bins, window `k` of a region `[a, b]` with
`n` bins and `h = (b - a)/n` being `[a + (k-1)h, a + kh]` and the last ending at `b` (`average_regions`'s
own arithmetic); otherwise both equal to `t_from_si.(axis, :t_ns)`, as for data never averaged.
"""
function _readable_bin_edges(@nospecialize(problem::AbstractFitProblem), axis::AbstractVector)
    meta = _channel_problem(problem, 1).data.meta
    nt = length(axis)
    keys_present = all(k -> haskey(meta, k), ("regions_starts", "regions_ends", "regions_nbins"))
    if keys_present && sum(meta["regions_nbins"]) == nt
        lo = Vector{Float64}(undef, nt)
        hi = Vector{Float64}(undef, nt)
        k = 0
        for (a, b, n) in zip(meta["regions_starts"], meta["regions_ends"], meta["regions_nbins"])
            h = (b - a) / n
            for i in 1:n
                k += 1
                lo[k] = a + (i - 1) * h
                hi[k] = i == n ? b : a + i * h
            end
        end
        return t_from_si.(lo, :t_ns), t_from_si.(hi, :t_ns)
    end
    t_ns = Vector{Float64}(t_from_si.(axis, :t_ns))
    return t_ns, copy(t_ns)
end

"""
    _readable_channels!(f, result::FitResult, problem::AbstractFitProblem) -> nothing

`channels/<name>/` for each channel of `result`, in order (Phase 4.M spec D10): `lambda_nm` from the
fitted experiment's axis; `data`, `sigma` and `mask` (`UInt8`) the channel problem's own, every
pixel; `model` `measure(result.experiments[c])`, the expression `write_result` writes as the result
file's `channels/<name>/model`; and the χ² per bin, computed once here over the problem's
`pixel_index` (`_readable_chi2_bins`), `npixels` per bin beside it, `chi2_total` their sum, and the
channel's `weight`. Like every function of this file it is not specialized on the result's or the
problem's type (`@nospecialize`, Phase 4.M spec D6): `fit(deck)` calls it for every problem type it
fits, and it is compiled once.
"""
function _readable_channels!(f, @nospecialize(result::FitResult), @nospecialize(problem::AbstractFitProblem))
    for (c, summ) in enumerate(result.channels)
        prefix = "channels/$(summ.name)"
        ch = _channel_problem(problem, c)
        expt = result.experiments[c]
        model = Matrix{Float64}(measure(expt))
        D = ch.data.image
        sig = ch.sigma
        chi2_bins, npix = _readable_chi2_bins(model, D, sig, ch.pixel_index)
        f["$prefix/lambda_nm"] = Vector{Float64}(lambda_from_si.(expt.lambda, :lambda_nm))
        f["$prefix/data"] = Matrix{Float64}(D)
        f["$prefix/sigma"] = Matrix{Float64}(sig)
        f["$prefix/model"] = model
        f["$prefix/mask"] = UInt8.(ch.mask)
        f["$prefix/chi2"] = chi2_bins
        f["$prefix/npixels"] = npix
        f["$prefix/chi2_total"] = sum(chi2_bins)
        f["$prefix/weight"] = Float64(summ.weight)
    end
    return nothing
end

"""
    _readable_chi2_bins(model, D, sig, pixel_index) -> (chi2, npixels)

The χ² and the fitted-pixel count of each time bin over the pixels `pixel_index` lists (Phase 4.M spec
D10): column `j` of a pixel `i` is `(i - 1) ÷ n_λ + 1`, and each pixel adds `((model - D)/σ)²` to its
column in `pixel_index` order. A function barrier: the loop is compiled for the arrays' own types, while
the callers above are compiled once for every problem and result type (`@nospecialize`).
"""
function _readable_chi2_bins(model::AbstractMatrix, D::AbstractMatrix, sig::AbstractMatrix,
    pixel_index::AbstractVector{<:Integer})
    nl, nt = size(D)
    chi2 = zeros(Float64, nt)
    npix = zeros(Int64, nt)
    for i in pixel_index
        j = (i - 1) ÷ nl + 1
        chi2[j] += abs2((model[i] - D[i]) / sig[i])
        npix[j] += 1
    end
    return chi2, npix
end

"""
    _readable_unit_keys(bindings::Vector{Binding}) -> Dict{String,String}

Each bound name's `unit_key`, from its first binding in `ParameterSpec.bindings` of either kind, a
plasma leaf or an effect parameter (Phase 4.M spec D11; every binding of one name shares one factor,
D9). A name bound to nothing is absent, and `_readable_unit` writes `"-"` for it.
"""
function _readable_unit_keys(bindings::Vector{Binding})
    units = Dict{String,String}()
    for b in bindings
        haskey(units, b.name) || (units[b.name] = b.unit_key)
    end
    return units
end

"""
    _readable_unit(units::AbstractDict, name::AbstractString) -> String

`units/<name>`'s text (Phase 4.M spec D11; the user's U4 of 2026-10-14): the name's `unit_key`, or
`"-"` when it has none (a dimensionless target, a key already in SI, or a name bound to nothing). Never
`""`: h5py cannot read an empty scalar string as JLD2 writes it.
"""
_readable_unit(units::AbstractDict, name::AbstractString) = (u = get(units, name, ""); isempty(u) ? "-" : u)

"""
    _readable_params!(f, result::FitResult) -> nothing

`params/<name>`, `units/<name>` and `flags/<name>` for every variable and then every dependent of
`result.pspec` (Phase 4.M spec D11, D12): the values `FitResult.values` holds, unconverted (each in
its own deck unit, Phase 3 spec D26), one `Float64` and one `String` for a scalar variable; the flag
`"F"` for a fitted variable, `"H"` for one not fitted (held or frozen), `"B"` at each entry
`_near_bound_entries(result)` lists, `"D"` for a dependent, and `""` wherever the value is `NaN`.
"""
function _readable_params!(f, @nospecialize(result::FitResult))
    pspec = result.pspec
    units = _readable_unit_keys(pspec.bindings)
    entries = _near_bound_entries(result)
    for v in pspec.variables
        vals = result.values[v.name]
        letter = v.vary ? "F" : "H"
        flags = String[isnan(x) ? "" : letter for x in vals]
        for e in entries
            e.variable == v.name && (flags[v.time_varying ? e.bin : 1] = "B")
        end
        f["params/$(v.name)"] = v.time_varying ? Vector{Float64}(vals) : Float64(vals[1])
        f["units/$(v.name)"] = _readable_unit(units, v.name)
        f["flags/$(v.name)"] = v.time_varying ? flags : flags[1]
    end
    for d in pspec.dependents
        vals = result.values[d.name]
        f["params/$(d.name)"] = Vector{Float64}(vals)
        f["units/$(d.name)"] = _readable_unit(units, d.name)
        f["flags/$(d.name)"] = String[isnan(x) ? "" : "D" for x in vals]
    end
    return nothing
end

"""
    _readable_fit!(f, result::FitResult, gn_decrease) -> nothing

The `fit/` summary (Phase 4.M spec D10): `chi2` (the weighted total for a joint fit, as `fit/chi2`
in the result file), `chi2_per_n` (`chi2 / npixels`), `R`, `objective`, `status`, `converged`
(`UInt8`), `termination`, `iterations`, `npixels`, `ntheta`, and `gn_decrease` when given.
"""
function _readable_fit!(f, @nospecialize(result::FitResult), gn_decrease)
    f["fit/chi2"] = Float64(result.chi2)
    f["fit/chi2_per_n"] = Float64(result.chi2) / result.npixels
    f["fit/R"] = Float64(result.R)
    f["fit/objective"] = Float64(result.objective)
    f["fit/status"] = string(result.status)
    f["fit/converged"] = UInt8(result.converged)
    f["fit/termination"] = string(result.termination)
    f["fit/iterations"] = Int64(result.iterations)
    f["fit/npixels"] = Int64(result.npixels)
    f["fit/ntheta"] = Int64(result.pspec.ntheta)
    gn_decrease === nothing || (f["fit/gn_decrease"] = Float64(gn_decrease))
    return nothing
end

"""
    _readable_laplace!(f, result::FitResult, laplace::LaplaceResult) -> nothing

`laplace/` (Phase 4.M spec D10): `decrease`, and per fitted variable `sigma/<name>` (the raw
`sigma_post`, in the variable's own unit, unscaled), `sigma_samp/<name>` and `near_bound/<name>`
(`UInt8`), each `(n_t)` for a time-varying variable and one value for a scalar one, as `params`.
"""
function _readable_laplace!(f, @nospecialize(result::FitResult), laplace::LaplaceResult)
    tv = Dict{String,Bool}(v.name => v.time_varying for v in result.pspec.variables)
    f["laplace/decrease"] = Float64(laplace.decrease)
    for (name, sp) in laplace.sigma_post
        ss = laplace.sigma_samp[name]
        nb = UInt8.(laplace.near_bound[name])
        f["laplace/sigma/$name"] = tv[name] ? Vector{Float64}(sp) : Float64(sp[1])
        f["laplace/sigma_samp/$name"] = tv[name] ? Vector{Float64}(ss) : Float64(ss[1])
        f["laplace/near_bound/$name"] = tv[name] ? nb : nb[1]
    end
    return nothing
end

"""
    _readable_scatter!(f, result::FitResult, scatter::ScatterResult) -> nothing

`scatter/` (Phase 4.M spec D10): per fine window `t_ns` (`t_fine` converted), `bin` (`coarse_bin`),
`status`, `decrease`, `chi2`, and per free variable `params/<name>` (the window fits' values),
`near_bound/<name>` (`UInt8`); per fitted bin `sigma_stat/<name>` (`s_stat`) and (Phase 4.N spec D7)
`p16/<name>`, `median/<name>`, `p84/<name>`; per held variable (Phase 4.N spec D7) `held/<name>`, per
fine window; and per channel, under `result`'s channel names in order, `channels/<name>/data`,
`sigma`, `mask` (`UInt8`) and `model`, `(n_λ, n_w)` each, the fine windows' own. `m`, `counts` and
`n_near_bound` stay in the result file.
"""
function _readable_scatter!(f, @nospecialize(result::FitResult), scatter::ScatterResult)
    f["scatter/t_ns"] = Vector{Float64}(t_from_si.(scatter.t_fine, :t_ns))
    f["scatter/bin"] = Int64.(scatter.coarse_bin)
    f["scatter/status"] = String[string(st) for st in scatter.status]
    f["scatter/decrease"] = Vector{Float64}(scatter.decrease)
    f["scatter/chi2"] = Vector{Float64}(scatter.chi2)
    for (name, v) in scatter.values
        f["scatter/params/$name"] = Vector{Float64}(v)
        f["scatter/near_bound/$name"] = UInt8.(scatter.near_bound[name])
        f["scatter/sigma_stat/$name"] = Vector{Float64}(scatter.s_stat[name])
        f["scatter/p16/$name"] = Vector{Float64}(scatter.p16[name])
        f["scatter/median/$name"] = Vector{Float64}(scatter.median[name])
        f["scatter/p84/$name"] = Vector{Float64}(scatter.p84[name])
    end
    for (name, v) in scatter.held
        f["scatter/held/$name"] = Vector{Float64}(v)
    end
    for (c, summ) in enumerate(result.channels)
        prefix = "scatter/channels/$(summ.name)"
        f["$prefix/data"] = Matrix{Float64}(scatter.data[c])
        f["$prefix/sigma"] = Matrix{Float64}(scatter.sigma[c])
        f["$prefix/mask"] = UInt8.(scatter.mask[c])
        f["$prefix/model"] = Matrix{Float64}(scatter.model[c])
    end
    return nothing
end
