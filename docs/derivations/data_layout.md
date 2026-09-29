# The JLD2 file layouts

> **Front matter (Step 2.4, 2026-09-11).** Windows 11, Julia 1.12.7; JLD2 0.6.6. Written
> alongside `src/data/streak.jl` and `src/data/io.jl`.

This document is the normative description of the two JLD2 layouts `write_streak`/`read_streak`
and `write_trace`/`read_trace` implement ([OQ-15](../../dev/open_questions.md#oq-15); spec D19,
D20). JLD2 is used only as a container for **plain numeric arrays and strings** — no Julia type
information — so that the subset an HDF5 reader such as h5py understands is exactly what is
written. There is no Python on this machine, so h5py-readability is a property of the layout,
checked by IO3 (every value in the file `isa Union{AbstractArray{<:Real}, AbstractString,
Real}`), not by an actual h5py read.

## 1. The streak layout, `otis-streak-1`

| Dataset | Type | Shape | Required |
|---|---|---|---|
| `layout_version` | `String` | scalar, `"otis-streak-1"` | yes |
| `lambda_nm` | `Float64` | `(N_λ)` | yes |
| `t_ns` | `Float64` | `(N_t)` | yes |
| `image` | `Float64` | `(N_λ, N_t)` | yes |
| `sigma` | `Float64` | `(N_λ, N_t)` | no |
| `mask` | `UInt8` | `(N_λ, N_t)`, `1` = use the pixel | no |
| `meta/source` | `String` | free text | yes |
| `meta/created_unix` | `Float64` | seconds since the Unix epoch | yes |
| `meta/otis_version` | `String` | `string(pkgversion(OTIS))` | yes |
| `meta/lambda_probe_nm` | `Float64` | | when a probe is given |
| `meta/theta_deg` | `Float64` | | when a collection is given |
| `meta/polarization_deg` | `Float64` | | when a probe is given |
| `meta/k_model` | `String` | `"exact"` or `"approximate"` | when a collection is given |
| `meta/regions_source_t` | `Float64` | `(N_t,source)` | when `average_regions` produced this streak |
| `meta/regions_t_shift` | `Float64` | scalar, s | when `average_regions` produced this streak |
| `meta/regions_min_usable_fraction` | `Float64` | scalar | when `average_regions` produced this streak |
| `meta/regions_starts` | `Float64` | `(N_regions,)`, s | when `average_regions` produced this streak |
| `meta/regions_ends` | `Float64` | `(N_regions,)`, s | when `average_regions` produced this streak |
| `meta/regions_nbins` | `Int` | `(N_regions,)` | when `average_regions` produced this streak |
| `meta/regions_intervals` | `String` | `(N_regions,)`, `""` for none | when `average_regions` produced this streak |
| `meta/regions_window_counts` | `Int` | `(N_t,)`, `\|W_k\|` | when `average_regions` produced this streak |
| `meta/regions_usable_counts` | `Int` | `(N_λ, N_t)`, `\|U_k,l\|` | when `average_regions` produced this streak |
| `meta/<user key>` | as given | | every entry of `sd.meta` |

**The nine `regions_*` keys (Step 4.A.2, 2026-09-23; spec D16, amended).** Written by
`average_regions` (`src/data/average.jl`) when it produces a streak by averaging a source streak
onto a user-given time axis ([OQ-34](../../dev/open_questions.md#oq-34)): `regions_source_t` is
the source axis in s, **before** the channel's fixed timing shift; `regions_t_shift` is that
shift, Δt, in s; the rest describe the regions and the per-window pixel counts the averaging used.
**These add no dataset the table above did not already have a place for and bump no layout
version**: all nine are `Real`s, `String`s or arrays of those, exactly the type `_check_meta_value`
already accepts and `write_streak`'s `meta/$k` loop already writes without a type map. **The
prefix is `regions_`, not `regions/`**, and this is load-bearing, not stylistic: `/` is a JLD2
group separator, and a meta key containing one produces a nested group `read_streak` cannot read
back as a scalar or an array (`StreakData`'s constructor refuses such a key outright, D26). A
source streak whose own `meta` already carries a key beginning `regions_` is a loud error at
`average_regions`, since these nine describe an operation that must not be silently reapplied.

**Orientation, in bold because it is the trap.** Arrays are written in Julia's **column-major
order with wavelength first** — `image[l, j]` is wavelength `l`, time `j`. An h5py reader,
which is row-major, sees the **transpose**, `(N_t, N_λ)`. `write_streak` and `read_streak` both
carry this statement in their docstrings; IO7 is the test that would catch a transposed write
(it uses a 64×16, not a square, streak precisely so a transpose cannot pass by accident).

**The mask convention.** `mask[l, j] == true` (stored as `UInt8` `1`) means the pixel **is
used**. This is the opposite of the more common "mask out bad pixels" convention, so it is
stated here, in `StreakData`'s own docstring, and IO4 checks it directly (`1` where the source
`Bool` mask is `true`). A Julia `Bool` array is not portable across languages the way `UInt8`
is, which is why the file stores the latter.

**`created_unix`, not a human-readable date.** [OQ-1](../../dev/open_questions.md#oq-1)'s Phase
2 dependency list admits `LinearAlgebra` and `Random` only; `Dates` is not on it, so the
timestamp is a Unix float rather than an ISO-8601 string. **[propagate]** — Level 2 has proposed
OQ-1's Phase 3 list gain `Dates` and the layout gain `meta/created` alongside
`meta/created_unix` then; recorded in `dev/phase2_overall_summary.md`, not repeated as new here.

**The probe/collection meta keys are optional and given as a pair**, `write_streak(path, sd;
source, probe, collection)`, reflecting the Step 2.3a split of the geometry type (spec D40); D19
was written against the single `ScatteringGeometry` this replaced, and the four meta keys keep
their original names (`lambda_probe_nm`, `theta_deg`, `polarization_deg`, `k_model`) split
across the two structs exactly as the structs themselves are split.

**`lambda`/`t` round-trip to full double precision, not always bitwise.** Converting SI to nm/ns
on write and back on read is one division and one multiplication by the same `Float64` constant
(`1e-9`); this is not always an exact inverse at the last bit for an arbitrary SI value (measured:
1 mismatch in a random 64-point wavelength grid, off by a single ULP). `image`, `sigma` and
`mask` carry no unit conversion and round-trip exactly; IO1 compares `lambda`/`t` at `rtol =
1e-15` and the arrays bitwise.

**A dual streak cannot be written.** `write_streak` requires `eltype(sd.lambda) <: AbstractFloat`
(any plain float — IO10 tests `Float32`) and throws naming the actual element type otherwise — a
dual streak is a computation, not data.

### An untested h5py snippet

No Python is available on this machine to run this; it documents the intended read, marked
untested.

```python
import h5py
with h5py.File("streak.jld2", "r") as f:
    image = f["image"][:]          # shape (N_t, N_λ): h5py sees the transpose
    lambda_nm = f["lambda_nm"][:]  # shape (N_λ,)
    t_ns = f["t_ns"][:]            # shape (N_t,)
    mask = f["mask"][:] if "mask" in f else None   # 1 = use the pixel
    source = f["meta/source"][()].decode()
```

## 2. The trace layout, `otis-trace-1`

| Dataset | Type | Shape | Required |
|---|---|---|---|
| `layout_version` | `String` | scalar, `"otis-trace-1"` | yes |
| `axis_name` | `String` | scalar, e.g. `"t"` | yes |
| `axis` | `Float64` | `(N_t)`, **SI** | yes |
| `axis_unit` | `String` | scalar, `"s"` for `:t`, else `"SI"` | yes |
| `paths` | `String` | `(N_leaf)`, the SI leaf paths, row order | yes |
| `values` | `Float64` | `(N_leaf, N_t)`, SI | yes |
| `template/n_e_m3` | `Float64` | | yes |
| `template/electrons/names` | `String` | `(N_e)` | yes |
| `template/electrons/<name>/mass_kg` | `Float64` | | yes |
| `template/electrons/<name>/charge` | `Int64` | | yes |
| `template/electrons/<name>/u_ms` | `Float64` | | yes |
| `template/electrons/<name>/T_J` | `Float64` | | yes |
| `template/electrons/<name>/fraction` | `Float64` | | yes |
| `template/electrons/<name>/shape` | `String` | e.g. `"SuperGaussian"` | yes |
| `template/electrons/<name>/shape_p/<p>` | `Float64` | one per `parameters(f)` entry | when non-empty |
| `template/ions/…` | | the same, for ion species | yes |
| `meta/source`, `meta/created_unix`, `meta/otis_version` | | as the streak layout | yes |

**The axis is SI here, not `t_ns` as the streak layout uses.** A `PlasmaTrace`'s axis need not
be time ([OQ-26](../../dev/open_questions.md#oq-26)) — `axis_name` need not be `:t` — so there
is no universal boundary unit to convert to, and `axis_unit` names whatever unit `axis` is
actually in (`"s"` when `axis_name === :t`, else `"SI"`, the generic statement that no
conversion was applied). This asymmetry against the streak layout's `t_ns` is deliberate.

**What `read_trace` does not reconstruct.** The file has everything needed to rebuild a
`PlasmaTrace` **except** the machinery that turns `template/electrons/<name>/shape = "SuperGaussian"`
and its `shape_p` back into an actual `AbstractVelocityDistribution` object — that lookup is the
component registry of `conventions.md` 7.2, built at Step 3.6. `read_trace` therefore returns
plain `Dict`/array data (spec D20's `NamedTuple`), and IO6's round-trip test compares `axis`,
`values` and `paths` bit for bit against what was written, not against a reconstructed trace.

**Only the ground-truth trace is written in Phase 2** (`examples/`, Step 2.5); Step 3.6 built the
result writer below (roadmap Step 2.4, amended; `dev/phase2_overall_summary.md` A7).

## 3. The result layout, `otis-result-3`

**Bumped from `otis-result-1` at Step 3.C.2 (D35)**, one bump carrying `fit/termination`,
`fit/near_bound_fraction` and (with a scan) `scan/status` together, and joined within the same
version by `scan/termination` at Step 3.C.3 (carried item 2) -- adding a dataset inside a version
introduced by this same phase, before the phase closes, is not the second bump D35 forbade.

**Bumped from `otis-result-2` to `otis-result-3` at Step 4.A.3b (D40, chunk 7)**, one bump adding
a `channels/` group: `channels/names`, a `Vector{String}` in channel order, and, per channel,
`channels/<name>/weight`, `chi2` (unweighted), `npixels`, `nparams`, `reduced_chi2`,
`reduced_chi2_simple`, `lambda`, `model` (`measure` of the fitted `experiments[c]`),
`effects/<k>/type` and `effects/<k>/<field>` (`parameters(effect)` of the fitted experiment's
`k`-th effect, in SI), and, only when `write_result` is given the problem, `residual`
(`(M_c - D_c)/sigma_c`, `NaN` where the mask is false) and `mask` (`UInt8`). `fit/chi2` and
`fit/npixels` at the top level are now the **weighted total** and the **sum over channels**
(`Sigma_c w_c chi2_c` and `Sigma_c n_c`) for a joint fit -- this is what `otis-result-2`'s own
comment below anticipated ("a later joint EPW/IAW fit ... turns this into a version bump"), with
the group named `channels/`, not `datasets/`. **Every `otis-result-3` file has `channels/`**,
since every `FitResult` carries at least one `ChannelSummary` (D36): a single-channel fit (the
shorthand deck, or a one-channel channel-form deck) writes `channels/main`.

| Dataset | Type | Shape | Required |
|---|---|---|---|
| `layout_version` | `String` | scalar, `"otis-result-3"` | yes |
| `meta/source` | `String` | free text | yes |
| `meta/created` | `String` | ISO-8601 UTC, e.g. `"2026-09-15T12:00:00.000Z"` | yes |
| `meta/created_unix` | `Float64` | seconds since the Unix epoch | yes |
| `meta/otis_version` | `String` | `string(pkgversion(OTIS))` | yes |
| `deck` | `String` | the effective TOML deck (spec D33) | yes |
| `axis_name` | `String` | scalar, e.g. `"t"` | yes |
| `axis` | `Float64` | `(N_t)`, SI | yes |
| `axis_unit` | `String` | scalar, `"s"` for `:t`, else `"SI"` | yes |
| `axis_boundary_unit` | `String` | `"ns"` or `"s"`: the unit `t` and the regularizer's operators used | yes |
| `variables/<name>/values` | `Float64` | `(N_t)` or `(1)` for a scalar, in the variable's own unit | yes, one group per fitted variable |
| `variables/<name>/unit` | `String` | the boundary keyword it was bound through, `""` if none | yes |
| `variables/<name>/bounds` | `Float64` | `(2)` | yes |
| `variables/<name>/bins` | `Int64` | `(2)`: first and last bin of its interval | yes |
| `variables/<name>/time_varying` | `UInt8` | | yes |
| `variables/<name>/vary` | `UInt8` | | yes |
| `variables/<name>/near_bound` | `UInt8` | `(N_t)` or `(1)` | yes |
| `dependents/<name>/values` | `Float64` | `(N_t)` | yes, one group per dependent |
| `dependents/<name>/expression` | `String` | the deck text, verbatim | yes |
| `leaves/<si_path>/values` | `Float64` | `(N_t)`, SI | yes, one group per row of the fitted trace |
| `fit/theta` | `Float64` | `(ntheta)`, the minimizer in `y` | yes |
| `fit/chi2`, `fit/R`, `fit/objective`, `fit/weight` | `Float64` | scalar | yes |
| `fit/term_values` | `Float64` | `(n_terms)` | yes |
| `fit/npixels`, `fit/iterations`, `fit/f_calls`, `fit/g_calls` | `Int64` | scalar | yes |
| `fit/converged` | `UInt8` | scalar | yes |
| `fit/status`, `fit/optimizer` | `String` | scalar | yes |
| `fit/termination` | `String` | scalar, Optim's finer code (`:none` for the non-Optim optimizers) | yes |
| `fit/near_bound_fraction` | `Float64` | scalar, `_near_bound_fraction(result)` | yes |
| `fit/gn_decrease` | `Float64` | scalar, the Gauss–Newton predicted decrease at `fit/theta` over the free entries; `NaN` when `JᵀJ` there is not positive definite (Phase 4.B spec D5) | only when `write_result` is given it, which `fit(deck)` always does |
| `fit/near_bound_entries/variable` | `String` | `(N_nb)`: the variable of each fitted entry within `BOUND_PROXIMITY_TOL` of a bound, in `pspec.variables` order, then ascending bin | only when there is at least one such entry (Phase 4.D spec D7) |
| `fit/near_bound_entries/bin` | `Int64` | `(N_nb)`: the entry's bin on the fitted axis; `0` for a scalar variable | with `variable` |
| `fit/near_bound_entries/bound` | `String` | `(N_nb)`: `"lower"` or `"upper"`, the bound the value is nearer | with `variable` |
| `fit/elapsed` | `Float64` | scalar, seconds | yes |
| `trajectory/iteration` | `Int64` | `(n_iter)` | yes |
| `trajectory/objective`, `trajectory/chi2`, `trajectory/R` | `Float64` | `(n_iter)` | yes |
| `channels/names` | `String` | `(N_channels)`, channel order | yes |
| `channels/<name>/weight`, `chi2`, `reduced_chi2`, `reduced_chi2_simple` | `Float64` | scalar | yes, one group per channel |
| `channels/<name>/npixels`, `nparams` | `Int64` | scalar | yes |
| `channels/<name>/lambda` | `Float64` | `(N_lambda,c)`, m | yes |
| `channels/<name>/model` | `Float64` | `(N_lambda,c, N_t)`, `measure(experiments[c])` | yes |
| `channels/<name>/effects/<k>/type` | `String` | scalar, the effect's type name | yes, one group per effect of that channel |
| `channels/<name>/effects/<k>/<field>` | `Float64` | scalar, or `(n)` for a vector field | one per `parameters(effect)` entry |
| `channels/<name>/residual` | `Float64` | `(N_lambda,c, N_t)`, `(M-D)/sigma`, `NaN` where masked | only when `write_result` is given the problem |
| `channels/<name>/mask` | `UInt8` | `(N_lambda,c, N_t)` | only when `write_result` is given the problem |
| `scan/status`, `scan/termination` | `String` | `(K)`, one entry per requested weight | with a scan |
| `scan/weights` | `Float64` | `(K)` | only when a scan was written |
| `scan/chi2`, `scan/R`, `scan/chi2_normalized`, `scan/R_normalized` | `Float64` | `(K)` | with a scan |
| `scan/chi2_unreg`, `scan/R_unreg` | `Float64` | scalar | with a scan |
| `scan/rss/<variable>`, `scan/rss_normalized/<variable>` | `Float64` | `(K)` | only with a known ground truth |
| `scan/rss_unreg/<variable>` | `Float64` | scalar | only with a known ground truth |

**The L-curve's corner is not stored** (Phase 4.C spec D7): it is a report recomputed from the
file's own `scan/weights`, `scan/chi2` and `scan/R` by `lcurve_corner(r.scan.weights, r.scan.chi2,
r.scan.R)` (`r = read_result(path)`), bitwise the estimate the script that wrote the file saw, the
member fits' convergence flags apart (the file carries each member's `scan/status`, not its
`converged`).

**`fit/near_bound_entries/*` is optional inside `otis-result-3`** (Phase 4.D spec D7): written, by
every caller of `write_result`, only when some fitted (`vary`) variable has an entry within
`BOUND_PROXIMITY_TOL` of a bound, one element per entry, the same entries `fit(deck)`'s saturation
warning names; a file without it reads back `fit.near_bound_entries === nothing`. It is the list
form of the `variables/<name>/near_bound` flags over the fitted variables (a frozen variable's
flags are written there and never listed here), so the two always agree (NB4).

**`fit/chi2` and `fit/npixels` are the weighted total and the channel sum** (Step 4.A.3b, D40):
`fit/chi2 = Sigma_c w_c chi2_c` (what the optimizer minimized) and `fit/npixels = Sigma_c n_c`,
`1` and `1` for a single-channel fit. This is the version bump the `otis-result-2` text above
anticipated ("a later joint EPW/IAW fit ... turns this into a version bump"); the group is named
`channels/`, not `datasets/`, and carries the per-channel `chi2`/`npixels` that text foresaw,
alongside the fitted model, the effect parameters and, when the problem is given, the residual and
mask.

**`laplace/` and `calibration_scan/` groups, inside the same `otis-result-3` (Step 4.A.4a, D47).**
Neither is a version bump: D17 forbids a *second* bump within the phase, not a dataset added
inside the one bump already made, and both groups are written only when `write_result` is given
the matching keyword, so an old `otis-result-3` file (chunk 7's, with neither group) still reads
back exactly as before -- `read_result`'s `laplace`/`calibration_scan` fields are `nothing` when
the file carries no such group. **`laplace` must satisfy `laplace.theta == result.theta` exactly**
(a loud error otherwise, naming the mismatch): `write_result` has no other way to know the
`LaplaceResult` it is given was computed at the θ `result` itself reports. The `p x p` covariances
are never written (13 MB at `p = 1280`); only the per-variable σ and the near-bound flag are.

| Dataset | Type | Shape | Required |
|---|---|---|---|
| `laplace/decrease` | `Float64` | scalar | only with `laplace` |
| `laplace/variables/<name>/sigma_post`, `sigma_samp` | `Float64` | `(N_t)` or `(1)`, `NaN` kept | yes, one group per fitted variable |
| `laplace/variables/<name>/near_bound` | `UInt8` | `(N_t)` or `(1)` | yes |
| `calibration_scan/shifts_nm` | `Float64` | `(N_shifts)` | only with `calibration_scan` |
| `calibration_scan/channel` | `Int64` | scalar | yes |
| `calibration_scan/base_offset_nm` | `Float64` | scalar | yes |
| `calibration_scan/chi2`, `objective`, `decrease` | `Float64` | `(N_shifts)` | yes |
| `calibration_scan/status` | `String` | `(N_shifts)` | yes |
| `calibration_scan/variables/<name>/delta` | `Float64` | `(N_t or 1, N_shifts)` | yes, one group per variable |
| `calibration_scan/variables/<name>/s_sys` | `Float64` | `(N_t)` or `(1)` | yes |

**`scatter/` group, inside the same `otis-result-3` (Step 4.A.4b, D52).** Not a version bump, the
same style as `laplace/` and `calibration_scan/` above: written only when `write_result` is given
the `scatter` keyword, and `scatter.theta == result.theta` is required exactly (a loud error
otherwise, `"scatter was computed at a different θ than result.theta"`) -- `write_result` has no
other way to know the `ScatterResult` it is given was computed at the θ `result` itself reports.
One entry per **free** (`vary && time_varying`) fitted variable only; a scalar or a frozen variable
has no `scatter/variables/<name>` group at all.

| Dataset | Type | Shape | Required |
|---|---|---|---|
| `scatter/m` | `Int64` | `(N_regions)` | only with `scatter` |
| `scatter/t_fine` | `Float64` | `(N_fine)`, s | yes |
| `scatter/coarse_bin` | `Int64` | `(N_fine)` | yes |
| `scatter/counts` | `Int64` | `(N_fine, N_channels)` | yes |
| `scatter/status` | `String` | `(N_fine)` | yes |
| `scatter/decrease` | `Float64` | `(N_fine)` | yes |
| `scatter/variables/<name>/values` | `Float64` | `(N_fine)`, the variable's own unit | yes, one group per free variable |
| `scatter/variables/<name>/near_bound` | `UInt8` | `(N_fine)` | yes |
| `scatter/variables/<name>/s_stat` | `Float64` | `(N_coarse)`, **never divided by `sqrt(m)`** | yes |
| `scatter/variables/<name>/n_near_bound` | `Int64` | `(N_coarse)` | yes |

**`read_trace` gains an `as` keyword** (`:arrays`, unchanged and default, or `:trace`, spec D35),
once the component registry (`conventions.md` 7.2) exists to look shape names up in; the trace
layout itself, `TRACE_LAYOUT_VERSION` and `read_streak`/`write_streak`/`write_trace` are
untouched.

## 4. The deck's channel form, and the pooled axis (Step 4.A.3b, D40)

**`otis-deck-2`, no further bump.** `read_deck` accepts two mutually exclusive forms of a deck.
**The shorthand** has a bare `[data]`, `[geometry]` and optional `[instrument]`, exactly as every
deck before chunk 7 -- the single-channel path is bitwise unchanged. **The channel form** has a
root-level `channels = ["epw", "iaw"]` array (the channel order) and a `[channel.<name>]` table
per name, each holding an optional `weight` (default `1.0`) and its own `data`, `geometry` and
`instrument`. Mixing the two forms is a loud error, and so is a top-level key `read_deck` does not
know (D40 item 1, Pitfall 9) -- a `[channel.epw]` table in a deck `read_deck` did not check for
would otherwise be silently ignored and the fit would run single-channel.

**The regions move to one shared `[[fit.regions]]`.** Two cameras average onto one time axis
([OQ-34](../../dev/open_questions.md#oq-34), D39), so the averaging windows are declared once,
under `[fit]`, rather than once per channel; a `regions` key left inside a `[data]` or
`[channel.<name>.data]` table is refused, naming its new home. Per-camera keys (`file`,
`t_shift_ns`/`t_shift_s`, `min_usable_fraction`) stay in each channel's own `data` table.

**The pooled axis.** With one channel the fitted axis is that channel's own averaged `t̄`
(`docs/derivations/` -- the averaging rule itself is unchanged, D14), bitwise identical to a deck
without a channel form. With two or more channels, each output bin's time is the **pooled mean**
of every channel's window mean, weighted by that channel's own window pixel count:

```
t̄_k = ( Σ_c |W_c,k| · t̄_c,k ) / ( Σ_c |W_c,k| )
```

with `|W_c,k|` read from each channel's own `regions_window_counts` meta key (Section 1 above).
Every channel's averaged streak is then rebuilt on this one pooled axis before the `Experiment`s
and the shared `ParameterSpec` are built, so that `FitProblem`'s `rtol = 1e-12` axis check holds
for every channel against the one shared trace. The cost, accepted at the design stage: each
channel's data are labelled with a time a few picoseconds from their own mean, inside a window
usually a hundred picoseconds or more wide.

**The result file's `channels/<name>` groups (Section 3 above) are keyed by this same channel
order and these same names** -- the shorthand's one channel is always named `"main"`; a
one-channel channel-form deck can name it anything the deck's `channels` list says.

`write_result`/`read_result` (`src/data/result.jl`) are this layout's readers and writers, on the
same footing as `write_streak`/`read_streak` and `write_trace`/`read_trace` above.
