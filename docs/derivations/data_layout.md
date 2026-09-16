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
| `meta/<user key>` | as given | | every entry of `sd.meta` |

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

## 3. The result layout, `otis-result-1`

| Dataset | Type | Shape | Required |
|---|---|---|---|
| `layout_version` | `String` | scalar, `"otis-result-1"` | yes |
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
| `fit/elapsed` | `Float64` | scalar, seconds | yes |
| `trajectory/iteration` | `Int64` | `(n_iter)` | yes |
| `trajectory/objective`, `trajectory/chi2`, `trajectory/R` | `Float64` | `(n_iter)` | yes |
| `scan/weights` | `Float64` | `(K)` | only when a scan was written |
| `scan/chi2`, `scan/R`, `scan/chi2_normalized`, `scan/R_normalized` | `Float64` | `(K)` | with a scan |
| `scan/chi2_unreg`, `scan/R_unreg` | `Float64` | scalar | with a scan |
| `scan/rss/<variable>`, `scan/rss_normalized/<variable>` | `Float64` | `(K)` | only with a known ground truth |
| `scan/rss_unreg/<variable>` | `Float64` | scalar | only with a known ground truth |

`fit/chi2`, `fit/npixels` and `meta/source` stay scalars: Phase 3 fits one dataset at a time, and
a later joint EPW/IAW fit that needs per-dataset values turns this into a version bump
(`layout_version` grows a `datasets/<name>/` group carrying its own `chi2`/`npixels`), not a
silent break — the version string is what makes that possible, and it is why it is written even
though nothing in Phase 3 reads it back for dispatch.

**`read_trace` gains an `as` keyword** (`:arrays`, unchanged and default, or `:trace`, spec D35),
once the component registry (`conventions.md` 7.2) exists to look shape names up in; the trace
layout itself, `TRACE_LAYOUT_VERSION` and `read_streak`/`write_streak`/`write_trace` are
untouched.

`write_result`/`read_result` (`src/data/result.jl`) are this layout's readers and writers, on the
same footing as `write_streak`/`read_streak` and `write_trace`/`read_trace` above.
