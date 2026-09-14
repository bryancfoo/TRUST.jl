# The regularizer

> **Front matter (Step 3.3, 2026-09-13; normalization-after-operator revision, chunk 7
> remediation, 2026-09-13).** Windows 11, Julia 1.12.7; ForwardDiff 1.4.5. Written alongside
> `src/inversion/regularizer.jl`.

This document derives `src/inversion/regularizer.jl`: the three operators as divided differences
on a nonuniform axis, why `D2` is exact for a quadratic on any grid, why `R` is a plain sum over
rows and what that buys under rebinning, the four normalization modes and the anchor-bin rule that
applies them **after** the operator, the residual-row form and its closed-form Jacobian (including
`relative`'s quotient-rule term), and the interval-edge rule. Spec D22-D25,
[OQ-19](../../dev/open_questions.md#oq-19).

## 1. The operators, as divided differences on the axis

Let `u_j = x_j - ref_j` be one selected name's **reference-subtracted** trace at bin `j` (`x_j` in
the name's own unit, `ref_j` the term's reference trace, zero by default; Section 3 below covers
what happens to `u_j` after the operator), and `t_j` the axis in the axis keyword's own unit
(`pspec.axis_boundary`, nanoseconds for a trace built with `t_ns`) — **never the bin index**. Over
the bins `bins` of the term's own window (the whole axis when `interval == 0`), with `j` ranging
over `first(bins)+1 .. last(bins)-1` for `D2`, `first(bins) .. last(bins)-1` for `D1`, and
`first(bins) .. last(bins)` for `I`:

```
    D2:  row_j = 2 * ( (u_{j+1} - u_j)/(t_{j+1} - t_j) - (u_j - u_{j-1})/(t_j - t_{j-1}) )
                   / (t_{j+1} - t_{j-1})

    D1:  row_j = (u_{j+1} - u_j) / (t_{j+1} - t_j)

    I:   row_j = u_j
```

`D1` is the ordinary divided first difference: the slope of the secant line through `(t_j,
u_j)` and `(t_{j+1}, u_{j+1})`. `D2` is the divided second difference: it approximates `u''` by
differencing two neighbouring divided first differences and dividing by the *span* they cover
(`t_{j+1} - t_{j-1}`), which is what makes it exact on a nonuniform grid (below) rather than
merely consistent as the grid refines.

**Why time, not the bin index.** A `D2` written against the *index* difference (`u_{j+1} - 2u_j +
u_{j-1}`, with no division by any `Δt`) is a different quantity at every bin whenever the grid is
nonuniform, and it changes under rebinning even when the underlying trace does not: halving the
bin count would roughly quadruple an index-differenced curvature's typical size for no physical
reason, so `Λ` would have to be retuned every time a streak was read in at a different resolution.
Differentiating in `t` (the same axis `t` means inside a dependent expression, D8) is what keeps
`Λ` meaning the same thing under rebinning — RG1 asserts this on `test_regularizer.jl`'s
deliberately nonuniform ten-bin axis, `t_ns = [0.0, 0.31, 0.7, 1.05, 1.5, 1.9, 2.4, 2.75, 3.2,
3.6]`, precisely because a bin-index `D2` and a time `D2` disagree there and only one of them is
correct.

### `D2` is exact for a quadratic on any grid

Let `u(t) = a t^2 + b t + c`. Then

```
    (u_{j+1} - u_j)/(t_{j+1} - t_j) = a(t_{j+1} + t_j) + b
    (u_j - u_{j-1})/(t_j - t_{j-1}) = a(t_j + t_{j-1}) + b
```

(the standard identity for a quadratic's divided difference: `(u(p) - u(q))/(p - q) = a(p+q) + b`
for any two points `p ≠ q`, since `u(p)-u(q) = a(p^2-q^2)+b(p-q) = (p-q)(a(p+q)+b)`). Their
difference is `a(t_{j+1} - t_{j-1})`, and dividing by `(t_{j+1} - t_{j-1})` a second time and
doubling gives exactly `2a = u''(t)` — **for every triple of grid points, uniform or not**, with
no truncation term at all (a cubic or higher order would leave one, `O(Δt^2 u''''/12)` on a
near-uniform grid, RG15's discretization-error argument). This is not an approximation that
happens to be good; the algebra above is exact, which is why RG1 asserts `1e-12` relative rather
than a discretization-scaled tolerance.

## 2. `R` is a plain sum over rows, not a `Δt`-weighted integral

```
    R = sum over terms of  w_term * ||op(u - ref) / N||^2
```

(`N` the per-row anchor-bin divisor Section 3 derives) with no `Δt` weighting of any row. Consequently, rebinning a smooth trace by a factor `r` (so the
bin count and hence the row count both scale by `~1/r`) divides `R` by about `r`, the *same*
scaling `chi2`'s own pixel count follows under rebinning — which is what keeps `Λ` comparable
across binnings: `𝒪 = chi2 + Λ R` has both of its terms shrink together, so a `Λ` tuned at one
resolution stays roughly the right order of magnitude at another. A `Δt`-weighted integral would
instead keep `R` roughly *constant* under rebinning while `chi2` kept shrinking, decoupling the
two and forcing `Λ` to be retuned per resolution — the opposite of the property wanted here. A
weighted variant is left for Phase 4 if a use case wants it (`R` per unit time rather than per
row).

RG15 pins the *consequence* of this design rather than its definition: `D2`'s own truncation
error is `O(Δt^2 u''''/12)` (Section 1), so halving the bin count of a fixed, smooth trace
roughly quadruples each row's own squared truncation error while roughly halving the row count —
`R` per row is expected to be *stable* under rebinning to within that discretization error, not
exactly invariant, which is why RG15's tolerance (`15%`) is deliberately loose.

## 3. Row normalization: `N_r`, the anchor bin, and the reference

The operator (Section 1) reads the reference-subtracted trace `u_j = x_j - ref_j` directly — `x_j`
in the selected name's own unit (the unit its bounds are given in, for a fitted variable; whatever
unit the dependent's expression naturally produces, for a dependent), `ref_j` the term's reference
trace, zero ("off") by default. **Normalization is applied to the row the operator produces, not
to the trace the operator reads** (D24/D25, revised 2026-09-13; "The withdrawn design" below
covers the earlier convention and why it changed):

```
    g_r = (op(x - ref))_r          Section 1's operator output, row r, unnormalized
    rho_r = g_r / N_r              the term's r-th raw row
```

`N_r` is a single number per row, read off the trace at that row's own **anchor bin** `a(r)`:

| operator | row `r` spans local bins | anchor `a(r)` |
|---|---|---|
| `D2` | `r, r+1, r+2` | `r + 1`, the centred stencil's middle |
| `D1` | `r, r+1` | `r`, the forward difference's own base point |
| `I` | `r` | `r` |

`D1` anchors on its base point rather than some interpolated point between `r` and `r+1`: the row
is already one-sided (there is no symmetric point to prefer), and anchoring on the base point lets
`D1` and `I` share one rule (`a(r) = r`) while `D2`'s only symmetric choice is its own centre.
`src/inversion/regularizer.jl`'s `_operator_anchor` is this table; `_operator_coeffs` (Section 4)
calls it for its own `k`, so the coefficient table and this one read the same bin by construction —
one indexing convention, not two independently-written rules that happen to agree.

`N_r` itself is:

| mode | `N_r` |
|---|---|
| `absolute` | `s` |
| `initial` | `x⁰_{a(r)}` |
| `relative` | `x_{a(r)}` |
| `unit` | `1` |

`s` is the term's `scale`: the value given explicitly, or, for a fitted variable, the default
`b - a` (the variable's own bound width) — a dependent has no bound width and so has no default, a
loud error at construction (RG8). `x⁰_j` is the *starting* trace: the value the selected name
takes at every fitted variable's own `init` (built by evaluating the same per-bin value table
`unflatten` computes, but at `flatten(pspec)`'s start vector rather than the current θ, so a fitted
variable's `x⁰_j` is simply its own `init`, constant over every bin, and a dependent's is its
expression evaluated on that same constant background — RG22 exercises a dependent whose `x⁰_j` is
**not** constant, since its expression uses `t`).

`s` and `x⁰_{a(r)}` do not depend on `x` at any bin, so `∂N_r/∂x_j = 0` identically for `absolute`,
`initial` and `unit`. `relative`'s `N_r = x_{a(r)}` is the one exception: it is the trace's own
current value, so it depends on theta through bin `a(r)` — and *only* through that bin:
`∂N_r/∂x_j = delta_{j,a(r)}`. This single fact is what Section 4 turns into `relative`'s
quotient-rule term.

### The withdrawn design, and why it changed

**Until 2026-09-13 this file specified the opposite order: normalize the trace, then apply the
operator** (`u_j = N(x_j) = (x_j - ref_j)/N_j`, `row_r = (op(u))_r`, dividing *before* the
difference). Under that design `relative`'s default (`ref_j = 0`) made `N(x_j) = x_j/x_j = 1`
identically for every bin, so a `relative` `D1`/`D2` term with no explicit reference contributed
**exactly zero** — the mode was inert at its own default. Chunk 7 built exactly that (D24 as it
then read) and reported the degeneracy as a finding rather than "fixing" it
(`dev/reports/phase3_report.md`, 2026-09-13 entry).

The user's decision, the same day: **divide after the operator, not before, and use the same rule
for `initial` too**, so all four modes share the one expression `rho_r = g_r / N_r` above. This
also brings `relative` in line with foo2026 Table I, which normalizes the *second derivative* by
the local value ("if 'Relative' is listed each d²u/dt² is normalized by the local value of u")
rather than the trace itself — the substitution this file previously recorded here (using
`initial` where foo2026 says "Relative", [propagate] item A6) is withdrawn, along with the deck
comments that announced it (D38, chunk 12's own file; no deck existed yet when this remediation
was made, so there was nothing to edit there).

## 4. The residual-row form and its closed-form Jacobian

Every term's rows are stacked as extra residual rows under the `npixels` data rows (D25), weighted
by both the term's own `w_term` and the global `Λ`:

```
    rho_row = sqrt(Lambda * w_term) * g_r / N_r
    r_full  = [ r ; rho_1 ; rho_2 ; ... ]                    length npixels + reg.nrows
    R       = sum over terms of  w_term * ||g_r / N_r||^2    Lambda-FREE (Section 5)
    O(theta) = ||r_full||^2 = chi2 + Lambda * R
```

**The rows' Jacobian is written closed-form; it is never obtained by differentiating `r_full`
itself.** Fix one row `r` of term `T` selecting name `v`, with anchor bin `a = a(r)` and `x_{v,j}`
the value `v` takes at bin `j` (in `v`'s own unit). By the quotient rule,

```
    d(rho_r)/d(x_{v,j}) = sqrt(Lambda * w_term)
                           * [ (dg_r/dx_{v,j}) / N_r  -  (g_r / N_r^2) * (dN_r/dx_{v,j}) ]
```

**`dg_r/dx_{v,j}`** is the operator's own linear coefficient at bin `j` — Section 1's formula
differentiated by hand with respect to `u_j = x_j - ref_j` at fixed grid (`d(u_k)/dx_j =
delta_{j,k}`, so `dg_r/dx_j` is nonzero only at the row's own local bins): a `D2` row's
coefficients are `2/((t_j-t_{j-1})(t_{j+1}-t_{j-1}))`,
`-2(1/(t_{j+1}-t_j)+1/(t_j-t_{j-1}))/(t_{j+1}-t_{j-1})`, `2/((t_{j+1}-t_j)(t_{j+1}-t_{j-1}))` at
bins `j-1, j, j+1` respectively; a `D1` row's are `∓ 1/(t_{j+1}-t_j)`; an `I` row's is `1`.
`src/inversion/regularizer.jl`'s `_operator_coeffs` is this table, written independently of the
value formula `_operator_rows` computes, so that RG12's ForwardDiff oracle (which differentiates
`_operator_rows` through the value path) is a genuine check of this table rather than a comparison
of one formula against itself (`conventions.md` 8.2).

**`dN_r/dx_{v,j}`** is Section 3's closing fact: zero for `absolute`, `initial` and `unit` (`N_r`
does not depend on `x` there), and `delta_{j,a}` for `relative` (`N_r = x_a`, so its only nonzero
partial is with respect to the anchor bin itself). Substituting:

```
    absolute, initial, unit:  d(rho_r)/dx_{v,j} = sqrt(Lambda w_term) * c_{r,j} / N_r

    relative:  d(rho_r)/dx_{v,j} = sqrt(Lambda w_term) * [ c_{r,j}/N_r - g_r * delta_{j,a}/N_r^2 ]
```

with `c_{r,j}` the operator coefficient above. **The quotient-rule term fires on exactly one
column per row — the anchor's — and nowhere else**: every off-anchor column of a `relative` row
looks exactly like `unit`'s own formula divided by `N_r`, which is why RG13's own oracle
comparison, not a term-by-term eyeball, is what actually catches a dropped quotient-rule term
(`dev/reports/phase3_report.md`, this chunk's remediation entry).

**The chain to θ** is unchanged from before: for a selected **fitted variable**, `x_{v,j}` depends
on exactly one theta entry (its own scalar entry, or its own entry at bin `j` if time-varying), so
`dx/dy` (the logit transform's own derivative, `docs/derivations/parameters.md`, `_dx_dy(x, a, b) =
(x-a)(b-x)/(b-a)`, evaluated at `v`'s own bounds) is the whole chain. For a selected **dependent**,
the chain continues through the expression tree: `x_{v,j}` is the dependent's own value at bin `j`,
which may depend on several fitted variables at once, and its partials with respect to every theta
index active at bin `j` are obtained by a small forward-mode dual pass — seed every such entry,
evaluate the dependent's expression tree (`_eval`, D6, already generic in the seeded type), read
off the partials — exactly the technique the data Jacobian's own per-bin pass
(`docs/derivations/jacobian.md` Section 3, D18) uses to obtain a kernel column's partials, applied
here to the dependent's own scalar value instead of to the measured spectrum. This is legitimate,
targeted automatic differentiation of the expression tree's own chain rule; it is not the "closed
form, never autodifferentiated" rule's target, which is the operator's and the normalization's own
formulas above — those are hand-derived in Sections 1, 3 and this one and never touched by
ForwardDiff in the production path (`_regularizer_closed_form`, `jacobian.jl`'s `jvp!`/`jtvp!`).

## 5. `R` excludes `Λ`

`R` as defined in Section 4 carries no `Λ`: `R = sum_terms w_term ||g_r / N_r||^2`, and `𝒪 = chi2 +
Λ R` (foo2026 Eq. 8) is where `Λ` enters. This matters at `Λ = 0`: the unregularized fit's own
`R` is still well-defined and reported there (nothing about evaluating `R` requires `Λ ≠ 0`),
which is exactly the denominator the weight-scan's normalized `(chi2, R)` table (foo2026 Fig. 1,
Step 3.5) needs. RG11 pins this by comparing `sum(abs2, r_full[regularizer rows])` — which *does*
carry `Λ` folded into each row via `sqrt(Λ w_term)` — against `Λ * R` computed independently.

## 6. The interval-edge rule

A term's `interval` (`0` for the whole axis, else an index into `pspec.intervals`) restricts
*which bins the operator ever reads*: Section 1's `bins` range is the interval's own `FitInterval`
bins, and the operator's row range (`first(bins)+1 .. last(bins)-1` for `D2`, etc.) never reaches
outside it. **A difference that would straddle the interval's edge is therefore never built at
all** — not built and clamped, not built and discarded, simply never constructed, because the
loop bounds themselves stop short of it. This is what lets two terms on adjacent, non-overlapping
intervals (`"pre"`/`"post"`) leave a genuine discontinuity between them unpenalized (RG6): each
term is blind to the other side of the boundary by construction, and a single whole-axis term
placed over the same trace would instead see the jump (its own row spanning the boundary) and
penalize it heavily. Nothing about interval resolution assumes intervals partition or avoid
overlapping the axis (D5); a shared bin between two intervals is simply a bin that gets
regularized by both terms that claim it, and "two terms may select the same variable; their
contributions add" (D22) already covers that case with no special handling.
