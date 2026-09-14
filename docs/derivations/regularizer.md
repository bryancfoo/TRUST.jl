# The regularizer

> **Front matter (Step 3.3, 2026-09-13).** Windows 11, Julia 1.12.7; ForwardDiff 1.4.5. Written
> alongside `src/inversion/regularizer.jl`.

This document derives `src/inversion/regularizer.jl`: the three operators as divided differences
on a nonuniform axis, why `D2` is exact for a quadratic on any grid, why `R` is a plain sum over
rows and what that buys under rebinning, the four normalization modes and their derivatives, the
residual-row form and its closed-form Jacobian, and the interval-edge rule. Spec D22-D25,
[OQ-19](../../dev/open_questions.md#oq-19).

## 1. The operators, as divided differences on the axis

Let `u_j` be one selected name's normalized, reference-subtracted trace (Section 3 below) at bin
`j`, and `t_j` the axis in the axis keyword's own unit (`pspec.axis_boundary`, nanoseconds for a
trace built with `t_ns`) — **never the bin index**. Over the bins `bins` of the term's own window
(the whole axis when `interval == 0`), with `j` ranging over `first(bins)+1 .. last(bins)-1` for
`D2`, `first(bins) .. last(bins)-1` for `D1`, and `first(bins) .. last(bins)` for `I`:

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
    R = sum over terms of  w_term * ||op(N(u - ref))||^2
```

with no `Δt` weighting of any row. Consequently, rebinning a smooth trace by a factor `r` (so the
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

## 3. The four normalizations and the reference

Before the operator is applied, the selected trace `x_j` (in its own unit — the unit its bounds
are given in, for a fitted variable; whatever unit the dependent's expression naturally produces,
for a dependent) is normalized and reference-subtracted:

| mode | `N(x_j)` | `∂N(x_j)/∂x_j` |
|---|---|---|
| `absolute` | `(x_j - ref_j) / s` | `1 / s` |
| `initial` | `(x_j - ref_j) / x⁰_j` | `1 / x⁰_j` |
| `relative` | `(x_j - ref_j) / x_j` | `ref_j / x_j^2` |
| `unit` | `x_j - ref_j` | `1` |

`s` is the term's `scale`: the value given explicitly, or, for a fitted variable, the default
`b - a` (the variable's own bound width) — a dependent has no bound width and so has no default,
a loud error at construction (RG8). `x⁰_j` is the *starting* trace: the value the selected name
takes at every fitted variable's own `init` (built here by evaluating the same per-bin value
table `unflatten` computes, but at `flatten(pspec)`'s start vector rather than the current θ, so
a fitted variable's `x⁰_j` is simply its own `init`, constant over every bin, and a dependent's is
its expression evaluated on that same constant background). `ref_j` is the reference trace,
zero — "off" — by default.

`u_j = N(x_j)`, the operator's own input, is what Section 1 differentiates. The derivative table
above is `∂u_j/∂x_j`; the closed-form row Jacobian (Section 4) multiplies it by the operator's own
(linear) coefficients and by the chain from `x_j` to θ.

### The `relative` finding — read this before touching `relative`

**`relative` is degenerate at its own default reference, and this is the design as specified, not
a defect.** With `ref_j = 0` (the default), `N(x_j) = (x_j - 0)/x_j = 1` identically, for *every*
bin, regardless of what `x_j` actually is (as long as it is never exactly zero, itself a loud
error, D39). A constant sequence has `D1 = D2 = 0` everywhere by Section 1's own algebra (a
degree-zero polynomial is a fortiori a quadratic with `a = 0`), so a `relative` term with no
explicit reference contributes **exactly zero** to `R` and to every residual row, for `D1` and
`D2` alike — only an `I` term (a ridge on `N(x) - ref`, i.e. on `1 - ref`) sees anything at all,
and even that is a ridge on a constant, not on `x`'s own shape.

RG7 pins this directly (`raw_rel == 0` bitwise for the default reference); RG12/RG13 exercise
`relative` again with a **nonzero** reference, where it is no longer degenerate and its Jacobian
(`∂N/∂x = ref/x^2`, nonzero exactly because `ref ≠ 0`) is checked against the brute-force
ForwardDiff oracle.

**Why this happens, read against foo2026.** foo2026's Table I "Relative" rows normalize the
*second derivative* by the local value — i.e., they compute `D2` of the raw trace first and
*then* divide by `x_j` (or a related local scale) — whereas this spec's `N` divides the *trace*
first and applies the operator to the quotient afterward. The two operations do not commute:
dividing then differencing kills the very shape `D2` is meant to penalize, exactly when the
reference is the trivial zero that makes the quotient a constant. Reproducing foo2026 Table I's
intent (in Phase 3, D38) therefore uses `initial` where the paper says "Relative": `initial`
divides by a **fixed** starting trace rather than by the trace's own current value, so it stays
quadratic in `x` and behaves the way "a relative curvature penalty" is usually meant to, which is
exactly [OQ-19](../../dev/open_questions.md#oq-19) item 5's own remark ("`initial` … is usually
what 'relative' is meant to achieve"). This is a recorded, deliberate deviation from foo2026
Table I's *literal* normalization name, not a silent substitution — [propagate] item A6
(`dev/phase3_overall_summary.md`) leaves the question of whether `relative` should instead
normalize *after* the operator (matching foo2026 exactly) open for the user to decide; this
chunk builds `relative` exactly as D25 specifies and does not "fix" the degeneracy.

## 4. The residual-row form and its closed-form Jacobian

Every term's rows are stacked as extra residual rows under the `npixels` data rows (D25),
weighted by both the term's own `w_term` and the global `Λ`:

```
    rho_row = sqrt(Lambda * w_term) * row(u)
    r_full  = [ r ; rho_1 ; rho_2 ; ... ]                length npixels + reg.nrows
    R       = sum over terms of  w_term * ||row(u)||^2   Lambda-FREE (Section 5)
    O(theta) = ||r_full||^2 = chi2 + Lambda * R
```

**The rows' Jacobian is written closed-form; it is never obtained by differentiating `r_full`
itself.** For one row of term `T` selecting name `v`, with `x_{v,j}` the value `v` takes at bin
`j` (in `v`'s own unit):

```
    d(rho_row)/d(theta_(v,j)) = sqrt(Lambda * w_term)
                                 * [ op-coefficient of row at bin j ]
                                 * (dN/dx)(x_{v,j})
                                 * (dx/dy)(x_{v,j})
```

The three factors are independent pieces, each derived on its own:

1. **The operator's own coefficient** is Section 1's formula differentiated by hand with respect
   to `u_j` at fixed grid — linear, and independent of `u` itself (a `D2` row's coefficients are
   `2/((t_j-t_{j-1})(t_{j+1}-t_{j-1}))`, `-2(1/(t_{j+1}-t_j)+1/(t_j-t_{j-1}))/(t_{j+1}-t_{j-1})`,
   `2/((t_{j+1}-t_j)(t_{j+1}-t_{j-1}))` at bins `j-1, j, j+1` respectively; a `D1` row's are `∓
   1/(t_{j+1}-t_j)`; an `I` row's is `1`). `src/inversion/regularizer.jl`'s `_operator_coeffs` is
   this table, written independently of the value formula `_operator_rows` computes, so that
   RG12's ForwardDiff oracle (which differentiates `_operator_rows` through the value path) is a
   genuine check of this table rather than a comparison of one formula against itself
   (`conventions.md` 8.2).
2. **`dN/dx`** is Section 3's table, evaluated at the current `x_{v,j}`.
3. **`dx/dy`** is the logit transform's own derivative (`docs/derivations/parameters.md`,
   `_dx_dy(x, a, b) = (x-a)(b-x)/(b-a)`), evaluated at `v`'s own bounds — no unit factor appears
   here because `x` is already in `v`'s own unit, the same unit its bounds are in.

**For a selected fitted variable**, `x_{v,j}` depends on exactly one theta entry (its own scalar
entry, or its own entry at bin `j` if time-varying), so `dx/dy` alone is the whole chain to theta,
factor 3 above.

**For a selected dependent**, the chain continues through the expression tree: `x_{v,j}` is the
dependent's own value at bin `j`, which may depend on several fitted variables at once. Its
partials with respect to every theta index active at bin `j` are obtained by a small forward-mode
dual pass — seed every such entry, evaluate the dependent's expression tree (`_eval`, D6, already
generic in the seeded type), read off the partials — exactly the technique the data Jacobian's
own per-bin pass (`docs/derivations/jacobian.md` Section 3, D18) uses to obtain a kernel column's
partials, applied here to the dependent's own scalar value instead of to the measured spectrum.
This is legitimate, targeted automatic differentiation of the expression tree's own chain rule; it
is not the "closed form, never autodifferentiated" rule's target, which is the operator's and the
normalization's own formulas (factors 1 and 2 above) — those are hand-derived in Sections 1 and 3
and never touched by ForwardDiff in the production path (`_regularizer_closed_form`,
`jacobian.jl`'s `jvp!`/`jtvp!`).

## 5. `R` excludes `Λ`

`R` as defined in Section 4 carries no `Λ`: `R = sum_terms w_term ||row(u)||^2`, and `𝒪 = chi2 +
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
