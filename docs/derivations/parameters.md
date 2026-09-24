# The parameter layer

> **Front matter (Step 3.1, chunk 4, 2026-09-13).** Windows 11, Julia 1.12.7. Written alongside
> `src/inversion/parameters.jl` and the narrowing of `_probe_construction` in
> `src/inversion/expressions.jl` (chunk 3's file). `[propagate]`: `conventions.md` 9.1 names only
> `jacobian.md` and `regularizer.md` for Phase 3; this file is not in that list, and is written
> anyway per spec D41's own instruction, since the expression grammar and the binding rules have
> nowhere else to live.

## 1. The logit transform

Every fitted variable has finite bounds `a < b` and passes through one universal transform. In
the code's symbols, with `y` the unconstrained value an optimizer works in and `x` the variable's
own value:

```
s(y)   = 1 / (1 + exp(-y))                       the logistic
x      = a + (b - a) · s(y)                      _from_unconstrained
y      = log((x - a) / (b - x))                  _to_unconstrained
dx/dy  = (x - a)(b - x) / (b - a)                 _dx_dy, evaluated from x
```

`dx/dy` is written from `x`, not `y`: algebraically `(b-a)·s(y)·(1-s(y))` is the same quantity, but
the `x`-form needs no second call to `exp` and is what the Jacobian's chain rule (Step 3.2) uses.

**Why bounds must be finite.** An infinite bound would make `s(y)` map onto a half-open or open
interval reaching `±∞`, and `_to_unconstrained` of any interior `x` would still be finite while the
*range* itself is not representable as `b - a`. Rather than special-case an infinite bound, the
design refuses it: "positive and unbounded" is written as `exp` inside a **dependent** expression
(`ne_cm3 = "10^logn"` with `logn` bounded, Pitfall 9), never as an unbounded `FittedVariable`.

**Overflow safety.** `_logistic` branches on the sign of `y` (at `y`'s *primal* value for a dual,
`conventions.md` 5.12) so that the argument to `exp` is never positive:

```julia
_logistic(y) = y >= 0 ? inv(1 + exp(-y)) : (e = exp(y); e / (1 + e))
```

**Verified directly, 2026-09-13, rather than assumed from the spec's own rationale** (which reads
"the naive form ... overflows to `Inf` ... and returns `NaN`" without saying which of the two
possible single formulas, or at which sign of `y`): on this machine, the single formula
`1/(1+exp(-y))`, applied unconditionally, does **not** itself produce `NaN` — `exp(-y)` overflows
to `Inf` at very negative `y`, and `1/(1+Inf)` is `0.0`, the correctly-rounded (if imprecise)
answer, not `NaN`. The formula that **does** overflow to `NaN` is the other single formula,
`exp(y)/(1+exp(y))`, applied unconditionally at large **positive** `y`: `exp(y)` overflows to
`Inf`, and `Inf/(1+Inf)` is `Inf/Inf`, which is `NaN`. The branched form above sidesteps both
failure modes at once by keeping `exp`'s argument non-positive on both sides of `y = 0`, where
`exp` cannot overflow — the specific pairing of "which formula on which side" is what matters, not
just "no `Inf`".

**What overflow safety does *not* buy: strict interiority at extreme `y`.** Measured directly
(2026-09-13, this machine):

```
_logistic(-800.0)                      = 0.0            exactly
_logistic(800.0)                       = 1.0            exactly
_from_unconstrained(-800.0, 50, 800)   = 50.0            exactly a
saturates to exactly 1.0 for y >~ 37.0
underflows to exactly 0.0 for y <~ -746.0
```

Float64 has no representable value between `1 - eps(1.0)/2 ≈ 1 - 1.11e-16` and `1.0`, so `s(y)` is
the correctly-rounded `1.0` for any `y` past about `37` — this is a property of *double-precision
density near 1*, not of this formula, and no alternative formula changes it (confirmed
2026-09-13: computing the positive branch as `1 - e/(1+e)` with `e = exp(-y)`, mirroring the
negative branch's own trick, saturates at the identical threshold, because `1 - tiny` rounds to
`1.0` regardless of how `tiny` was produced). The negative side reaches much further, to about
`y ≈ -746`, because subnormal doubles extend density near `0` far below `eps(1.0)`'s scale near
`1` — a genuine asymmetry in what Float64 can represent, not a code asymmetry. `_from_unconstrained`
therefore guarantees `a <= x <= b` and finiteness for every finite `y`, never strict interiority at
the extremes; `FittedVariable`'s own `init` is required strictly inside precisely because *that*
value is read once and never subject to this saturation the way an optimizer's iterate can be.

## 2. The expression grammar

A dependent's text is parsed with `Meta.parse` and lowered into a typed tree
(`AbstractExprNode`, `src/inversion/expressions.jl`), never `eval`ed. The accepted and rejected
`Expr` heads:

| Accepted | Becomes |
|---|---|
| a number literal | `ConstNode` |
| `:t` | `TimeNode` |
| a name in scope (a fitted variable or another dependent) | `SlotNode` |
| `Expr(:call, f, args...)`, `f` a comparison (`<,<=,==,!=,>,>=`) | `CompareNode` |
| `Expr(:call, :ifelse, cond, a, b)` | `IfElseNode` |
| `Expr(:call, f, args...)`, any other `f` resolved in `Base`/`SpecialFunctions` | `CallNode` |
| `Expr(:if, cond, a, b)` (`a ? b : c`) | `IfElseNode` |
| `Expr(:&&, a, b)` / `Expr(:\|\|, a, b)` | `CallNode` over `&`/`\|` |
| `Expr(:block, ...)` with one non-`LineNumberNode` statement | that statement |

| Rejected (loud error naming the construct) | `Expr` head |
|---|---|
| assignment | `:(=)` |
| indexing | `:ref` |
| a loop | `:for`, `:while` |
| a function definition | `:function`, `:->` |
| a macro call | `:macrocall` |
| a qualified name | `:.` |
| a string literal | `:string` |
| a tuple | `:tuple` |

**Indexing being unrepresentable is what enforces "pointwise in time."** A dependent may combine
the bin-`j` values of any variables and dependents with `t`, never bin `j-1` or any other bin's
value — `Expr(:ref, ...)` simply has no lowering. This is not a convenience restriction: it is what
keeps Step 3.2's Jacobian block-diagonal-in-`j` (per bin, before the temporal IRF spreads it),
since nothing in the grammar can make one bin's dependent value depend on another bin's variable
value.

**The read-time probe is two halves**, revised 2026-09-13 after the first cut (checking finiteness
at placeholder values `slot=1.0, t=0.0`) was found to refuse `"1/(x-1)"`, `"log(t)"`, `"1/t"`,
`"sqrt(t-1)"` and `"exp(x*1000)"` — five expressions that are perfectly fine at real values and
`Inf`/`-Inf`/a `DomainError` only at the placeholder:

1. **Structural** (`_probe_construction`, `_parse_expression`, at the dependent's read time):
   placeholder `slot=1.0`, `t=1.0`; converts **only a `MethodError`** (a wrong arity, an
   uncallable resolved name) into a loud failure; ignores every other exception and every
   non-finite result, since a placeholder cannot distinguish a broken expression from a good one
   evaluated at the wrong point.
2. **Value** (`_probe_initial_values`, `ParameterSpec`'s constructor): every dependent evaluated at
   the *real* `init` values of every variable, at *every bin of the trace's axis* (dependents carry
   no interval of their own; D12 evaluates them unconditionally at every bin), requiring a finite
   `Real`. This is what catches a genuine `DomainError` or a typo that parses to something
   meaningless, and it costs one pass over `N_t × N_dependent` at read time.

## 3. Bindings

A binding plugs one fitted variable or dependent's *name* into one target — a plasma leaf
(`kind = :leaf`) or an entry of an instrument effect's `parameters` (`kind = :effect`) — by writing
that name as a string where the deck (or, here, the hand-built problem) expects a number.

**A variable carries no unit.** Its `bounds` and `init` are read in the unit of whichever key it is
first bound through; every *other* binding of that same variable must convert through the
identical factor. This is stricter than a same-row check: `T_eV` and `T_keV` are the same leaf row
with different factors, and a variable bound to both would give its one `bounds` pair two
different meanings, so it is refused (PM14) even though `ions.D.u_kms` and `ions.C.u_kms` — two
different leaves, the same factor — are exactly the shared-flow idiom the shock problem needs
(PM15) and are accepted.

**Instrument targets must be scalar.** A binding to an effect parameter whose name is a
time-varying variable, or a dependent that transitively reads one (walking the same dependency
graph the topological sort already built, so the check is free), is refused: the post-kernel
effects act on the whole streak once, and `Experiment` holds one effect tuple, not one per bin.

**The probe and the collection are fixed.** A binding whose target key begins `probe.` or
`collection.` is refused, naming `WavelengthCalibration` — since Step 4.A.1a (2026-09-21,
OQ-24) the effect a fit binds instead when the wavelength axis itself needs to move; the probe
wavelength itself is never bindable.

**An effect-parameter binding's factor comes from `_INSTRUMENT_BINDING`** (`src/config/deck.jl`,
D9, Step 4.A.1b, closing Phase 3's B5), not from an assumption that the deck key already equals
`parameters(effect)`'s own key: `_build_effect_and_pending!` looks the deck key up in that table
to get the `parameters` field, the factor and the unit key together, and `_resolve_bindings`
(here) writes them onto the `Binding` from the target's fourth and fifth elements. **A hand-built
binding still defaults to `factor = 1.0`, `unit_key = ""`** (the two- and three-element target
forms, `(idx, field)`/`(idx, field, element)`): only the deck path and a target written out in the
five-element form, `(idx, field, element, factor, unit_key)`, carry a non-trivial factor. The
one-conversion-factor rule above therefore covers an effect binding exactly as it covers a leaf
one — a variable bound to `electrons.e.T_eV` and to `spectral_irf.fwhm_nm` is refused, naming both
keys, even though only one of them is a plasma leaf. **`gradient_smearing.width` is the one instrument key
that still refuses a variable reference**: its factor is the smeared leaf's own
(`_leaf_si_path(e.leaf)[2]`), not a constant of the key, so it cannot live in a static table.

**Two variables may never write one leaf**, but one variable may write many leaves (the
one-conversion-factor rule above is what keeps that meaningful). A leaf the trace does not already
tabulate gains a new row, built once at `ParameterSpec`'s construction with the union of the
original leaves (in their original order) and every newly bound leaf (in binding order), each new
row initialized to the template's own constant value.

## 4. Theta's layout and the index map

Theta is **variable-major**: in declaration order over the `vary = true` variables, a scalar
variable contributes one entry and a time-varying one contributes one entry per bin of its own
interval (the whole axis, if it has none), in increasing bin order.

```
index(v)     = offsets[v]                                for a scalar v
index(v, j)  = offsets[v] + (j - first(bins(v)))         for a time-varying v, j in bins(v)
```

**Why variable-major and not bin-major.** The Jacobian is never dense (Step 3.2), so contiguity in
theta buys no memory locality there; and intervals make bin-major non-uniform in exactly the same
way variable-major is non-uniform (a bin's set of active entries already needs the explicit
per-bin list `bin_entries` regardless of layout). Variable-major wins on every count that remains:
a regularizer term's rows for one variable are one contiguous `theta` slice, and `flatten`'s and a
`FitResult`'s per-variable trace are views, not gathers.

### Worked example

Three variables over a six-bin axis `t_ns = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]`, declared in this
order:

- `f_D`, a **scalar**, bound to `ions.D.fraction`;
- `T_e`, a **time-varying** variable on the **whole axis**, bound to `electrons.e.T_eV`;
- `u_pre`, **time-varying** on the named interval `"mid"`, `t_range_ns = [0.6, 1.8]` — which
  resolves to bins `2:4` (`searchsortedfirst`/`searchsortedlast` against the axis, since the axis
  is strictly increasing and the bins in any window are therefore contiguous) — bound to
  `ions.D.u_kms`.

```
variable   slot   vary?   offset   width
f_D          1     yes       1       1
T_e          2     yes       2       6
u_pre        3     yes       8       3

ntheta = 1 + 6 + 3 = 10

bin_entries[1] = [2]          bin_slots[1] = [2]
bin_entries[2] = [3, 8]       bin_slots[2] = [2, 3]
bin_entries[3] = [4, 9]       bin_slots[3] = [2, 3]
bin_entries[4] = [5, 10]      bin_slots[4] = [2, 3]
bin_entries[5] = [6]          bin_slots[5] = [2]
bin_entries[6] = [7]          bin_slots[6] = [2]

global_entries = [1]          global_slots = [1]
```

`slot` is a variable's position in the per-bin value vector (D8: variables first, in declaration
order, then dependents in topological order); it is not the same number as a theta *index* — `T_e`
is slot 2 but occupies theta indices 2 through 7, one per bin. `bin_entries[j]`/`bin_slots[j]` list
only the *time-varying* variables active at bin `j` (`u_pre` is absent from bins 1, 5 and 6); a
scalar variable (`f_D`) never appears there, appearing instead once in `global_entries`, since it
is fitted once and used at every bin regardless of which bin is being evaluated.

Every line above is reproduced exactly by PM22, written out by hand against the same six-bin,
three-variable problem before the code that builds it, per the chunk's own trap 1.

## 5. The channel on a binding, and `unflatten`'s channel (Phase 4.A, Step 4.A.3a, D32, D33)

`ParameterSpec` may be built over several channels' `Experiment`s at once,
`ParameterSpec((expt_1, …, expt_C); …)` (`ParameterSpec(expt; …)` is `ParameterSpec((expt,); …)`,
bitwise), one experiment per channel, all sharing **one** trace object (checked at construction: a
mismatched trace is a loud error naming which channel). A leaf binding still means exactly what it
always has — the shared plasma every channel reads — but an **effect** binding must now also say
*which* channel's `Experiment.effects` its `effect` index counts into, since two channels each have
their own effect chain and their own numbering.

`Binding.channel` carries this: **`0` for a `:leaf` binding** (a leaf reaches every channel, so the
channel field is meaningless for it and is never read), and **the target channel for an `:effect`
binding**. Three target forms resolve it:

- the three original tuples, `(effect, field)` / `(effect, field, element)` /
  `(effect, field, element, factor, unit_key)`, are unchanged and always mean **channel 1** —
  a single-channel spec never writes anything else, so every existing hand-built binding and every
  deck-driven one keeps compiling and keeps meaning what it always meant;
- the new `NamedTuple` form, `(channel = c, effect = i, field = :f[, element, factor, unit_key])`,
  names its channel explicitly and is the only form a joint spec's channel-2-and-beyond bindings
  use; an unknown key, a missing `channel`/`effect`/`field`, or a `channel` outside
  `1:length(expts)` is a loud error naming the variable (PM29's three errors).

A variable may be bound into **more than one channel** — the same field of the same effect
*index* in two different channels is two different parameters and is legal (D32); the same field
of the same effect index **within one channel** is still refused by `_rebuild_effect`'s
`only(matching)`, exactly as before. The one-conversion-factor rule (Section 3) is unchanged and
reads **per variable name**, so a variable bound into both channels' `Sensitivity.scale` (a shared
gain) is one factor, checked across every one of its bindings regardless of channel.

**`unflatten(expt, pspec, theta; channel = 1)`** applies every `:leaf` binding exactly as before
(the shared plasma does not care which channel is asking), but only the `:effect` bindings with
`b.channel == channel` rebuild `expt`'s own effects — so calling `unflatten` on channel 2's
`Experiment` with `channel = 2` sees only channel 2's own effect bindings, and channel 1's
`scale`/background/calibration bindings are simply never looked at, by construction, not by a
value happening to be absent. **Every call to `unflatten` on a `FitProblem`'s own experiment must
pass `channel = problem.channel`** (Pitfall 23): the two production call sites are
`residuals!`/`jvp!`'s underlying `residuals!(r, problem::FitProblem, theta)`
(`src/inversion/residuals.jl`) and `jacobian(problem::FitProblem, theta)`
(`src/inversion/jacobian.jl`); `_build_result` (`src/inversion/optimizers.jl`) calls it once per
channel of whichever problem it was given, `channel = ch.channel` for each. Missing the keyword
anywhere silently reads as "channel 1", which applies a channel-2 binding to channel 1's effect of
the same index whenever the two happen to share a field name, and raises a `parameters`-layer
error (or does nothing at all) when they do not — the trap is silent precisely when it looks most
innocuous, which is why MC8 and the chunk 6 discrimination check's revert (a) exist.

The Jacobian's own channel filter (`src/inversion/jacobian.jl`) mirrors `unflatten`'s in three
places, all gated on the same `b.channel == channel` test: `_collect_kernel_bindings` (which
kernel-stage effect bindings seed the per-bin dual pass), the kernel-global selection loop (which
scalar variables get a `kernel_globals` column at all), and `_build_chain_globals` (which chain-
stage bindings get a closed-form column). A variable bound only into another channel therefore
never becomes a global of this channel's Jacobian: its column is identically zero because nothing
ever writes to it, not because something computes zero and writes it (JT18).
