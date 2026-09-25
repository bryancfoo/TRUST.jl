# The uncertainty reports: the minimum, the Laplace covariance, the calibration scan

> **Front matter (Step 4.A.4a, 2026-09-25).** Windows 11, Julia 1.12.7; ForwardDiff 1.4.5 (main
> environment), 1.4.6 (test environment — a pre-existing skew between `Manifest.toml` and
> `test/Manifest.toml`, flagged by Pkg's own warning on every `Pkg.test()` run; see OP21 below).
> Written alongside `src/inversion/uncertainty.jl`. Spec D43-D47,
> `dev/phase4A_instructions.md` Section 10, [OQ-24](../../dev/open_questions.md#oq-24),
> [OQ-36](../../dev/open_questions.md#oq-36).

This document derives `src/inversion/uncertainty.jl`: the Gauss-Newton predicted decrease that
every report below checks before it trusts a point as a minimum; the normal matrices `JᵀJ` and
`J_dᵀJ_d`, assembled without ever forming `J` densely; the polish step that reaches that
minimum; the Laplace covariance's posterior and sampling forms; and the calibration scan.

## 1. Why every report checks it is at a minimum

A joint fit's own `converged = true` is not proof of a minimum (`dev/reports/phase4A_report.md`,
chunk 8's finding F2): the deck-start LBFGS fit reports `converged` at an objective 42x above a
reachable one. The check used throughout this file is the decrease a full Gauss-Newton step would
achieve, in **𝒪's own units**, where a change of 1 is one σ ([OQ-36](../../dev/open_questions.md#oq-36)):
a threshold stated this way means the same thing on every problem, unlike a threshold on the raw
gradient norm, whose scale is set by the bound widths of whichever variables happen to be fitted
(`dx/dy` in the chain rule, `docs/derivations/parameters.md`).

## 2. The normal matrices, without a dense `J`

With `J` the Jacobian of `r_full` (data rows and, when present, regularizer rows) and `J_d` its
data rows alone,

```
JᵀJ    = sum_k  (J e_k)ᵀ (J e_k)        assembled column by column
J_dᵀJ_d = sum_k  (J_d e_k)ᵀ (J_d e_k)
```

`normal_matrices` builds both from `ntheta` unit vectors `e_k`, each costing one `jvp!` (`J e_k`,
every row) and two `jtvp!` (`Jᵀ(J e_k)` for the first, `J_dᵀ(J_d e_k)` for the second, the second
built by zeroing `(J e_k)`'s regularizer rows before the second `jtvp!` — a data-rows-only
`jtvp!` is exactly `J_dᵀ` applied to `J_d e_k`, since `jtvp!`'s regularizer contribution reads only
the rows past `problem.npixels`, which are now zero). `J` itself is never formed: at `ntheta =
1280` (the full-resolution two-channel problem, `p = ntheta`) a dense `J` would be **807 MB**
(Pitfall 13, `dev/phase4A_instructions.md` Section 14 pitfall 13); the block-structured
`jvp!`/`jtvp!` machinery of `docs/derivations/jacobian.md` costs a few MB of blocks instead. Cost
is `ntheta` `jvp!` and `2·ntheta` `jtvp!` calls; Level 2 measured 0.6-0.8 s at `ntheta = 128` on
the full-size two-channel problem (Section 12 row 13), reproduced in this chunk's own hand-off.

**With no regularizer, `problem.npixels == nrows`**, so "zero the regularizer rows" zeroes an
*empty* range — `JdtJd` is therefore **bitwise** equal to `JtJ` with no special-casing at all
(UN2): both matrices are built from literally the same `col` on every column. This is not a
coincidence pinned after the fact; it falls directly out of the `npixels == nrows` identity, and
UN1's discrimination revert (a) — skip the zeroing unconditionally — makes `JdtJd` disagree with
its own dense oracle by 6.4e-5 relative exactly where a regularizer *is* present, which is the
guard this identity cannot provide on its own.

## 3. `gn_decrease`: the Gauss-Newton predicted decrease

Write `𝒪 = rᵀr`, so `g = ∇_y 𝒪 = 2Jᵀr` (`gradient!`). The Gauss-Newton step solves the linearized
normal equations `(JᵀJ) Δ = -Jᵀr = -g/2`, i.e. `Δ = -(JᵀJ)⁻¹g/2`. Substituting into the quadratic
model `𝒪(y+Δ) ≈ 𝒪(y) + gᵀΔ + Δᵀ(JᵀJ)Δ` (the Gauss-Newton Hessian is `2JᵀJ`, so the quadratic term
of the Taylor expansion, `(1/2)Δᵀ(2JᵀJ)Δ`, is `ΔᵀJᵀJΔ`):

```
predicted decrease = -(gᵀΔ + ΔᵀJᵀJΔ)
                    = -( -gᵀ(JᵀJ)⁻¹g/2  +  gᵀ(JᵀJ)⁻¹g/4 )
                    = gᵀ(JᵀJ)⁻¹g / 4
```

restricted to the free (not near-bound) entries `f`: `gn_decrease = g[f]ᵀ (JtJ[f,f])⁻¹ g[f] / 4`,
one Cholesky solve. **Near-bound entries are excluded from every solve in this file** (Pitfall 33):
there `dx/dy → 0` (the logit's derivative, `docs/derivations/parameters.md`), so a near-bound
column of `J` is *identically* zero and `JᵀJ` restricted to the full set is singular to round-off
— Level 2 measured eigenvalues of -2.2e-10 with a near-bound entry left in, and a `cholesky` call
over that set throws `PosDefException`.

## 4. `polish`: reaching the minimum

From a starting θ (typically an optimizer's own "converged" result, D43's own table shows this can
sit at a Gauss-Newton decrease of 1.25 — more than a σ away from a true minimum), `polish` repeats
up to `maxsteps` times: recompute the free set, `JᵀJ` and `g`; record `gn_decrease`; stop if it is
`<= tol` (default `1e-6`, two orders below OQ-36's "a hundredth of a σ" refusal threshold `laplace`
and `calibration_scan` use, so a polished point comfortably clears their own gate); otherwise take
the Gauss-Newton step `Δ[f] = -(JtJ[f,f])⁻¹ g[f] / 2` with backtracking (`α = 1`, halved down to
`1/1024` while `𝒪` fails to decrease). D43's own measured table on the MC fixture: `1.25 → 0.90`
(the staged start plus one LM iteration) `→ 2.0e-4 → 7.7e-8 → 6.3e-11 → 5.8e-14` over four GN
polish steps, each step's actual objective drop matching its own prediction to three digits
(0.903 predicted against 0.902 measured) — the quadratic model above is trustworthy exactly where
it is used, close to a minimum.

The returned `FitResult` keeps the original solve's own `status`, `converged`, `iterations`,
`f_calls`, `g_calls`, `termination` and `optimizer` unchanged — `polish` is bookkeeping on top of
an existing result, not a new kind of fit — with the trajectory gaining one row per **accepted**
step, numbered after the solve's own last, and `elapsed` including the polish's own wall time.

## 5. The Laplace covariance

At a checked minimum θ (`laplace` throws otherwise, naming the decrease found, if it exceeds
`max_decrease = 1e-2`, OQ-36's "a hundredth of a σ"):

```
Cov_y^post = (JᵀJ)⁻¹
Cov_y^samp = (JᵀJ)⁻¹ J_dᵀJ_d (JᵀJ)⁻¹
sigma_x    = |dx/dy| · sqrt(diag(Cov_x))
```

both restricted to the free set, `NaN` elsewhere. **The posterior form is the credible width under
the smoothness prior**: it is the ordinary Laplace approximation to the posterior covariance of a
Gaussian-prior, Gaussian-likelihood problem (`𝒪` is `-2 log posterior` up to a constant,
[OQ-20](../../dev/open_questions.md#oq-20)), correct if the prior (the regularizer, penalizing a
particular curvature) is taken at face value. **The sampling form is what a bootstrap or the
per-slice scatter (Step 4.A.4b) would actually measure**: it is the sandwich covariance of a
point estimator under repeated *data* noise alone, `J_dᵀJ_d` standing in for the data term's own
Fisher information while the regularizer contributes only through the shared `(JᵀJ)⁻¹` that
determines *where* the fit sits, not through its own (fictitious) sampling variability. **The two
agree exactly when there is no regularizer** (`J_d == J`, so `J_dᵀJ_d == JᵀJ` bitwise, Section 2
above) and differ once one is present: on the MC fixture, `cov_y` diagonal entries differ by
5.3e-4 relative, which becomes 2.6e-4 on `sigma` after the `sqrt` (UN4) — a smoothness prior that
narrows the posterior relative to what the data alone would say is exactly what a regularizer is
*for*, so the two forms disagreeing is the expected outcome, not a bug to chase.

Per fitted variable (`vary == true`, dependents have no bound and so no `dx/dy` of their own,
[propagate] — reported nowhere in this chunk), σ has the same shape as `FitResult.values[name]`:
one entry per bin for a time-varying variable (`NaN` outside its own interval, matching
`_build_result`'s own convention), length 1 for a scalar. A near-bound entry's σ is `NaN`, flagged
in `near_bound[name]`, never a number (Pitfall 33 again).

## 6. The calibration scan

For each requested shift `δ_k` (nm, including `0.0`), channel `channel`'s own
`WavelengthCalibration.offset` is fixed at its own value plus `δ_k · 1e-9`, the fit is
re-solved warm-started from `theta_star` and then polished, and every fitted variable's movement
is read off against the `δ = 0` point:

```
Δθ_v,j(δ_k) = θ*_v,j(δ_k) − θ*_v,j(0)
s_v,j^sys   = max_k |Δθ_v,j(δ_k)|
```

**The rebuild is the identity at `δ = 0`**: the channel's `Experiment` is reconstructed from the
same trace object (`Pitfall 7`, `experiment.trace === pspec.trace` for every experiment) with a
`WavelengthCalibration` whose `offset` is numerically unchanged, so `solve` from `theta_star`
either meets `g_abstol` immediately or rejects its very first trial step and stops by D44 — and
the subsequent `polish`, already below `tol`, takes no step. `theta_star` therefore comes back
**bitwise** at `δ = 0` (CS1), which is what makes `delta`'s own `δ = 0` column exactly zero without
any special-casing of that point in the loop. **The offset is fixed, never fitted**: a binding of
the target channel's calibration `:offset` field is refused at construction (Pitfall 36) — the scan
*is* the alternative to fitting it.

On the MC fixture (channel 2, `[-0.01, 0.0, 0.01]` nm), χ² is 104.67 / 100.46 / 119.46: the
fixture's own offset (0.01 nm, unbound) is the scan's own minimum, as it must be, since the
synthetic data were generated at it.

## 7. CS4: a structural failure to reach `tol`, not a bug

On the reproductions' own 128 x 32 EPW problem (96 fitted parameters, `weight = 2e-4`), the
calibration scan's `-0.1` nm point warm-started from θ\*(0) settles, under `LevenbergMarquardt`'s
default seed, into a point with `gn_decrease = 0.0057` -- above `polish`'s `tol = 1e-6` -- and
`polish`'s own first Gauss-Newton step finds no decreasing `alpha` down to `1/2^20`, though `g'·s`
is negative (a genuine descent direction under the local linear model). Scanning `chi2` as a pure
function of the one coordinate the step moves most (`logn` at bin 27, holding everything else
fixed) shows why: `chi2` is smooth on both sides but sits about 0.19 *lower* only in a window
roughly `1e-5` wide in `x` around θ\*(0)'s own value there -- a needle, not a basin, which a
Gauss-Newton step built from the smooth curvature just outside it cannot navigate into or out of
from a point sitting at its bottom. Eight different `LevenbergMarquardt` seeds and `LBFGS` all
land within a factor of about 2 of each other's residual `gn_decrease` (0.0031-0.0083) at this
same shift, confirming the needle is a structural property of this fit's own noise realization,
not an artefact of one optimizer seed; `normal_matrices`, `gn_decrease`, `polish` and
`calibration_scan` are each independently verified correct on the fast suite's own MC fixture
(UN1-UN6). `test/slow/test_uncertainty.jl`'s CS4 asserts `all(scan.decrease .<= 1e-6)` exactly as
D46/Section 10.1 states it, and is expected to fail on this one point -- reported, not tuned away
(`CLAUDE.md`: "a fit that does not reach its stated accuracy is a finding... do not tune inits,
bounds, weights, the regularizer or the optimizer to make it pass"). See the chunk 9 hand-off for
the full diagnostic trace.

## 8. OP21 and the ForwardDiff manifest skew

D44's fix (`src/inversion/optimizers.jl`, the `LevenbergMarquardt` reject branch) stops on a
rejected trial step whose predicted reduction is at or below round-off
(`2·pred <= 1e-12 · max(2F, 1.0)`), giving `:f_tolerance` rather than letting `mu` double without
bound. OP21 restarts LM at its own converged joint result and checks the outcome. Level 2's own
scratch measurement (a plain script run under the main environment, ForwardDiff 1.4.5) found this
**bitwise**, one iteration. Run through the actual test harness (`Pkg.test()`, which resolves
`test/Manifest.toml` — ForwardDiff 1.4.6, a version this repository has carried as a standing skew
against the main `Manifest.toml` since before this chunk, flagged by Pkg's own warning on every
run), the restart consistently takes **two** iterations and lands about 9.8e-9 away from the first
solve's own θ. The fix itself is not in question — neither environment throws, and both terminate
by the round-off criterion — but OP21's literal "bitwise, one iteration" sits exactly on the knife
edge that criterion tests, so this tiny, version-driven difference in the assembled Jacobian is
enough to move it. `test/inversion/test_optimizers.jl`'s OP21 row therefore asserts
`isapprox(...; rtol = 1e-6)` rather than `==`, with this paragraph's reasoning inlined as a comment
at the row itself; see the chunk 9 hand-off for the user's decision on record.
