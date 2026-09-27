# Velocity-distribution shapes

This document derives what `src/distributions/distribution.jl` and
`src/distributions/maxwellian.jl` implement, in the package's own normalization. See
`docs/derivations/plasma_dispersion.md` for `Z` and `Z'`.

## 1. The three normalizations, and why the variance is `1/2`

A shape `φ(x)`, `x = (v−u)/v_th`, is required to satisfy

    ∫ φ(x) dx = 1,      ∫ x φ(x) dx = 0,      ∫ x² φ(x) dx = 1/2.

Density, flow and temperature belong to the `Species`, not to the shape; `f_lab` (Section 3)
restores them. The `1/2` is not an arbitrary choice — it is one half of a matched pair with the
thermal speed `v_th = √(2T/m)` (`conventions.md` 1.4). Substituting `v = v_th x + u` into the
second **central** moment of the dimensional distribution,

    ∫ (v−u)² f_lab(v) dv = ∫ (v_th x)² φ(x) dx = v_th² ∫ x² φ(x) dx = v_th² · (1/2),

and with `v_th² = 2T/m`,

    ∫ (v−u)² f_lab(v) dv = (2T/m)(1/2) = T/m,

recovering the physical second moment of a distribution with temperature `T` and mass `m`,
independent of the `2` in `v_th`. Had the normalization instead fixed `∫x²φ dx = 1` (as if `x`
were already in units of one standard deviation), the `2` in `v_th` would have to move
somewhere else to keep this identity — the two conventions are inseparable, and Pitfall 2 is
what happens when only one of them is changed.

## 2. The Maxwellian's constants and the derivation of `𝒞φ(ξ) = Z'(ξ)`

The Maxwellian shape is

    φ(x) = e^{−x²}/√π,      φ'(x) = −2x e^{−x²}/√π,

which satisfies the three normalizations of Section 1 exactly (they are the standard Gaussian
moments `∫e^{−x²}dx/√π = 1`, odd first moment `0`, and `∫x²e^{−x²}dx/√π = 1/2`).

Its Cauchy integral is, by definition (`conventions.md` 1.5),

    𝒞φ(ξ) = ∫ φ'(x)/(x−ξ) dx = (1/√π) ∫ (−2x e^{−x²})/(x−ξ) dx.

Write `−2x = −2(x−ξ) − 2ξ` (adding and subtracting `ξ` in the numerator, exactly the trick used
for `Z'` in `docs/derivations/plasma_dispersion.md` Section 3) and split the integral:

    𝒞φ(ξ) = (1/√π) [ −2 ∫ e^{−x²} dx − 2ξ ∫ e^{−x²}/(x−ξ) dx ]
           = −2 − 2ξ Z(ξ)
           = −2(1 + ξZ(ξ))
           = Z'(ξ),

using `(1/√π)∫e^{−x²}dx = 1` and `(1/√π)∫e^{−x²}/(x−ξ)dx = Z(ξ)`. This is why
`cauchy_dfdv(::Maxwellian, ξ) = Zprime(ξ)` is exact, not an approximation, and why the
Maxwellian's own value, derivative and Cauchy integral all bypass the generic quadrature
machinery of Section 4.

## 3. `f_lab`, and its three normalizations in dimensional form

The dimensional, unit-normalized 1-D distribution along k̂ is

    f_lab(sp, v) = φ((v−u)/v_th) / v_th,

the one place in the package the shift and the `1/v_th` scaling appear (`conventions.md` 1.5,
1.4). Substituting `x = (v−u)/v_th`, `dv = v_th dx`:

    ∫ f_lab(v) dv = ∫ φ(x) dx = 1,
    ∫ v f_lab(v) dv = ∫ (v_th x + u) φ(x) dx = v_th·0 + u·1 = u,
    ∫ (v−u)² f_lab(v) dv = T/m               (Section 1),

so `f_lab` integrates to 1, has mean `u`, and has second central moment `T/m` — the three
normalizations of Section 1, restored to their dimensional, physical form.

## 4. The Plemelj split of the generic Cauchy-integral fallback

For a shape with no closed-form Cauchy integral, `cauchy_dfdv` falls back to quadrature
(spec D9). On the real axis, the defining Plemelj form is

    𝒞φ(ξ) = P∫ φ'(x)/(x−ξ) dx + iπ φ'(ξ).

Substituting `x = ξ + u` (so that `ξ` appears only in the integrand, never in the quadrature
limits — mandatory under `ForwardDiff`, `conventions.md` 5.12, Pitfall 1) and splitting the
symmetric window `[−L, L]` from the tails:

    P∫_{−∞}^{∞} φ'(ξ+u)/u du
      = P∫_{−L}^{L} φ'(ξ+u)/u du + ∫_{−∞}^{−L} φ'(ξ+u)/u du + ∫_{L}^{∞} φ'(ξ+u)/u du.

Add and subtract `φ'(ξ)/u` inside the window integral:

    P∫_{−L}^{L} φ'(ξ+u)/u du = ∫_{−L}^{L} [φ'(ξ+u) − φ'(ξ)]/u du + φ'(ξ) · P∫_{−L}^{L} du/u.

**The second term is exactly zero.** Substituting `u → −u` in `∫_{ε}^{L} du/u` gives
`∫_{−L}^{−ε} du/u = −∫_{ε}^{L} du/u`, so for every `ε > 0` the two halves of the principal value
cancel exactly; the limit `ε → 0` changes nothing. This is why the fallback's code
(`_cauchy_real`) never computes a term for it — it is not "negligible," it is algebraically
zero for any window width `L`.

What remains is `∫_{−L}^{L} [φ'(ξ+u) − φ'(ξ)]/u du`, whose integrand is smooth even at `u = 0`
(it is a difference quotient converging to `φ''(ξ)`, handled explicitly as a removable
singularity, Pitfall 8), plus the two tail integrals, plus the residue `iπφ'(ξ)`. `L = 6`
(`CAUCHY_WINDOW`) is wide enough that the tails are negligible for the Maxwellian's Gaussian
decay and, per spec D9, for the heavier-tailed shapes arriving at Step 1.SG.1.

## 5. `SuperGaussian`, the second concrete shape

`SuperGaussian` (Step 1.SG.1) is derived in `docs/derivations/supergaussian.md`. Its Cauchy
integral is served, on the real axis and inside a domain in `(ξ, m)`, from a Chebyshev table
built at precompile time (Step 1.SG.2); the generic quadrature fallback above remains its
reference method and the table's own generator. *(The table machinery itself was generalized and
renamed — a general `ChebyshevTable` type, per-table degree caps and a signed ξ domain — at
Phase 4.B Step 4.B.2a, 2026-09-26, spec D13; `SG_TABLE` and `SG_DM_TABLE` are bitwise unchanged
and keep their names.)*

## 6. A shape's derivative in its own parameters

`conventions.md` Section 4's `AbstractVelocityDistribution` entry carries the contract every
shape is built against: **"A shape's derivative with respect to each of its own parameters is
available without numerical quadrature, by closed form or by a table built at include time"**
(the user, 2026-09-17, [OQ-38](../../dev/open_questions.md#oq-38)); **"the guard:
`spectral_density` under a dual in each parameter of every shape in `SHAPE_REGISTRY` allocates
zero and costs at most 5× the same call under a dual in T"** (`conventions.md` 8.7's one
admitted ratio assertion, amended 2026-09-17,
[OQ-39](../../dev/open_questions.md#oq-39)). This section names which route each shape takes and
points at the guard that checks it, rather than restating the clause: it lives in
`test/distributions/test_shape_cost.jl`, gated `"shape cost"`, tests SC1-SC4 of
`dev/phase3C_instructions.md` Section 8.

**`Maxwellian`** meets the contract trivially: `parameters(f) = NamedTuple()`, so there is
nothing to differentiate. The guard's loop is vacuous for it, and SC1 reports that fact rather
than letting it pass silently.

**`SuperGaussian`** meets it by table, built at Step 3.C.4 and derived in full in
`docs/derivations/supergaussian.md` Section 7a. `∂φ/∂m` is served, inside `SG_DM_TABLE`'s domain
(the same `(|x|, m) ∈ [0, 10] × [1.5, 8]` as the Cauchy table `SG_TABLE`), from a second
Chebyshev table built at include time. The quadrature form — `∂φ/∂m` obtained by
`ForwardDiff.derivative` through the generic call path, which runs `_dgamma_da`'s adaptive
quadrature — remains both the table's reference and its generator, and, outside the domain, the
documented slow fallback: exactly the role the Cauchy integral's own quadrature fallback plays
in Section 4 above. Neither quadrature is deleted or sped up; the contract is met by not calling
it on the fast path, not by making it faster.

**`Kappa` and `HermiteExpansion`** (Step 4.2, not yet built) are expected, as an expectation and
not a claim, to meet the contract in closed form: kappa's κ-derivative through
`exp(−κ log(…))`, with `digamma` for `x₀(κ)`'s own derivative; Hermite's coefficients entering
`φ` linearly, so each one's derivative is just the corresponding basis function. **The guard
checks this expectation rather than assuming it**: `test_shape_cost.jl`'s `SC1` fails the
instant either shape is registered in `SHAPE_REGISTRY` without a matching entry in the test's own
`_SC_SHAPES` table, at a representative parameter value inside its own valid range, so a slow or
allocating parameter derivative is a test failure on the day it is written — not a profile
finding three phases later, which is what happened to `SuperGaussian`'s.
