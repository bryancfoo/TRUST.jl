# The `SkewNormal` shape

Phase 4.B, Step 4.B.4 (spec D35-D38, D42, D47). One file per non-Maxwellian shape
(`conventions.md` 9.1), `generalized_normal.md`'s form.

## 1. What the shape is, and what it is not

`SkewNormal` is Azzalini's skew-normal distribution, located and scaled to mean 0 and variance ½:

    φ(x) = (2/ω) ϕ(z) Φ(αz),   z = (x − x_c)/ω,

with `ϕ` the standard normal density and `Φ` the standard normal CDF. **It is not a Hermite or
Gram–Charlier skew**: those add a signed correction term to a Gaussian and can go negative;
`φ` above is a **product** of two positive factors (`ϕ(z) ≥ 0`, `Φ(αz) ∈ (0, 1)`) and is
**positive everywhere**, for every finite `α`. `α = 0` collapses `Φ(0·z) = 1/2` for every `x`, and
`φ` reduces to the Maxwellian exactly (Section 2). `α > 0` **skews the tail toward positive `x`**,
i.e. along `+k̂`: the `Φ(αz)` factor is larger where `z` and `α` have the same sign, pulling mass
toward that side (SK7's reflection identity makes this precise: `φ(x; −α) = φ(−x; α)`).

**Why this shape, for this package.** A heat flux or a return current leaves an asymmetric
signature on an electron population's velocity distribution along **k̂** — more mass moving one
way than the other — which neither `Maxwellian`, `SuperGaussian`, `Kappa` nor `GeneralizedNormal`
can represent (all four are even in `x − u`, the asymmetry already absorbed into the flow `u`
itself). `SkewNormal` is the phase's one skewed shape (`HermiteExpansion`, OQ-42 item 3, is not
built for now; D30).

## 2. The constants

With the standard skew-normal `Z` having density `2ϕ(z)Φ(αz)`, mean `√(2/π) δ` and variance
`1 − 2δ²/π` where `δ = α/√(1+α²)` (Azzalini's own parametrization), the affine image
`X = x_c + ωZ` has mean `x_c + ωμ_δ` and variance `ω²v_δ`. Setting the mean to `0` and the
variance to `½` (`conventions.md` 1.5) gives

    δ = α/√(1+α²),   μ_δ = δ√(2/π),   v_δ = 1 − 2δ²/π,
    ω = 1/√(2v_δ),   x_c = −ωμ_δ                                      (`_sn_constants`)

**`α = 0` gives `δ = 0`, `μ_δ = 0`, `v_δ = 1`, `ω = 1/√2`, `x_c = 0`** (computed as `−ω·0.0 =
−0.0`, so `SkewNormal(0.0).xc` is `-0.0`, not `0.0`; SK2 tests it with `iszero`, never `===`), and
`φ(x) = (2·√2)·[exp(−x²)/√(2π)]·(1/2) = exp(−x²)/√π`, the Maxwellian exactly (SK4).

**Skewness.** `γ₁ = ((4−π)/2) μ_δ³/v_δ^{3/2}`, from the standard skew-normal's own third
cumulant scaled to this shape's variance: `0.1369` at `α = 1`, `0.6670` at `α = 3` (the table's
edge), `0.8510` at `α = 5`, and `γ₁ → ((4−π)/2)·(2/π)^{3/2} ≈ 0.9953` as `α → ∞` — Azzalini's shape
can never be skewed past this limit, however large `α` is (SK6 checks `skewness` against this
closed form).

## 3. The moments

Differentiating `φ(x) = (2/ω)ϕ(z)Φ(αz)` in `x` gives

    φ'(x) = (2α/ω²) ϕ(z)ϕ(αz) − ((x − x_c)/ω²) φ(x)

(the first term from `d/dx Φ(αz) = (α/ω)ϕ(αz)`, the second from `d/dx ϕ(z) = −(z/ω)ϕ(z)`, both
chain-ruled through `z = (x−x_c)/ω`). Multiplying by `xⁿ` and integrating by parts,
`∫ xⁿ φ' dx = −n M_{n−1}` (the boundary term vanishes: `φ' xⁿ → 0` as `|x| → ∞`), so

    −n M_{n−1} = (2α/ω²) ∫ xⁿ ϕ(z)ϕ(αz) dx − (1/ω²) ∫ xⁿ (x−x_c) φ(x) dx.

Writing `x = x_c + ωz` and expanding `ϕ(z)ϕ(αz) = τ√(2/π)/ω · [normal density of mean `x_c`,
sd `τ`]` with `τ = ω/√(1+α²)` (the product of two centred normal densities is itself a scaled
normal density, a standard identity), and expanding `∫xⁿ(x−x_c)φ dx = M_{n+1} − x_c M_n`, both
sides reduce to raw-moment recursions in `x_c`-centred form, giving the three-term recursion of
Section 10.3:

    M_0 = 1,   M_1 = 0 (exactly, by the mean normalization),
    M_{k+1} = x_c M_k + k ω² M_{k−1} + g h_k        (k ≥ 1),
    h_0 = 1,   h_1 = x_c,   h_{k+1} = x_c h_k + k τ² h_{k−1},
    s = 1/√(1+α²),   τ = ωs,   g = 2αωs/√(2π),

`h_k` being the raw moments of a normal of mean `x_c` and standard deviation `τ` (the family
`ϕ(z)ϕ(αz)` reduces to). **`M₁` is set to exactly zero** in the code, not read off the recursion's
own first step (which gives it to rounding, about `1e-16`, since the mean normalization is
enforced by construction, not by the recursion): a consumer that needs a true zero (a Laurent
series' odd-term skip, `conventions.md` Section 4) must see one exactly, though `SkewNormal` itself
has no such consumer (Section 5). `M₂ = ½` follows from the recursion at `k = 1` (the variance
normalization, checked bitwise nowhere but verified to `2.1e-13` against quadrature, SK6); `M₃`
gives the skewness of Section 2. All moments are finite for every finite `α`, unlike `Kappa`'s.

**Checked** (SK6) against `quadgk` for `n = 0` to `8` at six `α` values (worst measured
**2.13e-13**, this session, against `rtol = 1e-11, atol = 1e-12`).

## 4. The α-partial of φ

By the chain rule through every quantity `_sn_constants` derives, with `δ' = (1+α²)^{−3/2}`,
`μ' = √(2/π) δ'`, `v' = −4δδ'/π`, `ω' = −ω v'/(2v_δ)`, `x_c' = −(ω'μ_δ + ωμ')`,
`z' = −(x_c' + zω')/ω`:

    ∂φ/∂α = φ · [ −ω'/ω − z z' + (ϕ(αz)/Φ(αz)) · (z + α z') ]

(the three terms are, in order, the `1/ω` prefactor's own α-dependence, `ϕ(z)`'s through `z(α)`,
and `Φ(αz)`'s through both `α` itself and `z(α)`). **No hand-written rule is needed in the code**:
`φ` is built from `exp` and `erfc` alone (`_std_normal_pdf`, `_std_normal_cdf`), and ForwardDiff
differentiates both through DiffRules' closed-form rules, so a dual `α` through
`SkewNormal(alpha)(x)` differentiates the whole shape correctly and allocation-free with no
second table, exactly `Kappa`'s and `GeneralizedNormal`'s reason (`conventions.md` Section 4's
derivative clause, OQ-38).

SK16 checks the closed form above against `ForwardDiff.derivative(aa -> _sn_uncached(aa)(x),
alpha)` on a grid including `x = 0` (worst measured **1.6e-16**, this session, well inside
`rtol = 1e-11`), and again checks `spectral_density`'s α-derivative against a central difference
(worst measured **2.1e-24** against a floor of about **2.2e-22**, this session), for six `α`
including one that runs the fallback.

## 5. The Cauchy integral

**The Plemelj form** (`conventions.md` 1.5): `𝒞φ(ξ) = R(ξ) + iπφ'(ξ)` on the real axis, `R(ξ) = P∫
φ'(x)/(x−ξ) dx`; a table holds `R` alone, and the imaginary part is exact and never tabulated.

**The reflection.** `φ(x; −α) = φ(−x; α)` (Section 1), so `φ'(x; −α) = −φ'(−x; α)` and, changing
variables `x → −x` in the principal-value integral,

    R(ξ; −α) = R(−ξ; α),       Im 𝒞φ_{−α}(ξ) = −Im 𝒞φ_α(−ξ)   (SK7).

**Why R is not even in ξ, and why the table is signed.** Every other shape this package builds is
even in `x` (a Maxwellian, super-Gaussian or kappa profile is symmetric about its mean), so `φ'` is
odd and `R` is automatically even: a table over `ξ ≥ 0` read at `|ξ|` is correct **for those
shapes**. `SkewNormal`'s `φ` is asymmetric for `α ≠ 0`, so `φ'` has no such parity and neither does
`R`: `R(0.4; 1.2) − R(−0.4; 1.2) = 0.4785` (Section 10.3, Level 2's measurement). **`SN_TABLE` is
built over the signed domain `ξ ∈ [−20, 20]`** (`ChebyshevTable`'s `xi_lo < 0` contract, D13), and
`cauchy_dfdv` reads it at the signed `ξ`, never `abs(ξ)` — the mistake `abs(ξ)` would make is
invisible at `α = 0` (SK4, SK11's reduction test), which is exactly why SK10 checks `+ξ` and `−ξ`
separately against an independent oracle, SK15(f) checks `R(ξ) ≠ R(−ξ)` directly, and SK25 checks
that the table itself, not just the shape, differs from its own value at `|ξ|`.

**Why there is no Laurent series branch.** `_cauchy_series` (`conventions.md` Section 4,
`distribution.jl`) was written for a symmetric shape: it skips only an *exactly* zero moment and
declares convergence at the first term below `rtol · |sum|`. A skewed shape's odd moments are
small but not zero, so an odd term can look negligible (`M₁ ~ 1e-17` after the code's exact-zero
override, or genuinely tiny for a mildly skewed shape) while the next even term is not yet small
enough — the series can report `converged = true` while wrong. Level 2 measured this directly on a
scratch `SkewNormal`: at `α ∈ {0.01, 0.1, 3.0}` and `ξ ∈ {±10.5, ±15}` it returned **converged, 0.7 %
to 1.5 % wrong**. A wrong answer reported as converged is the worst failure a branch can have, so
`SkewNormal` has **no series branch**, and `_cauchy_series` itself is not touched — it stays
`SuperGaussian`'s and `GeneralizedNormal`'s, bitwise. SK15(b) checks that no such branch exists in
`cauchy_dfdv`'s two-branch order.

## 6. The table, and beyond it

**The coordinate is `α` itself, domain `α ∈ [−3, 3] × ξ ∈ [−20, 20]`, signed.**

- **Why `α ∈ [−3, 3]`**: this covers skewness `|γ₁| ≤ 0.6670` of the shape's limit `0.9953`
  (Section 2); a wider fast range would need a different coordinate (the branch points of
  `δ(α) = α/√(1+α²)` are at `α = ±i`, so no rational reparametrization removes the limit the way
  `s = 1/κ` removes `Kappa`'s κ → ∞) or split tables, and is not built now (D30 item 4).
- **Why `|ξ| ≤ 20`, not 10**: `Kappa`'s own reason — an electron shape at low `T_e` in an EPW
  window reaches `|ξ| ≈ 8`–10, and a heavy-tailed or skewed `R` is still far from its asymptote at
  `ξ = 10`.
- **The degree cap in `α` is 64, for this table only** (through chunk 4's per-table caps, D13):
  Level 2 measured `48` failing to converge (`5.6e-9`) and `64` converging (`2.2e-11`); no other
  table's cap moves.
- **The generator at `α = 0`** is this shape's own quadrature (`_sn_uncached(0.0)` is the
  Maxwellian to rounding), **not `Zprime`** — so that `Zprime` stays an independent oracle for
  SK11 at that edge, the same reason `KAPPA_TABLE`'s and `GN_TABLE`'s generators avoid their own
  Maxwellian-edge oracle.

**`τ = 1e-9`, the value target with one order of margin (rule 8.9), and it is not lowered**: a
table direction that does not converge is a stop, `Kappa`'s and `GeneralizedNormal`'s rule.

**The build, as measured (chunk 7, 2026-09-27, this machine, Julia 1.12.7)**:

| Quantity | Value |
|---|---|
| Panels | 10 |
| ξ-degrees (`nxi`) | 16, 32, 32, 32, 32, 32, 32, 32, 32, 16 |
| `np` (α-degree) | 64 |
| Slice length `L` | 298 |
| Bisections | 0 |
| Every `xi_converged[p]` | `true` |
| `p_converged` | `true` |
| Reference evaluations | 19 370 |
| Warm build time | 0.77 s (a second call; the machine's normal jitter, D23 item 7) |
| `Base.summarysize(SN_TABLE)` | 155 778 bytes |

identical, panel for panel, to Level 2's build quoted in spec D36 (10 panels,
`16/32/32/32/32/32/32/32/32/16`, `np = 64`, `L = 298`, 19 370 evaluations, ≈0.69 s warm).

**Certification, chunk 7's measured worst errors**:

| Row | Against | Worst measured | Target |
|---|---|---|---|
| SK9 | the generator, 60 seeded points over the whole signed domain + every edge × `α ∈ {−3,0,3}` | 4.12e-12 | `rtol=atol=1e-8` |
| SK10(a) | a 64-bit BigFloat principal-value quadrature, `+ξ` and `−ξ` separately | 2.52e-11 | `atol=1e-9` |
| SK10(b) | the same, on the `\|α\| > 3` fallback | 4.44e-16 | `atol=1e-9` |
| SK11 | `Re Zprime` at the Maxwellian edge `α=0` | 5.99e-14 | `atol=1e-9` |
| SK13 | central differences of the generator, `∂R/∂α` | 0.088 % of its own floor | `rtol=1e-6`, floor per 8.4 |

(Level 2's own build measured `1.7e-11`, `2.5e-11`, `6.0e-14` and `1.2e-10` respectively against
SK9, SK10(a), SK11 and SK13's counterparts; the figures agree to the precision either measurement
distinguishes, the same deterministic build.)

**Beyond the table** (`\|α\| > 3` or `\|ξ\| > 20`): the quadrature fallback `_sn_cauchy_reference`,
exactly `_kappa_cauchy_reference`'s and `_gn_cauchy_reference`'s structure — direct quadrature
(`_cauchy_upper`) for `Im ξ > 0`, the Plemelj principal-value quadrature (`_cauchy_real`) for real
`ξ`, and an error for `Im ξ < 0` (the residue continuation needs a complex-valued `φ'`, which
`dfdx(::SkewNormal, ::Complex)` does not provide). Correct, and **25 to 100 µs a call** (D30 item
4; measured `≤ 2.2e-15` against BigFloat at `α ∈ {4, −6, 10, −20}`) — a documented cost for an ion
species carrying `SkewNormal` across a wide window, where `ξ_i` can exceed 20, or for a fit whose
bound on `α` sits outside `±3`.

**Chunk 7's own throughput measurements** (Section 12 row 23, ns/call, single-threaded, this
machine, one measurement each, not a median): the table (`SkewNormal(0.7)` at `ξ = 1.3`) **187 ns**,
at `ξ = −1.3` **260 ns** (a different panel, ξ-degree the same, 32); the fallback on `α` outside the
table (`SkewNormal(4.0)` at `ξ = 1.3`) **41.7 µs**; the fallback on `ξ` outside the table
(`SkewNormal(0.7)` at `ξ = 25.0`) **35.9 µs**. Both fallback figures sit inside D30 item 4's range
(25 to 100 µs).
