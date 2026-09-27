# The principal-value table R(xi; m) = P integral phi'(x;m)/(x-xi) dx on the real axis: the
# super-Gaussian's own build constants, its sampler and the module consts SG_TABLE and
# SG_DM_TABLE, built once at include time (conventions.md 5.6) and read-only afterwards. The
# general (xi, p) Chebyshev table machinery itself (the struct, the adaptive fit, the builder,
# the slice and evaluator) moved to chebyshev_table.jl and was generalized and renamed at Phase
# 4.B Step 4.B.2a (spec D13): SuperGaussianTable is now ChebyshevTable, _sg_build_table is now
# _build_cheb_table, and so on, with no numerical change to SG_TABLE or SG_DM_TABLE (D14).
# See docs/derivations/supergaussian.md Section 7.

"""
    SG_PANEL_EDGES, SG_XI_MAX, SG_M_LO, SG_M_HI, SG_DEGREE_START, SG_DEGREE_CAP_XI,
    SG_DEGREE_CAP_M, SG_BISECT_MAX, SG_FIT_TOL, SG_DM_FIT_TOL

The table build's constants (spec D10, D12; `SG_DM_FIT_TOL` added at Phase 3.C spec D4): the
default panel edges before bisection; the domain `|ξ| ∈ [0, SG_XI_MAX]`, `m ∈ [SG_M_LO, SG_M_HI]`,
shared by both tables built from them; the starting degree in each direction; the per-panel
ξ-degree cap and the single global m-degree cap; the maximum number of times any one original
panel may be bisected; `τ` for the Cauchy table (`SG_FIT_TOL`), the trailing-coefficient fit
target (D12, revised 2026-09-08 round 1: starts at `1e-9`, the value target with one order of
margin, and — driven **only** by the derivative certification TB7/TB8, never by a single-point
test — may be lowered by one factor of ten, to a floor of `1e-10`, if TB7/TB8 does not pass at
`1e-9`. The floor is `1e-10`, not lower, because the generator itself (`CAUCHY_RTOL = 1e-10`)
measures only ~1e-11 relative against a BigFloat oracle: τ below `1e-10` fits the generator's own
quadrature noise rather than the true interpolation error, which is exactly what drove the degrees
up without the accuracy improving in round 1's `τ = 1e-12` build. Pinned and reported — never
asserted at runtime); and `τ` for the m-partial table (`SG_DM_FIT_TOL`, D4), starting at `1e-10`
(the 1e-9 accuracy target with one order of margin) and lowerable **once**, to a floor of `1e-11`,
driven only by DM3 or DM5 failing at `1e-9`, never by a single point — the floor sits one order
below the Cauchy table's because the m-partial generator `_dgamma_da` itself runs at
`rtol = 1e-12`, one order tighter than `_cauchy_real`'s `CAUCHY_RTOL = 1e-10`.

These are `Float64` and `Int` constants of the *build*, not quantities that follow an argument's
element type (`conventions.md` 5.1's clarification of 2026-09-08 covers this exactly): they are
never read after `_build_cheb_table` runs. Every domain check in `src/` and `test/` reads
`SG_TABLE.xi_max`, `SG_TABLE.p_lo`, `SG_TABLE.p_hi` (or `SG_DM_TABLE`'s, which are the same
values) instead of these constants ([OQ-31](../../dev/open_questions.md#oq-31) item 3; the
fields were `m_lo`/`m_hi` before Phase 4.B spec D13's rename).

`SG_DEGREE_CAP_XI`, `SG_DEGREE_CAP_M` and `SG_BISECT_MAX` are passed to `_build_cheb_table` as
the required keywords `cap_xi`, `cap_p` and `bisect_max` (Phase 4.B spec D13): the general
builder takes a table's caps as arguments now, but `SG_TABLE` and `SG_DM_TABLE` still build from
these same module constants, unchanged.
"""
const SG_PANEL_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
const SG_XI_MAX = 10.0
const SG_M_LO = 1.5
const SG_M_HI = 8.0
const SG_DEGREE_START = 8
const SG_DEGREE_CAP_XI = 64
const SG_DEGREE_CAP_M = 48
const SG_BISECT_MAX = 4
const SG_FIT_TOL = 1e-9
const SG_DM_FIT_TOL = 1e-10

"""
    _sg_principal_value(m, xi) -> Real

One reference value `R(ξ; m)`, the real part of the reference Cauchy integral evaluated on the
*uncached* form `_sg_uncached(m)` — no slice, so the table's generator never depends on the
table it is building (spec D11, Pitfall 2). `_sg_cauchy_reference` on a real `ξ` is
`_cauchy_real`, which reads only `dfdx` and `_d2fdx2`, at Phase 0's `CAUCHY_RTOL`/`CAUCHY_ATOL`.
`SG_TABLE`'s sampler, in the general `sampler(p, x)` order `_build_cheb_table` expects (Phase
4.B spec D13).

Spec D11; `docs/derivations/supergaussian.md` Section 7.
"""
_sg_principal_value(m, xi) = real(_sg_cauchy_reference(_sg_uncached(m), xi))

"""
    SG_TABLE::ChebyshevTable
    SG_DM_TABLE::ChebyshevTable

The module-level table `const`s (`conventions.md` 5.6), each built once by `_build_cheb_table` at
include time and read-only afterwards. `SG_TABLE` tabulates `R(ξ; m)`, the Cauchy-integral
principal value, from `_sg_principal_value`; `SG_DM_TABLE` tabulates `∂φ/∂m` on the same
`(|x|, m)` domain and panels, from `_sg_dphi_dm_reference` (Phase 3.C spec D4, D9).
`SuperGaussian(m)`'s public constructor computes its `slice` from `SG_TABLE`, and, when `m`
carries a dual, its `dm_slice` from `SG_DM_TABLE`; `_sg_uncached(m)`, used only by the two
generators above, never touches either table. **`SG_TABLE` is bound first**: `SuperGaussian
(m::Real)`'s promotion reads `eltype(SG_TABLE.coeffs)`, so `SG_DM_TABLE`'s declaration must not
be swapped ahead of it (Phase 3.C spec D9), and `chebyshev_table.jl`, which defines
`_build_cheb_table` and `ChebyshevTable`, must be included before this file (Phase 4.B spec
D13: after `distributions/maxwellian.jl`, before `distributions/supergaussian.jl`, ahead of this
file's own `supergaussian_table.jl` position).

**Bitwise unchanged at Phase 4.B Step 4.B.2a** (spec D14): the general builder's caps are passed
here as `SG_DEGREE_START`, `SG_DEGREE_CAP_XI`, `SG_DEGREE_CAP_M` and `SG_BISECT_MAX`, the same
four module constants as before the rename, and `SG_XI_MAX` (`SG_PANEL_EDGES[end] == 10.0`) is
recovered by the builder from `edges[end]` rather than a removed sixth positional argument, to
the same value, bit for bit.
"""
const SG_TABLE = _build_cheb_table(_sg_principal_value, SG_PANEL_EDGES, SG_FIT_TOL,
                                    SG_M_LO, SG_M_HI;
                                    degree_start=SG_DEGREE_START, cap_xi=SG_DEGREE_CAP_XI,
                                    cap_p=SG_DEGREE_CAP_M, bisect_max=SG_BISECT_MAX)
const SG_DM_TABLE = _build_cheb_table(_sg_dphi_dm_reference, SG_PANEL_EDGES, SG_DM_FIT_TOL,
                                       SG_M_LO, SG_M_HI;
                                       degree_start=SG_DEGREE_START, cap_xi=SG_DEGREE_CAP_XI,
                                       cap_p=SG_DEGREE_CAP_M, bisect_max=SG_BISECT_MAX)
