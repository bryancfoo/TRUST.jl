# The principal-value table R(xi; m) = P integral phi'(x;m)/(x-xi) dx on the real axis: the
# Chebyshev-Lobatto machinery, the adaptive per-panel fit, the table object, and the module
# const SG_TABLE, built once at include time (conventions.md 5.6) and read-only afterwards.
# See docs/derivations/supergaussian.md Section 7.

"""
    SG_PANEL_EDGES, SG_XI_MAX, SG_M_LO, SG_M_HI, SG_DEGREE_START, SG_DEGREE_CAP_XI,
    SG_DEGREE_CAP_M, SG_BISECT_MAX, SG_FIT_TOL

The table build's constants (spec D10, D12): the default panel edges before bisection; the
domain `|ξ| ∈ [0, SG_XI_MAX]`, `m ∈ [SG_M_LO, SG_M_HI]`; the starting degree in each direction;
the per-panel ξ-degree cap and the single global m-degree cap; the maximum number of times any
one original panel may be bisected; and `τ`, the trailing-coefficient fit target (D12, revised
2026-09-08 round 1: starts at `1e-9`, the value target with one order of margin, and — driven
**only** by the derivative certification TB7/TB8, never by a single-point test — may be lowered by
one factor of ten, to a floor of `1e-10`, if TB7/TB8 does not pass at `1e-9`. The floor is `1e-10`,
not lower, because the generator itself (`CAUCHY_RTOL = 1e-10`) measures only ~1e-11 relative
against a BigFloat oracle: τ below `1e-10` fits the generator's own quadrature noise rather than
the true interpolation error, which is exactly what drove the degrees up without the accuracy
improving in round 1's `τ = 1e-12` build. Pinned and reported — never asserted at runtime).

These are `Float64` and `Int` constants of the *build*, not quantities that follow an argument's
element type (`conventions.md` 5.1's clarification of 2026-09-08 covers this exactly): they are
never read after `_sg_build_table` runs. Every domain check in `src/` and `test/` reads
`SG_TABLE.xi_max`, `SG_TABLE.m_lo`, `SG_TABLE.m_hi` instead of these constants
([OQ-31](../../dev/open_questions.md#oq-31) item 3).
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

"""
    SuperGaussianTable{L,FT<:AbstractFloat}

The immutable table object (spec D10): `edges` (`np+1` panel edges, ascending, `edges[1] == 0`,
`edges[end] == xi_max`), `nxi` (`np` per-panel ξ-degrees), `offset` (`np` zero-based starts of
each panel's block within a slice), `nm` (the one global m-degree), `coeffs` (the
`L*(nm+1)`-length flat tensor-coefficient array, in slice order: slice entry `i`'s `(nm+1)`
m-coefficients are `coeffs[(i-1)*(nm+1)+1 : i*(nm+1)]`, spec D10's Layout), `xi_converged` (`np`
booleans, one per panel: did the ξ-direction trailing-coefficient check pass at the degree finally
used there?), `m_converged` (one boolean for the whole table: did the m-direction check pass, for
every panel, at `nm`, the degree finally used?), `m_lo`, `m_hi`, `xi_max` (the domain), and `tau`
(the fit target actually used to build it). `L = Σ_p (nxi[p]+1)` is the slice length and the
struct's second type parameter.

**"At the cap" and "failing at the cap" are different** (spec D9, corrected 2026-09-08 round 1):
a panel or the m-direction that reaches its degree cap and passes the trailing-coefficient check
there is a fine panel; `xi_converged[p]` or `m_converged` being `false` means a direction stopped
*while still failing* the check — that, and only that, is the finding
[OQ-31](../../dev/open_questions.md#oq-31) item 2 asks for, and TB3 asserts on these flags rather
than on `nxi[p] < SG_DEGREE_CAP_XI`.

Built once, by `_sg_build_table()`, into the module `const` `SG_TABLE`
(`conventions.md` 5.6, the second named exception to "no global mutable state").

Spec D10; `docs/derivations/supergaussian.md` Section 7.
"""
struct SuperGaussianTable{L,FT<:AbstractFloat}
    edges::Vector{FT}
    nxi::Vector{Int}
    offset::Vector{Int}
    nm::Int
    coeffs::Vector{FT}
    xi_converged::Vector{Bool}
    m_converged::Bool
    m_lo::FT
    m_hi::FT
    xi_max::FT
    tau::FT
end

"""
    _cheb_nodes(n::Integer) -> Vector{Float64}

The `n + 1` Chebyshev-Lobatto (second-kind) nodes of degree `n`, `t_k = cos(π k / n)` for
`k = 0, …, n`, descending from `t_0 = +1` to `t_n = −1`.

Spec D9.
"""
_cheb_nodes(n::Integer) = [cos(pi * k / n) for k in 0:n]

"""
    _cheb_map(x, a, b) -> Real

Map `x ∈ [a, b]` to `t = (2x − a − b)/(b − a) ∈ [−1, 1]`. Generic in `x`'s type, so a dual `x`
gives a dual `t` (spec D9).
"""
_cheb_map(x, a, b) = (2x - a - b) / (b - a)

"""
    _cheb_unmap(t, a, b) -> Real

The inverse of `_cheb_map`: `x = (t(b − a) + a + b)/2`, used only by the table generator to turn
Lobatto nodes in `[−1, 1]` into sample points in `[a, b]`.
"""
_cheb_unmap(t, a, b) = (t * (b - a) + a + b) / 2

"""
    _cheb_coeffs(values::AbstractVector{<:Real}) -> Vector{Float64}

The discrete Chebyshev transform of `values = f.(_cheb_nodes(n))`, `n = length(values) - 1`:

    c_j = (2/n) Σ_{k=0}^{n} '' f_k cos(π j k / n),      j = 0, …, n

with `''` meaning the `k = 0` and `k = n` terms of the sum halved, and then `c_0` and `c_n`
themselves halved in place, so that `p(t) = Σ_j c̃_j T_j(t)` is a plain (unweighted) sum. Both
halvings are needed (spec D9, Pitfall 8; TB1 pins this convention).
"""
function _cheb_coeffs(values::AbstractVector{<:Real})
    n = length(values) - 1
    c = Vector{Float64}(undef, n + 1)
    for j in 0:n
        s = 0.0
        for k in 0:n
            w = (k == 0 || k == n) ? 0.5 : 1.0
            s += w * values[k+1] * cos(pi * j * k / n)
        end
        c[j+1] = (2 / n) * s
    end
    c[1] /= 2
    c[end] /= 2
    return c
end

"""
    _cheb_eval(c::AbstractVector, off::Integer, n::Integer, t) -> Real

Clenshaw's recurrence for the degree-`n` Chebyshev series stored in `c[off+1 : off+n+1]`
(`off` zero-based), evaluated at `t`:

    b_{n+1} = b_{n+2} = 0
    b_j = c̃_j + 2 t b_{j+1} − b_{j+2},   j = n, n−1, …, 1
    p(t) = c̃_0 + t b_1 − b_2

The accumulators take `promote_type(eltype(c), typeof(t))`, never a bare `0.0` (`conventions.md`
5.1's Clenshaw clarification): with `Float64` coefficients and a `ForwardDiff.Dual` `t`, this
returns a `Dual` of the same tag and allocates nothing. `off` and `n` let this read a flat
`Vector` (the table's `coeffs`) and a flat `NTuple` (a shape's cached `slice`) with the same code
(spec D9).
"""
function _cheb_eval(c, off::Integer, n::Integer, t)
    ZT = promote_type(eltype(c), typeof(t))
    b_jp1 = zero(ZT)
    b_jp2 = zero(ZT)
    for j in n:-1:1
        b_j = c[off+j+1] + 2t * b_jp1 - b_jp2
        b_jp2 = b_jp1
        b_jp1 = b_j
    end
    return c[off+1] + t * b_jp1 - b_jp2
end

"""
    _sg_principal_value(m, xi) -> Real

One reference value `R(ξ; m)`, the real part of the reference Cauchy integral evaluated on the
*uncached* form `_sg_uncached(m)` — no slice, so the table's generator never depends on the
table it is building (spec D11, Pitfall 2). `_sg_cauchy_reference` on a real `ξ` is
`_cauchy_real`, which reads only `dfdx` and `_d2fdx2`, at Phase 0's `CAUCHY_RTOL`/`CAUCHY_ATOL`.

Spec D11; `docs/derivations/supergaussian.md` Section 7.
"""
_sg_principal_value(m, xi) = real(_sg_cauchy_reference(_sg_uncached(m), xi))

"""
    _tensor_coeffs(F::AbstractMatrix{<:Real}) -> Matrix{Float64}

The per-panel tensor-product Chebyshev fit of spec D9: given `F[k+1, l+1] = R(ξ_k, m_l)` at the
Lobatto nodes in ξ (rows, degree `nxi = size(F,1)-1`) and in m (columns, degree
`nm = size(F,2)-1`), transform first down each column (the ξ direction) and then along each row
(the m direction), returning `C[j+1, i+1]`, `j = 0…nxi`, `i = 0…nm`.
"""
function _tensor_coeffs(F::AbstractMatrix{<:Real})
    nxi1, nm1 = size(F)
    G = Matrix{Float64}(undef, nxi1, nm1)
    for l in 1:nm1
        G[:, l] = _cheb_coeffs(view(F, :, l))
    end
    C = Matrix{Float64}(undef, nxi1, nm1)
    for j in 1:nxi1
        C[j, :] = _cheb_coeffs(view(G, j, :))
    end
    return C
end

"""
    _trailing_ok(C::AbstractMatrix{<:Real}, dim::Integer, tau) -> Bool

The trailing-coefficient convergence test of spec D9 along dimension `dim` (`1` for ξ, the rows
of `C`; `2` for m, the columns): with `a_k = max` over the other dimension of `|C|`, requires the
**last two** trailing values (never one alone — one can vanish by symmetry, Pitfall 7) to be at
most `tau` times the largest.
"""
function _trailing_ok(C::AbstractMatrix{<:Real}, dim::Integer, tau)
    n1 = size(C, dim)
    a = [maximum(abs, selectdim(C, dim, k)) for k in 1:n1]
    return max(a[end-1], a[end]) <= tau * maximum(a)
end

"""
    _sg_sample_grid!(memo, a, b, nxi, nm, tau) -> Matrix{Float64}

Samples `R(ξ, m)` on the `(nxi+1) × (nm+1)` Lobatto grid of panel `[a, b] × [m_lo, m_hi]`,
memoized by the exact `(ξ, m)` floating-point coordinate: a Lobatto node shared between degree
`n` and degree `2n` (`t_k` at degree `n` is `t_{2k}` at degree `2n`, spec D9) maps to the
identical `Float64` value under `_cheb_unmap` (division by exactly-representable integers, so
IEEE division of the mathematically equal ratio rounds identically), so doubling a degree only
pays for the new nodes. `memo` is build-time scratch, thrown away with the panel
(`conventions.md` Section 7's "no cache beyond the table const").
"""
function _sg_sample_grid!(memo::Dict{Tuple{Float64,Float64},Float64}, a, b, nxi, nm)
    xi_vals = [_cheb_unmap(t, a, b) for t in _cheb_nodes(nxi)]
    m_vals = [_cheb_unmap(t, SG_M_LO, SG_M_HI) for t in _cheb_nodes(nm)]
    F = Matrix{Float64}(undef, nxi + 1, nm + 1)
    for (kk, xi) in enumerate(xi_vals), (ll, mm) in enumerate(m_vals)
        F[kk, ll] = get!(() -> _sg_principal_value(mm, xi), memo, (xi, mm))
    end
    return F
end

"""
    _sg_adapt_panel!(final::Vector, a, b, depth::Integer, tau)

Fits one ξ-panel `[a, b]` adaptively in both directions (spec D9): starting at
`SG_DEGREE_START` in ξ and in m, doubling whichever direction's trailing-coefficient test
(`_trailing_ok`) fails, capped at `SG_DEGREE_CAP_XI` and `SG_DEGREE_CAP_M`. A ξ degree that
would need to exceed its cap is bisected instead, to a depth of at most `SG_BISECT_MAX` splits of
the *original* panel (Pitfall 19; the build must terminate); an m degree that would need to
exceed its cap is simply capped, accepted, and left for TB3 to report as a finding
([OQ-31](../../dev/open_questions.md#oq-31) item 2) — never bisected, since there is one global
m-degree for the whole table. Pushes `(a, b, nxi, nm, C, memo)` NamedTuples for the finished
panel(s), in ascending ξ order, onto `final`.
"""
function _sg_adapt_panel!(final::Vector, a::Float64, b::Float64, depth::Integer, tau)
    memo = Dict{Tuple{Float64,Float64},Float64}()
    nxi = SG_DEGREE_START
    nm = SG_DEGREE_START
    local C
    while true
        F = _sg_sample_grid!(memo, a, b, nxi, nm)
        C = _tensor_coeffs(F)
        xi_ok = _trailing_ok(C, 1, tau)
        m_ok = _trailing_ok(C, 2, tau)
        if xi_ok && m_ok
            push!(final, (a = a, b = b, nxi = nxi, nm = nm, C = C, memo = memo))
            return
        end
        if !xi_ok && nxi >= SG_DEGREE_CAP_XI && depth < SG_BISECT_MAX
            mid = (a + b) / 2
            _sg_adapt_panel!(final, a, mid, depth + 1, tau)
            _sg_adapt_panel!(final, mid, b, depth + 1, tau)
            return
        end
        next_nxi = (!xi_ok && nxi < SG_DEGREE_CAP_XI) ? min(2nxi, SG_DEGREE_CAP_XI) : nxi
        next_nm = (!m_ok && nm < SG_DEGREE_CAP_M) ? min(2nm, SG_DEGREE_CAP_M) : nm
        if next_nxi == nxi && next_nm == nm
            push!(final, (a = a, b = b, nxi = nxi, nm = nm, C = C, memo = memo))
            return
        end
        nxi, nm = next_nxi, next_nm
    end
end

"""
    _sg_build_table() -> SuperGaussianTable

The whole build (spec D9–D12), run once at include time: **Pass 1** adapts every panel of
`SG_PANEL_EDGES` independently in both directions (`_sg_adapt_panel!`, with bisection), yielding
the final panel edges, each panel's own `nxi` and its own *required* `nm`. **Pass 2** sets
`nm = max` over every panel's required `nm` (one global m-degree,
[OQ-31](../../dev/open_questions.md#oq-31) item 2) and rebuilds any panel whose own `nm` fell
short, reusing that panel's memo so only the new m-nodes cost a reference evaluation, then
re-checks `_trailing_ok` in *both* directions on each panel's final tensor — "the degree finally
used" (spec D9) is `nxi[p]` for the ξ direction and the shared `nm` for the m direction, and
re-checking after Pass 2's rebuild is what makes that literal: a panel's own Pass-1 `nm` can be
smaller than the global `nm`, and the extra m-columns Pass 2 adds could in principle change either
direction's row/column maxima. `xi_converged[p]` is that per-panel ξ recheck; `m_converged` is the
AND of the m recheck over every panel, since one global `nm` either serves the whole table or it
does not. Assembles the flat `coeffs` array in slice order (spec D10's Layout) and returns the
`SuperGaussianTable`.

Deterministic, machine-independent, silent, `Float64` throughout, never `BigFloat`, and bounded
by construction (`SG_BISECT_MAX`, `SG_DEGREE_CAP_XI`, `SG_DEGREE_CAP_M`) — the build runs at
every precompile and must terminate there (Pitfall 4, Pitfall 19).
"""
function _sg_build_table()
    tau = SG_FIT_TOL
    edges0 = collect(Float64, SG_PANEL_EDGES)
    panels = NamedTuple[]
    for p in 1:length(edges0)-1
        _sg_adapt_panel!(panels, edges0[p], edges0[p+1], 0, tau)
    end
    nm_global = maximum(p.nm for p in panels)
    final = NamedTuple[]
    for p in panels
        C = p.nm < nm_global ? _tensor_coeffs(_sg_sample_grid!(p.memo, p.a, p.b, p.nxi, nm_global)) : p.C
        xi_converged = _trailing_ok(C, 1, tau)
        m_converged = _trailing_ok(C, 2, tau)
        push!(final, (a = p.a, b = p.b, nxi = p.nxi, C = C,
                      xi_converged = xi_converged, m_converged = m_converged))
    end
    edges = Float64[final[1].a]
    for p in final
        push!(edges, p.b)
    end
    nxi_vec = Int[p.nxi for p in final]
    xi_converged_vec = Bool[p.xi_converged for p in final]
    m_converged = all(p.m_converged for p in final)
    offset = Int[]
    off = 0
    for n in nxi_vec
        push!(offset, off)
        off += n + 1
    end
    L = off
    coeffs = Vector{Float64}(undef, L * (nm_global + 1))
    for (pi, p) in enumerate(final)
        for j in 0:p.nxi
            slice_idx = offset[pi] + j
            coeffs[slice_idx*(nm_global+1)+1:(slice_idx+1)*(nm_global+1)] .= view(p.C, j + 1, :)
        end
    end
    return SuperGaussianTable{L,Float64}(edges, nxi_vec, offset, nm_global, coeffs,
                                          xi_converged_vec, m_converged,
                                          SG_M_LO, SG_M_HI, SG_XI_MAX, tau)
end

"""
    _sg_panel_index(table, x) -> Int

The panel owning a nonnegative *primal* `x` (spec D15): left-closed, right-open, with the last
panel closed, so that `x = 0` gives panel 1 and `x = table.xi_max` gives the last panel. Adjacent
panels share their endpoint node, so both give the same value there to round-off (TB9).
"""
_sg_panel_index(table, x) = clamp(searchsortedlast(table.edges, x), 1, length(table.edges) - 1)

"""
    _sg_slice(table::SuperGaussianTable{L}, m) where {L} -> NTuple{L}

The table's ξ-coefficients at this `m`, one panel after another, flattened: for every slice
entry `i = 1, …, L`, the one-dimensional Clenshaw evaluation in `m` of `coeffs`' `i`-th
`(nm+1)`-block. `Val(L)`, never a plain `L`, so the result is `NTuple{L, FT}`, inferable and
heap-free — measured at `4.47 µs`, zero allocation, `L = 102` (spec D3, D13). With a dual `m`
this returns an `NTuple` of duals of the same tag, which is how the m-derivative flows through a
cached shape.

Spec D13.
"""
function _sg_slice(table::SuperGaussianTable{L}, m) where {L}
    s = _cheb_map(m, table.m_lo, table.m_hi)
    return ntuple(i -> _cheb_eval(table.coeffs, (i - 1) * (table.nm + 1), table.nm, s), Val(L))
end

"""
    _sg_table_eval(table, slice, x) -> Real

The per-bin evaluator: `R` at `|ξ| = x ≥ 0` (possibly a `Dual`), read out of a shape's own
`slice` by one panel's one-dimensional Clenshaw in ξ. The panel index is taken at `x`'s primal
value (`_primal_value`, spec 5.12 — an index is never built from a dual); the Clenshaw evaluation
itself is generic in `x`'s type. Measured at `52.7 ns`, zero allocation, `n_ξ = 16` (spec D13).

Spec D13.
"""
function _sg_table_eval(table, slice, x)
    p = _sg_panel_index(table, _primal_value(x))
    a = table.edges[p]
    b = table.edges[p+1]
    return _cheb_eval(slice, table.offset[p], table.nxi[p], _cheb_map(x, a, b))
end

"""
    SG_TABLE::SuperGaussianTable

The module-level table `const` (`conventions.md` 5.6), built once by `_sg_build_table()` at
include time and read-only afterwards. `SuperGaussian(m)`'s public constructor computes its
`slice` from it; `_sg_uncached(m)`, used only by the generator above, never touches it.
"""
const SG_TABLE = _sg_build_table()
