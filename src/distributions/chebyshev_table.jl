# The general (xi, p) Chebyshev table machinery: the Chebyshev-Lobatto nodes, the discrete
# transform and Clenshaw evaluation, the per-panel adaptive fit, the table object and its
# builder. A table serves the Cauchy integral (or any other (xi, p) quantity) of a table-backed
# shape from a Chebyshev fit built once at include time (conventions.md 5.6) and read-only
# afterwards. Moved here from supergaussian_table.jl and generalized at Phase 4.B Step 4.B.2a
# (spec D13): a general name, per-table degree caps and a signed xi domain. `SuperGaussian`'s own
# table build stays in supergaussian_table.jl; see docs/derivations/supergaussian.md Section 7.

"""
    _cheb_nodes(n::Integer) -> Vector{Float64}

The `n + 1` Chebyshev-Lobatto (second-kind) nodes of degree `n`, `t_k = cos(π k / n)` for
`k = 0, …, n`, descending from `t_0 = +1` to `t_n = −1`.

Spec D9; moved and generalized at Phase 4.B spec D13.
"""
_cheb_nodes(n::Integer) = [cos(pi * k / n) for k in 0:n]

"""
    _cheb_map(x, a, b) -> Real

Map `x ∈ [a, b]` to `t = (2x − a − b)/(b − a) ∈ [−1, 1]`. Generic in `x`'s type, so a dual `x`
gives a dual `t` (spec D9). `[a, b]` need not contain zero: a signed panel `[a, b]` with `a < 0`
maps exactly the same way (Phase 4.B spec D13).
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
    _tensor_coeffs(F::AbstractMatrix{<:Real}) -> Matrix{Float64}

The per-panel tensor-product Chebyshev fit of spec D9: given `F[k+1, l+1] = R(ξ_k, p_l)` at the
Lobatto nodes in ξ (rows, degree `nxi = size(F,1)-1`) and in `p` (columns, degree
`np = size(F,2)-1`), transform first down each column (the ξ direction) and then along each row
(the `p` direction), returning `C[j+1, i+1]`, `j = 0…nxi`, `i = 0…np`.
"""
function _tensor_coeffs(F::AbstractMatrix{<:Real})
    nxi1, np1 = size(F)
    G = Matrix{Float64}(undef, nxi1, np1)
    for l in 1:np1
        G[:, l] = _cheb_coeffs(view(F, :, l))
    end
    C = Matrix{Float64}(undef, nxi1, np1)
    for j in 1:nxi1
        C[j, :] = _cheb_coeffs(view(G, j, :))
    end
    return C
end

"""
    _trailing_ok(C::AbstractMatrix{<:Real}, dim::Integer, tau) -> Bool

The trailing-coefficient convergence test of spec D9 along dimension `dim` (`1` for ξ, the rows
of `C`; `2` for `p`, the columns): with `a_k = max` over the other dimension of `|C|`, requires
the **last two** trailing values (never one alone — one can vanish by symmetry, Pitfall 7) to be
at most `tau` times the largest.
"""
function _trailing_ok(C::AbstractMatrix{<:Real}, dim::Integer, tau)
    n1 = size(C, dim)
    a = [maximum(abs, selectdim(C, dim, k)) for k in 1:n1]
    return max(a[end-1], a[end]) <= tau * maximum(a)
end

"""
    ChebyshevTable{L,FT<:AbstractFloat}

The immutable table object (spec D10; generalized at Phase 4.B spec D13, moved from
`SuperGaussianTable`): `edges` (`np_panels+1` panel edges, ascending — `edges[1] == xi_lo`,
`edges[end] == xi_max`; `np_panels = length(nxi)`), `nxi` (per-panel ξ-degrees), `offset`
(zero-based starts of each panel's block within a slice), `np` (the one global degree in the
table's parameter coordinate `p` — **not a panel count**, which is `length(nxi)`; named after
`nxi`, Pitfall 9), `coeffs` (the `L*(np+1)`-length flat tensor-coefficient array, in slice order:
slice entry `i`'s `(np+1)` p-coefficients are `coeffs[(i-1)*(np+1)+1 : i*(np+1)]`, spec D10's
Layout), `xi_converged` (one boolean per panel: did the ξ-direction trailing-coefficient check
pass at the degree finally used there?), `p_converged` (one boolean for the whole table: did the
p-direction check pass, for every panel, at `np`, the degree finally used?), `p_lo`, `p_hi` (the
domain in `p`), `xi_lo` (`edges[1]`; `0.0` for an **even** shape's table, whose callers pass
`|ξ|` — `SG_TABLE`'s case — and negative for a **signed** one, whose callers pass the signed ξ
directly and must never take `abs` first, Pitfall 2), `xi_max` (`edges[end]`), `tau` (the fit
target actually used to build it), and the build's own four degree caps
(`degree_start`, `cap_xi`, `cap_p`, `bisect_max`, recorded so a test can read them back — Phase
4.B spec D13, D14's CT1). `L = Σ_panel (nxi[panel]+1)` is the slice length and the struct's
second type parameter.

`p` is the table's parameter coordinate: `m` for `SG_TABLE`, `SG_DM_TABLE` and `GN_TABLE`, `s = 1/κ`
for `KAPPA_TABLE`, `α` for `SN_TABLE`.

**"At the cap" and "failing at the cap" are different** (spec D9, corrected 2026-09-08 round 1):
a panel or the p-direction that reaches its degree cap and passes the trailing-coefficient check
there is a fine panel; `xi_converged[panel]` or `p_converged` being `false` means a direction
stopped *while still failing* the check — that, and only that, is a finding
([OQ-31](../../dev/open_questions.md#oq-31) item 2, for `SG_TABLE`'s own case).

Built once per table by `_build_cheb_table(sampler, edges, tau, p_lo, p_hi; degree_start, cap_xi,
cap_p, bisect_max)`, into a module `const` (`conventions.md` 5.6, the second named exception to
"no global mutable state"): `SG_TABLE` and `SG_DM_TABLE` in `supergaussian_table.jl`, `KAPPA_TABLE`
in `kappa.jl`, `GN_TABLE` in `generalized_normal.jl`, `SN_TABLE` in `skew_normal.jl`.
**Each table states its own four caps as required keywords, with no defaults, so one table's
need never moves another's** (Phase 4.B spec D13): `SG_DM_TABLE` already reaches
`SG_DEGREE_CAP_M = 48`, and a later table may need a different cap without touching that one.

Renamed from `SuperGaussianTable` at Phase 4.B Step 4.B.2a, 2026-09-26 (spec D13): field `nm` is
now `np`, `m_converged` is now `p_converged`, `m_lo`/`m_hi` are now `p_lo`/`p_hi`; `SG_TABLE` and
`SG_DM_TABLE` themselves keep their names, being the super-Gaussian's own. **Bitwise unchanged
for `SuperGaussian`** (D14): the rename touches no numeric value.

Spec D10; Phase 3.C spec D4, D9; Phase 4.B spec D13; `docs/derivations/supergaussian.md`
Section 7.
"""
struct ChebyshevTable{L,FT<:AbstractFloat}
    edges::Vector{FT}
    nxi::Vector{Int}
    offset::Vector{Int}
    np::Int
    coeffs::Vector{FT}
    xi_converged::Vector{Bool}
    p_converged::Bool
    p_lo::FT
    p_hi::FT
    xi_lo::FT          # edges[1]; 0.0 for an even shape's table, negative for a signed one
    xi_max::FT
    tau::FT
    degree_start::Int  # the build's four caps, as passed
    cap_xi::Int
    cap_p::Int
    bisect_max::Int
end

"""
    _cheb_sample_grid!(memo, sampler, a, b, nxi, np, p_lo, p_hi) -> Matrix{Float64}

Samples `sampler(p, x)` on the `(nxi+1) × (np+1)` Lobatto grid of panel `[a, b] × [p_lo, p_hi]`,
memoized by the exact `(x, p)` floating-point coordinate: a Lobatto node shared between degree
`n` and degree `2n` (`t_k` at degree `n` is `t_{2k}` at degree `2n`, spec D9) maps to the
identical `Float64` value under `_cheb_unmap` (division by exactly-representable integers, so
IEEE division of the mathematically equal ratio rounds identically), so doubling a degree only
pays for the new nodes. `memo` is build-time scratch, thrown away with the panel
(`conventions.md` Section 7's "no cache beyond the table const"). `sampler` and the domain
(`p_lo`, `p_hi`) are parameters, not hard-coded (Phase 3.C spec D9; generalized at Phase 4.B spec
D13): `_sg_principal_value` builds `SG_TABLE` and `_sg_dphi_dm_reference` builds `SG_DM_TABLE`,
both `sampler(m, x)` on the same domain today, matching the general `sampler(p, x)` order.

Renamed from `_sg_sample_grid!` at Phase 4.B Step 4.B.2a (spec D13).
"""
function _cheb_sample_grid!(memo::Dict{Tuple{Float64,Float64},Float64}, sampler, a, b, nxi, np,
                             p_lo, p_hi)
    xi_vals = [_cheb_unmap(t, a, b) for t in _cheb_nodes(nxi)]
    p_vals = [_cheb_unmap(t, p_lo, p_hi) for t in _cheb_nodes(np)]
    F = Matrix{Float64}(undef, nxi + 1, np + 1)
    for (kk, xi) in enumerate(xi_vals), (ll, pp) in enumerate(p_vals)
        F[kk, ll] = get!(() -> sampler(pp, xi), memo, (xi, pp))
    end
    return F
end

"""
    _cheb_adapt_panel!(final::Vector, sampler, a, b, depth::Integer, tau, p_lo, p_hi, caps::NamedTuple)

Fits one ξ-panel `[a, b]` adaptively in both directions (spec D9): starting at
`caps.degree_start` in ξ and in `p`, doubling whichever direction's trailing-coefficient test
(`_trailing_ok`) fails, capped at `caps.cap_xi` and `caps.cap_p`. A ξ degree that would need to
exceed its cap is bisected instead, to a depth of at most `caps.bisect_max` splits of the
*original* panel (Pitfall 19; the build must terminate); a p degree that would need to exceed its
cap is simply capped, accepted, and left for the caller's own convergence test to report as a
finding — never bisected, since there is one global p-degree for the whole table. Pushes
`(a, b, nxi, np, C, memo)` NamedTuples for the finished panel(s), in ascending ξ order, onto
`final`. `sampler`, `tau` and the domain (`p_lo`, `p_hi`) are threaded through (Phase 3.C spec
D9).

**`caps` is a single `NamedTuple` argument threaded through every call, including the two
recursive calls on bisection** (Phase 4.B spec D13, Pitfall 3): a bisected panel must read the
*same* caps as its parent, never the module constants that served this function before D13 — a
call that forgot to pass `caps` on recursion would let a bisected panel silently fall back to
whatever the caller's own scope offered, which is exactly what CT2's discrimination check
reverts and catches.

Renamed from `_sg_adapt_panel!` at Phase 4.B Step 4.B.2a (spec D13); the degree caps, module
constants before this step, are now per-table arguments.
"""
function _cheb_adapt_panel!(final::Vector, sampler, a::Float64, b::Float64, depth::Integer, tau,
                             p_lo, p_hi, caps::NamedTuple)
    memo = Dict{Tuple{Float64,Float64},Float64}()
    nxi = caps.degree_start
    np = caps.degree_start
    local C
    while true
        F = _cheb_sample_grid!(memo, sampler, a, b, nxi, np, p_lo, p_hi)
        C = _tensor_coeffs(F)
        xi_ok = _trailing_ok(C, 1, tau)
        p_ok = _trailing_ok(C, 2, tau)
        if xi_ok && p_ok
            push!(final, (a = a, b = b, nxi = nxi, np = np, C = C, memo = memo))
            return
        end
        if !xi_ok && nxi >= caps.cap_xi && depth < caps.bisect_max
            mid = (a + b) / 2
            _cheb_adapt_panel!(final, sampler, a, mid, depth + 1, tau, p_lo, p_hi, caps)
            _cheb_adapt_panel!(final, sampler, mid, b, depth + 1, tau, p_lo, p_hi, caps)
            return
        end
        next_nxi = (!xi_ok && nxi < caps.cap_xi) ? min(2nxi, caps.cap_xi) : nxi
        next_np = (!p_ok && np < caps.cap_p) ? min(2np, caps.cap_p) : np
        if next_nxi == nxi && next_np == np
            push!(final, (a = a, b = b, nxi = nxi, np = np, C = C, memo = memo))
            return
        end
        nxi, np = next_nxi, next_np
    end
end

"""
    _build_cheb_table(sampler, edges, tau, p_lo, p_hi;
                      degree_start::Integer, cap_xi::Integer, cap_p::Integer,
                      bisect_max::Integer) -> ChebyshevTable

The whole build (spec D9–D12; parameterized on its sampler and domain at Phase 3.C spec D9;
generalized at Phase 4.B spec D13), run once at include time for each table: **Pass 1** adapts
every panel of `edges` independently in both directions (`_cheb_adapt_panel!`, with bisection),
yielding the final panel edges, each panel's own `nxi` and its own *required* `np`. **Pass 2**
sets `np = max` over every panel's required `np` (one global p-degree,
[OQ-31](../../dev/open_questions.md#oq-31) item 2) and rebuilds any panel whose own `np` fell
short, reusing that panel's memo so only the new p-nodes cost a reference evaluation, then
re-checks `_trailing_ok` in *both* directions on each panel's final tensor — "the degree finally
used" (spec D9) is `nxi[panel]` for the ξ direction and the shared `np` for the p direction, and
re-checking after Pass 2's rebuild is what makes that literal. `xi_converged[panel]` is that
per-panel ξ recheck; `p_converged` is the AND of the p recheck over every panel, since one global
`np` either serves the whole table or it does not. Assembles the flat `coeffs` array in slice
order (spec D10's Layout), records `xi_lo = edges[1]` and `xi_max = edges[end]` (Phase 4.B spec
D13: the old sixth positional argument `xi_max` is gone, taken from `edges` instead — for
`SG_TABLE`, `edges[end] == SG_XI_MAX == 10.0`, so the field's value is unchanged, bit for bit),
and the build's own four caps, and returns the `ChebyshevTable`.

**The caps are required keywords with no default** (Phase 4.B spec D13): every table states its
own `degree_start`, `cap_xi`, `cap_p` and `bisect_max`, so that raising one table's cap can never
silently move another's build. `edges` may start below zero: `_cheb_adapt_panel!`'s algorithm is
unchanged for a signed domain (Level 2 built signed tables with an unchanged copy of it on
2026-09-26), and `_cheb_panel_index` already works for `x` in `[xi_lo, xi_max]`.

Deterministic, machine-independent, silent, `Float64` throughout, never `BigFloat`, and bounded
by construction (`bisect_max`, `cap_xi`, `cap_p`) — the build runs at every precompile and must
terminate there (Pitfall 4, Pitfall 19).

Renamed from `_sg_build_table` at Phase 4.B Step 4.B.2a (spec D13); `SG_TABLE` and `SG_DM_TABLE`
are built from it in `supergaussian_table.jl`, unchanged bit for bit (D14).
"""
function _build_cheb_table(sampler, edges, tau, p_lo, p_hi;
                            degree_start::Integer, cap_xi::Integer, cap_p::Integer,
                            bisect_max::Integer)
    caps = (degree_start = degree_start, cap_xi = cap_xi, cap_p = cap_p, bisect_max = bisect_max)
    edges0 = collect(Float64, edges)
    panels = NamedTuple[]
    for p in 1:length(edges0)-1
        _cheb_adapt_panel!(panels, sampler, edges0[p], edges0[p+1], 0, tau, p_lo, p_hi, caps)
    end
    np_global = maximum(p.np for p in panels)
    final = NamedTuple[]
    for p in panels
        C = p.np < np_global ?
            _tensor_coeffs(_cheb_sample_grid!(p.memo, sampler, p.a, p.b, p.nxi, np_global, p_lo, p_hi)) :
            p.C
        xi_converged = _trailing_ok(C, 1, tau)
        p_converged = _trailing_ok(C, 2, tau)
        push!(final, (a = p.a, b = p.b, nxi = p.nxi, C = C,
                      xi_converged = xi_converged, p_converged = p_converged))
    end
    edges_out = Float64[final[1].a]
    for p in final
        push!(edges_out, p.b)
    end
    nxi_vec = Int[p.nxi for p in final]
    xi_converged_vec = Bool[p.xi_converged for p in final]
    p_converged = all(p.p_converged for p in final)
    offset = Int[]
    off = 0
    for n in nxi_vec
        push!(offset, off)
        off += n + 1
    end
    L = off
    coeffs = Vector{Float64}(undef, L * (np_global + 1))
    for (pi, p) in enumerate(final)
        for j in 0:p.nxi
            slice_idx = offset[pi] + j
            coeffs[slice_idx*(np_global+1)+1:(slice_idx+1)*(np_global+1)] .= view(p.C, j + 1, :)
        end
    end
    xi_lo = edges_out[1]
    xi_max = edges_out[end]
    return ChebyshevTable{L,Float64}(edges_out, nxi_vec, offset, np_global, coeffs,
                                      xi_converged_vec, p_converged,
                                      p_lo, p_hi, xi_lo, xi_max, tau,
                                      degree_start, cap_xi, cap_p, bisect_max)
end

"""
    _cheb_panel_index(table, x) -> Int

The panel owning a *primal* `x` (spec D15; generalized at Phase 4.B spec D13 for a signed
domain): left-closed, right-open, with the last panel closed, so that `x = table.xi_lo` gives
panel 1 and `x = table.xi_max` gives the last panel. Adjacent panels share their endpoint node,
so both give the same value there to round-off (TB9). Unchanged since Phase 1.SG: `x` need not
be nonnegative, since `searchsortedlast` on `table.edges` works identically whether `edges[1]` is
`0.0` or negative.

Renamed from `_sg_panel_index` at Phase 4.B Step 4.B.2a (spec D13).
"""
_cheb_panel_index(table, x) = clamp(searchsortedlast(table.edges, x), 1, length(table.edges) - 1)

"""
    _cheb_slice(table::ChebyshevTable{L}, p) where {L} -> NTuple{L}

The table's ξ-coefficients at this `p`, one panel after another, flattened: for every slice
entry `i = 1, …, L`, the one-dimensional Clenshaw evaluation in `p` of `coeffs`' `i`-th
`(np+1)`-block. `Val(L)`, never a plain `L`, so the result is `NTuple{L, FT}`, inferable and
heap-free — measured at `5.53 µs`, zero allocation, `L = 198` for `SG_TABLE` (spec D3, D13,
round 2). With a dual `p` this returns an `NTuple` of duals of the same tag, which is how a
p-derivative flows through a cached shape.

Renamed from `_sg_slice` at Phase 4.B Step 4.B.2a (spec D13).
"""
function _cheb_slice(table::ChebyshevTable{L}, p) where {L}
    s = _cheb_map(p, table.p_lo, table.p_hi)
    return ntuple(i -> _cheb_eval(table.coeffs, (i - 1) * (table.np + 1), table.np, s), Val(L))
end

"""
    _cheb_table_eval(table, slice, x) -> Real

The per-bin evaluator: the table's value at ξ = `x` (possibly a `Dual`), read out of a shape's
own `slice` by one panel's one-dimensional Clenshaw in ξ. The panel index is taken at `x`'s
primal value (`_primal_value`, spec 5.12 — an index is never built from a dual); the Clenshaw
evaluation itself is generic in `x`'s type. Measured at `39.3 ns`, zero allocation, for
`SG_TABLE` (spec D13, round 2).

**`x` is in `[table.xi_lo, table.xi_max]`** (Phase 4.B spec D13): an **even** shape's table has
`xi_lo == 0` and its callers pass `|ξ|` (`SuperGaussian`'s case, unchanged); a table with
`xi_lo < 0` is evaluated at the **signed** ξ, and passing `|ξ|` to it is wrong (Pitfall 2) — it
would silently fold the negative half of the domain onto the positive one.

Renamed from `_sg_table_eval` at Phase 4.B Step 4.B.2a (spec D13).
"""
function _cheb_table_eval(table, slice, x)
    panel = _cheb_panel_index(table, _primal_value(x))
    a = table.edges[panel]
    b = table.edges[panel+1]
    return _cheb_eval(slice, table.offset[panel], table.nxi[panel], _cheb_map(x, a, b))
end
