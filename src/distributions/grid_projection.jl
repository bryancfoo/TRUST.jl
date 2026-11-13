# The projection of a distribution onto a GaussianGrid: positive weights by a deterministic least-squares
# fit of f in the logarithm of the weights, Levenberg-Marquardt with a convergence test, on a collocation set
# four times finer than the grid; it reports its own error in f and in the Cauchy integral. Not a fit of
# data and never on the fit path. See docs/derivations/gaussian_grid.md Section 7. Phase G.1 Step G.1.3,
# spec D14 (OQ-68 item 6).

using LinearAlgebra: cholesky, Symmetric

"""
    GRID_PROJECTION_G_BOUNDS, GRID_PROJECTION_RTOL, GRID_PROJECTION_MAXITER

The projection's constants (Phase G.1 spec D14). Every log-weight is held in `GRID_PROJECTION_G_BOUNDS`
`= (-200.0, 200.0)` (a weight `e^{−200}` of the default's is nothing in Float64, and the bound keeps a
node the source leaves empty from walking to `−∞`); the iteration stops, converged, at the first
accepted step that lowers the sum of squares by less than `GRID_PROJECTION_RTOL = 1e-12` of itself, or
when no step lowers it at a damping above `1e12` (the minimum to working precision); and it stops,
**not** converged, after `GRID_PROJECTION_MAXITER = 500` iterations.
"""
const GRID_PROJECTION_G_BOUNDS = (-200.0, 200.0)
const GRID_PROJECTION_RTOL = 1e-12
const GRID_PROJECTION_MAXITER = 500

"""
    project_to_grid(source, target::GaussianGrid) -> (f, error_f, error_cauchy, iterations, converged)

The distribution of `source` written on `target`'s grid: a `GaussianGrid` with `target`'s centre,
spacing, width, symmetry and default weights, and the log-weights `g` that best represent `source`
([OQ-68](../../dev/open_questions.md#oq-68) item 6). `source` is a `Species` or `SpeciesSpec` of either
family (a shape with its `u`, `T` and mass, through `f_lab`; a lab-frame distribution itself), or any
`AbstractLabDistribution`, a `GaussianGrid` on another grid among them.

**The method** (Phase G.1 spec D14), deterministic: the unnormalized comb `p(v) = Σₙ w⁰ₙ e^{g_{π(n)}}
G(v; vₙ, σ)` (`π(n)` the parameter node `n` reads) is fitted to the source's `f(v)` by least squares
on the **collocation set** from `v₁ − 5σ` to `v_Nv + 5σ` in steps of `dv/4`, the residuals scaled by
`dv`. **The start** is the source sampled at the nodes, `w⁰ₙ e^{gₙ} = f(vₙ) dv` (for the symmetric
option, the mean over each mirrored pair), floored at `1e-300` and every `g` held in
`GRID_PROJECTION_G_BOUNDS`. **Levenberg–Marquardt in `g`**: the step solves `(JᵀJ + λ D) δ = −Jᵀr`,
`D` the diagonal of `JᵀJ` floored at `1e-12` of its largest entry, by Cholesky; `λ` starts at `1e-3`,
divides by 3 on an accepted step and multiplies by 5 on a rejected one; the stopping test is
`GRID_PROJECTION_RTOL`'s and `GRID_PROJECTION_MAXITER`'s.

**The errors it reports**, on the **evaluation set** from `v₁` to `v_Nv` in steps of `dv/8`:
`error_f = max |f − f_source| / max |f_source|` and `error_cauchy = max |𝒞f − 𝒞f_source| / max |𝒞f_source|`,
complex magnitudes, `𝒞` in lab velocity (for a shape `𝒞φ(ξ)/v_th²`, `ξ = (v − u)/v_th`). They measure
the grid's representation of the source, which is limited by the grid (`docs/derivations/gaussian_grid.md`
Section 7): a feature narrower than `σ`, a tail beyond the last node, a negative lobe, none of which a
positive comb on a fixed grid can hold.

`target` must be a plain `GaussianGrid{Float64}`; the projection is not differentiable and never on the fit
path. Its `g` are ignored: the result's are the fit's.

`docs/derivations/gaussian_grid.md` Section 7; Phase G.1 spec D14.
"""
project_to_grid(sp::Union{Species, SpeciesSpec}, target::GaussianGrid{Float64}) =
    _project_to_grid(v -> f_lab(sp, v), v -> _lab_cauchy(sp.f, sp, v), target)

project_to_grid(source::AbstractLabDistribution, target::GaussianGrid{Float64}) =
    _project_to_grid(v -> source(v), v -> cauchy_dfdv(source, v), target)

# The Cauchy integral in lab velocity of a species' distribution (Phase G.1 spec D14): a shape's in its own
# reduced variable divided by v_th^2 (conventions.md 1.12), a lab-frame distribution's itself.
function _lab_cauchy(f::AbstractVelocityDistribution, sp, v)
    vth = thermal_speed(sp)
    return cauchy_dfdv(f, (v - sp.u) / vth) / vth^2
end
_lab_cauchy(f::AbstractLabDistribution, sp, v) = cauchy_dfdv(f, v)

function _project_to_grid(fsrc, csrc, target::GaussianGrid{Float64, Nv, NP}) where {Nv, NP}
    nodes = [_grid_node(target, n) for n in 1:Nv]
    par = [_grid_parameter(n, Val(Nv), Val(NP)) for n in 1:Nv]
    g, iterations, converged = _project_lm(fsrc, nodes, par, NP, collect(target.w0), target.sigma, target.dv)
    f = reconstruct(target, NamedTuple{_grid_names(Val(NP))}(ntuple(k -> g[k], Val(NP))))
    error_f, error_cauchy = _projection_errors(fsrc, csrc, f, nodes[1], nodes[end], target.dv)
    return (f = f, error_f = error_f, error_cauchy = error_cauchy, iterations = iterations, converged = converged)
end

# The least-squares fit of spec D14, on plain vectors and not specialized on the source, so that it compiles
# once rather than once per source type and node count (the projection is never on the fit path).
function _project_lm(@nospecialize(fsrc), nodes::Vector{Float64}, par::Vector{Int}, NP::Int,
                     w0::Vector{Float64}, sigma::Float64, dv::Float64)
    Nv = length(nodes)
    vc = collect(range(nodes[1] - 5sigma, nodes[end] + 5sigma; step = dv / 4))
    b = Float64[dv * fsrc(v) for v in vc]
    G = [dv * exp(-((v - nodes[n]) / sigma)^2) / (sigma * sqrt(pi)) for v in vc, n in 1:Nv]
    lo, hi = GRID_PROJECTION_G_BOUNDS

    # the start: the source sampled at the nodes, summed over each mirrored pair with the symmetric option
    c0 = Float64[max(fsrc(nodes[n]) * dv, 1e-300) for n in 1:Nv]
    g = zeros(NP)
    for k in 1:NP
        members = findall(==(k), par)
        g[k] = clamp(log(sum(c0[members]) / sum(w0[members])), lo, hi)
    end

    c = similar(w0)
    _projection_weights!(c, g, w0, par)
    r = G * c .- b
    cost = sum(abs2, r)
    lambda = 1e-3
    iterations = 0
    converged = false
    J = zeros(length(vc), NP)
    while iterations < GRID_PROJECTION_MAXITER
        iterations += 1
        _projection_weights!(c, g, w0, par)
        fill!(J, 0.0)
        for n in 1:Nv, i in eachindex(vc)
            J[i, par[n]] += G[i, n] * c[n]
        end
        A = J' * J
        grad = J' * r
        dmax = maximum(A[k, k] for k in 1:NP)
        D = [max(A[k, k], 1e-12 * dmax) for k in 1:NP]
        accepted = false
        while lambda <= 1e12
            M = copy(A)
            for k in 1:NP
                M[k, k] += lambda * D[k]
            end
            g_new = clamp.(g .- (cholesky(Symmetric(M)) \ grad), lo, hi)
            _projection_weights!(c, g_new, w0, par)
            r_new = G * c .- b
            cost_new = sum(abs2, r_new)
            if cost_new < cost
                converged = cost - cost_new <= GRID_PROJECTION_RTOL * cost_new
                g, r, cost = g_new, r_new, cost_new
                lambda /= 3
                accepted = true
                break
            end
            lambda *= 5
        end
        accepted || (converged = true)
        converged && break
    end
    return g, iterations, converged
end

# The weights w0_n exp(g_par(n)) into c (spec D14).
function _projection_weights!(c::Vector{Float64}, g::Vector{Float64}, w0::Vector{Float64}, par::Vector{Int})
    for n in eachindex(c)
        c[n] = w0[n] * exp(g[par[n]])
    end
    return c
end

# The reported errors of spec D14, on the evaluation set v_1 to v_Nv in steps of dv/8.
function _projection_errors(@nospecialize(fsrc), @nospecialize(csrc), f::GaussianGrid, v1::Float64, vN::Float64, dv::Float64)
    ve = range(v1, vN; step = dv / 8)
    error_f = maximum(abs(f(v) - fsrc(v)) for v in ve) / maximum(abs(fsrc(v)) for v in ve)
    error_cauchy = maximum(abs(cauchy_dfdv(f, v) - csrc(v)) for v in ve) / maximum(abs(csrc(v)) for v in ve)
    return error_f, error_cauchy
end
