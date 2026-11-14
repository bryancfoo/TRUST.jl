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

# ------------------------------------------------------------------------------------------
# A trace written on grids, and a grid sized for a trace (Phase G.2 Step G.2.1, spec D5 to D7;
# OQ-69 items 2 and 3)
# ------------------------------------------------------------------------------------------

_grid_species_set(p::Plasma, set::Symbol) = set === :electrons ? p.electrons : p.ions

"""
    _grid_source(trace, source, caller) -> (set, idxs, keys)

A source of `project_to_grid` or `size_grid`, checked against `trace`'s template (Phase G.2 spec D5):
one species' key, `"electrons.<name>"` or `"ions.<name>"`, or a tuple or vector of such keys. Returns
the set (`:electrons` or `:ions`), the species' positions in it and the keys, in the order given.
Refused, each message beginning with `caller`: anything but a key or a collection of keys; an empty
collection; a key of another form; a key that names no species; keys from both sets; a key given twice;
species of different masses or charges; a species whose charge is a leaf of the trace.
"""
function _grid_source(trace::PlasmaTrace, source, caller::String)
    ks = if source isa AbstractString
        [String(source)]
    elseif source isa Union{Tuple, AbstractVector}
        all(k -> k isa AbstractString, source) || throw(ArgumentError(
            "$caller: a source's keys must be strings; got $(source)"))
        String[String(k) for k in source]
    else
        throw(ArgumentError("$caller: a source is a species key (\"electrons.<name>\" or \"ions.<name>\") " *
                            "or a tuple or vector of such keys; got $(typeof(source))"))
    end
    isempty(ks) && throw(ArgumentError("$caller: a source needs at least one species key"))
    set = :none
    idxs = Int[]
    for key in ks
        segs = split(key, '.')
        (length(segs) == 2 && (segs[1] == "electrons" || segs[1] == "ions")) || throw(ArgumentError(
            "$caller: \"$key\" is not a species key; expected \"electrons.<name>\" or \"ions.<name>\""))
        s = Symbol(segs[1])
        idx = findfirst(==(segs[2]), _grid_species_set(trace.template, s).names)
        idx === nothing && throw(ArgumentError("$caller: \"$key\" names no species of the trace"))
        (set === :none || set === s) || throw(ArgumentError(
            "$caller: a source's species are all electrons or all ions; got \"$(ks[1])\" and \"$key\""))
        idx in idxs && throw(ArgumentError("$caller: \"$key\" is given twice"))
        set = s
        push!(idxs, idx)
    end
    species_set = _grid_species_set(trace.template, set)
    rows = set === :electrons ? trace.electron_rows : trace.ion_rows
    sp1 = species_set[idxs[1]]
    for k in eachindex(idxs)
        sp = species_set[idxs[k]]
        (sp.mass == sp1.mass && sp.charge == sp1.charge) || throw(ArgumentError(
            "$caller: \"$(ks[k])\" and \"$(ks[1])\" differ in mass or charge; one source is one mass " *
            "and one charge"))
        rows[idxs[k]].charge == 0 || throw(ArgumentError(
            "$caller: \"$(ks[k])\"'s charge is a leaf of the trace; a source's charge must be constant"))
    end
    return set, idxs, ks
end

"""
    size_grid(trace::PlasmaTrace, source; spacing = 0.2, extent = 4, width = 1.2, symmetric = false)
        -> GaussianGrid{Float64}

The grid `project_to_grid` would write `source` on: a `GaussianGrid` at `g = 0` spanning every state
the source's species take in `trace` (Phase G.2, [OQ-69](../../dev/open_questions.md#oq-69) item 3).
`source` is one species' key, `"electrons.<name>"` or `"ions.<name>"`, or a tuple or vector of keys of
one set with one mass and one charge. Over every bin and every species of the source, with `T_hot` and
`T_cold` the highest and lowest `T`, `u_lo` and `u_hi` the lowest and highest `u`, and `m` the mass:

    v_ref = √(2 T_hot/m),    dv = spacing · √(2 T_cold/m),    σ = width · dv,
    u₀ = (u_lo + u_hi)/2,    Nv = 2 ⌈((u_hi − u_lo)/2 + extent · v_ref)/dv⌉ + 1,

the smallest odd node count whose end nodes reach `u_lo − extent · v_ref` and `u_hi + extent · v_ref`
(an end that falls between two nodes takes the outer one). The default weights are the Maxwellian of
`T_hot` at `u₀`: `GaussianGrid`'s temperature form at `T_ref = T_hot`, its spacing `dv/v_ref`. So the
grid resolves the coldest state and spans the hottest; in every colder bin the projection's `g` fall
in the tails, the harmless sign (`docs/derivations/gaussian_grid.md` Section 11). Each `T` and `u` is
read from `plasma(trace, j)`, so a source that is already gridded is sized from its computed moments.

`symmetric = true` makes a symmetric grid and is refused unless every flow of the source is the same.
Refused also: a `spacing`, `extent` or `width` that is not finite and positive; a source
`project_to_grid` refuses; and whatever `GaussianGrid` refuses (`spacing · width · √(T_cold/T_hot)`
must be below 1). **The defaults are the user's of 2026-11-13** (Phase G.2 spec D6): spacing 0.2 and extent
4, where OQ-69 proposed 0.25 and 5, which left the EPW ablation problem's twin 0.74 σ off its parametric
streak at the super-Gaussian's flat top; Hold 1 revisits all three numbers.

`docs/derivations/gaussian_grid.md` Section 11; Phase G.2 spec D6.
"""
function size_grid(trace::PlasmaTrace{Float64}, source; spacing::Real = 0.2, extent::Real = 4,
                   width::Real = 1.2, symmetric::Bool = false)
    set, idxs, _ = _grid_source(trace, source, "size_grid")
    for (name, x) in (("spacing", spacing), ("extent", extent), ("width", width))
        (isfinite(x) && x > 0) || throw(ArgumentError("size_grid: $name must be finite and positive; got $x"))
    end
    sp_hot = sp_cold = _grid_species_set(trace.template, set)[idxs[1]]
    u_lo, u_hi = Inf, -Inf
    T_hot, T_cold = -Inf, Inf
    for j in 1:nbins(trace)
        species_set = _grid_species_set(plasma(trace, j), set)
        for idx in idxs
            sp = species_set[idx]
            sp.T > T_hot && ((T_hot, sp_hot) = (sp.T, sp))
            sp.T < T_cold && ((T_cold, sp_cold) = (sp.T, sp))
            u_lo = min(u_lo, sp.u)
            u_hi = max(u_hi, sp.u)
        end
    end
    (symmetric && u_lo != u_hi) && throw(ArgumentError(
        "size_grid: symmetric = true needs one flow for the whole source; its flows run from " *
        "$(u_lo / 1e3) to $(u_hi / 1e3) km/s"))
    v_ref = thermal_speed(sp_hot)
    grid_kwargs = (mass_kg = Float64(sp_hot.mass), T_ref_eV = T_hot / EV, u0_ms = (u_lo + u_hi) / 2,
                   spacing = spacing * thermal_speed(sp_cold) / v_ref, width = width, symmetric = symmetric)
    dv = GaussianGrid(; nodes = 3, grid_kwargs...).dv
    half = ceil(Int, ((u_hi - u_lo) / 2 + extent * v_ref) / dv)
    return GaussianGrid(; nodes = 2 * half + 1, grid_kwargs...)
end

# The source's fraction in one bin, the left-to-right sum over its species in the order given (spec D5).
function _source_fraction(species_set, idxs::Vector{Int})
    total = species_set[idxs[1]].fraction
    for k in 2:length(idxs)
        total += species_set[idxs[k]].fraction
    end
    return total
end

# One bin of one source on its grid (spec D5): one species is G.1's projection of that species; several
# are the projection of their fraction-weighted mean, with equal weights in a bin where every fraction is 0.
function _project_source(species_set, idxs::Vector{Int}, grid::GaussianGrid{Float64})
    length(idxs) == 1 && return project_to_grid(species_set[idxs[1]], grid)
    members = Any[species_set[i] for i in idxs]
    total = _source_fraction(species_set, idxs)
    weights = total > 0 ? Float64[sp.fraction for sp in members] : ones(length(members))
    norm = total > 0 ? total : Float64(length(members))
    function fsrc(v)
        s = weights[1] * f_lab(members[1], v)
        for i in 2:length(members)
            s += weights[i] * f_lab(members[i], v)
        end
        return s / norm
    end
    function csrc(v)
        s = weights[1] * _lab_cauchy(members[1].f, members[1], v)
        for i in 2:length(members)
            s += weights[i] * _lab_cauchy(members[i].f, members[i], v)
        end
        return s / norm
    end
    return _project_to_grid(fsrc, csrc, grid)
end

_projection_target(g::GaussianGrid{Float64}) = g
_projection_target(g) = throw(ArgumentError(
    "project_to_grid: each source's grid must be a GaussianGrid{Float64}; got $(typeof(g))"))

"""
    project_to_grid(trace::PlasmaTrace, source => grid, …; names = nothing)
        -> (trace, error_f, error_cauchy, iterations, converged)

`trace` with each `source` replaced by **one** species carrying a `GaussianGrid` on `grid`, projected
bin by bin (Phase G.2, [OQ-69](../../dev/open_questions.md#oq-69) item 2). A source is one species'
key, `"electrons.<name>"` or `"ions.<name>"`, or a tuple or vector of keys of one set with one mass and
one charge; `grid` a `GaussianGrid{Float64}`, whose centre, spacing, width, symmetry and default the
result takes (`size_grid` makes one). In bin `j` the new species' fraction is `Σᵢ fractionᵢⱼ` and its
distribution the projection of

    fⱼ(v) = Σᵢ fractionᵢⱼ f_lab,ᵢⱼ(v) / Σᵢ fractionᵢⱼ

onto `grid`, the sum over the source's species in the order given (equal weights in a bin where every
fraction is 0); for one species it is Phase G.1's `project_to_grid(species, grid)`, bitwise. For
electrons the fractions are of `n_e`, and for ions of one charge they are in the ratio of the number
densities, so `fⱼ` is the number-weighted mixture either way.

**The trace returned.** The new species takes the first key's name, or `names[s]` for source `s`
(`names` a tuple or vector of strings, one per source), and the first key's place in its set; the other
species of a merged source are removed. Its template distribution is `grid` itself and its template
fraction the sum of the sources' template fractions. The sources' `T`, `u` and shape-parameter leaves
leave the trace; its fraction is a leaf, at the place of the first source fraction row, wherever any
source's fraction was one; its parameters arrive as rows, `<set>.<name>.f.g1` onwards, after every other
row, source by source. Every other species, row and value is copied unchanged, bitwise.

**Returned beside it**, one entry per source in the order given, each a vector over the bins: the
projection's `error_f` and `error_cauchy` (Phase G.1's, `docs/derivations/gaussian_grid.md` Section 7),
its `iterations` and whether it `converged`. A bin that does not converge is reported, never thrown.

Refused: a source `_grid_source` refuses (another form, no such species, both sets, two masses or charges,
a charge that is a leaf), a species in two sources, a grid that is not a `GaussianGrid{Float64}`, a
`names` of the wrong length; and whatever `Plasma` refuses of the new template (a name used twice).
`trace` must be a plain `Float64` trace; the projection is not differentiable and never on the fit path.

`docs/derivations/gaussian_grid.md` Section 11; Phase G.2 spec D5.
"""
function project_to_grid(trace::PlasmaTrace{Float64}, sources::Pair...; names = nothing)
    isempty(sources) && throw(ArgumentError("project_to_grid: no source => grid pair given"))
    parsed = [_grid_source(trace, first(s), "project_to_grid") for s in sources]
    grids = [_projection_target(last(s)) for s in sources]
    NS = length(sources)
    for a in 1:NS, b in (a + 1):NS
        parsed[a][1] === parsed[b][1] || continue
        for (k, idx) in enumerate(parsed[b][2])
            idx in parsed[a][2] && throw(ArgumentError(
                "project_to_grid: \"$(parsed[b][3][k])\" is in two sources"))
        end
    end
    newnames = if names === nothing
        String[_grid_species_set(trace.template, parsed[s][1]).names[parsed[s][2][1]] for s in 1:NS]
    else
        length(names) == NS || throw(ArgumentError(
            "project_to_grid: names has $(length(names)) entries for $NS sources"))
        String[String(n) for n in names]
    end
    nt = nbins(trace)
    gvals = [Matrix{Float64}(undef, length(parameters(grids[s])), nt) for s in 1:NS]
    fracs = [Vector{Float64}(undef, nt) for _ in 1:NS]
    error_f = [Vector{Float64}(undef, nt) for _ in 1:NS]
    error_cauchy = [Vector{Float64}(undef, nt) for _ in 1:NS]
    iterations = [Vector{Int}(undef, nt) for _ in 1:NS]
    converged = [Vector{Bool}(undef, nt) for _ in 1:NS]
    for j in 1:nt
        p = plasma(trace, j)
        for s in 1:NS
            species_set = _grid_species_set(p, parsed[s][1])
            res = _project_source(species_set, parsed[s][2], grids[s])
            g = parameters(res.f)
            for k in 1:length(g)
                gvals[s][k, j] = g[k]
            end
            fracs[s][j] = _source_fraction(species_set, parsed[s][2])
            error_f[s][j] = res.error_f
            error_cauchy[s][j] = res.error_cauchy
            iterations[s][j] = res.iterations
            converged[s][j] = res.converged
        end
    end
    newtrace = _gridded_trace(trace, parsed, grids, newnames, gvals, fracs)
    return (trace = newtrace, error_f = error_f, error_cauchy = error_cauchy, iterations = iterations,
            converged = converged)
end

# The trace project_to_grid returns (spec D5): the new template, then the rows in their order.
function _gridded_trace(trace::PlasmaTrace{Float64}, parsed, grids, newnames, gvals, fracs)
    template = trace.template
    source_of(set, idx) = findfirst(ps -> ps[1] === set && idx in ps[2], parsed)
    specs = Dict{Symbol,Vector{Any}}(:electrons => Any[], :ions => Any[])
    for set in (:electrons, :ions)
        species_set = _grid_species_set(template, set)
        for idx in 1:length(species_set)
            s = source_of(set, idx)
            if s === nothing
                push!(specs[set], spec(species_set[idx]))
            elseif parsed[s][2][1] == idx
                sp = species_set[idx]
                fraction = _source_fraction(species_set, parsed[s][2])
                push!(specs[set], set === :electrons ?
                    electrons(; name = newnames[s], f = grids[s], fraction = fraction) :
                    ion(; name = newnames[s], mass_kg = sp.mass, charge = sp.charge, f = grids[s], fraction = fraction))
            end
        end
    end
    new_template = Plasma(; n_e_m3 = template.n_e, electrons = specs[:electrons], ions = specs[:ions])

    paths = LeafPath[]
    rows = Vector{Float64}[]
    fraction_done = falses(length(parsed))
    for (i, path) in enumerate(trace.paths)
        ref = _resolve_leaf(template, path)
        s = ref.set === :n_e ? nothing : source_of(ref.set, ref.index)
        if s === nothing
            push!(paths, path)
            push!(rows, trace.values[i, :])
        elseif ref.field === :fraction && !ref.shape && !fraction_done[s]
            fraction_done[s] = true
            push!(paths, LeafPath([String(ref.set), newnames[s], "fraction"]))
            push!(rows, fracs[s])
        end
    end
    for s in eachindex(parsed)
        pnames = keys(parameters(grids[s]))
        for k in eachindex(pnames)
            push!(paths, LeafPath([String(parsed[s][1]), newnames[s], "f", String(pnames[k])]))
            push!(rows, gvals[s][k, :])
        end
    end
    nt = nbins(trace)
    values_mat = Matrix{Float64}(undef, length(paths), nt)
    for i in eachindex(rows), j in 1:nt
        values_mat[i, j] = rows[i][j]
    end
    refs = [_resolve_leaf(new_template, p) for p in paths]
    return _finish_trace(new_template, trace.axis_name, copy(trace.axis), paths, refs, values_mat)
end
