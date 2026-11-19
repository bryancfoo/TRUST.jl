# A gridded fit from Julia (Phase G.4 Steps G.4.2 and G.4.4a; OQ-71 items 4 and 9): the fitted variables of a
# GaussianGrid's log-weights, one of them held at 0, and two starts for them from a gridded trace, its own g or the
# Maxwellian of each bin's moments. Nothing here fits anything or reads a deck: a caller adds the variables and bindings
# to its own, and builds the ParameterSpec, the Regularizer and the FitProblem as for any fit.

"""
    grid_variables(grid::GaussianGrid, key::AbstractString; prefix, bounds, time_varying = true, interval = 0,
                   hold = :centre) -> (; variables, bindings, names, held)

The fitted variables of the log-weights of a species that carries `grid` ([OQ-71](../../dev/open_questions.md#oq-71)
item 4): one `FittedVariable` per parameter of the grid (`g1` to `gNP`: the node count, or `⌈Nv/2⌉` with the
symmetric option), parameter `k` named `prefix * string(k)`, with `bounds` and `init = 0`, the default distribution;
**one of them, parameter `held`, is not varied and is held at 0**.

- `key` is the species' key as a trace spells it, `"electrons.<name>"` or `"ions.<name>"`.
- `prefix` is a stem that makes a variable name (`FittedVariable`'s rule, a Julia identifier).
- `bounds` is `(lower, upper)` in g, dimensionless, the same for every parameter, with `lower < 0 < upper`; required,
  with no default.
- `time_varying` and `interval` are every variable's, as `FittedVariable` takes them.
- `hold` is `:centre`, the parameter of the node nearest the grid's centre `u₀`: node `(Nv + 1)/2` for an odd node
  count, **the lower of the two middle nodes**, `Nv/2`, for an even one, and with the symmetric option the middle
  parameter, the last (`dev/gridded_plan.md` GQ-4); or an integer `k` in `1:NP`, parameter `k`.

Returns `variables` (parameter order, the held one among them), `bindings` (`name => "<key>.f.g<k>"` for every
parameter, the held one too, the leaf keys of a gridded species), `names` (the names in parameter order: the `select`
of `VelocityRegularizerTerm(pspec; select = names, grid = grid)`) and `held` (the held parameter's index).

**Why one is held.** `f` is divided by the sum of its weights, so adding one constant to every `g` of a bin changes
nothing: exactly one flat direction per bin and no other. Holding one `g` at 0 removes it, and the other `NP − 1`
then describe every positive unit-integral comb on the grid once (`docs/derivations/gaussian_grid.md` Section 12;
`dev/gridded_plan.md` M9). A fitted `g_n` is then the logarithm of `f(v_n)/f(v_held)` over the default's same ratio,
so `bounds` bound that ratio, and a fitted distribution is read from its normalized weights, never from raw `g`.

Refused, each with a message naming the input: a `key` of another form; a `prefix` that does not make a variable
name; `bounds` that are not two finite numbers with `lower < 0 < upper`; a `hold` that is neither `:centre` nor an
integer in `1:NP`.
"""
function grid_variables(grid::GaussianGrid{FT, Nv, NP}, key::AbstractString; prefix::AbstractString, bounds,
                        time_varying::Bool = true, interval::Integer = 0, hold = :centre) where {FT, Nv, NP}
    _grid_key(key, "grid_variables")
    occursin(_VARIABLE_NAME_RE, prefix) || throw(ArgumentError(
        "grid_variables: prefix \"$prefix\" does not make a variable name; expected $(_VARIABLE_NAME_RE.pattern)"))
    (length(bounds) == 2 && all(b -> b isa Real && isfinite(b), bounds) && bounds[1] < 0 < bounds[2]) ||
        throw(ArgumentError("grid_variables: bounds must be two finite numbers (lower, upper) in g with " *
                            "lower < 0 < upper, since every parameter starts at 0, the default; got $(repr(bounds))"))
    held = if hold === :centre
        cld(Nv, 2)
    elseif hold isa Integer && !(hold isa Bool) && 1 <= hold <= NP
        Int(hold)
    else
        throw(ArgumentError("grid_variables: hold must be :centre or an integer in 1:$NP, the grid's parameters; " *
                            "got $(repr(hold))"))
    end
    lo, hi = Float64(bounds[1]), Float64(bounds[2])
    names = [string(prefix, k) for k in 1:NP]
    variables = [FittedVariable(names[k]; bounds = (lo, hi), init = 0.0, time_varying = time_varying,
                                vary = k != held, interval = interval) for k in 1:NP]
    bindings = [names[k] => string(key, ".f.g", k) for k in 1:NP]
    return (; variables, bindings, names, held)
end

"""
    grid_start(trace::PlasmaTrace, key::AbstractString, family) -> Vector{Pair{String, Vector{Float64}}}

A start for `family`, what `grid_variables` returned, from `trace`, a trace whose species `key` carries the family's
grid (in practice the `trace` of `project_to_grid`'s result; [OQ-71](../../dev/open_questions.md#oq-71) item 4). At
each bin it reads the species' `g` and **subtracts the held parameter's `g` from every `g` of that bin**, which leaves
the distribution what it was, and returns `name => values` over the trace's bins for every varied parameter, in
parameter order: the form `flatten(pspec, values)` takes. Every bin of `trace` is read, whatever the variables'
interval.

A value not strictly inside the family's bounds is **refused, naming the parameter, the bin and the value**: nothing
is clipped, and a value on a bound has no logit. Refused as well: a key of another form, or one that names no species
of the trace; a species that does not carry a `GaussianGrid`; a family whose parameter count is not the grid's, or
whose bindings are not to `key`'s log-weights (the family records no grid, so a grid of the same parameter count is
taken for the family's); a family whose variables are not time-varying.
"""
function grid_start(trace::PlasmaTrace, key::AbstractString, family)
    set, idx, NP = _grid_family(trace, key, family, "grid_start")
    h = family.held
    nt = nbins(trace)
    out = [family.names[k] => Vector{Float64}(undef, nt) for k in 1:NP if k != h]
    for j in 1:nt
        g = parameters(_grid_species_set(plasma(trace, j), set)[idx].f)
        gh = g[h]
        q = 0
        for k in 1:NP
            k == h && continue
            q += 1
            last(out[q])[j] = _grid_start_value(family.variables[k], j, Float64(g[k] - gh), "grid_start")
        end
    end
    return out
end

"""
    grid_maxwellian_start(trace::PlasmaTrace, key::AbstractString, family) -> Vector{Pair{String, Vector{Float64}}}

A start for `family`, what `grid_variables` returned, that carries no shape: at each bin of `trace`, **the Maxwellian
of the species' own flow and variance**, written on the species' grid against the grid's default weights and
re-centred on the held parameter ([OQ-71](../../dev/open_questions.md#oq-71), "The start hold": the user's choice of
2026-11-16). `trace` is a trace whose species `key` carries the family's grid, in practice the `trace` of
`project_to_grid`'s result. At bin `j`, with `ū = mean_velocity(f)` and `V = velocity_variance(f)` of that bin's
distribution `f`, node velocities `vₙ`, the grid's kernel width `σ` and default weights `w⁰ₙ`:

    gₙ = −((vₙ − ū)/s)² − ln w⁰ₙ,   s = √(2V − σ²),

and then `g_held` is subtracted from every `gₙ` of the bin, which leaves the distribution what it was. The node
weights `w⁰ₙ e^{gₙ}` are then a Gaussian of variance `s²/2` about `ū`, and each node's own Gaussian adds `σ²/2`, so
the comb has the bin's variance `V` (OQ-68 item 3; to the nodes' sampling of that Gaussian). On `size_grid`'s grids,
whose default is itself a Gaussian, `g` is a quadratic in `v` and the velocity term's rows are zero on it. Only the
two moments of each bin are read: the projection's tails, floored at −200 or left unconstrained, are not.

Returns what `grid_start` returns, `name => values` over the trace's bins for every varied parameter, in parameter
order, the form `flatten(pspec, values)` takes; every bin of `trace` is read, whatever the variables' interval. With
the symmetric option, parameter `k` takes node `k`'s value (the bin's own distribution is then symmetric about `u₀`,
and so is its Maxwellian).

Refused as `grid_start` refuses, naming this function: a value not strictly inside the family's bounds, **naming the
parameter, the bin and the value** (nothing is clipped; a bin whose distribution has no spread beyond the kernel's own
has no finite value and is refused by the same rule); a key of another form, or one that names no species of the
trace; a species that does not carry a `GaussianGrid`; a family whose parameter count is not the grid's, or whose
bindings are not to `key`'s log-weights; a family whose variables are not time-varying.
"""
function grid_maxwellian_start(trace::PlasmaTrace, key::AbstractString, family)
    set, idx, NP = _grid_family(trace, key, family, "grid_maxwellian_start")
    h = family.held
    nt = nbins(trace)
    out = [family.names[k] => Vector{Float64}(undef, nt) for k in 1:NP if k != h]
    g = Vector{Float64}(undef, NP)
    for j in 1:nt
        f = _grid_species_set(plasma(trace, j), set)[idx].f
        ubar = Float64(mean_velocity(f))
        s = sqrt(max(2 * Float64(velocity_variance(f)) - f.sigma^2, 0.0))
        for k in 1:NP
            g[k] = -((_grid_node(f, k) - ubar) / s)^2 - log(f.w0[k])
        end
        q = 0
        for k in 1:NP
            k == h && continue
            q += 1
            last(out[q])[j] = _grid_start_value(family.variables[k], j, g[k] - g[h], "grid_maxwellian_start")
        end
    end
    return out
end

# A species key, "electrons.<name>" or "ions.<name>", as its set and name; anything else refused, naming `caller`.
function _grid_key(key::AbstractString, caller::String)
    segs = split(key, '.')
    (length(segs) == 2 && (segs[1] == "electrons" || segs[1] == "ions") && !isempty(segs[2])) || throw(ArgumentError(
        "$caller: \"$key\" is not a species key; expected \"electrons.<name>\" or \"ions.<name>\""))
    return Symbol(segs[1]), String(segs[2])
end

# A start's checks of `family` against `trace` and `key`, shared by both starts, each refusal naming `caller`: the
# species' set, its index in the set, and its grid's parameter count.
function _grid_family(trace::PlasmaTrace, key::AbstractString, family, caller::String)
    set, name = _grid_key(key, caller)
    species = _grid_species_set(trace.template, set)
    idx = findfirst(==(name), species.names)
    idx === nothing && throw(ArgumentError("$caller: \"$key\" names no species of the trace"))
    NP = _grid_start_parameters(species[idx].f, key, caller)
    nf = length(family.names)
    nf == NP || throw(ArgumentError(
        "$caller: \"$key\" carries a grid of $NP parameters, but the family has $nf; it was made for another grid"))
    for k in 1:NP
        want = string(key, ".f.g", k)
        family.bindings[k] == (family.names[k] => want) || throw(ArgumentError(
            "$caller: the family binds \"$(first(family.bindings[k]))\" to \"$(last(family.bindings[k]))\", " *
            "not to \"$want\"; it was made for another species"))
    end
    all(v -> v.time_varying, family.variables) || throw(ArgumentError(
        "$caller: the family's variables are not time-varying; a start per bin is for a time-varying family"))
    return set, idx, NP
end

# A start's value `x` of variable `v` at bin `j`, refused unless strictly inside the variable's bounds.
function _grid_start_value(v, j::Int, x::Float64, caller::String)
    v.lower < x < v.upper || throw(ArgumentError(
        "$caller: \"$(v.name)\" at bin $j is $x once the held parameter's g is subtracted, not strictly inside its " *
        "bounds ($(v.lower), $(v.upper))"))
    return x
end

# The parameter count of a species' GaussianGrid; any other distribution refused, naming `caller` (one method per
# family, never `isa`: conventions.md Section 4).
_grid_start_parameters(::GaussianGrid{FT, Nv, NP}, key, caller) where {FT, Nv, NP} = NP
_grid_start_parameters(f, key, caller) = throw(ArgumentError(
    "$caller: \"$key\" does not carry a GaussianGrid; its distribution is a $(nameof(typeof(f)))"))
