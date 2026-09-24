# The parameter layer (Step 3.1c, chunk 4 of Phase 3): the logit transform, fitted variables,
# named intervals, dependents' topological order, bindings to leaves and effect parameters,
# ParameterSpec, theta's layout and index map, flatten/unflatten. This is the one mechanism that
# makes "any numeric leaf fittable" true (OQ-17): the per-bin kernel never sees a bound name, a
# transform or an expression -- it only ever sees a plain Experiment. Spec D3-D13.

# ------------------------------------------------------------------------------------------
# D3: the logit transform -- the one copy in src/
# ------------------------------------------------------------------------------------------

"""
    _logistic(y::Real) -> Real

The logistic function `s(y) = 1/(1+exp(-y))`, written overflow-safe: `y >= 0` uses
`inv(1+exp(-y))`, `y < 0` uses `exp(y)/(1+exp(y))`, so the argument to `exp` is never positive.
**Verified 2026-09-13** (see `docs/derivations/parameters.md` section 1 for the numbers): the
single formula `1/(1+exp(-y))`, applied unconditionally, does not itself produce `NaN` on this
machine (it saturates to the same `0.0`/`1.0` this branched form does); the formula that does
overflow to `NaN` is the *other* single formula, `exp(y)/(1+exp(y))`, applied unconditionally at
large **positive** `y` (`exp(y)` overflows to `Inf`, and `Inf/(1+Inf)` is `Inf/Inf`). Branching
on the sign of `y` and always calling `exp` on a non-positive argument (this function's actual
form) avoids both failure modes, regardless of which one-formula version a reader has in mind
(PM3). The branch is on the **value** of `y` (`_primal_value`, `src/constants.jl`), which for a
dual changes neither the value nor the derivative (`conventions.md` 5.12).

**This does not make `s(y)` strictly between 0 and 1 for every finite `y`.** Float64 has no
representable value between `1 - eps(1.0)/2` and `1.0`, so `s(y)` is **exactly** `1.0` for any
`y ≳ 37`, and, symmetrically, `exp(y)` itself underflows to exactly `0.0` for `y ≲ -746`, so `s(y)`
is exactly `0.0` there. Both are the correctly-rounded double value, not a computation error, and
no formula avoids them (D3; confirmed against the spec's own PM3/PM4 wording, 2026-09-13 -- see
`docs/derivations/parameters.md`). `_from_unconstrained` therefore only guarantees `a <= x <= b`
and finiteness, never strict interiority, at the extremes of `y`.

Spec D3.
"""
function _logistic(y::Real)
    yv = _primal_value(y)
    return yv >= 0 ? inv(one(y) + exp(-y)) : (e = exp(y); e / (one(e) + e))
end

"""
    _from_unconstrained(y::Real, a::Real, b::Real) -> Real

`x = a + (b - a) * _logistic(y)`, the logit's forward map from the unconstrained `y` an optimizer
works in onto the fitted variable's bounded range `(a, b)`. Generic in the element type of `y` so
a dual flows through (`conventions.md` 5.1). Guarantees `a <= x <= b` and finiteness for every
finite `y` (see `_logistic`'s docstring for why not strict interiority at extreme `y`).

Spec D3.
"""
_from_unconstrained(y::Real, a::Real, b::Real) = a + (b - a) * _logistic(y)

"""
    _to_unconstrained(x::Real, a::Real, b::Real) -> Real

`y = log((x - a) / (b - x))`, the logit's inverse map, `_from_unconstrained`'s inverse on `(a, b)`.
Returns `±Inf` when `x` is on or outside a bound, which is why `FittedVariable` requires `init`
strictly inside its bounds (D39) rather than catching it here.

Spec D3.
"""
_to_unconstrained(x::Real, a::Real, b::Real) = log((x - a) / (b - x))

"""
    _dx_dy(x::Real, a::Real, b::Real) -> Real

`dx/dy = (x - a)(b - x) / (b - a)`, the logit's derivative evaluated **from `x`**, so it needs no
second `exp` call; algebraically identical to `(b-a)*s(y)*(1-s(y))` but this is the form the
Jacobian's chain rule uses (Step 3.2). Vanishes at both bounds -- D13's bound-proximity flag exists
because of this.

Spec D3.
"""
_dx_dy(x::Real, a::Real, b::Real) = (x - a) * (b - x) / (b - a)

# ------------------------------------------------------------------------------------------
# D4: FittedVariable
# ------------------------------------------------------------------------------------------

const _VARIABLE_NAME_RE = r"^[A-Za-z_][A-Za-z0-9_]*$"

"""
    FittedVariable(name; bounds, init, time_varying = false, vary = true, interval = 0)

A scalar or per-bin fitted quantity: finite bounds `(a, b)`, an initial value strictly inside
them, whether it varies over the trace axis, whether it is actually fitted (`vary = false` freezes
it at `init`, removing it from theta but keeping its name usable by dependents and the
regularizer), and which named interval (an index into `ParameterSpec.intervals`, `0` for the
whole axis) it lives on. `name` must match `r"^[A-Za-z_][A-Za-z0-9_]*\$"` (a valid Julia
identifier, since it is also written into dependent expressions).

**A variable carries no unit**: `bounds` and `init` are read in the unit of the key it is later
bound to (`conventions.md` 7.6, D9). All four numeric fields are stored as `Float64`, never `FT`
-- they are user input, read once, never differentiated.

Spec D4.
"""
struct FittedVariable
    name::String
    lower::Float64
    upper::Float64
    init::Float64
    time_varying::Bool
    vary::Bool
    interval::Int
end

function FittedVariable(name::AbstractString; bounds, init::Real, time_varying::Bool=false,
                         vary::Bool=true, interval::Integer=0)
    occursin(_VARIABLE_NAME_RE, name) || throw(ArgumentError(
        "\"$name\" is not a valid variable name; expected $(_VARIABLE_NAME_RE.pattern)"))
    length(bounds) == 2 || throw(ArgumentError(
        "FittedVariable \"$name\": bounds must give exactly [lower, upper]; got $bounds"))
    lower, upper = Float64(bounds[1]), Float64(bounds[2])
    (isfinite(lower) && isfinite(upper)) || throw(ArgumentError(
        "FittedVariable \"$name\": bounds must be finite; got ($lower, $upper)"))
    lower < upper || throw(ArgumentError(
        "FittedVariable \"$name\": lower bound $lower is not below the upper bound $upper"))
    initf = Float64(init)
    (lower < initf < upper) || throw(ArgumentError(
        "FittedVariable \"$name\": init $initf must lie strictly inside the bounds ($lower, $upper)"))
    return FittedVariable(String(name), lower, upper, initf, time_varying, vary, Int(interval))
end

# ------------------------------------------------------------------------------------------
# D5: FitInterval
# ------------------------------------------------------------------------------------------

"""
    FitInterval

A named time window: `lo`/`hi` in SI (the trace axis's own unit, s for a `:t` axis) and `bins`,
the `UnitRange` of axis indices `j` with `lo <= axis[j] <= hi` (inclusive at both ends),
computed at construction. `bins` is a `UnitRange` and not a `Vector` because the axis is strictly
increasing (`PlasmaTrace` guarantees it), so the set of bins in a window is always contiguous --
this is what lets a variable's theta block be a contiguous slice (D11).

Spec D5.
"""
struct FitInterval
    name::String
    lo::Float64
    hi::Float64
    bins::UnitRange{Int}
end

const _INTERVAL_RANGE_KEYS = (t_range_ns=:t_ns, t_range_s=:t_s)

"""
    FitInterval(name::AbstractString, axis::AbstractVector{<:Real}; t_range_ns | t_range_s)

Build a `FitInterval` against `axis` (SI, e.g. `trace.axis`), reading its boundary keyword through
`ACCEPTED_UNITS`'s `t` row exactly as `PlasmaTrace`'s own axis keyword is (D5). Exactly one of
`t_range_ns`/`t_range_s` must be given, a two-element `[lo, hi]`. `name = ""` builds an anonymous
interval (the inline `t_range_ns = [a, b]` form D5 describes for a variable, dependent or
regularizer term). An empty window (no bin of `axis` inside `[lo, hi]`) is a loud error (D39).

Not given an explicit Julia signature by the spec (D5 describes only the fields and the deck
form); this constructor's shape -- `axis` positional, the boundary keyword mirroring
`PlasmaTrace`'s -- is Level 3's choice under rule 11 zero, flagged in the chunk 4 report.
"""
function FitInterval(name::AbstractString, axis::AbstractVector{<:Real}; kwargs...)
    present = Tuple(k for k in keys(_INTERVAL_RANGE_KEYS) if haskey(kwargs, k))
    if length(present) != 1
        throw(ArgumentError(
            "FitInterval \"$name\": expected exactly one of $(keys(_INTERVAL_RANGE_KEYS)); got $present"))
    end
    key = present[1]
    rng = kwargs[key]
    length(rng) == 2 || throw(ArgumentError(
        "FitInterval \"$name\": \"$key\" must give exactly [lo, hi]; got $(collect(rng))"))
    factor = Float64(getfield(getfield(ACCEPTED_UNITS, :t), _INTERVAL_RANGE_KEYS[key]))
    lo, hi = Float64(rng[1]) * factor, Float64(rng[2]) * factor
    first_j = searchsortedfirst(axis, lo)
    last_j = searchsortedlast(axis, hi)
    first_j <= last_j || throw(ArgumentError(
        "FitInterval \"$name\" [$lo, $hi] contains no bin of the axis"))
    return FitInterval(String(name), lo, hi, first_j:last_j)
end

_interval_bins(v::FittedVariable, intervals::Vector{FitInterval}, nt::Int) =
    v.interval == 0 ? (1:nt) : intervals[v.interval].bins

# ------------------------------------------------------------------------------------------
# D8: DependentVariable, topological order, and the per-bin value vector's name map
# ------------------------------------------------------------------------------------------

"""
    DependentVariable{N<:AbstractExprNode}

A named expression over fitted variables, other dependents and `t`: `name`, the deck string
`text` (verbatim, for the effective deck and the result), and the parsed, resolved `node`.

Spec D8.
"""
struct DependentVariable{N<:AbstractExprNode}
    name::String
    text::String
    node::N
end

_referenced_slots!(acc::Set{Int}, ::ConstNode) = nothing
_referenced_slots!(acc::Set{Int}, ::TimeNode) = nothing
_referenced_slots!(acc::Set{Int}, n::SlotNode) = push!(acc, n.slot)
_referenced_slots!(acc::Set{Int}, n::CallNode) = foreach(a -> _referenced_slots!(acc, a), n.args)
function _referenced_slots!(acc::Set{Int}, n::CompareNode)
    _referenced_slots!(acc, n.lhs)
    _referenced_slots!(acc, n.rhs)
    return nothing
end
function _referenced_slots!(acc::Set{Int}, n::IfElseNode)
    _referenced_slots!(acc, n.cond)
    _referenced_slots!(acc, n.iftrue)
    _referenced_slots!(acc, n.iffalse)
    return nothing
end

"""
    _topological_dependents(variables, dependents_input, nvar) -> (deps::Tuple, names::Dict{String,Int})

Parse every `name => text` pair of `dependents_input` and order them topologically (D8): a
dependent may reference only fitted variables, `t`, and dependents that come **before** it in the
returned tuple. Two passes are needed because `_parse_expression` resolves a name to a slot
**index**, so the dependency graph among dependent names (needed to find that order) must be built
from a first, provisional parse before the final slot numbers -- and hence the final trees -- can
be assigned. A cycle is a loud error naming it (D39, PM25); so is a name declared both as a
variable and a dependent.

Spec D8.
"""
function _topological_dependents(variables::Vector{FittedVariable}, dependents_input, nvar::Int)
    dep_pairs = [(String(nm), String(text)) for (nm, text) in dependents_input]
    ndep = length(dep_pairs)
    dep_names = [p[1] for p in dep_pairs]
    dep_text = Dict{String,String}(dep_pairs)

    var_names = Dict{String,Int}(v.name => i for (i, v) in enumerate(variables))
    for nm in dep_names
        haskey(var_names, nm) && throw(ArgumentError(
            "\"$nm\" is declared both as a fitted variable and as a dependent"))
    end

    provisional = copy(var_names)
    for (k, nm) in enumerate(dep_names)
        provisional[nm] = nvar + k
    end

    edges = [Int[] for _ in 1:ndep]
    for (k, nm) in enumerate(dep_names)
        node = _parse_expression(dep_text[nm], provisional; context=nm)
        refs = Set{Int}()
        _referenced_slots!(refs, node)
        for r in refs
            r > nvar && push!(edges[k], r - nvar)
        end
    end

    order = Int[]
    state = zeros(Int, ndep)   # 0 unvisited, 1 in progress, 2 done
    stack = Int[]
    function visit(k::Int)
        state[k] == 2 && return nothing
        if state[k] == 1
            i0 = findfirst(==(k), stack)
            cyc = stack[i0:end]
            chain = join((dep_names[c] for c in cyc), " -> ")
            throw(ArgumentError("dependent \"$(dep_names[k])\" is cyclic: $chain -> $(dep_names[k])"))
        end
        state[k] = 1
        push!(stack, k)
        for r in edges[k]
            visit(r)
        end
        pop!(stack)
        state[k] = 2
        push!(order, k)
        return nothing
    end
    for k in 1:ndep
        visit(k)
    end

    final_names = copy(var_names)
    for (pos, k) in enumerate(order)
        final_names[dep_names[k]] = nvar + pos
    end

    final_deps = Vector{DependentVariable}(undef, ndep)
    for (pos, k) in enumerate(order)
        nm = dep_names[k]
        node = _parse_expression(dep_text[nm], final_names; context=nm)
        final_deps[pos] = DependentVariable(nm, dep_text[nm], node)
    end

    return Tuple(final_deps), final_names
end

"""
    _time_varying_transitively(variables, dependents, nvar) -> Vector{Bool}

For each dependent (in the topological order `dependents` is stored in), whether it references,
directly or through another dependent, any time-varying fitted variable. Used by the
scalar-instrument rule (D9): binding a *dependent* to an instrument parameter is refused when this
is true, exactly as binding a time-varying *variable* to one directly is. A single forward pass
suffices because a dependent only references earlier slots.
"""
function _time_varying_transitively(variables::Vector{FittedVariable}, dependents::Tuple, nvar::Int)
    ndep = length(dependents)
    tv = falses(ndep)
    for (k, dep) in enumerate(dependents)
        refs = Set{Int}()
        _referenced_slots!(refs, dep.node)
        flag = false
        for r in refs
            flag |= r <= nvar ? variables[r].time_varying : tv[r-nvar]
        end
        tv[k] = flag
    end
    return tv
end

# ------------------------------------------------------------------------------------------
# D9: Binding
# ------------------------------------------------------------------------------------------

"""
    Binding

One variable or dependent (`name`, and its `slot` in the per-bin value vector) plugged into one
target: a plasma leaf (`kind === :leaf`, `path`/`row`) or an effect parameter (`kind === :effect`,
`effect`/`field`/`element`, `element = 0` for a scalar parameter). `factor` converts the variable's
own value to SI (`1.0`, `unit_key = ""` for a dimensionless target); every binding of one variable
must share one `factor` (D9's tightening of `conventions.md` 7.6). The two kinds share a struct so
that `ParameterSpec.bindings` can be one homogeneous `Vector`; the fields the other kind does not
use hold harmless placeholders (`0`, `Symbol("")`, an unused dummy `LeafPath`), never read.

`channel` (Step 4.A.3a, D32) is meaningful only for an `:effect` binding: which channel's
`Experiment` the binding's `effect` index counts into (`_resolve_bindings` writes `0` for a
`:leaf` binding, since every channel shares the one plasma). The ten-argument constructor below
sets `channel = 1`, so a hand-built `Binding` from before this chunk keeps compiling.

Spec D9, D32.
"""
struct Binding
    name::String
    slot::Int
    kind::Symbol
    path::LeafPath
    row::Int
    effect::Int
    field::Symbol
    element::Int
    factor::Float64
    unit_key::String
    channel::Int
end

Binding(name, slot, kind, path, row, effect, field, element, factor, unit_key) =
    Binding(name, slot, kind, path, row, effect, field, element, factor, unit_key, 1)

const _DUMMY_LEAF_PATH = LeafPath(["_"])

function _check_unit_factor!(name_factor::Dict{String,Tuple{Float64,String}}, name::AbstractString,
                              factor::Float64, unit_key::AbstractString)
    if haskey(name_factor, name)
        existing_factor, existing_key = name_factor[name]
        if existing_factor != factor
            existing_label = isempty(existing_key) ? "a dimensionless target" : existing_key
            new_label = isempty(unit_key) ? "a dimensionless target" : unit_key
            throw(ArgumentError(
                "\"$name\" is bound through both $existing_label and $new_label, which have different units"))
        end
    else
        name_factor[name] = (Float64(factor), String(unit_key))
    end
    return nothing
end

"""
    _resolve_bindings(bindings_input, variables, names, nvar, dep_tv, template, trace,
                      effects_by_channel) -> Vector{Binding}

Resolve every `name => target` pair into a `Binding`, checking, in order: the name is a declared
variable or dependent; a probe/collection key is refused naming `WavelengthCalibration`; the target leaf or
effect parameter exists; the one-conversion-factor rule (D9); the scalar-instrument rule for an
`:effect` target (D9); the one-variable-per-leaf rule for a `:leaf` target (D9). `row` is left `0`
for a leaf the trace does not yet tabulate; `ParameterSpec`'s constructor fills it in after the
trace rebuild.

`effects_by_channel` is a `Tuple` of each channel's `Experiment.effects` (length 1 outside a joint
spec). `target` may be (Step 4.A.3a, D32): a leaf key string (unchanged, `channel = 0`, D32: every
channel shares the plasma); one of the three original `Tuple` forms `(effect, field)`,
`(effect, field, element)` or `(effect, field, element, factor, unit_key)` (unchanged, `channel =
1`, resolved against `effects_by_channel[1]`); or a `NamedTuple` with required `channel`, `effect`
and `field` and optional `element` (`0`), `factor` (`1.0`) and `unit_key` (`""`), resolved against
`effects_by_channel[target.channel]` -- an unknown key, a missing required key, or a channel
outside `1:length(effects_by_channel)` is a loud error naming the variable (PM29's three errors).

Spec D9, D32.
"""
function _resolve_bindings(bindings_input, variables::Vector{FittedVariable},
                            names::Dict{String,Int}, nvar::Int, dep_tv::AbstractVector{Bool},
                            template::Plasma, trace::PlasmaTrace, effects_by_channel::Tuple)
    nchannels = length(effects_by_channel)
    pairs_list = collect(bindings_input)
    bindings = Vector{Binding}(undef, length(pairs_list))
    name_factor = Dict{String,Tuple{Float64,String}}()
    leaf_owner = Dict{String,String}()

    for (i, kv) in enumerate(pairs_list)
        name, target = kv
        name = String(name)
        haskey(names, name) || throw(ArgumentError(
            "binding names \"$name\", which is not a fitted variable or a dependent"))
        slot = names[name]

        if target isa AbstractString
            key = String(target)
            first_seg = split(key, '.')[1]
            (first_seg == "probe" || first_seg == "collection") && throw(ArgumentError(
                "\"$name\" cannot be bound to \"$key\": the probe and the collection are fixed. " *
                "Bind the wavelength axis through a WavelengthCalibration effect in the " *
                "instrument chain instead."))
            si_path, factor = _leaf_si_path(key)
            path = LeafPath(si_path)
            _resolve_leaf(template, path)
            unit_key = factor == 1.0 ? "" : String(split(key, '.')[end])
            _check_unit_factor!(name_factor, name, factor, unit_key)
            if haskey(leaf_owner, si_path) && leaf_owner[si_path] != name
                throw(ArgumentError("both \"$(leaf_owner[si_path])\" and \"$name\" are bound to \"$si_path\""))
            end
            leaf_owner[si_path] = name
            row = _leaf_row(trace, path)
            bindings[i] = Binding(name, slot, :leaf, path, row, 0, Symbol(""), 0, factor, unit_key, 0)
        elseif target isa NamedTuple
            allowed_keys = (:channel, :effect, :field, :element, :factor, :unit_key)
            for k in keys(target)
                k in allowed_keys || throw(ArgumentError(
                    "binding \"$name\": unknown key :$k in the named-tuple target; valid keys are $allowed_keys"))
            end
            (haskey(target, :channel) && haskey(target, :effect) && haskey(target, :field)) || throw(ArgumentError(
                "binding \"$name\": the named-tuple target requires :channel, :effect and :field"))
            channel = Int(target.channel)
            (1 <= channel <= nchannels) || throw(ArgumentError(
                "binding \"$name\": channel $channel is out of range 1:$nchannels"))
            effect_idx = Int(target.effect)
            field = Symbol(target.field)
            element = haskey(target, :element) ? Int(target.element) : 0
            factor = haskey(target, :factor) ? Float64(target.factor) : 1.0
            unit_key = haskey(target, :unit_key) ? String(target.unit_key) : ""
            effects = effects_by_channel[channel]
            (1 <= effect_idx <= length(effects)) || throw(ArgumentError(
                "binding \"$name\": effect index $effect_idx is out of range 1:$(length(effects)) for channel $channel"))
            e = effects[effect_idx]
            pnt = parameters(e)
            isempty(pnt) && throw(ArgumentError(
                "\"$name\" cannot be bound to $(typeof(e).name.name): it has no fittable parameters"))
            haskey(pnt, field) || throw(ArgumentError(
                "\"$name\": $(typeof(e).name.name) has no parameter \"$field\"; valid names are $(keys(pnt))"))
            is_tv = slot <= nvar ? variables[slot].time_varying : dep_tv[slot-nvar]
            is_tv && throw(ArgumentError(
                "\"$name\" is time-varying and cannot be bound to an instrument parameter"))
            _check_unit_factor!(name_factor, name, factor, unit_key)
            bindings[i] = Binding(name, slot, :effect, _DUMMY_LEAF_PATH, 0, effect_idx, field, element, factor, unit_key, channel)
        else
            effect_idx = Int(target[1])
            field = Symbol(target[2])
            element = length(target) >= 3 ? Int(target[3]) : 0
            factor = length(target) >= 4 ? Float64(target[4]) : 1.0
            unit_key = length(target) >= 5 ? String(target[5]) : ""
            effects = effects_by_channel[1]
            (1 <= effect_idx <= length(effects)) || throw(ArgumentError(
                "binding \"$name\": effect index $effect_idx is out of range 1:$(length(effects))"))
            e = effects[effect_idx]
            pnt = parameters(e)
            isempty(pnt) && throw(ArgumentError(
                "\"$name\" cannot be bound to $(typeof(e).name.name): it has no fittable parameters"))
            haskey(pnt, field) || throw(ArgumentError(
                "\"$name\": $(typeof(e).name.name) has no parameter \"$field\"; valid names are $(keys(pnt))"))
            is_tv = slot <= nvar ? variables[slot].time_varying : dep_tv[slot-nvar]
            is_tv && throw(ArgumentError(
                "\"$name\" is time-varying and cannot be bound to an instrument parameter"))
            _check_unit_factor!(name_factor, name, factor, unit_key)
            bindings[i] = Binding(name, slot, :effect, _DUMMY_LEAF_PATH, 0, effect_idx, field, element, factor, unit_key, 1)
        end
    end
    return bindings
end

# ------------------------------------------------------------------------------------------
# D10, D11: ParameterSpec and theta's index map
# ------------------------------------------------------------------------------------------

"""
    ParameterSpec{FT<:Real,D<:Tuple,TR<:PlasmaTrace{FT}}

The parameter layer over one `Experiment`: the fitted variables, the dependents (a `Tuple`, in
topological order, so the per-bin evaluation loop unrolls and stays type-stable -- the same style
`Experiment` carries its effects in), the named intervals, the resolved bindings, the name-to-slot
map of the per-bin value vector, and theta's layout (D11): `offsets`/`widths` per variable,
`bin_entries`/`bin_slots` per bin, `global_entries`/`global_slots`, and `ntheta`. `trace` is the
**rebuilt** `PlasmaTrace` (D9): the original leaves in their original order, then every bound leaf
the original trace did not tabulate, in binding order, each new row holding the template's own
constant value until the first `unflatten`.

Spec D10, D11.
"""
struct ParameterSpec{FT<:Real,D<:Tuple,TR<:PlasmaTrace{FT}}
    variables::Vector{FittedVariable}
    dependents::D
    intervals::Vector{FitInterval}
    bindings::Vector{Binding}
    slots::Dict{String,Int}
    nslots::Int
    axis::Vector{FT}
    axis_boundary::Vector{Float64}
    axis_unit::String
    offsets::Vector{Int}
    widths::Vector{Int}
    bin_entries::Vector{Vector{Int}}
    bin_slots::Vector{Vector{Int}}
    global_entries::Vector{Int}
    global_slots::Vector{Int}
    ntheta::Int
    trace::TR
end

const _AXIS_UNIT_KEYS = (ns=:t_ns, s=:t_s)

function _rebuild_trace_with_bindings(trace::PlasmaTrace, bindings::Vector{Binding})
    new_leaf_paths = LeafPath[]
    new_leaf_names = String[]
    for b in bindings
        b.kind === :leaf || continue
        b.row == 0 || continue
        b.path in new_leaf_paths && continue
        push!(new_leaf_paths, b.path)
        push!(new_leaf_names, string(b.path))
    end
    isempty(new_leaf_paths) && return trace, bindings

    leaves = Vector{Pair{String,Vector{Float64}}}()
    nt = nbins(trace)
    for (i, path) in enumerate(trace.paths)
        row = collect(Float64, @view trace.values[i, :])
        push!(leaves, string(path) => row)
    end
    for path in new_leaf_paths
        ref = _resolve_leaf(trace.template, path)
        v0 = Float64(_leaf_value(trace.template, ref))
        push!(leaves, string(path) => fill(v0, nt))
    end

    new_trace = PlasmaTrace(trace.template; axis_name=trace.axis_name, t_s=trace.axis, leaves=leaves)

    row_of = Dict{String,Int}(string(p) => i for (i, p) in enumerate(new_trace.paths))
    new_bindings = Vector{Binding}(undef, length(bindings))
    for (i, b) in enumerate(bindings)
        if b.kind === :leaf
            new_bindings[i] = Binding(b.name, b.slot, b.kind, b.path, row_of[string(b.path)],
                b.effect, b.field, b.element, b.factor, b.unit_key, b.channel)
        else
            new_bindings[i] = b
        end
    end
    return new_trace, new_bindings
end

function _build_index_map(variables::Vector{FittedVariable}, intervals::Vector{FitInterval}, nt::Int)
    nvar = length(variables)
    offsets = zeros(Int, nvar)
    widths = zeros(Int, nvar)
    pos = 1
    for (i, v) in enumerate(variables)
        v.vary || continue
        w = v.time_varying ? length(_interval_bins(v, intervals, nt)) : 1
        offsets[i] = pos
        widths[i] = w
        pos += w
    end
    ntheta = pos - 1

    bin_entries = [Int[] for _ in 1:nt]
    bin_slots = [Int[] for _ in 1:nt]
    global_entries = Int[]
    global_slots = Int[]
    for (i, v) in enumerate(variables)
        v.vary || continue
        if v.time_varying
            bins = _interval_bins(v, intervals, nt)
            for (k, j) in enumerate(bins)
                push!(bin_entries[j], offsets[i] + (k - 1))
                push!(bin_slots[j], i)
            end
        else
            push!(global_entries, offsets[i])
            push!(global_slots, i)
        end
    end
    return offsets, widths, bin_entries, bin_slots, global_entries, global_slots, ntheta
end

"""
    _slot_defined_at(slot, j, variables, nvar, intervals) -> Bool

Whether `vals[slot]` (the per-bin value vector) holds a meaningful value at bin `j`: always true
for a dependent's slot (`slot > nvar`, evaluated at every bin, D12 step 3) and for a scalar or
frozen variable; for a varying trace variable, only within its own interval.
"""
function _slot_defined_at(slot::Int, j::Int, variables::Vector{FittedVariable}, nvar::Int,
                           intervals::Vector{FitInterval})
    slot > nvar && return true
    v = variables[slot]
    (!v.vary || !v.time_varying || v.interval == 0) && return true
    return j in intervals[v.interval].bins
end

"""
    _check_interval_reads(dep, variables, intervals, axis_boundary, nvar, nt)

D39's construction-time check: a dependent that reads a trace variable outside its interval at a
bin where the dependent is itself evaluated (every bin, D12 step 3) is a loud error. Checked by a
dry run: evaluate `dep.node` once per bin with every slot at the placeholder `1.0` and `t` at the
bin's **real** axis value, using a recording vector that logs which slots the lazy `IfElseNode`
evaluation (D6) actually reads; for every recorded read of a variable slot with a non-whole-axis
interval, the bin must lie in that interval. This validates the canonical guard idiom exactly
(`"t < 3.4 ? u_pre : u_post"`, since `t` is real at each bin) but cannot validate a branch whose
*condition* itself depends on a fitted variable's value (the branch actually taken then depends on
the fit, not on this placeholder) -- flagged as a known limitation in the chunk 4 report.
"""
struct _ReadRecorder <: AbstractVector{Float64}
    n::Int
    reads::Vector{Int}
end
Base.size(r::_ReadRecorder) = (r.n,)
function Base.getindex(r::_ReadRecorder, i::Int)
    push!(r.reads, i)
    return 1.0
end

function _check_interval_reads(dep::DependentVariable, variables::Vector{FittedVariable},
                                intervals::Vector{FitInterval}, axis_boundary::Vector{Float64},
                                nvar::Int, ndep::Int, nt::Int)
    for j in 1:nt
        recorder = _ReadRecorder(nvar + ndep, Int[])
        _eval(dep.node, recorder, axis_boundary[j])
        for slot in recorder.reads
            slot <= nvar || continue
            v = variables[slot]
            v.interval == 0 && continue
            bins = intervals[v.interval].bins
            j in bins || throw(ArgumentError(
                "dependent \"$(dep.name)\" reads \"$(v.name)\", which is not defined on bin $j " *
                "(\"$(v.name)\" is defined on bins $bins)"))
        end
    end
    return nothing
end

"""
    _probe_initial_values(variables, dependents, axis_boundary, nt)

D7 item 4's **value** half (PM26): evaluate every dependent at the real `init` values of every
fitted variable, at every bin, requiring a finite `Real`. This is what catches a genuine
`DomainError` (`log` of something negative at the actual initial trace) or a typo that parses to
something meaningless -- the structural probe in `_parse_expression` (`_probe_construction`)
deliberately cannot, since it only has synthetic placeholder values (D7). Every dependent is
evaluated at every bin of the **whole axis** (D12 step 3 does the same, unconditionally; a
dependent has no interval of its own), not just "its interval" in any variable-specific sense.
"""
function _probe_initial_values(variables::Vector{FittedVariable}, dependents::Tuple,
                                axis_boundary::Vector{Float64}, nt::Int)
    nvar = length(variables)
    vals = Vector{Float64}(undef, nvar + length(dependents))
    for i in 1:nvar
        vals[i] = variables[i].init
    end
    for j in 1:nt
        for (k, dep) in enumerate(dependents)
            t = axis_boundary[j]
            local v
            try
                v = _eval(dep.node, vals, t)
            catch e
                throw(ArgumentError(
                    "dependent \"$(dep.name)\" could not be evaluated at its initial values: " *
                    "$(sprint(showerror, e)) at bin $j (t = $t) (\"$(dep.text)\")"))
            end
            (v isa Real && isfinite(v)) || throw(ArgumentError(
                "dependent \"$(dep.name)\" could not be evaluated at its initial values: " *
                "got $(repr(v)) at bin $j (t = $t) (\"$(dep.text)\")"))
            vals[nvar+k] = v
        end
    end
    return nothing
end

"""
    ParameterSpec(expts::Tuple{Vararg{Experiment}}; variables, dependents = (), intervals = (),
                  bindings, axis_unit::Symbol = :ns)
    ParameterSpec(expt::Experiment; variables, dependents = (), intervals = (), bindings,
                  axis_unit::Symbol = :ns)

Build the parameter layer over one or more channels' experiments (Step 4.A.3a, D32); the
single-`Experiment` form is `ParameterSpec((expt,); ...)`, bitwise. `variables` an iterable of
`FittedVariable`; `dependents` an iterable of `name => expression_text` pairs; `intervals` an
iterable of `FitInterval`; `bindings` an iterable of `name => target` pairs, `target` a boundary
leaf key string (`"electrons.e.T_eV"`, meaning channel 0 -- every channel shares the plasma), one
of the original `(effect_index, field)` / `(effect_index, field, element)` /
`(effect_index, field, element, factor, unit_key)` tuples (meaning channel 1), or a `NamedTuple`
`(channel, effect, field[, element, factor, unit_key])` naming any channel (D32). `axis_unit`
(`:ns` or `:s`) is the unit `t` means inside a dependent and the regularizer's derivatives are
taken in (D8) -- **not given an explicit keyword by D10's abbreviated constructor signature;
Level 3's addition, flagged under rule 11 zero** (chunk 4 report), since `PlasmaTrace` itself does
not retain which boundary keyword built its axis.

Every `expts[c].trace` must be `=== expts[1].trace` (D32: one trace object, OQ-33) -- a loud error
naming `c` otherwise.

Order of construction (D10): validate the variables; resolve the intervals (already done by the
caller, D5); parse the dependents and topologically sort them (D7, D8), checking interval reads
(D39); resolve every binding against the shared trace and each channel's own effects (D9, D32);
rebuild the trace with the union of leaves (D9); build the index map (D11); probe-evaluate the
dependents at the initial values over every bin (D7 item 4, PM26).

Spec D10, D32.
"""
function ParameterSpec(expts::Tuple{Vararg{Experiment}}; variables, dependents=(), intervals=(),
                        bindings, axis_unit::Symbol=:ns)
    isempty(expts) && throw(ArgumentError("ParameterSpec: at least one experiment is required"))
    haskey(_AXIS_UNIT_KEYS, axis_unit) || throw(ArgumentError(
        "ParameterSpec: axis_unit must be :ns or :s; got :$axis_unit"))
    factor = Float64(getfield(getfield(ACCEPTED_UNITS, :t), _AXIS_UNIT_KEYS[axis_unit]))

    trace0 = expts[1].trace
    for c in 2:length(expts)
        expts[c].trace === trace0 || throw(ArgumentError(
            "ParameterSpec: experiment $c's trace is not === experiment 1's; every channel must " *
            "share one trace object"))
    end

    vars = collect(FittedVariable, variables)
    ivals = collect(FitInterval, intervals)
    nvar = length(vars)
    nt = nbins(trace0)

    seen_names = Set{String}()
    for v in vars
        v.name in seen_names && throw(ArgumentError("\"$(v.name)\" is declared more than once as a fitted variable"))
        push!(seen_names, v.name)
        v.interval == 0 || (1 <= v.interval <= length(ivals)) || throw(ArgumentError(
            "FittedVariable \"$(v.name)\": interval index $(v.interval) is out of range 1:$(length(ivals))"))
    end

    axis_boundary = Float64.(trace0.axis) ./ factor

    dep_tuple, names = _topological_dependents(vars, dependents, nvar)
    ndep = length(dep_tuple)
    for dep in dep_tuple
        _check_interval_reads(dep, vars, ivals, axis_boundary, nvar, ndep, nt)
    end
    dep_tv = _time_varying_transitively(vars, dep_tuple, nvar)

    resolved_bindings = _resolve_bindings(bindings, vars, names, nvar, dep_tv, trace0.template,
        trace0, Tuple(e.effects for e in expts))

    trace, resolved_bindings = _rebuild_trace_with_bindings(trace0, resolved_bindings)

    offsets, widths, bin_entries, bin_slots, global_entries, global_slots, ntheta =
        _build_index_map(vars, ivals, nt)

    _probe_initial_values(vars, dep_tuple, axis_boundary, nt)

    FT = eltype(trace)
    return ParameterSpec(vars, dep_tuple, ivals, resolved_bindings, names,
        nvar + ndep, Vector{FT}(trace.axis), axis_boundary, String(axis_unit), offsets, widths,
        bin_entries, bin_slots, global_entries, global_slots, ntheta, trace)
end

function ParameterSpec(expt::Experiment; variables, dependents=(), intervals=(), bindings,
                        axis_unit::Symbol=:ns)
    return ParameterSpec((expt,); variables=variables, dependents=dependents, intervals=intervals,
        bindings=bindings, axis_unit=axis_unit)
end

# ------------------------------------------------------------------------------------------
# D13: the bound-proximity flag
# ------------------------------------------------------------------------------------------

"""
    BOUND_PROXIMITY_TOL

The relative distance from a bound (of the bound's own range `b - a`) within which a fitted value
is flagged as stalled against it (D13): `1e-3`. At that distance `dx/dy ≈ 1e-3(b-a)`, so a unit
step in `y` (the scale L-BFGS works in, since the logit makes every `y` O(1)) moves `x` by one
part in a thousand of its range -- the optimizer is effectively stalled there.
"""
const BOUND_PROXIMITY_TOL = 1e-3

"""
    SATURATION_FRACTION

The fraction of `theta`'s coordinates that must sit within `BOUND_PROXIMITY_TOL` of a bound before
a gradient-norm stop (`Optim.g_converged`) is relabelled `:saturated` rather than trusted as a
real minimum (D31, Step 3.C.2). Measured contrast (`dev/reports/phase3C_report.md`, chunk 5):
`0.0` at a healthy start on every cut-down problem, `1.0` at EPW's saturated (2-iteration) stop,
and `0.1438`/`0.01563` at multi-flow's and shock's own cap-terminated, non-saturated ends -- any
threshold in `(0.15, 1.0]` separates the saturated case from the non-saturated ones, and `0.5`
sits in the middle of that gap with a factor of 3 of margin on each side.

**This is a detector, not a tuning knob.** Narrowing it to make a test pass is a defect: a fit
that lands between 0.15 and 0.5 is a finding to report, not a reason to move this constant.

Spec D31.
"""
const SATURATION_FRACTION = 0.5

"""
    _near_bound(x::Real, a::Real, b::Real) -> Bool

Whether `x` is within `BOUND_PROXIMITY_TOL` of the range `b - a` of either bound. A start value
**on** a bound is a construction-time error (D4, D39); this is a post-fit **flag**, never a
refusal (D13).
"""
_near_bound(x::Real, a::Real, b::Real) = min(x - a, b - x) < BOUND_PROXIMITY_TOL * (b - a)

# ------------------------------------------------------------------------------------------
# D12: flatten and unflatten
# ------------------------------------------------------------------------------------------

"""
    flatten(pspec::ParameterSpec) -> Vector{Float64}

Theta at every `vary = true` variable's `init`: for a trace variable, the same `init` at every bin
of its interval. This is the start vector every fit begins from (D12).
"""
function flatten(pspec::ParameterSpec)
    theta = Vector{Float64}(undef, pspec.ntheta)
    nt = nbins(pspec.trace)
    for (i, v) in enumerate(pspec.variables)
        v.vary || continue
        y = _to_unconstrained(v.init, v.lower, v.upper)
        if v.time_varying
            bins = _interval_bins(v, pspec.intervals, nt)
            for k in 1:length(bins)
                theta[pspec.offsets[i]+(k-1)] = y
            end
        else
            theta[pspec.offsets[i]] = y
        end
    end
    return theta
end

"""
    flatten(pspec::ParameterSpec, values) -> Vector{Float64}

Theta built from given `x` values rather than each variable's `init`: `values` an iterable of
`name => x` pairs, `x` a scalar (used at every bin of the variable's interval) or, for a trace
variable, a per-bin `AbstractVector` indexed by the trace's own bin number. A variable not named in
`values` falls back to its `init`. **Not given an explicit shape by D10's abbreviated docstring;
Level 3's choice, flagged under rule 11 zero** (chunk 4 report).
"""
function flatten(pspec::ParameterSpec, values)
    valdict = Dict{String,Any}(String(k) => v for (k, v) in values)
    theta = Vector{Float64}(undef, pspec.ntheta)
    nt = nbins(pspec.trace)
    for (i, v) in enumerate(pspec.variables)
        v.vary || continue
        if v.time_varying
            bins = _interval_bins(v, pspec.intervals, nt)
            xv = get(valdict, v.name, v.init)
            for (k, j) in enumerate(bins)
                x = xv isa AbstractVector ? xv[j] : xv
                theta[pspec.offsets[i]+(k-1)] = _to_unconstrained(Float64(x), v.lower, v.upper)
            end
        else
            x = get(valdict, v.name, v.init)
            theta[pspec.offsets[i]] = _to_unconstrained(Float64(x), v.lower, v.upper)
        end
    end
    return theta
end

function _widen_trace(trace::PlasmaTrace{FT0}, ::Type{T}) where {FT0,T}
    FT0 === T && return trace
    new_template = _widen_plasma(trace.template, T)
    new_values = Matrix{T}(trace.values)
    refs = [_resolve_leaf(new_template, p) for p in trace.paths]
    return _finish_trace(new_template, trace.axis_name, T.(trace.axis), trace.paths, refs, new_values)
end

"""
    _eval_dependents!(vals::AbstractVector, deps::Tuple, t, slot::Int)

Evaluate every dependent of `deps` (a `Tuple` of `DependentVariable`, in topological order) against
`vals` and the bin's time `t`, writing each result into `vals[slot], vals[slot+1], ...`. Written by
recursion over the `Tuple` with `Base.tail`, in `measure.jl`'s `_accumulate!` style, **not** a
`for` loop over `enumerate(deps)`: `deps` is heterogeneous (each dependent's `node` is a distinct
concrete `AbstractExprNode` type, D6), so a plain loop cannot be specialized for every element and
falls back to a dynamic dispatch per dependent inside `unflatten`'s per-bin, per-sweep hot path
(D8's own instruction; the precedent this codebase already has). The recursion terminates on
`Tuple{}` and unrolls completely at compile time for any fixed number of dependents.

Spec D8.
"""
_eval_dependents!(vals::AbstractVector, ::Tuple{}, t, slot::Int) = nothing
function _eval_dependents!(vals::AbstractVector, deps::Tuple, t, slot::Int)
    dep = first(deps)
    vals[slot] = _eval(dep.node, vals, t)
    return _eval_dependents!(vals, Base.tail(deps), t, slot + 1)
end

function _rebuild_effect(e::AbstractInstrumentEffect, bs::Vector{Binding}, vals::AbstractVector)
    old = parameters(e)
    ks = keys(old)
    newvals = map(ks) do fname
        fval = old[fname]
        matching = [b for b in bs if b.field === fname]
        if isempty(matching)
            fval
        elseif fval isa Tuple
            ntuple(length(fval)) do ii
                m = findfirst(b -> b.element == ii, matching)
                m === nothing ? fval[ii] : vals[matching[m].slot] * matching[m].factor
            end
        else
            b = only(matching)
            vals[b.slot] * b.factor
        end
    end
    return reconstruct(e, NamedTuple{ks}(newvals))
end

"""
    unflatten(expt::Experiment, pspec::ParameterSpec, theta::AbstractVector; channel::Int = 1) -> Experiment

Undo the transform, evaluate the dependents bin by bin, write the trace table's rows and each
bound effect through `reconstruct`, and return the resulting `Experiment` (D12).

**`channel` (Step 4.A.3a, D33) selects which channel's effects `expt` is rebuilt against**: leaf
bindings are applied exactly as today (every channel shares the plasma, D32), but only `:effect`
bindings with `b.channel == channel` rebuild `expt`'s effects -- so a variable bound only into
another channel never touches this one's, silently and by construction. `channel = 1` alone for a
single-channel fit. **Every `unflatten(problem.experiment, problem.pspec, ...)` call on a
`FitProblem` must pass `channel = problem.channel`** (Pitfall 23); the calls in `src/` are listed
in the chunk 6 report.

**`expt` must be built from `pspec.trace`** -- the same `PlasmaTrace` object `ParameterSpec`'s
constructor produced (including any leaves it added), typically `Experiment(; trace = pspec.trace,
probe = ..., collection = ..., effects = ..., prefactor = ...)`, called once before the first
`unflatten` -- so that the in-place leaf write below (`pspec.trace.values`, which `expt.trace`
`===`) is visible through `expt` too. This precondition is implied but not stated as such by D12's
own wording ("the same Experiment object ... whose trace is the very trace whose values were just
written"); Level 3 flags it explicitly here since nothing else states it (chunk 4 report).

`T = eltype(theta)` may be a `ForwardDiff.Dual`; `unflatten` then allocates a widened `Matrix{T}`
trace and a new `Experiment` (the one case that is not `===` its input regardless of bindings).
`unflatten` is not in the per-bin kernel and may allocate; `plasma(trace, j)` and `_bin_kernel!`
afterwards still must not (TR14, XP13).

Spec D12.
"""
function unflatten(expt::Experiment, pspec::ParameterSpec, theta::AbstractVector; channel::Int=1)
    length(theta) == pspec.ntheta || throw(ArgumentError(
        "unflatten: theta has length $(length(theta)), but this ParameterSpec has ntheta = $(pspec.ntheta)"))
    T = eltype(theta)
    nvar = length(pspec.variables)
    nt = nbins(pspec.trace)
    vals = Vector{T}(undef, pspec.nslots)

    work_trace = _widen_trace(pspec.trace, T)

    scalar_x = Vector{T}(undef, nvar)
    for i in 1:nvar
        v = pspec.variables[i]
        v.time_varying && continue
        scalar_x[i] = v.vary ? _from_unconstrained(theta[pspec.offsets[i]], v.lower, v.upper) :
                      convert(T, v.init)
    end

    for j in 1:nt
        for i in 1:nvar
            v = pspec.variables[i]
            if !v.vary
                vals[i] = convert(T, v.init)
            elseif !v.time_varying
                vals[i] = scalar_x[i]
            else
                vals[i] = T(NaN)
            end
        end
        for (idx, slot) in zip(pspec.bin_entries[j], pspec.bin_slots[j])
            v = pspec.variables[slot]
            vals[slot] = _from_unconstrained(theta[idx], v.lower, v.upper)
        end
        tj = convert(T, pspec.axis_boundary[j])
        _eval_dependents!(vals, pspec.dependents, tj, nvar + 1)
        for b in pspec.bindings
            b.kind === :leaf || continue
            _slot_defined_at(b.slot, j, pspec.variables, nvar, pspec.intervals) || continue
            work_trace.values[b.row, j] = vals[b.slot] * b.factor
        end
    end

    has_effect_binding = any(b -> b.kind === :effect && b.channel == channel, pspec.bindings)
    widened = work_trace !== pspec.trace

    if !has_effect_binding && !widened
        return expt
    end

    new_effects = ntuple(length(expt.effects)) do ei
        e = expt.effects[ei]
        bs = Binding[b for b in pspec.bindings if b.kind === :effect && b.effect == ei && b.channel == channel]
        isempty(bs) ? e : _rebuild_effect(e, bs, vals)
    end

    return Experiment(; trace=work_trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=new_effects, prefactor=expt.prefactor)
end

# --- show (spec D40) ---

function Base.show(io::IO, pspec::ParameterSpec)
    print(io, "ParameterSpec($(length(pspec.variables)) variables, " *
              "$(length(pspec.dependents)) dependents, ntheta = $(pspec.ntheta), " *
              "$(nbins(pspec.trace)) bins)")
end

function Base.show(io::IO, ::MIME"text/plain", pspec::ParameterSpec)
    println(io, "ParameterSpec")
    println(io, "  variables ($(length(pspec.variables))):")
    for v in pspec.variables
        kind = v.time_varying ? "trace" : "scalar"
        status = v.vary ? "fitted" : "frozen at $(v.init)"
        println(io, "    $(v.name): $kind, [$(v.lower), $(v.upper)], $status")
    end
    println(io, "  dependents ($(length(pspec.dependents))):")
    for dep in pspec.dependents
        println(io, "    $(dep.name) = \"$(dep.text)\"")
    end
    @printf(io, "  ntheta = %d over %d bins\n", pspec.ntheta, nbins(pspec.trace))
end
