# The regularizer: composable terms along the trace axis, each a divided-difference operator
# applied to a reference-subtracted trace and THEN normalized row by row, contributing extra
# residual rows with a closed-form Jacobian (Step 3.3, chunk 7 of Phase 3; normalization-after-
# operator revision, chunk 7 remediation, 2026-09-13). Spec D22-D25, OQ-19.
#
# Three things this file must get right, each with its own trap in the chunk prompt: the
# operators differentiate with respect to TIME (pspec.axis_boundary), never bin index (D23); the
# rows are extra data for the residual vector, not a separate Hessian, so R excludes the global
# weight Lambda and the rows' Jacobian is written closed-form rather than autodifferentiated
# (D25); and each row's normalization divisor N_r is evaluated at that row's own ANCHOR bin --
# `D2`'s centred-stencil middle, `D1`'s and `I`'s own base bin, exactly the bin `_operator_coeffs`
# calls `k` -- not at every bin of the term's window (D24/D25 revised 2026-09-13: `relative` is no
# longer degenerate at the default zero reference, since the divisor is the trace's own value, not
# `(x-ref)/x`).
#
# Phase G.3 (OQ-70, spec D3-D8) adds the second concrete term, VelocityRegularizerTerm, which
# differences an ordered list of names across velocity at each time bin (its section is the last
# before `show`). The walkers below read only a term's rows, weight and normalization and call its
# own _term_raw / _term_closed_form, so a Regularizer holds terms of either type.

import ForwardDiff

# ------------------------------------------------------------------------------------------
# D22: the term struct, the Regularizer, and their construction
# ------------------------------------------------------------------------------------------

"""
    AbstractRegularizerTerm

The regularizer term interface. `RegularizerTerm` (D22) differences one name along the time axis:
it is parameterized by its `operator` field rather than by its own type, so that `operator = :I`
(a ridge penalty) and `operator = :D2`/`:D1` are the same struct. `VelocityRegularizerTerm`
(Phase G.3, [OQ-70](../../dev/open_questions.md#oq-70)) differences an ordered list of names across
velocity at each time bin. A concrete term has the fields `rows`, `weight`, `normalization` and
`interval`, and methods of `_term_raw` (its rows), `_term_closed_form` (its rows and their Jacobian
entries), `_with_rows` (the term with its row range in a `Regularizer`) and `_term_line` (its line
in a `Regularizer`'s listing); nothing else of it is read outside its own code, and the two types
are told apart by those methods, never by `isa`.

Spec D22; Phase G.3 spec D7.
"""
abstract type AbstractRegularizerTerm end

"""
    RegularizerTerm <: AbstractRegularizerTerm

One regularizer term: `select`/`slots` name the fitted variables or dependents it acts on (the
group `"time_varying"` is expanded to every time-varying **fitted** variable at construction, not
stored as a literal group name); `operator` is `:D2`, `:D1` or `:I` (D23); `normalization` is
`:absolute`, `:initial`, `:relative` or `:unit` (D24); `scale` is the `absolute` mode's fixed
divisor, `0.0` meaning "use the selected variable's own bound width" (D24; a dependent has no
default and construction refuses one); `weight` is the term's own scalar `w_term`; `interval` is
an index into `ParameterSpec.intervals`, `0` for the whole axis; `reference` is the per-bin
reference trace, one entry per bin of the term's own window, all zero when off (D24); `rows` is
this term's row range within the whole `Regularizer` (D25) -- filled with a **local** `1:n` range
by the outer constructor below and corrected to the term's true global offset by `Regularizer`'s
own constructor when more than one term is combined.

Spec D22.
"""
struct RegularizerTerm <: AbstractRegularizerTerm
    select::Vector{String}
    slots::Vector{Int}
    operator::Symbol
    normalization::Symbol
    scale::Float64
    weight::Float64
    interval::Int
    reference::Vector{Float64}
    rows::UnitRange{Int}
end

"""
    Regularizer{T<:Tuple}

A list of terms (a `Tuple`, so the row assembly in `residuals.jl`/`jacobian.jl`
unrolls, exactly as `Experiment.effects` and `ParameterSpec.dependents` are), the global weight
`weight` (Lambda), and `nrows`, the total row count every term's `rows` range partitions. The terms
are `RegularizerTerm`s and, since Phase G.3, `VelocityRegularizerTerm`s, in the caller's order; a
regularizer of time terms alone has the type it had before.

Spec D22; Phase G.3 spec D7.
"""
struct Regularizer{T<:Tuple}
    terms::T
    weight::Float64
    nrows::Int
end

function _operator_row_count(op::Symbol, n::Int)
    if op === :D2
        return n - 2
    elseif op === :D1
        return n - 1
    elseif op === :I
        return n
    else
        throw(ArgumentError("RegularizerTerm: operator must be :D2, :D1 or :I; got :$op"))
    end
end

_operator_min_bins(op::Symbol) = op === :D2 ? 3 : op === :D1 ? 2 : 1

"""
    _resolve_term_interval(pspec, interval; who = "RegularizerTerm") -> Int

`interval` as a term's `interval` field stores it: `0` (default, or explicitly given) for the
whole axis, an `Int` taken as a 1-based index into `pspec.intervals` directly, or a name looked up
against `iv.name` for every `iv in pspec.intervals`. Inline `t_range_ns` ranges are the deck
layer's job (chunk 12), which registers an anonymous `FitInterval` into `pspec.intervals` before
building a `RegularizerTerm` that names it -- not given an explicit shape by D22; Level 3's choice
under rule 11 zero, flagged in the chunk 7 report. `who` names the term type in a refusal; shared
with `VelocityRegularizerTerm` since Phase G.3 (spec D8).
"""
function _resolve_term_interval(pspec::ParameterSpec, interval; who::AbstractString="RegularizerTerm")
    if interval isa Integer
        iv = Int(interval)
        (iv == 0 || (1 <= iv <= length(pspec.intervals))) || throw(ArgumentError(
            "$who: interval index $iv is out of range 1:$(length(pspec.intervals))"))
        return iv
    elseif interval isa AbstractString
        idx = findfirst(iv -> iv.name == interval, pspec.intervals)
        idx === nothing && throw(ArgumentError("$who: no interval named \"$interval\""))
        return idx
    else
        throw(ArgumentError(
            "$who: interval must be an Int index into pspec.intervals, an interval " *
            "name, or 0 for the whole axis; got $(typeof(interval))"))
    end
end

# A term's bins: the whole axis, or its interval's. Every concrete term has an `interval` field.
_term_bins(term::AbstractRegularizerTerm, pspec::ParameterSpec) =
    term.interval == 0 ? (1:nbins(pspec.trace)) : pspec.intervals[term.interval].bins

"""
    _check_term_inside_variables(pspec, selected, slots, intv, bins, who)

Phase 4.C spec D13, shared by both term types since Phase G.3 (spec D8): a term's bins must lie
inside every selected variable's own interval, since a time-varying fitted variable on an interval
is `NaN` in `_value_table` outside it. Refuses the first name that is undefined at one of `bins`,
naming it and both intervals; `who` names the term type. The check is `_slot_defined_at`'s own.
"""
function _check_term_inside_variables(pspec::ParameterSpec, selected, slots, intv::Int, bins,
                                      who::AbstractString)
    nvar = length(pspec.variables)
    for (nm, slot) in zip(selected, slots)
        all(j -> _slot_defined_at(slot, j, pspec.variables, nvar, pspec.intervals), bins) && continue
        v = pspec.variables[slot]
        term_iv = intv == 0 ? "the whole axis" : "interval \"$(pspec.intervals[intv].name)\""
        throw(ArgumentError(
            "$who: the term selecting \"$nm\" is on $term_iv (bins $bins), but " *
            "\"$nm\" is defined only on its own interval \"$(pspec.intervals[v.interval].name)\" " *
            "(bins $(pspec.intervals[v.interval].bins)); a term's interval must lie inside every " *
            "selected variable's own interval"))
    end
    return nothing
end

"""
    RegularizerTerm(pspec::ParameterSpec; select, operator::Symbol, normalization::Symbol,
                     scale::Real = 0.0, weight::Real = 1.0, interval = 0, reference = 0.0)

Resolve one term against `pspec` (D22): `select` a name, a `Vector` of names, or `"time_varying"`
(expanded to every time-varying fitted variable, D22); `operator` one of `:D2`, `:D1`, `:I`;
`normalization` one of `:absolute`, `:initial`, `:relative`, `:unit`; `scale` the `absolute`
mode's fixed divisor (`0.0`, the default, means "use the selected variable's own bound width" --
refused at construction for a selected dependent, D24); `interval` as `_resolve_term_interval`
resolves it; `reference` a number (constant over the term's bins), a vector (length = the term's
own bin count), or a string parsed as a `t`-only expression (no variables in scope, D22, so a
reference naming a fitted variable is a loud error).

`rows` is filled `1:n` here (this term considered alone); `Regularizer`'s own constructor
corrects it to the term's true offset when combining more than one term (D25).

Not given an explicit Julia signature by D22, which describes only the struct fields and the
deck form; this shape (a `ParameterSpec`-resolving keyword constructor, mirroring
`FitInterval`'s and `ParameterSpec`'s own) is Level 3's choice under rule 11 zero, flagged in the
chunk 7 report.

**A term's interval must lie inside every selected variable's own interval** (Phase 4.C spec D13):
a time-varying fitted variable on an interval is undefined outside it (`_value_table` holds `NaN`
there), so a term reading it there would make `R` `NaN` with no error; the constructor refuses it,
naming the variable and both intervals. The check is on bins, not times, and is `_slot_defined_at`'s
own: a dependent, a scalar or frozen variable, or a variable on the whole axis is defined at every
bin and never refused.

Spec D22.
"""
function RegularizerTerm(pspec::ParameterSpec; select, operator::Symbol, normalization::Symbol,
                          scale::Real=0.0, weight::Real=1.0, interval=0, reference=0.0)
    nvar = length(pspec.variables)
    var_names = Dict{String,Int}(v.name => i for (i, v) in enumerate(pspec.variables))
    dep_names = Dict{String,Int}(d.name => nvar + i for (i, d) in enumerate(pspec.dependents))
    valid_names = sort(vcat(collect(keys(var_names)), collect(keys(dep_names)), ["time_varying"]))

    selected = if select isa AbstractString && select == "time_varying"
        [v.name for v in pspec.variables if v.time_varying]
    elseif select isa AbstractString
        [String(select)]
    else
        collect(String, select)
    end
    isempty(selected) && throw(ArgumentError(
        "RegularizerTerm: select = \"time_varying\" resolves to no variables"))

    slots = Int[]
    for nm in selected
        if haskey(var_names, nm)
            push!(slots, var_names[nm])
        elseif haskey(dep_names, nm)
            push!(slots, dep_names[nm])
        else
            throw(ArgumentError(
                "\"$nm\" is not a fitted variable, a dependent, or \"time_varying\"; " *
                "valid names are $valid_names"))
        end
    end

    operator in (:D2, :D1, :I) || throw(ArgumentError(
        "RegularizerTerm: operator must be :D2, :D1 or :I; got :$operator"))
    normalization in (:absolute, :initial, :relative, :unit) || throw(ArgumentError(
        "RegularizerTerm: normalization must be :absolute, :initial, :relative or :unit; " *
        "got :$normalization"))

    intv = _resolve_term_interval(pspec, interval)
    bins = intv == 0 ? (1:nbins(pspec.trace)) : pspec.intervals[intv].bins
    n = length(bins)
    needed = _operator_min_bins(operator)
    n >= needed || throw(ArgumentError(
        "RegularizerTerm: interval has $n bins, but operator :$operator needs at least $needed"))

    # Phase 4.C spec D13: a term's bins must lie inside every selected variable's own interval.
    _check_term_inside_variables(pspec, selected, slots, intv, bins, "RegularizerTerm")

    scale_f = Float64(scale)
    if normalization === :absolute && scale_f == 0.0
        for (nm, slot) in zip(selected, slots)
            slot > nvar && throw(ArgumentError(
                "RegularizerTerm: selects the dependent \"$nm\", so `scale` must be given explicitly"))
        end
    end

    refvec = if reference isa AbstractString
        node = _parse_expression(reference, Dict{String,Int}(); context="regularizer reference")
        [Float64(_eval(node, Float64[], pspec.axis_boundary[j])) for j in bins]
    elseif reference isa AbstractVector
        length(reference) == n || throw(ArgumentError(
            "RegularizerTerm: reference has length $(length(reference)), but the term has $n bins"))
        Float64.(reference)
    elseif reference isa Real
        fill(Float64(reference), n)
    else
        throw(ArgumentError(
            "RegularizerTerm: reference must be a number, a vector, or an expression string in t"))
    end

    nrows_term = sum(_operator_row_count(operator, n) for _ in selected)
    return RegularizerTerm(selected, slots, operator, normalization, scale_f, Float64(weight),
                            intv, refvec, 1:nrows_term)
end

"""
    Regularizer(terms; weight::Real = 1.0) -> Regularizer

Combine already-resolved terms (each built with `RegularizerTerm(pspec; ...)` or
`VelocityRegularizerTerm(pspec; ...)` against the same `ParameterSpec`, in any mix and order) into
one `Regularizer`, fixing up every term's `rows` to its true offset within the combined row
numbering, in the order given (D25: "`Regularizer.nrows` is the total and `RegularizerTerm.rows` is
each term's range within it"). `weight` is the global Lambda, default `1.0`. The terms are stored
as a `Tuple` of their own types, so a regularizer of time terms alone is what it was before Phase
G.3 (spec D7).

Not given an explicit Julia signature by D22 (only the struct fields and the deck form); this
shape is Level 3's choice under rule 11 zero, flagged in the chunk 7 report.

Spec D22, D25; Phase G.3 spec D7.
"""
function Regularizer(terms; weight::Real=1.0)
    ts = collect(AbstractRegularizerTerm, terms)
    isempty(ts) && throw(ArgumentError("Regularizer: at least one term is required"))
    fixed = Vector{AbstractRegularizerTerm}(undef, length(ts))
    pos = 1
    for (i, t) in enumerate(ts)
        nr = length(t.rows)
        fixed[i] = _with_rows(t, pos:(pos + nr - 1))
        pos += nr
    end
    return Regularizer(Tuple(fixed), Float64(weight), pos - 1)
end

# The term with its row range in a Regularizer (spec D25; one method per term type, Phase G.3 spec D7).
_with_rows(t::RegularizerTerm, rows::UnitRange{Int}) =
    RegularizerTerm(t.select, t.slots, t.operator, t.normalization, t.scale, t.weight, t.interval,
                    t.reference, rows)

# ------------------------------------------------------------------------------------------
# D23: the operators, as divided differences on the axis
# ------------------------------------------------------------------------------------------

"""
    _operator_rows(op::Symbol, u::AbstractVector, t::AbstractVector) -> Vector

The operator's row values (D23), generic in `eltype(u)` so a dual `u` differentiates (used by the
brute-force oracle path, `_regularizer_raw_rows`/`_full_residuals`, never by the closed-form
Jacobian): `D2` is the divided second difference (exact for a quadratic on any grid, RG1), `D1`
the divided first difference, `I` the trace itself. `t` is `pspec.axis_boundary` restricted to
the term's own bins -- the axis keyword's own unit, never the bin index (D23, Pitfall 11).
"""
function _operator_rows(op::Symbol, u::AbstractVector, t::AbstractVector)
    n = length(u)
    if op === :D2
        return [2 * ((u[k+1] - u[k]) / (t[k+1] - t[k]) - (u[k] - u[k-1]) / (t[k] - t[k-1])) /
                (t[k+1] - t[k-1]) for k in 2:(n-1)]
    elseif op === :D1
        return [(u[k+1] - u[k]) / (t[k+1] - t[k]) for k in 1:(n-1)]
    else
        return collect(u)
    end
end

"""
    _operator_anchor(op::Symbol, r::Int) -> Int

The **anchor bin** `a(r)` of row `r` (1-based within one selected name's block), the single local
bin a row's normalization divisor `N_r` is evaluated at (D24 revised 2026-09-13): `D2`'s centred
three-point stencil `r, r+1, r+2` anchors on its middle, `r + 1`; `D1`'s two-point stencil `r,
r+1` anchors on its base point, `r`; `I`'s one-bin row anchors on itself, `r`. This is the same
`k`/`r` `_operator_coeffs` uses to build its own coefficient table, factored out so the two never
drift apart -- there is exactly one indexing convention, not two.
"""
_operator_anchor(op::Symbol, r::Int) = op === :D2 ? r + 1 : r

"""
    _operator_coeffs(op::Symbol, r::Int, t::AbstractVector) -> Tuple

The operator's own **linear** coefficients for its `r`-th row (1-based within one selected
name's block), as `(local_bin_index, coefficient)` pairs of `∂row_r/∂u[local_bin_index]` --
D23's formulas differentiated by hand, independent of `u` itself (the operator is linear in the
reference-subtracted trace). This is the analytic half of D25's closed-form row Jacobian; it is
never autodifferentiated, and it is written independently of `_operator_rows`'s value formula so
that RG12's ForwardDiff oracle (which differentiates `_operator_rows` through `_term_raw`) is a
genuine check of this table, not a comparison of one formula against itself (`conventions.md`
8.2). `k = _operator_anchor(op, r)` is the row's own anchor bin (D24/D25).
"""
function _operator_coeffs(op::Symbol, r::Int, t::AbstractVector)
    k = _operator_anchor(op, r)
    if op === :D2
        d1 = t[k] - t[k-1]
        d2 = t[k+1] - t[k]
        d3 = t[k+1] - t[k-1]
        return ((k - 1, 2 / (d1 * d3)), (k, -2 * (1 / d2 + 1 / d1) / d3), (k + 1, 2 / (d2 * d3)))
    elseif op === :D1
        d = t[k+1] - t[k]
        return ((k, -1 / d), (k + 1, 1 / d))
    else
        return ((k, 1.0),)
    end
end

# ------------------------------------------------------------------------------------------
# The per-bin value table (generic, differentiable): the raw trace value of every slot (fitted
# variable or dependent) at every bin, mirroring unflatten's own per-bin computation for its own
# purpose. ParameterSpec does not retain this after construction (D7 item 4's probe validates and
# discards it), so it is recomputed here -- flagged as a design point in the chunk 7 report.
# ------------------------------------------------------------------------------------------

function _value_table(pspec::ParameterSpec, theta::AbstractVector{T}) where {T<:Real}
    nvar = length(pspec.variables)
    nt = nbins(pspec.trace)
    vals = Matrix{T}(undef, pspec.nslots, nt)
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
                vals[i, j] = convert(T, _frozen_value(v, j, pspec.intervals))
            elseif !v.time_varying
                vals[i, j] = scalar_x[i]
            else
                vals[i, j] = T(NaN)
            end
        end
        for (idx, slot) in zip(pspec.bin_entries[j], pspec.bin_slots[j])
            v = pspec.variables[slot]
            vals[slot, j] = _from_unconstrained(theta[idx], v.lower, v.upper)
        end
        tj = convert(T, pspec.axis_boundary[j])
        col = view(vals, :, j)
        _eval_dependents!(col, pspec.dependents, tj, nvar + 1)
    end
    return vals
end

_needs_initial(reg::Regularizer) = any(t -> t.normalization === :initial, reg.terms)
_needs_initial(term::RegularizerTerm) = term.normalization === :initial

function _effective_scale(term::RegularizerTerm, pspec::ParameterSpec, slot::Int)
    term.scale != 0.0 && return term.scale
    nvar = length(pspec.variables)
    slot <= nvar || throw(ArgumentError(
        "RegularizerTerm: selects a dependent, so `scale` must be given explicitly"))
    v = pspec.variables[slot]
    return v.upper - v.lower
end

# ------------------------------------------------------------------------------------------
# D24/D25: the raw rows, generic (the AD-differentiable value path -- the oracle side of rule
# 8.2), and the residual rows _full_residuals stacks under the data residuals.
# ------------------------------------------------------------------------------------------

function _term_raw(term::RegularizerTerm, pspec::ParameterSpec, vtable::AbstractMatrix{T},
                    x0table::Union{Nothing,AbstractMatrix{Float64}}) where {T<:Real}
    bins = _term_bins(term, pspec)
    t = view(pspec.axis_boundary, bins)
    n = length(bins)
    out = Vector{T}(undef, length(term.rows))
    pos = 0
    for (ni, slot) in enumerate(term.slots)
        scale = term.normalization === :absolute ? _effective_scale(term, pspec, slot) : 0.0
        u = Vector{T}(undef, n)
        for k in 1:n
            j = bins[k]
            u[k] = vtable[slot, j] - term.reference[k]
        end
        rows = _operator_rows(term.operator, u, t)
        nr = length(rows)
        for r in 1:nr
            a = _operator_anchor(term.operator, r)
            j = bins[a]
            if term.normalization === :absolute
                rows[r] = rows[r] / scale
            elseif term.normalization === :initial
                x0 = x0table[slot, j]
                x0 == 0 && throw(ArgumentError(
                    "\"$(term.select[ni])\" has a starting value of zero at bin $j"))
                rows[r] = rows[r] / x0
            elseif term.normalization === :relative
                xa = vtable[slot, j]
                _primal_value(xa) == 0 && throw(ArgumentError(
                    "\"$(term.select[ni])\" has a value of zero at bin $j"))
                rows[r] = rows[r] / xa
            end
        end
        out[pos+1:pos+nr] .= rows
        pos += nr
    end
    return out
end

function _term_raw(term::RegularizerTerm, pspec::ParameterSpec, theta::AbstractVector{T}) where {T<:Real}
    vtable = _value_table(pspec, theta)
    x0table = _needs_initial(term) ? _value_table(pspec, flatten(pspec)) : nothing
    return _term_raw(term, pspec, vtable, x0table)
end

"""
    _regularizer_raw_rows(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector) -> Vector

Every term's raw `op(u - ref) / N` rows, `N` the row's own anchor-bin divisor (D23/D24, revised
2026-09-13), **unweighted** (no `sqrt(Lambda * w_term)`), stacked in `reg`'s row order. Generic in
`eltype(theta)`: this is the function RG12's `ForwardDiff.jacobian` oracle differentiates by way
of `_regularizer_residual_rows`/`_full_residuals`, never the closed-form path (`conventions.md`
8.2).
"""
function _regularizer_raw_rows(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector{T}) where {T<:Real}
    vtable = _value_table(pspec, theta)
    x0table = _needs_initial(reg) ? _value_table(pspec, flatten(pspec)) : nothing
    out = Vector{T}(undef, reg.nrows)
    for term in reg.terms
        out[term.rows] .= _term_raw(term, pspec, vtable, x0table)
    end
    return out
end

"""
    _regularizer_residual_rows(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector) -> Vector

The regularizer's contribution to `r_full` (D25): `rho_row = sqrt(Lambda * w_term) * raw_row`,
generic in `eltype(theta)` so `_full_residuals` (`residuals.jl`) differentiates through it for
RG12/RG13/RG19's oracle.
"""
function _regularizer_residual_rows(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector{T}) where {T<:Real}
    vtable = _value_table(pspec, theta)
    x0table = _needs_initial(reg) ? _value_table(pspec, flatten(pspec)) : nothing
    out = Vector{T}(undef, reg.nrows)
    for term in reg.terms
        raw = _term_raw(term, pspec, vtable, x0table)
        out[term.rows] .= sqrt(reg.weight * term.weight) .* raw
    end
    return out
end

"""
    _regularizer_value(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector) -> Real

`R = sum_terms w_term * ||op(u - ref) / N||^2` (D25, revised 2026-09-13), **Lambda-free**. Built
from the same AD-generic raw-row path `_regularizer_residual_rows` stacks (evaluating a value
needs no closed form; only the row *Jacobian* is required to be closed-form, D25).
"""
function _regularizer_value(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector{T}) where {T<:Real}
    raw = _regularizer_raw_rows(reg, pspec, theta)
    Rval = zero(T)
    for term in reg.terms
        Rval += term.weight * sum(abs2, view(raw, term.rows))
    end
    return Rval
end

# ------------------------------------------------------------------------------------------
# D25: the closed-form Jacobian of the rows -- never autodifferentiated. For a selected fitted
# variable the chain to theta is the trivial one-entry dx/dy (D3); for a selected dependent, the
# chain rule continues through the expression tree, obtained by a small forward-mode dual pass
# seeded on every fitted entry active at that bin (the same technique D18 uses for the data
# Jacobian, applied here to the dependent's own value rather than to the kernel).
# ------------------------------------------------------------------------------------------

"""
    _dependent_value_and_grad(pspec, theta, j, slot) -> (value, active::Vector{Int}, grad::Vector{Float64})

`vals[slot]` at bin `j` and its partials with respect to every theta index active at bin `j`
(`pspec.bin_entries[j]`, this bin's own time-varying entries, and `pspec.global_entries`, every
scalar fitted variable) -- obtained by `ForwardDiff.gradient` of a closure that reconstructs
`vals[1:nvar]` from a length-`length(active)` seed vector and runs `_eval_dependents!` (D8's own
evaluator, reused unchanged). This is legitimate, D18-style AD through the expression tree's
chain rule, not the row-Jacobian autodifferentiation D25 forbids -- see `_operator_coeffs`'s
docstring for that half.
"""
function _dependent_value_and_grad(pspec::ParameterSpec, theta::AbstractVector{Float64}, j::Int, slot::Int)
    nvar = length(pspec.variables)
    active = vcat(pspec.bin_entries[j], pspec.global_entries)
    active_slot = vcat(pspec.bin_slots[j], pspec.global_slots)
    slot_pos = Dict{Int,Int}(s => k for (k, s) in enumerate(active_slot))

    f = function (z::AbstractVector{T}) where {T}
        vals = Vector{T}(undef, pspec.nslots)
        for i in 1:nvar
            v = pspec.variables[i]
            if !v.vary
                vals[i] = convert(T, _frozen_value(v, j, pspec.intervals))
            else
                pos = get(slot_pos, i, 0)
                vals[i] = pos == 0 ? T(NaN) : _from_unconstrained(z[pos], v.lower, v.upper)
            end
        end
        tj = convert(T, pspec.axis_boundary[j])
        _eval_dependents!(vals, pspec.dependents, tj, nvar + 1)
        return vals[slot]
    end

    isempty(active) && return f(Float64[]), Int[], Float64[]
    z0 = Float64[theta[i] for i in active]
    return f(z0), active, ForwardDiff.gradient(f, z0)
end

"""
    _slot_value_and_grad(pspec, theta, j, slot) -> (value, active::Vector{Int}, grad::Vector{Float64})

`vals[slot]` at bin `j` and its partials with respect to theta (D25). For a fitted variable
(`slot <= nvar`) this is closed-form and trivial: a frozen variable contributes no theta entry at
all (its value its `init`, or its held series' entry at `j`, Phase 4.I spec D7); a scalar
variable's one entry is `pspec.offsets[slot]`; a time-varying variable's entry at
bin `j` is looked up in `pspec.bin_entries[j]`/`bin_slots[j]`, or the variable is undefined at
this bin (outside its own interval) and contributes nothing. For a dependent (`slot > nvar`) this
defers to `_dependent_value_and_grad`.
"""
function _slot_value_and_grad(pspec::ParameterSpec, theta::AbstractVector{Float64}, j::Int, slot::Int)
    nvar = length(pspec.variables)
    if slot > nvar
        return _dependent_value_and_grad(pspec, theta, j, slot)
    end
    v = pspec.variables[slot]
    if !v.vary
        return _frozen_value(v, j, pspec.intervals), Int[], Float64[]
    elseif !v.time_varying
        idx = pspec.offsets[slot]
        x = _from_unconstrained(theta[idx], v.lower, v.upper)
        return x, [idx], [_dx_dy(x, v.lower, v.upper)]
    else
        pos = findfirst(==(slot), pspec.bin_slots[j])
        pos === nothing && return NaN, Int[], Float64[]
        idx = pspec.bin_entries[j][pos]
        x = _from_unconstrained(theta[idx], v.lower, v.upper)
        return x, [idx], [_dx_dy(x, v.lower, v.upper)]
    end
end

"""
    _term_closed_form(term, pspec, theta, vtable, x0table, cache) -> (raw::Vector{Float64}, entries)

One term's raw row values and the **unweighted** closed-form Jacobian entries
`(local_row, theta_index, ∂raw_row/∂theta_index)` (D25, revised 2026-09-13): each row is
`raw_row = g_r / N_r`, `g_r` the operator applied to the reference-subtracted trace
(`_operator_rows`, D23) and `N_r` the row's normalization divisor evaluated at its own anchor bin
`a(r)` (`_operator_anchor`, D24's table -- `s` for `absolute`, the starting trace `x0` at `a(r)`
for `initial`, the current trace `x` at `a(r)` for `relative`, `1` for `unit`). Its derivative with
respect to `x_j` (bin `j`'s raw trace value) is `∂g_r/∂x_j / N_r`, from `_operator_coeffs`'s linear
coefficient at bin `j`, **plus**, for `relative` alone and only when `j` is the row's own anchor
`a(r)`, the quotient-rule term `-g_r / N_r^2` (`N_r` depends on `x_j` only there). Each is then
chained to theta by `_slot_value_and_grad`. `cache` memoizes `_slot_value_and_grad` by `(slot,
bin)` across every row and every term of one `Regularizer` evaluation, since neighbouring
`D2`/`D1` rows and overlapping terms reuse the same bins (D25: "cache it rather than running a
second one").
"""
function _term_closed_form(term::RegularizerTerm, pspec::ParameterSpec, theta::AbstractVector{Float64},
                            vtable::AbstractMatrix{Float64}, x0table::Union{Nothing,AbstractMatrix{Float64}},
                            cache::Dict{Tuple{Int,Int},Tuple{Float64,Vector{Int},Vector{Float64}}})
    bins = _term_bins(term, pspec)
    t = view(pspec.axis_boundary, bins)
    n = length(bins)
    raw = Vector{Float64}(undef, length(term.rows))
    entries = Tuple{Int,Int,Float64}[]
    pos = 0
    for (ni, slot) in enumerate(term.slots)
        scale = term.normalization === :absolute ? _effective_scale(term, pspec, slot) : 0.0
        u = Vector{Float64}(undef, n)
        for k in 1:n
            j = bins[k]
            u[k] = vtable[slot, j] - term.reference[k]
        end
        rows_here = _operator_rows(term.operator, u, t)
        nr = length(rows_here)
        for r in 1:nr
            a = _operator_anchor(term.operator, r)
            j_anchor = bins[a]
            g = rows_here[r]
            if term.normalization === :absolute
                Nr = scale
            elseif term.normalization === :initial
                Nr = x0table[slot, j_anchor]
                Nr == 0 && throw(ArgumentError(
                    "\"$(term.select[ni])\" has a starting value of zero at bin $j_anchor"))
            elseif term.normalization === :relative
                Nr = vtable[slot, j_anchor]
                Nr == 0 && throw(ArgumentError(
                    "\"$(term.select[ni])\" has a value of zero at bin $j_anchor"))
            else
                Nr = 1.0
            end
            raw[pos+r] = g / Nr

            for (klocal, opcoeff) in _operator_coeffs(term.operator, r, t)
                j = bins[klocal]
                key = (slot, j)
                sv = get(cache, key, nothing)
                if sv === nothing
                    val, active, grad = _slot_value_and_grad(pspec, theta, j, slot)
                    cache[key] = (val, active, grad)
                else
                    val, active, grad = sv
                end
                c = opcoeff / Nr
                if term.normalization === :relative && klocal == a
                    c -= g / Nr^2
                end
                for (m, aidx) in enumerate(active)
                    push!(entries, (pos + r, aidx, c * grad[m]))
                end
            end
        end
        pos += nr
    end
    return raw, entries
end

"""
    _regularizer_closed_form(reg, pspec, theta::AbstractVector{Float64}) -> (raw_all, rows, cols, coeffs)

Every term's raw rows and the **weighted** (`sqrt(Lambda * w_term)`) closed-form Jacobian
entries, as parallel `rows`/`cols`/`coeffs` vectors (`rows` in `reg`'s global row numbering,
`cols` a theta index): `∂rho_row/∂theta_col = coeffs[k]`. This is what `jacobian.jl` stores on
`FitJacobian` and what `jvp!`/`jtvp!` consume for the regularizer's rows (D25).
"""
function _regularizer_closed_form(reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector{Float64})
    vtable = _value_table(pspec, theta)
    x0table = _needs_initial(reg) ? _value_table(pspec, flatten(pspec)) : nothing
    cache = Dict{Tuple{Int,Int},Tuple{Float64,Vector{Int},Vector{Float64}}}()
    raw_all = Vector{Float64}(undef, reg.nrows)
    rows = Int[]
    cols = Int[]
    coeffs = Float64[]
    for term in reg.terms
        raw, entries = _term_closed_form(term, pspec, theta, vtable, x0table, cache)
        raw_all[term.rows] .= raw
        s = sqrt(reg.weight * term.weight)
        for (lr, tidx, c) in entries
            push!(rows, term.rows[lr])
            push!(cols, tidx)
            push!(coeffs, s * c)
        end
    end
    return raw_all, rows, cols, coeffs
end

"""
    _regularizer_gradient!(g, reg::Regularizer, pspec::ParameterSpec, theta::AbstractVector{Float64}) -> g

`∇R` (D25), **Lambda-free**: `g[k] = sum over rows of 2 * w_term * raw_row * ∂raw_row/∂theta_k`,
built from the same closed-form entries `_regularizer_closed_form` computes for the residual
rows, but with the term weight alone (no `sqrt(Lambda * w_term)`), matching `R`'s own
Lambda-freedom.
"""
function _regularizer_gradient!(g::AbstractVector, reg::Regularizer, pspec::ParameterSpec,
                                 theta::AbstractVector{Float64})
    fill!(g, 0.0)
    vtable = _value_table(pspec, theta)
    x0table = _needs_initial(reg) ? _value_table(pspec, flatten(pspec)) : nothing
    cache = Dict{Tuple{Int,Int},Tuple{Float64,Vector{Int},Vector{Float64}}}()
    for term in reg.terms
        raw, entries = _term_closed_form(term, pspec, theta, vtable, x0table, cache)
        for (lr, tidx, c) in entries
            g[tidx] += 2 * term.weight * raw[lr] * c
        end
    end
    return g
end

# ------------------------------------------------------------------------------------------
# Phase G.3 (OQ-70; spec D3-D8): the regularizer along velocity. An ordered list of names, each at
# the velocity of its node in m/s, differenced across velocity at every bin of the term's interval.
# One operator, :D3; one normalization, :unit. Its rows enter the walkers above through its own
# _term_raw and _term_closed_form.
# ------------------------------------------------------------------------------------------

"""
    VelocityRegularizerTerm <: AbstractRegularizerTerm
    VelocityRegularizerTerm(pspec::ParameterSpec; select, velocities, operator = :D3,
                            normalization = :unit, weight = 1.0, interval = 0)
    VelocityRegularizerTerm(pspec::ParameterSpec; select, grid, operator = :D3,
                            normalization = :unit, weight = 1.0, interval = 0)

A regularizer term along **velocity**: at every time bin of its interval it differences an ordered
list of entries, each a name (a fitted variable or a dependent) at the velocity of its node in m/s,
where `RegularizerTerm` differences one name along the time axis. Its one operator, `:D3`, is the
third derivative of the cubic through four neighbouring entries times the square root of the
stencil's middle interval: for the entries' values `u₁ … u_N` at one bin, at velocities
`v₁ < … < v_N`, row `i = 1 … N − 3` is

    rowᵢ = √(v_{i+2} − v_{i+1}) · 6 · Σ_{a=i}^{i+3} u_a / Π_{b≠a} (v_a − v_b)        [(m/s)^(−5/2)]

6 times Newton's third divided difference is exact for a cubic on any nodes and zero for any
quadratic, so for the log-weights of every Maxwellian on a `GaussianGrid` whose default is a
Maxwellian; the square root makes the rows' sum of squares in one bin approximate
`∫ (∂³u/∂v³)² dv` over `[v₂, v_{N−1}]`, so that it stays put when the nodes are refined. On evenly
spaced nodes a row is `(u_{i+3} − 3u_{i+2} + 3u_{i+1} − u_i)/dv^(5/2)`. There are `N − 3` rows per
bin, stacked bin by bin over the interval's bins and within a bin by `i`; across bins the term is a
plain sum, as the time term is. A weight on it is dimensional, in (m/s)⁵
(`docs/derivations/regularizer.md` Section 8).

**The explicit form**: `select` is `N ≥ 4` names and `velocities` `N` finite, strictly increasing
numbers in m/s (no unit suffix, so SI, `conventions.md` 1.2), in node order. **The grid form**:
`grid` is a `GaussianGrid` and `select` the names of the variables bound to its parameters, in
parameter order (`g1` to `gNP`); node `n` becomes the entry `select[k]`, `k` the parameter node `n`
reads (`n` itself, or `min(n, Nv + 1 − n)` with the symmetric option), at the node's velocity
`u₀ + (n − (Nv + 1)/2) dv`. Both forms store the expanded list of entries and nothing of the grid:
the term never reads a reference temperature, a spacing or a width, so the same distribution costs
the same on any grid that holds it. **A name may repeat** (with the symmetric option every
parameter but an odd count's middle one appears twice): a row's coefficients on one θ index are
summed into one Jacobian entry. An entry may be a time-varying fitted variable, a scalar one, one
that is not varied (a constant in its rows, with no Jacobian column) or a dependent.

`operator` accepts `:D3` alone and defaults to it; that default does not change when another
operator is added. `normalization` accepts `:unit` alone, the rows as written. `weight` is the
term's own scalar under the global Λ; `interval` an index into `pspec.intervals`, an interval's
name, or `0` for the whole axis, and it must lie inside every selected variable's own interval
(Phase 4.C spec D13). Refused, each with a message: both or neither of `velocities` and `grid`; an
operator other than `:D3`; a normalization other than `:unit`; a length mismatch; a `select` whose
length is not the grid's parameter count; fewer than four entries; a velocity that is not finite,
or velocities not strictly increasing; a name that is neither a fitted variable nor a dependent;
an interval as `RegularizerTerm` refuses one.

`rows` is the term's row range in a `Regularizer`, `1:n` when built alone.

[OQ-70](../../dev/open_questions.md#oq-70); Phase G.3 spec D3 to D8.
"""
struct VelocityRegularizerTerm <: AbstractRegularizerTerm
    select::Vector{String}
    slots::Vector{Int}
    velocities::Vector{Float64}
    operator::Symbol
    normalization::Symbol
    weight::Float64
    interval::Int
    rows::UnitRange{Int}
end

# The operators a velocity term accepts (OQ-70 item 5: :D3 alone for now). Another operator is one
# more value here and one more branch in each of the four lookups below, never a new type.
const _VELOCITY_OPERATORS = (:D3,)

_velocity_operator_error(op::Symbol) = ArgumentError("VelocityRegularizerTerm: operator must be " *
    join((":$o" for o in _VELOCITY_OPERATORS), " or ") * "; got :$op")

"""
    _velocity_stencil(op::Symbol) -> Int

The number of neighbouring entries one row of the velocity operator `op` reads (OQ-70 item 5): 4
for `:D3`, so a term needs at least four entries. Any other operator is refused here, which is the
constructor's check.
"""
function _velocity_stencil(op::Symbol)
    op === :D3 && return 4
    throw(_velocity_operator_error(op))
end

# The rows per bin of a velocity term with n entries: one per position of the stencil.
_velocity_row_count(op::Symbol, n::Int) = n - _velocity_stencil(op) + 1

"""
    _velocity_interval(op::Symbol, i::Int, v::AbstractVector) -> Float64

Row `i`'s quadrature interval in m/s: for `:D3` the stencil's **middle** interval
`v[i+2] − v[i+1]`, not its outer span over three, so that the middle intervals of consecutive rows
tile `[v₂, v_{N−1}]` and the rows' sum of squares approximates an integral over velocity (OQ-70
item 2). A row carries its square root.
"""
function _velocity_interval(op::Symbol, i::Int, v::AbstractVector)
    op === :D3 && return v[i+2] - v[i+1]
    throw(_velocity_operator_error(op))
end

"""
    _velocity_rows(op::Symbol, u::AbstractVector, v::AbstractVector) -> Vector

One bin's rows of the velocity operator `op` on the entries' values `u` at the velocities `v`
(m/s), generic in `eltype(u)` so that a dual `u` differentiates: the value path and the side the
ForwardDiff oracle differentiates, never the closed-form Jacobian. For `:D3`, row `i` is
`√(v[i+2] − v[i+1])` times 6 times Newton's third divided difference on entries `i … i+3`, written
**recursively**, a difference of two second divided differences over the outer span (as
`_operator_rows` writes `D2`), independently of `_velocity_coeffs`' product form.
"""
function _velocity_rows(op::Symbol, u::AbstractVector, v::AbstractVector)
    op === :D3 || throw(_velocity_operator_error(op))
    return [sqrt(_velocity_interval(op, i, v)) * 6 *
            ((((u[i+3] - u[i+2]) / (v[i+3] - v[i+2]) - (u[i+2] - u[i+1]) / (v[i+2] - v[i+1])) / (v[i+3] - v[i+1]) -
              ((u[i+2] - u[i+1]) / (v[i+2] - v[i+1]) - (u[i+1] - u[i]) / (v[i+1] - v[i])) / (v[i+2] - v[i])) /
             (v[i+3] - v[i])) for i in 1:_velocity_row_count(op, length(u))]
end

"""
    _velocity_coeffs(op::Symbol, i::Int, v::AbstractVector) -> Tuple

Row `i`'s linear coefficients, `(entry_index, ∂rowᵢ/∂u[entry_index])` pairs, independent of the
values: the closed-form half of the term's Jacobian, never autodifferentiated. For `:D3` the
coefficient of `u_a` is `6 √(v[i+2] − v[i+1]) / Π_{b≠a} (v_a − v_b)` over the stencil's other three
entries, the **product form** of the divided difference, written independently of `_velocity_rows`'
recursive one so that the ForwardDiff oracle is a real check (`conventions.md` 8.2).
"""
function _velocity_coeffs(op::Symbol, i::Int, v::AbstractVector)
    op === :D3 || throw(_velocity_operator_error(op))
    q = 6 * sqrt(_velocity_interval(op, i, v))
    va, vb, vc, vd = v[i], v[i+1], v[i+2], v[i+3]
    return ((i, q / ((va - vb) * (va - vc) * (va - vd))),
            (i + 1, q / ((vb - va) * (vb - vc) * (vb - vd))),
            (i + 2, q / ((vc - va) * (vc - vb) * (vc - vd))),
            (i + 3, q / ((vd - va) * (vd - vb) * (vd - vc))))
end

function VelocityRegularizerTerm(pspec::ParameterSpec; select, velocities=nothing, grid=nothing,
                                  operator::Symbol=:D3, normalization::Symbol=:unit, weight::Real=1.0,
                                  interval=0)
    (velocities === nothing) == (grid === nothing) && throw(ArgumentError(
        "VelocityRegularizerTerm: give exactly one of `velocities` (the explicit form) and `grid` " *
        "(the grid form); got " * (grid === nothing ? "neither" : "both")))
    _velocity_stencil(operator)       # refuses an operator other than :D3
    normalization === :unit || throw(ArgumentError(
        "VelocityRegularizerTerm: normalization must be :unit; got :$normalization (:relative and " *
        ":initial would divide by a log-weight, which is zero at the default distribution, and " *
        ":absolute's fixed divisor is the weight under another name)"))
    names = select isa AbstractString ? [String(select)] : collect(String, select)
    if grid === nothing
        length(names) == length(velocities) || throw(ArgumentError(
            "VelocityRegularizerTerm: select has $(length(names)) names but velocities has " *
            "$(length(velocities))"))
        return _velocity_term(pspec, names, Float64.(collect(velocities)), operator, weight, interval)
    end
    entries, vel = _velocity_grid_entries(grid, names)
    return _velocity_term(pspec, entries, vel, operator, weight, interval)
end

"""
    _velocity_grid_entries(grid::GaussianGrid, names) -> (entries::Vector{String}, velocities::Vector{Float64})

The grid form's expansion (Phase G.3 spec D4): `names` name the grid's `NP` parameters in
parameter order; node `n` of `Nv` becomes the entry `names[_grid_parameter(n, …)]` at the velocity
`_grid_node(grid, n)`, in m/s. With the symmetric option every parameter but an odd count's middle
one appears twice, and the list has `Nv` entries. Nothing else of the grid is read.
"""
function _velocity_grid_entries(grid::GaussianGrid{FT, Nv, NP}, names::Vector{String}) where {FT, Nv, NP}
    length(names) == NP || throw(ArgumentError(
        "VelocityRegularizerTerm: the grid has $NP parameters (g1 to g$NP), so select must name $NP " *
        "variables, in parameter order; got $(length(names))"))
    entries = [names[_grid_parameter(n, Val(Nv), Val(NP))] for n in 1:Nv]
    vel = [_grid_node(grid, n) for n in 1:Nv]
    return entries, vel
end

"""
    _velocity_term(pspec, names, velocities, operator, weight, interval) -> VelocityRegularizerTerm

The one inner path of both constructor forms (Phase G.3 spec D4), checking in order: the entries'
count against the operator's stencil; the velocities, finite and strictly increasing; the names,
each a fitted variable or a dependent; the interval (`_resolve_term_interval`); and that it lies
inside every selected variable's own (`_check_term_inside_variables`, Phase 4.C spec D13). `rows`
is `1:n`, the term alone.
"""
function _velocity_term(pspec::ParameterSpec, names::Vector{String}, velocities::Vector{Float64},
                        operator::Symbol, weight::Real, interval)
    n = length(names)
    width = _velocity_stencil(operator)
    n >= width || throw(ArgumentError(
        "VelocityRegularizerTerm: operator :$operator needs at least $width entries; got $n"))
    for (a, v) in enumerate(velocities)
        isfinite(v) || throw(ArgumentError("VelocityRegularizerTerm: velocity $a is not finite; got $v"))
        (a == 1 || v > velocities[a-1]) || throw(ArgumentError(
            "VelocityRegularizerTerm: the velocities must be strictly increasing; velocity $a ($v m/s) " *
            "is not above velocity $(a - 1) ($(velocities[a-1]) m/s)"))
    end
    slots = Vector{Int}(undef, n)
    for (a, nm) in enumerate(names)
        slot = get(pspec.slots, nm, 0)
        slot == 0 && throw(ArgumentError(
            "VelocityRegularizerTerm: \"$nm\" is not a fitted variable or a dependent; valid names " *
            "are $(sort(collect(keys(pspec.slots))))"))
        slots[a] = slot
    end
    intv = _resolve_term_interval(pspec, interval; who="VelocityRegularizerTerm")
    bins = intv == 0 ? (1:nbins(pspec.trace)) : pspec.intervals[intv].bins
    _check_term_inside_variables(pspec, names, slots, intv, bins, "VelocityRegularizerTerm")
    nrows = _velocity_row_count(operator, n) * length(bins)
    return VelocityRegularizerTerm(names, slots, velocities, operator, :unit, Float64(weight), intv,
                                   1:nrows)
end

# The term with its row range in a Regularizer (Phase G.3 spec D7).
_with_rows(t::VelocityRegularizerTerm, rows::UnitRange{Int}) =
    VelocityRegularizerTerm(t.select, t.slots, t.velocities, t.operator, t.normalization, t.weight,
                            t.interval, rows)

# The raw rows, generic (the value path and the ForwardDiff oracle's side, spec D6): at each bin of
# the interval, the entries' values from the shared value table, then _velocity_rows. `x0table` is
# unused (`:unit` only); the argument keeps the walkers' one call for both term types.
function _term_raw(term::VelocityRegularizerTerm, pspec::ParameterSpec, vtable::AbstractMatrix{T},
                    x0table::Union{Nothing,AbstractMatrix{Float64}}) where {T<:Real}
    bins = _term_bins(term, pspec)
    n = length(term.slots)
    nr = _velocity_row_count(term.operator, n)
    out = Vector{T}(undef, length(term.rows))
    u = Vector{T}(undef, n)
    for (k, j) in enumerate(bins)
        for a in 1:n
            u[a] = vtable[term.slots[a], j]
        end
        out[((k - 1) * nr + 1):(k * nr)] .= _velocity_rows(term.operator, u, term.velocities)
    end
    return out
end

"""
    _term_closed_form(term::VelocityRegularizerTerm, pspec, theta, vtable, x0table, cache) -> (raw, entries)

The velocity term's raw rows (from `_velocity_rows`, the value path) and its **unweighted**
closed-form Jacobian entries `(local_row, theta_index, ∂raw_row/∂theta_index)` (spec D6): each
row's coefficients `_velocity_coeffs`, each entry's value and partials `_slot_value_and_grad`
(memoized in `cache` by `(slot, bin)`, shared with every other term of the evaluation), so a held
entry contributes no column and a dependent its chain. **Within one row the contributions to one
theta index are summed into one entry** (OQ-70 item 3): a repeated name, or two dependents of one
variable, would otherwise give two entries on one row and column, which `jvp!` and `jtvp!` add
correctly and which `_gauss_newton_diagonal` squared one by one until Phase G.4 (it now sums a row's
entries on a column before squaring, spec D6 of that phase). The entries of a row are in the order
their theta indices first appear in it.
"""
function _term_closed_form(term::VelocityRegularizerTerm, pspec::ParameterSpec, theta::AbstractVector{Float64},
                            vtable::AbstractMatrix{Float64}, x0table::Union{Nothing,AbstractMatrix{Float64}},
                            cache::Dict{Tuple{Int,Int},Tuple{Float64,Vector{Int},Vector{Float64}}})
    bins = _term_bins(term, pspec)
    n = length(term.slots)
    nr = _velocity_row_count(term.operator, n)
    raw = Vector{Float64}(undef, length(term.rows))
    entries = Tuple{Int,Int,Float64}[]
    u = Vector{Float64}(undef, n)
    cols = Int[]        # one row's theta indices, in the order they first appear
    sums = Float64[]    # and the sum of the row's coefficients on each
    for (k, j) in enumerate(bins)
        for a in 1:n
            u[a] = vtable[term.slots[a], j]
        end
        rows_here = _velocity_rows(term.operator, u, term.velocities)
        for r in 1:nr
            lr = (k - 1) * nr + r
            raw[lr] = rows_here[r]
            empty!(cols)
            empty!(sums)
            for (a, c) in _velocity_coeffs(term.operator, r, term.velocities)
                slot = term.slots[a]
                key = (slot, j)
                sv = get(cache, key, nothing)
                if sv === nothing
                    sv = _slot_value_and_grad(pspec, theta, j, slot)
                    cache[key] = sv
                end
                _, active, grad = sv
                for (m, aidx) in enumerate(active)
                    p = findfirst(==(aidx), cols)
                    if p === nothing
                        push!(cols, aidx)
                        push!(sums, c * grad[m])
                    else
                        sums[p] += c * grad[m]
                    end
                end
            end
            for (aidx, s) in zip(cols, sums)
                push!(entries, (lr, aidx, s))
            end
        end
    end
    return raw, entries
end

# --- show (spec D40) ---

function Base.show(io::IO, term::RegularizerTerm)
    print(io, "RegularizerTerm(", join(term.select, ", "), "; :", term.operator, ", :",
        term.normalization, ", weight = ", term.weight, ")")
end

function Base.show(io::IO, reg::Regularizer)
    print(io, "Regularizer(", length(reg.terms), " terms, weight = ", reg.weight, ")")
end

function Base.show(io::IO, ::MIME"text/plain", reg::Regularizer)
    println(io, "Regularizer")
    println(io, "  weight (Lambda) = ", reg.weight)
    println(io, "  terms (", length(reg.terms), "), ", reg.nrows, " rows:")
    for term in reg.terms
        println(io, "    ", _term_line(term), ", rows ", term.rows)
    end
end

# One term's line in a Regularizer's listing, before its rows (one method per term type, Phase G.3
# spec D7; a time term's line is the one this listing printed before).
_term_line(term::RegularizerTerm) = string(join(term.select, ", "), ": :", term.operator, ", :",
    term.normalization, ", weight = ", term.weight)
_term_line(term::VelocityRegularizerTerm) = string(_velocity_entries_phrase(term), ": :",
    term.operator, ", :", term.normalization, ", weight = ", term.weight)

# A velocity term's entries in one phrase (Phase G.3 spec D7): the first and last names, the count,
# and the velocity range in m/s to four significant digits.
_velocity_entries_phrase(term::VelocityRegularizerTerm) = string(first(term.select), " … ",
    last(term.select), " (", length(term.select), " entries, ", round(first(term.velocities); sigdigits=4),
    " to ", round(last(term.velocities); sigdigits=4), " m/s)")

function Base.show(io::IO, term::VelocityRegularizerTerm)
    print(io, "VelocityRegularizerTerm(", _velocity_entries_phrase(term), "; :", term.operator, ", :",
        term.normalization, ", weight = ", term.weight, ")")
end
