# The dependent-expression layer (Step 3.1b, chunk 3 of Phase 3): a typed tree over fitted
# variables, other dependents and t, built by lowering Julia's own parse tree and interpreted by
# ordinary Julia dispatch -- never eval'ed (OQ-17: the reasons are world age, type stability
# inside unflatten, and letting a deck run arbitrary statements, not sandboxing). The resolved
# function of a CallNode/CompareNode is a *type parameter*, not a Function field, so the whole
# tree evaluates type-stably even under a ForwardDiff.Dual vals vector -- the property the
# Jacobian's hot path (Step 3.2 onward) depends on. Nothing here is consumed yet: ParameterSpec,
# bindings and unflatten are Step 3.1c (chunk 4); no exports until then. Spec D6, D7.

import SpecialFunctions

# ------------------------------------------------------------------------------------------
# D6: the typed tree and its evaluator
# ------------------------------------------------------------------------------------------

"""
    AbstractExprNode

The abstract supertype of every node in a dependent's typed expression tree (D6): `ConstNode`,
`SlotNode`, `TimeNode`, `CallNode`, `CompareNode`, `IfElseNode`. Every concrete node is immutable
and concrete once built; evaluate a tree with `_eval`.
"""
abstract type AbstractExprNode end

"""
    ConstNode{FT<:Real}

A numeric literal. Every literal in a parsed dependent becomes `ConstNode(Float64(x))`
(`_parse_expression`), so that, e.g., the `2` in `"2x"` does not make the subtree
`Int`-flavoured. Spec D6.
"""
struct ConstNode{FT<:Real} <: AbstractExprNode
    value::FT
end

"""
    SlotNode

A reference to slot `slot` of the per-bin value vector that `_eval` is called against: the
fitted variables first, then the dependents, in slot order (D8, Step 3.1c). Spec D6.
"""
struct SlotNode <: AbstractExprNode
    slot::Int
end

"""
    TimeNode

The bin's time `t`, in the axis keyword's own unit (nanoseconds when the trace was built with
`t_ns`, seconds with `t_s`; D8). Spec D6.
"""
struct TimeNode <: AbstractExprNode end

"""
    CallNode{F,A<:Tuple}

A call `f(args...)`, `args` a `Tuple` of `AbstractExprNode`s. **`f` is stored by value as a type
parameter** (`typeof(exp)`, `typeof(+)`, ...), never as an `f::Function` field:
`CallNode{typeof(exp), Tuple{SlotNode}}` is a concrete type, so `node.f(...)` resolves at compile
time and the whole tree, including `_eval`, is type-stable under a `ForwardDiff.Dual` `vals` --
the property `unflatten`'s per-bin, per-sweep hot path (Step 3.2 onward) depends on. An
`f::Function` field would box every call behind a dynamic dispatch and make that path type
unstable. Spec D6.
"""
struct CallNode{F,A<:Tuple} <: AbstractExprNode
    f::F
    args::A
end

"""
    CompareNode{F,L<:AbstractExprNode,R<:AbstractExprNode}

A comparison `lhs f rhs`, `f` one of `<, <=, ==, !=, >, >=`, stored by value as a type parameter
for the same reason as `CallNode`. Spec D6.
"""
struct CompareNode{F,L<:AbstractExprNode,R<:AbstractExprNode} <: AbstractExprNode
    f::F
    lhs::L
    rhs::R
end

"""
    IfElseNode{C<:AbstractExprNode,A<:AbstractExprNode,B<:AbstractExprNode}

The conditional `cond ? iftrue : iffalse`, or `ifelse(cond, iftrue, iffalse)` written as a call.
**Only the taken branch is evaluated** -- a documented departure from `Base.ifelse`'s eager
form -- so that the untaken branch may be undefined at the current values (e.g. `log` of a
quantity that is only positive on the branch actually taken). This is the mechanism an interval
uses to switch a leaf between two variables, `u = "t < 3.4 ? u_pre : u_post"`, without ever
evaluating the inactive one. Spec D6.
"""
struct IfElseNode{C<:AbstractExprNode,A<:AbstractExprNode,B<:AbstractExprNode} <: AbstractExprNode
    cond::C
    iftrue::A
    iffalse::B
end

"""
    _eval(node::AbstractExprNode, vals::AbstractVector{T}, t) where {T} -> T

Evaluate `node` against the per-bin value vector `vals` and the bin's time `t`. Generic in the
element type of `vals` (`conventions.md` 5.1), so a `ForwardDiff.Dual` `vals` produces a dual
result; type-stable because every node's resolved function is a type parameter, not a field
(D6). `map` over a node's `args::Tuple` unrolls at compile time; only the taken branch of an
`IfElseNode` is evaluated.
"""
_eval(n::ConstNode, vals::AbstractVector{T}, t) where {T} = convert(T, n.value)
_eval(n::SlotNode,  vals, t) = @inbounds vals[n.slot]
_eval(::TimeNode,   vals::AbstractVector{T}, t) where {T} = convert(T, t)
_eval(n::CallNode,  vals, t) = n.f(map(a -> _eval(a, vals, t), n.args)...)
_eval(n::CompareNode, vals, t) = n.f(_eval(n.lhs, vals, t), _eval(n.rhs, vals, t))
_eval(n::IfElseNode,  vals, t) = _eval(n.cond, vals, t) ? _eval(n.iftrue, vals, t) :
                                                          _eval(n.iffalse, vals, t)

# ------------------------------------------------------------------------------------------
# D7: parsing, resolution and the read-time checks
# ------------------------------------------------------------------------------------------

const _COMPARISON_SYMBOLS = (:<, :(<=), :(==), :(!=), :>, :(>=))

"""
    NONSMOOTH_FUNCTIONS

The functions D7's one sanctioned warning watches for (`conventions.md` 5.9): applying one of
these to an argument whose subtree involves a fitted or dependent variable (not only `t` and
constants) makes the objective non-smooth there. A comparison operator and an `IfElseNode`'s
condition are checked the same way even though only the comparison operators, not `ifelse`
itself, appear in this tuple: `ifelse`'s non-smoothness is carried entirely by its `IfElseNode`
lowering, checked directly by `_check_nonsmooth`.
"""
const NONSMOOTH_FUNCTIONS = (min, max, abs, sign, clamp, floor, ceil, round, mod, rem,
                              <, <=, ==, !=, >, >=)

_grammar_error(construct::AbstractString, text::AbstractString, context::AbstractString) =
    throw(ArgumentError(
        "dependent \"$context\" uses $construct, which is not allowed in a dependent " *
        "expression (\"$text\")"))

"""
    _resolve_function(fsym::Symbol, text, context) -> Function

Resolve a call's function name (D7 item 3): `Base` first, then `SpecialFunctions`, else a loud
error naming the two modules searched. The resolved object must be a `Function`; anything else
(a constant such as `pi`, a type, a module) is a loud error naming what it resolved to.
"""
function _resolve_function(fsym::Symbol, text::AbstractString, context::AbstractString)
    if isdefined(Base, fsym)
        f = getfield(Base, fsym)
    elseif isdefined(SpecialFunctions, fsym)
        f = getfield(SpecialFunctions, fsym)
    else
        throw(ArgumentError(
            "\"$fsym\" is not a fitted variable, a dependent, `t`, or a function in Base or " *
            "SpecialFunctions (dependent \"$context\": \"$text\")"))
    end
    f isa Function || throw(ArgumentError(
        "\"$fsym\" resolves to $(repr(f)), which is not a function " *
        "(dependent \"$context\": \"$text\")"))
    return f
end

"""
    _resolve_name(sym::Symbol, names, text, context) -> Union{TimeNode,SlotNode}

Resolve a bare name used as a value: `:t` becomes a `TimeNode`, a key of `names` becomes a
`SlotNode`, anything else is a loud error (D7 item 2, D39).
"""
function _resolve_name(sym::Symbol, names::Dict{String,Int}, text::AbstractString,
                        context::AbstractString)
    sym === :t && return TimeNode()
    key = string(sym)
    haskey(names, key) && return SlotNode(names[key])
    throw(ArgumentError(
        "\"$sym\" is not a fitted variable, a dependent, `t`, or a function in Base or " *
        "SpecialFunctions (dependent \"$context\": \"$text\")"))
end

_lower(ex::Number, names::Dict{String,Int}, text::AbstractString, context::AbstractString) =
    ConstNode(Float64(ex))
_lower(ex::AbstractString, names::Dict{String,Int}, text::AbstractString, context::AbstractString) =
    _grammar_error("string", text, context)
_lower(ex::Symbol, names::Dict{String,Int}, text::AbstractString, context::AbstractString) =
    _resolve_name(ex, names, text, context)
_lower(ex::Expr, names::Dict{String,Int}, text::AbstractString, context::AbstractString) =
    _lower_expr(ex, names, text, context)
_lower(ex, names::Dict{String,Int}, text::AbstractString, context::AbstractString) =
    _grammar_error("a $(typeof(ex))", text, context)

function _lower_call(ex::Expr, names::Dict{String,Int}, text::AbstractString,
                      context::AbstractString)
    fex = ex.args[1]
    argexs = ex.args[2:end]
    fex isa Expr && fex.head === :. && _grammar_error("qualified name", text, context)
    fex isa Symbol || _grammar_error("a computed call", text, context)
    if fex in _COMPARISON_SYMBOLS
        length(argexs) == 2 || throw(ArgumentError(
            "dependent \"$context\": \"$fex\" takes exactly two arguments (\"$text\")"))
        lhs = _lower(argexs[1], names, text, context)
        rhs = _lower(argexs[2], names, text, context)
        return CompareNode(getfield(Base, fex), lhs, rhs)
    elseif fex === :ifelse
        length(argexs) == 3 || throw(ArgumentError(
            "dependent \"$context\": ifelse takes exactly three arguments (\"$text\")"))
        cond = _lower(argexs[1], names, text, context)
        iftrue = _lower(argexs[2], names, text, context)
        iffalse = _lower(argexs[3], names, text, context)
        return IfElseNode(cond, iftrue, iffalse)
    else
        f = _resolve_function(fex, text, context)
        argnodes = Tuple(_lower(a, names, text, context) for a in argexs)
        return CallNode(f, argnodes)
    end
end

function _lower_expr(ex::Expr, names::Dict{String,Int}, text::AbstractString,
                      context::AbstractString)
    head = ex.head
    if head === :call
        return _lower_call(ex, names, text, context)
    elseif head === :if
        length(ex.args) == 3 || _grammar_error("a conditional", text, context)
        cond = _lower(ex.args[1], names, text, context)
        iftrue = _lower(ex.args[2], names, text, context)
        iffalse = _lower(ex.args[3], names, text, context)
        return IfElseNode(cond, iftrue, iffalse)
    elseif head === :&&
        lhs = _lower(ex.args[1], names, text, context)
        rhs = _lower(ex.args[2], names, text, context)
        return CallNode(&, (lhs, rhs))
    elseif head === :||
        lhs = _lower(ex.args[1], names, text, context)
        rhs = _lower(ex.args[2], names, text, context)
        return CallNode(|, (lhs, rhs))
    elseif head === :block
        stmts = [a for a in ex.args if !(a isa LineNumberNode)]
        length(stmts) == 1 || _grammar_error("multiple statements", text, context)
        return _lower(stmts[1], names, text, context)
    elseif head === :(=)
        lhs = ex.args[1]
        construct = (lhs isa Expr && lhs.head === :call) ? "function definition" : "assignment"
        _grammar_error(construct, text, context)
    elseif head === :ref
        _grammar_error("indexing", text, context)
    elseif head === :for || head === :while
        _grammar_error("loop", text, context)
    elseif head === :function || head === :->
        _grammar_error("function definition", text, context)
    elseif head === :macrocall
        _grammar_error("macro call", text, context)
    elseif head === :.
        _grammar_error("qualified name", text, context)
    elseif head === :string
        _grammar_error("string", text, context)
    elseif head === :tuple
        _grammar_error("tuple", text, context)
    else
        _grammar_error("a $(head) expression", text, context)
    end
end

"""
    _probe_construction(node, names, text, context)

D7 item 4's **structural** half, revised 2026-09-13: evaluate `node` once against a synthetic
value vector (every slot at `1.0`, `t = 1.0`) and convert **only a `MethodError`** — a wrong
arity, or a name that resolved to something not callable on numbers, `"exp(a, b)"` (XR14) — into
a loud failure now, at the moment the dependent's tree is built, rather than a surprise deep in
the Jacobian's hot path.

**This probe does not, and must not, claim anything about finiteness or domain validity.** Any
exception other than `MethodError`, and any non-finite result, is silently ignored: a synthetic
placeholder cannot tell a broken expression from a perfectly good one evaluated at the wrong
point. `"log(t)"`, `"1/t"`, `"1/(x - 1)"`, `"sqrt(t - 1)"` and `"exp(x*1000)"` are all legitimate
dependents that are `Inf`, `-Inf` or a `DomainError` at this probe's placeholder values, and the
first cut of this check (which also required a finite result) refused all five — a read-time
check that refuses `log(t)` is worse than no check. The placeholder for `t` is `1.0`, not `0.0`,
so that the canonical `t < X` idiom and `log(t)` alike are unlikely to trip even the `MethodError`
pass by accident; it is otherwise free, since nothing here depends on it.

The **value** half — evaluating every dependent at its real `init` values, at every bin of its
interval, and requiring a finite `Real` there — is `ParameterSpec`'s own probe (D8, D10, Step
3.1c, PM26), built once real values and bins exist. That is the check a genuine `DomainError` or
a typo that parses to something meaningless is caught by; this one only guarantees the call is
*structurally* callable at all, with synthetic values chosen so a conditional guarding a domain
restriction is not itself what trips it (D6's lazy `IfElseNode` already keeps an untaken branch
from being probed).
"""
function _probe_construction(node::AbstractExprNode, names::Dict{String,Int},
                              text::AbstractString, context::AbstractString)
    nslots = isempty(names) ? 0 : maximum(values(names))
    vals = ones(Float64, nslots)
    try
        _eval(node, vals, 1.0)
    catch e
        e isa MethodError || return nothing
        throw(ArgumentError(
            "dependent \"$context\" could not be evaluated at its initial values: " *
            "$(sprint(showerror, e)) (\"$text\")"))
    end
    return nothing
end

_contains_slot(::ConstNode) = false
_contains_slot(::SlotNode) = true
_contains_slot(::TimeNode) = false
_contains_slot(n::CallNode) = any(_contains_slot, n.args)
_contains_slot(n::CompareNode) = _contains_slot(n.lhs) || _contains_slot(n.rhs)
_contains_slot(n::IfElseNode) =
    _contains_slot(n.cond) || _contains_slot(n.iftrue) || _contains_slot(n.iffalse)

function _warn_nonsmooth_at(context::AbstractString, fname::AbstractString)
    @warn "dependent \"$context\" applies the non-smooth function $fname to an argument that " *
          "involves a fitted or dependent variable: the objective may have a kink or a jump " *
          "there, a fitter may stall or oscillate where the branch flips, and the Jacobian is " *
          "one-sided at that point"
    return nothing
end

"""
    _check_nonsmooth(node::AbstractExprNode, context::AbstractString)

Walk `node` and warn (`conventions.md` 5.9, the one sanctioned warning) at every `CallNode`
whose `f` is in `NONSMOOTH_FUNCTIONS`, every `CompareNode`, and every `IfElseNode`, when the
argument (the condition, for an `IfElseNode`) has a subtree containing a `SlotNode` -- a fitted
or dependent variable. An argument built only from `TimeNode` and `ConstNode` draws no warning:
that is how an interval switches a leaf between two variables without triggering it. The deck is
never refused for this; it only warns. Spec D7.
"""
_check_nonsmooth(::ConstNode, context::AbstractString) = nothing
_check_nonsmooth(::SlotNode, context::AbstractString) = nothing
_check_nonsmooth(::TimeNode, context::AbstractString) = nothing

function _check_nonsmooth(n::CallNode, context::AbstractString)
    if n.f in NONSMOOTH_FUNCTIONS && any(_contains_slot, n.args)
        _warn_nonsmooth_at(context, string(nameof(n.f)))
    end
    foreach(a -> _check_nonsmooth(a, context), n.args)
    return nothing
end

function _check_nonsmooth(n::CompareNode, context::AbstractString)
    if _contains_slot(n.lhs) || _contains_slot(n.rhs)
        _warn_nonsmooth_at(context, string(nameof(n.f)))
    end
    _check_nonsmooth(n.lhs, context)
    _check_nonsmooth(n.rhs, context)
    return nothing
end

function _check_nonsmooth(n::IfElseNode, context::AbstractString)
    _contains_slot(n.cond) && _warn_nonsmooth_at(context, "ifelse")
    _check_nonsmooth(n.cond, context)
    _check_nonsmooth(n.iftrue, context)
    _check_nonsmooth(n.iffalse, context)
    return nothing
end

"""
    _parse_expression(text::AbstractString, names::Dict{String,Int}; context::AbstractString) -> AbstractExprNode

Parse `text` as a Julia expression (`Meta.parse`), lower it into a typed `AbstractExprNode` tree
and resolve every call's function name, `Base` first, then `SpecialFunctions` (D7). Accepts a
number literal, `t`, a name in `names`, a call, a comparison, `ifelse`, the conditional
`a ? b : c`, `&&`, `||`, and a parenthesized single-statement block; everything else the parser
can produce -- assignment, indexing, a loop, a function definition, a macro call, a qualified
name, a string, a tuple -- is a loud error naming the construct. **Indexing being unrepresentable
is what enforces "pointwise in time"**: a dependent may combine the bin-`j` values of variables
and `t`, never bin `j - 1`, which is what keeps Step 3.2's block-structured Jacobian valid.

After the tree is built it is probe-evaluated once (`_probe_construction`, D7 item 4) so a
structurally broken call (an arity error that parses, e.g. `"exp(a, b)"`) is a loud failure now,
not a surprise deep in the Jacobian's hot path; and it is walked once for D7's one sanctioned
warning (`_check_nonsmooth`), which never refuses the deck.

`context` names the dependent (or the text itself, for a standalone probe) for every message
this call can raise. A deck is trusted input in the same class as a script deck: this is not a
sandbox, and the reason expressions are not `eval`ed is world age, type stability and statement
execution (OQ-17), not safety from a malicious deck. Spec D7.
"""
function _parse_expression(text::AbstractString, names::Dict{String,Int};
                            context::AbstractString)
    ex = try
        Meta.parse(text)
    catch e
        throw(ArgumentError(
            "dependent \"$context\" could not be parsed as a Julia expression: " *
            "$(sprint(showerror, e)) (\"$text\")"))
    end
    (ex === nothing || (ex isa Expr && ex.head === :incomplete)) && throw(ArgumentError(
        "dependent \"$context\" could not be parsed as a Julia expression (\"$text\")"))
    node = _lower(ex, names, text, context)
    _probe_construction(node, names, text, context)
    _check_nonsmooth(node, context)
    return node
end
