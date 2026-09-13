using Test
using OTIS
using OTIS: _parse_expression, _eval
import ForwardDiff
import SpecialFunctions

# XR1 to XR14 against dev/phase3_instructions.md Section 4's Step 3.1 table (spec D6, D7), gated
# "expressions". XR13 (cycle detection over a *set* of named dependents) is not built here: it
# needs D8's topological sort, which does not exist until ParameterSpec (Step 3.1c, chunk 4) --
# see the chunk 3 hand-off report for the finding and the user's approval to defer it there.

# A shared names table covering every slot XR1-XR11 uses, and the plain Float64 values evaluated
# against it; XR12 builds its own small table since it is specifically about declaration order.
const _XR_NAMES = Dict("logn" => 1, "f_D" => 2, "f_C" => 3, "x" => 4, "a" => 5, "b" => 6,
                        "u_pre" => 7, "u_post" => 8)
const _XR_VALS = [19.5, 0.3, 0.7, 2.0, 0.5, 1.5, 100.0, 300.0]

_parse_and_check(text::AbstractString; names::Dict{String,Int} = _XR_NAMES,
                  context::AbstractString = text) = _parse_expression(text, names; context = context)

@testset "XR1: literals, names and t" begin
    n1 = _parse_expression("1.5", Dict{String,Int}(); context = "XR1a")
    @test _eval(n1, Float64[], 0.0) === 1.5

    n2 = _parse_expression("logn", _XR_NAMES; context = "XR1b")
    for t in (0.0, 1.0, 2.0)
        @test _eval(n2, _XR_VALS, t) === 19.5
    end

    n3 = _parse_expression("t", Dict{String,Int}(); context = "XR1c")
    for t in (0.0, 1.0, 2.0)
        @test _eval(n3, Float64[], t) === t
    end
end

@testset "XR2: arithmetic and Julia functions" begin
    cases = (
        ("10^logn",      () -> 10.0^19.5,                    0.0),
        ("1 - f_D",      () -> 1.0 - 0.3,                     0.0),
        ("2x",           () -> 2.0 * 2.0,                     0.0),
        ("x^2",          () -> 2.0^2,                         0.0),
        ("exp(-t/1.35)", () -> exp(-2.0 / 1.35),               2.0),
        ("tanh(a)",      () -> tanh(0.5),                      0.0),
        ("hypot(a, b)",  () -> hypot(0.5, 1.5),                0.0),
        ("atan(a, b)",   () -> atan(0.5, 1.5),                 0.0),
        ("erf(a)",       () -> SpecialFunctions.erf(0.5),      0.0),
        ("gamma(a)",     () -> SpecialFunctions.gamma(0.5),    0.0),
    )
    for (text, oracle, t) in cases
        node = _parse_expression(text, _XR_NAMES; context = "XR2:" * text)
        v = _eval(node, _XR_VALS, t)
        @test isapprox(v, oracle(); rtol = 1e-15)
    end
end

@testset "XR3: the conditional, both spellings" begin
    n_ternary = _parse_expression("t < 3.4 ? u_pre : u_post", _XR_NAMES; context = "XR3a")
    n_ifelse = _parse_expression("ifelse(t < 3.4, u_pre, u_post)", _XR_NAMES; context = "XR3b")
    upre, upost = _XR_VALS[_XR_NAMES["u_pre"]], _XR_VALS[_XR_NAMES["u_post"]]
    for t in (1.0, 3.4, 5.0)
        expected = t < 3.4 ? upre : upost
        @test _eval(n_ternary, _XR_VALS, t) === expected
        @test _eval(n_ifelse, _XR_VALS, t) === expected
    end
end

@testset "XR4: the taken branch only is evaluated" begin
    node = _parse_expression("t < 1.0 ? 1.0 : log(-1.0)", Dict{String,Int}(); context = "XR4")
    @test _eval(node, Float64[], 0.5) === 1.0
end

@testset "XR5: duals flow" begin
    scalar_cases = (
        ("10^logn", "logn", 19.5),
        ("1 - f_D", "f_D", 0.3),
        ("2x", "x", 2.0),
        ("x^2", "x", 2.0),
        ("tanh(a)", "a", 0.5),
        ("hypot(a, b)", "a", 0.5),
        ("atan(a, b)", "a", 0.5),
        ("erf(a)", "a", 0.5),
        ("gamma(a)", "a", 0.5),
    )
    for (text, varname, x0) in scalar_cases
        node = _parse_expression(text, _XR_NAMES; context = "XR5:" * text)
        slot = _XR_NAMES[varname]
        f = y -> begin
            vals = Vector{typeof(y)}(undef, length(_XR_VALS))
            for i in eachindex(_XR_VALS)
                vals[i] = i == slot ? y : _XR_VALS[i]
            end
            _eval(node, vals, 0.0)
        end
        ad = ForwardDiff.derivative(f, x0)
        hh = 1e-6 * abs(x0)
        fd = (f(x0 + hh) - f(x0 - hh)) / (2hh)
        floor = 1e-6 * max(abs(fd), 1e-8)
        @test isapprox(ad, fd; rtol = 1e-6, atol = floor)
    end

    node_t = _parse_expression("exp(-t/1.35)", Dict{String,Int}(); context = "XR5:exp(-t/1.35)")
    # TimeNode's _eval takes its element type from `vals`, not from `t` (D6), so an empty `vals`
    # must still be typed to match `t` for a derivative purely with respect to t to flow.
    ft = tt -> _eval(node_t, Vector{typeof(tt)}(), tt)
    t0 = 2.0
    ad_t = ForwardDiff.derivative(ft, t0)
    hh = 1e-6 * t0
    fd_t = (ft(t0 + hh) - ft(t0 - hh)) / (2hh)
    @test isapprox(ad_t, fd_t; rtol = 1e-6, atol = 1e-6 * max(abs(fd_t), 1e-8))
end

@testset "XR6: type stability" begin
    texts = ("10^logn", "1 - f_D", "2x", "x^2", "exp(-t/1.35)", "tanh(a)",
              "hypot(a, b)", "atan(a, b)", "erf(a)", "gamma(a)")
    dvals = ForwardDiff.Dual{:XR6}.(_XR_VALS, 1.0)
    for text in texts
        node = _parse_expression(text, _XR_NAMES; context = "XR6:" * text)
        v = @inferred _eval(node, _XR_VALS, 2.0)
        @test v isa Float64
        dv = @inferred _eval(node, dvals, 2.0)
        @test dv isa ForwardDiff.Dual
    end
end

@testset "XR7: the grammar refuses what it must" begin
    @test_throws r"assignment" _parse_expression(
        "a = 1", Dict{String,Int}(); context = "XR7a")
    @test_throws r"indexing" _parse_expression(
        "u[j-1]", Dict{String,Int}(); context = "XR7b")
    @test_throws r"loop" _parse_expression(
        "for i in 1:3 end", Dict{String,Int}(); context = "XR7c")
    @test_throws r"function definition" _parse_expression(
        "f(x) = x", Dict{String,Int}(); context = "XR7d")
    @test_throws r"function definition" _parse_expression(
        "x -> x", Dict{String,Int}(); context = "XR7e")
    @test_throws r"macro call" _parse_expression(
        "@show x", Dict{String,Int}(); context = "XR7f")
    @test_throws r"qualified name" _parse_expression(
        "Base.exp(x)", Dict{String,Int}(); context = "XR7g")
    @test_throws r"string" _parse_expression(
        "\"s\"", Dict{String,Int}(); context = "XR7h")
    @test_throws r"tuple" _parse_expression(
        "(a, b)", Dict{String,Int}(); context = "XR7i")
end

@testset "XR8: unknown names" begin
    err = nothing
    try
        _parse_expression("nosuchvar", _XR_NAMES; context = "XR8a")
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("nosuchvar", err.msg)
    @test occursin("Base", err.msg) && occursin("SpecialFunctions", err.msg)

    err2 = nothing
    try
        _parse_expression("nosuchfun(a)", _XR_NAMES; context = "XR8b")
    catch e
        err2 = e
    end
    @test err2 isa ArgumentError
    @test occursin("nosuchfun", err2.msg)
    @test occursin("Base", err2.msg) && occursin("SpecialFunctions", err2.msg)
end

@testset "XR9: a name that is not a function" begin
    err = nothing
    try
        _parse_expression("pi(a)", _XR_NAMES; context = "XR9")
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("resolves to", err.msg)
    @test occursin("which is not a function", err.msg)
end

@testset "XR10: the non-smooth warning fires" begin
    @test_logs (:warn, r"min") _parse_and_check("min(u_pre, u_post)")
    @test_logs (:warn, r"abs") _parse_and_check("abs(a)")
    @test_logs (:warn, r"clamp") _parse_and_check("clamp(a, 0.0, 10.0)")
    @test_logs (:warn, r">") _parse_and_check("u_pre > u_post")
    # a conditional whose condition names a variable: the condition (a comparison over a
    # variable) is itself a CompareNode, so it draws its own warning in addition to the
    # IfElseNode's -- see the chunk 3 report for this as a design point worth Level 2's review.
    @test_logs (:warn, r"ifelse") (:warn, r">") _parse_and_check("u_pre > u_post ? a : b")
end

@testset "XR11: a condition on t alone does not warn" begin
    @test_logs _parse_and_check("t < 3.4 ? u_pre : u_post")
end

@testset "XR12: declaration order does not matter to parsing" begin
    names = Dict("f_D" => 1, "f_C" => 2, "g" => 3)
    # g's expression is parsed BEFORE f_C's, the reverse of their dependency order; nothing in
    # _parse_expression/_eval assumes an order, so filling vals dependency-first still works.
    node_g = _parse_expression("f_C * 2", names; context = "g")
    node_fC = _parse_expression("1 - f_D", names; context = "f_C")
    vals = zeros(Float64, 3)
    vals[1] = 0.3
    vals[2] = _eval(node_fC, vals, 0.0)
    vals[3] = _eval(node_g, vals, 0.0)
    @test isapprox(vals[2], 0.7; rtol = 1e-15)
    @test isapprox(vals[3], 1.4; rtol = 1e-15)
end

@testset "XR14: the read-time probe catches a bad call" begin
    err = nothing
    try
        _parse_expression("exp(a, b)", Dict("a" => 1, "b" => 2); context = "XR14")
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("could not be evaluated at its initial values", err.msg)
end
