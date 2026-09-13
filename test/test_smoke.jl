using Test
using OTIS
using REPL: REPL  # gives Base.Docs.doc its methods (empty otherwise); see HS1 below

@testset "smoke" begin
    @test OTIS isa Module
end

@testset "HS1: every exported name has a docstring" begin
    for nm in names(OTIS)
        doc_str = string(Base.Docs.doc(getfield(OTIS, nm)))
        @test !isempty(doc_str)
        @test !occursin("No documentation found", doc_str)
    end
end

@testset "HS3: no stale Gain wording" begin
    src_dir = joinpath(@__DIR__, "..", "src")
    allowed = ("`Gain` until 2026-09-10", "bitwise `Gain(gain = scale)`")
    found_in = String[]
    for (root, _, files) in walkdir(src_dir)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            for line in eachline(path)
                occursin("Gain", line) || continue
                any(occursin(a, line) for a in allowed) && continue
                push!(found_in, "$path: $line")
            end
        end
    end
    @test isempty(found_in)
end
