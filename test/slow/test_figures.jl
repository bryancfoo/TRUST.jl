using Test
using TRUST
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const FIGURE_NAMES = ("epw_ablation.png", "iaw_multiflow.png", "iaw_shock.png",
    "instrument_chain.png", "noise.png")
const FIGURES_DIR = joinpath(@__DIR__, "..", "..", "examples", "figures")
const MAKE_FIGURES = joinpath(FIGURES_DIR, "make_figures.jl")
const PNG_SIGNATURE = UInt8[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]

function _run_make_figures(outdir)
    run(`$(Base.julia_cmd()) --project=$FIGURES_DIR $MAKE_FIGURES $outdir`)
    return nothing
end

# One run serves FG1-FG3 (they check different properties of the same output); FG4 needs its own
# two fresh runs to compare across.
const _OUTDIR = mktempdir()
_run_make_figures(_OUTDIR)

@testset "FG1: make_figures.jl writes exactly the five files, into a given outdir" begin
    written = sort(readdir(_OUTDIR))
    @test written == sort(collect(FIGURE_NAMES))
end

@testset "FG2: every file is a PNG, 10 kB to 400 kB" begin
    for name in FIGURE_NAMES
        path = joinpath(_OUTDIR, name)
        sig = open(io -> read(io, 8), path)
        @test sig == PNG_SIGNATURE
        sz = filesize(path)
        @info "FG2" name sz
        @test 10_000 < sz < 400_000
    end
end

@testset "FG3: the five together are under 1.5 MB" begin
    total = sum(filesize(joinpath(_OUTDIR, name)) for name in FIGURE_NAMES)
    @info "FG3" total
    @test total < 1_500_000
end

@testset "FG4: reproducible -- same file sizes, and bitwise-identical underlying data" begin
    outdir1 = mktempdir()
    outdir2 = mktempdir()
    _run_make_figures(outdir1)
    _run_make_figures(outdir2)
    for name in FIGURE_NAMES
        sz1 = filesize(joinpath(outdir1, name))
        sz2 = filesize(joinpath(outdir2, name))
        @test sz1 == sz2
    end

    # The load-bearing half (spec D37): make_figures.jl writes only its five PNGs (FG1), so the
    # three noisy streaks it draws cannot be recovered from its output. What is tested directly is
    # the property that makes them reproducible in the first place -- synthesize() with the same
    # fixed seed (D24) on the same deterministic measure() pipeline gives the same image twice.
    for problem in (epw_ablation_problem(), iaw_multiflow_problem(), iaw_shock_problem())
        expt = build_experiment(problem)
        sd_a = synthesize(expt, problem.noise; rng=Xoshiro(problem.seed))
        sd_b = synthesize(expt, problem.noise; rng=Xoshiro(problem.seed))
        @test all(sd_a.image .=== sd_b.image)
    end
end
