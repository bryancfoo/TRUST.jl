# The gridded identifiability script (Phase G.2 Step G.2.3; OQ-69 item 8), in the slow tier since Phase G.4's Step
# G.4.0 (the user, 2026-11-16, OQ-71 (6) and (7)). Gate "gridded identifiability (slow)": EX15, the fast suite's
# `examples` gate's row until then, unchanged; and EX16, the script on a channel whose amplitude is a Sensitivity
# scale per bin. Nothing is asserted of the script's numbers but EX16's comparison with the same problem's scalar
# scale.

using Test
using TRUST

# EX15 (Phase G.2 Step G.2.3, spec D16; OQ-69 item 8): the gridded identifiability script runs. Nothing is asserted of
# its numbers; that it runs end to end at a cut-down size (the EPW problem at 32 x 8, one chosen bin, one neighbour
# each side), writes its two tables, and returns counts and a sensitivity map of the right shapes, is.
include(joinpath(@__DIR__, "..", "..", "examples", "diagnostics", "gridded_identifiability.jl"))

@testset "EX15: the gridded identifiability script runs at a cut-down size" begin
    mktempdir() do dir
        r = run_problem(epw_ablation_problem(; nlambda=32, nt=8); outdir=dir, bins=[4], neighbours=1)
        @test isfile(joinpath(dir, "counts.tsv")) && isfile(joinpath(dir, "sensitivity_electrons_e.tsv"))
        @test r.species == ["electrons.e"] && r.bins == [4] && r.block == 76
        @test size(r.counts) == (1, 3) && size(r.shared) == (1, 3)
        @test all(r.counts .>= 0) && issorted(vec(r.counts); rev=true) && issorted(vec(r.shared); rev=true)
        @test size(r.sensitivity[1]) == (77, 1) && isnan(r.sensitivity[1][39, 1])
        @test count(isnan, r.sensitivity[1]) == 1 && all(>=(0), filter(!isnan, r.sensitivity[1]))
    end
end

# EX16 (Phase G.4 Step G.4.0, spec D3; OQ-71 (7)): the script on a channel whose amplitude is a Sensitivity scale per
# bin (Phase 4.I), which Experiment refused on a reading's window of bins until the scale was cut to it. A per-bin
# scale never stands beside a TemporalIRF (Experiment refuses it), so the EPW problem at 32 x 8 is taken without its
# temporal IRF and a reading's window is its own bins. (a) The scale rising by 1 % a bin: both readings run, with EX15's
# shapes. (b) The scale constant, against the same problem with that scale as one number: a reading of one bin has
# one amplitude, the scalar problem's, so its counts and sensitivities are the scalar problem's, bitwise; a reading of
# neighbours has one amplitude per bin, whose span contains the scalar amplitude's column, so no count and no
# sensitivity is above the scalar problem's.
@testset "EX16: the script with a Sensitivity scale per bin" begin
    base = epw_ablation_problem(; nlambda=32, nt=8)
    nt = nbins(base.trace)
    noirf = Tuple(e for e in base.effects if !(e isa TemporalIRF))
    scaled(s) = Tuple(e isa Sensitivity ? Sensitivity(scale=s) : e for e in noirf)
    mktempdir() do dir
        ra = run_problem(merge(base, (; effects=scaled([2.0e-4 * (1 + 0.01 * (j - 1)) for j in 1:nt])));
                         outdir=joinpath(dir, "a"), bins=[4], neighbours=1)
        @test isfile(joinpath(dir, "a", "counts.tsv"))
        @test ra.block == 76 && size(ra.counts) == (1, 3) && size(ra.shared) == (1, 3)
        @test all(ra.counts .>= 0) && issorted(vec(ra.counts); rev=true) && issorted(vec(ra.shared); rev=true)
        @test count(isnan, ra.sensitivity[1]) == 1 && all(>=(0), filter(!isnan, ra.sensitivity[1]))

        rs = run_problem(merge(base, (; effects=noirf)); outdir=joinpath(dir, "s"), bins=[4], neighbours=1)
        rb = run_problem(merge(base, (; effects=scaled(fill(2.0e-4, nt)))); outdir=joinpath(dir, "b"), bins=[4],
                         neighbours=1)
        @test rb.counts == rs.counts
        @test isequal(rb.sensitivity[1], rs.sensitivity[1])
        @test all(rb.shared .<= rs.shared)
        seen = .!isnan.(rs.shared_sensitivity[1])
        @test all(rb.shared_sensitivity[1][seen] .<= rs.shared_sensitivity[1][seen])
        @info "EX16 (b): the reading of neighbours, per-bin against scalar" rb.shared rs.shared maximum(rb.shared_sensitivity[1][seen] ./ rs.shared_sensitivity[1][seen])
    end
end
