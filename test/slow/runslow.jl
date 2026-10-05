# The slow tier's runner (conventions.md 8.6, spec D27): one @safetestset per slow file. This
# environment carries CairoMakie, which the fast suite (test/Project.toml) must never see
# (OQ-15), so this file is run by hand, never included from test/runtests.jl.
#
# Wrapped in one outer @testset (Phase 4.A, D30, carried into chunk 6), exactly as
# test/runtests.jl wraps its own gates (conventions.md 8.6, A32): a top-level @safetestset throws
# the instant it records a failure, which used to abort this script and silently skip every gate
# after it -- Phase 3.C's three known, untuned findings (FR2, FR4, FR6) fail "foo2026
# reproductions (cut-down)" third, so "streak averaging (slow)" (SA1) never ran in the normal
# invocation. A nested testset records the failure and lets its siblings run.
using SafeTestsets
using Test

@testset "OTIS (slow)" verbose = true begin
    @safetestset "Makie extension" begin include("test_makie_ext.jl") end
    @safetestset "figure gallery" begin include("test_figures.jl") end
    @safetestset "foo2026 reproductions (cut-down)" begin include("test_reproduce.jl") end
    @safetestset "streak averaging (slow)" begin include("test_average.jl") end
    @safetestset "two-channel problem (slow)" begin include("test_two_channel.jl") end
    @safetestset "staged fit (slow)" begin include("test_deck_staged.jl") end
    @safetestset "pixel integration (slow)" begin include("test_pixel_integration.jl") end
    @safetestset "resonance width (slow)" begin include("test_deck_resonance.jl") end
    @safetestset "tabulated k-smearing (slow)" begin include("test_tabulated_k_smearing.jl") end
    @safetestset "uncertainty (slow)" begin include("test_uncertainty.jl") end
end
