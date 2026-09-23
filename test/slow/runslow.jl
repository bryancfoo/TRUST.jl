# The slow tier's runner (conventions.md 8.6, spec D27): one @safetestset per slow file. This
# environment carries CairoMakie, which the fast suite (test/Project.toml) must never see
# (OQ-15), so this file is run by hand, never included from test/runtests.jl.
using SafeTestsets

@safetestset "Makie extension" begin include("test_makie_ext.jl") end
@safetestset "figure gallery" begin include("test_figures.jl") end
@safetestset "foo2026 reproductions (cut-down)" begin include("test_reproduce.jl") end
@safetestset "streak averaging (slow)" begin include("test_average.jl") end
