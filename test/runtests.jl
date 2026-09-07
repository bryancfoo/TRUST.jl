using SafeTestsets

@safetestset "smoke" begin include("test_smoke.jl") end
@safetestset "constants" begin include("test_constants.jl") end
@safetestset "plasma dispersion" begin include("test_plasma_dispersion.jl") end
# one line per test file, added as each step lands
