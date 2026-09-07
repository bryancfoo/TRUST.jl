using SafeTestsets

@safetestset "smoke" begin include("test_smoke.jl") end
@safetestset "constants" begin include("test_constants.jl") end
@safetestset "plasma dispersion" begin include("test_plasma_dispersion.jl") end
@safetestset "distribution interface" begin include("distributions/test_distribution.jl") end
@safetestset "Maxwellian" begin include("distributions/test_maxwellian.jl") end
# one line per test file, added as each step lands
