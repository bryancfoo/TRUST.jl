using SafeTestsets

@safetestset "smoke" begin include("test_smoke.jl") end
# one line per test file, added as each step lands
