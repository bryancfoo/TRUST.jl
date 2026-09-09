using SafeTestsets

@safetestset "smoke" begin include("test_smoke.jl") end
@safetestset "constants" begin include("test_constants.jl") end
@safetestset "plasma dispersion" begin include("test_plasma_dispersion.jl") end
@safetestset "distribution interface" begin include("distributions/test_distribution.jl") end
@safetestset "Maxwellian" begin include("distributions/test_maxwellian.jl") end
@safetestset "SuperGaussian" begin include("distributions/test_supergaussian.jl") end
@safetestset "SuperGaussian table" begin include("distributions/test_supergaussian_table.jl") end
@safetestset "species" begin include("plasma/test_species.jl") end
@safetestset "plasma parameters" begin include("plasma/test_parameters.jl") end
@safetestset "plasma" begin include("plasma/test_plasma.jl") end
@safetestset "plasma trace" begin include("plasma/test_trace.jl") end
@safetestset "susceptibility" begin include("scattering/test_susceptibility.jl") end
@safetestset "spectral density" begin include("scattering/test_spectral_density.jl") end
@safetestset "scattering geometry" begin include("scattering/test_geometry.jl") end
@safetestset "spectrum" begin include("scattering/test_spectrum.jl") end
@safetestset "super-Gaussian spectrum" begin include("scattering/test_spectrum_supergaussian.jl") end
@safetestset "instrument effects" begin include("instrument/test_effects.jl") end
@safetestset "instrument background" begin include("instrument/test_background.jl") end
@safetestset "instrument experiment" begin include("instrument/test_experiment.jl") end
@safetestset "instrument measure" begin include("instrument/test_measure.jl") end
