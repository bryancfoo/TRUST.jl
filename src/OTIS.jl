module OTIS

include("constants.jl")
include("plasma_dispersion.jl")
include("distributions/distribution.jl")
include("distributions/maxwellian.jl")
include("plasma/species.jl")
include("plasma/parameters.jl")     # methods on Species and SpeciesSpec
include("plasma/plasma.jl")         # Plasma constructor uses Species and the unit table

# types
export AbstractVelocityDistribution, Maxwellian, SpeciesSpec, Species, SpeciesSet, Plasma

# shape interface
export dfdx, cauchy_dfdv, moment, skewness, excess_kurtosis, parameters, reconstruct

# plasma dispersion
export Z, Zprime

# species builders and accessors
export electrons, ion, spec
export thermal_speed, plasma_frequency, debye_length, alpha, xi, f_lab

# plasma quantities
export mean_charge_state, mean_ion_mass, neutrality_residual

# constants
export ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY, SPEED_OF_LIGHT,
       ATOMIC_MASS_UNIT, EV

# boundary conversions
export T_to_si, T_from_si, n_to_si, n_from_si, u_to_si, u_from_si,
       mass_to_si, mass_from_si, lambda_to_si, lambda_from_si,
       t_to_si, t_from_si, theta_to_si, theta_from_si

end
