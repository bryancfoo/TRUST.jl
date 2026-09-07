module OTIS

include("constants.jl")
include("plasma_dispersion.jl")

# constants
export ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY, SPEED_OF_LIGHT,
       ATOMIC_MASS_UNIT, EV

# boundary conversions
export T_to_si, T_from_si, n_to_si, n_from_si, u_to_si, u_from_si,
       mass_to_si, mass_from_si, lambda_to_si, lambda_from_si,
       t_to_si, t_from_si, theta_to_si, theta_from_si

# plasma dispersion
export Z, Zprime

end
