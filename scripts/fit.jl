# julia --project scripts/fit.jl deck.toml -- the command-line entry point for `fit` (spec D33).
# Not part of the package (no include in src/TRUST.jl): it is a thin wrapper, not a component
# other code depends on.

using TRUST

length(ARGS) == 1 || error("usage: julia --project scripts/fit.jl deck.toml")
fit(ARGS[1])
