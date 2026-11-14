# Step G.2.2: the wide-dual figures beside the cost guard (Phase G.2 spec D14; OQ-69 item 6; the G.1 summary's A8),
# the slow tier's namesake of test/distributions/test_shape_cost.jl. Gate "shape cost (slow)": SCW1. Measured and
# reported, never asserted: one spectral_density call under a dual of N_P + 3 = 44 partials (N_P the default
# grid's 41 parameters), for every registered distribution, the dual seeded in the distribution's own parameters
# (parameter k in partial k) and, for one with no parameter (Maxwellian), in its species' T; the time per call, a
# throughput over 2000 calls, and the bytes one call allocates, behind a function barrier (conventions.md 5.2's
# measurement rules, 8.7). The fixture is the fast file's: n_e = 5e19 cm^-3, electrons carrying the distribution
# at 500 eV (a lab-frame one from itself alone), deuterons at 250 eV, 263.25 nm, 60 degrees, 245 nm. The one
# assertion is that the table covers SHAPE_REGISTRY.

using Test
using TRUST
using TRUST: SHAPE_REGISTRY, ELECTRON_MASS
const ForwardDiff = TRUST.ForwardDiff   # test/slow's environment has no ForwardDiff of its own

const _SCW_SHAPES = Dict{String,Any}(
    "Maxwellian"        => Maxwellian(),
    "SuperGaussian"     => SuperGaussian(3.0),
    "Kappa"             => Kappa(3.5),
    "GeneralizedNormal" => GeneralizedNormal(3.5),
    "SkewNormal"        => SkewNormal(0.7),
    "HermiteExpansion"  => HermiteExpansion(c3 = 0.02, c4 = 0.01),
    "GaussianGrid"      => GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 500.0),
)
const _SCW_NP = 44
const _SCW_N = 2000
const _SCW_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _SCW_COLLECTION = Collection(theta_deg = 60, k_model = :exact)

struct _SCWTag end
_scw_dual(value, k) = ForwardDiff.Dual{_SCWTag}(value, ForwardDiff.Partials(ntuple(i -> Float64(i == k), Val(_SCW_NP))))

# Every parameter of f dual (parameter k in partial k); a distribution with none keeps its plain self.
function _scw_seeded(f)
    p = parameters(f)
    isempty(p) && return f
    return reconstruct(f, NamedTuple{keys(p)}(ntuple(k -> _scw_dual(p[k], k), Val(length(p)))))
end
_scw_electrons(f::AbstractVelocityDistribution) =
    isempty(parameters(f)) ? electrons(T_eV = _scw_dual(500.0, 1), f = f) : electrons(T_eV = 500.0, f = _scw_seeded(f))
_scw_electrons(f::AbstractLabDistribution) = electrons(f = _scw_seeded(f))
_scw_plasma(f) = Plasma(n_e_cm3 = 5e19, electrons = (_scw_electrons(f),), ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))

_scw_consume(s::ForwardDiff.Dual) = ForwardDiff.value(s) + sum(ForwardDiff.partials(s))

function _scw_measure(plasma)
    (k, omega, _) = wavevector(_SCW_PROBE, _SCW_COLLECTION, 245e-9, plasma)
    barrier() = _scw_consume(spectral_density(plasma, k, omega))
    barrier()
    bytes = @allocated barrier()
    acc = 0.0
    t = @elapsed for _ in 1:_SCW_N
        acc += barrier()
    end
    isfinite(acc) || error("SCW1: a non-finite accumulator")
    return (bytes = bytes, us_per_call = t / _SCW_N * 1e6)
end

@testset "SCW1: one spectral_density call under a dual of 44 partials, every registered distribution (reported)" begin
    @test Set(keys(_SCW_SHAPES)) == Set(keys(SHAPE_REGISTRY))
    for name in sort(collect(keys(_SCW_SHAPES)))
        f = _SCW_SHAPES[name]
        r = _scw_measure(_scw_plasma(f))
        @info "SCW1 (latency in a loop of $(_SCW_N), barriered spectral_density, a dual of $(_SCW_NP) partials)" distribution = name seeded = (isempty(parameters(f)) ? "T" : join(string.(keys(parameters(f)))[1:min(end, 3)], ", ") * (length(parameters(f)) > 3 ? ", …" : "")) bytes = r.bytes us_per_call = r.us_per_call
    end
end
