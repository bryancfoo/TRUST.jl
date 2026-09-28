# Step 3.C.5: the shape-parameter derivative contract and the cost guard (spec D11, D12;
# dev/phase3C_instructions.md Section 8). Iterates SHAPE_REGISTRY and is shape-agnostic by
# construction: Kappa and HermiteExpansion enter this guard the day Step 4.2 registers them,
# which is the entire point of the step. conventions.md 8.1: a new file, distinct from the shape
# files it walks. SC1-SC4, exactly as Section 8's Test table specifies them.

using Test
using OTIS
using OTIS: SHAPE_REGISTRY, _SHAPE_OPTIONS, _build_shape
import ForwardDiff

# --- The representative-instance table (spec D11's forcing function; corrected 2026-09-18,
# Section 8's Build text). SHAPE_REGISTRY maps a name to a constructor, and there is no generic
# way to learn a shape's valid parameter range, or even its constructor's arity, from the
# registry alone (Maxwellian() takes nothing, SuperGaussian(m) takes one positional argument).
# SC1 asserts this table covers SHAPE_REGISTRY exactly, so registering a new shape fails this
# test until someone adds it here at a representative parameter value inside its own valid
# range -- that failure is the mechanism by which Step 4.2's shapes get measured the day they
# are written. ---

const _SC_SHAPES = Dict{String,Any}(
    "Maxwellian"    => Maxwellian(),
    "SuperGaussian" => SuperGaussian(3.0),
    "Kappa"         => Kappa(3.5),  # mid-domain in s (0.286), not an integer: coincides with no oracle point
    "GeneralizedNormal" => GeneralizedNormal(3.5),  # mid-domain, and no oracle point of GN10
    "SkewNormal"        => SkewNormal(0.7),  # inside the table, off every oracle point
)

# --- The fixture: the package's standard collective EPW recipe (test_supergaussian_dm.jl's
# DM12, and chunk 3's own Section 12 row 6, same n_e/T_e/T_i/geometry), one electron species
# carrying the shape under test, one deuteron species, one (k, omega) inside the EPW feature. ---

const _SC_PROBE = ProbeBeam(lambda_probe_nm = 263.25)
const _SC_COLLECTION = Collection(theta_deg = 60, k_model = :exact)
const _SC_LAMBDA_S = 245e-9  # the midpoint of the 512-point axis (235e-9, 255e-9) RF1a/DP use

_sc_plasma(f, T_eV) = Plasma(n_e_cm3 = 5e19,
                              electrons = (electrons(T_eV = T_eV, f = f),),
                              ions = (ion(mass_amu = 2.014, charge = 1, T_eV = 250.0),))

function _sc_fixture_alpha()
    plasma = _sc_plasma(Maxwellian(), 500.0)
    (k, _, _) = wavevector(_SC_PROBE, _SC_COLLECTION, _SC_LAMBDA_S, plasma)
    return alpha(plasma.electrons[1], k)
end

# --- Seeding the dual through reconstruct (conventions.md 8.11), never by hand: this is the
# path a fit takes, and the only way a fixture can see the thing being measured. Both duals
# carry three partials, one of them nonzero, matching chunk 3's own fixture, so the p-dual and
# T-dual calls run the identical code shape and only the ratio means something. ---

const _SCTag = :ShapeCost
_sc_dual(value) = ForwardDiff.Dual{_SCTag}(value, 1.0, 0.0, 0.0)
_sc_seed(f, p::Symbol, dual) = reconstruct(f, merge(parameters(f), NamedTuple{(p,)}((dual,))))

# --- The barrier (conventions.md 5.2's two measurement rules): every measured call consumes the
# result as a number, value and partials, so nothing can be elided. Wrapped in a function so that
# plasma/k/omega are locals, never a non-const global (a top-level closure over a global is what
# measures dynamic dispatch instead of the call). ---

_sc_consume(s::ForwardDiff.Dual) = ForwardDiff.value(s) + sum(ForwardDiff.partials(s))
_sc_consume(s::Real) = s

function _sc_allocated(plasma)
    (k, omega, _) = wavevector(_SC_PROBE, _SC_COLLECTION, _SC_LAMBDA_S, plasma)
    barrier() = _sc_consume(spectral_density(plasma, k, omega))
    barrier()                      # warm up
    return @allocated barrier()
end

function _sc_throughput(plasma, N::Int)
    (k, omega, _) = wavevector(_SC_PROBE, _SC_COLLECTION, _SC_LAMBDA_S, plasma)
    barrier() = _sc_consume(spectral_density(plasma, k, omega))
    barrier()                      # warm up
    acc = 0.0
    elapsed = @elapsed for _ in 1:N
        acc += barrier()
    end
    isfinite(acc) || error("_sc_throughput: accumulator went non-finite on $(typeof(plasma))")
    return elapsed / N * 1e9       # ns/call -- a throughput, not a latency (conventions.md 8.7)
end

# --- One measurement per (shape, parameter): SC2's allocation figure, SC3's ratio in both
# orders, SC4's T-dual/plain ratio, all from the same three plasmas so nothing is rebuilt
# between rows (pitfall 3: the T-dual control differs from the p-dual case in exactly one
# thing -- same tag, same three partials, same plasma recipe). ---

const _SC_N = 50_000  # chunk 3's own loop count for this call (Section 12 row 6)

function _sc_measure(name::String, f, p::Symbol, N::Int)
    v0 = parameters(f)[p]
    f_pdual = _sc_seed(f, p, _sc_dual(v0))

    plasma_plain = _sc_plasma(f, 500.0)
    plasma_tdual = _sc_plasma(f, _sc_dual(500.0))
    plasma_pdual = _sc_plasma(f_pdual, 500.0)

    allocated_pdual = _sc_allocated(plasma_pdual)

    # Back-to-back, then reversed (D11: "the larger ratio is the one asserted").
    t_pdual_1 = _sc_throughput(plasma_pdual, N)
    t_tdual_1 = _sc_throughput(plasma_tdual, N)
    t_tdual_2 = _sc_throughput(plasma_tdual, N)
    t_pdual_2 = _sc_throughput(plasma_pdual, N)

    # SC4's baseline, in both orders on its own four calls (Phase 4.B spec D28: one plain
    # measurement read 0.665 at fdc1b7d). A bound fails only when both orders agree.
    t_tdual_3 = _sc_throughput(plasma_tdual, N)
    t_plain_1 = _sc_throughput(plasma_plain, N)
    t_plain_2 = _sc_throughput(plasma_plain, N)
    t_tdual_4 = _sc_throughput(plasma_tdual, N)

    ratio_1 = t_pdual_1 / t_tdual_1
    ratio_2 = t_pdual_2 / t_tdual_2

    return (shape = name, parameter = p, allocated_pdual = allocated_pdual,
            t_pdual_1 = t_pdual_1, t_tdual_1 = t_tdual_1, ratio_1 = ratio_1,
            t_tdual_2 = t_tdual_2, t_pdual_2 = t_pdual_2, ratio_2 = ratio_2,
            ratio_pt = max(ratio_1, ratio_2),
            t_tdual_3 = t_tdual_3, t_plain_1 = t_plain_1, ratio_tp_1 = t_tdual_3 / t_plain_1,
            t_plain_2 = t_plain_2, t_tdual_4 = t_tdual_4, ratio_tp_2 = t_tdual_4 / t_plain_2)
end

const _SC_RESULTS = let
    out = NamedTuple[]
    for name in sort(collect(keys(_SC_SHAPES)))
        f = _SC_SHAPES[name]
        ps = sort(collect(keys(parameters(f))))
        isempty(ps) && continue
        for p in ps
            push!(out, _sc_measure(name, f, p, _SC_N))
        end
    end
    out
end

# --- SC1: the registry was actually walked, and the guard cannot silently stop covering it ---

@testset "SC1: the registry was actually walked" begin
    registry_keys = Set(keys(SHAPE_REGISTRY))
    table_keys = Set(keys(_SC_SHAPES))
    missing_keys = setdiff(registry_keys, table_keys)
    @test isempty(missing_keys) || error(
        "SC1: SHAPE_REGISTRY has registered shape(s) $(sort(collect(missing_keys))) not " *
        "covered by _SC_SHAPES; add each to _SC_SHAPES at a representative parameter value " *
        "inside its own valid range.")
    @test table_keys == registry_keys

    with_params = sort([name for (name, f) in _SC_SHAPES if !isempty(parameters(f))])
    without_params = sort([name for (name, f) in _SC_SHAPES if isempty(parameters(f))])
    @info "SC1: registered shapes, with and without fittable parameters" with_params without_params
    @test !isempty(with_params)   # else the guard below measures nothing (D11)

    alpha_e = _sc_fixture_alpha()
    @info "SC1: the fixture's alpha_e (must stay collective, > 1)" alpha_e
    @test alpha_e > 1
end

# --- SC2: zero allocation under a shape-parameter dual (the deterministic half; rule 5.2's
# dual-pass clause with the m-exemption withdrawn) ---

@testset "SC2: zero allocation under a shape-parameter dual" begin
    @test !isempty(_SC_RESULTS)  # the loop must have measured something (SC1's own guard, restated)
    for r in _SC_RESULTS
        @info "SC2 (allocation, barriered spectral_density(plasma, k, omega))" shape=r.shape parameter=r.parameter allocated_bytes=r.allocated_pdual
        @test r.allocated_pdual == 0
    end
end

# --- SC3: the throughput ratio, back-to-back in both orders, the larger asserted. Deliberately
# wide (22x measured before Step 3.C.4, about 1.2x expected after); this is the suite's first
# and only timing assertion, admitted by conventions.md 8.7 as amended 2026-09-17 (OQ-39): a
# ratio of two throughputs measured back-to-back in one process does not vary the way an
# absolute time does. ---

@testset "SC3: the p-dual/T-dual throughput ratio, back-to-back both orders" begin
    for r in _SC_RESULTS
        @info "SC3 (throughput, N=$(_SC_N), ns/call, barriered spectral_density(plasma, k, omega))" shape=r.shape parameter=r.parameter t_pdual_order1=r.t_pdual_1 t_tdual_order1=r.t_tdual_1 ratio_order1=r.ratio_1 t_tdual_order2=r.t_tdual_2 t_pdual_order2=r.t_pdual_2 ratio_order2=r.ratio_2 asserted_ratio=r.ratio_pt
        @test r.ratio_pt <= 5
    end
end

# --- SC4: the baseline is a real baseline -- catches a "T-dual" that is not dual or a "plain"
# call that is not plain, which would make SC3 meaningless. Deliberately very wide: a fixture
# sanity check, not a performance bound. Measured in both orders on its own four calls since
# Phase 4.B spec D28 (2026-09-26): the lower bound on the larger ratio, the upper on the smaller,
# so one noisy measurement cannot fail it. ---

@testset "SC4: the T-dual/plain ratio is a real baseline" begin
    for r in _SC_RESULTS
        @info "SC4 (throughput, N=$(_SC_N), ns/call, barriered spectral_density(plasma, k, omega))" shape=r.shape parameter=r.parameter t_tdual_order1=r.t_tdual_3 t_plain_order1=r.t_plain_1 ratio_order1=r.ratio_tp_1 t_plain_order2=r.t_plain_2 t_tdual_order2=r.t_tdual_4 ratio_order2=r.ratio_tp_2
        @test max(r.ratio_tp_1, r.ratio_tp_2) >= 1
        @test min(r.ratio_tp_1, r.ratio_tp_2) <= 20
    end
end

# --- SC5: a shape registered in one name table and forgotten in the other (Phase 4.B spec D19).
# _SHAPE_OPTIONS (the deck layer) and SHAPE_REGISTRY (io.jl's read path) are not folded into one
# table (summary B4); this is the guard that catches the one failure mode a fold would prevent. ---

@testset "SC5: _SHAPE_OPTIONS and SHAPE_REGISTRY name the same shapes" begin
    @test Set(keys(_SHAPE_OPTIONS)) == Set(keys(SHAPE_REGISTRY))
    for (name, ctor) in SHAPE_REGISTRY
        @test ctor isa Type
        @test hasmethod(_build_shape, Tuple{Type{ctor},Dict{String,Float64}})
    end
end
