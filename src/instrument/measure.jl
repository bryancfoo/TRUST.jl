# ideal_streak, measure and the per-bin kernel; foreach_bin! and BinCache, the loop over time
# bins, and pin_blas_threads. This is the only file in the package that mentions threads
# (conventions.md Section 6). Spec D16, D17.

using LinearAlgebra: mul!
using LinearAlgebra.BLAS: set_num_threads, get_num_threads
using OhMyThreads: @tasks, DynamicScheduler

"""
    _effect_eltype(e::AbstractInstrumentEffect) -> Type

`promote_type` over `values(parameters(e))`, or `Bool` (the promotion identity) for an effect
with no parameters. A value that is itself a `Tuple` (`PolynomialBackground`'s `coefficients`)
is flattened one level first, so every entry is promoted individually rather than the tuple's
own (non-numeric) type being fed to `promote_type`.
"""
function _effect_eltype(e::AbstractInstrumentEffect)
    p = values(parameters(e))
    isempty(p) && return Bool
    types = Type[]
    for v in p
        if v isa Tuple
            for vi in v
                push!(types, typeof(vi))
            end
        else
            push!(types, typeof(v))
        end
    end
    return promote_type(types...)
end

"""
    _streak_eltype(expt::Experiment) -> Type{<:Real}

The element type of `measure(expt)`/`ideal_streak(expt)`: `promote_type` of `eltype(expt.trace)`,
the geometry's own element type, `eltype(expt.lambda)`, `eltype(expt.t)`, `_effect_eltype(e)` for
every effect (plus `eltype(e.image)` for an `ImageBackground`). **This is the one place the
element type is decided; every cache and output buffer downstream is typed from it** (Pitfall
3).

Spec D16.
"""
function _streak_eltype(expt::Experiment)
    FT = promote_type(eltype(expt.trace), typeof(expt.probe.lambda), typeof(expt.collection.theta),
        eltype(expt.lambda), eltype(expt.t))
    for e in expt.effects
        FT = promote_type(FT, _effect_eltype(e))
        e isa ImageBackground && (FT = promote_type(FT, eltype(e.image)))
    end
    return FT
end

"""
    BinCache{FT<:Real}

The per-task scratch space for the per-bin kernel: a length-`N_λ` column buffer.
"""
mutable struct BinCache{FT<:Real}
    column::Vector{FT}
end
_make_cache(::Type{FT}, nlambda::Integer) where {FT} = BinCache{FT}(Vector{FT}(undef, nlambda))

"""
    foreach_bin!(kernel!, nbins::Integer, cache_factory; scheduler::Symbol = :auto)

Calls `kernel!(j, cache)` once for every `j` in `1:nbins`. `cache_factory()` is called once per
task and its result passed to every call that task makes. **This is the only function in the
package that mentions threads or chunking** (`conventions.md` 6.2); caches are per task, never
selected by the running task's own thread number (`conventions.md` 6.3, Pitfall 1: never index
a cache by the identifier `Threads` assigns the current thread). `scheduler` is `:auto` (the
default) or `:serial`.

`:serial`, or `:auto` with `Threads.nthreads() == 1`, runs a plain loop with one cache. `:auto`
with more than one thread runs OhMyThreads' `@tasks` with a `DynamicScheduler` and a per-task
`@local` cache — **verified bitwise identical to the serial loop** on this machine at `-t 4`
with OhMyThreads 0.8.6 (Level 2, 2026-09-09; TH1-TH3 repeat the check here). No reduction
crosses bins in this phase (every bin writes its own column of a preallocated array), so the
bitwise identity is a property of the design, not of the scheduler.

Spec D17.
"""
function foreach_bin!(kernel!, nbins::Integer, cache_factory; scheduler::Symbol=:auto)
    scheduler in (:auto, :serial) || throw(ArgumentError(
        "foreach_bin!: scheduler must be :auto or :serial; got :$scheduler"))
    if scheduler === :serial || Threads.nthreads() == 1
        cache = cache_factory()
        for j in 1:nbins
            kernel!(j, cache)
        end
    else
        @tasks for j in 1:nbins
            @set scheduler = DynamicScheduler()
            @local cache = cache_factory()
            kernel!(j, cache)
        end
    end
    return nothing
end

"""
    pin_blas_threads(n::Integer = 1) -> Int

Sets BLAS to `n` threads and returns the count it had before. OTIS threads over time bins
itself (`foreach_bin!`); BLAS on many threads oversubscribes the machine against that, and
`conventions.md` 6.6 asks for BLAS on one thread "in the driver" — `measure` has no driver of
its own in this phase and never calls this implicitly, so call it yourself before a large
`measure` (TH3 calls it before comparing threaded and serial results, since a multithreaded
BLAS `mul!` may reassociate).
"""
function pin_blas_threads(n::Integer=1)
    previous = get_num_threads()
    set_num_threads(n)
    return previous
end

"""
    _kernel_effects(expt::Experiment) -> Tuple

The leading run of `expt.effects` with `_chain_order == 1` (a prefix, since the tuple is
sorted): the in-kernel effects, extracted once per streak, not per bin.
"""
function _kernel_effects(expt::Experiment)
    effects = expt.effects
    n = 0
    for e in effects
        _chain_order(e) == 1 || break
        n += 1
    end
    return effects[1:n]
end

"""
    _bin_kernel!(P::AbstractMatrix, expt::Experiment, j::Integer, cache::BinCache)

The per-bin kernel: materializes `plasma(expt.trace, j)`, zeroes and accumulates into
`cache.column` over the in-kernel effects' samples, then writes the column into `P[:, j]`.
Allocation-free in the value pass and the dual pass alike (XP13).

Spec D16.
"""
function _bin_kernel!(P::AbstractMatrix, expt::Experiment, j::Integer, cache::BinCache)
    p = plasma(expt.trace, j)
    col = cache.column
    fill!(col, zero(eltype(col)))
    _accumulate!(col, p, expt.probe, expt.collection, expt.lambda, _kernel_effects(expt),
        one(eltype(col)), _prefactor_flag(expt.prefactor))
    @inbounds for l in eachindex(col)
        P[l, j] = col[l]
    end
    return P
end

function _accumulate!(col, p, probe, collection, lambda, ::Tuple{}, w, first_order)
    @inbounds for l in eachindex(lambda)
        col[l] += w * _spectrum_at(p, probe, collection, lambda[l], first_order)
    end
    return col
end

function _accumulate!(col, p, probe, collection, lambda, effects::Tuple, w, first_order)
    e = first(effects)
    rest = Base.tail(effects)
    for (wq, pq, probeq, collectionq) in samples(e, p, probe, collection)
        _accumulate!(col, pq, probeq, collectionq, lambda, rest, w * wq, first_order)
    end
    return col
end

"""
    ideal_streak!(P::AbstractMatrix, expt::Experiment; scheduler::Symbol = :auto) -> P

Fills `P` (`N_λ x N_t`, wavelength first so a column is a contiguous view, Pitfall 15) with the
ideal (pre-instrument) streak, one column per bin via `foreach_bin!`.

Spec D16.
"""
function ideal_streak!(P::AbstractMatrix, expt::Experiment; scheduler::Symbol=:auto)
    nl, nt_ = length(expt.lambda), length(expt.t)
    size(P) == (nl, nt_) || throw(ArgumentError(
        "ideal_streak!: P is $(size(P)) but the experiment's axes are $nl x $nt_"))
    FT = eltype(P)
    cache_factory = () -> _make_cache(FT, nl)
    foreach_bin!(nbins(expt.trace), cache_factory; scheduler=scheduler) do j, cache
        _bin_kernel!(P, expt, j, cache)
    end
    return P
end

"""
    ideal_streak(expt::Experiment; scheduler::Symbol = :auto) -> Matrix

Allocates `Matrix{_streak_eltype(expt)}(undef, N_λ, N_t)` and calls `ideal_streak!`.

Spec D16.
"""
function ideal_streak(expt::Experiment; scheduler::Symbol=:auto)
    FT = _streak_eltype(expt)
    P = Matrix{FT}(undef, length(expt.lambda), length(expt.t))
    return ideal_streak!(P, expt; scheduler=scheduler)
end

_needs_destination(::AbstractInstrumentEffect) = false
_needs_destination(::SpectralIRF) = true
_needs_destination(::TemporalIRF) = true

apply!(dest, src, ::SpectralIRF, op) = mul!(dest, op, src)
apply!(dest, src, ::TemporalIRF, op) = mul!(dest, src, transpose(op))
function apply!(dest, src, e::Sensitivity{FT,Nothing,Nothing}, op) where {FT}
    dest === src || copyto!(dest, src)
    dest .*= e.scale
    return dest
end
function apply!(dest, src, e::Sensitivity, op)
    dest === src || copyto!(dest, src)
    dest .*= e.scale .* op
    return dest
end
function apply!(dest, src, e::ConstantBackground, op)
    dest === src || copyto!(dest, src)
    dest .+= e.level
    return dest
end
function apply!(dest, src, e::PolynomialBackground, op)
    dest === src || copyto!(dest, src)
    dest .+= op
    return dest
end
function apply!(dest, src, e::RationalBackground, op)
    dest === src || copyto!(dest, src)
    dest .+= op
    return dest
end
function apply!(dest, src, e::ImageBackground, op)
    size(e.image) == size(src) || throw(ArgumentError(
        "ImageBackground: background image is $(size(e.image)) but the streak is $(size(src))"))
    dest === src || copyto!(dest, src)
    dest .+= e.scale .* e.image
    return dest
end

"""
    measure(expt::Experiment; scheduler::Symbol = :auto) -> Matrix

The measured streak: `ideal_streak(expt)` walked through the detector effects (`_chain_order >
1`) in canonical order via `apply!`, swapping between two buffers only where `SpectralIRF`/
`TemporalIRF` need a separate destination (`_needs_destination`). Does not touch BLAS (spec
D17's `pin_blas_threads` is the caller's job, from Step 2.3).

Spec D16.
"""
function measure(expt::Experiment; scheduler::Symbol=:auto)
    P = ideal_streak(expt; scheduler=scheduler)
    spare = similar(P)
    cur, buf = P, spare
    for i in eachindex(expt.effects)
        e = expt.effects[i]
        _chain_order(e) == 1 && continue
        op = expt.operators[i]
        if _needs_destination(e)
            apply!(buf, cur, e, op)
            cur, buf = buf, cur
        else
            apply!(cur, cur, e, op)
        end
    end
    return cur
end
