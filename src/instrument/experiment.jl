# Experiment: a PlasmaTrace, a ScatteringGeometry, the detector axes, and the instrument chain
# in canonical order. conventions.md Section 4; spec D15.

using Printf: @printf

"""
    Experiment{FT<:Real, T<:PlasmaTrace{FT}, G<:ScatteringGeometry, L<:AbstractVector,
               A<:AbstractVector, E<:Tuple, O<:Tuple}

A `PlasmaTrace`, a `ScatteringGeometry`, the detector axes (`lambda` in m, `t` in s, both
strictly increasing), and the instrument chain **already sorted into canonical order** —
kernel effects, `SpectralIRF`, `TemporalIRF`, `Gain`, backgrounds
([OQ-13](../../dev/open_questions.md#oq-13)). `operators` is parallel to `effects`: the
value `_prepare` built for that effect, or `nothing`. `FT` is the trace's own element type;
the geometry, the axes and the effects may carry a different (possibly dual) type of their own
— `_streak_eltype` is where all of that is finally reconciled (spec D16).

Built with `Experiment(; trace, geometry, lambda_nm | lambda_m, t_ns | t_s = nothing, effects =
(), prefactor::Symbol = :first_order)`. The time axis is the trace's: if `t_ns`/`t_s` is
omitted, `expt.t` is `trace.axis`; if given, it must agree with `trace.axis` to `rtol = 1e-12`
(else an error naming the first bin that disagrees), and the *stored* axis is still
`trace.axis` — two sources of truth for one axis is the trap this closes.

Spec D15.
"""
struct Experiment{FT<:Real,T<:PlasmaTrace{FT},G<:ScatteringGeometry,L<:AbstractVector,
    A<:AbstractVector,E<:Tuple,O<:Tuple}
    trace::T
    geometry::G
    lambda::L
    t::A
    effects::E
    operators::O
    prefactor::Symbol
end

"""
    _si_axis(quantity::Symbol, kwargs) -> AbstractVector

Like `_si_value`, but for a vector-valued keyword: returns the given array **unchanged**
(`===`) when its unit keyword's factor is exactly `1` (the canonical SI keyword given
directly — `lambda_m`, `t_s`), and a freshly broadcast array otherwise. `_si_value` always
multiplies, which for a `Vector` allocates a new array even when the factor is `1.0` and so
never preserves identity; `Experiment`'s `expt.lambda` is required to be `===` the array the
user gave when it was already SI (EF12, Pitfall 11).
"""
function _si_axis(quantity::Symbol, kwargs)
    row = getfield(ACCEPTED_UNITS, quantity)
    present = Tuple(k for k in keys(row) if haskey(kwargs, k))
    if length(present) == 0
        throw(ArgumentError("no unit keyword given for $quantity; expected exactly one of $(keys(row))"))
    elseif length(present) > 1
        throw(ArgumentError("more than one unit keyword given for $quantity: $present; expected exactly one"))
    end
    key = present[1]
    factor = getfield(row, key)
    raw = kwargs[key]
    return isone(factor) ? raw : raw .* factor
end

function _check_lambda_axis(lambda::AbstractVector)
    isempty(lambda) && throw(ArgumentError("wavelength axis must be strictly increasing"))
    for l in 2:length(lambda)
        lambda[l] > lambda[l-1] || throw(ArgumentError("wavelength axis must be strictly increasing"))
    end
    return nothing
end

function _stable_sort_effects(effects::Tuple)
    v = collect(effects)
    sort!(v; by=_chain_order, alg=Base.Sort.MergeSort)
    return Tuple(v)
end

const _CHAIN_CLASS_NAME = Dict(2 => "SpectralIRF", 3 => "TemporalIRF", 4 => "Gain")

function _check_unique_chain_classes(effects::Tuple)
    counts = Dict{Int,Int}()
    for e in effects
        c = _chain_order(e)
        haskey(_CHAIN_CLASS_NAME, c) || continue
        counts[c] = get(counts, c, 0) + 1
    end
    for (c, n) in counts
        n > 1 && throw(ArgumentError("Experiment: only one $(_CHAIN_CLASS_NAME[c]) is allowed"))
    end
    return nothing
end

function Experiment(; trace::PlasmaTrace, geometry::ScatteringGeometry, effects=(),
    prefactor::Symbol=:first_order, kwargs...)
    t_row = getfield(ACCEPTED_UNITS, :t)
    t_given = any(k -> haskey(kwargs, k), keys(t_row))

    lambda = _si_axis(:lambda, kwargs)
    _reject_extra_kwargs(kwargs, (:lambda_nm, :lambda_m, :t_ns, :t_s))
    _prefactor_flag(prefactor)

    _check_lambda_axis(lambda)

    if t_given
        t_raw = _si_axis(:t, kwargs)
        length(t_raw) == length(trace.axis) || throw(ArgumentError(
            "Experiment: the given t has $(length(t_raw)) bins but the trace's axis has $(length(trace.axis))"))
        for j in eachindex(t_raw)
            isapprox(t_raw[j], trace.axis[j]; rtol=1e-12) || throw(ArgumentError(
                "Experiment: the given t disagrees with the trace's axis at bin $j"))
        end
    end
    t = trace.axis

    effects_tuple = Tuple(effects)
    sorted_effects = _stable_sort_effects(effects_tuple)
    _check_unique_chain_classes(sorted_effects)

    operators = map(e -> _prepare(e, lambda, t), sorted_effects)

    FT = eltype(trace)
    return Experiment{FT,typeof(trace),typeof(geometry),typeof(lambda),typeof(t),
        typeof(sorted_effects),typeof(operators)}(
        trace, geometry, lambda, t, sorted_effects, operators, prefactor)
end

"""
    Base.axes(expt::Experiment) -> (lambda, t)

A plain two-tuple of the coordinate vectors of `measure(expt)`'s two dimensions, in order.
Not re-exported: this is a method on `Base.axes`.

Spec D15.
"""
Base.axes(expt::Experiment) = (expt.lambda, expt.t)

# --- show (spec D15, D29) ---

function Base.show(io::IO, expt::Experiment)
    print(io, "Experiment($(length(expt.lambda)) x $(length(expt.t)), $(length(expt.effects)) effects)")
end

function Base.show(io::IO, ::MIME"text/plain", expt::Experiment)
    println(io, "Experiment")
    @printf(io, "  lambda: %e to %e nm (%d bins)\n", lambda_from_si(expt.lambda[1], :lambda_nm),
        lambda_from_si(expt.lambda[end], :lambda_nm), length(expt.lambda))
    @printf(io, "  t:      %e to %e ns (%d bins)\n", t_from_si(expt.t[1], :t_ns),
        t_from_si(expt.t[end], :t_ns), length(expt.t))
    print(io, "  geometry: ")
    show(io, expt.geometry)
    println(io)
    println(io, "  chain:")
    for e in expt.effects
        print(io, "    ")
        show(io, e)
        println(io, "  (islinear=$(islinear(e)))")
    end
end
