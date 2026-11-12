# The lab-frame family of distributions and its first member, GaussianGrid: the reduced distribution of
# one species along k-hat as a sum of Gaussians of one width at evenly spaced, fixed velocity nodes, with
# positive weights w_n = w0_n exp(g_n) against a default distribution; the g_n are the parameters. Its
# Cauchy integral and every parameter derivative are closed form through Zprime, no table.
# See docs/derivations/gaussian_grid.md. Phase G.1 Step G.1.1, spec D3-D7 (OQ-68).

"""
    AbstractLabDistribution

The abstract type of a **lab-frame distribution**: the reduced distribution of one species along **k̂**
as a function of lab velocity, `f(v)` in s/m, with `∫ f(v) dv = 1` (`conventions.md` 1.12). It is
**not** an `AbstractVelocityDistribution`: it is not a dimensionless shape, the second and third
normalizations of `conventions.md` 1.5 do not apply to it, and nothing about it is an exception to them.
A `Species` carries either family in its distribution slot `f`; for one carrying a lab-frame
distribution, `u` and `T` are the distribution's mean velocity and variance temperature, computed
wherever the species is built and never given, varied or fitted.

A member provides the call `f(v)`, `dfdv(f, v)`, `cauchy_dfdv(f, v)` (the Cauchy integral in lab
velocity, `𝒞f(v_φ) = ∫ f′(v)/(v − v_φ) dv` on the Landau contour, in s²/m²), `moment(f, n) = ∫ vⁿ f dv`,
`mean_velocity(f)`, `velocity_variance(f)`, `parameters(f)` and `reconstruct(f, values)`. The first and
only member is `GaussianGrid`.

`conventions.md` 1.12, Section 4; [OQ-68](../../dev/open_questions.md#oq-68) item 1;
`docs/derivations/gaussian_grid.md`.
"""
abstract type AbstractLabDistribution end

"""
    dfdv(f::AbstractLabDistribution, v) -> Number

The derivative `f′(v)` of a lab-frame distribution, in s²/m², at the lab velocity `v` (m/s). The
lab-frame family's counterpart of a shape's `dfdx`, whose argument is the reduced velocity; there is no
generic method.

`conventions.md` 1.12, Section 4.
"""
function dfdv end

"""
    mean_velocity(f::AbstractLabDistribution) -> Real

The mean velocity `ū = ∫ v f(v) dv` of a lab-frame distribution, in m/s: the `u` of a species that
carries it (`conventions.md` 1.12).
"""
function mean_velocity end

"""
    velocity_variance(f::AbstractLabDistribution) -> Real

The variance `V = ∫ (v − ū)² f(v) dv` of a lab-frame distribution, in m²/s²: a species of mass `m` that
carries it has `T = m V`, its variance temperature along **k̂** (`conventions.md` 1.12).
"""
function velocity_variance end

"""
    GaussianGrid{FT<:Real, Nv, NP} <: AbstractLabDistribution
    GaussianGrid(; mass_amu | mass_kg, T_ref_eV | T_ref_keV, u0_kms | u0_ms = 0, nodes = 41,
                 spacing = 0.25, width = 1.2, symmetric = false)
    GaussianGrid(; dv_kms | dv_ms, default_weights, u0_kms | u0_ms = 0, width = 1.2, symmetric = false)

A distribution in lab velocity on a fixed grid: `Nv` Gaussians of one width `σ` at evenly spaced nodes,
with positive weights,

    f(v) = (1/m₀) Σₙ wₙ G(v; vₙ, σ),    G(v; c, s) = e^{−((v − c)/s)²} / (s √π),    m₀ = Σₙ wₙ,
    vₙ = u₀ + (n − (Nv + 1)/2) dv,    σ = width · dv,    wₙ = w⁰ₙ exp(gₙ),

so `f` has unit integral for any weights and is positive everywhere. **The `gₙ` are its parameters**,
`g1` to `gNv` in node order (`parameters`); `g = 0` is the default distribution `w⁰`, and adding one
constant to every `gₙ` changes nothing, since `f` is divided by `m₀`. **The grid is fixed**: the centre,
the spacing, the width and the default weights are fields, never parameters, the same at every time bin
of a fit ([OQ-68](../../dev/open_questions.md#oq-68) item 2, the user's decision).

**The temperature form** (the user's inputs): the centre `u₀`, a reference temperature `T_ref`, the
number of nodes and the spacing as a fraction of the reference thermal speed `v_ref = √(2 T_ref/m)` of
the species' mass `m`, so `dv = spacing · v_ref`. Its default is **the Maxwellian of temperature `T_ref`
and flow `u₀`**, written on the grid in closed form, `w⁰ₙ = dv · G(vₙ; u₀, s₀)` with `s₀ = √(v_ref² −
σ²)`: a Gaussian narrowed by exactly the width the comb's own Gaussians add back, so `spacing · width`
must be below 1. The defaults (41 nodes at 0.25 `v_ref` with `width = 1.2`) span ±5 `v_ref` and
represent that Maxwellian's Cauchy integral to about 1e-4 of its peak (`docs/derivations/gaussian_grid.md`
Section 6).

**The explicit form** takes the spacing `dv` itself and the default weights `default_weights`
(any `Nv ≥ 3` finite positive numbers; their scale is irrelevant), with no temperature.

**The symmetric option** (off by default): `wₙ = w_{Nv+1−n}`, the parameters then `g1` to `g⌈Nv/2⌉`
(`NP = cld(Nv, 2)`), so `f` is even about `u₀`. The explicit form's `default_weights` must then be
symmetric themselves.

**Refused**, each naming the input and its value: fewer than 3 nodes; a spacing, width, `dv`, `T_ref` or
mass that is not finite and positive; `spacing · width ≥ 1` in the temperature form; a non-finite `u₀`;
a default weight that is not finite and positive; asymmetric default weights with `symmetric = true`;
both forms' keywords at once, or neither; and, from `reconstruct`, a non-finite `g` or weights whose sum
is not finite and positive. The grid's inputs are plain numbers (`Float64` fields); `FT` is the
parameters' element type, so dual `g` make the distribution, and a species that carries it, dual.

`FT` is the element type of `g`, the cached weights `w` and `m₀`; `Nv` the number of nodes; `NP` the
number of parameters (`Nv`, or `cld(Nv, 2)` with the symmetric option). Every field is `isbits`.

`conventions.md` 1.12, Section 4; [OQ-68](../../dev/open_questions.md#oq-68) items 1 to 4; Phase G.1
spec D3, D4; `docs/derivations/gaussian_grid.md`.
"""
struct GaussianGrid{FT<:Real, Nv, NP} <: AbstractLabDistribution
    u0::Float64
    dv::Float64
    sigma::Float64
    w0::NTuple{Nv, Float64}
    g::NTuple{NP, FT}
    w::NTuple{Nv, FT}
    m0::FT
    function GaussianGrid{FT, Nv, NP}(u0::Float64, dv::Float64, sigma::Float64, w0::NTuple{Nv, Float64},
                                      g::NTuple{NP, FT}) where {FT<:Real, Nv, NP}
        (NP == Nv || NP == cld(Nv, 2)) || throw(ArgumentError(
            "GaussianGrid: $NP parameters for $Nv nodes; expected $Nv, or $(cld(Nv, 2)) with the symmetric option"))
        for i in 1:NP
            isfinite(g[i]) || throw(ArgumentError("GaussianGrid: g$i must be finite; got $(g[i])"))
        end
        w = ntuple(n -> w0[n] * exp(g[_grid_parameter(n, Val(Nv), Val(NP))]), Val(Nv))
        m0 = w[1]
        for n in 2:Nv
            m0 += w[n]
        end
        (isfinite(m0) && _primal_value(m0) > 0) || throw(ArgumentError(
            "GaussianGrid: the weights w0 exp(g) sum to $m0; the sum must be finite and positive"))
        return new{FT, Nv, NP}(u0, dv, sigma, w0, g, w, m0)
    end
end

# The parameter that node n reads: n itself, or the nearer of n and its mirror Nv + 1 - n with the
# symmetric option (Phase G.1 spec D3). Nv and NP are type parameters, so this folds at compile time.
_grid_parameter(n::Int, ::Val{Nv}, ::Val{NP}) where {Nv, NP} = NP == Nv ? n : min(n, Nv + 1 - n)

# The node v_n = u0 + (n - (Nv + 1)/2) dv, in m/s (OQ-68 item 2).
_grid_node(f::GaussianGrid{FT, Nv}, n::Int) where {FT, Nv} = f.u0 + (n - (Nv + 1) / 2) * f.dv

"""
    _grid_names(::Val{NP}) -> NTuple{NP, Symbol}

The parameter names `(:g1, :g2, …, :gNP)`, a compile-time constant (a generated function, so that
`parameters` and `reconstruct` build no `Symbol` at run time and allocate nothing).
"""
@generated _grid_names(::Val{NP}) where {NP} = Tuple(Symbol("g", i) for i in 1:NP)

# The promoted float type of a NamedTuple's values, at compile time (Phase G.1 spec D3).
@generated _grid_values_eltype(::Type{NamedTuple{K, V}}) where {K, V} = float(promote_type(fieldtypes(V)...))

# --- the keyword constructors (Phase G.1 spec D4) ---

function GaussianGrid(; width::Real = 1.2, symmetric::Bool = false, kwargs...)
    has_T = haskey(kwargs, :T_ref_eV) || haskey(kwargs, :T_ref_keV)
    has_dv = haskey(kwargs, :dv_kms) || haskey(kwargs, :dv_ms)
    (has_T && has_dv) && throw(ArgumentError(
        "GaussianGrid: give a reference temperature (T_ref_eV or T_ref_keV) or a velocity spacing " *
        "(dv_kms or dv_ms), not both"))
    has_T && return _gaussian_grid_from_temperature(width, symmetric, kwargs)
    has_dv && return _gaussian_grid_from_spacing(width, symmetric, kwargs)
    throw(ArgumentError(
        "GaussianGrid: no grid given; expected T_ref_eV or T_ref_keV with mass_amu or mass_kg (the " *
        "temperature form), or dv_kms or dv_ms with default_weights (the explicit form)"))
end

_grid_positive(name::String, x::Real) = (isfinite(x) && x > 0) || throw(ArgumentError(
    "GaussianGrid: $name must be finite and positive; got $x"))

function _grid_centre(kwargs)
    u0 = Float64(_si_value(:u0, kwargs, 0.0))
    isfinite(u0) || throw(ArgumentError("GaussianGrid: u0 must be finite; got $u0 m/s"))
    return u0
end

function _grid_nodes(nodes)
    (nodes isa Integer && nodes >= 3) || throw(ArgumentError(
        "GaussianGrid: nodes must be an integer of at least 3; got $nodes"))
    return Int(nodes)
end

function _gaussian_grid_from_temperature(width::Real, symmetric::Bool, kwargs)
    _reject_extra_kwargs(kwargs, (:mass_amu, :mass_kg, :T_ref_eV, :T_ref_keV, :u0_kms, :u0_ms, :nodes,
                                  :spacing, :width, :symmetric))
    mass = Float64(_si_value(:mass, kwargs))
    T_ref = Float64(_si_value(:T_ref, kwargs))
    u0 = _grid_centre(kwargs)
    Nv = _grid_nodes(get(kwargs, :nodes, 41))
    spacing = Float64(get(kwargs, :spacing, 0.25))
    w = Float64(width)
    _grid_positive("mass", mass)
    _grid_positive("T_ref", T_ref)
    _grid_positive("spacing", spacing)
    _grid_positive("width", w)
    spacing * w < 1 || throw(ArgumentError(
        "GaussianGrid: spacing · width must be below 1 for the Maxwellian default (its width " *
        "√(v_ref² − σ²) needs σ < v_ref); got spacing = $spacing, width = $w"))
    v_ref = sqrt(2 * T_ref / mass)    # the reference thermal speed, conventions.md 1.4's v_th at T_ref
    dv = spacing * v_ref
    sigma = w * dv
    s0 = sqrt(v_ref^2 - sigma^2)
    # w0_n = dv G(v_n; u0, s0), written from the offset (n - (Nv+1)/2) dv so that it is exactly symmetric
    w0 = ntuple(n -> dv * exp(-(((n - (Nv + 1) / 2) * dv) / s0)^2) / (s0 * sqrt(pi)), Nv)
    return _gaussian_grid(u0, dv, sigma, w0, symmetric)
end

function _gaussian_grid_from_spacing(width::Real, symmetric::Bool, kwargs)
    _reject_extra_kwargs(kwargs, (:dv_kms, :dv_ms, :default_weights, :u0_kms, :u0_ms, :width, :symmetric))
    dv = Float64(_si_value(:dv, kwargs))
    u0 = _grid_centre(kwargs)
    haskey(kwargs, :default_weights) || throw(ArgumentError(
        "GaussianGrid: the explicit form (dv_kms or dv_ms) needs default_weights"))
    raw = kwargs[:default_weights]
    Nv = _grid_nodes(length(raw))
    w = Float64(width)
    _grid_positive("dv", dv)
    _grid_positive("width", w)
    w0 = ntuple(n -> Float64(raw[n]), Nv)
    for n in 1:Nv
        (isfinite(w0[n]) && w0[n] > 0) || throw(ArgumentError(
            "GaussianGrid: default_weights[$n] must be finite and positive; got $(w0[n])"))
    end
    if symmetric
        for n in 1:Nv
            w0[n] == w0[Nv + 1 - n] || throw(ArgumentError(
                "GaussianGrid: symmetric = true needs default_weights symmetric about the centre " *
                "(default_weights[n] == default_weights[Nv + 1 - n]); they differ at n = $n"))
        end
    end
    return _gaussian_grid(u0, dv, w * dv, w0, symmetric)
end

# The function barrier from the run-time node count to the type (Phase G.1 spec D4): g = 0.
function _gaussian_grid(u0::Float64, dv::Float64, sigma::Float64, w0::NTuple{Nv, Float64}, symmetric::Bool) where {Nv}
    NP = symmetric ? cld(Nv, 2) : Nv
    return GaussianGrid{Float64, Nv, NP}(u0, dv, sigma, w0, ntuple(_ -> 0.0, NP))
end

# --- the contract (OQ-68 item 4; Phase G.1 spec D5) ---

"""
    (f::GaussianGrid)(v::Number) -> Number

`f(v) = (1/m₀) Σₙ wₙ e^{−((v − vₙ)/σ)²} / (σ√π)`, in s/m, at the lab velocity `v` (m/s): real, complex or
dual. A loop over the cached weights; allocation-free.

[OQ-68](../../dev/open_questions.md#oq-68) item 4; `docs/derivations/gaussian_grid.md` Section 1.
"""
function (f::GaussianGrid{FT, Nv})(v::Number) where {FT, Nv}
    s = f.w[1] * exp(-((v - _grid_node(f, 1)) / f.sigma)^2)
    for n in 2:Nv
        s += f.w[n] * exp(-((v - _grid_node(f, n)) / f.sigma)^2)
    end
    return s / (f.m0 * f.sigma * sqrt(pi))
end

"""
    dfdv(f::GaussianGrid, v::Number) -> Number

`f′(v) = (1/m₀) Σₙ wₙ (−2 (v − vₙ)/σ²) G(v; vₙ, σ) = −(2/(m₀ σ² √π)) Σₙ wₙ zₙ e^{−zₙ²}`, `zₙ = (v − vₙ)/σ`,
in s²/m², at real, complex or dual `v`. Closed form; allocation-free.

[OQ-68](../../dev/open_questions.md#oq-68) item 4; `docs/derivations/gaussian_grid.md` Section 1.
"""
function dfdv(f::GaussianGrid{FT, Nv}, v::Number) where {FT, Nv}
    z = (v - _grid_node(f, 1)) / f.sigma
    s = f.w[1] * z * exp(-z^2)
    for n in 2:Nv
        z = (v - _grid_node(f, n)) / f.sigma
        s += f.w[n] * z * exp(-z^2)
    end
    return -2 * s / (f.m0 * f.sigma^2 * sqrt(pi))
end

"""
    cauchy_dfdv(f::GaussianGrid, v::Number) -> Complex

The Cauchy integral in lab velocity,

    𝒞f(v_φ) = ∫ f′(v)/(v − v_φ) dv = (1/(m₀ σ²)) Σₙ wₙ Z′((v_φ − vₙ)/σ)      [s²/m²],

on the Landau contour (`conventions.md` 1.12), from `∫ G′(v; c, s)/(v − v_φ) dv = Z′((v_φ − c)/s)/s²`. Its
argument is a velocity (m/s), real, complex in either half-plane (the comb is entire, so there is no
branch) or dual; `Zprime`'s dual methods carry every derivative, and a partial in one `gₙ` is the single
term `(wₙ/m₀)(Z′((v_φ − vₙ)/σ)/σ² − 𝒞f(v_φ))`. On the real axis `Im 𝒞f = π f′`. No table, no quadrature;
allocation-free.

[OQ-68](../../dev/open_questions.md#oq-68) item 4; `docs/derivations/gaussian_grid.md` Section 2.
"""
function cauchy_dfdv(f::GaussianGrid{FT, Nv}, v::Number) where {FT, Nv}
    s = f.w[1] * Zprime((v - _grid_node(f, 1)) / f.sigma)
    for n in 2:Nv
        s += f.w[n] * Zprime((v - _grid_node(f, n)) / f.sigma)
    end
    return s / (f.m0 * f.sigma^2)
end

"""
    moment(f::GaussianGrid, n::Integer) -> Real

`∫ vⁿ f(v) dv` in (m/s)ⁿ, closed form for every `n ≥ 0`: each Gaussian's moment about the origin,

    ∫ vⁿ G(v; vₖ, σ) dv = Σ_{j even ≤ n} C(n, j) vₖ^{n−j} σʲ (j − 1)!!/2^{j/2},

weighted by `wₖ/m₀`. `moment(f, 0)` is 1 to round-off. Throws for `n < 0`.

[OQ-68](../../dev/open_questions.md#oq-68) item 4; `docs/derivations/gaussian_grid.md` Section 3.
"""
function moment(f::GaussianGrid{FT, Nv}, n::Integer) where {FT, Nv}
    n >= 0 || throw(ArgumentError("moment is defined for n >= 0; got n = $n"))
    acc = zero(FT)
    for k in 1:Nv
        a = _grid_node(f, k)
        t = 0.0
        binom = 1.0
        for j in 0:n
            iseven(j) && (t += binom * a^(n - j) * f.sigma^j * _hermite_gaussian_moment(j))
            binom *= (n - j) / (j + 1)
        end
        acc += f.w[k] * t
    end
    return acc / f.m0
end

"""
    mean_velocity(f::GaussianGrid) -> Real

`ū = (Σₙ wₙ vₙ)/m₀`, in m/s. Allocation-free.
"""
function mean_velocity(f::GaussianGrid{FT, Nv}) where {FT, Nv}
    s = f.w[1] * _grid_node(f, 1)
    for n in 2:Nv
        s += f.w[n] * _grid_node(f, n)
    end
    return s / f.m0
end

"""
    velocity_variance(f::GaussianGrid) -> Real

`V = (Σₙ wₙ [(vₙ − ū)² + σ²/2])/m₀`, in m²/s²: the spread of the nodes about the mean plus each Gaussian's
own variance `σ²/2`. Two passes (the mean first); allocation-free.
"""
function velocity_variance(f::GaussianGrid{FT, Nv}) where {FT, Nv}
    ubar = mean_velocity(f)
    half_sigma_sq = f.sigma^2 / 2
    s = f.w[1] * ((_grid_node(f, 1) - ubar)^2 + half_sigma_sq)
    for n in 2:Nv
        s += f.w[n] * ((_grid_node(f, n) - ubar)^2 + half_sigma_sq)
    end
    return s / f.m0
end

"""
    parameters(f::GaussianGrid) -> NamedTuple

`(g1 = …, g2 = …, …)`, the log-weights relative to the default in node order: `Nv` of them, or
`cld(Nv, 2)` with the symmetric option. The grid and the default are not parameters.
"""
parameters(f::GaussianGrid{FT, Nv, NP}) where {FT, Nv, NP} = NamedTuple{_grid_names(Val(NP))}(f.g)

"""
    reconstruct(f::GaussianGrid, values::NamedTuple) -> GaussianGrid

A `GaussianGrid` on `f`'s grid with every parameter replaced by `values`, which must carry exactly
`parameters(f)`'s names in order (a loud error otherwise). The element type is the promoted float type
of the values, so dual values give a dual distribution; the weights and `m₀` are recomputed.
Allocation-free.
"""
function reconstruct(f::GaussianGrid{FT0, Nv, NP}, values::NamedTuple) where {FT0, Nv, NP}
    keys(values) === _grid_names(Val(NP)) || throw(ArgumentError(
        "GaussianGrid: reconstruct needs the parameters g1 to g$NP in order; got $(keys(values))"))
    FT = _grid_values_eltype(typeof(values))
    return GaussianGrid{FT, Nv, NP}(f.u0, f.dv, f.sigma, f.w0, ntuple(i -> FT(values[i]), Val(NP)))
end

# A species' element type over its distribution's parameters (species.jl's _shape_eltype; Phase G.1 spec D8).
_shape_eltype(::GaussianGrid{FT}) where {FT} = FT

"""
    Base.show(io::IO, f::GaussianGrid)

One line: the node count, the centre and spacing in km/s, the width in spacings, and `symmetric` when
the option is on: `GaussianGrid(41 nodes, u0 = 0.0 km/s, dv = 3923.2 km/s, width = 1.2)`. The
parameters are not printed.
"""
function Base.show(io::IO, f::GaussianGrid{FT, Nv, NP}) where {FT, Nv, NP}
    print(io, "GaussianGrid(", Nv, " nodes, u0 = ", round(f.u0 / 1e3; digits = 1), " km/s, dv = ",
          round(f.dv / 1e3; digits = 1), " km/s, width = ", round(f.sigma / f.dv; digits = 3),
          NP < Nv ? ", symmetric)" : ")")
end
