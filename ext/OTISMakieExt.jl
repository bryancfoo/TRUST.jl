# Makie plotting methods for the eight stubs declared in src/data/streak.jl. Spec D26: plain
# methods on package-declared stubs, not @recipe (a @recipe's plot function lives in this
# extension, not in OTIS, and MakieCore is not a hard dependency, OQ-1).
module OTISMakieExt

using Makie

using OTIS
using OTIS: PlasmaTrace, StreakData, LeafPath, _resolve_leaf, _leaf_row, _leaf_display_unit,
    _leaf_value

function _streak_heatmap!(ax, lambda, t, image; kwargs...)
    return Makie.heatmap!(ax, t, lambda, permutedims(image); kwargs...)
end

function OTIS.streakplot(lambda, t, image; kwargs...)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="t", ylabel="λ")
    hm = _streak_heatmap!(ax, lambda, t, image; kwargs...)
    Makie.Colorbar(fig[1, 2], hm)
    return fig
end

function OTIS.streakplot(sd::StreakData; kwargs...)
    lambda_nm = lambda_from_si.(sd.lambda, :lambda_nm)
    t_ns = t_from_si.(sd.t, :t_ns)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="t (ns)", ylabel="λ (nm)")
    hm = _streak_heatmap!(ax, lambda_nm, t_ns, sd.image; kwargs...)
    Makie.Colorbar(fig[1, 2], hm)
    return fig
end

function OTIS.streakplot!(ax, sd::StreakData; kwargs...)
    lambda_nm = lambda_from_si.(sd.lambda, :lambda_nm)
    t_ns = t_from_si.(sd.t, :t_ns)
    return _streak_heatmap!(ax, lambda_nm, t_ns, sd.image; kwargs...)
end

function OTIS.spectrumplot(lambda, P; kwargs...)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="λ", ylabel="P")
    Makie.lines!(ax, lambda, P; kwargs...)
    return fig
end

function OTIS.spectrumplot!(ax, lambda, P; kwargs...)
    return Makie.lines!(ax, lambda, P; kwargs...)
end

# The row of trace.values named by path, in its canonical boundary unit, against the axis in ns
# (D26); a leaf that resolves but is not tabulated (constant on this trace) is a flat line at the
# template's value, exactly what plasma(trace, j) would show at every bin.
function _trace_series(trace::PlasmaTrace, path::AbstractString)
    lp = LeafPath(path)
    ref = _resolve_leaf(trace.template, lp)
    row = _leaf_row(trace, lp)
    unit_key, factor = _leaf_display_unit(lp)
    t_ns = t_from_si.(trace.axis, :t_ns)
    y = row == 0 ? fill(_leaf_value(trace.template, ref) / factor, nbins(trace)) :
        trace.values[row, :] ./ factor
    return t_ns, y, unit_key
end

function OTIS.traceplot(trace::PlasmaTrace, path::AbstractString; sigma=nothing, kwargs...)
    _, _, unit_key = _trace_series(trace, path)
    fig = Makie.Figure()
    ylabel = isempty(unit_key) ? path : "$path ($unit_key)"
    ax = Makie.Axis(fig[1, 1]; xlabel="t (ns)", ylabel=ylabel)
    OTIS.traceplot!(ax, trace, path; sigma=sigma, kwargs...)
    return fig
end

function OTIS.traceplot!(ax, trace::PlasmaTrace, path::AbstractString; sigma=nothing, kwargs...)
    t_ns, y, _ = _trace_series(trace, path)
    if sigma !== nothing
        Makie.band!(ax, t_ns, y .- sigma, y .+ sigma; color=(:steelblue, 0.25))
    end
    return Makie.lines!(ax, t_ns, y; kwargs...)
end

function OTIS.lcurveplot(chi2::AbstractVector, R::AbstractVector; weights=nothing, kwargs...)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="R", ylabel="χ²")
    OTIS.lcurveplot!(ax, chi2, R; weights=weights, kwargs...)
    return fig
end

function OTIS.lcurveplot!(ax, chi2::AbstractVector, R::AbstractVector; weights=nothing, kwargs...)
    color = weights === nothing ? (:steelblue) : weights
    return Makie.scatterlines!(ax, R, chi2; color=color, kwargs...)
end

end # module OTISMakieExt
