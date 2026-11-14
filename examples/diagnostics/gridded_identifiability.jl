# Phase G.2 Step G.2.3: what an experiment determines of a gridded distribution (spec D15 to D17; OQ-69 item 8; the
# plan's measurements M4 and M5, through the package). A DIAGNOSTIC, not a fit: nothing here changes the package,
# and nothing is asserted of its numbers.
#
# Given a parametric problem (its experiment or experiments, each with its σ) and a grid for each species to be
# gridded, it projects those species onto the grids (project_to_grid) and, at chosen bins, forms by ForwardDiff
# through `measure` the Jacobian of each channel's measured streak at that bin with respect to the gridded species'
# log-weights and to the nuisances a fit would free: every other row of the gridded trace (ln n_e, the fractions,
# the other species' leaves) and each channel's amplitude and background; a gridded species has no temperature or
# flow among them, and the g of each gridded species' centre node (the node nearest u0) is held out of the block
# (the plan's GQ-4). Each row is divided by its σ, the nuisance columns are projected out, and the singular values
# of the log-weight block are counted above 1, 3 and 10: a combination with singular value s is determined to 1/s
# in ln(weight). Two readings: each bin alone; and the bin with `neighbours` bins on each side sharing one set of
# log-weights, each bin with its own nuisances (the limit of strong time regularization). The sensitivity per node
# is the column norm of the same projected block; a node whose sensitivity is below `unseen` (1 by default: a unit
# change in its g moves the data by less than one σ in all) is marked unseen.
#
# Run, from the repository root, for the four synthetic problems:
#   julia --project=examples/figures examples/diagnostics/gridded_identifiability.jl [cut|full]
# (`--project=.` also runs it, without the figure, which needs CairoMakie). It writes, for each problem, into
# runs/gridded_identifiability/<cut|full>/<problem>/: counts.tsv (the counts per bin, both readings),
# sensitivity_<species>.tsv (per node and chosen bin, a `*` after a value below `unseen`), and
# identifiability.png (the sensitivity map and the counts). A problem built from any parametric deck can be handed
# to `identifiability` through `channels_from_problem(problem, theta)`.

using TRUST
using LinearAlgebra: svd, svdvals, norm
const ForwardDiff = TRUST.ForwardDiff   # the package's own: a script environment need not list it (spec D17)

# problems.jl, unless the includer has it already (test/instrument/test_examples.jl does, and Pkg.test() warns of every
# method defined twice).
isdefined(@__MODULE__, :twin_cameras) || include(joinpath(@__DIR__, "..", "problems.jl"))

const IDENT_COUNT_LEVELS = (1, 3, 10)
const IDENT_CHUNK = 8    # one ForwardDiff chunk for every reading and problem, so the dual path compiles once

"""
    channels_from_problem(problem, theta) -> Vector{NamedTuple}

The channels `identifiability` takes, from a fit problem built from a parametric deck at the parameter vector
`theta` (a fit's result, say): each channel's experiment at `theta` (`TRUST.unflatten`), its frozen σ and its mask.
"""
channels_from_problem(p::TRUST.FitProblem, theta) =
    [(; experiment=TRUST.unflatten(p.experiment, p.pspec, theta), sigma=p.sigma, mask=p.mask)]
channels_from_problem(jp::TRUST.JointFitProblem, theta) =
    [(; experiment=TRUST.unflatten(ch.experiment, jp.pspec, theta; channel=c), sigma=ch.sigma, mask=ch.mask)
     for (c, ch) in enumerate(jp.channels)]

# The bins each side whose temporal IRF reaches the chosen bin (spec D16): six of its widths, or none.
function _ident_window(expt)
    t = expt.trace.axis
    length(t) < 2 && return 0
    dt = (t[end] - t[1]) / (length(t) - 1)
    w = 0
    for e in expt.effects
        e isa TemporalIRF && e.kernel isa TRUST.GaussianKernel && (w = max(w, ceil(Int, 6 * e.kernel.sigma / dt)))
    end
    return w
end

# The amplitude and background parameters of one channel's effects (spec D16): Sensitivity's scale, every
# background's parameters; with the effects rebuilt from a vector of values.
_ident_is_nuisance(e) = e isa Union{Sensitivity, ConstantBackground, PolynomialBackground, RationalBackground, ImageBackground}
function _ident_effect_values(effects)
    vals = Float64[]
    for e in effects
        _ident_is_nuisance(e) || continue
        for v in values(parameters(e))
            append!(vals, v isa AbstractArray ? vec(Float64.(v)) : [Float64(v)])
        end
    end
    return vals
end
function _ident_effects(effects, x)
    k = 0
    return map(effects) do e
        _ident_is_nuisance(e) || return e
        p = parameters(e)
        newvals = map(values(p)) do v
            if v isa AbstractArray
                n = length(v)
                out = reshape(x[k+1:k+n], size(v))
                k += n
                out
            else
                k += 1
                x[k]
            end
        end
        reconstruct(e, NamedTuple{keys(p)}(newvals))
    end
end

"""
    identifiability(channels, sources, grids; names = nothing, bins = nothing, neighbours = 2, unseen = 1.0)
        -> NamedTuple

`channels` a vector of `(; experiment, sigma, mask)` (the mask optional, `true` = use the pixel), each the
parametric experiment and its σ; `sources` and `grids` as `project_to_grid` takes them, one grid per source;
`bins` the chosen bins of the first channel (default eight spread evenly over its axis; another channel uses its
bin nearest in time); `neighbours` the bins on each side that share the log-weights in the second reading.
Returns `(; species, nodes, times, counts, shared, sensitivity, shared_sensitivity, unseen, block)`: per chosen
bin the counts above 1, 3 and 10 of each reading, and per gridded species a matrix of the sensitivity per node
(rows) and chosen bin (columns), its centre node `NaN`. Spec D15 to D17.
"""
function identifiability(channels, sources, grids; names=nothing, bins=nothing, neighbours::Int=2, unseen::Real=1.0)
    twins = [project_to_grid(ch.experiment.trace, (s => g for (s, g) in zip(sources, grids))...; names=names).trace
             for ch in channels]
    tw1 = twins[1]
    # the gridded species' rows: <set>.<name>.f.g<k>, the last rows of every twin, source by source (spec D5)
    srcset = [first(split(s isa AbstractString ? s : first(s), '.')) for s in sources]
    newnames = names === nothing ?
        [split(s isa AbstractString ? s : first(s), '.')[2] for s in sources] : collect(String.(names))
    grow = Dict{Int,Tuple{Int,Int}}()          # row of the trace => (source, parameter)
    block = Tuple{Int,Int}[]                    # (source, parameter) in the block's column order
    for (i, path) in enumerate(tw1.paths)
        segs = path.segments
        s = findfirst(k -> segs[1] == srcset[k] && segs[2] == newnames[k], eachindex(sources))
        (s !== nothing && length(segs) == 4 && segs[3] == "f") || continue
        k = parse(Int, segs[4][2:end])
        grow[i] = (s, k)
        k == cld(length(grids[s].w0), 2) || push!(block, (s, k))   # the centre node's g is held (GQ-4)
    end
    nuis_rows = [i for i in eachindex(tw1.paths) if !haskey(grow, i)]
    nt1 = nbins(tw1)
    chosen = bins === nothing ? unique(round.(Int, range(1, nt1; length=min(8, nt1)))) : collect(bins)
    times = tw1.axis[chosen]
    function reading(j1, n)
        cols = [begin
                    tw = twins[c]
                    jc = argmin(abs.(tw.axis .- tw1.axis[j1]))
                    max(1, jc - n):min(nbins(tw), jc + n)
                end for c in eachindex(channels)]
        J = _ident_jacobian(channels, twins, cols, grow, block, nuis_rows)
        nb = length(block)
        B = J[:, 1:nb]
        Nn = J[:, nb+1:end]
        P = _ident_project(B, Nn)
        sv = svdvals(P)
        return [count(>(lv), sv) for lv in IDENT_COUNT_LEVELS], [norm(P[:, k]) for k in 1:nb], sv
    end
    counts = Matrix{Int}(undef, length(chosen), 3)
    shared = Matrix{Int}(undef, length(chosen), 3)
    sens = [fill(NaN, length(g.g), length(chosen)) for g in grids]
    ssens = [fill(NaN, length(g.g), length(chosen)) for g in grids]
    for (q, j) in enumerate(chosen)
        c1, norms1, _ = reading(j, 0)
        c2, norms2, _ = reading(j, neighbours)
        counts[q, :] .= c1
        shared[q, :] .= c2
        for (b, (s, k)) in enumerate(block)
            sens[s][k, q] = norms1[b]
            ssens[s][k, q] = norms2[b]
        end
    end
    nodes = [[g.u0 + (n - (length(g.w0) + 1) / 2) * g.dv for n in 1:length(g.w0)] for g in grids]
    return (; species=["$(srcset[s]).$(newnames[s])" for s in eachindex(sources)], nodes, bins=chosen, times,
            counts, shared, sensitivity=sens, shared_sensitivity=ssens, unseen=Float64(unseen), block=length(block))
end

# The whitened Jacobian of every channel's streak at the bins `cols[c]` (spec D16): the block's log-weights shared
# by every bin and channel, every other row of each bin its own nuisance (shared across channels at one bin
# index of each), each channel's amplitude and background its own. Through `measure` on a window of each trace.
function _ident_jacobian(channels, twins, cols, grow, block, nuis_rows)
    nb = length(block)
    nwin = length(cols[1])
    nn = length(nuis_rows) * nwin
    eff0 = [_ident_effect_values(ch.experiment.effects) for ch in channels]
    nx = nb + nn + sum(length, eff0)
    blockrow = Dict{Tuple{Int,Int},Int}()
    for (i, sk) in grow
        blockrow[sk] = i
    end
    function F(x)
        FT = eltype(x)
        out = FT[]
        e0 = nb + nn
        for (c, ch) in enumerate(channels)
            tw = twins[c]
            W = _ident_window(ch.experiment)
            lo, hi = max(1, first(cols[c]) - W), min(nbins(tw), last(cols[c]) + W)
            win = lo:hi
            vals = Matrix{FT}(tw.values[:, win])
            for (b, sk) in enumerate(block)
                for j in cols[c]
                    vals[blockrow[sk], j - lo + 1] += x[b]
                end
            end
            for (q, j) in enumerate(cols[c]), (ri, i) in enumerate(nuis_rows)
                q <= nwin || continue
                vals[i, j - lo + 1] += x[nb + (q - 1) * length(nuis_rows) + ri]   # additive, in SI: the columns are normalized
            end
            refs = [TRUST._resolve_leaf(tw.template, p) for p in tw.paths]
            trace = TRUST._finish_trace(TRUST._widen_plasma(tw.template, FT), tw.axis_name, Vector{FT}(tw.axis[win]),
                                        tw.paths, refs, vals)
            ne = length(eff0[c])
            effects = _ident_effects(ch.experiment.effects, FT.(eff0[c]) .+ x[e0+1:e0+ne])
            e0 += ne
            ex = Experiment(trace=trace, probe=ch.experiment.probe, collection=ch.experiment.collection,
                            lambda_m=ch.experiment.lambda, effects=effects, prefactor=ch.experiment.prefactor,
                            gain=ch.experiment.gain)
            M = measure(ex)
            for j in cols[c]
                col = M[:, j - lo + 1]
                s = ch.sigma[:, j]
                use = hasproperty(ch, :mask) && ch.mask !== nothing ? ch.mask[:, j] : trues(length(col))
                append!(out, col[use] ./ s[use])
            end
        end
        return out
    end
    x0 = zeros(nx)
    return ForwardDiff.jacobian(F, x0, ForwardDiff.JacobianConfig(F, x0, ForwardDiff.Chunk{IDENT_CHUNK}()))
end

# B with the span of N's columns removed (spec D16): N's columns normalized, an orthonormal basis of its range from
# an SVD kept above 1e-10 of its largest singular value (a column that is a combination of the others adds
# nothing, and QR without pivoting would project out a direction that is not in the range).
function _ident_project(B, N)
    size(N, 2) == 0 && return B
    Nn = similar(N)
    for k in axes(N, 2)
        c = norm(N[:, k])
        Nn[:, k] = c > 0 ? N[:, k] ./ c : N[:, k]
    end
    F = svd(Nn)
    r = count(>(1e-10 * F.S[1]), F.S)
    Q = F.U[:, 1:r]
    return B - Q * (Q' * B)
end

"""
    run_problem(problem; outdir, bins = nothing, neighbours = 2, unseen = 1.0) -> NamedTuple

`identifiability` on one of the four synthetic problems and its twin's grids (`twin_cameras`, `twin_sources`),
each camera a channel with its noise model's σ on the parametric streak; writes counts.tsv and
sensitivity_<species>.tsv into `outdir`, and identifiability.png when CairoMakie is loaded.
"""
function run_problem(problem; outdir, bins=nothing, neighbours=2, unseen=1.0)
    cams = twin_cameras(problem)
    channels = [(; experiment=c.parametric, sigma=sigma(c.noise, measure(c.parametric)), mask=nothing) for c in cams]
    sources, names = twin_sources(problem.name)
    r = identifiability(channels, sources, cams[1].grids; names=names, bins=bins, neighbours=neighbours, unseen=unseen)
    mkpath(outdir)
    open(joinpath(outdir, "counts.tsv"), "w") do io
        println(io, "# $(problem.name): log-weights in the block $(r.block); nuisances every other trace row and each channel's amplitude and background")
        println(io, "bin\tt_ns\tbin_s>1\tbin_s>3\tbin_s>10\tshared_s>1\tshared_s>3\tshared_s>10")
        for (q, j) in enumerate(r.bins)
            println(io, join([string(j), string(round(r.times[q] * 1e9; digits=4)), string.(r.counts[q, :])...,
                              string.(r.shared[q, :])...], "\t"))
        end
    end
    for (s, sp) in enumerate(r.species)
        open(joinpath(outdir, "sensitivity_$(replace(sp, "." => "_")).tsv"), "w") do io
            println(io, "# $(sp): column norm of the projected whitened block per node (rows) and chosen bin (columns); * below $(r.unseen) (unseen); NaN the held centre node")
            println(io, "v_kms\t" * join(("t=$(round(t * 1e9; digits=4))" for t in r.times), "\t"))
            for n in eachindex(r.nodes[s])
                k = n   # the parameter of node n (a symmetric grid would fold; the twins' grids are full)
                x = r.sensitivity[s][k, :]
                cells = [isnan(v) ? "NaN" : string(round(v; sigdigits=3), v < r.unseen ? "*" : "") for v in x]
                println(io, round(r.nodes[s][n] / 1e3; digits=1), "\t", join(cells, "\t"))
            end
        end
    end
    isdefined(Main, :CairoMakie) && Base.invokelatest(_ident_figure, r, problem.name, joinpath(outdir, "identifiability.png"))
    return r
end

# The figure (spec D17), drawn only where CairoMakie is loaded: per gridded species, log10 of the sensitivity over
# velocity and time with the unseen nodes outlined, and the counts of both readings against time.
function _ident_figure(r, name, path)
    CM = Main.CairoMakie
    fig = CM.Figure(size=(420 * (length(r.species) + 1), 380))
    for (s, sp) in enumerate(r.species)
        ax = CM.Axis(fig[1, s]; title="$name: $sp, log10 sensitivity", xlabel="t (ns)", ylabel="v (km/s)")
        z = log10.(max.(r.sensitivity[s], 1e-6))
        hm = CM.heatmap!(ax, r.times .* 1e9, r.nodes[s] ./ 1e3, permutedims(z))
        CM.contour!(ax, r.times .* 1e9, r.nodes[s] ./ 1e3, permutedims(z); levels=[log10(r.unseen)], color=:white)
        CM.Colorbar(fig[2, s], hm; vertical=false)
    end
    ax = CM.Axis(fig[1, length(r.species) + 1]; title="singular values above 1, 3, 10", xlabel="t (ns)")
    for (k, lv) in enumerate(IDENT_COUNT_LEVELS)
        CM.scatterlines!(ax, r.times .* 1e9, r.counts[:, k]; label="bin alone, > $lv")
        CM.scatterlines!(ax, r.times .* 1e9, r.shared[:, k]; linestyle=:dash, label="shared, > $lv")
    end
    CM.Legend(fig[2, length(r.species) + 1], ax; orientation=:horizontal, nbanks=3, framevisible=false)
    CM.save(path, fig)
    return path
end

function main(args)
    size_ = isempty(args) ? "cut" : args[1]
    problems = size_ == "full" ?
        [epw_ablation_problem(), iaw_multiflow_problem(), iaw_shock_problem(), bimaxwellian_hydrogen_problem()] :
        [epw_ablation_problem(; nlambda=128, nt=32), iaw_multiflow_problem(; nlambda=128, nt=32),
         iaw_shock_problem(; nlambda=128, nt=32),
         bimaxwellian_hydrogen_problem(; nlambda_epw=128, nlambda_iaw=128, nt_epw=32, nt_iaw=30)]
    root = joinpath(dirname(dirname(@__DIR__)), "runs", "gridded_identifiability", size_)
    for p in problems
        t = @elapsed r = run_problem(p; outdir=joinpath(root, p.name))
        @info "gridded_identifiability: $(p.name) ($size_)" seconds = t block = r.block bins = r.bins counts_bin = r.counts counts_shared = r.shared
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isdefined(Main, :CairoMakie) || try
        @eval Main using CairoMakie
    catch
        @info "gridded_identifiability: CairoMakie is not in this environment; the figure is skipped " *
              "(run with --project=examples/figures)"
    end
    main(ARGS)
end
