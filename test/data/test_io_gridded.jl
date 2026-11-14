# Step G.2.2: the trace file holds a grid (Phase G.2 spec D11, D12; OQ-69 item 4). Gate "gridded file": GL1 to
# GL5. The layout trust-trace-3: a gridded species' group stores its registry name, its template log-weights,
# its grid and no u or T; read_trace(as = :trace) returns the grid, the default and every weight to the bit; a
# trust-trace-2 file, which holds no gridded species, reads exactly as before. The oracle is the trace the file
# was written from (conventions.md 8.2); every row is exact.

using Test
using TRUST
using TRUST: ELECTRON_MASS
import JLD2
using JLD2: jldopen
using Random: Xoshiro, randn

# A trace with two gridded species (electrons on the default grid, deuterons on a symmetric one), a shape
# species beside them (carbon), and g, n_e and fraction rows varying over three bins.
function _gl_trace()
    ge = GaussianGrid(; mass_kg = ELECTRON_MASS, T_ref_eV = 700.0, u0_kms = 50.0)
    gD = GaussianGrid(; mass_amu = 2.014, T_ref_eV = 250.0, nodes = 21, symmetric = true)
    rng = Xoshiro(11)
    ge = reconstruct(ge, NamedTuple{keys(parameters(ge))}(Tuple(0.2 .* randn(rng, 41))))
    tmpl = Plasma(n_e_cm3 = 5e19, electrons = (electrons(f = ge),),
                  ions = (ion(name = "D", mass_amu = 2.014, charge = 1, f = gD, fraction = 0.9),
                          ion(name = "C", mass_amu = 12.011, charge = 6, T_eV = 300.0, u_kms = 20.0, fraction = 0.1)))
    G = 0.3 .* randn(rng, 41, 3)
    GD = 0.2 .* randn(rng, 11, 3)
    return PlasmaTrace(tmpl; t_ns = [0.0, 1.0, 2.0],
        leaves = ["n_e_cm3" => [5e19, 6e19, 7e19], "electrons.e.f" => G, "ions.D.f" => GD,
                  "ions.D.fraction" => [0.9, 0.85, 0.8], "ions.C.fraction" => [0.1, 0.15, 0.2]])
end

@testset "GL1: trust-trace-3 stores a gridded species' grid and no u or T" begin
    tr = _gl_trace()
    mktempdir() do dir
        path = joinpath(dir, "gl1.jld2")
        write_trace(path, tr; source = "GL1")
        r = read_trace(path)
        @test r.layout_version == "trust-trace-3"
        @test r.paths == string.(tr.paths)
        @test all(r.values .=== tr.values)
        e = r.template["electrons"]["species"]["e"]
        @test Set(keys(e)) == Set(["mass_kg", "charge", "fraction", "shape", "shape_p", "grid"])
        @test e["shape"] == "GaussianGrid"
        g = tr.template.electrons[1].f
        @test Set(keys(e["shape_p"])) == Set("g$i" for i in 1:41)
        @test all(e["shape_p"]["g$i"] === g.g[i] for i in 1:41)
        grid = e["grid"]
        @test Set(keys(grid)) == Set(["u0_ms", "dv_ms", "sigma_ms", "nodes", "symmetric", "default_weights", "node_velocities_ms"])
        @test grid["u0_ms"] === g.u0 && grid["dv_ms"] === g.dv && grid["sigma_ms"] === g.sigma
        @test grid["nodes"] === Int64(41) && grid["symmetric"] === UInt8(0)
        @test all(grid["default_weights"] .=== collect(g.w0))
        @test all(grid["node_velocities_ms"] .=== [g.u0 + (n - 21) * g.dv for n in 1:41])
        D = r.template["ions"]["species"]["D"]
        @test D["grid"]["symmetric"] === UInt8(1) && D["grid"]["nodes"] === Int64(21) && length(D["shape_p"]) == 11
        @test !haskey(D, "u_ms") && !haskey(D, "T_J")
        C = r.template["ions"]["species"]["C"]                      # a shape species, as in trust-trace-2
        @test Set(keys(C)) == Set(["mass_kg", "charge", "u_ms", "T_J", "fraction", "shape", "shape_p"])
        @test C["u_ms"] === tr.template.ions[2].u && C["T_J"] === tr.template.ions[2].T
    end
end

@testset "GL2: read_trace(as = :trace) returns the grid, the default and every weight to the bit" begin
    tr = _gl_trace()
    mktempdir() do dir
        path = joinpath(dir, "gl2.jld2")
        write_trace(path, tr; source = "GL2")
        rb = read_trace(path; as = :trace)
        @test rb.template.electrons[1].f === tr.template.electrons[1].f
        @test rb.template.ions[1].f === tr.template.ions[1].f
        @test rb.template.ions[2].f === tr.template.ions[2].f
        @test rb.template.electrons[1].T === tr.template.electrons[1].T
        @test rb.template.electrons[1].u === tr.template.electrons[1].u
        @test rb.paths == tr.paths && all(rb.values .=== tr.values)
        for j in 1:3
            p, q = plasma(rb, j), plasma(tr, j)
            @test p.electrons[1].f === q.electrons[1].f && p.ions[1].f === q.ions[1].f
            @test p.electrons[1].T === q.electrons[1].T && p.ions[1].u === q.ions[1].u
            @test p.ions[2].n === q.ions[2].n
        end
    end
end

@testset "GL3: a trace with no gridded species has the same content in both layouts, and trust-trace-2 reads" begin
    tmpl = Plasma(n_e_cm3 = 5e19, electrons = (electrons(T_eV = 500.0, f = SuperGaussian(3.0)),),
                  ions = (ion(name = "D", mass_amu = 2.014, charge = 1, T_eV = 250.0),))
    tr = PlasmaTrace(tmpl; t_ns = [0.0, 1.0], leaves = ["electrons.e.f.m" => [3.0, 3.5]])
    mktempdir() do dir
        path = joinpath(dir, "gl3.jld2")
        write_trace(path, tr; source = "GL3")
        names = String[]
        jldopen(path, "r") do f
            walk(g, pre) = for k in keys(g)
                obj = g[k]
                full = isempty(pre) ? k : "$pre/$k"
                obj isa JLD2.Group ? walk(obj, full) : push!(names, full)
            end
            walk(f, "")
        end
        @test sort(names) == sort(["layout_version", "axis_name", "axis", "axis_unit", "paths", "values",
            "template/n_e_m3", "template/electrons/names", "template/ions/names",
            "template/electrons/e/mass_kg", "template/electrons/e/charge", "template/electrons/e/u_ms",
            "template/electrons/e/T_J", "template/electrons/e/fraction", "template/electrons/e/shape",
            "template/electrons/e/shape_p/m", "template/ions/D/mass_kg", "template/ions/D/charge",
            "template/ions/D/u_ms", "template/ions/D/T_J", "template/ions/D/fraction", "template/ions/D/shape",
            "meta/source", "meta/created_unix", "meta/trust_version"])
        new = read_trace(path)
        jldopen(path, "a+") do f
            delete!(f, "layout_version")
            f["layout_version"] = "trust-trace-2"
        end
        old = read_trace(path)
        @test old.layout_version == "trust-trace-2"
        @test isequal(merge(old, (layout_version = "trust-trace-3",)), new)
        rb = read_trace(path; as = :trace)
        @test all(plasma(rb, j).electrons[1].f === plasma(tr, j).electrons[1].f for j in 1:2)
    end
end

@testset "GL4: read_trace's refusals" begin
    tr = _gl_trace()
    mktempdir() do dir
        old = joinpath(dir, "old.jld2")
        jldopen(old, "w") do f
            f["layout_version"] = "trust-trace-1"
        end
        @test_throws "has layout_version \"trust-trace-1\"; expected \"trust-trace-2\" or \"trust-trace-3\"" read_trace(old)
        path = joinpath(dir, "gl4.jld2")
        write_trace(path, tr)
        jldopen(path, "a+") do f
            delete!(f, "template/electrons/e/grid/nodes")
            f["template/electrons/e/grid/nodes"] = Int64(40)
        end
        @test_throws "read_trace: a GaussianGrid's grid/nodes is 40 but its grid/default_weights has 41 entries" read_trace(path; as = :trace)
        path2 = joinpath(dir, "gl4b.jld2")
        write_trace(path2, tr)
        jldopen(path2, "a+") do f
            delete!(f, "template/electrons/e/shape_p/g41")
        end
        @test_throws "read_trace: a GaussianGrid of 41 nodes has 41 parameters; the file stores 40" read_trace(path2; as = :trace)
    end
end

@testset "GL5: every value a gridded trace file holds is a plain number, a string, or an array of either" begin
    tr = _gl_trace()
    mktempdir() do dir
        path = joinpath(dir, "gl5.jld2")
        write_trace(path, tr)
        ok = Bool[]
        jldopen(path, "r") do f
            walk(g) = for k in keys(g)
                obj = g[k]
                obj isa JLD2.Group ? walk(obj) :
                    push!(ok, obj isa Union{AbstractArray{<:Real}, AbstractArray{<:AbstractString}, AbstractString, Real})
            end
            walk(f)
        end
        @test !isempty(ok) && all(ok)
    end
end
