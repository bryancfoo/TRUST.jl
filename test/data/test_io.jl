using Test
using TRUST
import JLD2
using JLD2: jldopen

function _walk_leaf_values(f, prefix="")
    vals = Pair{String,Any}[]
    container = isempty(prefix) ? f : f[prefix]
    for k in keys(container)
        full = isempty(prefix) ? k : "$prefix/$k"
        v = f[full]
        if v isa JLD2.Group
            append!(vals, _walk_leaf_values(f, full))
        else
            push!(vals, full => v)
        end
    end
    return vals
end

@testset "IO1: the full streak round trip" begin
    nl, nt = 64, 16
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    image = rand(nl, nt)
    sigma = rand(nl, nt) .+ 0.1
    mask = [isodd(l + j) for l in 1:nl, j in 1:nt]
    meta = Dict{String,Any}("a" => "hello", "b" => 3.5, "c" => [1.0, 2.0], "d" => 7)
    sd = StreakData(lambda_m=lam, t_s=t, image=image, sigma=sigma, mask=mask, meta=meta)

    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd; source="test")
        sd2 = read_streak(path)
        # lambda/t round-trip through the nm/ns boundary (SI -> /1e-9 -> file -> *1e-9 -> SI):
        # dividing and multiplying by the same non-power-of-2 Float64 constant is not always an
        # exact inverse (one value in a random 64-point grid was already found to differ by a
        # single ULP), so these compare to full double precision rather than bitwise. image,
        # sigma and mask carry no unit conversion and round-trip exactly (docs/derivations/data_layout.md).
        @test isapprox(sd2.lambda, sd.lambda; rtol=1e-15)
        @test isapprox(sd2.t, sd.t; rtol=1e-15)
        @test all(sd2.image .=== sd.image)
        @test all(sd2.sigma .=== sd.sigma)
        @test sd2.mask == sd.mask
        @test sd2.meta["a"] == "hello"
        @test sd2.meta["b"] == 3.5
        @test sd2.meta["c"] == [1.0, 2.0]
        @test sd2.meta["d"] == 7
    end
end

@testset "IO2: the round trip without sigma and mask" begin
    nl, nt = 16, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt))
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd)
        sd2 = read_streak(path)
        @test sd2.sigma === nothing
        @test sd2.mask === nothing
    end
end

@testset "IO3: the file holds only plain arrays and strings" begin
    nl, nt = 16, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt), sigma=rand(nl, nt) .+ 0.1,
        mask=trues(nl, nt), meta=Dict{String,Any}("x" => "y", "z" => 1.5))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60)
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd; source="t", probe=probe, collection=collection)
        jldopen(path, "r") do f
            for (name, v) in _walk_leaf_values(f)
                @test v isa Union{AbstractArray{<:Real},AbstractString,Real}
            end
        end
    end
end

@testset "IO4: the mask is stored as UInt8" begin
    nl, nt = 8, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    mask = [isodd(l + j) for l in 1:nl, j in 1:nt]
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt), mask=mask)
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd)
        jldopen(path, "r") do f
            @test eltype(f["mask"]) === UInt8
            @test all((f["mask"] .== 1) .== mask)
        end
    end
end

@testset "IO5: malformed files" begin
    nl, nt = 8, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt))

    mktempdir() do dir
        path1 = joinpath(dir, "no_image.jld2")
        jldopen(path1, "w") do f
            f["layout_version"] = TRUST.STREAK_LAYOUT_VERSION
            f["lambda_nm"] = lambda_from_si.(lam, :lambda_nm)
            f["t_ns"] = t_from_si.(t, :t_ns)
        end
        @test_throws "is missing the required dataset \"image\"" read_streak(path1)

        path2 = joinpath(dir, "bad_version.jld2")
        jldopen(path2, "w") do f
            f["layout_version"] = "trust-streak-0"
            f["lambda_nm"] = lambda_from_si.(lam, :lambda_nm)
            f["t_ns"] = t_from_si.(t, :t_ns)
            f["image"] = rand(nl, nt)
        end
        @test_throws "has layout_version \"trust-streak-0\"; expected \"trust-streak-1\"" read_streak(path2)

        path3 = joinpath(dir, "wrong_size.jld2")
        jldopen(path3, "w") do f
            f["layout_version"] = TRUST.STREAK_LAYOUT_VERSION
            f["lambda_nm"] = lambda_from_si.(lam, :lambda_nm)
            f["t_ns"] = t_from_si.(t, :t_ns)
            f["image"] = rand(nl, nt + 1)
        end
        @test_throws "image is" read_streak(path3)
    end
end

@testset "IO6: the trace round trip" begin
    template = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500, f=SuperGaussian(3.0)),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=250),))
    t_ns = collect(range(2.0, 5.0; length=16))
    n_e_vec = collect(range(5e19, 6e19; length=16))
    Te_vec = collect(range(500.0, 550.0; length=16))
    m_vec = collect(range(3.0, 3.5; length=16))
    trace = PlasmaTrace(template; t_ns=t_ns,
        leaves=("n_e_cm3" => n_e_vec, "electrons.e.T_eV" => Te_vec, "electrons.e.f.m" => m_vec))

    mktempdir() do dir
        path = joinpath(dir, "trace.jld2")
        write_trace(path, trace; source="test")
        r = read_trace(path)
        @test r.layout_version == TRUST.TRACE_LAYOUT_VERSION
        @test all(r.axis .=== trace.axis)
        @test all(r.values .=== trace.values)
        @test r.paths == string.(trace.paths)
        @test r.template["electrons"]["names"] == ["e"]
        @test r.template["electrons"]["species"]["e"]["mass_kg"] == TRUST.ELECTRON_MASS
        @test r.template["electrons"]["species"]["e"]["charge"] == -1
        @test r.template["electrons"]["species"]["e"]["shape"] == "SuperGaussian"
        @test haskey(r.template["electrons"]["species"]["e"]["shape_p"], "m")
        @test r.template["ions"]["names"] == ["i"]
        @test r.template["ions"]["species"]["i"]["charge"] == 1
    end
end

@testset "IO7: the orientation" begin
    nl, nt = 64, 16
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    image = rand(nl, nt)
    sd = StreakData(lambda_m=lam, t_s=t, image=image)
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd)
        jldopen(path, "r") do f
            @test size(f["image"]) == (nl, nt)
            @test f["image"][3, 5] === sd.image[3, 5]
        end
    end
end

@testset "IO8: the meta keys" begin
    nl, nt = 8, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt))
    probe = ProbeBeam(lambda_probe_nm=263.25, polarization_deg=10.0)
    collection = Collection(theta_deg=60, k_model=:exact)
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        before = time()
        write_streak(path, sd; source="s", probe=probe, collection=collection)
        after = time()
        jldopen(path, "r") do f
            @test f["meta/source"] == "s"
            @test f["meta/created_unix"] isa Float64
            @test before - 60 <= f["meta/created_unix"] <= after + 60
            @test f["meta/trust_version"] isa String
            @test isapprox(f["meta/lambda_probe_nm"], 263.25; rtol=1e-12)
            @test isapprox(f["meta/theta_deg"], 60.0; rtol=1e-12)
            @test isapprox(f["meta/polarization_deg"], 10.0; rtol=1e-12)
            @test f["meta/k_model"] == "exact"
        end
    end
end

@testset "IO9: a user meta key survives" begin
    nl, nt = 8, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt),
        meta=Dict{String,Any}("shot" => "s12345", "energy_j" => 42.5))
    mktempdir() do dir
        path = joinpath(dir, "streak.jld2")
        write_streak(path, sd)
        sd2 = read_streak(path)
        @test sd2.meta["shot"] == "s12345"
        @test sd2.meta["energy_j"] == 42.5
    end
end

@testset "IO10: a non-Float64 streak" begin
    nl, nt = 8, 4
    lam32 = Float32.(collect(range(235e-9, 255e-9; length=nl)))
    t32 = Float32.(collect(range(2.0e-9, 3.0e-9; length=nt)))
    sd32 = StreakData(lambda_m=lam32, t_s=t32, image=rand(Float32, nl, nt))
    mktempdir() do dir
        path = joinpath(dir, "streak32.jld2")
        write_streak(path, sd32)
        sd2 = read_streak(path)
        @test isapprox(sd2.lambda, Float64.(sd32.lambda); rtol=1e-7)
        @test isapprox(sd2.image, Float64.(sd32.image); rtol=1e-7)
    end

    import ForwardDiff
    lam_dual = [ForwardDiff.Dual(l, 1.0) for l in collect(range(235e-9, 255e-9; length=nl))]
    t_dual = [ForwardDiff.Dual(tt, 0.0) for tt in collect(range(2.0e-9, 3.0e-9; length=nt))]
    image_dual = [ForwardDiff.Dual(x, 0.0) for x in rand(nl, nt)]
    sd_dual = StreakData(lambda_m=lam_dual, t_s=t_dual, image=image_dual)
    mktempdir() do dir
        path = joinpath(dir, "streak_dual.jld2")
        @test_throws "streak element type must be a plain float" write_streak(path, sd_dual)
    end
end

# ------------------------------------------------------------------------------------------
# IO11, IO12: Phase 4.N's housekeeping (spec D19, D20; OQ-65 items 2 and 5)
# ------------------------------------------------------------------------------------------

@testset "IO11: a Bool in meta writes and reads back as a Bool" begin
    sd = StreakData(lambda_nm=[262.0, 263.0], t_ns=[0.0, 1.0], image=[1.0 2.0; 3.0 4.0],
        meta=Dict{String,Any}("flag" => true, "flags" => [true, false]))
    mktempdir() do dir
        back = read_streak(write_streak(joinpath(dir, "bool.jld2"), sd; source="io11"))
        @test back.meta["flag"] === true
        @test back.meta["flags"] isa Vector{Bool} && back.meta["flags"] == [true, false]
    end
end

@testset "IO12: an empty source or meta string is written as \"-\" and read back as \"\"" begin
    sd = StreakData(lambda_nm=[262.0, 263.0], t_ns=[0.0, 1.0], image=[1.0 2.0; 3.0 4.0],
        meta=Dict{String,Any}("note" => "", "tag" => "x", "names" => ["", "a"]))
    trace = PlasmaTrace(Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0)]); t_ns=[0.0, 1.0])
    mktempdir() do dir
        path = write_streak(joinpath(dir, "empty.jld2"), sd)
        jldopen(path, "r") do f
            @test f["meta/source"] == "-"
            @test f["meta/note"] == "-"
            @test f["meta/tag"] == "x"
            @test f["meta/names"] == ["", "a"]
        end
        back = read_streak(path)
        @test back.meta["source"] == "" && back.meta["note"] == ""
        @test back.meta["tag"] == "x" && back.meta["names"] == ["", "a"]
        # a file written before Phase 4.N, holding "" itself, reads the same
        jldopen(path, "a+") do f
            delete!(f, "meta/note")
            f["meta/note"] = ""
        end
        @test read_streak(path).meta["note"] == ""
        tpath = write_trace(joinpath(dir, "trace.jld2"), trace)
        @test jldopen(f -> f["meta/source"], tpath, "r") == "-"
    end
end

# ------------------------------------------------------------------------------------------
# IO13: the otis- alias of the streak layout (the 2026-10-18 rename, Level 1 decision D3)
# ------------------------------------------------------------------------------------------

@testset "IO13: read_streak accepts the otis- spelling of the current number only" begin
    nl, nt = 8, 4
    lam = collect(range(235e-9, 255e-9; length=nl))
    t = collect(range(2.0e-9, 3.0e-9; length=nt))
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt), sigma=rand(nl, nt) .+ 0.1,
        mask=trues(nl, nt), meta=Dict{String,Any}("a" => "hello"))
    rewrite(path, version) = jldopen(path, "a+") do f
        delete!(f, "layout_version")
        f["layout_version"] = version
    end
    mktempdir() do dir
        path = write_streak(joinpath(dir, "streak.jld2"), sd; source="io13")
        ref = read_streak(path)
        @test jldopen(f -> f["layout_version"], path, "r") == "trust-streak-1"   # writers never emit otis-

        # 1: the current number, old spelling: reads, and every field is the untouched file's (exact)
        rewrite(path, "otis-streak-1")
        @test jldopen(f -> f["layout_version"], path, "r") == "otis-streak-1"
        old = read_streak(path)
        @test all(isequal(getfield(old, k), getfield(ref, k)) for k in fieldnames(StreakData))

        # 2: a wrong number in the old spelling is refused, naming the canonical string
        rewrite(path, "otis-streak-0")
        @test_throws "has layout_version \"trust-streak-0\"; expected \"trust-streak-1\"" read_streak(path)
    end
end
