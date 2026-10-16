using Test
using OTIS

_sd_axes(nl, nt) = (collect(range(235e-9, 255e-9; length=nl)), collect(range(2.0e-9, 3.0e-9; length=nt)))

@testset "SD1: construction and unit conversion" begin
    nl, nt = 16, 4
    lam_nm = collect(range(235.0, 255.0; length=nl))
    t_ns = collect(range(2.0, 3.0; length=nt))
    image = rand(nl, nt)
    sd_nm = StreakData(lambda_nm=lam_nm, t_ns=t_ns, image=image)
    sd_m = StreakData(lambda_m=lambda_to_si.(lam_nm, :lambda_nm), t_s=t_to_si.(t_ns, :t_ns), image=image)
    @test sd_nm.lambda == sd_m.lambda
    @test sd_nm.t == sd_m.t
end

@testset "SD2: every StreakData error of D28" begin
    nl, nt = 8, 4
    lam, t = _sd_axes(nl, nt)
    lam_m = collect(lam)
    image = rand(nl, nt)

    bad_lam = copy(lam_m); bad_lam[3] = bad_lam[2]
    @test_throws "must be strictly increasing" StreakData(lambda_m=bad_lam, t_s=t, image=image)

    @test_throws "does not match the axes" StreakData(lambda_m=lam_m, t_s=t, image=rand(nl, nt + 1))
    @test_throws "does not match the axes" StreakData(lambda_m=lam_m, t_s=t, image=image, sigma=rand(nl, nt + 1))
    @test_throws "does not match the axes" StreakData(lambda_m=lam_m, t_s=t, image=image, mask=trues(nl, nt + 1))

    bad_sigma = rand(nl, nt); bad_sigma[1, 1] = -1.0
    @test_throws "sigma must be strictly positive" StreakData(lambda_m=lam_m, t_s=t, image=image, sigma=bad_sigma)

    @test_throws "must be a Real, a String" StreakData(lambda_m=lam_m, t_s=t, image=image,
        meta=Dict{String,Any}("bad" => (1, 2, 3)))
end

@testset "SD3: the mask survives as Bool and true means use" begin
    nl, nt = 8, 4
    lam, t = _sd_axes(nl, nt)
    image = rand(nl, nt)
    mask = [isodd(l + j) for l in 1:nl, j in 1:nt]
    sd = StreakData(lambda_m=lam, t_s=t, image=image, mask=mask)
    @test eltype(sd.mask) === Bool
    @test count(sd.mask) == count(mask)
end

@testset "SD4: size, nbins, show" begin
    nl, nt = 8, 4
    lam, t = _sd_axes(nl, nt)
    image = rand(nl, nt)
    sd = StreakData(lambda_m=lam, t_s=t, image=image)
    @test size(sd) == (nl, nt)
    @test nbins(sd) == nt
    @test occursin("StreakData", sprint(show, sd))
    @test occursin("StreakData", sprint(io -> show(io, MIME("text/plain"), sd)))
end

@testset "SD5: the plotting stubs exist and throw without Makie" begin
    nl, nt = 8, 4
    lam, t = _sd_axes(nl, nt)
    image = rand(nl, nt)
    sd = StreakData(lambda_m=lam, t_s=t, image=image)
    @test_throws MethodError streakplot(sd)
    @test_throws MethodError spectrumplot(lam, image[:, 1])
    @test_throws MethodError lcurveplot([1.0, 2.0], [3.0, 4.0])
end

# ------------------------------------------------------------------------------------------
# SD6 (D26, A19: this is Phase 2's file, the fix belongs here beside the other meta checks): a
# meta key containing "/" is refused at construction, naming the key; a key without one still
# round-trips through write_streak/read_streak, type included.
# ------------------------------------------------------------------------------------------

@testset "SD6: a meta key containing / is refused; a regions_-prefixed key round-trips" begin
    nl, nt = 4, 3
    lam, t = _sd_axes(nl, nt)
    image = rand(nl, nt)

    @test_throws "must not contain" StreakData(lambda_m=lam, t_s=t, image=image,
        meta=Dict{String,Any}("bad/key" => 1.0))
    @test_throws "regions/source_t" StreakData(lambda_m=lam, t_s=t, image=image,
        meta=Dict{String,Any}("regions/source_t" => 1.0))

    sd = StreakData(lambda_m=lam, t_s=t, image=image,
        meta=Dict{String,Any}("regions_source_t" => collect(1.0:9.0),
            "regions_usable_counts" => Matrix{Int}(reshape(1:12, nl, nt))))
    mktempdir() do dir
        path = joinpath(dir, "sd6.jld2")
        write_streak(path, sd; source="test")
        back = read_streak(path)
        @test back.meta["regions_source_t"] == collect(1.0:9.0)
        @test back.meta["regions_usable_counts"] == Matrix{Int}(reshape(1:12, nl, nt))
        @test eltype(back.meta["regions_usable_counts"]) <: Integer
    end
end

@testset "SD7: _masked_image blanks the masked-out pixels; without a mask it is the image itself" begin
    # Phase 4.N spec D18 (OQ-65 item 1): what streakplot draws; tested here without Makie.
    image = [1.0 2.0 3.0; 4.0 5.0 6.0]
    sd0 = StreakData(lambda_nm=[262.0, 263.0], t_ns=[0.0, 1.0, 2.0], image=image)
    @test OTIS._masked_image(sd0) === sd0.image
    mask = [true false true; true true false]
    sd = StreakData(lambda_nm=[262.0, 263.0], t_ns=[0.0, 1.0, 2.0], image=image, mask=mask)
    m = OTIS._masked_image(sd)
    @test m isa Matrix{Float64} && m !== sd.image
    @test isnan.(m) == .!mask
    @test m[mask] == image[mask]
    @test sd.image == [1.0 2.0 3.0; 4.0 5.0 6.0]
end
