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
