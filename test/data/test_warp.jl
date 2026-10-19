using Test
using TRUST
using REPL: REPL  # gives Base.Docs.doc its methods (empty otherwise); see DW5 below
using JLD2: jldopen

# DW1-DW7 against dev/phase3_instructions.md Section 11 (spec D36), Step 3.8's WarpMap/dewarp/
# read_warpmap. No real OMEGA data is used here (DW1-DW7 need none, D36); all synthetic.

_axes(nl, nt) = (collect(range(500.0e-9, 510.0e-9; length=nl)), collect(range(1.0e-9, 1.5e-9; length=nt)))

# ------------------------------------------------------------------------------------------
# DW1: an identity warp is the identity (bitwise)
# ------------------------------------------------------------------------------------------

@testset "DW1: an identity warp is the identity" begin
    nl, nt = 12, 7
    lam, t = _axes(nl, nt)
    image = rand(nl, nt)
    sigma = rand(nl, nt) .+ 0.1
    sd = StreakData(lambda_m=lam, t_s=t, image=image, sigma=sigma)

    source_row = Float64[i for i in 1:nl, j in 1:nt]
    source_col = Float64[j for i in 1:nl, j in 1:nt]
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam, t_s=t)

    out = dewarp(sd, wm)
    @test all(out.image .=== sd.image)
    @test all(out.sigma .=== sd.sigma)
    @test all(out.mask)
end

# ------------------------------------------------------------------------------------------
# DW2: a known synthetic warp inverts (deliberately loose, 1e-3 relative: set by bilinear
# interpolation, not by the code -- see the bound computed and stated below)
# ------------------------------------------------------------------------------------------

@testset "DW2: a known synthetic warp inverts" begin
    nl, nt = 40, 30
    lam, t = _axes(nl, nt)

    # A smooth analytic image, defined once and for all directly on the integer pixel grid, so
    # dewarp's expected value at any fractional (row, col) is exactly f evaluated there --
    # bilinear interpolation's only error against that continuum value (DW2's own tolerance).
    f(p, q) = 10.0 + sin(2pi * p / nl) + 0.5 * cos(2pi * q / nt)
    image = Float64[f(p, q) for p in 1:nl, q in 1:nt]
    sd = StreakData(lambda_m=lam, t_s=t, image=image)

    # A smooth, small-amplitude, invertible warp map -- amplitude kept well under one pixel so
    # every destination pixel's four source corners stay in bounds.
    warp_row(i, j) = i + 0.4 * sin(2pi * j / nt)
    warp_col(i, j) = j + 0.3 * cos(2pi * i / nl)
    source_row = Float64[warp_row(i, j) for i in 1:nl, j in 1:nt]
    source_col = Float64[warp_col(i, j) for i in 1:nl, j in 1:nt]
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam, t_s=t)

    out = dewarp(sd, wm)
    expected = Float64[f(warp_row(i, j), warp_col(i, j)) for i in 1:nl, j in 1:nt]

    # The warp's own amplitude (0.4, 0.3 pixels) can push a pixel right at the boundary (i = 1 or
    # nl, j = 1 or nt) outside the image; that is DW4's case, not this one, so this row's own
    # assertion is over the interior, where every destination pixel's source point stays in
    # bounds by construction (amplitude < 1 pixel from the boundary).
    interior_i, interior_j = 2:(nl-1), 2:(nt-1)
    @test all(out.mask[interior_i, interior_j])
    # Bilinear interpolation error is O(h^2 |f''|) with h = 1 pixel here (f is defined directly
    # on the pixel-index grid): |f_pp| <= (2pi/nl)^2 = 0.0247, |f_qq| <= 0.5*(2pi/nt)^2 = 0.0219,
    # so the error is bounded by roughly (1/8)*(0.0247 + 0.0219) =~ 0.0058 absolute on a function
    # of magnitude ~10 -- about 6e-4 relative, under the 1e-3 asserted here with margin.
    @test all(isapprox.(out.image[interior_i, interior_j], expected[interior_i, interior_j];
        rtol=1e-3, atol=1e-6))
end

# ------------------------------------------------------------------------------------------
# DW3: sigma propagates in quadrature (tight, 1e-14 relative: the closed form, D36)
# ------------------------------------------------------------------------------------------

@testset "DW3: sigma propagates in quadrature" begin
    nl, nt = 6, 5
    lam, t = _axes(nl, nt)
    image = rand(nl, nt)
    sigma = fill(4.0, nl, nt)
    sd = StreakData(lambda_m=lam, t_s=t, image=image, sigma=sigma)

    # A half-pixel shift in both axes: every interior destination pixel's four bilinear weights
    # are all 0.25 (D36's closed form), so sigma_out = sigma * sqrt(4 * 0.25^2) = sigma / 2.
    source_row = Float64[i + 0.5 for i in 1:nl, j in 1:nt]
    source_col = Float64[j + 0.5 for i in 1:nl, j in 1:nt]
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam, t_s=t)
    out = dewarp(sd, wm)

    for i in 1:(nl-1), j in 1:(nt-1)
        @test out.mask[i, j]
        @test isapprox(out.sigma[i, j], 4.0 / 2; rtol=1e-14)
    end
    # the last row/column's +0.5 shift needs a source pixel one beyond the image: out of bounds
    @test all(.!out.mask[nl, :])
    @test all(.!out.mask[:, nt])
end

# ------------------------------------------------------------------------------------------
# DW4: the mask is conservative (tight; never clamped, D36)
# ------------------------------------------------------------------------------------------

@testset "DW4: the mask is conservative" begin
    nl, nt = 5, 5
    lam, t = _axes(nl, nt)
    image = rand(nl, nt)
    mask = trues(nl, nt)
    mask[2, 2] = false
    sd = StreakData(lambda_m=lam, t_s=t, image=image, mask=mask)

    source_row = fill(1.0, nl, nt)
    source_col = fill(1.0, nl, nt)
    # destination (1,1): a half-pixel sample straddling source (1,1),(1,2),(2,1),(2,2) -- (2,2)
    # is mask-false, so this destination pixel must be excluded even though it is fully in bounds.
    source_row[1, 1] = 1.5
    source_col[1, 1] = 1.5
    # destination (2,2): a source point outside the image (never clamped back in).
    source_row[2, 2] = nl + 5.0
    source_col[2, 2] = 1.0
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam, t_s=t)
    out = dewarp(sd, wm)

    @test out.mask[1, 1] == false
    @test out.mask[2, 2] == false
end

# ------------------------------------------------------------------------------------------
# DW5: the metadata says what it is (tight: substring match, in the data AND the docstring)
# ------------------------------------------------------------------------------------------

@testset "DW5: the metadata says what it is" begin
    nl, nt = 4, 3
    lam, t = _axes(nl, nt)
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt))
    source_row = Float64[i for i in 1:nl, j in 1:nt]
    source_col = Float64[j for i in 1:nl, j in 1:nt]
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam, t_s=t)
    out = dewarp(sd, wm)

    @test out.meta["dewarped"] == 1.0
    @test occursin("diagonal-approximate", out.meta["sigma_note"])
    @test occursin("correlates the noise over about one pixel", out.meta["sigma_note"])

    doc_str = string(Base.Docs.doc(dewarp))
    @test occursin("diagonal-approximate", doc_str)
    @test occursin("correlates the noise", doc_str)
end

# ------------------------------------------------------------------------------------------
# DW6: read_warpmap round-trips a file this code wrote in the same dataset layout (D36: this
# does NOT prove JLD2 opens the collaborator's own HDF5 file -- see the chunk 14 report).
#
# source_row/source_col carry no unit conversion and round-trip bitwise, exactly as IO1 (data/
# test_io.jl) treats image/sigma/mask; lambda/t go through the nm/ns boundary the same way
# StreakData's own axes do, which IO1 already found is not always bitwise-exact (one value in a
# random 64-point grid differed by a single ULP) -- so those two are compared at 1e-15 relative,
# the same tolerance and the same reason as IO1.
# ------------------------------------------------------------------------------------------

@testset "DW6: read_warpmap round-trips a file this code wrote" begin
    nl, nt = 6, 4
    source_row = rand(nl, nt) .* (nl - 1) .+ 1
    source_col = rand(nl, nt) .* (nt - 1) .+ 1
    lam, t = _axes(nl, nt)
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam, t_s=t)

    mktempdir() do dir
        path = joinpath(dir, "test_dewarp_data.h5")
        deformation_map = Array{Float64}(undef, nl, nt, 2)
        deformation_map[:, :, 1] = wm.source_row
        deformation_map[:, :, 2] = wm.source_col
        jldopen(path, "w") do f
            f["deformation_map"] = deformation_map
            f["time_indices"] = t_from_si.(wm.t, :t_ns)
            f["lambda_nm"] = lambda_from_si.(wm.lambda, :lambda_nm)
        end

        wm2 = read_warpmap(path)
        @test wm2.source_row == wm.source_row
        @test wm2.source_col == wm.source_col
        @test isapprox(wm2.lambda, wm.lambda; rtol=1e-15)
        @test isapprox(wm2.t, wm.t; rtol=1e-15)
    end
end

# ------------------------------------------------------------------------------------------
# DW7: size mismatch (D39's fragment)
# ------------------------------------------------------------------------------------------

@testset "DW7: size mismatch" begin
    nl, nt = 5, 4
    lam, t = _axes(nl, nt)
    sd = StreakData(lambda_m=lam, t_s=t, image=rand(nl, nt))

    lam_wrong, _ = _axes(nl + 1, nt)
    source_row = Float64[i for i in 1:(nl+1), j in 1:nt]
    source_col = Float64[j for i in 1:(nl+1), j in 1:nt]
    wm = WarpMap(source_row=source_row, source_col=source_col, lambda_m=lam_wrong, t_s=t)

    @test_throws "the warp map is" dewarp(sd, wm)
end
