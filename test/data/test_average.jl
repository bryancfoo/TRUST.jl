using Test
using OTIS
using OTIS: _operator_rows, _operator_anchor
using Random: Xoshiro
using Statistics: median

# AV1 to AV15 against dev/phase4A_instructions.md Section 7 (spec D13, D14, D16, D26), gated
# "streak averaging". Step 4.A.2, chunk 5: average_regions, the first function in the package
# that averages, rebins or resamples a StreakData along either axis.

_av_axis(nl, nt) = (collect(range(500.0e-9, 500.0e-9 + (nl - 1) * 1.0e-9; length=nl)),
    collect(range(0.0, (nt - 1) * 1.0e-9; length=nt)))

# ------------------------------------------------------------------------------------------
# AV1: AveragingRegion construction and validation
# ------------------------------------------------------------------------------------------

@testset "AV1: AveragingRegion construction and validation" begin
    r = AveragingRegion(; nbins=4, t_range_ns=[0.0, 8.0])
    @test r.lo == 0.0
    @test r.hi == 8.0e-9
    @test r.nbins == 4
    @test r.interval == ""

    r2 = AveragingRegion(; nbins=2, t_range_s=[1.0e-9, 3.0e-9], interval="pre")
    @test r2.lo == 1.0e-9
    @test r2.hi == 3.0e-9
    @test r2.interval == "pre"

    @test_throws "expected exactly one of" AveragingRegion(; nbins=2, t_range_ns=[0.0, 1.0], t_range_s=[0.0, 1.0e-9])
    @test_throws "expected exactly one of" AveragingRegion(; nbins=2)
    @test_throws "nbins must be >= 1" AveragingRegion(; nbins=0, t_range_ns=[0.0, 1.0])
    @test_throws "hi must be greater than lo" AveragingRegion(; nbins=2, t_range_ns=[1.0, 1.0])
    @test_throws "hi must be greater than lo" AveragingRegion(; nbins=2, t_range_ns=[2.0, 1.0])
    @test_throws "must be finite" AveragingRegion(; nbins=2, t_range_ns=[0.0, Inf])
    @test_throws "must be finite" AveragingRegion(; nbins=2, t_range_ns=[NaN, 1.0])

    @test occursin("AveragingRegion", sprint(show, r2))
end

# ------------------------------------------------------------------------------------------
# AV2: a region equal to the pixel grid reproduces the input (bitwise for image/mask/axis;
# sigma at rtol = 4*eps(Float64) -- see the file header comment for why sigma is not bitwise)
# ------------------------------------------------------------------------------------------

@testset "AV2: regions equal to the pixel grid reproduce the input" begin
    nl, nt = 5, 6
    lam, t = _av_axis(nl, nt)
    d = t[2] - t[1]
    image = Float64[10 * l + j for l in 1:nl, j in 1:nt]
    sigma = Float64[0.3 + 0.01 * (10 * l + j) for l in 1:nl, j in 1:nt]
    sd = StreakData(; lambda_m=lam, t_s=t, image=image, sigma=sigma)

    region = AveragingRegion(; nbins=nt, t_range_s=[t[1] - d / 2, t[end] + d / 2])
    out = average_regions(sd, [region])

    @test out.lambda == sd.lambda
    @test out.t == sd.t
    @test all(out.image .=== sd.image)
    @test all(out.mask)
    @test isapprox(out.sigma, sd.sigma; rtol=4 * eps(Float64))
    sigma_bitwise = all(out.sigma .=== sd.sigma)
    @info "AV2: sigma bitwise on this fixture?" sigma_bitwise

    # The construction above (D14's own recipe, [t_1 - d/2, t_N + d/2]) puts hi half a pixel
    # BEYOND the last source pixel, so it never actually lands exactly on one -- a lone region's
    # "closed at the top" rule (D14) is not exercised by it at all (t_N is captured whether the
    # top is open or closed). A boundary landing EXACTLY on the last pixel is the case that rule
    # guards: nbins = 1 so the single window's own hi IS the last pixel's value.
    region_exact = AveragingRegion(; nbins=1, t_range_s=[t[1], t[end]])
    out_exact = average_regions(sd, [region_exact])
    @test out_exact.meta["regions_window_counts"] == [nt]
    @test isapprox(out_exact.t[1], sum(t) / nt; rtol=1e-14)
    @test all(out_exact.mask)
end

# ------------------------------------------------------------------------------------------
# AV3: the by-hand oracle. A 4x9 image, two masked pixels, a partly covered window and a fully
# unusable row -- one region of 3 bins over the whole axis, h = 3 pixels/window exactly.
#
# t_ns = 0..8 (9 pixels), region [-0.5, 8.5] ns, nbins = 3, h = 3 ns:
#   W1 = [-0.5, 2.5) -> j = 1,2,3 (t = 0,1,2)
#   W2 = [2.5, 5.5)  -> j = 4,5,6 (t = 3,4,5)
#   W3 = [5.5, 8.5]  -> j = 7,8,9 (t = 6,7,8)   (closed at top: lone region, D14)
# image[l,j] = 10*l + j (l = 1..4, j = 1..9); sigma[l,j] = 1.0 everywhere.
# mask: all true except mask[2,5] = false (row 2, W2) and mask[3,8] = false (row 3, W3), and
# mask[4,1] = mask[4,2] = mask[4,3] = false (row 4's whole W1 -- the fully unusable row/window).
#
# By hand:
#   row 1 (no masking):        Dbar = [12, 15, 18],           sigmabar = 1/sqrt(3) each
#   row 2 (W2 missing j=5):    Dbar = [22, 25, 28],  W2 sigmabar = sqrt(2)/2, else 1/sqrt(3)
#   row 3 (W3 missing j=8):    Dbar = [32, 35, 38],  W3 sigmabar = sqrt(2)/2, else 1/sqrt(3)
#   row 4 (W1 fully unusable): Dbar = [0, 45, 48],   W1 sigmabar = 1.0 (D14's precedence rule),
#                                                     else 1/sqrt(3)
# usable_counts = [3 3 3; 3 2 3; 3 3 2; 0 3 3], window_counts = [3, 3, 3], tbar = [1,4,7] ns.
# ------------------------------------------------------------------------------------------

@testset "AV3: the by-hand oracle" begin
    nl, nt = 4, 9
    lam = collect(range(500.0e-9, 503.0e-9; length=nl))
    t = collect(0:(nt-1)) .* 1.0e-9
    image = Float64[10 * l + j for l in 1:nl, j in 1:nt]
    sigma = fill(1.0, nl, nt)
    mask = trues(nl, nt)
    mask[2, 5] = false
    mask[3, 8] = false
    mask[4, 1] = false
    mask[4, 2] = false
    mask[4, 3] = false
    sd = StreakData(; lambda_m=lam, t_s=t, image=image, sigma=sigma, mask=mask)

    region = AveragingRegion(; nbins=3, t_range_s=[-0.5e-9, 8.5e-9])
    out = average_regions(sd, [region])

    inv3 = 1 / sqrt(3)
    inv2 = sqrt(2) / 2

    Dbar_expected = [12.0 15.0 18.0
        22.0 25.0 28.0
        32.0 35.0 38.0
        0.0 45.0 48.0]
    sigmabar_expected = [inv3 inv3 inv3
        inv3 inv2 inv3
        inv3 inv3 inv2
        1.0 inv3 inv3]
    mask_expected = trues(4, 3)
    mask_expected[4, 1] = false
    usable_expected = [3 3 3
        3 2 3
        3 3 2
        0 3 3]

    @test isapprox(out.image, Dbar_expected; rtol=1e-14)
    @test isapprox(out.sigma, sigmabar_expected; rtol=1e-14)
    @test out.mask == mask_expected
    @test out.meta["regions_usable_counts"] == usable_expected
    @test isapprox(out.t, [1.0e-9, 4.0e-9, 7.0e-9]; rtol=1e-14)
    @test out.meta["regions_window_counts"] == [3, 3, 3]

    @info "AV3: oracle numbers" Dbar_expected sigmabar_expected
end

# ------------------------------------------------------------------------------------------
# AV4: sigma-bar divides by the count (tight, rtol = 1e-14: the closed form, D14)
# ------------------------------------------------------------------------------------------

@testset "AV4: sigma-bar divides by the count" begin
    n = 5
    t = collect(1:n) .* 1.0e-9
    s0 = 2.0
    image = fill(0.0, 1, n)
    sigma = fill(s0, 1, n)
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t, image=image, sigma=sigma)
    region = AveragingRegion(; nbins=1, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    out = average_regions(sd, [region])
    @test isapprox(out.sigma[1, 1], s0 / sqrt(n); rtol=1e-14)
end

# ------------------------------------------------------------------------------------------
# AV5: a window with no usable pixel in a row -- exact, and holds at f_min = 0.0 too (D14's
# precedence rule: the |U|=0 rule beats the f_min rule)
# ------------------------------------------------------------------------------------------

@testset "AV5: a window with no usable pixel in a row" begin
    t = collect(1:4) .* 1.0e-9
    image = [1.0 2.0 3.0 4.0]
    sigma = fill(1.0, 1, 4)
    mask = falses(1, 4)
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t, image=image, sigma=sigma, mask=mask)
    region = AveragingRegion(; nbins=1, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    for fmin in (0.0, 0.5, 1.0)
        out = average_regions(sd, [region]; min_usable_fraction=fmin)
        @test out.image[1, 1] == 0.0
        @test out.sigma[1, 1] == 1.0
        @test out.mask[1, 1] == false
    end
end

# ------------------------------------------------------------------------------------------
# AV6: the usable-fraction rule at f_min = 0.0, 0.5, 1.0 on a window with 3 of 4 pixels usable
# ------------------------------------------------------------------------------------------

@testset "AV6: the usable-fraction rule" begin
    t = collect(1:4) .* 1.0e-9
    image = [1.0 2.0 3.0 4.0]
    mask = [true true true false]
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t, image=image, mask=mask)
    region = AveragingRegion(; nbins=1, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    @test average_regions(sd, [region]; min_usable_fraction=0.0).mask[1, 1] == true
    @test average_regions(sd, [region]; min_usable_fraction=0.5).mask[1, 1] == true
    @test average_regions(sd, [region]; min_usable_fraction=1.0).mask[1, 1] == false
end

# ------------------------------------------------------------------------------------------
# AV7: t-bar is one vector over all of the window's pixels, masked or not (bitwise on t-bar;
# D-bar and sigma-bar in the masked rows change)
# ------------------------------------------------------------------------------------------

@testset "AV7: t-bar ignores the mask" begin
    t = collect(1:4) .* 1.0e-9
    image = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]
    sigma = fill(1.0, 2, 4)
    mask_a = trues(2, 4)
    mask_b = trues(2, 4)
    mask_b[1, 2] = false
    mask_b[2, 3] = false
    sd_a = StreakData(; lambda_m=[500.0e-9, 501.0e-9], t_s=t, image=image, sigma=sigma, mask=mask_a)
    sd_b = StreakData(; lambda_m=[500.0e-9, 501.0e-9], t_s=t, image=image, sigma=sigma, mask=mask_b)
    region = AveragingRegion(; nbins=1, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    out_a = average_regions(sd_a, [region])
    out_b = average_regions(sd_b, [region])

    @test out_a.t == out_b.t
    @test out_a.image[1, 1] != out_b.image[1, 1]
    @test out_a.image[2, 1] != out_b.image[2, 1]
    @test out_a.sigma[1, 1] != out_b.sigma[1, 1]
    @test out_a.sigma[2, 1] != out_b.sigma[2, 1]
end

# ------------------------------------------------------------------------------------------
# AV8: the time shift moves which window a pixel lands in -- shifting the data and the regions
# by the same amount gives the same image (rtol = 1e-14) and the same axis up to the shift
# (rtol = 1e-12: added then averaged, vs averaged then added)
# ------------------------------------------------------------------------------------------

@testset "AV8: the time shift" begin
    t = collect(0:5) .* 1.0e-9
    image = Float64[10 * l + j for l in 1:1, j in 1:6]
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t, image=image)

    region_a = AveragingRegion(; nbins=2, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    out_a = average_regions(sd, [region_a])

    shift_ns = 0.3
    dt = shift_ns * 1.0e-9
    region_b = AveragingRegion(; nbins=2, t_range_s=[t[1] - 0.5e-9 + dt, t[end] + 0.5e-9 + dt])
    out_b = average_regions(sd, [region_b]; t_shift_ns=shift_ns)

    @test isapprox(out_b.image, out_a.image; rtol=1e-14)
    @test isapprox(out_b.t, out_a.t .+ dt; rtol=1e-12)

    @test_throws "more than one unit keyword" average_regions(sd, [region_a]; t_shift_ns=1.0, t_shift_s=1.0e-9)
end

# ------------------------------------------------------------------------------------------
# AV9: loud errors
# ------------------------------------------------------------------------------------------

@testset "AV9: loud errors" begin
    t = collect(0:5) .* 1.0e-9
    image = fill(1.0, 1, 6)
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t, image=image)

    r_overlap_a = AveragingRegion(; nbins=1, t_range_s=[0.0, 2.0e-9])
    r_overlap_b = AveragingRegion(; nbins=1, t_range_s=[1.0e-9, 3.0e-9])
    @test_throws "sorted and non-overlapping" average_regions(sd, [r_overlap_a, r_overlap_b])

    r_unsorted_a = AveragingRegion(; nbins=1, t_range_s=[2.0e-9, 3.0e-9])
    r_unsorted_b = AveragingRegion(; nbins=1, t_range_s=[0.0, 1.0e-9])
    @test_throws "sorted and non-overlapping" average_regions(sd, [r_unsorted_a, r_unsorted_b])

    r_fine = AveragingRegion(; nbins=100, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    @test_throws "has no source pixel" average_regions(sd, [r_fine])

    r_ok = AveragingRegion(; nbins=1, t_range_s=[t[1] - 0.5e-9, t[end] + 0.5e-9])
    @test_throws "min_usable_fraction" average_regions(sd, [r_ok]; min_usable_fraction=-0.1)
    @test_throws "min_usable_fraction" average_regions(sd, [r_ok]; min_usable_fraction=1.1)

    sd_bad_meta = StreakData(; lambda_m=[500.0e-9], t_s=t, image=image,
        meta=Dict{String,Any}("regions_foo" => 1.0))
    @test_throws "regions_" average_regions(sd_bad_meta, [r_ok])
end

# ------------------------------------------------------------------------------------------
# AV10: the model at the averaged axis. A constant-in-time plasma, so the model is the same at
# every bin regardless of where t-bar falls; compared to a noisy synthetic streak averaged onto
# two regions of unequal bin width, within a 3*sigmabar envelope on every unmasked pixel
# (deliberately loose -- one noisy realization, fixed seed AV10_SEED). A small image (6 lambda,
# 8 output bins = 48 compared pixels) keeps the expected number of >3sigma exceedances small
# (~48*0.0027 ~ 0.13) for a single fixed-seed realization.
# ------------------------------------------------------------------------------------------

const AV10_SEED = 20260923

@testset "AV10: the model at the averaged axis (constant plasma)" begin
    template = Plasma(; n_e_cm3=1.0e19, electrons=(electrons(T_eV=300.0),),
        ions=(ion(mass_amu=2.014, charge=1, T_eV=200.0),))
    nt = 240
    t_ns = collect(range(0.0, 4.0; length=nt))
    trace = PlasmaTrace(template; t_ns=t_ns, leaves=())
    lambda = collect(range(262.0e-9, 264.5e-9; length=6))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60, k_model=:exact)
    effects = (Sensitivity(scale=1.0e-6),)
    expt = Experiment(; trace=trace, probe=probe, collection=collection, lambda_m=lambda, effects=effects)

    sd = synthesize(expt, GaussianNoise(; a=1.0e-4, b=0.0); rng=Xoshiro(AV10_SEED))

    r1 = AveragingRegion(; nbins=5, t_range_ns=[0.0, 1.6])
    r2 = AveragingRegion(; nbins=3, t_range_ns=[1.6, 4.0])
    avg = average_regions(sd, [r1, r2])

    trace_avg = PlasmaTrace(template; t_s=avg.t, axis_name=:t, leaves=())
    expt_avg = Experiment(; trace=trace_avg, probe=probe, collection=collection, lambda_m=lambda, effects=effects)
    model = measure(expt_avg)

    within = abs.(avg.image .- model) .<= 3 .* avg.sigma
    @test all(within[avg.mask])
end

# ------------------------------------------------------------------------------------------
# AV11: rebin invariance of the time derivatives (OQ-19 item 5), deliberately loose
# (rtol = 1e-2). The regularizer's own R = sum(row^2) (unweighted) scales with row COUNT under
# rebinning by design (OQ-19: "R is a plain sum over rows, not a Delta-t-weighted integral", so
# that Lambda stays comparable across binnings) -- it is NOT the quantity that can be
# rebin-invariant. What IS comparable between a fine and a coarse grid is a genuine Riemann-sum
# approximation to the continuum integral integral (D2)^2 dt, weighting each row by its own
# local spacing (D2's row r anchors at bin a = r+1, always interior, so t[a-1] and t[a+1] both
# exist); this is Level 3's own choice of what "the term's value" means quantitatively (rule 11
# zero), flagged in the chunk 5 report for Level 2's review.
# ------------------------------------------------------------------------------------------

function _av11_riemann_d2_value(u::AbstractVector, t::AbstractVector, lo::Real, hi::Real)
    rows = _operator_rows(:D2, u, t)
    s = 0.0
    for r in eachindex(rows)
        a = _operator_anchor(:D2, r)
        (lo <= t[a] <= hi) || continue
        w = (t[a+1] - t[a-1]) / 2
        s += rows[r]^2 * w
    end
    return s
end

@testset "AV11: rebin invariance of the D2 operator" begin
    cubic(t_ns) = 1.0 + 0.3 * t_ns - 0.05 * t_ns^2 + 0.004 * t_ns^3

    nt_fine = 2000
    t_fine_ns = collect(range(0.0, 20.0; length=nt_fine))
    t_fine_s = t_fine_ns .* 1.0e-9
    u_fine = cubic.(t_fine_ns)
    d = t_fine_s[2] - t_fine_s[1]

    image = reshape(u_fine, 1, nt_fine)
    sigma = fill(1.0, 1, nt_fine)
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t_fine_s, image=image, sigma=sigma)

    nbins_coarse = 40
    region = AveragingRegion(; nbins=nbins_coarse, t_range_s=[t_fine_s[1] - d / 2, t_fine_s[end] + d / 2])
    avg = average_regions(sd, [region])
    u_avg = vec(avg.image)

    # D2's own operator drops the outermost window of whichever grid it runs on (its 3-point
    # stencil has no row anchored at bin 1 or bin n). That loss is a much larger FRACTION of a
    # 40-bin coarse grid than of a 2000-pixel fine one, so an edge-to-edge sum over each grid's
    # own rows compares different physical time spans -- an artifact of the comparison, not of
    # rebin invariance. Restrict both sums to one shared interior window, excluding one coarse
    # bin width from each end of the domain, so both grids' Riemann sums cover the same span.
    h_coarse = (t_fine_s[end] - t_fine_s[1] + d) / nbins_coarse
    lo_inner = t_fine_s[1] + h_coarse
    hi_inner = t_fine_s[end] - h_coarse

    value_fine = _av11_riemann_d2_value(u_fine, t_fine_s, lo_inner, hi_inner)
    value_avg = _av11_riemann_d2_value(u_avg, avg.t, lo_inner, hi_inner)
    rel_agreement = abs(value_avg - value_fine) / abs(value_fine)

    @info "AV11: D2 term value agreement" value_fine value_avg rel_agreement
    @test isapprox(value_avg, value_fine; rtol=1e-2)
end

# ------------------------------------------------------------------------------------------
# AV12: regions of unequal bin width join cleanly. t_ns = 0..39 (nt=40, 1 ns spacing); region 1
# [-0.5, 20] ns / 6 bins meets region 2 [20, 39.5] ns / 3 bins exactly at the shared pixel t=20
# ns (region 1's last window therefore half-open, D14's adjacency rule). Hand-verified window
# counts: region 1 -> [3,4,3,4,3,3] (sum 20), region 2 -> [7,6,7] (sum 20).
# ------------------------------------------------------------------------------------------

@testset "AV12: regions of unequal bin width join cleanly" begin
    nt = 40
    t_ns = collect(0:(nt-1))
    t_s = Float64.(t_ns) .* 1.0e-9
    cubic(x) = 2.0 + 0.2 * x - 0.03 * x^2 + 0.002 * x^3
    image = reshape(cubic.(Float64.(t_ns)), 1, nt)
    sigma = fill(1.0, 1, nt)
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t_s, image=image, sigma=sigma)

    mid = t_s[21]  # t = 20 ns, shared exactly between the two regions
    r1 = AveragingRegion(; nbins=6, t_range_s=[t_s[1] - 0.5e-9, mid])
    r2 = AveragingRegion(; nbins=3, t_range_s=[mid, t_s[end] + 0.5e-9])
    avg = average_regions(sd, [r1, r2])

    @test issorted(avg.t)
    @test allunique(avg.t)
    @test avg.meta["regions_window_counts"] == [3, 4, 3, 4, 3, 3, 7, 6, 7]

    u = vec(avg.image)
    rows = _operator_rows(:D2, u, avg.t)
    @info "AV12: D2 rows across the join" rows
    @test all(isfinite, rows)
    med = median(abs.(rows))
    @test all(abs.(rows) .<= 10 * med)
end

# ------------------------------------------------------------------------------------------
# AV13: meta carries all nine keys with the right types and lengths, and a round trip preserves
# them exactly
# ------------------------------------------------------------------------------------------

@testset "AV13: the nine meta keys, and their round trip" begin
    nl, nt = 3, 9
    lam, t = _av_axis(nl, nt)
    image = Float64[10 * l + j for l in 1:nl, j in 1:nt]
    sigma = fill(1.0, nl, nt)
    sd = StreakData(; lambda_m=lam, t_s=t, image=image, sigma=sigma)

    r1 = AveragingRegion(; nbins=2, t_range_s=[t[1] - 0.5e-9, t[4] - 0.5e-9], interval="pre")
    r2 = AveragingRegion(; nbins=2, t_range_s=[t[4] - 0.5e-9, t[end] + 0.5e-9])
    out = average_regions(sd, [r1, r2]; t_shift_ns=0.1, min_usable_fraction=0.7)

    m = out.meta
    @test m["regions_source_t"] isa Vector{Float64}
    @test length(m["regions_source_t"]) == nt
    @test m["regions_t_shift"] isa Float64
    @test m["regions_min_usable_fraction"] isa Float64
    @test m["regions_min_usable_fraction"] == 0.7
    @test m["regions_starts"] isa Vector{Float64} && length(m["regions_starts"]) == 2
    @test m["regions_ends"] isa Vector{Float64} && length(m["regions_ends"]) == 2
    @test m["regions_nbins"] isa Vector{Int} && m["regions_nbins"] == [2, 2]
    @test m["regions_intervals"] isa Vector{String} && m["regions_intervals"] == ["pre", ""]
    @test m["regions_window_counts"] isa Vector{Int} && length(m["regions_window_counts"]) == 4
    @test m["regions_usable_counts"] isa Matrix{Int} && size(m["regions_usable_counts"]) == (nl, 4)

    mktempdir() do dir
        path = joinpath(dir, "av13.jld2")
        write_streak(path, out; source="test")
        back = read_streak(path)
        @test back.meta["regions_source_t"] == m["regions_source_t"]
        @test back.meta["regions_nbins"] == m["regions_nbins"]
        @test back.meta["regions_intervals"] == m["regions_intervals"]
        @test back.meta["regions_usable_counts"] == m["regions_usable_counts"]
    end
end

# ------------------------------------------------------------------------------------------
# AV14: the deck. A deck with [[fit.regions]] (moved from [[data.regions]] at chunk 7, D40
# amendment 11) builds a problem whose trace axis is the averaged axis; both regions and
# t_range_ns is a loud error; a region's interval becomes a FitInterval with the region's own
# bins; the round trip through effective_deck/read_deck reproduces the trace axis bitwise and the
# FitInterval bins; deck.sections is never mutated by build_problem; t_shift_ns without regions is
# a loud error.
# ------------------------------------------------------------------------------------------

function _av_write_streak(path::AbstractString; nl=8, nt=12, lambda_lo=262.0, lambda_hi=264.5,
    t_lo=0.0, t_hi=11.0)
    lambda_m = collect(range(lambda_lo, lambda_hi; length=nl)) .* 1.0e-9
    t_s = collect(range(t_lo, t_hi; length=nt)) .* 1.0e-9
    image = fill(30.0, nl, nt)
    sigma = fill(1.0, nl, nt)
    sd = StreakData(; lambda_m=lambda_m, t_s=t_s, image=image, sigma=sigma)
    write_streak(path, sd; source="test")
    return sd
end

function _av_deck_text(outdir::AbstractString, data_block::AbstractString;
    fit_extra::AbstractString="")
    return """
    deck_version = "otis-deck-2"

    $data_block

    [geometry]
    lambda_probe_nm = 263.25
    theta_deg = 60.0

    [fit.variables]
    T_D = { bounds = [50.0, 800.0], init = 200.0 }

    $fit_extra

    [plasma]
    n_e_cm3 = 3.0e19

    [plasma.electrons.e]
    fraction = 1.0
    T_eV = 400.0

    [plasma.ions.D]
    mass_amu = 2.014
    charge = 1
    fraction = 1.0
    T_eV = "T_D"
    u_kms = 0.0

    [instrument]
    sensitivity = { scale = 1.0e-6 }

    [fit]
    sigma = "supplied"

    [optimizer]
    procedure = "lbfgs"
    lbfgs = { iterations = 3 }

    [output]
    dir = "$(replace(outdir, "\\" => "/"))"
    """
end

@testset "AV14: the deck" begin
    mktempdir() do dir
        datafile = joinpath(dir, "data.jld2")
        _av_write_streak(datafile)
        outdir = joinpath(dir, "out")

        data_block = """
        [data]
        file = "$(replace(datafile, "\\" => "/"))"

        [[fit.regions]]
        t_range_ns = [0.0, 5.5]
        nbins = 4
        interval = "pre"

        [[fit.regions]]
        t_range_ns = [5.5, 11.0]
        nbins = 3
        """
        deckpath = joinpath(dir, "deck.toml")
        write(deckpath, _av_deck_text(outdir, data_block))
        deck = read_deck(deckpath)
        before = deepcopy(deck.sections)

        problem, optimizer, _ = OTIS.build_problem(deck)
        @test length(problem.experiment.t) == 7
        @test issorted(problem.experiment.t)
        # isequal, not ==: the optimizer section carries LBFGS's time_limit = NaN, and NaN == NaN
        # is false while isequal(NaN, NaN) is true (Julia's hash-consistent equality).
        @test isequal(deck.sections, before)

        pre_idx = findfirst(iv -> iv.name == "pre", problem.pspec.intervals)
        @test pre_idx !== nothing
        @test problem.pspec.intervals[pre_idx].bins == 1:4

        # both regions and t_range_ns: a loud error (D40: the message names [[fit.regions]] now)
        bad_block = replace(data_block, "[data]" => "[data]\nt_range_ns = [0.0, 11.0]")
        bad_path = joinpath(dir, "bad1.toml")
        write(bad_path, _av_deck_text(outdir, bad_block))
        @test_throws "and the deck gives [[fit.regions]]" read_deck(bad_path)

        # t_shift_ns without regions: a loud error (D40: the message names [[fit.regions]] now)
        no_regions_block = """
        [data]
        file = "$(replace(datafile, "\\" => "/"))"
        t_shift_ns = 0.1
        """
        bad_path2 = joinpath(dir, "bad2.toml")
        write(bad_path2, _av_deck_text(outdir, no_regions_block))
        @test_throws "without [[fit.regions]]" read_deck(bad_path2)

        # the round trip: build_problem -> effective_deck -> read_deck -> build_problem
        eff = effective_deck(deck, problem, optimizer)
        eff_path = joinpath(dir, "eff.toml")
        write(eff_path, eff)
        deck2 = read_deck(eff_path)
        problem2, _, _ = OTIS.build_problem(deck2)
        @test problem2.experiment.t == problem.experiment.t
        pre_idx2 = findfirst(iv -> iv.name == "pre", problem2.pspec.intervals)
        @test problem2.pspec.intervals[pre_idx2].bins == problem.pspec.intervals[pre_idx].bins
    end
end

# ------------------------------------------------------------------------------------------
# AV15: adjacent regions share an endpoint without double-counting (D14's amendment). Built from
# ONE set of Float64 values so the shared-endpoint equality is exact.
# ------------------------------------------------------------------------------------------

@testset "AV15: adjacent regions share an endpoint without double-counting" begin
    t_s = collect(0:12) .* 1.0e-10
    image = reshape(collect(1.0:13.0), 1, 13)
    sd = StreakData(; lambda_m=[500.0e-9], t_s=t_s, image=image)

    r1 = AveragingRegion(; nbins=2, t_range_s=[t_s[1], t_s[7]])
    r2 = AveragingRegion(; nbins=2, t_range_s=[t_s[7], t_s[13]])
    out = average_regions(sd, [r1, r2])

    wc = out.meta["regions_window_counts"]
    @test sum(wc[1:2]) == 6
    @test sum(wc[3:4]) == 7

    # region-level sums: region 1 owns pixels 1:6 (sum 21), region 2 owns pixels 7:13 (sum 70)
    region1_total = sum(out.image[1, 1:2] .* wc[1:2])
    region2_total = sum(out.image[1, 3:4] .* wc[3:4])
    @test isapprox(region1_total, sum(image[1, 1:6]); rtol=1e-14)
    @test isapprox(region2_total, sum(image[1, 7:13]); rtol=1e-14)
end
