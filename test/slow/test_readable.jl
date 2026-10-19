using Test
using TRUST

# Step 4.M.1's slow rows (Phase 4.M spec D16), gated "readable (slow)": RD11 to RD13 fit the RD deck
# through fit(deck), two cameras with an interval, a held series, two dependents, two scalars (one ending
# on its bound) and a frozen variable bound to an instrument parameter, a problem type nothing in the fast
# suite compiles (about 25 s of compilation), so they are kept out of the phase's share
# (conventions.md 8.6). The fast rows are test/data/test_readable.jl's. fit(deck) pins BLAS to one
# thread for the rest of the process (conventions.md 6.6); this file restores the count it found, as
# test_deck_staged.jl does.

include(joinpath(@__DIR__, "..", "fixtures", "readable_fixture.jl"))

const _RDS_NBLAS = pin_blas_threads()
pin_blas_threads(_RDS_NBLAS)

const _RDS_DIR = mktempdir()
const _RDS_PA = _rd_streak_a(joinpath(_RDS_DIR, "a.jld2"))
const _RDS_PB = _rd_streak_b(joinpath(_RDS_DIR, "b.jld2"); masked=2)
const _RDS_HOLD = write_start(joinpath(_RDS_DIR, "hold.jld2"); axis_ns=[0.0, 1.2],
    values=Dict("T_e" => [390.0, 410.0]), units=Dict("T_e" => "T_eV"), source="rd_test")
const _RDS_OUT = joinpath(_RDS_DIR, "out")
const _RDS_DECK = read_deck(_rd_write(_RDS_DIR, "rd.toml", _rd_deck_text(_RDS_PA, _RDS_PB, _RDS_HOLD, _RDS_OUT)))
const _RDS_RESULT = _rd_fit(_RDS_DECK)
const _RDS_PROBLEM = first(TRUST.build_problem(_RDS_DECK))

# ------------------------------------------------------------------------------------------
# RD11: the RD deck's arrays and chi2 (OQ-63 (a), (b)): an interval, a held series, masked pixels
# ------------------------------------------------------------------------------------------

@testset "RD11: the RD deck: data, sigma, mask, model and chi2 per bin" begin
    rd = _rd_dict(joinpath(_RDS_OUT, "readable.h5"))
    re = _rd_dict(joinpath(_RDS_OUT, "result.jld2"))
    for (c, name) in enumerate(("a", "b"))
        ch = TRUST._channel_problem(_RDS_PROBLEM, c)
        @test isequal(rd["channels/$name/data"], ch.data.image)
        @test isequal(rd["channels/$name/sigma"], ch.sigma)
        @test rd["channels/$name/mask"] == UInt8.(ch.mask)
        @test isequal(rd["channels/$name/model"], re["channels/$name/model"])
    end
    @test count(iszero, rd["channels/b/mask"]) == 6
    TRUST.jldopen(joinpath(_RDS_OUT, "readable.h5"), "r") do f
        for name in ("a", "b")
            chi2, n = _rd_chi2_bins(f, "channels/$name")
            @test isapprox(f["channels/$name/chi2"], chi2; rtol=1e-12)
            @test f["channels/$name/npixels"] == n
            @test isapprox(f["channels/$name/chi2_total"], re["channels/$name/chi2"]; rtol=1e-12)
        end
        @test Set(first.(_rd_leaves(f))) == _rd_expected_keys(_RDS_RESULT, ("a", "b"); deck=true, gn=true)
    end
end

# ------------------------------------------------------------------------------------------
# RD12: every variable and dependent: values, units and flags (OQ-63 (c); U2)
# ------------------------------------------------------------------------------------------

@testset "RD12: params, units and flags: an interval, held, frozen, scalars, a bound, dependents" begin
    r = _RDS_RESULT
    units = Dict("T_D" => "T_eV", "u_D" => "u_kms", "T_e" => "T_eV", "s_b" => "-", "bg" => "-",
        "w_b" => "fwhm_nm", "ne" => "n_e_cm3", "ratio" => "-")
    flags = Dict{String,Any}("T_D" => ["F", "F", "F"], "u_D" => ["", "F", "F"], "T_e" => ["H", "H", "H"],
        "s_b" => "F", "bg" => "B", "w_b" => "H", "ne" => ["D", "D", "D"], "ratio" => ["D", "D", "D"])
    @test TRUST._near_bound_entries(r) == [(variable="bg", bin=0, bound=:lower)]
    rd = _rd_dict(joinpath(_RDS_OUT, "readable.h5"))
    @test Set(n for n in keys(units)) == Set(keys(r.values))
    for (name, unit) in units
        v = r.values[name]
        if name in ("s_b", "bg", "w_b")
            @test rd["params/$name"] isa Float64
            @test rd["params/$name"] === v[1]
        else
            @test rd["params/$name"] isa Vector{Float64}
            @test isequal(rd["params/$name"], v)
        end
        @test rd["units/$name"] == unit
        @test rd["flags/$name"] == flags[name]
    end
    @test isnan.(rd["params/u_D"]) == [true, false, false]
    @test rd["params/ne"] == fill(3.0e19, 3)
    @test isequal(rd["params/T_e"], r.values["T_e"])
    @test all(390.0 .< rd["params/T_e"] .< 410.0)
end

# ------------------------------------------------------------------------------------------
# RD13: laplace/ with scalars and a near-bound entry; fit(deck)'s result beside a rebuilt problem (OQ-63 (g))
# ------------------------------------------------------------------------------------------

@testset "RD13: laplace/ with scalars and a near-bound entry; the rebuilt problem's file" begin
    pol = polish(_RDS_PROBLEM, _RDS_RESULT).result
    lap = laplace(_RDS_PROBLEM, pol.theta)
    mktempdir() do dir
        rd = _rd_dict(write_readable(joinpath(dir, "lap.h5"), pol, _RDS_PROBLEM; laplace=lap))
        @test Set(k[length("laplace/sigma/")+1:end] for k in keys(rd) if startswith(k, "laplace/sigma/")) ==
              Set(["T_D", "u_D", "s_b", "bg"])
        @test isequal(rd["laplace/sigma/u_D"], lap.sigma_post["u_D"])
        for name in ("s_b", "bg")
            @test rd["laplace/sigma/$name"] isa Float64
            @test isequal(rd["laplace/sigma/$name"], lap.sigma_post[name][1])
            @test isequal(rd["laplace/sigma_samp/$name"], lap.sigma_samp[name][1])
            @test rd["laplace/near_bound/$name"] === UInt8(lap.near_bound[name][1])
        end
        @test isnan(rd["laplace/sigma/bg"]) && rd["laplace/near_bound/bg"] === 0x01

        eff = read(joinpath(_RDS_OUT, "effective_deck.toml"), String)
        again = _rd_dict(write_readable(joinpath(dir, "again.h5"), _RDS_RESULT, _RDS_PROBLEM; deck=eff))
        fitted = _rd_dict(joinpath(_RDS_OUT, "readable.h5"))
        @test Set(keys(again)) == setdiff(Set(keys(fitted)), ["fit/gn_decrease"])
        @test sort([k for k in keys(again) if k != "source" && !isequal(again[k], fitted[k])]) == String[]
    end
end
