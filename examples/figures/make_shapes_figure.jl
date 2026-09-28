# Phase 4.B Step 4.B.5: the shapes and what each does to the IAW and EPW features. Spec D40.
# A physics figure (conventions.md 10.4's exception): no fit, no noise, about a second of
# spectral_density calls. Not part of make_figures.jl/FIGURE_NAMES: FG1 asserts exactly five
# files.
#
# Run by hand:
#   julia --project=examples/figures -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=examples/figures examples/figures/make_shapes_figure.jl [outdir]
#
# `outdir` defaults to this directory (the committed location).

using CairoMakie
using OTIS

Makie.set_theme!(fontsize=14)

function _save_info(path, fig)
    t0 = time()
    save(path, fig; px_per_unit=1)
    dt = time() - t0
    @info "figure" name = basename(path) bytes = filesize(path) seconds = dt
    return path
end

# The five shapes shown in every panel, in one fixed order and color so a reader can match a
# curve across panels from the one legend below them; the Maxwellian black, the rest the default
# categorical palette (spec D40; the legend moved below the panels at Phase 4.C spec D3).
const _PALETTE = Makie.wong_colors()
const _SHAPES = (
    ("Maxwellian", Maxwellian(), :black),
    ("Kappa, κ=3.0", Kappa(3.0), _PALETTE[1]),
    ("GeneralizedNormal, m=4.0", GeneralizedNormal(4.0), _PALETTE[2]),
    ("SkewNormal, α=3.0", SkewNormal(3.0), _PALETTE[3]),
    ("SkewNormal, α=-3.0", SkewNormal(-3.0), _PALETTE[4]),
)

# KP23's fixture, exactly (test/distributions/test_kappa.jl's _kp_iaw_plasma, _KP_PROBE,
# _KP_COLLECTION): n_e = 5e19 cm^-3, T_e = 500 eV, deuterons at 250 eV, probe 263.25 nm, theta =
# 60 deg, k_model = :exact.
_iaw_plasma(f) = Plasma(n_e_cm3=5e19, electrons=(electrons(T_eV=500.0, f=f),),
    ions=(ion(mass_amu=2.014, charge=1, T_eV=250.0),))
const _PROBE = ProbeBeam(lambda_probe_nm=263.25)
const _COLLECTION = Collection(theta_deg=60, k_model=:exact)

function panel_shapes!(ax)
    xs = collect(range(-5, 5; length=401))
    for (label, f, color) in _SHAPES
        lines!(ax, xs, f.(xs); color=color, label=label)
    end
    return ax
end

function panel_iaw!(ax)
    plasma_M = _iaw_plasma(Maxwellian())
    (k, _, _) = wavevector(_PROBE, _COLLECTION, 263.25e-9, plasma_M)
    cs = sqrt((500.0 + 3 * 250.0) * 1.602176634e-19 / (2.014 * 1.66053906660e-27))
    xr = collect(range(-3, 3; length=1201))
    omegas = xr .* (k * cs)

    s_M = [spectral_density(plasma_M, k, om) for om in omegas]
    norm = maximum(s_M)
    for (label, f, color) in _SHAPES
        plasma_f = _iaw_plasma(f)
        s = [spectral_density(plasma_f, k, om) for om in omegas]
        lines!(ax, xr, s ./ norm; color=color, label=label)
    end
    return ax
end

function panel_epw!(ax)
    lambda_s_nm = collect(230.0:0.05:258.0)
    lambda_s_m = lambda_s_nm .* 1e-9

    plasma_M = _iaw_plasma(Maxwellian())
    s_M = map(lambda_s_m) do lam
        (k, om, _) = wavevector(_PROBE, _COLLECTION, lam, plasma_M)
        spectral_density(plasma_M, k, om)
    end
    norm = maximum(s_M)
    for (label, f, color) in _SHAPES
        plasma_f = _iaw_plasma(f)
        s = map(lambda_s_m) do lam
            (k, om, _) = wavevector(_PROBE, _COLLECTION, lam, plasma_f)
            spectral_density(plasma_f, k, om)
        end
        lines!(ax, lambda_s_nm, s ./ norm; color=color, label=label)
    end
    return ax
end

function main(outdir=@__DIR__)
    mkpath(outdir)

    fig = Figure(size=(1500, 500))

    ax_a = Axis(fig[1, 1]; xlabel="x", ylabel="φ(x)", yscale=log10, title="Velocity shapes, variance ½")
    panel_shapes!(ax_a)
    ylims!(ax_a, 1e-6, 2.0)

    ax_b = Axis(fig[1, 2]; xlabel="ω / (k c_s)", ylabel="S(k,ω) / max(S_Maxwellian)")
    panel_iaw!(ax_b)

    ax_c = Axis(fig[1, 3]; xlabel="λ_s (nm)", ylabel="S(k,ω) / max(S_Maxwellian)")
    panel_epw!(ax_c)

    Legend(fig[2, 1:3], ax_a; orientation=:horizontal, framevisible=false)

    return _save_info(joinpath(outdir, "shapes.png"), fig)
end

main(length(ARGS) >= 1 ? ARGS[1] : @__DIR__)
