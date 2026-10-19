# Step 4.E.1: why does one time bin's m leave the truth in a converged GeneralizedNormal fit?
# Spec dev/phase4E_instructions.md D3 to D13 (OQ-52, the user's decision of 2026-09-30, scope (b):
# "Let's investigate it in case it points at a deeper issue").
#
# A MEASUREMENT, not a fix: nothing here changes a shape, a table, a deck, the forward model or an
# optimizer, and nothing is built on what it shows. Step 4.D.2 (generalized_normal_blur.jl) found
# that the staged fit of the two-channel shapes problem converges with one bin's m far from the
# truth (bin 25: m 5.4658 against 4.6111, EPW chi2/n 1032.94 in that bin) while every other bin's
# m is within 0.11 of it. A stationary point that bad is either a genuine local minimum or a point
# where the computed gradient vanishes and the true one does not (a derivative defect). The four
# measurements, in OQ-52's order, each printed in full whatever the one before it showed:
#
#   1. The gradient: central differences of the objective against its autodifferentiated
#      gradient at the reported fit, along bin 25's four coordinates, bin 21's (the control) and
#      the direction from the fit to the truth in bin 25; all in theta, the logit's y.
#   2. The landscape: the objective along bin 25's m from 3.0 to 6.5 with everything else held
#      at the fit (a slice), then with bin 25's logn, T_e and T_D re-optimized at each m (a
#      profile); and along the straight line from the fit to the truth in bin 25.
#   3. The start: a refit from the truth; the stage at which bin 25 leaves the truth; the staged
#      fit from m's init 4.5; the staged fit with L-BFGS for its joint stage.
#   4. The model: bin 25's Jacobian columns against bin 21's; the (m, xi) pairs its pixels
#      evaluate against GN_TABLE's panels; the width of the EPW resonance against one wavelength
#      pixel; the objective along m on a fine grid; and the shape's m-partials at one pixel
#      against central differences of the table's own generator.
#
# The streaks are synthesized first, with BLAS at its default thread count, as examples/shapes.jl
# does (bitwise runs/shapes_generalized_normal_<camera>_streak.jld2 when those files exist); then
# BLAS is pinned to one thread for every fit and measurement (conventions.md 6.6). The staged fit
# is written out inline, exactly as examples/problems.jl's staged_joint_fit does it, so that the
# joint Levenberg-Marquardt result before polish is kept; its polished theta is checked bitwise
# against staged_joint_fit's own.
#
# Every criterion the spec states is applied here and printed as a verdict; the last lines name
# the outcome by the spec's D13. Its output goes to the phase report, never to a committed file;
# the streaks and decks it writes live in a temporary directory and are deleted.
#
# Run:
#   julia --project=. examples/diagnostics/generalized_normal_excursion.jl
#
# Budget: 30 minutes (hard limit 60); Level 2's two runs took 6.7 minutes each on the machine it
# was written on (spec D12).

using TRUST
using Printf
using LinearAlgebra: BLAS, dot, norm, cond, eigen, Symmetric, Diagonal, diag
import ForwardDiff
import TOML

include(joinpath(@__DIR__, "..", "problems.jl"))

const _EXAMPLES_DIR = normpath(joinpath(@__DIR__, ".."))
const _RUNS_DIR = normpath(joinpath(@__DIR__, "..", "..", "runs"))

const BIN = 25                          # the outlying bin of Step 4.D.2's fit
const CONTROL = 21                      # a healthy bin, at the truth's peak
const NAMES = ("logn", "T_e", "m_e", "T_D")
const STEPS = [10.0^-k for k in 1:7]    # central-difference steps in y, D5
const TOL_FD = 1e-2                     # D5: the sweep has converged
const TOL_AGREE = 0.1                   # D5: FD and AD agree; a coordinate is stationary
const TOL_AD = 1e-8                     # D6: the AD paths agree with each other
const DEPART = 0.2                      # D10: a bin's m has left the truth
const M_GRID = collect(3.0:0.25:6.5)    # D8: the slice and the profile
const PROFILE_CAP = 400                 # D8: the profile's solver
const PROFILE_TOL = 1e-8                # D8: its predicted decrease, in the objective's units
const COND_DEGENERATE = 1e4             # D11
const GN13_RTOL = 1e-6                  # D11: GN13's own tolerance on dR/dm

function say(args...)
    println(args...)
    flush(stdout)
    return nothing
end

fmt(v) = join((@sprintf("%.6e", x) for x in v), "  ")
fmt2(v) = join((@sprintf("%.2e", x) for x in v), "  ")

# examples/fit_shapes.jl's deck for GeneralizedNormal (with_wavelength_calibration), its data files
# pointed at this run's streaks; with `m_init`, m_e's init replaced through TOML (D10).
function deck_path(dir, epw_path, iaw_path; m_init=nothing, tag="gn")
    text = with_wavelength_calibration(read(joinpath(_EXAMPLES_DIR, "shapes_generalized_normal.toml"), String))
    for (camera, path) in (("epw", epw_path), ("iaw", iaw_path))
        old = "runs/shapes_generalized_normal_$(camera)_streak.jld2"
        occursin(old, text) || error("deck_path: \"$old\" not found")
        text = replace(text, old => replace(path, "\\" => "/"))
    end
    out = replace(joinpath(dir, "out_$(tag)"), "\\" => "/")
    text = replace(text, "dir = \"runs/shapes/generalized_normal/joint\"" => "dir = \"$out\"")
    if m_init !== nothing
        raw = TOML.parse(text)
        raw["fit"]["variables"]["m_e"]["init"] = Float64(m_init)
        buf = IOBuffer()
        TOML.print(buf, raw; sorted=true)
        text = String(take!(buf))
    end
    path = joinpath(dir, "deck_$(tag).toml")
    write(path, text)
    return path
end

function main()
    t_start = time()
    say("BLAS.get_num_threads() = ", BLAS.get_num_threads(), ", Threads.nthreads() = ", Threads.nthreads())
    mktempdir() do dir
        # ------------------------------------------------------------------------------
        # 0. The streaks and the reported fit (Step 4.D.2's, bitwise)
        # ------------------------------------------------------------------------------
        t0 = time()
        prob = shapes_problem(:generalized_normal)
        paths = String[]
        for camera in (:epw, :iaw)
            sd = synthesize_channel(prob, camera)
            path = joinpath(dir, "gn_$(camera)_streak.jld2")
            write_streak(path, sd; source="generalized_normal_excursion", probe=prob.probe, collection=prob.collection)
            push!(paths, path)
            ref = joinpath(_RUNS_DIR, "shapes_generalized_normal_$(camera)_streak.jld2")
            same = isfile(ref) ? (r = read_streak(ref); all(r.image .=== sd.image) && all(r.sigma .=== sd.sigma)) : nothing
            say("synthesized ", camera, ": ", size(sd.image), "; bitwise runs/shapes_generalized_normal_$(camera)_streak.jld2: ",
                same === nothing ? "no such file" : string(same))
        end
        pin_blas_threads()
        say("pinned: BLAS.get_num_threads() = ", BLAS.get_num_threads())
        dp = deck_path(dir, paths[1], paths[2])
        text = read(dp, String)

        # staged_joint_fit (examples/problems.jl), written out so that the joint LM result is kept.
        stage = mktempdir() do sdir
            res = Dict{String,FitResult}()
            for name in ("epw", "iaw")
                path = joinpath(sdir, "$name.toml")
                write(path, single_channel_deck_text(text, name))
                p, opt, st = TRUST.build_problem(read_deck(path))
                res[name] = solve(p, opt; start=st)
            end
            res
        end
        epw, iaw = stage["epw"], stage["iaw"]
        problem, _, _ = TRUST.build_problem(read_deck(dp))
        ps = problem.pspec
        start = flatten(ps, ["logn" => epw.values["logn"], "T_e" => epw.values["T_e"],
            "m_e" => epw.values["m_e"], "T_D" => iaw.values["T_D"]])
        lm = solve(problem, LevenbergMarquardt(); start=start)
        pol = polish(problem, lm; tol=1e-6)
        r = pol.result
        theta = copy(r.theta)
        t_inline = time() - t0
        t1 = time()
        st = staged_joint_fit(dp; shape_variable="m_e")
        say(@sprintf("staged fit written out: %.1f s; staged_joint_fit: %.1f s; theta bitwise staged_joint_fit's: %s",
            t_inline, time() - t1, string(all(st.result.theta .=== theta))))
        O_fit = objective(problem, theta)
        gd = gn_decrease(problem, theta)
        say(@sprintf("reported fit: status %s, converged %s, LM iterations %d, polish's decreases %s, gn_decrease %.3e",
            r.status, r.converged, lm.iterations, join((@sprintf("%.3e", x) for x in pol.decreases), " "), gd))
        say(@sprintf("  chi2/n EPW %.4f, IAW %.4f; objective O* = %.10e; objective === sum(abs2, _full_residuals): %s",
            r.channels[1].chi2 / r.channels[1].npixels, r.channels[2].chi2 / r.channels[2].npixels, O_fit,
            string(O_fit === sum(abs2, TRUST._full_residuals(problem, theta)))))

        slot(name) = findfirst(v -> v.name == name, ps.variables)
        idx(name, j) = (s = slot(name); ps.bin_entries[j][findfirst(==(s), ps.bin_slots[j])])
        xval(name, th, j) = (v = ps.variables[slot(name)]; TRUST._from_unconstrained(th[idx(name, j)], v.lower, v.upper))
        dxdy(name, th, j) = (v = ps.variables[slot(name)]; x = xval(name, th, j); (x - v.lower) * (v.upper - x) / (v.upper - v.lower))
        vm = ps.variables[slot("m_e")]
        setm(th, j, m) = (th2 = copy(th); th2[idx("m_e", j)] = TRUST._to_unconstrained(m, vm.lower, vm.upper); th2)

        base = Dict(prob.leaves)
        t_ns = t_from_si.(ps.axis, :t_ns)
        truth = Dict("logn" => log10.(base["n_e_cm3"].(t_ns)), "T_e" => base["electrons.e.T_eV"].(t_ns),
            "m_e" => base["electrons.e.f.m"].(t_ns), "T_D" => base["ions.D.T_eV"].(t_ns))
        theta_truth = flatten(ps, [k => truth[k] for k in NAMES])
        m_fit, m_true = xval("m_e", theta, BIN), truth["m_e"][BIN]
        say(@sprintf("bin %d (t = %.4f ns): fit  logn %.6f  T_e %.4f  m %.4f  T_D %.4f", BIN, t_ns[BIN],
            (xval(n, theta, BIN) for n in NAMES)...))
        say(@sprintf("bin %d (t = %.4f ns): true logn %.6f  T_e %.4f  m %.4f  T_D %.4f", BIN, t_ns[BIN],
            (truth[n][BIN] for n in NAMES)...))
        say(@sprintf("[section 0: %.1f s]", time() - t0))

        # ------------------------------------------------------------------------------
        # 1. The gradient
        # ------------------------------------------------------------------------------
        say("=" ^ 100)
        say("MEASUREMENT 1: the gradient at the reported fit (theta is y, the logit's unconstrained coordinate)")
        t0 = time()
        rfull = TRUST._full_residuals(problem, theta)
        Jac = jacobian(problem, theta)
        g = zeros(ps.ntheta)
        gradient!(g, problem, theta)
        g2 = zeros(ps.ntheta)
        TRUST.jtvp!(g2, problem, Jac, rfull)
        g2 .*= 2
        say(@sprintf("gradient! === 2 jtvp!(jacobian, r): %s; ||g||_2 = %.4e, ||g||_inf = %.4e",
            string(all(g .=== g2)), norm(g), maximum(abs, g)))
        col = zeros(length(rfull))
        verdicts1 = String[]
        scaled1 = Float64[]
        function check(label, d, gd_ad)
            TRUST.jvp!(col, problem, Jac, d)
            H = dot(col, col)
            s = 2 * sqrt(H)
            g_col = 2 * dot(col, rfull)
            g_dual = ForwardDiff.derivative(t -> objective(problem, theta .+ t .* d), 0.0)
            D = [(objective(problem, theta .+ h .* d) - objective(problem, theta .- h .* d)) / (2h) for h in STEPS]
            succ = [abs(D[k] - D[k+1]) for k in 1:length(D)-1]
            k = argmin(succ)
            Dstar, E = D[k], succ[k]
            ad_ok = abs(g_col - gd_ad) <= TOL_AD * s && abs(g_dual - gd_ad) <= TOL_AD * s
            verdict = if E / s > TOL_FD
                "FD NOT CONVERGED"
            elseif abs(Dstar - gd_ad) / s > TOL_AGREE
                "DERIVATIVE DEFECT"
            elseif abs(gd_ad) / s <= TOL_AGREE && abs(Dstar) / s <= TOL_AGREE
                "agree, stationary"
            else
                "agree, NOT stationary"
            end
            ad_ok || (verdict *= "; AD PATHS DISAGREE")
            push!(verdicts1, verdict)
            push!(scaled1, abs(gd_ad) / s)
            say(@sprintf("%s: scale 2sqrt(|Jd|^2) = %.4e", label, s))
            say(@sprintf("   AD: gradient! %.6e | 2(Jd)'r %.6e | ForwardDiff through objective %.6e", gd_ad, g_col, g_dual))
            say("   FD, h = 1e-1 .. 1e-7: ", fmt(D))
            say("   (FD - AD)/scale:      ", fmt2((D .- gd_ad) ./ s))
            say(@sprintf("   quoted h = %.0e: FD %.6e, sweep error %.2e, (FD - AD)/scale %+.2e, AD/scale %+.2e -> %s",
                STEPS[k], Dstar, E / s, (Dstar - gd_ad) / s, gd_ad / s, verdict))
        end
        for j in (BIN, CONTROL), n in NAMES
            d = zeros(ps.ntheta)
            d[idx(n, j)] = 1.0
            check(@sprintf("%-4s bin %d (x = %.6g, dx/dy = %.4g)", n, j, xval(n, theta, j), dxdy(n, theta, j)), d, g[idx(n, j)])
        end
        d_truth = zeros(ps.ntheta)
        for n in NAMES
            d_truth[idx(n, BIN)] = theta_truth[idx(n, BIN)] - theta[idx(n, BIN)]
        end
        check(@sprintf("the direction fit -> truth in bin %d (y: %s)", BIN,
            join((@sprintf("%s %+.4f", n, d_truth[idx(n, BIN)]) for n in NAMES), ", ")), d_truth, dot(g, d_truth))
        m1_defect = any(v -> occursin("DEFECT", v) || occursin("DISAGREE", v), verdicts1)
        m1_stationary = all(v -> v == "agree, stationary", verdicts1)
        say("verdicts: ", join(verdicts1, " | "))
        say(@sprintf("[section 1: %.1f s]", time() - t0))

        # ------------------------------------------------------------------------------
        # 2. The landscape
        # ------------------------------------------------------------------------------
        say("=" ^ 100)
        say("MEASUREMENT 2: the objective along bin $(BIN)'s m (O - O*; parts: EPW chi2, IAW chi2, EPW bin-$(BIN) chi2/n, each regularizer term)")
        t0 = time()
        ch1 = problem.channels[1]
        nl1 = size(ch1.data.image, 1)
        rows_bin = problem.row_ranges[1][[k for (k, i) in enumerate(ch1.pixel_index) if div(i - 1, nl1) + 1 == BIN]]
        function parts(th)
            rr = TRUST._full_residuals(problem, th)
            terms = [sum(abs2, view(rr, problem.npixels .+ term.rows)) for term in problem.regularizer.terms]
            return (O=sum(abs2, rr), epw=sum(abs2, view(rr, problem.row_ranges[1])), iaw=sum(abs2, view(rr, problem.row_ranges[2])),
                bin=sum(abs2, view(rr, rows_bin)) / length(rows_bin), terms=terms)
        end
        pline(q) = @sprintf("%+.6e  %.6e  %.6e  %9.2f  %s", q.O - O_fit, q.epw, q.iaw, q.bin, join((@sprintf("%.4e", x) for x in q.terms), " "))
        grid = sort(unique(vcat(M_GRID, [m_fit, m_true])))
        say("slice (every other coordinate at the fit): m, O - O*, EPW chi2, IAW chi2, bin chi2/n, terms")
        slice = Float64[]
        for m in grid
            q = parts(setm(theta, BIN, m))
            push!(slice, q.O - O_fit)
            say(@sprintf("  %.4f  %s", m, pline(q)))
        end
        nmin_slice = count(i -> slice[i] < slice[i-1] && slice[i] < slice[i+1], 2:length(slice)-1)
        say("line from the fit (s = 0) to the truth (s = 1) in bin $(BIN)'s four coordinates: s, m, O - O*, parts")
        line = Float64[]
        for sfrac in 0.0:0.1:1.0
            th = theta .+ sfrac .* d_truth
            q = parts(th)
            push!(line, q.O - O_fit)
            say(@sprintf("  %.1f  %.4f  %s", sfrac, xval("m_e", th, BIN), pline(q)))
        end
        free = [idx("logn", BIN), idx("T_e", BIN), idx("T_D", BIN)]
        function reoptimize(th0)
            th = copy(th0)
            rr = TRUST._full_residuals(problem, th)
            Ocur = sum(abs2, rr)
            mu = 1e-3
            c = zeros(length(rr))
            C = zeros(length(rr), length(free))
            dec = NaN
            for it in 1:PROFILE_CAP
                J = jacobian(problem, th)
                for (q, i) in enumerate(free)
                    e = zeros(ps.ntheta)
                    e[i] = 1.0
                    TRUST.jvp!(c, problem, J, e)
                    C[:, q] .= c
                end
                A = C' * C
                gg = C' * rr
                dec = dot(gg, A \ gg)
                dec <= PROFILE_TOL && return th, Ocur, it, dec, "converged"
                accepted = false
                for _ in 1:30
                    step = -((A + mu * Diagonal(diag(A))) \ gg)
                    tn = copy(th)
                    tn[free] .+= step
                    rn = TRUST._full_residuals(problem, tn)
                    On = sum(abs2, rn)
                    if On < Ocur
                        th, rr, Ocur = tn, rn, On
                        mu = max(mu / 3, 1e-12)
                        accepted = true
                        break
                    end
                    mu *= 4
                end
                accepted || return th, Ocur, it, dec, "no step lowers O"
            end
            return th, Ocur, PROFILE_CAP, dec, "cap"
        end
        th_chk, O_chk, it_chk, dec_chk, why_chk = reoptimize(theta)
        say(@sprintf("the profile's solver started at the fit itself: O - O* %+.3e after %d iterations (%s, last predicted decrease %.2e)",
            O_chk - O_fit, it_chk, why_chk, dec_chk))
        say("profile (bin $(BIN)'s logn, T_e, T_D re-optimized from the fit at each m): m, O - O*, parts, logn, T_e, T_D, iterations, stop")
        profile = Float64[]
        conv = Bool[]
        for m in grid
            th, Op, it, dec, why = reoptimize(setm(theta, BIN, m))
            q = parts(th)
            push!(profile, Op - O_fit)
            push!(conv, why != "cap")
            say(@sprintf("  %.4f  %s  %.5f %.3f %.3f  %d %s (%.1e)", m, pline(q), xval("logn", th, BIN), xval("T_e", th, BIN),
                xval("T_D", th, BIN), it, why, dec))
        end
        cm = findall(conv)
        pmins = [grid[cm[i]] for i in 2:length(cm)-1 if profile[cm[i]] < profile[cm[i-1]] && profile[cm[i]] < profile[cm[i+1]]]
        i_true = findfirst(==(m_true), grid)
        m2_deeper = profile[i_true] < 0
        m2_rises = maximum(line) > 0
        say(@sprintf("readings: slice's interior local minima on the grid %d; profile's interior local minima (stopped points) at m = %s; profile at the truth's m %+.6e (the truth's basin deeper: %s); the line's highest point %+.6e at s = %.1f (it rises above O*: %s)",
            nmin_slice, join((@sprintf("%.4f", x) for x in pmins), ", "), profile[i_true], string(m2_deeper), maximum(line),
            (argmax(line) - 1) / 10, string(m2_rises)))
        say(@sprintf("[section 2: %.1f s]", time() - t0))

        # ------------------------------------------------------------------------------
        # 3. The start
        # ------------------------------------------------------------------------------
        say("=" ^ 100)
        say("MEASUREMENT 3: the start")
        t0 = time()
        departed(values) = [j for j in eachindex(values) if abs(values[j] - truth["m_e"][j]) > DEPART]
        function report(label, res, t)
            m = res.values["m_e"]
            dep = abs.(m .- truth["m_e"])
            jw = argmax(dep)
            say(@sprintf("%-32s %6.1f s  %-12s it %4d  O - O* %+.6e  EPW %.4f  IAW %.4f  m(bin %d) %.4f  worst bin %d (m %.4f against %.4f)  bins with |m - truth| > %.1f: %s",
                label, t, res.status, res.iterations, objective(problem, res.theta) - O_fit, res.channels[1].chi2 / res.channels[1].npixels,
                res.channels[2].chi2 / res.channels[2].npixels, BIN, m[BIN], jw, m[jw], truth["m_e"][jw], DEPART,
                isempty(departed(m)) ? "none" : join(departed(m), ", ")))
            return m[BIN]
        end
        say(@sprintf("stages of the reported fit at bin %d (truth: m %.4f):", BIN, m_true))
        say(@sprintf("  EPW-only stage:  logn %.6f  T_e %.4f  m %.4f  (bins with |m - truth| > %.1f: %s)", epw.values["logn"][BIN],
            epw.values["T_e"][BIN], epw.values["m_e"][BIN], DEPART, isempty(departed(epw.values["m_e"])) ? "none" : join(departed(epw.values["m_e"]), ", ")))
        say(@sprintf("  IAW-only stage:  logn %.6f  T_e %.4f  m %.4f  T_D %.4f", iaw.values["logn"][BIN], iaw.values["T_e"][BIN],
            iaw.values["m_e"][BIN], iaw.values["T_D"][BIN]))
        stage_m = [("the joint start", start), ("the joint LM", lm.theta), ("polish", theta)]
        for (label, th) in stage_m
            say(@sprintf("  %-16s logn %.6f  T_e %.4f  m %.4f  T_D %.4f  O - O* %+.6e", label * ":", (xval(n, th, BIN) for n in NAMES)...,
                objective(problem, th) - O_fit))
        end
        first_stage = abs(epw.values["m_e"][BIN] - m_true) > DEPART ? "the EPW-only stage" :
                      abs(xval("m_e", lm.theta, BIN) - m_true) > DEPART ? "the joint LM" :
                      abs(m_fit - m_true) > DEPART ? "polish" : "none"
        say("  the first stage with bin $(BIN)'s |m - truth| > $(DEPART): ", first_stage)
        t = @elapsed begin
            lm_t = solve(problem, LevenbergMarquardt(); start=theta_truth)
            pol_t = polish(problem, lm_t; tol=1e-6)
        end
        say(@sprintf("the truth itself: O - O* %+.6e", objective(problem, theta_truth) - O_fit))
        m_refit = report("refit from the truth (LM, polish)", pol_t.result, t)
        say(@sprintf("  its gn_decrease %.3e; polish's decreases %s", gn_decrease(problem, pol_t.result.theta),
            join((@sprintf("%.3e", x) for x in pol_t.decreases), " ")))
        m3_stays = abs(m_refit - m_true) <= DEPART && objective(problem, pol_t.result.theta) < O_fit
        dp45 = deck_path(dir, paths[1], paths[2]; m_init=4.5, tag="init45")
        t = @elapsed st45 = staged_joint_fit(dp45; shape_variable="m_e")
        report("staged, m_e init 4.5", st45.result, t)
        say(@sprintf("  its EPW-only stage: m(bin %d) %.4f", BIN, st45.epw.values["m_e"][BIN]))
        t = @elapsed stl = staged_joint_fit(dp; optimizer=LBFGS(), shape_variable="m_e")
        report("staged, joint stage LBFGS", stl.result, t)
        say(@sprintf("readings: the refit from the truth stays (|m - truth| <= %.1f at bin %d, O below O*): %s; the excursion enters at: %s",
            DEPART, BIN, string(m3_stays), first_stage))
        say(@sprintf("[section 3: %.1f s]", time() - t0))

        # ------------------------------------------------------------------------------
        # 4. The model
        # ------------------------------------------------------------------------------
        say("=" ^ 100)
        say("MEASUREMENT 4: the model")
        t0 = time()
        conds = Dict{Tuple{String,Int},Float64}()
        mnorm = Dict{Tuple{String,Int},Float64}()
        for (label, th) in (("fit", theta), ("truth", theta_truth)), j in (BIN, CONTROL)
            J = jacobian(problem, th)
            c = zeros(problem.npixels + problem.regularizer.nrows)
            C = zeros(problem.npixels, 4)
            for (q, n) in enumerate(NAMES)
                e = zeros(ps.ntheta)
                e[idx(n, j)] = 1.0
                TRUST.jvp!(c, problem, J, e)
                C[:, q] .= view(c, 1:problem.npixels)
            end
            G = C' * C
            dn = sqrt.(diag(G))
            R = G ./ (dn * dn')
            R3 = R[1:3, 1:3]
            ev = eigen(Symmetric(R3))
            conds[(label, j)] = cond(R3)
            mnorm[(label, j)] = dn[3]
            say(@sprintf("%-5s bin %d: data-row column norms (logn, T_e, m, T_D) %s", label, j, join((@sprintf("%.4e", x) for x in dn), " ")))
            say(@sprintf("   correlations logn-T_e %+.6f  logn-m %+.6f  T_e-m %+.6f; cond (logn, T_e, m) %.4e; smallest eigenvalue %.3e along %s",
                R[1, 2], R[1, 3], R[2, 3], cond(R3), ev.values[1], join((@sprintf("%+.3f", x) for x in ev.vectors[:, 1]), " ")))
        end
        n_off_table = 0
        for j in (BIN, CONTROL), c in 1:2
            ch = problem.channels[c]
            expt = unflatten(ch.experiment, ps, theta; channel=c)
            pl = plasma(expt.trace, j)
            sp = pl.electrons[1]
            nl = length(expt.lambda)
            rows = [(i - 1) % nl + 1 for i in ch.pixel_index if div(i - 1, nl) + 1 == j]
            xs = Float64[]
            for l in rows
                kk, om, _ = wavevector(expt.probe, expt.collection, expt.lambda[l], pl)
                push!(xs, xi(sp, kk, om))
            end
            ax = abs.(xs)
            nser, nfall = 0, 0
            for x in xs
                abs(x) <= TRUST.GN_TABLE.xi_max && continue
                _, ok = TRUST._cauchy_series(sp.f, x)
                ok ? (nser += 1) : (nfall += 1)
            end
            n_off_table += nser + nfall
            panels = sort(unique([TRUST._cheb_panel_index(TRUST.GN_TABLE, a) for a in ax if a <= TRUST.GN_TABLE.xi_max]))
            say(@sprintf("bin %d %s at the fit: %d used pixels, m %.4f, xi %.4f to %.4f (|xi| %.4f to %.4f); table %d (panels %s), series %d, fallback %d",
                j, ch.name, length(xs), sp.f.m, minimum(xs), maximum(xs), minimum(ax), maximum(ax), count(<=(TRUST.GN_TABLE.xi_max), ax),
                join(panels, ","), nser, nfall))
        end
        say("GN_TABLE's panel edges in |xi|: ", join((@sprintf("%g", e) for e in TRUST.GN_TABLE.edges), ", "))
        widths = Dict{Tuple{String,Int},Float64}()
        peaks = Dict{Tuple{String,Int},Float64}()
        for (label, th) in (("fit", theta), ("truth", theta_truth)), j in (BIN, CONTROL)
            expt = unflatten(problem.channels[1].experiment, ps, th; channel=1)
            pl = plasma(expt.trace, j)
            lam = expt.lambda
            dl = lam[2] - lam[1]
            S = spectrum(pl, expt.probe, expt.collection, lam)
            k = argmax(S)
            lo, hi = lam[k-4] - dl / 2, lam[k+4] + dl / 2
            lf = collect(range(lo, hi; length=9 * 256 + 1))
            Sf = spectrum(pl, expt.probe, expt.collection, lf)
            kf = argmax(Sf)
            a, b = kf, kf
            while a > 1 && Sf[a] > Sf[kf] / 2
                a -= 1
            end
            while b < length(Sf) && Sf[b] > Sf[kf] / 2
                b += 1
            end
            w = (lf[b] - lf[a]) / dl
            widths[(label, j)] = w
            peaks[(label, j)] = Sf[kf] / S[k]
            ratios = Float64[]
            for q in k-3:k+3
                sel = findall(x -> lam[q] - dl / 2 <= x < lam[q] + dl / 2, lf)
                push!(ratios, S[q] / (sum(Sf[sel]) / length(sel)))
            end
            kk, om, _ = wavevector(expt.probe, expt.collection, lf[kf], pl)
            sp = pl.electrons[1]
            x = xi(sp, kk, om)
            say(@sprintf("%-5s bin %d: m %.4f  EPW resonance at %.4f nm, FWHM %.4f nm = %.3f wavelength pixels (pixel %.5f nm); its peak %.3f x the largest pixel value; xi there %.4f, dphi/dx %.3e",
                label, j, sp.f.m, lf[kf] * 1e9, (lf[b] - lf[a]) * 1e9, w, dl * 1e9, Sf[kf] / S[k], x, dfdx(sp.f, x)))
            say("   pixel value over its pixel's average, pixels k-3 .. k+3 about the largest: ", join((@sprintf("%.3f", q) for q in ratios), " "))
        end
        for (j, mlo, mhi) in ((BIN, 4.40, 5.80), (CONTROL, 4.40, 5.20))
            ms = collect(mlo:0.002:mhi)
            vals = [objective(problem, setm(theta, j, m)) - O_fit for m in ms]
            mins = [ms[i] for i in 2:length(ms)-1 if vals[i] < vals[i-1] && vals[i] < vals[i+1]]
            say(@sprintf("fine slice, bin %d, m %.2f to %.2f by 0.002 (every other coordinate at the fit): interior local minima at m = %s; largest O - O* %.4e at m %.3f",
                j, mlo, mhi, join((@sprintf("%.3f", x) for x in mins), ", "), maximum(vals), ms[argmax(vals)]))
            say("   every 0.02 in m: ", join((@sprintf("%.2f:%+.3e", ms[i], vals[i]) for i in 1:10:length(ms)), "  "))
        end
        row_ok = true
        for (label, th) in (("fit", theta), ("truth", theta_truth))
            expt = unflatten(problem.channels[1].experiment, ps, th; channel=1)
            pl = plasma(expt.trace, BIN)
            S = spectrum(pl, expt.probe, expt.collection, expt.lambda)
            k = argmax(S)
            kk, om, _ = wavevector(expt.probe, expt.collection, expt.lambda[k], pl)
            x = xi(pl.electrons[1], kk, om)
            m = pl.electrons[1].f.m
            h = 1e-4
            ad_R = ForwardDiff.derivative(mm -> real(cauchy_dfdv(GeneralizedNormal(mm), x)), m)
            fd_R = (TRUST._gn_principal_value(m + h, x) - TRUST._gn_principal_value(m - h, x)) / (2h)
            ad_I = ForwardDiff.derivative(mm -> imag(cauchy_dfdv(GeneralizedNormal(mm), x)), m)
            fd_I = pi * (dfdx(TRUST._gn_uncached(m + h), x) - dfdx(TRUST._gn_uncached(m - h), x)) / (2h)
            ad_p = ForwardDiff.derivative(mm -> GeneralizedNormal(mm)(x), m)
            fd_p = (TRUST._gn_uncached(m + h)(x) - TRUST._gn_uncached(m - h)(x)) / (2h)
            tab = ForwardDiff.derivative(mm -> TRUST._cheb_table_eval(TRUST.GN_TABLE, TRUST._cheb_slice(TRUST.GN_TABLE, mm), x), m)
            seeds = [s * ForwardDiff.derivative(t -> real(cauchy_dfdv(GeneralizedNormal(s * t), x)), s * m) for s in (1.0, -1.0)]
            rel = [abs(ad_R - fd_R) / abs(fd_R), abs(ad_I - fd_I) / abs(fd_I), abs(ad_p - fd_p) / abs(fd_p)]
            same = seeds[1] === tab && seeds[2] === tab
            row_ok &= all(<=(GN13_RTOL), rel) && same
            say(@sprintf("row-level check, %s, bin %d's largest EPW pixel (%d): m %.6f, xi %.6f", label, BIN, k, m, x))
            say(@sprintf("   dR/dm    AD %.10e  FD of the generator %.10e  relative %.2e; seeds +1 and -1 === the table's own derivative: %s",
                ad_R, fd_R, rel[1], string(same)))
            say(@sprintf("   dIm/dm   AD %.10e  FD (pi dphi/dx)      %.10e  relative %.2e", ad_I, fd_I, rel[2]))
            say(@sprintf("   dphi/dm  AD %.10e  FD                   %.10e  relative %.2e", ad_p, fd_p, rel[3]))
        end
        m4_unresolved = widths[("fit", BIN)] < 1 && widths[("truth", BIN)] >= 1
        say(@sprintf("readings: resonance narrower than one pixel at the fit's bin %d: %s (%.3f), at the truth's: %s (%.3f); m's column at the fit's bin %d %s the truth's (%.4e against %.4e); cond (logn, T_e, m) at the fit's bin %d %.4e, at bin %d %.4e (degenerate, >= %.0e: %s); pixels off the table %d; row-level check within %.0e and seeds bitwise: %s",
            BIN, string(widths[("fit", BIN)] < 1), widths[("fit", BIN)], string(widths[("truth", BIN)] < 1), widths[("truth", BIN)],
            BIN, mnorm[("fit", BIN)] < mnorm[("truth", BIN)] ? "is smaller than" : "is not smaller than", mnorm[("fit", BIN)],
            mnorm[("truth", BIN)], BIN, conds[("fit", BIN)], CONTROL, conds[("fit", CONTROL)], COND_DEGENERATE,
            string(conds[("fit", BIN)] >= COND_DEGENERATE), n_off_table, GN13_RTOL, string(row_ok)))
        say(@sprintf("[section 4: %.1f s]", time() - t0))

        # ------------------------------------------------------------------------------
        # The outcome, by the spec's D13
        # ------------------------------------------------------------------------------
        say("=" ^ 100)
        outcome = if m1_defect || !row_ok
            "(A) a derivative defect"
        elseif m1_stationary && m2_deeper && m2_rises && m3_stays && m4_unresolved
            "(B) the model: the EPW resonance narrower than a wavelength pixel at the fit"
        elseif m1_stationary && m2_deeper && m2_rises && m3_stays
            "(C) a wrong basin"
        else
            "(D) unknown"
        end
        say("OUTCOME by D13: ", outcome)
        say(@sprintf("  brackets: [B] %d  [M] %.4f  [T] %.4f  [W] %.3f  [R] %.3f  [Wt] %.3f  [DO] %.6e  [MR] %.4f  [ST] %s  [G] %.2e",
            BIN, m_fit, m_true, widths[("fit", BIN)], peaks[("fit", BIN)], widths[("truth", BIN)], objective(problem, pol_t.result.theta) - O_fit,
            m_refit, first_stage, maximum(scaled1)))
    end
    say(@sprintf("total wall time %.1f s", time() - t_start))
    return nothing
end

main()
