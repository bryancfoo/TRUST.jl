using Test
using OTIS
using OTIS: _full_residuals, _regularizer_closed_form, _regularizer_value, _regularizer_raw_rows,
    _regularizer_gradient!, _operator_rows
import ForwardDiff

# RG1 to RG21 against dev/phase3_instructions.md Section 6's RG table (spec D22-D25), gated
# "regularizer". The axis throughout is deliberately nonuniform (D22's own prompt), so that an
# operator written against the bin index rather than the time fails.

const _RG_T_NS = [0.0, 0.31, 0.7, 1.05, 1.5, 1.9, 2.4, 2.75, 3.2, 3.6]

# ------------------------------------------------------------------------------------------
# shared fixtures
# ------------------------------------------------------------------------------------------

function _rg_template()
    return Plasma(; n_e_cm3=3e19,
        electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0, fraction=0.7),
              ion(name="C", mass_amu=12.0, charge=6, T_eV=200.0, u_kms=0.0, fraction=0.3)])
end

function _rg_expt(t_ns::AbstractVector{<:Real}=_RG_T_NS)
    trace = PlasmaTrace(_rg_template(); t_ns=collect(Float64, t_ns))
    probe = ProbeBeam(lambda_probe_nm=263.25)
    collection = Collection(theta_deg=60.0, k_model=:exact)
    lambda_m = collect(range(262.0, 264.5; length=12)) .* 1e-9
    return Experiment(trace=trace, probe=probe, collection=collection, lambda_m=lambda_m)
end

# One time-varying variable "u1" (bound to ions.D.u_kms, so its own unit is km/s), one scalar
# variable "f_D" (bound to ions.D.fraction), and two dependents: "f_C" (constant across bins,
# RG8's "absolute on a dependent" case) and "dep1" (time-varying, so its own D2 rows are
# nontrivial -- RG12/RG19's dependent-selection case).
function _rg_pspec(expt::Experiment; intervals=FitInterval[])
    v_u1 = FittedVariable("u1"; bounds=(-100.0, 100.0), init=5.0, time_varying=true)
    v_fD = FittedVariable("f_D"; bounds=(0.0, 1.0), init=0.7)
    return ParameterSpec(expt; variables=[v_u1, v_fD], intervals=intervals,
        dependents=["f_C" => "1 - f_D", "dep1" => "2*u1 - 1"],
        bindings=["u1" => "ions.D.u_kms", "f_D" => "ions.D.fraction"])
end

function _rg_working(expt::Experiment, pspec::ParameterSpec)
    return Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)
end

function _rg_data(working::Experiment, pspec::ParameterSpec, theta::AbstractVector)
    truth = unflatten(working, pspec, theta)
    image = measure(truth)
    sig = sigma(GaussianNoise(a=1.0, b=0.02), image)
    return StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=sig)
end

# A FitProblem carrying a three-term Regularizer: an absolute D2 term on "u1", a relative D2
# term on "u1" with a nonzero reference (RG13, so the mode is not degenerate), and a D2 term on
# the dependent "dep1" (RG19, the chain-rule-through-the-expression case). RG11, RG12, RG13 and
# RG19 all share this fixture; each reads only the term(s) its own row names.
function _rg_problem_with_reg()
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    working = _rg_working(expt, pspec)
    theta0 = flatten(pspec)
    data = _rg_data(working, pspec, theta0)

    term_abs = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:absolute,
        scale=50.0, weight=3.0)
    term_rel = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:relative,
        reference=2.0, weight=1.0)
    term_dep = RegularizerTerm(pspec; select=["dep1"], operator=:D2, normalization=:unit,
        weight=1.0)
    reg = Regularizer([term_abs, term_rel, term_dep]; weight=0.25)

    problem = FitProblem(working, pspec, data; regularizer=reg)
    theta = theta0 .+ 0.05 .* (1:length(theta0))
    return problem, theta, reg
end

# ------------------------------------------------------------------------------------------
# RG1 to RG5: the operators themselves
# ------------------------------------------------------------------------------------------

@testset "RG1: D2 of a quadratic is exact on the nonuniform axis" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    xvalues = [3.5 * t^2 - 2t + 7 for t in _RG_T_NS]
    theta = flatten(pspec, ["u1" => xvalues])

    term = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit)
    reg = Regularizer([term])
    raw = _regularizer_raw_rows(reg, pspec, theta)

    @test length(raw) == length(_RG_T_NS) - 2
    worst = maximum(abs.(raw .- 7.0) ./ 7.0)
    @info "RG1 worst relative error against 2*3.5" worst
    @test worst <= 1e-12
end

@testset "RG2: D2 of a linear trace is zero" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    xvalues = [4t + 1 for t in _RG_T_NS]
    theta = flatten(pspec, ["u1" => xvalues])

    term = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit)
    reg = Regularizer([term])
    raw = _regularizer_raw_rows(reg, pspec, theta)

    scale = maximum(abs, xvalues)
    worst = maximum(abs, raw)
    @info "RG2 worst absolute D2 of a linear trace, against 1e-12*scale" worst bound = 1e-12 * scale
    @test worst <= 1e-12 * scale
end

@testset "RG3: D1 of a linear trace is its slope" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    xvalues = [4t + 1 for t in _RG_T_NS]
    theta = flatten(pspec, ["u1" => xvalues])

    term = RegularizerTerm(pspec; select=["u1"], operator=:D1, normalization=:unit)
    reg = Regularizer([term])
    raw = _regularizer_raw_rows(reg, pspec, theta)

    @test length(raw) == length(_RG_T_NS) - 1
    worst = maximum(abs.(raw .- 4.0) ./ 4.0)
    @info "RG3 worst relative error against the slope 4.0" worst
    @test worst <= 1e-13
end

@testset "RG4: I returns the trace" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    xvalues = [3.5 * t^2 - 2t + 7 for t in _RG_T_NS]
    theta = flatten(pspec, ["u1" => xvalues])

    term = RegularizerTerm(pspec; select=["u1"], operator=:I, normalization=:unit)
    reg = Regularizer([term])
    raw = _regularizer_raw_rows(reg, pspec, theta)

    @test length(raw) == length(_RG_T_NS)
    # not bitwise === xvalues: theta round-trips xvalues through the logit transform
    # (_to_unconstrained then _from_unconstrained), which is not bit-exact. 1e-13 relative is
    # what that round trip actually achieves; flagged in the chunk 7 report as a deviation from
    # the spec's literal "===" against this test's own methodology, not against the "I" operator
    # itself (_operator_rows(:I, u, t) is exactly collect(u), an identity with no roundoff of
    # its own).
    worst = maximum(abs.(raw .- xvalues) ./ abs.(xvalues))
    @info "RG4 worst relative error against xvalues (the logit round trip)" worst
    @test isapprox(raw, xvalues; rtol=1e-13)
end

@testset "RG5: row counts, whole axis and an interval" begin
    expt = _rg_expt()
    N = length(_RG_T_NS)
    pre = FitInterval("pre", expt.trace.axis; t_range_ns=[0.0, 1.5])   # bins 1:5
    pspec = _rg_pspec(expt; intervals=[pre])

    for (op, n) in ((:D2, N - 2), (:D1, N - 1), (:I, N))
        term = RegularizerTerm(pspec; select=["u1"], operator=op, normalization=:unit)
        @test length(term.rows) == n
    end
    for (op, n) in ((:D2, 3), (:D1, 4), (:I, 5))
        term = RegularizerTerm(pspec; select=["u1"], operator=op, normalization=:unit,
            interval="pre")
        @test length(term.rows) == n
    end
end

# ------------------------------------------------------------------------------------------
# RG6: an interval term has no straddling row
# ------------------------------------------------------------------------------------------

@testset "RG6: an interval term has no straddling row" begin
    expt = _rg_expt()
    pre = FitInterval("pre", expt.trace.axis; t_range_ns=[0.0, 1.5])    # bins 1:5
    post = FitInterval("post", expt.trace.axis; t_range_ns=[1.9, 3.6]) # bins 6:10
    pspec = _rg_pspec(expt; intervals=[pre, post])

    # the step (5.0) must stay well inside u1's (-100, 100) bounds; the curvature (1e-5) is six
    # orders below it, matching the row's own "dominates by six orders" description.
    xvalues = Vector{Float64}(undef, length(_RG_T_NS))
    for j in 1:5
        xvalues[j] = 1e-5 * _RG_T_NS[j]^2
    end
    for j in 6:10
        xvalues[j] = 5.0 + 1e-5 * (_RG_T_NS[j] - _RG_T_NS[6])^2
    end
    theta = flatten(pspec, ["u1" => xvalues])

    term_whole = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit)
    R_whole = _regularizer_value(Regularizer([term_whole]), pspec, theta)

    term_pre = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit,
        interval="pre")
    term_post = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit,
        interval="post")
    R_split = _regularizer_value(Regularizer([term_pre, term_post]; weight=1.0), pspec, theta)

    @info "RG6 R_split vs R_whole" R_split R_whole ratio = R_split / R_whole
    @test R_split < 1e-6 * R_whole
end

# ------------------------------------------------------------------------------------------
# RG7: the four normalizations, and relative's degeneracy at the default reference
# ------------------------------------------------------------------------------------------

@testset "RG7: the four normalizations, and the anchor-bin rule" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    # a genuine quadratic: linear would make D2 ~ 0, turning a relative-error comparison into a
    # measurement of round-off (conventions.md 8.4's own floor rationale). Never zero, so
    # :relative's default reference is well defined too.
    xvalues = [10.0 + t + 2t^2 for t in _RG_T_NS]
    theta = flatten(pspec, ["u1" => xvalues])
    t = Float64.(_RG_T_NS)
    _relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-13 * maximum(abs, b)))

    # absolute, explicit scale -- a constant divisor commutes exactly with a linear operator, so
    # this number is expected to be the same whether the divisor is applied before or after D2
    # (both the withdrawn design and the current one agree here; not a special case in the code).
    term_abs = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:absolute,
        scale=20.0)
    raw_abs = _regularizer_raw_rows(Regularizer([term_abs]), pspec, theta)
    expected_abs = _operator_rows(:D2, xvalues ./ 20.0, t)
    @info "RG7 absolute (explicit scale) worst relative error, normalize-after vs. normalize-before" worst = _relerr(raw_abs, expected_abs)
    @test _relerr(raw_abs, expected_abs) <= 1e-13

    # absolute, default scale (the bound width, 200.0 for u1's (-100, 100))
    term_def = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:absolute)
    raw_def = _regularizer_raw_rows(Regularizer([term_def]), pspec, theta)
    expected_def = _operator_rows(:D2, xvalues ./ 200.0, t)
    @info "RG7 absolute (default scale) worst relative error, normalize-after vs. normalize-before" worst = _relerr(raw_def, expected_def)
    @test _relerr(raw_def, expected_def) <= 1e-13

    # initial (u1's own init is 5.0, constant over the whole axis -- a flat divisor, same
    # commuting argument as absolute's)
    term_init = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:initial)
    raw_init = _regularizer_raw_rows(Regularizer([term_init]), pspec, theta)
    expected_init = _operator_rows(:D2, xvalues ./ 5.0, t)
    @info "RG7 initial (flat u0) worst relative error, normalize-after vs. normalize-before" worst = _relerr(raw_init, expected_init)
    @test _relerr(raw_init, expected_init) <= 1e-13

    # unit
    term_unit = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit)
    raw_unit = _regularizer_raw_rows(Regularizer([term_unit]), pspec, theta)
    expected_unit = _operator_rows(:D2, xvalues, t)
    @info "RG7 unit worst relative error, normalize-after vs. normalize-before" worst = _relerr(raw_unit, expected_unit)
    @test _relerr(raw_unit, expected_unit) <= 1e-13

    # relative, default (zero) reference: normalization is applied AFTER the operator now (D24/D25
    # revised 2026-09-13), at each row's own anchor bin a(r) -- D2's centred-stencil middle
    # (r + 1), D1's and I's own base bin (r). N_r = x_{a(r)}, the trace's OWN current value, so
    # with ref = 0 a zero row here would mean the divisor is still being applied to the trace
    # before the operator (the old, withdrawn design), not this one -- a zero must NOT appear.
    # Each expected row is written out by hand from _operator_rows and the anchor rule, not from
    # any function under test.
    for (op, anchor) in ((:D2, r -> r + 1), (:D1, r -> r), (:I, r -> r))
        term_rel = RegularizerTerm(pspec; select=["u1"], operator=op, normalization=:relative)
        raw_rel = _regularizer_raw_rows(Regularizer([term_rel]), pspec, theta)
        g = _operator_rows(op, xvalues, t)
        expected_rel = [g[r] / xvalues[anchor(r)] for r in eachindex(g)]
        @info "RG7 relative/$op worst relative error against the hand-written anchor formula" worst = _relerr(raw_rel, expected_rel)
        @test _relerr(raw_rel, expected_rel) <= 1e-13
    end
end

# ------------------------------------------------------------------------------------------
# RG8: absolute on a dependent with no scale
# ------------------------------------------------------------------------------------------

@testset "RG8: absolute on a dependent with no scale" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    @test_throws "selects the dependent \"dep1\", so `scale` must be given explicitly" RegularizerTerm(
        pspec; select=["dep1"], operator=:D2, normalization=:absolute)
end

# ------------------------------------------------------------------------------------------
# RG9: the reference
# ------------------------------------------------------------------------------------------

@testset "RG9: the reference" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    xvalues = [3.5 * t^2 - 2t + 7 for t in _RG_T_NS]
    theta = flatten(pspec, ["u1" => xvalues])

    # an I term with the trace itself as its reference gives R == 0 (up to the logit round trip
    # theta puts xvalues through, RG4's same deviation from a literal "==0")
    term_self = RegularizerTerm(pspec; select=["u1"], operator=:I, normalization=:unit,
        reference=xvalues)
    R_self = _regularizer_value(Regularizer([term_self]), pspec, theta)
    @info "RG9 R_self (an I term referenced against its own trace)" R_self
    @test R_self < 1e-20

    # a constant reference shifts the I rows by exactly that constant
    term_shift = RegularizerTerm(pspec; select=["u1"], operator=:I, normalization=:unit,
        reference=5.0)
    raw_shift = _regularizer_raw_rows(Regularizer([term_shift]), pspec, theta)
    worst_shift = maximum(abs.(raw_shift .- (xvalues .- 5.0)))
    @info "RG9 worst absolute error, the constant-shift row" worst_shift bound = 1e-14 * maximum(abs, xvalues)
    @test worst_shift <= 1e-14 * maximum(abs, xvalues)

    # D1 and D2 are unchanged by a constant reference
    for op in (:D1, :D2)
        term_ref0 = RegularizerTerm(pspec; select=["u1"], operator=op, normalization=:unit,
            reference=0.0)
        term_ref5 = RegularizerTerm(pspec; select=["u1"], operator=op, normalization=:unit,
            reference=5.0)
        raw0 = _regularizer_raw_rows(Regularizer([term_ref0]), pspec, theta)
        raw5 = _regularizer_raw_rows(Regularizer([term_ref5]), pspec, theta)
        @test isapprox(raw0, raw5; rtol=1e-14)
    end
end

# ------------------------------------------------------------------------------------------
# RG10: a reference naming a fitted variable is refused
# ------------------------------------------------------------------------------------------

@testset "RG10: a reference naming a fitted variable is refused" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    @test_throws "is not a fitted variable, a dependent" RegularizerTerm(
        pspec; select=["u1"], operator=:I, normalization=:unit, reference="u1")
end

# ------------------------------------------------------------------------------------------
# RG11: R against the closed-form sum, and the pin that R excludes Lambda
# ------------------------------------------------------------------------------------------

@testset "RG11: R against the closed-form sum, R excludes Lambda" begin
    problem, theta, reg = _rg_problem_with_reg()
    pspec = problem.pspec

    R = _regularizer_value(reg, pspec, theta)

    raw = _regularizer_raw_rows(reg, pspec, theta)
    R_by_hand = 0.0
    for term in reg.terms
        R_by_hand += term.weight * sum(abs2, view(raw, term.rows))
    end
    relerr_R = abs(R - R_by_hand) / abs(R_by_hand)
    @info "RG11 relative error, R against the closed-form sum" relerr_R
    @test isapprox(R, R_by_hand; rtol=1e-13)

    # "𝒪 - chi2" read as sum(abs2, _full_residuals) - chi2(problem, theta): both terms are
    # ~2e10 here (a deliberately large perturbation, shared with RG12/13/19's fixture), so
    # computing their difference from the two full sums loses precision at chi2's own O(eps)
    # (~1e-6 absolute) and swamps R's ~90 -- a cancellation artifact of that subtraction, not of
    # the regularizer. Slicing r_full's own regularizer rows before squaring avoids it and is a
    # more direct test of the same claim (D25's residual-row form and "R excludes Lambda").
    r_full = _full_residuals(problem, theta)
    np = problem.npixels
    obj_minus_chi2 = sum(abs2, view(r_full, (np+1):length(r_full)))
    relerr_lambda = abs(obj_minus_chi2 - reg.weight * R) / abs(reg.weight * R)
    @info "RG11 relative error, (O - chi2) against Lambda*R" relerr_lambda
    @test isapprox(obj_minus_chi2, reg.weight * R; rtol=1e-13)
end

# ------------------------------------------------------------------------------------------
# RG12, RG13, RG19: the residual rows' closed-form Jacobian against brute-force ForwardDiff
# ------------------------------------------------------------------------------------------

@testset "RG12, RG13, RG19: the regularizer rows' Jacobian against brute force" begin
    problem, theta, reg = _rg_problem_with_reg()
    pspec = problem.pspec
    ntheta = pspec.ntheta
    np = problem.npixels

    J_dense_bruteforce = ForwardDiff.jacobian(th -> _full_residuals(problem, th), theta)
    J_reg_ad = J_dense_bruteforce[(np+1):end, :]

    raw_all, rows, cols, coeffs = _regularizer_closed_form(reg, pspec, theta)
    J_reg_cf = zeros(reg.nrows, ntheta)
    for (r, c, v) in zip(rows, cols, coeffs)
        J_reg_cf[r, c] += v
    end

    function _worst_relerr(J_ad, J_cf, rowrange, nθ)
        worst = 0.0
        for k in 1:nθ
            colref = @view J_ad[rowrange, k]
            colcf = @view J_cf[rowrange, k]
            floor_k = 1e-10 * maximum(abs, colref; init=0.0)
            relerr = maximum(abs.(colcf .- colref) ./ max.(abs.(colref), floor_k, 1e-300))
            worst = max(worst, relerr)
        end
        return worst
    end

    worst_all = _worst_relerr(J_reg_ad, J_reg_cf, 1:reg.nrows, ntheta)
    @info "RG12 worst relative error, all regularizer rows" worst_all
    @test worst_all <= 1e-10

    term_abs, term_rel, term_dep = reg.terms
    worst_rel = _worst_relerr(J_reg_ad, J_reg_cf, term_rel.rows, ntheta)
    @info "RG13 worst relative error, the nonzero-reference relative term" worst_rel
    @test worst_rel <= 1e-10

    worst_dep = _worst_relerr(J_reg_ad, J_reg_cf, term_dep.rows, ntheta)
    @info "RG19 worst relative error, the dependent-selecting term" worst_dep
    @test worst_dep <= 1e-10

    # RG13's other half: relative at the DEFAULT (zero) reference. Under the withdrawn design
    # (normalize-before-operator) this Jacobian was trivially zero everywhere -- relative was
    # degenerate there, D25's own finding. Under the current one (normalize-after-operator,
    # D24/D25 revised 2026-09-13) it is not, and this is exactly the row that fails if the
    # quotient-rule term (`-g_r / N_r^2`, fired only on the anchor column) is dropped: every
    # off-anchor column stays right even without it, so only this oracle comparison catches it.
    term_rel0 = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:relative)
    reg0 = Regularizer([term_rel0])
    problem0 = FitProblem(problem.experiment, pspec, problem.data; regularizer=reg0)
    np0 = problem0.npixels

    J_dense0 = ForwardDiff.jacobian(th -> _full_residuals(problem0, th), theta)
    J_reg_ad0 = J_dense0[(np0+1):end, :]

    raw_all0, rows0, cols0, coeffs0 = _regularizer_closed_form(reg0, pspec, theta)
    J_reg_cf0 = zeros(reg0.nrows, ntheta)
    for (r, c, v) in zip(rows0, cols0, coeffs0)
        J_reg_cf0[r, c] += v
    end

    worst_rel0 = _worst_relerr(J_reg_ad0, J_reg_cf0, 1:reg0.nrows, ntheta)
    @info "RG13 worst relative error, the DEFAULT (zero-reference) relative term" worst_rel0
    @test worst_rel0 <= 1e-10
end

# ------------------------------------------------------------------------------------------
# RG14: the regularizer's own gradient against finite differences
# ------------------------------------------------------------------------------------------

@testset "RG14: grad R against finite differences, all four normalizations" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    theta0 = flatten(pspec)
    theta = theta0 .+ 0.03 .* (1:length(theta0))

    for (mode, scale, ref) in ((:absolute, 40.0, 0.0), (:initial, 0.0, 0.0),
                                (:relative, 0.0, 2.0), (:unit, 0.0, 0.0))
        term = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=mode,
            scale=scale, reference=ref)
        reg = Regularizer([term])

        g = zeros(length(theta))
        _regularizer_gradient!(g, reg, pspec, theta)

        h = 1e-6
        g_fd = similar(theta)
        for i in eachindex(theta)
            tp = copy(theta); tp[i] += h
            tm = copy(theta); tm[i] -= h
            g_fd[i] = (_regularizer_value(reg, pspec, tp) - _regularizer_value(reg, pspec, tm)) / (2h)
        end

        floorv = 1e-6 * maximum(abs, g_fd)
        relerr = maximum(abs.(g .- g_fd) ./ max.(abs.(g_fd), floorv, 1e-300))
        @info "RG14 worst relative error" mode relerr
        @test relerr <= 1e-6
    end
end

# ------------------------------------------------------------------------------------------
# RG15: rebinning changes R per row by less than D2's own discretization error
# ------------------------------------------------------------------------------------------

@testset "RG15: R per row is stable under rebinning" begin
    t40 = collect(range(0.0, 4.0; length=40))
    t20 = t40[1:2:end]

    pspec40 = _rg_pspec(_rg_expt(t40))
    pspec20 = _rg_pspec(_rg_expt(t20))
    theta40 = flatten(pspec40, ["u1" => sin.(2 .* t40)])
    theta20 = flatten(pspec20, ["u1" => sin.(2 .* t20)])

    term40 = RegularizerTerm(pspec40; select=["u1"], operator=:D2, normalization=:unit)
    term20 = RegularizerTerm(pspec20; select=["u1"], operator=:D2, normalization=:unit)
    R40 = _regularizer_value(Regularizer([term40]), pspec40, theta40)
    R20 = _regularizer_value(Regularizer([term20]), pspec20, theta20)

    per_row_40 = R40 / length(term40.rows)
    per_row_20 = R20 / length(term20.rows)
    @info "RG15 R per row, 40 bins vs 20 bins" per_row_40 per_row_20
    @test isapprox(per_row_40, per_row_20; rtol=0.15)
end

# ------------------------------------------------------------------------------------------
# RG16: two terms selecting the same variable add
# ------------------------------------------------------------------------------------------

@testset "RG16: terms add" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    theta0 = flatten(pspec)
    theta = theta0 .+ 0.02 .* (1:length(theta0))

    term_a = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit, weight=1.0)
    term_b = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit, weight=2.0)
    R_ab = _regularizer_value(Regularizer([term_a, term_b]), pspec, theta)

    term_c = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit, weight=3.0)
    R_c = _regularizer_value(Regularizer([term_c]), pspec, theta)

    relerr = abs(R_ab - R_c) / abs(R_c)
    @info "RG16 relative error, two weight-1/2 terms against one weight-3 term" relerr
    @test isapprox(R_ab, R_c; rtol=1e-13)
end

# ------------------------------------------------------------------------------------------
# RG17: the "time_varying" group
# ------------------------------------------------------------------------------------------

@testset "RG17: the time_varying group selects every time-varying fitted variable" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    term = RegularizerTerm(pspec; select="time_varying", operator=:I, normalization=:unit)
    @test term.select == ["u1"]
end

# ------------------------------------------------------------------------------------------
# RG18: an unknown selector
# ------------------------------------------------------------------------------------------

@testset "RG18: an unknown selector" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    @test_throws "is not a fitted variable, a dependent, or \"time_varying\"; valid names are" RegularizerTerm(
        pspec; select=["nosuchname"], operator=:I, normalization=:unit)
end

# ------------------------------------------------------------------------------------------
# RG20: an interval too short for the operator
# ------------------------------------------------------------------------------------------

@testset "RG20: an interval too short for the operator" begin
    expt = _rg_expt()
    short = FitInterval("short", expt.trace.axis; t_range_ns=[0.0, 0.31])   # bins 1:2
    pspec = _rg_pspec(expt; intervals=[short])
    @test_throws "has 2 bins, but operator :D2 needs at least 3" RegularizerTerm(
        pspec; select=["u1"], operator=:D2, normalization=:unit, interval="short")
end

# ------------------------------------------------------------------------------------------
# RG21: show
# ------------------------------------------------------------------------------------------

@testset "RG21: show" begin
    expt = _rg_expt()
    pspec = _rg_pspec(expt)
    term = RegularizerTerm(pspec; select=["u1"], operator=:D2, normalization=:unit, weight=2.0)
    reg = Regularizer([term]; weight=0.25)

    s = sprint(show, reg)
    @test occursin("1 terms", s)
    @test occursin("0.25", s)

    ml = sprint((io, x) -> show(io, MIME("text/plain"), x), reg)
    @test occursin("u1", ml)
end

# ------------------------------------------------------------------------------------------
# RG22: initial on a dependent whose starting trace varies with t
# ------------------------------------------------------------------------------------------

@testset "RG22: initial on a dependent whose starting trace varies with t" begin
    # "decay" is a function of t alone (no fitted variable), so its value at any theta equals its
    # value at flatten(pspec)'s start vector -- unlike u1's own `init`, which is a scalar and so
    # flat across bins (RG7's "initial selecting a plain fitted variable"), decay's starting trace
    # u0_j varies with t. This is the case D24/D25's revision (2026-09-13) treats differently from
    # the withdrawn design: dividing by u0 before the operator does not commute with dividing by
    # u0_{a(r)} after it when u0 itself is not flat.
    expt = _rg_expt()
    v_u1 = FittedVariable("u1"; bounds=(-100.0, 100.0), init=5.0, time_varying=true)
    pspec = ParameterSpec(expt; variables=[v_u1],
        dependents=["decay" => "5.0 + 2.0 * exp(-t / 2.0)"],
        bindings=["u1" => "ions.D.u_kms"])
    theta0 = flatten(pspec)
    theta = theta0 .+ 0.05 .* (1:length(theta0))
    t = Float64.(_RG_T_NS)
    u0 = [5.0 + 2.0 * exp(-tj / 2.0) for tj in t]

    for (op, anchor) in ((:D2, r -> r + 1), (:D1, r -> r))
        term = RegularizerTerm(pspec; select=["decay"], operator=op, normalization=:initial)
        raw = _regularizer_raw_rows(Regularizer([term]), pspec, theta)

        # decay does not depend on theta, so its own trace equals u0 at every theta, including the
        # perturbed one above; expected rows are (op(u0 - 0))_r / u0_{a(r)}, written out by hand
        # from _operator_rows and u0's own per-bin values -- not op((u0 - ref)/u0)_r, the withdrawn
        # design, and not computed via _term_raw/_regularizer_raw_rows, the code under test.
        g = _operator_rows(op, u0, t)
        expected = [g[r] / u0[anchor(r)] for r in eachindex(g)]
        worst = maximum(abs.(raw .- expected) ./ max.(abs.(expected), 1e-13 * maximum(abs, expected)))
        @info "RG22 worst relative error, initial on a t-varying dependent" op worst
        @test worst <= 1e-13
    end
end
