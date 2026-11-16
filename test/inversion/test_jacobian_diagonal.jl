# Step G.4.1: the Gauss-Newton diagonal's regularizer rows (Phase G.4 spec D5 to D7; OQ-71 item 3). Gate
# "gauss-newton diagonal": GD1 to GD5. diag(JᵀJ) is Σ_rows (∂ρ_row/∂θ_col)², and a row's entry on a column is the sum
# of its closed-form triples there; a :D2 or :D1 row on a scalar fitted variable, or on a dependent that reads one, has
# one triple per bin of its stencil on the same θ index. The oracle is CV2's (test_optimizers_diagnostics.jl): the dense
# diagonal from jvp! on unit vectors, which adds the triples linearly and shares no code with the diagonal
# (conventions.md 8.2).

using Test
using TRUST
using TRUST: jvp!, jacobian, _gauss_newton_diagonal, _add_regularizer_diagonal!, _regularizer_closed_form

# ------------------------------------------------------------------------------------------
# the fixture (spec D7): P1's plasma (test_optimizers_diagnostics.jl) on five uneven bins and twelve wavelengths, no
# instrument function, so that the forward model compiles in seconds
# ------------------------------------------------------------------------------------------

const _GD_T_NS = [0.0, 0.4, 1.0, 1.5, 2.2]

function _gd_template()
    return Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
end

function _gd_expt(lambda_nm; scale=1.0e-6)
    trace = PlasmaTrace(_gd_template(); t_ns=_GD_T_NS)
    return Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact), lambda_m=lambda_nm .* 1e-9,
        effects=(Sensitivity(scale=scale), ConstantBackground(level=25.0)))
end

# T_D per bin; u_D and s scalar; the dependent T_e reads T_D and the scalar s.
_gd_variables() = [FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true),
                   FittedVariable("u_D"; bounds=(-200.0, 200.0), init=30.0),
                   FittedVariable("s"; bounds=(0.5, 2.0), init=1.2)]
const _GD_DEPENDENTS = ["T_e" => "2 * T_D * s"]
const _GD_BINDINGS = ["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms", "T_e" => "electrons.e.T_eV"]

# Every term that repeats a row and a column (spec D5): :D2 and :D1 on a scalar under each normalization, and :D2 on
# a dependent that reads a scalar; then two that do not, :D2 on a variable per bin and :I on a scalar.
function _gd_repeating_terms(pspec)
    return [RegularizerTerm(pspec; select="u_D", operator=:D2, normalization=:unit),
            RegularizerTerm(pspec; select="u_D", operator=:D1, normalization=:relative),
            RegularizerTerm(pspec; select="s", operator=:D2, normalization=:absolute, scale=0.5),
            RegularizerTerm(pspec; select="s", operator=:D1, normalization=:initial),
            RegularizerTerm(pspec; select="T_e", operator=:D2, normalization=:unit)]
end
_gd_plain_terms(pspec) = [RegularizerTerm(pspec; select="T_D", operator=:D2, normalization=:relative),
                          RegularizerTerm(pspec; select="u_D", operator=:I, normalization=:unit)]

function _gd_data(working, pspec, theta)
    image = measure(unflatten(working, pspec, theta))
    return StreakData(lambda_m=working.lambda, t_s=working.t, image=image, sigma=0.05 .* image .+ 1.0)
end

_gd_working(expt, pspec) = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
    lambda_m=expt.lambda, effects=expt.effects, prefactor=expt.prefactor)

const _GD_LAMBDA = collect(range(262.0, 264.5; length=12))

# The data at the truth, the diagonal taken away from it (spec D7): theta at the truth would make every :relative
# divisor its truth and every :D2 row of a constant zero, which hides nothing, but away from it every row is live.
function _gd_problem(terms_of; weight=1.0e-2)
    expt = _gd_expt(_GD_LAMBDA)
    pspec = ParameterSpec(expt; variables=_gd_variables(), dependents=_GD_DEPENDENTS, bindings=_GD_BINDINGS)
    working = _gd_working(expt, pspec)
    truth = flatten(pspec, ["T_D" => [180.0, 220.0, 260.0, 240.0, 300.0], "u_D" => 25.0, "s" => 1.1])
    reg = Regularizer(terms_of(pspec); weight=weight)
    problem = FitProblem(working, pspec, _gd_data(working, pspec, truth); regularizer=reg)
    theta = flatten(pspec, ["T_D" => [210.0, 190.0, 300.0, 230.0, 280.0], "u_D" => -40.0, "s" => 1.4])
    return problem, theta
end

function _gd_joint(terms_of; weight=1.0e-2)
    e1 = _gd_expt(_GD_LAMBDA)
    e2 = Experiment(trace=e1.trace, probe=e1.probe, collection=e1.collection,
        lambda_m=collect(range(262.6, 263.9; length=10)) .* 1e-9,
        effects=(Sensitivity(scale=2.0e-6), ConstantBackground(level=10.0)))
    pspec = ParameterSpec((e1, e2); variables=_gd_variables(), dependents=_GD_DEPENDENTS, bindings=_GD_BINDINGS)
    w1, w2 = _gd_working(e1, pspec), _gd_working(e2, pspec)
    truth = flatten(pspec, ["T_D" => [180.0, 220.0, 260.0, 240.0, 300.0], "u_D" => 25.0, "s" => 1.1])
    reg = Regularizer(terms_of(pspec); weight=weight)
    ch1 = FitProblem(w1, pspec, _gd_data(w1, pspec, truth); channel=1, name="one")
    ch2 = FitProblem(w2, pspec, _gd_data(w2, pspec, truth); channel=2, name="two")
    jp = JointFitProblem((ch1, ch2); weights=[1.0, 0.5], regularizer=reg)
    theta = flatten(pspec, ["T_D" => [210.0, 190.0, 300.0, 230.0, 280.0], "u_D" => -40.0, "s" => 1.4])
    return jp, theta
end

# CV2's oracle: column k of the dense J_full (data and regularizer rows) by jvp! on e_k, its squared norm.
function _gd_oracle(problem, Jac, nrows)
    n = problem.pspec.ntheta
    out = zeros(problem.npixels + nrows)
    oracle = zeros(n)
    for k in 1:n
        v = zeros(n)
        v[k] = 1.0
        jvp!(out, problem, Jac, v)
        oracle[k] = sum(abs2, out)
    end
    return oracle
end

# The (row, column) pairs on which a regularizer has two or more nonzero triples.
function _gd_repeats(rows, cols, coeffs)
    seen = Dict{Tuple{Int,Int},Int}()
    for m in eachindex(rows)
        coeffs[m] == 0 && continue
        seen[(rows[m], cols[m])] = get(seen, (rows[m], cols[m]), 0) + 1
    end
    return [k for (k, n) in seen if n >= 2]
end

# The same Jacobian with its regularizer triples emptied: the data rows' share of the diagonal.
_gd_data_only(J) = TRUST.FitJacobian{Float64}(J.blocks, J.kernel_globals, J.chain_globals, J.kernel_global_idx,
    J.chain_global_idx, J.chain_blocks, J.chain_block_idx, J.response, J.Klambda, J.Kt, Int[], Int[], Float64[])

# Each triple's square added by itself, the formula until Phase G.4 (spec D6), for the fixture's own check that it
# discriminates.
function _gd_squares(d0, rows, cols, coeffs)
    d = copy(d0)
    for m in eachindex(rows)
        d[cols[m]] += Float64(coeffs[m])^2
    end
    return d
end

# ------------------------------------------------------------------------------------------
# GD1: the fixture repeats a row and a column in every term that should, and in no other
# ------------------------------------------------------------------------------------------

@testset "GD1: which terms repeat a row and a column" begin
    problem, theta = _gd_problem(p -> vcat(_gd_repeating_terms(p), _gd_plain_terms(p)))
    reg = problem.regularizer
    _, rows, cols, coeffs = _regularizer_closed_form(reg, problem.pspec, theta)
    rep = _gd_repeats(rows, cols, coeffs)
    for (i, term) in enumerate(reg.terms)
        n = count(rc -> rc[1] in term.rows, rep)
        @test (n > 0) == (i <= 5)
    end
    @test issorted(rows)    # spec D6: a row's triples together, the rows in order
end

# ------------------------------------------------------------------------------------------
# GD2, GD3: the diagonal against the dense one, single-channel and joint
# ------------------------------------------------------------------------------------------

@testset "GD2: FitProblem, the diagonal against jvp! on unit vectors" begin
    problem, theta = _gd_problem(p -> vcat(_gd_repeating_terms(p), _gd_plain_terms(p)))
    Jac = jacobian(problem, theta)
    oracle = _gd_oracle(problem, Jac, problem.regularizer.nrows)
    @test maximum(oracle) > 0
    d = _gauss_newton_diagonal(problem, Jac)
    floor_d = 1e-12 * maximum(oracle)
    for k in eachindex(d)
        @test isapprox(d[k], oracle[k]; rtol=1e-12, atol=floor_d)
    end
    # the fixture discriminates: each triple's square by itself misses the oracle where a row repeats a column
    old = _gd_squares(_gauss_newton_diagonal(problem, _gd_data_only(Jac)), Jac.reg_rows, Jac.reg_cols, Jac.reg_coeffs)
    @test maximum(abs.(old .- oracle) ./ oracle) > 1e-3
end

@testset "GD3: JointFitProblem, the diagonal against jvp! on unit vectors" begin
    jp, theta = _gd_joint(p -> vcat(_gd_repeating_terms(p), _gd_plain_terms(p)))
    J = jacobian(jp, theta)
    oracle = _gd_oracle(jp, J, jp.regularizer.nrows)
    @test maximum(oracle) > 0
    d = _gauss_newton_diagonal(jp, J)
    floor_d = 1e-12 * maximum(oracle)
    for k in eachindex(d)
        @test isapprox(d[k], oracle[k]; rtol=1e-12, atol=floor_d)
    end
end

# ------------------------------------------------------------------------------------------
# GD4: where no row repeats a column, the diagonal is bitwise each triple's square added in turn
# ------------------------------------------------------------------------------------------

@testset "GD4: no repeat, bitwise the triple-by-triple sum" begin
    problem, theta = _gd_problem(_gd_plain_terms)
    Jac = jacobian(problem, theta)
    @test isempty(_gd_repeats(Jac.reg_rows, Jac.reg_cols, Jac.reg_coeffs))
    @test !isempty(Jac.reg_rows)
    d = _gauss_newton_diagonal(problem, Jac)
    d0 = _gauss_newton_diagonal(problem, _gd_data_only(Jac))
    @test d == _gd_squares(d0, Jac.reg_rows, Jac.reg_cols, Jac.reg_coeffs)
end

# ------------------------------------------------------------------------------------------
# GD5: the summing step on hand-written triples, and its refusal
# ------------------------------------------------------------------------------------------

@testset "GD5: _add_regularizer_diagonal! by hand" begin
    d = [1.0, 0.0, 0.0, 0.0]
    _add_regularizer_diagonal!(d, [1, 1, 1, 2, 2, 3], [3, 3, 4, 3, 2, 3], [1.0, 2.0, 5.0, 7.0, -0.5, 0.25])
    @test d == [1.0, 0.25, 9.0 + 49.0 + 0.0625, 25.0]
    # two triples that cancel give nothing; the old formula gave 2c²
    d = zeros(2)
    _add_regularizer_diagonal!(d, [4, 4], [2, 2], [3.0, -3.0])
    @test d == [0.0, 0.0]
    d = [2.0]
    @test _add_regularizer_diagonal!(d, Int[], Int[], Float64[]) === d && d == [2.0]
    @test_throws "row 1 after row 2" _add_regularizer_diagonal!(zeros(3), [1, 2, 1], [1, 2, 3], [1.0, 1.0, 1.0])
end
