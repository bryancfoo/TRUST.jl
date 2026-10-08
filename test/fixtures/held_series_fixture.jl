# The held series' fixture (Phase 4.I spec D13), shared by test/inversion/test_held_series.jl (HD1,
# HD2, HD5, HD6) and its slow namesake under test/slow/ (HD3, HD4): test_jacobian.jl's P2 chain (ten
# bins, SpectralIRF, TemporalIRF, Sensitivity), so that the brute-force Jacobian compiles nothing new.
# T_D is held on the interval "mid" (bins 3 to 8 of ten, 0.6 to 2.4 ns) at _HD_HELD, u_D fitted on the
# whole axis, and the sensitivity scale held at 2e-6 (its init 1e-6): a held scalar bound to a
# chain-stage parameter, which no Jacobian took before Phase 4.I (spec D11). The regularizer selects
# u_D, T_D (constant rows, no column) and the dependent Tu = T_D * u_D inside the interval, whose
# closed-form row derivative carries T_D's held value.

const _HD_HELD = [150.0, 180.0, 220.0, 260.0, 300.0, 330.0]

function _hd_expt()
    template = Plasma(; n_e_cm3=3e19, electrons=[electrons(T_eV=400.0)],
        ions=[ion(name="D", mass_amu=2.014, charge=1, T_eV=200.0, u_kms=0.0)])
    trace = PlasmaTrace(template; t_ns=collect(range(0.0, 3.0; length=10)))
    return Experiment(trace=trace, probe=ProbeBeam(lambda_probe_nm=263.25),
        collection=Collection(theta_deg=60.0, k_model=:exact),
        lambda_m=collect(range(262.0, 264.5; length=32)) .* 1e-9,
        effects=(SpectralIRF(fwhm_nm=0.08), TemporalIRF(fwhm_ns=0.4), Sensitivity(scale=1.0e-6)))
end

function _hd_problem(; T_held=_HD_HELD, T_interval=1)
    expt = _hd_expt()
    mid = FitInterval("mid", expt.trace.axis; t_range_ns=[0.6, 2.4])
    si = findfirst(e -> e isa Sensitivity, expt.effects)
    vars = [FittedVariable("T_D"; bounds=(50.0, 800.0), init=200.0, time_varying=true, interval=T_interval, held=T_held),
        FittedVariable("u_D"; bounds=(-300.0, 300.0), init=0.0, time_varying=true),
        FittedVariable("s"; bounds=(1e-7, 1e-5), init=1e-6, held=[2.0e-6])]
    pspec = ParameterSpec(expt; variables=vars, intervals=[mid],
        dependents=["Tu" => "t >= 0.6 && t <= 2.4 ? T_D * u_D : 0.0"],
        bindings=["T_D" => "ions.D.T_eV", "u_D" => "ions.D.u_kms", "s" => (si, :scale)])
    working = Experiment(; trace=pspec.trace, probe=expt.probe, collection=expt.collection,
        lambda_m=expt.lambda, effects=expt.effects)
    reg = Regularizer([RegularizerTerm(pspec; select=["u_D"], operator=:D2, normalization=:absolute),
        RegularizerTerm(pspec; select=["T_D"], operator=:D2, normalization=:absolute, scale=100.0,
            interval=T_interval),
        RegularizerTerm(pspec; select=["Tu"], operator=:D2, normalization=:absolute, scale=1.0e4,
            interval=1)]; weight=1e-2)
    theta0 = flatten(pspec)
    image = measure(unflatten(working, pspec, theta0))
    data = StreakData(lambda_m=working.lambda, t_s=working.t, image=image,
        sigma=sigma(GaussianNoise(a=1.0, b=0.02), image))
    return FitProblem(working, pspec, data; regularizer=reg), theta0 .+ 0.05 .* (1:length(theta0))
end
