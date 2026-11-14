"""The engine must conserve mass and reproduce known analytical limits."""

from __future__ import annotations

import numpy as np
import pytest

from vivosim import (
    RAT,
    Compound,
    IVBolus,
    IVInfusion,
    OralDose,
    Regimen,
    get_physiology,
    repeated,
    simulate,
)

DOSE_MG = 1.0


def inert() -> Compound:
    """A compound that distributes but is never eliminated."""
    return Compound(name="inert", default_partition_coefficient=2.0)


def cleared() -> Compound:
    """A compound cleared by the liver."""
    return Compound(
        name="cleared",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.2,
        hepatic_intrinsic_clearance_L_per_h=1.5,
    )


# -- conservation of mass -------------------------------------------------


def test_inert_compound_conserves_mass() -> None:
    result = simulate(inert(), RAT, Regimen(IVBolus(DOSE_MG)), duration_h=24)
    assert result.mass_balance_error() < 1e-6
    # With no elimination the dose stays in the body for the whole simulation.
    assert np.allclose(result.total_in_body(), DOSE_MG, atol=1e-6)
    assert result.total_eliminated()[-1] == pytest.approx(0.0, abs=1e-9)


def test_cleared_compound_conserves_mass() -> None:
    result = simulate(cleared(), RAT, Regimen(IVBolus(DOSE_MG)), duration_h=48)
    assert result.mass_balance_error() < 1e-6
    # Everything administered ends up either in the body or eliminated.
    total = result.total_in_body()[-1] + result.total_eliminated()[-1]
    assert total == pytest.approx(DOSE_MG, rel=1e-6)


def test_infusion_conserves_mass_during_and_after() -> None:
    regimen = Regimen(IVInfusion(amount_mg=DOSE_MG, duration_h=4.0))
    result = simulate(cleared(), RAT, regimen, duration_h=24)
    assert result.mass_balance_error() < 1e-6


def test_oral_dose_conserves_mass() -> None:
    regimen = Regimen(OralDose(DOSE_MG, absorption_rate_const_per_h=0.8))
    result = simulate(cleared(), RAT, regimen, duration_h=48)
    assert result.mass_balance_error() < 1e-6


def test_repeated_dosing_conserves_mass() -> None:
    regimen = repeated(IVBolus(DOSE_MG), every_h=6.0, count=4)
    result = simulate(cleared(), RAT, regimen, duration_h=48)
    assert result.mass_balance_error() < 1e-6
    assert regimen.total_amount_mg == pytest.approx(4 * DOSE_MG)


def test_no_dose_gives_an_empty_animal() -> None:
    result = simulate(cleared(), RAT, Regimen(), duration_h=10)
    assert np.allclose(result.amounts, 0.0)


# -- analytical limits ----------------------------------------------------


def test_equilibrium_tissue_ratio_approaches_kp() -> None:
    # With no elimination the system must settle so that every tissue sits at
    # Kp times the plasma concentration. This is the strongest single check
    # that the perfusion-limited exchange terms are written correctly.
    kp = {"muscle": 3.0, "adipose": 12.0, "brain": 0.4}
    compound = Compound(
        name="equilibrating",
        partition_coefficients=kp,
        default_partition_coefficient=1.0,
        blood_plasma_ratio=1.3,
    )
    result = simulate(compound, RAT, Regimen(IVBolus(DOSE_MG)), duration_h=400)

    plasma = result.plasma_concentration()[-1]
    for tissue, expected in kp.items():
        ratio = result.tissue_concentration(tissue)[-1] / plasma
        assert ratio == pytest.approx(expected, rel=1e-3), tissue


def test_steady_state_matches_clearance() -> None:
    # Under a long constant infusion the blood concentration must settle where
    # elimination balances input. Hepatic extraction of a well-stirred liver
    # gives CL = Q_h * CL_int_u / (Q_h + CL_int_u) on blood concentrations.
    compound = cleared()
    rate_mg_per_h = 0.5
    duration = 600.0
    regimen = Regimen(IVInfusion(amount_mg=rate_mg_per_h * duration, duration_h=duration))
    result = simulate(compound, RAT, regimen, duration_h=duration, n_points=2001)

    q_h = RAT.liver_inflow_L_per_h()
    cl_int_u = compound.hepatic_intrinsic_clearance_L_per_h * (
        compound.fraction_unbound_plasma / compound.blood_plasma_ratio
    )
    expected_blood_cl = q_h * cl_int_u / (q_h + cl_int_u)

    observed = result.blood_concentration("arterial")[-1]
    expected = rate_mg_per_h / expected_blood_cl
    assert observed == pytest.approx(expected, rel=1e-3)


def test_elimination_is_first_order_in_dose() -> None:
    # The model is linear, so doubling the dose doubles every exposure metric.
    small = simulate(cleared(), RAT, Regimen(IVBolus(1.0)), duration_h=24)
    large = simulate(cleared(), RAT, Regimen(IVBolus(2.0)), duration_h=24)
    assert large.auc("plasma") == pytest.approx(2 * small.auc("plasma"), rel=1e-6)
    assert large.cmax("plasma") == pytest.approx(2 * small.cmax("plasma"), rel=1e-6)


def test_oral_route_shows_first_pass_loss() -> None:
    # An oral dose passes through the liver before reaching the circulation, so
    # its systemic exposure is lower than the same dose given intravenously.
    compound = cleared()
    iv = simulate(compound, RAT, Regimen(IVBolus(DOSE_MG)), duration_h=72)
    oral = simulate(
        compound,
        RAT,
        Regimen(OralDose(DOSE_MG, absorption_rate_const_per_h=1.0)),
        duration_h=72,
    )
    assert oral.auc("plasma") < iv.auc("plasma")
    # Absorption takes time, so the oral peak comes later and is lower.
    assert oral.tmax("plasma") > iv.tmax("plasma")
    assert oral.cmax("plasma") < iv.cmax("plasma")


def test_no_first_pass_without_hepatic_clearance() -> None:
    # Without metabolism the liver only delays an oral dose; all of it still
    # reaches the circulation, so exposure matches the intravenous route.
    compound = inert()
    duration = 400.0
    iv = simulate(compound, RAT, Regimen(IVBolus(DOSE_MG)), duration_h=duration)
    oral = simulate(
        compound,
        RAT,
        Regimen(OralDose(DOSE_MG, absorption_rate_const_per_h=2.0)),
        duration_h=duration,
    )
    assert oral.plasma_concentration()[-1] == pytest.approx(
        iv.plasma_concentration()[-1], rel=1e-4
    )


def test_bolus_appears_immediately_in_venous_blood() -> None:
    result = simulate(inert(), RAT, Regimen(IVBolus(DOSE_MG)), duration_h=24)
    # The dose enters the venous pool, so t=0 holds the whole dose there.
    assert result.amount("venous")[0] == pytest.approx(DOSE_MG)
    assert result.amount("arterial")[0] == pytest.approx(0.0)


def test_delayed_bolus_is_absent_until_its_time() -> None:
    result = simulate(
        inert(), RAT, Regimen(IVBolus(DOSE_MG, time_h=5.0)), duration_h=10, n_points=1001
    )
    before = result.time_h < 5.0
    assert np.allclose(result.total_in_body()[before], 0.0, atol=1e-9)
    assert result.total_in_body()[-1] == pytest.approx(DOSE_MG, rel=1e-6)


def test_oral_lag_delays_absorption() -> None:
    regimen = Regimen(
        OralDose(DOSE_MG, absorption_rate_const_per_h=2.0, lag_h=3.0)
    )
    result = simulate(cleared(), RAT, regimen, duration_h=24, n_points=1001)
    before = result.time_h < 3.0
    assert np.allclose(result.plasma_concentration()[before], 0.0, atol=1e-12)
    assert result.cmax("plasma") > 0.0


def test_adipose_accumulates_a_lipophilic_compound() -> None:
    compound = Compound(
        name="lipophilic",
        partition_coefficients={"adipose": 50.0},
        default_partition_coefficient=1.0,
    )
    result = simulate(compound, RAT, Regimen(IVBolus(DOSE_MG)), duration_h=200)
    assert result.tissue_concentration("adipose")[-1] > result.tissue_concentration("muscle")[-1]


# -- scaling and plumbing -------------------------------------------------


def test_heavier_animal_dilutes_the_same_dose() -> None:
    light = simulate(inert(), get_physiology("rat", 0.2), Regimen(IVBolus(DOSE_MG)), duration_h=24)
    heavy = simulate(inert(), get_physiology("rat", 0.4), Regimen(IVBolus(DOSE_MG)), duration_h=24)
    assert heavy.cmax("plasma") < light.cmax("plasma")


def test_mouse_and_rat_both_run() -> None:
    for species in ("rat", "mouse"):
        result = simulate(
            cleared(), get_physiology(species), Regimen(IVBolus(DOSE_MG)), duration_h=24
        )
        assert result.mass_balance_error() < 1e-6


def test_explicit_output_grid_is_respected() -> None:
    grid = np.array([0.0, 0.5, 1.0, 4.0, 12.0])
    result = simulate(inert(), RAT, Regimen(IVBolus(DOSE_MG)), times_h=grid)
    assert np.array_equal(result.time_h, grid)


def test_summary_reports_the_expected_keys() -> None:
    result = simulate(cleared(), RAT, Regimen(IVBolus(DOSE_MG)), duration_h=24)
    summary = result.summary()
    assert set(summary) == {
        "cmax_mg_per_L",
        "tmax_h",
        "auc_mg_h_per_L",
        "mass_balance_error_mg",
    }


def test_unknown_compartment_is_reported_clearly() -> None:
    result = simulate(inert(), RAT, Regimen(IVBolus(DOSE_MG)), duration_h=1)
    with pytest.raises(KeyError, match="unknown tissue"):
        result.tissue_concentration("pancreas")
    with pytest.raises(KeyError, match="unknown compartment"):
        result.amount("nowhere")


def test_bad_output_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate(inert(), RAT, Regimen(IVBolus(1.0)), times_h=np.array([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="at least 2 points"):
        simulate(inert(), RAT, Regimen(IVBolus(1.0)), times_h=np.array([0.0]))


def test_csv_export_round_trips(tmp_path) -> None:
    result = simulate(cleared(), RAT, Regimen(IVBolus(DOSE_MG)), duration_h=6, n_points=25)
    path = tmp_path / "profile.csv"
    result.to_csv(str(path))
    loaded = np.genfromtxt(path, delimiter=",", names=True)
    assert loaded["time_h"].size == 25
    assert np.allclose(loaded["time_h"], result.time_h, atol=1e-6)
