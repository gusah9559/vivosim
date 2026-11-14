"""Comparing predictions against measured data."""

from __future__ import annotations

import numpy as np
import pytest

from vivosim import HUMAN, RAT, Compound, IVBolus, Regimen, simulate
from vivosim.validation import ObservedProfile, validate, validate_many


def compound() -> Compound:
    return Compound(
        name="example",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        hepatic_intrinsic_clearance_L_per_h=2.0,
    )


def simulation(duration_h: float = 24.0):
    return simulate(compound(), RAT, Regimen(IVBolus(1.0)), duration_h=duration_h)


def perfect_observation(result, times=(1.0, 4.0, 12.0)) -> ObservedProfile:
    """Measurements that agree exactly with a simulation."""
    grid = np.asarray(times, dtype=float)
    curve = np.interp(grid, result.time_h, result.plasma_concentration())
    return ObservedProfile(time_h=grid, concentration=curve, source="synthetic")


# -- the observed profile -------------------------------------------------


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="3 times but 2 concentrations"):
        ObservedProfile(time_h=np.array([0.0, 1.0, 2.0]), concentration=np.array([1.0, 2.0]))


def test_unordered_or_negative_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be in order"):
        ObservedProfile(time_h=np.array([0.0, 2.0, 1.0]), concentration=np.ones(3))
    with pytest.raises(ValueError, match="must be non-negative"):
        ObservedProfile(time_h=np.array([-1.0, 1.0]), concentration=np.ones(2))


def test_an_empty_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        ObservedProfile(time_h=np.array([]), concentration=np.array([]))


# -- the metrics ----------------------------------------------------------


def test_a_perfect_prediction_scores_exactly_one() -> None:
    result = simulation()
    comparison = validate(result, perfect_observation(result))
    assert comparison.average_fold_error == pytest.approx(1.0, rel=1e-9)
    assert comparison.absolute_average_fold_error == pytest.approx(1.0, rel=1e-9)
    assert comparison.fraction_within_two_fold == 1.0
    assert comparison.worst_fold_error == pytest.approx(1.0, rel=1e-9)


def test_a_uniformly_high_prediction_shows_its_bias() -> None:
    # Halving every measurement means the prediction is twice the truth
    # everywhere, which both metrics must report as exactly two-fold.
    result = simulation()
    truth = perfect_observation(result)
    halved = ObservedProfile(time_h=truth.time_h, concentration=truth.concentration / 2)
    comparison = validate(result, halved)
    assert comparison.average_fold_error == pytest.approx(2.0, rel=1e-6)
    assert comparison.absolute_average_fold_error == pytest.approx(2.0, rel=1e-6)


def test_bias_cancels_but_precision_does_not() -> None:
    # The reason both numbers are reported: a prediction that is twice too high
    # half the time and half too high the rest is unbiased on average while
    # being wrong everywhere.
    result = simulation()
    truth = perfect_observation(result, times=(2.0, 6.0))
    skewed = ObservedProfile(
        time_h=truth.time_h,
        concentration=np.array([truth.concentration[0] / 2, truth.concentration[1] * 2]),
    )
    comparison = validate(result, skewed)
    assert comparison.average_fold_error == pytest.approx(1.0, rel=1e-6)
    assert comparison.absolute_average_fold_error == pytest.approx(2.0, rel=1e-6)


def test_fraction_within_a_fold_tightens_as_the_bar_rises() -> None:
    result = simulation()
    truth = perfect_observation(result, times=(2.0, 6.0, 10.0))
    off = ObservedProfile(
        time_h=truth.time_h, concentration=truth.concentration * np.array([1.0, 1.4, 3.0])
    )
    comparison = validate(result, off)
    assert comparison.fraction_within(1.1) == pytest.approx(1 / 3)
    assert comparison.fraction_within(2.0) == pytest.approx(2 / 3)
    assert comparison.fraction_within(5.0) == 1.0


def test_an_impossible_fold_bar_is_rejected() -> None:
    result = simulation()
    with pytest.raises(ValueError, match="fold must be at least 1"):
        validate(result, perfect_observation(result)).fraction_within(0.5)


def test_undetected_measurements_are_excluded_not_counted() -> None:
    # A fold error is undefined where nothing was measured, and counting such a
    # point as agreement or disagreement would both be wrong.
    result = simulation()
    truth = perfect_observation(result, times=(1.0, 4.0, 12.0))
    with_zero = ObservedProfile(
        time_h=truth.time_h,
        concentration=np.array([truth.concentration[0], 0.0, truth.concentration[2]]),
    )
    comparison = validate(result, with_zero)
    assert comparison.n == 2
    assert comparison.excluded == 1
    assert comparison.average_fold_error == pytest.approx(1.0, rel=1e-9)


# -- guard rails ----------------------------------------------------------


def test_observations_past_the_simulation_are_refused() -> None:
    # Silently extrapolating would compare the prediction against nothing.
    result = simulation(duration_h=6.0)
    late = ObservedProfile(time_h=np.array([1.0, 20.0]), concentration=np.array([1.0, 0.1]))
    with pytest.raises(ValueError, match="observations run to 20 h"):
        validate(result, late)


def test_provenance_must_be_declared_honestly() -> None:
    result = simulation()
    assert validate(result, perfect_observation(result)).provenance == "prediction"
    assert (
        validate(result, perfect_observation(result), provenance="fit").provenance == "fit"
    )
    with pytest.raises(ValueError, match="must be 'prediction' or 'fit'"):
        validate(result, perfect_observation(result), provenance="pretty good")


def test_a_tissue_can_be_compared_as_well_as_plasma() -> None:
    result = simulation()
    grid = np.array([2.0, 8.0])
    liver = np.interp(grid, result.time_h, result.tissue_concentration("liver"))
    observed = ObservedProfile(time_h=grid, concentration=liver, matrix="liver")
    assert validate(result, observed).average_fold_error == pytest.approx(1.0, rel=1e-9)


# -- pooling --------------------------------------------------------------


def test_pooling_weights_each_measurement_once() -> None:
    result = simulation()
    truth = perfect_observation(result, times=(1.0, 4.0, 12.0))
    exact = validate(result, truth)
    doubled = validate(
        result,
        ObservedProfile(time_h=truth.time_h, concentration=truth.concentration / 2),
    )
    pooled = validate_many([exact, doubled])
    assert pooled["n"] == 6
    assert pooled["subjects"] == 2
    # Three points at 1x and three at 2x: the geometric mean is sqrt(2).
    assert pooled["average_fold_error"] == pytest.approx(np.sqrt(2.0), rel=1e-6)
    assert pooled["within_two_fold"] == 1.0


def test_pooling_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no validation results"):
        validate_many([])


# -- human physiology, added for exactly this purpose ---------------------


def test_human_physiology_is_balanced() -> None:
    assert abs(HUMAN.flow_balance_error()) < 1e-9
    assert abs(HUMAN.volume_balance_error()) < 1e-9


def test_human_derived_volumes_match_reference_values() -> None:
    # Checked against the figures any pharmacology text gives for a 70 kg
    # adult, so an arithmetic slip in the table cannot pass unnoticed.
    assert HUMAN.cardiac_output_L_per_h / 60 == pytest.approx(6.0, abs=0.5)
    assert HUMAN.plasma_volume_L == pytest.approx(3.0, abs=0.3)
    assert HUMAN.arterial_volume_L + HUMAN.venous_volume_L == pytest.approx(5.5, abs=0.5)
    assert HUMAN.volume_L("liver") == pytest.approx(1.8, abs=0.3)
    assert HUMAN.volume_L("brain") == pytest.approx(1.4, abs=0.3)


def test_a_human_simulation_conserves_mass() -> None:
    result = simulate(compound(), HUMAN, Regimen(IVBolus(100.0)), duration_h=24)
    assert result.mass_balance_error() < 1e-6
