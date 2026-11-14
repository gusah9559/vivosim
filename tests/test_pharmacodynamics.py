"""Linking concentration to a measured response."""

from __future__ import annotations

import numpy as np
import pytest

from vivosim import RAT, Compound, IVBolus, OralDose, Regimen, simulate
from vivosim.pharmacodynamics import EmaxModel, effect_of


def drug() -> Compound:
    return Compound(
        name="drug",
        default_partition_coefficient=2.0,
        partition_coefficients={"brain": 0.8},
        fraction_unbound_plasma=0.4,
        hepatic_intrinsic_clearance_L_per_h=2.0,
    )


def profile(model: EmaxModel, regimen: Regimen | None = None, duration_h: float = 24):
    result = simulate(
        drug(),
        RAT,
        regimen if regimen is not None else Regimen(IVBolus(1.0)),
        duration_h=duration_h,
        n_points=1201,
    )
    return effect_of(result, model)


# -- the curve itself -----------------------------------------------------


def test_no_compound_gives_the_baseline() -> None:
    model = EmaxModel(baseline=70.0, e_max=30.0, ec50_mg_per_L=2.0)
    assert model.response(0.0) == pytest.approx(70.0)


def test_ec50_gives_half_the_maximum_effect() -> None:
    model = EmaxModel(baseline=0.0, e_max=100.0, ec50_mg_per_L=2.0)
    assert model.response(2.0) == pytest.approx(50.0)


def test_response_saturates_at_emax() -> None:
    # The point of the model: biology runs out of receptors, so the response
    # flattens however much more compound arrives.
    model = EmaxModel(baseline=10.0, e_max=40.0, ec50_mg_per_L=1.0)
    assert model.response(1e6) == pytest.approx(50.0, rel=1e-4)
    assert model.maximum_response == 50.0


def test_a_suppressing_compound_lowers_the_response() -> None:
    model = EmaxModel(baseline=100.0, e_max=-60.0, ec50_mg_per_L=1.0)
    assert model.response(1.0) == pytest.approx(70.0)
    assert model.response(1e6) == pytest.approx(40.0, rel=1e-4)


def test_a_higher_hill_coefficient_is_more_switch_like() -> None:
    shallow = EmaxModel(e_max=100.0, ec50_mg_per_L=1.0, hill=1.0)
    steep = EmaxModel(e_max=100.0, ec50_mg_per_L=1.0, hill=8.0)
    # Both pass through the same half-maximal point...
    assert shallow.response(1.0) == pytest.approx(steep.response(1.0))
    # ...but the steep one is flatter below it and flatter above it.
    assert steep.response(0.5) < shallow.response(0.5)
    assert steep.response(2.0) > shallow.response(2.0)


def test_negative_concentrations_are_clamped() -> None:
    # Solver undershoot can leave a tiny negative concentration; it must not
    # propagate into a nonsensical response.
    model = EmaxModel(baseline=5.0, e_max=10.0, ec50_mg_per_L=1.0)
    assert model.response(-1e-9) == pytest.approx(5.0)


def test_model_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="ec50_mg_per_L must be positive"):
        EmaxModel(ec50_mg_per_L=0.0)
    with pytest.raises(ValueError, match="hill must be positive"):
        EmaxModel(hill=0.0)
    with pytest.raises(ValueError, match="equilibration_rate_const_per_h must be positive"):
        EmaxModel(equilibration_rate_const_per_h=0.0)


# -- response over time ---------------------------------------------------


def test_an_immediate_response_peaks_with_concentration() -> None:
    model = EmaxModel(e_max=100.0, ec50_mg_per_L=0.5)
    effect = profile(model)
    assert not model.has_delay
    assert np.array_equal(effect.effect_site_concentration, effect.driving_concentration)
    peak_concentration_time = effect.time_h[int(np.argmax(effect.driving_concentration))]
    assert effect.time_of_peak_h == pytest.approx(peak_concentration_time)


def test_an_effect_compartment_delays_the_peak() -> None:
    # The reason the compartment exists: response lags concentration, because
    # the compound has to reach its target and engage it.
    immediate = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.5))
    delayed = profile(
        EmaxModel(e_max=100.0, ec50_mg_per_L=0.5, equilibration_rate_const_per_h=0.4)
    )
    assert delayed.time_of_peak_h > immediate.time_of_peak_h


def test_a_slower_equilibration_delays_the_peak_further() -> None:
    fast = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.5, equilibration_rate_const_per_h=2.0))
    slow = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.5, equilibration_rate_const_per_h=0.2))
    assert slow.time_of_peak_h > fast.time_of_peak_h


def test_the_effect_site_never_overshoots_the_driving_concentration() -> None:
    # It is a lagging average of what came before, so it cannot exceed the peak
    # that produced it.
    effect = profile(
        EmaxModel(e_max=100.0, ec50_mg_per_L=0.5, equilibration_rate_const_per_h=0.5)
    )
    assert effect.effect_site_concentration.max() <= effect.driving_concentration.max()
    assert np.all(effect.effect_site_concentration >= -1e-12)


def test_a_very_fast_equilibration_approaches_the_immediate_case() -> None:
    immediate = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.5))
    almost = profile(
        EmaxModel(e_max=100.0, ec50_mg_per_L=0.5, equilibration_rate_const_per_h=500.0)
    )
    assert almost.area_under_effect() == pytest.approx(
        immediate.area_under_effect(), rel=0.02
    )


def test_a_brain_acting_compound_can_be_driven_by_brain_concentration() -> None:
    plasma_driven = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.5))
    brain_driven = profile(
        EmaxModel(e_max=100.0, ec50_mg_per_L=0.5, driving_site="brain")
    )
    # Brain partitioning is below 1 for this compound, so the same model driven
    # by brain concentration sees less compound and produces less response.
    assert brain_driven.area_under_effect() < plasma_driven.area_under_effect()


def test_an_unknown_driving_site_is_reported_clearly() -> None:
    result = simulate(drug(), RAT, Regimen(IVBolus(1.0)), duration_h=4)
    with pytest.raises(KeyError, match="unknown tissue"):
        effect_of(result, EmaxModel(driving_site="pancreas"))


# -- what the metrics mean ------------------------------------------------


def test_saturation_reports_how_much_headroom_is_left() -> None:
    # A potent compound at this dose runs the response up against its ceiling;
    # a weak one leaves most of the range unused.
    potent = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.001))
    weak = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=1000.0))
    assert potent.saturation > 0.95
    assert weak.saturation < 0.1


def test_a_saturated_response_barely_moves_when_the_dose_doubles() -> None:
    # The practical consequence of saturation, and a thing a linear intuition
    # gets badly wrong.
    model = EmaxModel(e_max=100.0, ec50_mg_per_L=0.01)
    single = profile(model, Regimen(IVBolus(1.0)))
    double = profile(model, Regimen(IVBolus(2.0)))
    assert double.peak_effect == pytest.approx(single.peak_effect, rel=0.02)


def test_time_above_a_threshold_grows_with_dose() -> None:
    model = EmaxModel(e_max=100.0, ec50_mg_per_L=1.0)
    small = profile(model, Regimen(IVBolus(0.5)), duration_h=48)
    large = profile(model, Regimen(IVBolus(5.0)), duration_h=48)
    assert large.time_above(50.0) > small.time_above(50.0)


def test_time_above_an_unreachable_threshold_is_zero() -> None:
    effect = profile(EmaxModel(e_max=10.0, ec50_mg_per_L=1.0))
    assert effect.time_above(1e6) == 0.0


def test_baseline_only_response_has_no_change() -> None:
    model = EmaxModel(baseline=42.0, e_max=100.0, ec50_mg_per_L=1.0)
    result = simulate(drug(), RAT, Regimen(), duration_h=12)
    effect = effect_of(result, model)
    assert np.allclose(effect.effect, 42.0)
    assert effect.peak_change == pytest.approx(0.0)
    assert effect.saturation == pytest.approx(0.0)


def test_repeated_dosing_stacks_the_response() -> None:
    from vivosim import repeated

    model = EmaxModel(e_max=100.0, ec50_mg_per_L=2.0, equilibration_rate_const_per_h=0.5)
    once = profile(model, Regimen(OralDose(2.0, absorption_rate_const_per_h=1.0)), 48)
    thrice = profile(
        model,
        repeated(OralDose(2.0, absorption_rate_const_per_h=1.0), every_h=8.0, count=3),
        48,
    )
    assert thrice.area_under_effect() > once.area_under_effect()


def test_summary_reports_the_expected_keys() -> None:
    summary = profile(EmaxModel(e_max=100.0, ec50_mg_per_L=0.5)).summary()
    assert set(summary) == {
        "peak_effect",
        "peak_change_from_baseline",
        "time_of_peak_h",
        "saturation",
        "area_under_effect",
    }
