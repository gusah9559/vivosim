"""Measuring an interaction: each compound alone against together."""

from __future__ import annotations

import numpy as np
import pytest

from vivosim import (
    RAT,
    Compound,
    Inhibition,
    IVBolus,
    IVInfusion,
    Metabolism,
    Regimen,
    assess_interaction,
    classify_ratio,
)

CYP = "CYP3A4"
OTHER = "CYP2D6"


def substrate(name: str = "victim", enzyme: str = CYP) -> Compound:
    return Compound(
        name=name,
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        metabolism=(Metabolism(enzyme=enzyme, vmax_mg_per_h=2.0, km_mg_per_L=0.5),),
    )


def inhibitor(name: str = "inhibitor", ki: float = 0.05, enzyme: str = CYP) -> Compound:
    return Compound(
        name=name,
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        inhibits=(Inhibition(enzyme=enzyme, ki_mg_per_L=ki),),
    )


def report(victim: Compound, other: Compound, **kwargs):
    return assess_interaction(
        [victim, other],
        RAT,
        [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=72.0))],
        duration_h=72,
        **kwargs,
    )


# -- classification -------------------------------------------------------


def test_bands_follow_the_standard_thresholds() -> None:
    assert classify_ratio(6.0) == "strong"
    assert classify_ratio(5.0) == "strong"
    assert classify_ratio(3.0) == "moderate"
    assert classify_ratio(2.0) == "moderate"
    assert classify_ratio(1.5) == "weak"
    assert classify_ratio(1.25) == "weak"
    assert classify_ratio(1.1) == "none"
    assert classify_ratio(1.0) == "none"


def test_a_fall_in_exposure_is_classified_by_its_own_size() -> None:
    assert classify_ratio(1 / 6.0) == "reduced strong"
    assert classify_ratio(1 / 3.0) == "reduced moderate"
    assert classify_ratio(1 / 1.5) == "reduced weak"


def test_a_negligible_fall_is_not_a_finding() -> None:
    # A ratio a hair under 1 must not be reported as an effect just because it
    # is on the low side of it.
    assert classify_ratio(0.999) == "none"
    assert classify_ratio(0.81) == "none"


def test_a_degenerate_ratio_is_not_a_finding() -> None:
    assert classify_ratio(0.0) == "none"
    assert classify_ratio(float("nan")) == "none"


# -- the comparison -------------------------------------------------------


def test_an_interaction_needs_two_compounds() -> None:
    with pytest.raises(ValueError, match="at least two compounds"):
        assess_interaction([substrate()], RAT, Regimen(IVBolus(1.0)), duration_h=8)


def test_both_runs_share_one_time_grid() -> None:
    # The profiles are subtracted point by point, so they must be sampled at
    # identical times.
    result = report(substrate(), inhibitor())
    for entry in result:
        assert np.array_equal(entry.alone.time_h, entry.together.time_h)
        assert np.array_equal(entry.alone.time_h, result.time_h)


def test_an_inhibitor_raises_the_victim_exposure() -> None:
    result = report(substrate(), inhibitor())
    victim = result["victim"]
    assert victim.auc_together > victim.auc_alone
    assert victim.auc_ratio > 2.0
    assert victim.classification == "moderate"
    assert victim.is_affected


def test_the_inhibitor_itself_is_unaffected() -> None:
    # It is not metabolised by the enzyme it blocks, so nothing about the
    # victim's presence should change it.
    result = report(substrate(), inhibitor())
    other = result["inhibitor"]
    assert other.auc_ratio == pytest.approx(1.0, rel=1e-6)
    assert other.classification == "none"
    assert not other.is_affected
    assert result.affected() == [result["victim"]]


def test_compounds_on_different_enzymes_show_no_interaction() -> None:
    result = report(substrate(), inhibitor(enzyme=OTHER))
    for entry in result:
        assert entry.auc_ratio == pytest.approx(1.0, rel=1e-4)
        assert not entry.is_affected
    assert result.affected() == []
    assert result.interacting_enzymes() == {}


def test_a_stronger_inhibitor_gives_a_larger_ratio() -> None:
    weak = report(substrate(), inhibitor(ki=1.0))["victim"].auc_ratio
    strong = report(substrate(), inhibitor(ki=0.01))["victim"].auc_ratio
    assert strong > weak > 1.0


def test_two_substrates_of_one_enzyme_affect_each_other() -> None:
    result = assess_interaction(
        [substrate("a"), substrate("b")],
        RAT,
        [Regimen(IVBolus(2.0)), Regimen(IVBolus(8.0))],
        duration_h=72,
    )
    # Each slows the other, so neither is unchanged.
    assert result["a"].auc_ratio > 1.0
    assert result["b"].auc_ratio > 1.0
    assert set(result.interacting_enzymes()[CYP]) == {"a", "b"}


def test_the_difference_profile_matches_the_two_runs() -> None:
    victim = report(substrate(), inhibitor())["victim"]
    expected = victim.together.plasma_concentration() - victim.alone.plasma_concentration()
    assert np.allclose(victim.difference(), expected)
    # Inhibition raises exposure, so the difference ends up positive.
    assert victim.difference().max() > 0


def test_largest_change_picks_the_victim() -> None:
    result = report(substrate(), inhibitor())
    assert result.largest_change().name == "victim"


def test_report_is_indexable_by_name_and_position() -> None:
    result = report(substrate(), inhibitor())
    assert result[0].name == "victim"
    assert result["victim"] is result[0]
    assert result.names == ["victim", "inhibitor"]
    assert len(result) == 2
    with pytest.raises(KeyError, match="no compound named"):
        result["nobody"]


def test_summary_reports_every_compound() -> None:
    summary = report(substrate(), inhibitor()).summary()
    assert set(summary) == {"victim", "inhibitor"}
    assert summary["victim"]["auc_ratio"] > 1.0
    assert summary["victim"]["classification"] == "moderate"


def test_solo_runs_match_a_plain_simulation() -> None:
    # The "alone" arm must be an ordinary single-compound simulation, or the
    # ratio compares against something that was never really alone.
    from vivosim import simulate

    victim = substrate()
    result = report(victim, inhibitor())
    direct = simulate(
        victim, RAT, Regimen(IVBolus(1.0)), times_h=result.time_h
    )
    assert result["victim"].auc_alone == pytest.approx(direct.auc("plasma"), rel=1e-9)


def test_mass_is_conserved_in_both_arms() -> None:
    result = report(substrate(), inhibitor())
    for entry in result:
        assert entry.alone.mass_balance_error() < 1e-6
        assert entry.together.mass_balance_error() < 1e-6
