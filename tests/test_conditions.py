"""Disease states, and the pharmacology they are supposed to reproduce."""

from __future__ import annotations

import pytest

from vivosim import (
    CONDITIONS,
    HEALTHY,
    RAT,
    Compound,
    Condition,
    IVBolus,
    Metabolism,
    Regimen,
    hepatic_impairment,
    obesity,
    renal_impairment,
    simulate,
)


def hepatic_drug(clearance: float = 3.0) -> Compound:
    """Cleared only by the liver."""
    return Compound(
        name="hepatic",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.2,
        hepatic_intrinsic_clearance_L_per_h=clearance,
    )


def renal_drug() -> Compound:
    """Excreted unchanged by the kidney, untouched by the liver."""
    return Compound(
        name="renal",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        renal_clearance_L_per_h=0.3,
    )


def lipophilic() -> Compound:
    return Compound(
        name="lipophilic",
        partition_coefficients={"adipose": 60.0},
        default_partition_coefficient=1.0,
        fraction_unbound_plasma=0.5,
        hepatic_intrinsic_clearance_L_per_h=1.0,
    )


def auc_under(condition: Condition, compound: Compound, duration_h: float = 48) -> float:
    sick_compound, sick_body = condition.apply(compound, RAT)
    return simulate(
        sick_compound, sick_body, Regimen(IVBolus(1.0)), duration_h=duration_h
    ).auc("plasma")


# -- the invariant that must never break ----------------------------------


@pytest.mark.parametrize("condition", list(CONDITIONS.values()), ids=list(CONDITIONS))
def test_every_condition_keeps_the_animal_balanced(condition: Condition) -> None:
    # Flows must still sum to cardiac output and volumes to body weight. A
    # condition that broke this would leak mass and quietly corrupt every
    # number downstream.
    body = condition.apply_physiology(RAT)
    assert abs(body.flow_balance_error()) < 1e-9
    assert abs(body.volume_balance_error()) < 1e-9


@pytest.mark.parametrize("condition", list(CONDITIONS.values()), ids=list(CONDITIONS))
def test_every_condition_still_conserves_mass(condition: Condition) -> None:
    compound, body = condition.apply(hepatic_drug(), RAT)
    result = simulate(compound, body, Regimen(IVBolus(1.0)), duration_h=24)
    assert result.mass_balance_error() < 1e-6


@pytest.mark.parametrize("condition", list(CONDITIONS.values()), ids=list(CONDITIONS))
def test_every_condition_keeps_parameters_physical(condition: Condition) -> None:
    compound, body = condition.apply(hepatic_drug(), RAT)
    assert 0.0 < compound.fraction_unbound_plasma <= 1.0
    assert compound.hepatic_intrinsic_clearance_L_per_h >= 0
    assert body.body_weight_kg > 0
    for spec in body.tissues.values():
        assert spec.volume_fraction > 0
        assert spec.flow_fraction >= 0


# -- healthy is the identity ----------------------------------------------


def test_healthy_changes_nothing() -> None:
    assert HEALTHY.is_healthy
    compound, body = HEALTHY.apply(hepatic_drug(), RAT)
    assert compound == hepatic_drug()
    assert body.body_weight_kg == RAT.body_weight_kg
    assert body.tissues == RAT.tissues


def test_an_impairment_is_not_healthy() -> None:
    assert not hepatic_impairment("mild").is_healthy
    assert not obesity(2.0).is_healthy


# -- validation -----------------------------------------------------------


def test_severity_must_be_a_known_grade() -> None:
    with pytest.raises(ValueError, match="severity must be one of"):
        hepatic_impairment("catastrophic")
    with pytest.raises(ValueError, match="severity must be one of"):
        renal_impairment("slight")


def test_multipliers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="liver_flow must be positive"):
        Condition(name="broken", description="", liver_flow=0.0)
    with pytest.raises(ValueError, match="fold must be positive"):
        obesity(0.0)


def test_unbound_fraction_cannot_exceed_one() -> None:
    # Severe impairment raises the unbound fraction, but a fraction is still a
    # fraction -- a compound cannot be more than entirely free.
    compound = hepatic_drug().replace(fraction_unbound_plasma=0.9)
    altered = hepatic_impairment("severe").apply_compound(compound)
    assert altered.fraction_unbound_plasma == 1.0


def test_impairment_scales_vmax_not_km() -> None:
    # Disease destroys enzyme; it does not change the enzyme's affinity.
    compound = Compound(
        name="saturable",
        default_partition_coefficient=2.0,
        metabolism=(Metabolism(enzyme="CYP3A4", vmax_mg_per_h=2.0, km_mg_per_L=0.5),),
    )
    altered = hepatic_impairment("severe").apply_compound(compound)
    assert altered.metabolism[0].km_mg_per_L == 0.5
    assert altered.metabolism[0].vmax_mg_per_h < 2.0


# -- the pharmacology --------------------------------------------------------


def test_hepatic_impairment_raises_exposure_progressively() -> None:
    healthy = auc_under(HEALTHY, hepatic_drug())
    mild = auc_under(hepatic_impairment("mild"), hepatic_drug())
    moderate = auc_under(hepatic_impairment("moderate"), hepatic_drug())
    severe = auc_under(hepatic_impairment("severe"), hepatic_drug())
    assert healthy < mild < moderate < severe


def test_hepatic_impairment_hits_a_low_extraction_compound_hardest() -> None:
    # A compound the liver clears slowly depends on enzyme capacity, so losing
    # enzyme hurts it most. One cleared as fast as blood arrives is limited by
    # flow and is partly shielded -- the same extraction-dependence the
    # population model shows, arrived at from the other direction.
    condition = hepatic_impairment("severe")
    low = auc_under(condition, hepatic_drug(0.1)) / auc_under(HEALTHY, hepatic_drug(0.1))
    high = auc_under(condition, hepatic_drug(400.0)) / auc_under(
        HEALTHY, hepatic_drug(400.0)
    )
    assert low > high


def test_renal_impairment_spares_a_liver_cleared_compound() -> None:
    # Nothing about this compound passes through the kidney, so losing kidney
    # function must leave it essentially untouched.
    healthy = auc_under(HEALTHY, hepatic_drug())
    impaired = auc_under(renal_impairment("severe"), hepatic_drug())
    assert impaired == pytest.approx(healthy, rel=0.02)


def test_renal_impairment_raises_exposure_of_a_renally_cleared_compound() -> None:
    healthy = auc_under(HEALTHY, renal_drug())
    moderate = auc_under(renal_impairment("moderate"), renal_drug())
    severe = auc_under(renal_impairment("severe"), renal_drug())
    assert healthy < moderate < severe
    assert severe / healthy > 1.2


def test_obesity_adds_weight_rather_than_swapping_tissue() -> None:
    body = obesity(3.0).apply_physiology(RAT)
    assert body.body_weight_kg > RAT.body_weight_kg
    # Fat mass grows; the other organs keep roughly their absolute size.
    assert body.volume_L("adipose") > RAT.volume_L("adipose")
    assert body.volume_L("liver") == pytest.approx(RAT.volume_L("liver"), rel=1e-6)
    assert body.volume_L("brain") == pytest.approx(RAT.volume_L("brain"), rel=1e-6)


def test_obesity_gives_a_lipophilic_compound_somewhere_to_hide() -> None:
    # The extra fat is a reservoir: the peak falls because more of the dose
    # leaves the blood, and more is still in the body at the end.
    condition = obesity(4.0)
    compound, body = condition.apply(lipophilic(), RAT)
    fat = simulate(compound, body, Regimen(IVBolus(1.0)), duration_h=120)
    lean = simulate(lipophilic(), RAT, Regimen(IVBolus(1.0)), duration_h=120)
    assert fat.total_in_body()[-1] > lean.total_in_body()[-1]


def test_obesity_barely_touches_a_water_soluble_compound() -> None:
    water_soluble = Compound(
        name="polar",
        default_partition_coefficient=0.6,
        fraction_unbound_plasma=0.8,
        hepatic_intrinsic_clearance_L_per_h=1.0,
    )
    fat = auc_under(obesity(4.0), water_soluble)
    lean = auc_under(HEALTHY, water_soluble)
    assert fat == pytest.approx(lean, rel=0.15)


def test_conditions_compose_with_the_population_model() -> None:
    # The point of having both: asking what a dose does in a population that
    # is not healthy.
    from vivosim import Variability, simulate_population

    compound, body = hepatic_impairment("moderate").apply(hepatic_drug(), RAT)
    population = simulate_population(
        compound,
        body,
        Regimen(IVBolus(1.0)),
        n=12,
        seed=3,
        duration_h=24,
        variability=Variability(hepatic_clearance=0.5),
    )
    assert population.size == 12
    assert population.fold_range() > 1.0
