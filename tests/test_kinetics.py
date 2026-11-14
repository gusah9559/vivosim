"""Saturable metabolism and competition between compounds."""

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
    OralDose,
    Regimen,
    Result,
    Study,
    simulate,
)
from vivosim.kinetics import intrinsic_clearance, metabolic_rates

CYP = "CYP3A4"
OTHER = "CYP2D6"


def saturable(name: str = "substrate", vmax: float = 2.0, km: float = 0.5) -> Compound:
    return Compound(
        name=name,
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        metabolism=(Metabolism(enzyme=CYP, vmax_mg_per_h=vmax, km_mg_per_L=km),),
    )


def linear(name: str = "linear", clearance: float = 4.0) -> Compound:
    return Compound(
        name=name,
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        hepatic_intrinsic_clearance_L_per_h=clearance,
    )


# -- validation -----------------------------------------------------------


def test_metabolism_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="km_mg_per_L must be positive"):
        Metabolism(enzyme=CYP, vmax_mg_per_h=1.0, km_mg_per_L=0.0)
    with pytest.raises(ValueError, match="vmax_mg_per_h must be non-negative"):
        Metabolism(enzyme=CYP, vmax_mg_per_h=-1.0, km_mg_per_L=1.0)
    with pytest.raises(ValueError, match="enzyme must be a non-empty name"):
        Metabolism(enzyme="", vmax_mg_per_h=1.0, km_mg_per_L=1.0)


def test_inhibition_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="ki_mg_per_L must be positive"):
        Inhibition(enzyme=CYP, ki_mg_per_L=0.0)


def test_one_enzyme_cannot_be_listed_twice() -> None:
    # Two entries for the same enzyme would double-count competition against it.
    with pytest.raises(ValueError, match="lists the same enzyme twice"):
        Compound(
            name="confused",
            metabolism=(
                Metabolism(enzyme=CYP, vmax_mg_per_h=1.0, km_mg_per_L=1.0),
                Metabolism(enzyme=CYP, vmax_mg_per_h=2.0, km_mg_per_L=3.0),
            ),
        )


def test_compounds_must_have_distinct_names() -> None:
    with pytest.raises(ValueError, match="distinct names"):
        simulate(
            [saturable("same"), saturable("same")],
            RAT,
            Regimen(IVBolus(1.0)),
            duration_h=4,
        )


def test_regimen_count_must_match_compound_count() -> None:
    with pytest.raises(ValueError, match="2 compounds but 3 regimens"):
        simulate(
            [saturable("a"), saturable("b")],
            RAT,
            [Regimen(IVBolus(1.0))] * 3,
            duration_h=4,
        )


# -- the rate expression itself -------------------------------------------


def test_rate_is_half_of_vmax_at_km() -> None:
    pathway = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    rates = metabolic_rates([(0, pathway)], {CYP: [(0, 2.0)]}, [2.0])
    assert rates[0] == pytest.approx(5.0)


def test_rate_approaches_vmax_when_saturated() -> None:
    pathway = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    rates = metabolic_rates([(0, pathway)], {CYP: [(0, 2.0)]}, [2000.0])
    assert rates[0] == pytest.approx(10.0, rel=1e-2)


def test_rate_is_first_order_well_below_km() -> None:
    # Far below Km the expression must collapse to CL_int = Vmax / Km.
    pathway = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    concentration = 1e-5
    rates = metabolic_rates([(0, pathway)], {CYP: [(0, 2.0)]}, [concentration])
    assert rates[0] == pytest.approx(5.0 * concentration, rel=1e-4)


def test_a_competitor_slows_metabolism() -> None:
    pathway = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    alone = metabolic_rates([(0, pathway)], {CYP: [(0, 2.0)]}, [2.0, 0.0])
    contested = metabolic_rates(
        [(0, pathway)], {CYP: [(0, 2.0), (1, 1.0)]}, [2.0, 5.0]
    )
    assert contested[0] < alone[0]


def test_competition_raises_apparent_km_but_not_vmax() -> None:
    # The signature of competitive inhibition: enough substrate still wins.
    pathway = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    inhibitors = {CYP: [(0, 2.0), (1, 1.0)]}
    saturating = metabolic_rates([(0, pathway)], inhibitors, [1e6, 5.0])
    assert saturating[0] == pytest.approx(10.0, rel=1e-3)


def test_a_compound_does_not_compete_with_itself() -> None:
    # Its own occupancy is already in the Michaelis-Menten denominator; counting
    # it again would halve the rate at Km.
    pathway = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    rates = metabolic_rates([(0, pathway)], {CYP: [(0, 2.0)]}, [2.0])
    assert rates[0] == pytest.approx(5.0)


def test_separate_enzymes_do_not_interact() -> None:
    first = Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0)
    alone = metabolic_rates([(0, first)], {CYP: [(0, 2.0)]}, [2.0, 0.0])
    beside = metabolic_rates(
        [(0, first)], {CYP: [(0, 2.0)], OTHER: [(1, 0.1)]}, [2.0, 99.0]
    )
    assert beside[0] == pytest.approx(alone[0])


def test_intrinsic_clearance_sums_every_pathway() -> None:
    pathways = (
        Metabolism(enzyme=CYP, vmax_mg_per_h=10.0, km_mg_per_L=2.0),
        Metabolism(enzyme=OTHER, vmax_mg_per_h=3.0, km_mg_per_L=3.0),
    )
    assert intrinsic_clearance(pathways, 1.5) == pytest.approx(5.0 + 1.0 + 1.5)


def test_compound_reports_its_linear_clearance() -> None:
    compound = saturable(vmax=2.0, km=0.5)
    assert compound.is_saturable
    assert compound.linear_hepatic_clearance_L_per_h == pytest.approx(4.0)
    assert not linear().is_saturable


# -- saturation in a simulation -------------------------------------------


def test_saturable_compound_conserves_mass() -> None:
    result = simulate(saturable(), RAT, Regimen(IVBolus(5.0)), duration_h=48)
    assert result.mass_balance_error() < 1e-6


def test_a_tiny_dose_matches_the_equivalent_linear_compound() -> None:
    # Well below Km a saturable compound must behave exactly like a first-order
    # one with CL_int = Vmax / Km. This is the limit that anchors the model.
    dose = 1e-4
    sat = simulate(saturable(vmax=2.0, km=0.5), RAT, Regimen(IVBolus(dose)), duration_h=24)
    lin = simulate(linear(clearance=4.0), RAT, Regimen(IVBolus(dose)), duration_h=24)
    assert sat.auc("plasma") == pytest.approx(lin.auc("plasma"), rel=1e-3)


def test_exposure_grows_faster_than_dose_once_saturated() -> None:
    # The defining consequence of saturation, and the reason a linear model is
    # dangerous at high doses: double the dose, more than double the exposure.
    small = simulate(saturable(), RAT, Regimen(IVBolus(1.0)), duration_h=96)
    large = simulate(saturable(), RAT, Regimen(IVBolus(10.0)), duration_h=96)
    ratio = large.auc("plasma") / small.auc("plasma")
    assert ratio > 10.5, f"expected more than dose-proportional exposure, got {ratio:.2f}x"


def test_a_linear_compound_stays_exactly_proportional() -> None:
    small = simulate(linear(), RAT, Regimen(IVBolus(1.0)), duration_h=48)
    large = simulate(linear(), RAT, Regimen(IVBolus(10.0)), duration_h=48)
    assert large.auc("plasma") == pytest.approx(10 * small.auc("plasma"), rel=1e-6)


def test_saturation_cannot_remove_more_than_vmax() -> None:
    # Elimination is capped by the enzyme, so over a window no more than
    # Vmax * duration can be cleared however much is given.
    vmax = 2.0
    duration = 24.0
    result = simulate(
        saturable(vmax=vmax), RAT, Regimen(IVBolus(500.0)), duration_h=duration
    )
    assert result.total_eliminated()[-1] <= vmax * duration * 1.001


def test_saturable_metabolism_works_by_the_oral_route() -> None:
    regimen = Regimen(OralDose(4.0, absorption_rate_const_per_h=1.0))
    result = simulate(saturable(), RAT, regimen, duration_h=72)
    assert result.mass_balance_error() < 1e-6
    assert result.cmax("plasma") > 0


# -- interaction between compounds ----------------------------------------


def test_simulating_several_compounds_returns_a_study() -> None:
    study = simulate(
        [saturable("victim"), saturable("rival")], RAT, Regimen(IVBolus(1.0)), duration_h=24
    )
    assert isinstance(study, Study)
    assert study.names == ["victim", "rival"]
    assert isinstance(study["victim"], Result)
    assert isinstance(study[0], Result)
    assert len(study) == 2
    assert study.mass_balance_error() < 1e-6


def test_unknown_compound_name_is_reported_clearly() -> None:
    study = simulate([saturable("a"), saturable("b")], RAT, Regimen(IVBolus(1.0)), duration_h=4)
    with pytest.raises(KeyError, match="no compound named"):
        study["c"]


def test_compounds_on_different_enzymes_do_not_affect_each_other() -> None:
    # The sharpest check that coupling happens only where it should: a compound
    # cleared by another enzyme must give byte-for-byte the same profile as it
    # does alone, however much of it is present.
    victim = saturable("victim", vmax=2.0, km=0.5)
    bystander = Compound(
        name="bystander",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        metabolism=(Metabolism(enzyme=OTHER, vmax_mg_per_h=1.0, km_mg_per_L=0.2),),
    )

    alone = simulate(victim, RAT, Regimen(IVBolus(2.0)), duration_h=48)
    together = simulate(
        [victim, bystander],
        RAT,
        [Regimen(IVBolus(2.0)), Regimen(IVBolus(50.0))],
        duration_h=48,
    )

    solo = alone.plasma_concentration()
    paired = together["victim"].plasma_concentration()
    # Agreement is judged against the size of the profile. The two runs carry
    # different numbers of states, so the solver distributes its error budget
    # differently and the long tail -- where both are numerically zero --
    # differs in the last digits without differing in substance.
    assert np.allclose(paired, solo, rtol=1e-5, atol=1e-6 * solo.max())
    assert together["victim"].auc("plasma") == pytest.approx(alone.auc("plasma"), rel=1e-6)
    assert together.interacting_enzymes() == {}


def test_a_rival_for_the_same_enzyme_raises_exposure() -> None:
    # Two compounds competing for one enzyme each clear more slowly, so both
    # are more exposed than either would be alone. This is the interaction that
    # cannot be seen by simulating them separately.
    victim = saturable("victim", vmax=2.0, km=0.5)
    rival = saturable("rival", vmax=2.0, km=0.5)

    alone = simulate(victim, RAT, Regimen(IVBolus(2.0)), duration_h=72)
    together = simulate(
        [victim, rival],
        RAT,
        [Regimen(IVBolus(2.0)), Regimen(IVBolus(8.0))],
        duration_h=72,
    )
    assert together["victim"].auc("plasma") > alone.auc("plasma")
    assert together.interacting_enzymes() == {CYP: ["victim", "rival"]}


def test_a_pure_inhibitor_raises_exposure_without_being_metabolised() -> None:
    victim = saturable("victim", vmax=2.0, km=0.5)
    inhibitor = Compound(
        name="inhibitor",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        inhibits=(Inhibition(enzyme=CYP, ki_mg_per_L=0.05),),
    )

    alone = simulate(victim, RAT, Regimen(IVBolus(1.0)), duration_h=72)
    together = simulate(
        [victim, inhibitor],
        RAT,
        [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=72.0))],
        duration_h=72,
    )
    assert together["victim"].auc("plasma") > alone.auc("plasma")
    # The inhibitor is not a substrate, so nothing clears it hepatically.
    assert together["inhibitor"].amount("eliminated_hepatic")[-1] == pytest.approx(0.0)


def test_a_stronger_inhibitor_raises_exposure_further() -> None:
    victim = saturable("victim", vmax=2.0, km=0.5)

    def with_ki(ki: float) -> float:
        inhibitor = Compound(
            name="inhibitor",
            default_partition_coefficient=2.0,
            fraction_unbound_plasma=0.5,
            inhibits=(Inhibition(enzyme=CYP, ki_mg_per_L=ki),),
        )
        study = simulate(
            [victim, inhibitor],
            RAT,
            [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=72.0))],
            duration_h=72,
        )
        return study["victim"].auc("plasma")

    assert with_ki(0.02) > with_ki(0.5)


def test_overwhelming_inhibition_approaches_no_metabolism() -> None:
    # In the limit of total enzyme blockade the victim must behave like a
    # compound with no hepatic clearance at all.
    victim = saturable("victim", vmax=2.0, km=0.5)
    unmetabolised = Compound(
        name="victim", default_partition_coefficient=2.0, fraction_unbound_plasma=0.5
    )
    blocker = Compound(
        name="blocker",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        inhibits=(Inhibition(enzyme=CYP, ki_mg_per_L=1e-6),),
    )

    blocked = simulate(
        [victim, blocker],
        RAT,
        [Regimen(IVBolus(1.0)), Regimen(IVInfusion(100.0, duration_h=48.0))],
        duration_h=48,
    )
    free = simulate(unmetabolised, RAT, Regimen(IVBolus(1.0)), duration_h=48)
    assert blocked["victim"].auc("plasma") == pytest.approx(free.auc("plasma"), rel=0.02)


def test_mass_is_conserved_for_every_compound_in_a_study() -> None:
    study = simulate(
        [saturable("a"), saturable("b"), linear("c")],
        RAT,
        [
            Regimen(IVBolus(2.0)),
            Regimen(OralDose(3.0, absorption_rate_const_per_h=0.8)),
            Regimen(IVInfusion(1.0, duration_h=3.0)),
        ],
        duration_h=48,
    )
    for result in study:
        assert result.mass_balance_error() < 1e-6


def test_one_regimen_can_be_shared_by_every_compound() -> None:
    study = simulate(
        [saturable("a"), linear("b")], RAT, Regimen(IVBolus(1.0)), duration_h=24
    )
    assert study["a"].model.regimen.total_amount_mg == 1.0
    assert study["b"].model.regimen.total_amount_mg == 1.0


def test_study_summary_is_keyed_by_compound() -> None:
    study = simulate(
        [saturable("a"), linear("b")], RAT, Regimen(IVBolus(1.0)), duration_h=24
    )
    summary = study.summary()
    assert set(summary) == {"a", "b"}
    assert summary["a"]["cmax_mg_per_L"] > 0
