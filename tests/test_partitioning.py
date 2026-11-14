"""Predicted partition coefficients must behave the way chemistry demands."""

from __future__ import annotations

import pytest

from vivosim import RAT, IVBolus, Regimen, simulate
from vivosim.partitioning import (
    MolecularProperties,
    apparent_logd,
    compound_from_properties,
    predict_partition_coefficients,
)
from vivosim.tissue_composition import (
    PLASMA_COMPOSITION,
    RAT_TISSUE_COMPOSITION,
    TissueComposition,
)


def props(logp: float, **kwargs) -> MolecularProperties:
    base = {"name": "test", "logp": logp, "fraction_unbound_plasma": 0.1}
    return MolecularProperties(**{**base, **kwargs})


# -- composition table ----------------------------------------------------


def test_every_rat_tissue_has_a_composition() -> None:
    # A missing composition silently falls back to Kp = 1, which would look
    # like a prediction rather than a gap.
    assert set(RAT_TISSUE_COMPOSITION) == set(RAT.tissues)


def test_adipose_is_the_most_lipid_rich_tissue() -> None:
    richest = max(RAT_TISSUE_COMPOSITION, key=lambda t: RAT_TISSUE_COMPOSITION[t].lipid_equivalent)
    assert richest == "adipose"


def test_phase_fractions_cannot_exceed_one() -> None:
    with pytest.raises(ValueError, match="exceeds 1"):
        TissueComposition(neutral_lipid=0.6, phospholipid=0.3, water=0.3)


def test_lipid_and_water_equivalents_split_phospholipid() -> None:
    c = TissueComposition(neutral_lipid=0.1, phospholipid=0.2, water=0.5)
    assert c.lipid_equivalent == pytest.approx(0.1 + 0.3 * 0.2)
    assert c.water_equivalent == pytest.approx(0.5 + 0.7 * 0.2)


# -- ionization correction ------------------------------------------------


def test_neutral_compound_keeps_its_logp() -> None:
    assert apparent_logd(3.0) == 3.0


def test_base_below_its_pka_loses_lipophilicity() -> None:
    # A base with pKa well above pH 7.4 is almost entirely protonated, so very
    # little of it is available to partition into lipid.
    assert apparent_logd(3.0, "base", pka=10.4) == pytest.approx(0.0, abs=0.01)


def test_acid_above_its_pka_loses_lipophilicity() -> None:
    assert apparent_logd(3.0, "acid", pka=4.4) == pytest.approx(0.0, abs=0.01)


def test_ionization_at_the_pka_halves_the_neutral_fraction() -> None:
    # At pH == pKa the compound is half ionized, costing log10(2).
    assert apparent_logd(3.0, "base", pka=7.4) == pytest.approx(3.0 - 0.3010, abs=1e-3)


def test_weakly_ionized_compound_is_barely_affected() -> None:
    assert apparent_logd(3.0, "base", pka=3.0) == pytest.approx(3.0, abs=1e-3)


def test_strong_ionization_does_not_overflow() -> None:
    # 10 ** (pKa - pH) would overflow a float here; the correction must not.
    value = apparent_logd(3.0, "base", pka=400.0)
    assert value < -300


def test_pka_is_required_for_ionizable_compounds() -> None:
    with pytest.raises(ValueError, match="pka is required"):
        apparent_logd(3.0, "base")
    with pytest.raises(ValueError, match="pka is required"):
        props(3.0, ionization="acid")


# -- predicted partitioning -----------------------------------------------


def test_lipophilic_compound_concentrates_in_adipose() -> None:
    kp = predict_partition_coefficients(props(4.0))
    # Adipose must come out ahead of every other tissue, by a wide margin.
    assert max(kp, key=lambda t: kp[t]) == "adipose"
    # Liver carries more phospholipid than muscle, so it trails adipose least.
    assert kp["adipose"] > 2.5 * kp["liver"]
    assert kp["adipose"] > 4 * kp["muscle"]


def test_plasma_binding_suppresses_adipose_uptake() -> None:
    # In this method adipose scales with fu_p while other tissues scale with
    # fu_p/fu_t, so heavy plasma binding narrows the adipose advantage. The
    # effect is real but it is a modelling assumption worth seeing explicitly.
    unbound = predict_partition_coefficients(props(4.0, fraction_unbound_plasma=1.0))
    bound = predict_partition_coefficients(props(4.0, fraction_unbound_plasma=0.1))
    assert unbound["adipose"] / unbound["muscle"] > bound["adipose"] / bound["muscle"]


def test_hydrophilic_compound_does_not_concentrate_anywhere() -> None:
    kp = predict_partition_coefficients(props(-1.0))
    # Nothing should accumulate: every tissue stays at or below plasma.
    assert all(value < 2.0 for value in kp.values()), kp


def test_partitioning_increases_monotonically_with_lipophilicity() -> None:
    adipose = [predict_partition_coefficients(props(p))["adipose"] for p in (0.0, 2.0, 4.0, 6.0)]
    assert adipose == sorted(adipose)
    assert adipose[-1] > adipose[0]


def test_ionization_reduces_predicted_partitioning() -> None:
    neutral = predict_partition_coefficients(props(3.0))
    base = predict_partition_coefficients(props(3.0, ionization="base", pka=10.0))
    assert base["adipose"] < neutral["adipose"]
    assert base["muscle"] < neutral["muscle"]


def test_tighter_plasma_binding_reduces_tissue_partitioning() -> None:
    # Compound held by plasma protein is not free to enter tissue, and adipose
    # scales directly with the unbound fraction.
    loose = predict_partition_coefficients(props(3.0, fraction_unbound_plasma=0.9))
    tight = predict_partition_coefficients(props(3.0, fraction_unbound_plasma=0.01))
    assert tight["adipose"] < loose["adipose"]


def test_a_tissue_identical_to_plasma_gives_kp_near_one() -> None:
    # Same composition as the reference phase, so the only departure from 1 is
    # the tissue-versus-plasma binding correction.
    kp = predict_partition_coefficients(
        props(2.0, fraction_unbound_plasma=1.0), {"clone": PLASMA_COMPOSITION}
    )
    assert kp["clone"] == pytest.approx(1.0, rel=1e-9)


def test_measured_olive_oil_coefficient_is_used_when_given() -> None:
    estimated = predict_partition_coefficients(props(3.0))
    measured = predict_partition_coefficients(props(3.0, olive_oil_water_logd=1.0))
    assert measured["adipose"] != pytest.approx(estimated["adipose"])
    # Only adipose uses the olive oil coefficient; the rest must be untouched.
    assert measured["muscle"] == pytest.approx(estimated["muscle"])


# -- end to end -----------------------------------------------------------


def test_predicted_compound_simulates_and_conserves_mass() -> None:
    compound = compound_from_properties(
        props(3.5, ionization="base", pka=8.0, blood_plasma_ratio=1.2),
        hepatic_intrinsic_clearance_L_per_h=1.0,
    )
    result = simulate(compound, RAT, Regimen(IVBolus(1.0)), duration_h=48)
    assert result.mass_balance_error() < 1e-6
    assert result.cmax("plasma") > 0


def test_lipophilic_compound_lingers_longer_than_hydrophilic() -> None:
    # The point of predicting distribution: a fat-seeking molecule has a large
    # reservoir to come back out of, so its exposure outlasts a polar one at
    # the same clearance.
    kwargs = {"hepatic_intrinsic_clearance_L_per_h": 2.0}
    fatty = compound_from_properties(props(5.0), **kwargs)
    polar = compound_from_properties(props(-0.5), **kwargs)

    fatty_result = simulate(fatty, RAT, Regimen(IVBolus(1.0)), duration_h=72)
    polar_result = simulate(polar, RAT, Regimen(IVBolus(1.0)), duration_h=72)

    remaining_fatty = fatty_result.total_in_body()[-1]
    remaining_polar = polar_result.total_in_body()[-1]
    assert remaining_fatty > remaining_polar


def test_compound_from_properties_carries_measurements_through() -> None:
    compound = compound_from_properties(
        props(2.0, blood_plasma_ratio=1.3, molecular_weight=250.0),
        renal_clearance_L_per_h=0.4,
    )
    assert compound.blood_plasma_ratio == 1.3
    assert compound.molecular_weight == 250.0
    assert compound.renal_clearance_L_per_h == 0.4
    assert compound.fraction_unbound_plasma == 0.1
    # Clearance is never predicted from structure, so an unset one stays zero.
    assert compound.hepatic_intrinsic_clearance_L_per_h == 0.0
