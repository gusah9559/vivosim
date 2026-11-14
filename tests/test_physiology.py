"""Physiological parameter sets must describe a closed, consistent animal."""

from __future__ import annotations

import pytest

from vivosim import MOUSE, RAT, SPECIES, Physiology, TissueSpec, get_physiology


@pytest.mark.parametrize("phys", list(SPECIES.values()), ids=list(SPECIES))
def test_flows_sum_to_cardiac_output(phys: Physiology) -> None:
    # Every unit of blood the heart ejects must come back, or the model leaks.
    assert abs(phys.flow_balance_error()) < 1e-9


@pytest.mark.parametrize("phys", list(SPECIES.values()), ids=list(SPECIES))
def test_volumes_sum_to_body_weight(phys: Physiology) -> None:
    assert abs(phys.volume_balance_error()) < 1e-9


@pytest.mark.parametrize("phys", list(SPECIES.values()), ids=list(SPECIES))
def test_liver_outflow_exceeds_its_arterial_supply(phys: Physiology) -> None:
    # The liver's outflow carries the portal tissues' flow as well as its own.
    assert phys.liver_inflow_L_per_h() > phys.flow_L_per_h(phys.liver_name)


def test_cardiac_output_scales_allometrically() -> None:
    light = RAT.at_weight(0.2)
    heavy = RAT.at_weight(0.4)
    ratio = heavy.cardiac_output_L_per_h / light.cardiac_output_L_per_h
    assert ratio == pytest.approx(2.0**0.75)


def test_volumes_scale_linearly_with_weight() -> None:
    assert RAT.at_weight(0.5).volume_L("liver") == pytest.approx(2 * RAT.volume_L("liver"))


def test_mouse_is_smaller_than_rat() -> None:
    assert MOUSE.body_weight_kg < RAT.body_weight_kg
    assert MOUSE.cardiac_output_L_per_h < RAT.cardiac_output_L_per_h


def test_get_physiology_is_case_insensitive() -> None:
    assert get_physiology("RAT").species == "rat"


def test_get_physiology_rejects_unknown_species() -> None:
    with pytest.raises(KeyError, match="no physiology for species"):
        get_physiology("elephant")


def test_get_physiology_applies_weight() -> None:
    assert get_physiology("rat", 0.3).body_weight_kg == 0.3


def test_with_tissue_does_not_mutate_the_original() -> None:
    extended = RAT.with_tissue("tumour", TissueSpec(0.01, 0.005))
    assert "tumour" in extended.tissues
    assert "tumour" not in RAT.tissues


def test_tissue_spec_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="volume_fraction must be positive"):
        TissueSpec(volume_fraction=0.0, flow_fraction=0.1)
    with pytest.raises(ValueError, match="flow_fraction must be non-negative"):
        TissueSpec(volume_fraction=0.1, flow_fraction=-0.1)
    with pytest.raises(ValueError, match="drains_to must be"):
        TissueSpec(volume_fraction=0.1, flow_fraction=0.1, drains_to="spleen")


def test_physiology_requires_lung_and_liver() -> None:
    with pytest.raises(ValueError, match="missing the required"):
        RAT.replace(tissues={"muscle": TissueSpec(0.4, 1.0)})


def test_physiology_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="body_weight_kg must be positive"):
        RAT.at_weight(0.0)
