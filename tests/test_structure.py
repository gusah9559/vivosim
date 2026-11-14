"""Reading molecular structures, and what they imply for distribution.

These tests need RDKit, which is an optional dependency, so they skip without
it. Molecules with well-known properties are used so that the computed
descriptors can be checked against reality rather than against the code.
"""

from __future__ import annotations

import pytest

from vivosim import RAT, Compound
from vivosim.partitioning import (
    compound_from_properties,
    predict_partition_coefficients,
    volume_of_distribution_L_per_kg,
)

pytest.importorskip("rdkit", reason="structure reading requires RDKit")

from vivosim.structure import describe_smiles, properties_from_smiles  # noqa: E402

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
DIAZEPAM = "CN1c2ccc(Cl)cc2C(=Nc3ccccc13)c4ccccc4"
PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"
BENZENE = "c1ccccc1"


# -- descriptors against reality ------------------------------------------


def test_molecular_weight_is_exact() -> None:
    # Molar mass is a property of the formula, so it must match the real value.
    assert describe_smiles(CAFFEINE).molecular_weight == pytest.approx(194.19, abs=0.05)
    assert describe_smiles(IBUPROFEN).molecular_weight == pytest.approx(206.28, abs=0.05)


def test_logp_ranks_molecules_correctly() -> None:
    # Crippen logP is an estimate, so test the ordering it gets right rather
    # than demanding accuracy it does not have: caffeine is water-soluble,
    # ibuprofen greasy, diazepam greasier still.
    caffeine = describe_smiles(CAFFEINE).logp
    ibuprofen = describe_smiles(IBUPROFEN).logp
    diazepam = describe_smiles(DIAZEPAM).logp
    assert caffeine < 1.0 < ibuprofen < diazepam


def test_canonical_smiles_is_returned() -> None:
    assert describe_smiles(BENZENE).smiles == "c1ccccc1"


def test_invalid_smiles_is_rejected() -> None:
    with pytest.raises(ValueError, match="could not parse SMILES"):
        describe_smiles("this is not a molecule")


# -- ionizable group detection --------------------------------------------


def test_carboxylic_acid_is_detected() -> None:
    descriptors = describe_smiles(IBUPROFEN)
    assert "carboxylic acid" in descriptors.acidic_groups
    assert descriptors.likely_ionization == "acid"
    assert descriptors.is_ionizable


def test_amine_is_detected_as_a_base() -> None:
    descriptors = describe_smiles(PROPRANOLOL)
    assert descriptors.basic_groups
    assert descriptors.likely_ionization == "base"


def test_neutral_molecule_has_no_ionizable_groups() -> None:
    descriptors = describe_smiles(BENZENE)
    assert not descriptors.is_ionizable
    assert descriptors.likely_ionization == "neutral"


def test_caffeine_is_treated_as_neutral() -> None:
    # Its nitrogens are amide- and aromatic-like, not basic amines, so it must
    # not be flagged as a base.
    assert describe_smiles(CAFFEINE).likely_ionization == "neutral"


# -- building properties ---------------------------------------------------


def test_ionizable_structure_demands_a_pka() -> None:
    # Silently defaulting an ionizable compound to neutral would distort every
    # partition coefficient, so it has to be refused.
    with pytest.raises(ValueError, match="pKa is not predicted from structure"):
        properties_from_smiles(IBUPROFEN, fraction_unbound_plasma=0.01)


def test_ionization_can_be_overridden_deliberately() -> None:
    properties = properties_from_smiles(
        IBUPROFEN, fraction_unbound_plasma=0.01, ionization="neutral"
    )
    assert properties.ionization == "neutral"


def test_measured_logp_overrides_the_computed_one() -> None:
    properties = properties_from_smiles(
        IBUPROFEN, fraction_unbound_plasma=0.01, pka=4.4, logp=3.97
    )
    assert properties.logp == 3.97


def test_neutral_structure_needs_no_pka() -> None:
    properties = properties_from_smiles(CAFFEINE, fraction_unbound_plasma=0.7)
    assert properties.pka is None
    assert properties.molecular_weight == pytest.approx(194.19, abs=0.05)


def test_name_defaults_to_the_canonical_smiles() -> None:
    assert properties_from_smiles(BENZENE, fraction_unbound_plasma=0.5).name == "c1ccccc1"


# -- distribution behaviour -----------------------------------------------


def test_ibuprofen_is_predicted_to_stay_in_plasma() -> None:
    # Ibuprofen is an acid that is ionized at pH 7.4 and about 99% bound to
    # plasma protein, and its measured volume of distribution is accordingly
    # tiny -- roughly 0.15 L/kg, close to plasma volume. A partitioning method
    # that did not reproduce a small Vss here would be unusable for acids.
    properties = properties_from_smiles(
        IBUPROFEN, fraction_unbound_plasma=0.01, pka=4.4, logp=3.97, name="ibuprofen"
    )
    compound = compound_from_properties(properties)
    vss = volume_of_distribution_L_per_kg(compound, RAT)
    assert vss < 1.0, f"expected a plasma-confined compound, got Vss {vss:.2f} L/kg"


def test_diazepam_is_predicted_to_distribute_into_tissue() -> None:
    # Diazepam is lipophilic and neutral, with a measured Vss around 1-2 L/kg,
    # well above total body water -- it must come out larger than ibuprofen.
    diazepam = compound_from_properties(
        properties_from_smiles(DIAZEPAM, fraction_unbound_plasma=0.02, logp=2.82)
    )
    ibuprofen = compound_from_properties(
        properties_from_smiles(
            IBUPROFEN, fraction_unbound_plasma=0.01, pka=4.4, logp=3.97
        )
    )
    assert volume_of_distribution_L_per_kg(diazepam, RAT) > volume_of_distribution_L_per_kg(
        ibuprofen, RAT
    )


def test_ionization_state_changes_the_prediction() -> None:
    # The same structure, treated as ionized or not, must partition
    # differently -- this is what makes the chemistry layer more than a
    # lipophilicity lookup.
    acid = properties_from_smiles(IBUPROFEN, fraction_unbound_plasma=0.01, pka=4.4)
    ignored = properties_from_smiles(
        IBUPROFEN, fraction_unbound_plasma=0.01, ionization="neutral"
    )
    assert (
        predict_partition_coefficients(acid)["muscle"]
        < predict_partition_coefficients(ignored)["muscle"]
    )


def test_volume_of_distribution_reflects_partitioning() -> None:
    confined = Compound(name="confined", default_partition_coefficient=0.1)
    spread = Compound(name="spread", default_partition_coefficient=20.0)
    assert volume_of_distribution_L_per_kg(confined, RAT) < volume_of_distribution_L_per_kg(
        spread, RAT
    )


def test_volume_of_distribution_is_weight_independent() -> None:
    # Vss per kg is a property of the compound, not of how big the animal is.
    compound = Compound(name="example", default_partition_coefficient=3.0)
    light = volume_of_distribution_L_per_kg(compound, RAT.at_weight(0.2))
    heavy = volume_of_distribution_L_per_kg(compound, RAT.at_weight(0.4))
    assert light == pytest.approx(heavy, rel=1e-9)
