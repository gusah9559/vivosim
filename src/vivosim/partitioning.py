"""Predicting tissue partition coefficients from molecular properties.

This is the layer that connects chemistry to the PBPK model. Instead of
supplying a `Kp` per tissue by hand, you describe the molecule -- how
lipophilic it is, whether it ionizes, how strongly it binds plasma protein --
and the partition coefficients are computed from the lipid and water content of
each tissue.

Method
------
`predict_partition_coefficients` implements the mechanistic tissue-composition
approach of:

    Poulin P, Theil FP (2002). "Prediction of pharmacokinetics prior to in vivo
    studies. 1. Mechanism-based prediction of volume of distribution."
    Journal of Pharmaceutical Sciences 91(1):129-156.

The compound is assumed to distribute by dissolving in the lipid and water of
each tissue, so for a non-adipose tissue::

    Kp = [D * Lip_t + Aq_t] / [D * Lip_p + Aq_p] * fu_p / fu_t

where ``Lip`` and ``Aq`` are the lipid- and water-equivalent volumes of tissue
and plasma, ``D`` the lipophilicity at physiological pH, and ``fu`` the unbound
fractions. Adipose tissue is handled separately, because partitioning into bulk
triglyceride is better described by an olive-oil/water coefficient than by
octanol/water, and unbound fraction in adipose is taken as 1.

What this does and does not buy you
-----------------------------------
What it predicts is *distribution*: where the compound goes and therefore how
long it stays. That is genuinely useful -- it is the difference between a
compound that clears in an hour and one that sits in fat for a week, and it
falls out of structure alone.

What it does not predict is *effect*. Nothing here says whether a compound
works, or what it does to the animal. Those need measured data. Treat predicted
Kp values as a structured starting hypothesis to be replaced by measurements,
not as an answer: the method is known to do poorly on compounds that are
actively transported, that bind tissue components specifically, or that are
strong bases accumulating in acidic phospholipid, for which the Rodgers &
Rowland extension is the better choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .compound import Compound
from .physiology import Physiology
from .tissue_composition import (
    PLASMA_COMPOSITION,
    RAT_TISSUE_COMPOSITION,
    TissueComposition,
)

__all__ = [
    "MolecularProperties",
    "apparent_logd",
    "compound_from_properties",
    "predict_partition_coefficients",
    "volume_of_distribution_L_per_kg",
]

PHYSIOLOGICAL_PH = 7.4

IonizationClass = Literal["neutral", "acid", "base"]


@dataclass(frozen=True)
class MolecularProperties:
    """The molecular properties that determine tissue partitioning.

    Attributes:
        name: Compound label.
        logp: Octanol/water partition coefficient of the neutral species,
            log10. Either measured or computed from structure; see
            `vivosim.structure.properties_from_smiles`.
        ionization: Whether the compound is neutral, an acid or a base at
            physiological pH. Determines how `pka` is applied.
        pka: Acid dissociation constant. Required unless the compound is
            neutral; ignored when it is.
        fraction_unbound_plasma: Fraction of plasma compound not bound to
            protein. Measured, typically by ultrafiltration -- it is not
            reliably predictable from structure, so it is required input.
        blood_plasma_ratio: Whole blood to plasma concentration ratio.
        molecular_weight: Molar mass in g/mol, carried through for reporting.
        olive_oil_water_logd: Olive-oil/water distribution coefficient, log10,
            used for adipose tissue. When omitted it is estimated from `logp`
            by the regression Poulin & Theil fitted for that purpose, which is
            an approximation -- supply a measured value when you have one.
    """

    name: str
    logp: float
    fraction_unbound_plasma: float
    ionization: IonizationClass = "neutral"
    pka: float | None = None
    blood_plasma_ratio: float = 1.0
    molecular_weight: float | None = None
    olive_oil_water_logd: float | None = None

    def __post_init__(self) -> None:
        if self.ionization not in ("neutral", "acid", "base"):
            raise ValueError(
                f"ionization must be 'neutral', 'acid' or 'base', got {self.ionization!r}"
            )
        if self.ionization != "neutral" and self.pka is None:
            raise ValueError(f"pka is required for a {self.ionization}")
        if not 0.0 < self.fraction_unbound_plasma <= 1.0:
            raise ValueError(
                "fraction_unbound_plasma must be in (0, 1], "
                f"got {self.fraction_unbound_plasma}"
            )
        if self.blood_plasma_ratio <= 0:
            raise ValueError(f"blood_plasma_ratio must be positive, got {self.blood_plasma_ratio}")


def apparent_logd(
    logp: float,
    ionization: IonizationClass = "neutral",
    pka: float | None = None,
    ph: float = PHYSIOLOGICAL_PH,
) -> float:
    """Lipophilicity at a given pH, corrected for ionization.

    Only the neutral form of a molecule partitions appreciably into lipid, so
    an ionizable compound is effectively less lipophilic than its logP
    suggests. The correction is the fraction of molecules that are un-ionized::

        acid: logD = logP - log10(1 + 10^(pH - pKa))
        base: logD = logP - log10(1 + 10^(pKa - pH))

    An acid above its pKa, or a base below it, is mostly ionized and so loses
    lipophilicity.

    Args:
        logp: Octanol/water partition coefficient of the neutral species.
        ionization: Whether the compound is neutral, an acid or a base.
        pka: Acid dissociation constant. Required unless neutral.
        ph: The pH to evaluate at.

    Returns:
        log10 of the apparent distribution coefficient.

    Raises:
        ValueError: If a pKa is needed but not supplied.
    """
    if ionization == "neutral":
        return logp
    if pka is None:
        raise ValueError(f"pka is required for a {ionization}")
    exponent = (ph - pka) if ionization == "acid" else (pka - ph)
    # log1p of a power, written to stay finite for strongly ionized compounds
    # where 10**exponent would overflow.
    ionized_term = (
        exponent * math.log(10.0) if exponent > 300 else math.log1p(10.0**exponent)
    )
    return logp - ionized_term / math.log(10.0)


def predict_partition_coefficients(
    properties: MolecularProperties,
    tissues: dict[str, TissueComposition] | None = None,
    *,
    plasma: TissueComposition = PLASMA_COMPOSITION,
    ph: float = PHYSIOLOGICAL_PH,
) -> dict[str, float]:
    """Tissue-to-plasma partition coefficients predicted from structure.

    Args:
        properties: The molecule's lipophilicity, ionization and plasma
            binding.
        tissues: Tissue compositions to predict for. Defaults to the rat table,
            whose keys match `vivosim.physiology.RAT`.
        plasma: Composition of the reference plasma phase.
        ph: pH at which to evaluate ionization.

    Returns:
        A mapping of tissue name to predicted Kp, ready to pass to `Compound`.
    """
    table = RAT_TISSUE_COMPOSITION if tissues is None else tissues

    logd = apparent_logd(properties.logp, properties.ionization, properties.pka, ph)
    d_octanol = 10.0**logd

    fu_p = properties.fraction_unbound_plasma
    # Poulin & Theil's assumption for unbound fraction in tissue: tissue
    # interstitial fluid holds about half the binding protein concentration of
    # plasma, so binding is correspondingly weaker.
    fu_tissue = 1.0 / (1.0 + ((1.0 - fu_p) / fu_p) * 0.5)

    plasma_denominator = d_octanol * plasma.lipid_equivalent + plasma.water_equivalent

    if properties.olive_oil_water_logd is not None:
        d_olive = 10.0**properties.olive_oil_water_logd
    else:
        d_olive = 10.0 ** _estimate_olive_oil_logd(logd)
    plasma_denominator_adipose = (
        d_olive * plasma.lipid_equivalent + plasma.water_equivalent
    )

    predicted: dict[str, float] = {}
    for name, composition in table.items():
        if name == "adipose":
            # Bulk triglyceride, and no appreciable protein binding to correct
            # for, so the unbound fraction in adipose is taken as 1.
            numerator = d_olive * composition.lipid_equivalent + composition.water_equivalent
            predicted[name] = (numerator / plasma_denominator_adipose) * fu_p
        else:
            numerator = d_octanol * composition.lipid_equivalent + composition.water_equivalent
            predicted[name] = (numerator / plasma_denominator) * (fu_p / fu_tissue)
    return predicted


def compound_from_properties(
    properties: MolecularProperties,
    *,
    tissues: dict[str, TissueComposition] | None = None,
    hepatic_intrinsic_clearance_L_per_h: float = 0.0,
    renal_clearance_L_per_h: float = 0.0,
    ph: float = PHYSIOLOGICAL_PH,
) -> Compound:
    """A simulation-ready `Compound` built from molecular properties.

    Partition coefficients are predicted; clearance is not. Metabolic clearance
    depends on which enzymes act on the molecule and how fast, which structure
    alone does not determine, so it is passed in -- normally scaled from a
    liver microsome or hepatocyte incubation.

    Args:
        properties: The molecule's properties.
        tissues: Tissue compositions to predict against.
        hepatic_intrinsic_clearance_L_per_h: Measured or scaled intrinsic
            hepatic clearance, referenced to unbound concentration.
        renal_clearance_L_per_h: Renal clearance of unchanged compound.
        ph: pH at which to evaluate ionization.

    Returns:
        A `Compound` carrying the predicted Kp values.
    """
    kp = predict_partition_coefficients(properties, tissues, ph=ph)
    return Compound(
        name=properties.name,
        partition_coefficients=kp,
        # Any tissue absent from the composition table falls back to this;
        # water-like partitioning is the conservative default.
        default_partition_coefficient=1.0,
        blood_plasma_ratio=properties.blood_plasma_ratio,
        fraction_unbound_plasma=properties.fraction_unbound_plasma,
        hepatic_intrinsic_clearance_L_per_h=hepatic_intrinsic_clearance_L_per_h,
        renal_clearance_L_per_h=renal_clearance_L_per_h,
        molecular_weight=properties.molecular_weight,
    )


def volume_of_distribution_L_per_kg(
    compound: Compound, physiology: Physiology
) -> float:
    """Steady-state volume of distribution implied by a compound's Kp values.

    Vss is the headline distribution number: the apparent volume the compound
    would need to occupy, at the plasma concentration measured, to account for
    all of it in the body. A value near plasma volume means the compound stays
    in the blood; a large one means it is hiding in tissue.

    It is the sum of the plasma volume and every tissue volume weighted by how
    strongly that tissue concentrates the compound::

        Vss = V_plasma + sum_t (V_t * Kp_t)

    Because it follows from the Kp values alone, it is the cheapest check on a
    predicted partitioning set: compare it against a measured Vss before
    trusting any simulation built on those predictions.

    Args:
        compound: The compound, carrying its partition coefficients.
        physiology: The animal whose tissue volumes to weight by.

    Returns:
        Steady-state volume of distribution in L/kg body weight.
    """
    total = physiology.plasma_volume_L
    total += sum(
        physiology.volume_L(tissue) * compound.kp(tissue) for tissue in physiology.tissues
    )
    return total / physiology.body_weight_kg


def _estimate_olive_oil_logd(logd_octanol: float) -> float:
    """Olive-oil/water logD estimated from octanol/water logD.

    Poulin & Theil fitted this relation to let the adipose calculation proceed
    when only octanol data is available. It is a regression across a chemically
    diverse set, so it carries real scatter; a measured olive-oil coefficient
    is always preferable for a compound whose adipose distribution matters.
    """
    return 1.115 * logd_octanol - 1.35
