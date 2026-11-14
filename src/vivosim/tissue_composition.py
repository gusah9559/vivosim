"""Tissue composition: the lipid and water content that drives partitioning.

A compound distributes into a tissue because it dissolves in that tissue's
lipid and water. So if you know how much neutral lipid, phospholipid and water
a tissue contains, and how lipophilic the compound is, the partition
coefficient follows from physical chemistry rather than from fitting.

That is what this table is for. It is the bridge between molecular structure
and the `Kp` values the PBPK model needs.

Fractional volumes are from the compilations used by the mechanistic
partitioning methods:

    Poulin P, Theil FP (2002). "Prediction of pharmacokinetics prior to in vivo
    studies. 1. Mechanism-based prediction of volume of distribution."
    Journal of Pharmaceutical Sciences 91(1):129-156.

    Rodgers T, Rowland M (2006). "Physiologically based pharmacokinetic
    modelling 2: predicting the tissue distribution of acids, very weak bases,
    neutrals and zwitterions." Journal of Pharmaceutical Sciences 95(6):1238-1257.

These are literature means transcribed for convenience. They are adequate for
exploring how structure drives distribution, but check them against the primary
tables before relying on a specific number, and prefer species- and
strain-specific composition where you have it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TissueComposition", "RAT_TISSUE_COMPOSITION", "PLASMA_COMPOSITION"]


@dataclass(frozen=True)
class TissueComposition:
    """Fractional volumes of the phases a compound can dissolve in.

    The three fractions need not sum to 1; the remainder is protein and solid
    matter, which the partitioning methods treat as inaccessible.

    Attributes:
        neutral_lipid: Volume fraction of neutral lipid (triglycerides). The
            dominant phase for lipophilic compounds, and why adipose tissue
            accumulates them.
        phospholipid: Volume fraction of phospholipid. Partly polar, so it is
            treated as 30% lipid-like and 70% water-like.
        water: Volume fraction of water, intracellular and extracellular
            combined.
    """

    neutral_lipid: float
    phospholipid: float
    water: float

    def __post_init__(self) -> None:
        for name in ("neutral_lipid", "phospholipid", "water"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1], got {value}")
        total = self.neutral_lipid + self.phospholipid + self.water
        if total > 1.0 + 1e-9:
            raise ValueError(f"phase fractions sum to {total:.3f}, which exceeds 1")

    @property
    def lipid_equivalent(self) -> float:
        """Effective lipid volume: neutral lipid plus 30% of phospholipid."""
        return self.neutral_lipid + 0.3 * self.phospholipid

    @property
    def water_equivalent(self) -> float:
        """Effective aqueous volume: water plus 70% of phospholipid."""
        return self.water + 0.7 * self.phospholipid


# Composition of rat tissues. Keys match the tissue names in
# `vivosim.physiology.RAT`, so a predicted Kp set drops straight into a model.
RAT_TISSUE_COMPOSITION: dict[str, TissueComposition] = {
    "adipose": TissueComposition(neutral_lipid=0.853, phospholipid=0.002, water=0.144),
    "bone": TissueComposition(neutral_lipid=0.0273, phospholipid=0.0027, water=0.439),
    "brain": TissueComposition(neutral_lipid=0.0392, phospholipid=0.0053, water=0.753),
    "gut": TissueComposition(neutral_lipid=0.0375, phospholipid=0.0124, water=0.718),
    "heart": TissueComposition(neutral_lipid=0.0115, phospholipid=0.0166, water=0.758),
    "kidney": TissueComposition(neutral_lipid=0.0207, phospholipid=0.0162, water=0.783),
    "liver": TissueComposition(neutral_lipid=0.0348, phospholipid=0.0252, water=0.751),
    "lung": TissueComposition(neutral_lipid=0.0219, phospholipid=0.0140, water=0.790),
    "muscle": TissueComposition(neutral_lipid=0.0238, phospholipid=0.0072, water=0.760),
    "skin": TissueComposition(neutral_lipid=0.0284, phospholipid=0.0111, water=0.651),
    "spleen": TissueComposition(neutral_lipid=0.0201, phospholipid=0.0198, water=0.788),
    # The lumped remainder compartment has no measured composition of its own.
    # Muscle is the closest single proxy, being the largest contributor to the
    # tissues it stands in for. Override it if that matters for your compound.
    "rest": TissueComposition(neutral_lipid=0.0238, phospholipid=0.0072, water=0.760),
}

# Plasma is the reference phase: every Kp is a ratio against it.
PLASMA_COMPOSITION = TissueComposition(
    neutral_lipid=0.0023, phospholipid=0.0013, water=0.960
)
