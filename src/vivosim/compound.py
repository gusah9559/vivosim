"""Compound-specific parameters: tissue partitioning and clearance.

A `Compound` holds everything about a molecule that a perfusion-limited PBPK
model needs, separately from the animal it is simulated in. The same compound
can therefore be run against any `Physiology`.

Partition coefficients (Kp) are the ratio of total tissue concentration to
total plasma concentration at equilibrium. They are normally either fitted to
tissue distribution data or predicted from physicochemistry; this module takes
them as given and only provides the bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = ["Compound"]


@dataclass(frozen=True)
class Compound:
    """Parameters describing how one compound distributes and is cleared.

    Attributes:
        name: Compound label, used for reporting.
        partition_coefficients: Tissue-to-plasma partition coefficients (Kp),
            keyed by tissue name. Tissues absent from this mapping fall back to
            `default_partition_coefficient`.
        default_partition_coefficient: Kp applied to any tissue without an
            explicit entry.
        blood_plasma_ratio: Whole blood to plasma concentration ratio. Converts
            the plasma-referenced Kp values onto the blood concentrations the
            circulation carries.
        fraction_unbound_plasma: Fraction of plasma drug not bound to protein.
            Only the unbound fraction is available to metabolising enzymes.
        hepatic_intrinsic_clearance_L_per_h: Intrinsic metabolic clearance in
            the liver, referenced to unbound concentration.
        renal_clearance_L_per_h: Clearance of unchanged drug by the kidney,
            referenced to the blood concentration leaving the kidney.
        molecular_weight: Molar mass in g/mol. Optional; used only to convert
            between mass and molar units when reporting.
    """

    name: str
    partition_coefficients: dict[str, float] = field(default_factory=dict)
    default_partition_coefficient: float = 1.0
    blood_plasma_ratio: float = 1.0
    fraction_unbound_plasma: float = 1.0
    hepatic_intrinsic_clearance_L_per_h: float = 0.0
    renal_clearance_L_per_h: float = 0.0
    molecular_weight: float | None = None

    def __post_init__(self) -> None:
        if self.default_partition_coefficient <= 0:
            raise ValueError(
                "default_partition_coefficient must be positive, "
                f"got {self.default_partition_coefficient}"
            )
        for tissue, kp in self.partition_coefficients.items():
            if kp <= 0:
                raise ValueError(f"partition coefficient for {tissue!r} must be positive, got {kp}")
        if self.blood_plasma_ratio <= 0:
            raise ValueError(f"blood_plasma_ratio must be positive, got {self.blood_plasma_ratio}")
        if not 0.0 < self.fraction_unbound_plasma <= 1.0:
            raise ValueError(
                "fraction_unbound_plasma must be in (0, 1], "
                f"got {self.fraction_unbound_plasma}"
            )
        if self.hepatic_intrinsic_clearance_L_per_h < 0:
            raise ValueError("hepatic_intrinsic_clearance_L_per_h must be non-negative")
        if self.renal_clearance_L_per_h < 0:
            raise ValueError("renal_clearance_L_per_h must be non-negative")
        if self.molecular_weight is not None and self.molecular_weight <= 0:
            raise ValueError(f"molecular_weight must be positive, got {self.molecular_weight}")

    def kp(self, tissue: str) -> float:
        """Tissue-to-plasma partition coefficient for one tissue."""
        return self.partition_coefficients.get(tissue, self.default_partition_coefficient)

    def kp_blood(self, tissue: str) -> float:
        """Tissue-to-blood partition coefficient for one tissue.

        The circulation carries whole blood, so the plasma-referenced Kp is
        divided by the blood:plasma ratio to give the ratio that governs
        exchange between tissue and perfusing blood.
        """
        return self.kp(tissue) / self.blood_plasma_ratio

    def replace(self, **changes: object) -> Compound:
        """A copy with individual fields overridden."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def with_partition_coefficients(self, **by_tissue: float) -> Compound:
        """A copy with specific tissue Kp values added or replaced."""
        merged = dict(self.partition_coefficients)
        merged.update(by_tissue)
        return replace(self, partition_coefficients=merged)
