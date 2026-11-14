"""A library of compounds to pick from.

Most of these are **archetypes**, not real drugs. They are named for the
behaviour they demonstrate -- lipophilic base, polar acid, high-extraction --
because the point of a picker is to let someone see how a *kind* of molecule
behaves, and because putting invented numbers under a real drug's name would
be a lie that looks like data.

The one exception is theophylline, whose parameters are literature values and
which the package validates against a published study. It is marked as such.

Every entry carries a note saying what it is for and what makes it behave the
way it does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compound import Compound
from .kinetics import Inhibition, Metabolism

__all__ = ["LibraryEntry", "LIBRARY", "get_compound", "names"]


@dataclass(frozen=True)
class LibraryEntry:
    """One pickable compound, with an explanation of why it behaves as it does.

    Attributes:
        key: Stable identifier used by the interface.
        label: Display name.
        note: What this compound demonstrates.
        measured: Whether the parameters are literature values for a real
            compound, or invented to illustrate a behaviour.
        compound: The parameters themselves.
        suggested_dose_mg_per_kg: A dose that puts the compound in an
            interesting range for the default species.
    """

    key: str
    label: str
    note: str
    measured: bool
    compound: Compound
    suggested_dose_mg_per_kg: float = 4.0


def _entry(
    key: str,
    label: str,
    note: str,
    compound: Compound,
    *,
    measured: bool = False,
    dose: float = 4.0,
) -> LibraryEntry:
    return LibraryEntry(
        key=key,
        label=label,
        note=note,
        measured=measured,
        compound=compound,
        suggested_dose_mg_per_kg=dose,
    )


LIBRARY: dict[str, LibraryEntry] = {
    entry.key: entry
    for entry in (
        _entry(
            "theophylline",
            "Theophylline",
            "Literature parameters for a real drug. The package predicts a "
            "published 12-subject study with these and lands 92% of "
            "measurements within two-fold.",
            Compound(
                name="Theophylline",
                default_partition_coefficient=0.70,
                blood_plasma_ratio=0.82,
                fraction_unbound_plasma=0.60,
                hepatic_intrinsic_clearance_L_per_h=2.8 / 0.60,
                molecular_weight=180.16,
            ),
            measured=True,
            dose=4.0,
        ),
        _entry(
            "lipophilic",
            "Lipophilic neutral",
            "Greasy and uncharged, so it floods into fat and leaves slowly. "
            "Watch adipose fill in the anatomy view and stay full long after "
            "plasma has cleared.",
            Compound(
                name="Lipophilic neutral",
                partition_coefficients={
                    "adipose": 55.0, "liver": 6.0, "muscle": 4.0, "brain": 3.0
                },
                default_partition_coefficient=3.0,
                blood_plasma_ratio=1.0,
                fraction_unbound_plasma=0.1,
                hepatic_intrinsic_clearance_L_per_h=4.0,
            ),
            dose=4.0,
        ),
        _entry(
            "polar_acid",
            "Polar acid",
            "Charged at body pH and heavily protein-bound, so it stays in the "
            "blood. Almost nothing reaches tissue -- the opposite extreme.",
            Compound(
                name="Polar acid",
                default_partition_coefficient=0.35,
                blood_plasma_ratio=0.9,
                fraction_unbound_plasma=0.02,
                hepatic_intrinsic_clearance_L_per_h=1.0,
                renal_clearance_L_per_h=0.15,
            ),
            dose=4.0,
        ),
        _entry(
            "renal",
            "Renally cleared",
            "Leaves unchanged in urine and is barely touched by the liver. "
            "Try it with renal impairment, then with hepatic impairment, and "
            "see which one matters.",
            Compound(
                name="Renally cleared",
                default_partition_coefficient=0.8,
                fraction_unbound_plasma=0.7,
                renal_clearance_L_per_h=0.4,
            ),
            dose=4.0,
        ),
        _entry(
            "high_extraction",
            "High extraction",
            "Cleared as fast as blood can deliver it to the liver, so its fate "
            "is set by blood flow rather than by enzyme. Losing enzyme barely "
            "touches it; losing liver blood flow does.",
            Compound(
                name="High extraction",
                default_partition_coefficient=2.0,
                fraction_unbound_plasma=0.8,
                hepatic_intrinsic_clearance_L_per_h=300.0,
            ),
            dose=4.0,
        ),
        _entry(
            "saturable",
            "Saturable",
            "Metabolised by a finite amount of enzyme. Double the dose and "
            "exposure more than doubles, because the enzyme is already full.",
            Compound(
                name="Saturable",
                default_partition_coefficient=2.0,
                fraction_unbound_plasma=0.5,
                metabolism=(
                    Metabolism(enzyme="CYP3A4", vmax_mg_per_h=2.0, km_mg_per_L=0.5),
                ),
            ),
            dose=8.0,
        ),
        _entry(
            "victim",
            "Interaction victim",
            "Cleared by CYP3A4 and nothing else. Harmless alone; add the "
            "inhibitor beside it and its exposure climbs several-fold.",
            Compound(
                name="Interaction victim",
                partition_coefficients={"adipose": 8.0, "liver": 4.0},
                default_partition_coefficient=2.0,
                fraction_unbound_plasma=0.5,
                metabolism=(
                    Metabolism(enzyme="CYP3A4", vmax_mg_per_h=2.0, km_mg_per_L=0.5),
                ),
            ),
            dose=4.0,
        ),
        _entry(
            "inhibitor",
            "CYP3A4 inhibitor",
            "Blocks CYP3A4 without being metabolised by it. On its own it does "
            "nothing interesting -- pick it *together with* the victim.",
            Compound(
                name="CYP3A4 inhibitor",
                default_partition_coefficient=2.0,
                fraction_unbound_plasma=0.5,
                inhibits=(Inhibition(enzyme="CYP3A4", ki_mg_per_L=0.05),),
            ),
            dose=40.0,
        ),
        _entry(
            "brain_penetrant",
            "Brain penetrant",
            "Crosses into the brain and concentrates there. Rotate the 3D view "
            "and watch the head light up before anything else.",
            Compound(
                name="Brain penetrant",
                partition_coefficients={"brain": 12.0, "adipose": 10.0, "liver": 3.0},
                default_partition_coefficient=1.5,
                fraction_unbound_plasma=0.3,
                hepatic_intrinsic_clearance_L_per_h=2.0,
            ),
            dose=4.0,
        ),
    )
}


def names() -> list[str]:
    """The library keys, in display order."""
    return list(LIBRARY)


def get_compound(key: str) -> Compound:
    """Look up a compound by key.

    Raises:
        KeyError: If no entry has that key.
    """
    if key not in LIBRARY:
        available = ", ".join(LIBRARY)
        raise KeyError(f"no compound {key!r} in the library; available: {available}")
    return LIBRARY[key].compound
