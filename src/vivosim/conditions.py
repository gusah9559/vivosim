"""Disease states: how an unhealthy animal handles a compound differently.

Every parameter set elsewhere describes a healthy animal. Real subjects are
often not healthy, and the difference is not cosmetic -- a damaged liver can
leave several times the exposure from the same dose, which is how a dose that
is routine in a healthy animal becomes an overdose in a sick one.

A `Condition` changes both sides of the problem, because disease does not
respect that separation. Hepatic impairment reduces liver blood flow *and*
enzyme capacity *and* raises the unbound fraction by lowering plasma protein,
and all three push exposure the same way.

What this does not do
---------------------
These are *pharmacokinetic* conditions: they describe how a body altered by
disease moves a compound around. Nothing here models the disease itself, its
progression, or whether a compound treats it. An impaired animal in this module
is a healthy animal with different numbers, not a sick one.

The severities are coarse, defensible multipliers rather than measurements of
any particular model of disease. Use them to ask how sensitive an outcome is to
organ function, and replace them with values from your own study before
drawing a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .compound import Compound
from .kinetics import Metabolism
from .physiology import Physiology

__all__ = [
    "Condition",
    "HEALTHY",
    "hepatic_impairment",
    "renal_impairment",
    "obesity",
    "CONDITIONS",
]


@dataclass(frozen=True)
class Condition:
    """An altered physiological state, as multipliers on a healthy animal.

    Attributes:
        name: Short label.
        description: What the state represents and what it changes.
        liver_flow: Multiplier on hepatic blood flow. Below 1 means blood is
            shunted past the liver, so less compound is presented to it.
        liver_volume: Multiplier on functional liver volume.
        kidney_flow: Multiplier on renal blood flow.
        adipose_volume: Multiplier on fat mass. Above 1 adds that mass to the
            animal's weight rather than taking it from another tissue.
        metabolic_capacity: Multiplier on hepatic intrinsic clearance and on
            Vmax -- how much working enzyme is left.
        renal_capacity: Multiplier on renal clearance.
        unbound_fraction: Multiplier on the unbound fraction in plasma. Above 1
            represents the lower plasma protein of liver disease, which frees
            more compound to distribute and to be cleared.
    """

    name: str
    description: str
    liver_flow: float = 1.0
    liver_volume: float = 1.0
    kidney_flow: float = 1.0
    adipose_volume: float = 1.0
    metabolic_capacity: float = 1.0
    renal_capacity: float = 1.0
    unbound_fraction: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "liver_flow",
            "liver_volume",
            "kidney_flow",
            "adipose_volume",
            "metabolic_capacity",
            "renal_capacity",
            "unbound_fraction",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

    @property
    def is_healthy(self) -> bool:
        """Whether this condition leaves everything unchanged."""
        return all(
            getattr(self, name) == 1.0
            for name in (
                "liver_flow",
                "liver_volume",
                "kidney_flow",
                "adipose_volume",
                "metabolic_capacity",
                "renal_capacity",
                "unbound_fraction",
            )
        )

    # -- application -------------------------------------------------------

    def apply(
        self, compound: Compound, physiology: Physiology
    ) -> tuple[Compound, Physiology]:
        """The compound and the animal as this condition leaves them."""
        return self.apply_compound(compound), self.apply_physiology(physiology)

    def apply_physiology(self, physiology: Physiology) -> Physiology:
        """The animal's body as this condition leaves it.

        Blood flows and tissue volumes are rebalanced so that flows still sum
        to cardiac output and volumes still sum to body weight. Without that
        the circulation would stop conserving mass, and every number downstream
        would be quietly wrong.
        """
        tissues = dict(physiology.tissues)
        body_weight = physiology.body_weight_kg

        # Flow diverted from an organ has to go somewhere, or the heart would
        # be ejecting more than returns. It is given to the lumped remainder.
        freed_flow = 0.0
        for organ, multiplier in (
            (physiology.liver_name, self.liver_flow),
            ("kidney", self.kidney_flow),
        ):
            if multiplier == 1.0 or organ not in tissues:
                continue
            spec = tissues[organ]
            new_flow = spec.flow_fraction * multiplier
            freed_flow += spec.flow_fraction - new_flow
            tissues[organ] = replace(spec, flow_fraction=new_flow)

        if freed_flow and "rest" in tissues:
            spec = tissues["rest"]
            tissues["rest"] = replace(
                spec, flow_fraction=max(1e-9, spec.flow_fraction + freed_flow)
            )

        if self.liver_volume != 1.0:
            spec = tissues[physiology.liver_name]
            lost = spec.volume_fraction * (1.0 - self.liver_volume)
            tissues[physiology.liver_name] = replace(
                spec, volume_fraction=spec.volume_fraction * self.liver_volume
            )
            if "rest" in tissues:
                rest = tissues["rest"]
                tissues["rest"] = replace(
                    rest, volume_fraction=max(1e-9, rest.volume_fraction + lost)
                )

        if self.adipose_volume != 1.0 and "adipose" in tissues:
            # Fat is gained, not swapped for other tissue: the animal gets
            # heavier and every other tissue keeps its absolute size, so its
            # fraction of the new weight falls.
            old_fraction = tissues["adipose"].volume_fraction
            added = old_fraction * (self.adipose_volume - 1.0)
            body_weight = body_weight * (1.0 + added)
            scale = 1.0 / (1.0 + added)
            for name, spec in tissues.items():
                fraction = spec.volume_fraction
                if name == "adipose":
                    fraction = old_fraction * self.adipose_volume
                tissues[name] = replace(spec, volume_fraction=fraction * scale)
            arterial = physiology.arterial_blood_fraction * scale
            venous = physiology.venous_blood_fraction * scale
            plasma = physiology.plasma_fraction * scale
            return physiology.replace(
                tissues=tissues,
                body_weight_kg=body_weight,
                arterial_blood_fraction=arterial,
                venous_blood_fraction=venous,
                plasma_fraction=plasma,
            )

        return physiology.replace(tissues=tissues, body_weight_kg=body_weight)

    def apply_compound(self, compound: Compound) -> Compound:
        """The compound's clearance and binding as this condition leaves them."""
        unbound = min(1.0, compound.fraction_unbound_plasma * self.unbound_fraction)
        metabolism = tuple(
            Metabolism(
                enzyme=pathway.enzyme,
                vmax_mg_per_h=pathway.vmax_mg_per_h * self.metabolic_capacity,
                km_mg_per_L=pathway.km_mg_per_L,
            )
            for pathway in compound.metabolism
        )
        return replace(
            compound,
            fraction_unbound_plasma=unbound,
            hepatic_intrinsic_clearance_L_per_h=(
                compound.hepatic_intrinsic_clearance_L_per_h * self.metabolic_capacity
            ),
            renal_clearance_L_per_h=compound.renal_clearance_L_per_h * self.renal_capacity,
            metabolism=metabolism,
        )


HEALTHY = Condition(name="healthy", description="An unmodified reference animal.")


def hepatic_impairment(severity: str = "moderate") -> Condition:
    """Reduced liver function.

    Liver disease attacks a compound's clearance from three directions at once:
    blood is shunted past the liver, the working enzyme mass falls, and lower
    plasma protein leaves more compound unbound. All three raise exposure for a
    compound the liver clears.

    Args:
        severity: ``"mild"``, ``"moderate"`` or ``"severe"``.

    Raises:
        ValueError: If the severity is not one of those three.
    """
    grades = {
        "mild": (0.90, 0.90, 0.75, 1.15),
        "moderate": (0.75, 0.75, 0.45, 1.40),
        "severe": (0.55, 0.60, 0.20, 1.90),
    }
    flow, volume, capacity, unbound = _grade(grades, severity, "hepatic_impairment")
    return Condition(
        name=f"hepatic impairment ({severity})",
        description=(
            "Reduced hepatic blood flow, functional liver mass and metabolic "
            "capacity, with a raised unbound fraction from lower plasma protein."
        ),
        liver_flow=flow,
        liver_volume=volume,
        metabolic_capacity=capacity,
        unbound_fraction=unbound,
    )


def renal_impairment(severity: str = "moderate") -> Condition:
    """Reduced kidney function.

    Matters for a compound excreted unchanged by the kidney, and barely at all
    for one cleared entirely by the liver -- which is itself a useful thing to
    be able to show.

    Args:
        severity: ``"mild"``, ``"moderate"`` or ``"severe"``.

    Raises:
        ValueError: If the severity is not one of those three.
    """
    grades = {
        "mild": (0.85, 0.70),
        "moderate": (0.60, 0.35),
        "severe": (0.35, 0.12),
    }
    flow, capacity = _grade(grades, severity, "renal_impairment")
    return Condition(
        name=f"renal impairment ({severity})",
        description="Reduced renal blood flow and clearance of unchanged compound.",
        kidney_flow=flow,
        renal_capacity=capacity,
    )


def obesity(fold: float = 2.5) -> Condition:
    """Increased fat mass.

    The extra fat is added to the animal's weight rather than taken from
    another tissue. A lipophilic compound gains a large, poorly perfused
    reservoir to fill and then slowly empty, which lowers its peak and
    lengthens its stay; a water-soluble one barely notices.

    Args:
        fold: How many times the reference fat mass the animal carries.

    Raises:
        ValueError: If ``fold`` is not above zero.
    """
    if fold <= 0:
        raise ValueError(f"fold must be positive, got {fold}")
    return Condition(
        name=f"obesity ({fold:g}x fat)",
        description=(
            "Increased adipose mass added to body weight, giving a lipophilic "
            "compound a larger and poorly perfused reservoir."
        ),
        adipose_volume=fold,
    )


def _grade(grades: dict[str, tuple[float, ...]], severity: str, where: str):
    key = severity.strip().lower()
    if key not in grades:
        available = ", ".join(grades)
        raise ValueError(f"{where} severity must be one of {available}; got {severity!r}")
    return grades[key]


CONDITIONS: dict[str, Condition] = {
    "healthy": HEALTHY,
    "hepatic impairment (mild)": hepatic_impairment("mild"),
    "hepatic impairment (moderate)": hepatic_impairment("moderate"),
    "hepatic impairment (severe)": hepatic_impairment("severe"),
    "renal impairment (moderate)": renal_impairment("moderate"),
    "renal impairment (severe)": renal_impairment("severe"),
    "obesity (2.5x fat)": obesity(2.5),
}
