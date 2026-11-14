"""Species physiology: organ volumes and blood flows.

Tissue volumes are expressed as fractions of body weight and blood flows as
fractions of cardiac output, so a parameter set scales to any body weight
within its species.

Reference values are literature means taken from:

    Brown RP, Delp MD, Lindstedt SL, Rhomberg LR, Beliles RP (1997).
    "Physiological parameter values for physiologically based pharmacokinetic
    models." Toxicology and Industrial Health 13(4):407-484.

They describe a typical healthy animal. Real studies deviate — strain, age,
sex, diet and disease all move these numbers. Override them with values
measured in your own study whenever you have them; see `Physiology.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = ["TissueSpec", "Physiology", "RAT", "MOUSE", "get_physiology", "SPECIES"]


@dataclass(frozen=True)
class TissueSpec:
    """A tissue compartment's share of body weight and cardiac output.

    Attributes:
        volume_fraction: Tissue volume as a fraction of total body weight,
            assuming unit density (1 kg == 1 L).
        flow_fraction: Blood flow through the tissue as a fraction of cardiac
            output. Zero for tissues modelled without their own perfusion.
        drains_to: Where venous outflow goes -- ``"venous"`` for the systemic
            venous pool, or ``"liver"`` for pre-hepatic (portal) tissues such
            as gut and spleen, whose outflow perfuses the liver first.
    """

    volume_fraction: float
    flow_fraction: float
    drains_to: str = "venous"

    def __post_init__(self) -> None:
        if self.volume_fraction <= 0:
            raise ValueError(f"volume_fraction must be positive, got {self.volume_fraction}")
        if self.flow_fraction < 0:
            raise ValueError(f"flow_fraction must be non-negative, got {self.flow_fraction}")
        if self.drains_to not in ("venous", "liver"):
            raise ValueError(f"drains_to must be 'venous' or 'liver', got {self.drains_to!r}")


@dataclass(frozen=True)
class Physiology:
    """A complete physiological parameter set for one animal.

    Attributes:
        species: Species label, used only for reporting.
        body_weight_kg: Total body weight.
        cardiac_output_coefficient: Cardiac output in L/h at 1 kg body weight.
            Scaled allometrically as ``coefficient * BW ** exponent``.
        cardiac_output_exponent: Allometric exponent for cardiac output.
        tissues: Named tissue compartments.
        arterial_blood_fraction: Arterial blood volume as a fraction of body
            weight.
        venous_blood_fraction: Venous blood volume as a fraction of body weight.
        plasma_fraction: Plasma volume as a fraction of body weight. Used to
            convert between blood and plasma concentrations when reporting.
    """

    species: str
    body_weight_kg: float
    cardiac_output_coefficient: float
    cardiac_output_exponent: float
    tissues: dict[str, TissueSpec]
    arterial_blood_fraction: float
    venous_blood_fraction: float
    plasma_fraction: float
    lung_name: str = field(default="lung")
    liver_name: str = field(default="liver")

    def __post_init__(self) -> None:
        if self.body_weight_kg <= 0:
            raise ValueError(f"body_weight_kg must be positive, got {self.body_weight_kg}")
        for required in (self.lung_name, self.liver_name):
            if required not in self.tissues:
                raise ValueError(f"physiology is missing the required {required!r} tissue")
        if self.tissues[self.lung_name].drains_to != "venous":
            raise ValueError("the lung must drain to the arterial side, not into another tissue")
        for name, spec in self.tissues.items():
            if spec.drains_to == "liver" and name == self.liver_name:
                raise ValueError("the liver cannot drain into itself")

    # -- derived quantities ------------------------------------------------

    @property
    def cardiac_output_L_per_h(self) -> float:
        """Total cardiac output, scaled allometrically to body weight."""
        return self.cardiac_output_coefficient * self.body_weight_kg**self.cardiac_output_exponent

    def volume_L(self, tissue: str) -> float:
        """Volume of one tissue in litres."""
        return self.tissues[tissue].volume_fraction * self.body_weight_kg

    def flow_L_per_h(self, tissue: str) -> float:
        """Blood flow through one tissue in litres per hour.

        For the lung this is the whole cardiac output, since all venous blood
        passes through it.
        """
        if tissue == self.lung_name:
            return self.cardiac_output_L_per_h
        return self.tissues[tissue].flow_fraction * self.cardiac_output_L_per_h

    def liver_inflow_L_per_h(self) -> float:
        """Total blood flow leaving the liver.

        The sum of the liver's own arterial supply and the flow arriving from
        every tissue that drains into it.
        """
        total = self.flow_L_per_h(self.liver_name)
        total += sum(
            self.flow_L_per_h(name)
            for name, spec in self.tissues.items()
            if spec.drains_to == "liver"
        )
        return total

    @property
    def arterial_volume_L(self) -> float:
        return self.arterial_blood_fraction * self.body_weight_kg

    @property
    def venous_volume_L(self) -> float:
        return self.venous_blood_fraction * self.body_weight_kg

    @property
    def plasma_volume_L(self) -> float:
        return self.plasma_fraction * self.body_weight_kg

    # -- construction helpers ---------------------------------------------

    def at_weight(self, body_weight_kg: float) -> Physiology:
        """This parameter set rescaled to a different body weight."""
        return replace(self, body_weight_kg=body_weight_kg)

    def replace(self, **changes: object) -> Physiology:
        """A copy with individual fields overridden."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def with_tissue(self, name: str, spec: TissueSpec) -> Physiology:
        """A copy with one tissue added or replaced."""
        tissues = dict(self.tissues)
        tissues[name] = spec
        return replace(self, tissues=tissues)

    def perfused_tissues(self) -> list[str]:
        """Tissue names excluding the lung, in insertion order."""
        return [name for name in self.tissues if name != self.lung_name]

    def flow_balance_error(self) -> float:
        """How far tissue flow fractions are from summing to cardiac output.

        Returns the signed difference between the total systemic flow implied
        by the tissue fractions and 1.0. A well-formed parameter set should be
        within a few percent of zero; the residual reflects rounding in the
        published means and tissues the set does not model.
        """
        systemic = sum(
            spec.flow_fraction for name, spec in self.tissues.items() if name != self.lung_name
        )
        return systemic - 1.0

    def volume_balance_error(self) -> float:
        """How far tissue and blood volumes are from summing to body weight.

        Returns the signed difference between the total modelled volume
        fraction and 1.0, assuming unit tissue density.
        """
        total = sum(spec.volume_fraction for spec in self.tissues.values())
        total += self.arterial_blood_fraction + self.venous_blood_fraction
        return total - 1.0


# Tissue volume fractions and flow fractions for an adult rat. Gut and spleen
# carry portal flow and so drain into the liver rather than straight to the
# venous pool; the liver's own `flow_fraction` is its hepatic artery supply.
RAT = Physiology(
    species="rat",
    body_weight_kg=0.25,
    cardiac_output_coefficient=14.0,
    cardiac_output_exponent=0.75,
    arterial_blood_fraction=0.0215,
    venous_blood_fraction=0.0525,
    plasma_fraction=0.0312,
    tissues={
        "lung": TissueSpec(volume_fraction=0.0050, flow_fraction=0.0),
        "liver": TissueSpec(volume_fraction=0.0366, flow_fraction=0.021),
        "gut": TissueSpec(volume_fraction=0.0270, flow_fraction=0.131, drains_to="liver"),
        "spleen": TissueSpec(volume_fraction=0.0020, flow_fraction=0.010, drains_to="liver"),
        "kidney": TissueSpec(volume_fraction=0.0073, flow_fraction=0.141),
        "heart": TissueSpec(volume_fraction=0.0033, flow_fraction=0.049),
        "brain": TissueSpec(volume_fraction=0.0057, flow_fraction=0.020),
        "muscle": TissueSpec(volume_fraction=0.4043, flow_fraction=0.278),
        "adipose": TissueSpec(volume_fraction=0.0700, flow_fraction=0.070),
        "skin": TissueSpec(volume_fraction=0.1903, flow_fraction=0.058),
        "bone": TissueSpec(volume_fraction=0.0730, flow_fraction=0.122),
        # Everything not modelled explicitly, carrying the balance of body
        # weight and cardiac output so that both close exactly. Without it the
        # named tissues account for only 90% of flow and the circulation would
        # not conserve mass.
        "rest": TissueSpec(volume_fraction=0.1015, flow_fraction=0.100),
    },
)

MOUSE = Physiology(
    species="mouse",
    body_weight_kg=0.025,
    cardiac_output_coefficient=14.0,
    cardiac_output_exponent=0.75,
    arterial_blood_fraction=0.0199,
    venous_blood_fraction=0.0485,
    plasma_fraction=0.0500,
    tissues={
        "lung": TissueSpec(volume_fraction=0.0073, flow_fraction=0.0),
        "liver": TissueSpec(volume_fraction=0.0549, flow_fraction=0.020),
        "gut": TissueSpec(volume_fraction=0.0422, flow_fraction=0.141, drains_to="liver"),
        "spleen": TissueSpec(volume_fraction=0.0035, flow_fraction=0.011, drains_to="liver"),
        "kidney": TissueSpec(volume_fraction=0.0167, flow_fraction=0.091),
        "heart": TissueSpec(volume_fraction=0.0050, flow_fraction=0.066),
        "brain": TissueSpec(volume_fraction=0.0165, flow_fraction=0.033),
        "muscle": TissueSpec(volume_fraction=0.3840, flow_fraction=0.159),
        "adipose": TissueSpec(volume_fraction=0.0700, flow_fraction=0.070),
        "skin": TissueSpec(volume_fraction=0.1650, flow_fraction=0.058),
        "bone": TissueSpec(volume_fraction=0.1070, flow_fraction=0.110),
        "rest": TissueSpec(volume_fraction=0.0595, flow_fraction=0.241),
    },
)

SPECIES: dict[str, Physiology] = {"rat": RAT, "mouse": MOUSE}


def get_physiology(species: str, body_weight_kg: float | None = None) -> Physiology:
    """Look up a built-in parameter set by species name.

    Args:
        species: Species name, case-insensitive. See `SPECIES` for the
            available sets.
        body_weight_kg: Body weight to scale to. Defaults to the reference
            weight for the species.

    Raises:
        KeyError: If the species has no built-in parameter set.
    """
    key = species.strip().lower()
    if key not in SPECIES:
        available = ", ".join(sorted(SPECIES))
        raise KeyError(f"no physiology for species {species!r}; available: {available}")
    base = SPECIES[key]
    return base if body_weight_kg is None else base.at_weight(body_weight_kg)
