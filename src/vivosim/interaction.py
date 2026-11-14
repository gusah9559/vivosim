"""Measuring a drug-drug interaction: each compound alone against together.

An interaction is not a property of a compound, it is a property of a pair. So
measuring one means running the same regimen twice -- once with the other
compound present and once without -- and comparing the exposure.

The comparison is expressed as an **AUC ratio**: exposure together divided by
exposure alone. That is the quantity regulators ask for, and the bands used to
describe it here follow the FDA's classification of an interaction's strength::

    >= 5          strong
    >= 2, < 5     moderate
    >= 1.25, < 2  weak
    < 1.25        not clinically meaningful

Those bands describe clinical significance in humans, and this is a preclinical
simulation of a model animal. The vocabulary is borrowed because it is what the
field uses to talk about magnitude; a ratio computed here is not a regulatory
finding about anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .compound import Compound
from .dosing import Regimen
from .physiology import Physiology
from .results import Result, Study
from .simulate import simulate

__all__ = ["CompoundInteraction", "InteractionReport", "assess_interaction", "classify_ratio"]


def classify_ratio(ratio: float) -> str:
    """Describe the strength of an exposure change.

    Args:
        ratio: Exposure together divided by exposure alone.

    Returns:
        One of ``"strong"``, ``"moderate"``, ``"weak"`` for increases, the same
        bands prefixed ``"reduced "`` when exposure falls instead, or
        ``"none"`` when the change is below the 1.25-fold threshold in either
        direction -- a negligible fall is not a finding, so it is not reported
        as one.
    """
    if not ratio > 0:
        return "none"
    fold = ratio if ratio >= 1.0 else 1.0 / ratio
    band = _band(fold)
    if band == "none":
        return "none"
    return band if ratio >= 1.0 else "reduced " + band


def _band(ratio: float) -> str:
    if ratio >= 5.0:
        return "strong"
    if ratio >= 2.0:
        return "moderate"
    if ratio >= 1.25:
        return "weak"
    return "none"


@dataclass(frozen=True)
class CompoundInteraction:
    """How one compound's exposure changed in the presence of the others.

    Attributes:
        name: The compound's name.
        alone: The result of simulating it on its own.
        together: The result of simulating it alongside the others.
        auc_alone: Plasma AUC on its own, in mg*h/L.
        auc_together: Plasma AUC alongside the others.
        cmax_alone: Peak plasma concentration on its own, in mg/L.
        cmax_together: Peak plasma concentration alongside the others.
    """

    name: str
    alone: Result
    together: Result
    auc_alone: float
    auc_together: float
    cmax_alone: float
    cmax_together: float

    @property
    def auc_ratio(self) -> float:
        """Exposure together divided by exposure alone."""
        return self.auc_together / self.auc_alone if self.auc_alone > 0 else float("nan")

    @property
    def cmax_ratio(self) -> float:
        """Peak concentration together divided by peak alone."""
        return self.cmax_together / self.cmax_alone if self.cmax_alone > 0 else float("nan")

    @property
    def classification(self) -> str:
        """The strength band of the exposure change."""
        return classify_ratio(self.auc_ratio)

    @property
    def is_affected(self) -> bool:
        """Whether the exposure change rises above the 1.25-fold threshold."""
        return self.classification != "none"

    def difference(self) -> NDArray[np.float64]:
        """Plasma concentration together minus alone, over the time grid."""
        return self.together.plasma_concentration() - self.alone.plasma_concentration()


@dataclass(frozen=True)
class InteractionReport:
    """The outcome of comparing compounds alone against together.

    Attributes:
        compounds: One entry per compound, in the order simulated.
        study: The joint simulation the comparison is drawn from.
    """

    compounds: list[CompoundInteraction]
    study: Study

    def __getitem__(self, key: str | int) -> CompoundInteraction:
        if isinstance(key, int):
            return self.compounds[key]
        for entry in self.compounds:
            if entry.name == key:
                return entry
        available = ", ".join(self.names)
        raise KeyError(f"no compound named {key!r}; available: {available}")

    def __iter__(self):
        return iter(self.compounds)

    def __len__(self) -> int:
        return len(self.compounds)

    @property
    def names(self) -> list[str]:
        return [entry.name for entry in self.compounds]

    @property
    def time_h(self) -> NDArray[np.float64]:
        return self.study.time_h

    def affected(self) -> list[CompoundInteraction]:
        """The compounds whose exposure changed by more than 1.25-fold."""
        return [entry for entry in self.compounds if entry.is_affected]

    def interacting_enzymes(self) -> dict[str, list[str]]:
        """Enzymes contested by more than one compound."""
        return self.study.interacting_enzymes()

    def largest_change(self) -> CompoundInteraction | None:
        """The compound whose exposure moved furthest from its solo value."""
        if not self.compounds:
            return None
        return max(
            self.compounds,
            key=lambda entry: abs(np.log(entry.auc_ratio))
            if entry.auc_ratio > 0
            else 0.0,
        )

    def summary(self) -> dict[str, dict[str, float | str]]:
        """A flat per-compound summary, convenient for reporting."""
        return {
            entry.name: {
                "auc_alone": entry.auc_alone,
                "auc_together": entry.auc_together,
                "auc_ratio": entry.auc_ratio,
                "cmax_ratio": entry.cmax_ratio,
                "classification": entry.classification,
            }
            for entry in self.compounds
        }


def assess_interaction(
    compounds: Sequence[Compound],
    physiology: Physiology,
    regimens: Regimen | Sequence[Regimen],
    **simulate_kwargs,
) -> InteractionReport:
    """Simulate compounds together and alone, and compare their exposures.

    Each compound is simulated once alongside the others and once by itself on
    the identical regimen and time grid, so the only difference between the two
    runs is the presence of the other compounds.

    Args:
        compounds: Two or more compounds to assess. One alone has nothing to
            interact with.
        physiology: The animal to simulate them in.
        regimens: One regimen per compound, or a single regimen for all.
        **simulate_kwargs: Passed through to `simulate` for both the joint and
            the solo runs, so the comparison is made on matching settings.

    Returns:
        An `InteractionReport`.

    Raises:
        ValueError: If fewer than two compounds are given.
    """
    compounds = list(compounds)
    if len(compounds) < 2:
        raise ValueError(
            "an interaction needs at least two compounds; "
            f"got {len(compounds)}"
        )

    study = simulate(compounds, physiology, regimens, **simulate_kwargs)
    assert isinstance(study, Study)

    # Reuse the joint run's time grid for the solo runs so the two profiles are
    # sampled identically and can be subtracted point by point.
    solo_kwargs = dict(simulate_kwargs)
    solo_kwargs.pop("duration_h", None)
    solo_kwargs.pop("n_points", None)
    solo_kwargs["times_h"] = study.time_h

    entries: list[CompoundInteraction] = []
    for index, compound in enumerate(compounds):
        together = study[index]
        alone = simulate(
            compound, physiology, together.model.regimen, **solo_kwargs
        )
        assert isinstance(alone, Result)
        entries.append(
            CompoundInteraction(
                name=compound.name,
                alone=alone,
                together=together,
                auc_alone=alone.auc("plasma"),
                auc_together=together.auc("plasma"),
                cmax_alone=alone.cmax("plasma"),
                cmax_together=together.cmax("plasma"),
            )
        )

    return InteractionReport(compounds=entries, study=study)
