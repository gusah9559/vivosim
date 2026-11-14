"""Several compounds in one animal, competing for the same enzymes.

Each compound gets its own `PBPKModel` and its own block of the state vector.
The compounds are otherwise independent: they do not share tissues, binding
sites or blood. The single coupling is hepatic metabolism -- if two compounds
are handled by the same enzyme, each slows the other down.

That coupling is why this cannot be done by simulating each compound separately
and overlaying the results. The interaction is a property of the pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .compound import Compound
from .dosing import Regimen
from .kinetics import Metabolism
from .model import PBPKModel
from .physiology import Physiology

__all__ = ["CoupledModel"]


@dataclass
class CoupledModel:
    """A set of per-compound models sharing one animal's enzymes.

    Attributes:
        models: One `PBPKModel` per compound, in the order given.
        physiology: The animal they all occupy.
    """

    models: list[PBPKModel]
    physiology: Physiology
    offsets: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("a coupled model needs at least one compound")

        names = [model.compound.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError(
                "compounds must have distinct names so their results can be "
                f"told apart, got {names}"
            )

        offset = 0
        for model in self.models:
            self.offsets.append(offset)
            offset += model.n_states
        self.n_states = offset

        self._slices = [
            slice(start, start + model.n_states)
            for start, model in zip(self.offsets, self.models, strict=True)
        ]

        # Saturable pathways, and who competes for each enzyme. A substrate
        # competes through its own Km -- occupying the site is what matters,
        # not whether the occupier is turned over.
        self._pathways: list[tuple[int, Metabolism]] = []
        self._inhibitors: dict[str, list[tuple[int, float]]] = {}
        for index, model in enumerate(self.models):
            compound = model.compound
            for pathway in compound.metabolism:
                self._pathways.append((index, pathway))
                self._inhibitors.setdefault(pathway.enzyme, []).append(
                    (index, pathway.km_mg_per_L)
                )
            for inhibition in compound.inhibits:
                self._inhibitors.setdefault(inhibition.enzyme, []).append(
                    (index, inhibition.ki_mg_per_L)
                )

        self._has_saturable = bool(self._pathways)

    # -- introspection -----------------------------------------------------

    @property
    def compounds(self) -> list[Compound]:
        return [model.compound for model in self.models]

    def block(self, index: int) -> slice:
        """The slice of the state vector belonging to one compound."""
        return self._slices[index]

    def interacting_enzymes(self) -> dict[str, list[str]]:
        """Enzymes contested by more than one compound, and who contests them.

        The pairs whose exposure cannot be predicted from single-compound
        simulations.
        """
        contested: dict[str, list[str]] = {}
        for enzyme, participants in self._inhibitors.items():
            if len(participants) > 1:
                contested[enzyme] = [
                    self.models[index].compound.name for index, _ in participants
                ]
        return contested

    def initial_state(self) -> NDArray[np.float64]:
        return np.zeros(self.n_states, dtype=float)

    # -- derivatives -------------------------------------------------------

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Time derivative of the combined state vector.

        Args:
            t: Time in hours.
            y: Combined state of every compound, in mg.

        Returns:
            dy/dt in mg/h.
        """
        dy = np.empty_like(y)

        rates = [0.0] * len(self.models)
        if self._has_saturable:
            # Every compound's enzyme-side concentration has to be known before
            # any rate can be computed, since each appears in the others'
            # denominators.
            from .kinetics import metabolic_rates

            unbound = [
                model.liver_unbound_mg_per_L(y[block])
                for model, block in zip(self.models, self._slices, strict=True)
            ]
            rates = metabolic_rates(self._pathways, self._inhibitors, unbound)

        for model, block, rate in zip(self.models, self._slices, rates, strict=True):
            dy[block] = model.rhs(t, y[block], rate)
        return dy


def build_coupled(
    compounds: list[Compound], physiology: Physiology, regimens: list[Regimen]
) -> CoupledModel:
    """Assemble a coupled model from compounds and their regimens.

    Args:
        compounds: The compounds to simulate together.
        physiology: The animal they share.
        regimens: One regimen per compound, in the same order.

    Returns:
        The assembled `CoupledModel`.
    """
    models = [
        PBPKModel(compound=compound, physiology=physiology, regimen=regimen)
        for compound, regimen in zip(compounds, regimens, strict=True)
    ]
    return CoupledModel(models=models, physiology=physiology)
