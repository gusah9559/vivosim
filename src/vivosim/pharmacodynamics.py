"""Linking concentration to effect.

Everything else in this package is pharmacokinetics: what the body does to the
compound. This module is the other direction -- what the compound does to the
body -- and it is a far weaker kind of prediction. Read this before using it.

What this can and cannot tell you
---------------------------------
Given a concentration and *measured* parameters describing how that compound's
concentration relates to a *measured* response, this computes how that response
rises and falls over time. It can tell you that an effect peaks two hours after
a dose, or that a second dose arrives before the first has worn off, or that
doubling a dose barely moves a response already near its ceiling.

It cannot tell you whether a compound works. The parameters here -- the maximum
achievable effect, the concentration giving half of it -- are not predictable
from a molecule. They come from an experiment that already measured the effect
you are asking about. Supply numbers from such an experiment and this
interpolates between them honestly; invent them and it will produce a confident
curve that means nothing.

Nor is "effect" a disease. A model like this describes one measured readout --
a receptor occupancy, an enzyme inhibited, a blood pressure. Whether moving
that readout cures anything is a different question, and not one any simulation
answers.

The model
---------
Response follows a sigmoid Emax curve, the standard description of a saturable
biological response::

    E = E0 + Emax * C^h / (EC50^h + C^h)

Saturation is the point: biology runs out of receptors, so response flattens
however much more compound arrives. Raising the Hill coefficient ``h`` steepens
the approach, making the response more switch-like.

Effect often lags concentration, because the compound has to reach the tissue
where it acts and engage its target. An optional effect compartment introduces
that delay as a first-order equilibration, so effect peaks after concentration
does rather than with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .results import Result

__all__ = ["EmaxModel", "EffectProfile", "effect_of"]


@dataclass(frozen=True)
class EmaxModel:
    """A measured relationship between concentration and response.

    Attributes:
        baseline: Response with no compound present (E0), in whatever units the
            response was measured in.
        e_max: Maximum change from baseline the compound can produce. Negative
            for a response the compound suppresses.
        ec50_mg_per_L: Concentration producing half of ``e_max``. Lower means
            more potent.
        hill: Hill coefficient. 1 is the ordinary hyperbolic response; higher
            values make it switch-like.
        driving_site: Which concentration drives the response -- ``"plasma"``,
            ``"blood"``, or a tissue name. A compound acting in the brain
            should be driven by brain concentration, not by plasma.
        equilibration_rate_const_per_h: Rate constant (ke0) of an effect
            compartment, introducing the delay between concentration and
            response. ``None`` means the response follows concentration
            instantly.
        response_name: Label for the response being modelled, used in reporting.
    """

    baseline: float = 0.0
    e_max: float = 100.0
    ec50_mg_per_L: float = 1.0
    hill: float = 1.0
    driving_site: str = "plasma"
    equilibration_rate_const_per_h: float | None = None
    response_name: str = "response"

    def __post_init__(self) -> None:
        if self.ec50_mg_per_L <= 0:
            raise ValueError(f"ec50_mg_per_L must be positive, got {self.ec50_mg_per_L}")
        if self.hill <= 0:
            raise ValueError(f"hill must be positive, got {self.hill}")
        if (
            self.equilibration_rate_const_per_h is not None
            and self.equilibration_rate_const_per_h <= 0
        ):
            raise ValueError(
                "equilibration_rate_const_per_h must be positive when given, "
                f"got {self.equilibration_rate_const_per_h}"
            )

    @property
    def has_delay(self) -> bool:
        """Whether the response lags concentration through an effect compartment."""
        return self.equilibration_rate_const_per_h is not None

    def response(self, concentration: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Response at a given concentration, with no time dependence.

        Args:
            concentration: Concentration in mg/L.

        Returns:
            Response in the units of `baseline` and `e_max`.
        """
        values = np.asarray(concentration, dtype=float)
        safe = np.clip(values, 0.0, None)
        powered = safe**self.hill
        return self.baseline + self.e_max * powered / (
            self.ec50_mg_per_L**self.hill + powered
        )

    @property
    def maximum_response(self) -> float:
        """The response approached as concentration grows without bound."""
        return self.baseline + self.e_max


def _effect_site(
    concentration: NDArray[np.float64],
    time_h: NDArray[np.float64],
    ke0: float,
) -> NDArray[np.float64]:
    """Concentration at the effect site, lagging the driving concentration.

    Solves ``dCe/dt = ke0 * (C - Ce)`` across the grid. The driving
    concentration is taken as linear within each interval, for which the
    equation has a closed form -- so the lag is exact for the sampled profile
    rather than being re-approximated by another numerical solver.
    """
    effect_site = np.zeros_like(concentration)
    for i in range(1, concentration.size):
        dt = time_h[i] - time_h[i - 1]
        if dt <= 0:
            effect_site[i] = effect_site[i - 1]
            continue
        decay = np.exp(-ke0 * dt)
        c0 = concentration[i - 1]
        slope = (concentration[i] - c0) / dt
        effect_site[i] = (
            effect_site[i - 1] * decay
            + c0 * (1.0 - decay)
            + slope * (dt - (1.0 - decay) / ke0)
        )
    return effect_site


@dataclass(frozen=True)
class EffectProfile:
    """A response over time, computed from a concentration profile.

    Attributes:
        time_h: The time grid.
        driving_concentration: The concentration the model was driven by.
        effect_site_concentration: Concentration at the effect site. Identical
            to the driving concentration when the model has no delay.
        effect: The response over time.
        model: The model that produced it.
    """

    time_h: NDArray[np.float64]
    driving_concentration: NDArray[np.float64]
    effect_site_concentration: NDArray[np.float64]
    effect: NDArray[np.float64]
    model: EmaxModel

    @property
    def peak_effect(self) -> float:
        """The largest response reached, measured from zero."""
        return float(np.max(self.effect))

    @property
    def peak_change(self) -> float:
        """The largest change from baseline reached."""
        changes = self.effect - self.model.baseline
        return float(changes[np.argmax(np.abs(changes))])

    @property
    def time_of_peak_h(self) -> float:
        """When the response was furthest from baseline."""
        changes = np.abs(self.effect - self.model.baseline)
        return float(self.time_h[int(np.argmax(changes))])

    @property
    def saturation(self) -> float:
        """How close the peak came to the model's ceiling, from 0 to 1.

        A value near 1 means the response was already maxed out, so a larger
        dose would add little. Near 0 means the response was still on the steep
        part of the curve.
        """
        if self.model.e_max == 0:
            return 0.0
        return float(abs(self.peak_change / self.model.e_max))

    def time_above(self, threshold: float) -> float:
        """How long the response stayed above a threshold, in hours.

        Args:
            threshold: The response level to count time above.

        Returns:
            Total time above it. Computed on the output grid, so a coarse grid
            gives a coarse answer.
        """
        above = self.effect > threshold
        if not np.any(above):
            return 0.0
        # Trapezoidal in the indicator, which credits partial intervals at the
        # edges rather than counting whole samples.
        return float(np.trapezoid(above.astype(float), self.time_h))

    def area_under_effect(self) -> float:
        """Area under the response-time curve, in response units times hours."""
        return float(np.trapezoid(self.effect, self.time_h))

    def summary(self) -> dict[str, float]:
        """Headline numbers describing the response."""
        return {
            "peak_effect": self.peak_effect,
            "peak_change_from_baseline": self.peak_change,
            "time_of_peak_h": self.time_of_peak_h,
            "saturation": self.saturation,
            "area_under_effect": self.area_under_effect(),
        }


def effect_of(result: Result, model: EmaxModel) -> EffectProfile:
    """Compute a response over time from a simulated concentration profile.

    Args:
        result: The simulation supplying the concentration.
        model: The measured concentration-response relationship.

    Returns:
        An `EffectProfile`.

    Raises:
        KeyError: If the model's driving site is not in the result.
    """
    site = model.driving_site
    if site == "plasma":
        driving = result.plasma_concentration()
    elif site == "blood":
        driving = result.blood_concentration()
    else:
        driving = result.tissue_concentration(site)

    if model.has_delay:
        assert model.equilibration_rate_const_per_h is not None
        effect_site = _effect_site(
            driving, result.time_h, model.equilibration_rate_const_per_h
        )
    else:
        effect_site = driving

    return EffectProfile(
        time_h=result.time_h,
        driving_concentration=driving,
        effect_site_concentration=effect_site,
        effect=model.response(effect_site),
        model=model,
    )
