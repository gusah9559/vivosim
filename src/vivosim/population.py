"""Simulating a population rather than one animal.

Why a single simulation is not an answer
----------------------------------------
Every result elsewhere in this package describes one animal built from mean
parameters. No such animal exists. Real individuals differ in body weight, in
organ size, in how much metabolising enzyme they express, in how tightly their
plasma binds a compound -- and those differences are large. Enzyme expression
in particular varies several-fold between individuals, which is why the same
dose can be uneventful in most animals and an overdose in a few.

So the question worth asking is not "what is the exposure?" but "what is the
*spread* of exposure, and what fraction of animals falls outside the range I
can accept?". That is what this module computes: it samples a population,
simulates each individual, and reports the distribution.

How the sampling works
----------------------
Each varied parameter is drawn from a lognormal distribution around the
reference value. Lognormal because these quantities are positive and
right-skewed -- a clearance can be three times the mean but never below zero --
and because it is the distribution the field uses. Variability is specified as
a coefficient of variation, which converts to the underlying sigma by
``sigma = sqrt(ln(1 + CV^2))``.

What this does *not* model
--------------------------
Parameters are sampled independently. Real ones are correlated: a heavier
animal has a bigger liver, and the enzymes in that liver tend to move together.
Ignoring those correlations widens some predicted distributions and narrows
others, so the percentiles here are an approximation of the spread, not a
calibrated one. Correlated sampling is the obvious next step.

Nor is any of this a probability that a compound *works*. The probabilities
computed here are about exposure -- whether a concentration lands inside a
window you specified -- and a drug being present at a given concentration is
not the same as a drug having an effect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .compound import Compound
from .dosing import Regimen
from .kinetics import Metabolism
from .physiology import Physiology
from .results import Result
from .simulate import simulate

__all__ = [
    "Variability",
    "Individual",
    "PopulationResult",
    "sample_population",
    "simulate_population",
]


@dataclass(frozen=True)
class Variability:
    """How much each parameter differs between individuals.

    Values are coefficients of variation: 0.3 means a standard deviation of
    30% of the mean. Zero holds a parameter fixed at its reference value.

    The defaults are deliberately moderate and generic. They are a starting
    point for exploring how sensitive an outcome is to variability, not
    measurements of any real population -- substitute values from your own
    study before drawing a conclusion from a percentile.

    Attributes:
        body_weight: Variation in body weight, which scales every organ volume
            and the cardiac output together.
        cardiac_output: Variation in cardiac output beyond what body weight
            already explains.
        hepatic_clearance: Variation in metabolic capacity -- first-order
            clearance and Vmax alike. Usually the largest single contributor,
            because enzyme expression differs several-fold between individuals.
        renal_clearance: Variation in renal clearance.
        fraction_unbound: Variation in the unbound fraction in plasma.
        partition_coefficients: Variation in tissue partitioning.
    """

    body_weight: float = 0.15
    cardiac_output: float = 0.10
    hepatic_clearance: float = 0.40
    renal_clearance: float = 0.25
    fraction_unbound: float = 0.20
    partition_coefficients: float = 0.15

    def __post_init__(self) -> None:
        for field_name in (
            "body_weight",
            "cardiac_output",
            "hepatic_clearance",
            "renal_clearance",
            "fraction_unbound",
            "partition_coefficients",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")

    @classmethod
    def none(cls) -> Variability:
        """A population with no variability, every individual identical.

        Useful as a control: a population simulated this way must reproduce the
        single-animal result exactly.
        """
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Individual:
    """One sampled animal and the compound as it behaves in them.

    Attributes:
        index: Position in the population.
        compound: The compound with this individual's clearance and binding.
        physiology: This individual's body.
    """

    index: int
    compound: Compound
    physiology: Physiology


def _lognormal(
    rng: np.random.Generator, reference: float, cv: float, size: int
) -> NDArray[np.float64]:
    """Draw positive values around a reference with a given coefficient of variation.

    The median is shifted so that the *mean* of the draws is the reference
    value, rather than the median -- otherwise a population would be
    systematically lower than the animal it was built from.
    """
    if cv <= 0 or reference == 0:
        return np.full(size, float(reference))
    sigma = np.sqrt(np.log(1.0 + cv * cv))
    mu = np.log(reference) - 0.5 * sigma * sigma
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def sample_population(
    compound: Compound,
    physiology: Physiology,
    n: int,
    variability: Variability | None = None,
    seed: int | None = None,
) -> list[Individual]:
    """Draw a population of individuals around a reference animal.

    Args:
        compound: The reference compound.
        physiology: The reference animal.
        n: How many individuals to draw.
        variability: How much each parameter differs between them.
        seed: Seed for the random generator, so a population can be reproduced.
            Always set this for anything you intend to report.

    Returns:
        The sampled individuals.

    Raises:
        ValueError: If ``n`` is below 1.
    """
    if n < 1:
        raise ValueError(f"a population needs at least one individual, got {n}")
    spread = variability if variability is not None else Variability()
    rng = np.random.default_rng(seed)

    weights = _lognormal(rng, physiology.body_weight_kg, spread.body_weight, n)
    outputs = _lognormal(
        rng, physiology.cardiac_output_coefficient, spread.cardiac_output, n
    )
    hepatic = _lognormal(rng, 1.0, spread.hepatic_clearance, n)
    renal = _lognormal(rng, 1.0, spread.renal_clearance, n)
    unbound = _lognormal(rng, compound.fraction_unbound_plasma, spread.fraction_unbound, n)
    partition = _lognormal(rng, 1.0, spread.partition_coefficients, n)

    individuals: list[Individual] = []
    for i in range(n):
        body = physiology.replace(
            body_weight_kg=float(weights[i]),
            cardiac_output_coefficient=float(outputs[i]),
        )
        # The unbound fraction is a proportion, so a draw cannot leave (0, 1].
        fu = float(min(1.0, max(1e-9, unbound[i])))
        scaled_kp = {
            tissue: value * float(partition[i])
            for tissue, value in compound.partition_coefficients.items()
        }
        metabolism = tuple(
            Metabolism(
                enzyme=pathway.enzyme,
                vmax_mg_per_h=pathway.vmax_mg_per_h * float(hepatic[i]),
                km_mg_per_L=pathway.km_mg_per_L,
            )
            for pathway in compound.metabolism
        )
        individual_compound = replace(
            compound,
            partition_coefficients=scaled_kp,
            default_partition_coefficient=(
                compound.default_partition_coefficient * float(partition[i])
            ),
            fraction_unbound_plasma=fu,
            hepatic_intrinsic_clearance_L_per_h=(
                compound.hepatic_intrinsic_clearance_L_per_h * float(hepatic[i])
            ),
            renal_clearance_L_per_h=compound.renal_clearance_L_per_h * float(renal[i]),
            metabolism=metabolism,
        )
        individuals.append(
            Individual(index=i, compound=individual_compound, physiology=body)
        )
    return individuals


@dataclass(frozen=True)
class PopulationResult:
    """Exposure across a simulated population.

    Attributes:
        time_h: The shared output time grid.
        concentrations: Plasma concentration for every individual, shaped
            ``(n_individuals, n_times)``.
        auc: Plasma AUC for every individual, in mg*h/L.
        cmax: Peak plasma concentration for every individual, in mg/L.
        individuals: The sampled individuals, in the same order.
        reference: The single simulation built from the reference animal, for
            comparison against the population it was drawn from.
    """

    time_h: NDArray[np.float64]
    concentrations: NDArray[np.float64]
    auc: NDArray[np.float64]
    cmax: NDArray[np.float64]
    individuals: list[Individual]
    reference: Result

    @property
    def size(self) -> int:
        return len(self.individuals)

    # -- distribution ------------------------------------------------------

    def percentile_band(
        self, lower: float = 5.0, upper: float = 95.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Lower, median and upper concentration envelopes over time.

        Args:
            lower: Lower percentile, 0-100.
            upper: Upper percentile, 0-100.

        Returns:
            Three arrays over the time grid: the lower envelope, the median,
            and the upper envelope. Note these are percentiles computed at each
            time point, so the envelope is not the profile of any one
            individual -- it is the band most individuals stay inside.
        """
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError(
                f"percentiles must satisfy 0 <= lower < upper <= 100, got {lower}, {upper}"
            )
        band = np.percentile(self.concentrations, [lower, 50.0, upper], axis=0)
        return band[0], band[1], band[2]

    def auc_percentiles(self, *points: float) -> dict[float, float]:
        """Selected percentiles of total exposure."""
        chosen = points or (5.0, 25.0, 50.0, 75.0, 95.0)
        values = np.percentile(self.auc, list(chosen))
        return {float(p): float(v) for p, v in zip(chosen, values, strict=True)}

    def fold_range(self, lower: float = 5.0, upper: float = 95.0) -> float:
        """How many times larger the high-exposure tail is than the low one.

        A single number for how much the population spreads. A fold range near
        1 means the dose behaves the same in everyone; a large one means some
        individuals see several times the exposure of others on the same dose.
        """
        low, high = np.percentile(self.auc, [lower, upper])
        return float(high / low) if low > 0 else float("inf")

    # -- probabilities -----------------------------------------------------

    def probability_above(self, threshold_mg_per_L: float) -> float:
        """Fraction of the population whose peak concentration exceeds a threshold.

        Use for a concentration you want to stay under. The threshold is yours
        to supply from measured data; nothing here knows what is safe.
        """
        return float(np.mean(self.cmax > threshold_mg_per_L))

    def probability_auc_within(self, low: float, high: float) -> float:
        """Fraction of the population whose exposure lands inside a target window.

        Args:
            low: Lower bound of the acceptable AUC, in mg*h/L.
            high: Upper bound.

        Returns:
            The fraction between the bounds, inclusive.

        Raises:
            ValueError: If the window is empty.
        """
        if not high > low:
            raise ValueError(f"the window must have high > low, got {low}, {high}")
        inside = (self.auc >= low) & (self.auc <= high)
        return float(np.mean(inside))

    def probability_auc_above(self, threshold: float) -> float:
        """Fraction of the population whose exposure exceeds a threshold."""
        return float(np.mean(self.auc > threshold))

    def summary(self) -> dict[str, Any]:
        """Headline numbers describing the population."""
        return {
            "n": self.size,
            "auc_median": float(np.median(self.auc)),
            "auc_p5": float(np.percentile(self.auc, 5)),
            "auc_p95": float(np.percentile(self.auc, 95)),
            "auc_fold_range": self.fold_range(),
            "cmax_median": float(np.median(self.cmax)),
            "cmax_p95": float(np.percentile(self.cmax, 95)),
        }


def simulate_population(
    compound: Compound,
    physiology: Physiology,
    regimen: Regimen,
    *,
    n: int = 200,
    variability: Variability | None = None,
    seed: int | None = None,
    duration_h: float | None = None,
    n_points: int = 301,
    **simulate_kwargs,
) -> PopulationResult:
    """Simulate a population of individuals on the same regimen.

    Every individual is simulated on the same time grid so their profiles can
    be compared point by point, and a reference animal built from the mean
    parameters is simulated alongside for comparison.

    Args:
        compound: The reference compound.
        physiology: The reference animal.
        regimen: The regimen given to every individual. Note this is a fixed
            dose, not a dose per kilogram, so body weight variation changes
            exposure as it would in a real fixed-dose study.
        n: How many individuals to simulate.
        variability: How much individuals differ. Defaults to `Variability()`.
        seed: Seed for the sampling, so a population can be reproduced.
        duration_h: Length of the simulation.
        n_points: Points on the shared output grid.
        **simulate_kwargs: Passed through to `simulate`.

    Returns:
        A `PopulationResult`.
    """
    total = duration_h if duration_h is not None else regimen.last_event_time_h() + 24.0
    grid = np.linspace(0.0, total, n_points)

    reference = simulate(
        compound, physiology, regimen, times_h=grid, **simulate_kwargs
    )
    assert isinstance(reference, Result)

    individuals = sample_population(compound, physiology, n, variability, seed)

    concentrations = np.zeros((n, grid.size), dtype=float)
    auc = np.zeros(n, dtype=float)
    cmax = np.zeros(n, dtype=float)

    for individual in individuals:
        result = simulate(
            individual.compound,
            individual.physiology,
            regimen,
            times_h=grid,
            **simulate_kwargs,
        )
        assert isinstance(result, Result)
        profile = result.plasma_concentration()
        concentrations[individual.index] = profile
        auc[individual.index] = result.auc("plasma")
        cmax[individual.index] = result.cmax("plasma")

    return PopulationResult(
        time_h=grid,
        concentrations=concentrations,
        auc=auc,
        cmax=cmax,
        individuals=individuals,
        reference=reference,
    )
