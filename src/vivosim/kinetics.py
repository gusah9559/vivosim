"""Enzyme kinetics: saturable metabolism and competition between compounds.

Why this exists
---------------
First-order clearance says a fixed *fraction* of what is present is removed per
unit time, so exposure is strictly proportional to dose and two compounds given
together never notice each other. Real metabolism is done by a finite number of
enzyme molecules. Supply enough substrate and they saturate: clearance falls,
exposure climbs faster than dose, and -- the part that matters clinically -- a
second compound competing for the same enzyme displaces the first.

Drug-drug interactions all start here. A compound that merely occupies an
enzyme raises the exposure of everything else that enzyme clears, which is why
combinations can be dangerous at doses that are each safe alone.

The model
---------
Metabolism of substrate ``i`` by enzyme ``E`` follows Michaelis-Menten kinetics
with competitive inhibition by everything else bound to ``E``::

    v_i = Vmax_i * Cu_i / (Km_i * (1 + sum_j Cu_j / Ki_j) + Cu_i)

Concentrations are unbound, since only free compound reaches the enzyme. Each
competing *substrate* of the same enzyme inhibits with ``Ki = Km`` -- occupying
the site is what competes, whether or not the occupier is turned over -- and a
pure inhibitor declares its own ``Ki`` without being metabolised.

Competitive inhibition raises apparent Km and leaves Vmax untouched: enough
substrate still outruns the inhibitor. Other mechanisms (non-competitive,
mechanism-based inactivation, induction) behave differently and are not
implemented; see the README for what is missing.

At concentrations well below Km the expression collapses to first-order
clearance with ``CL_int = Vmax / Km``, so a saturable compound behaves linearly
when it is dilute, as it should.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Metabolism", "Inhibition", "metabolic_rates", "intrinsic_clearance"]


@dataclass(frozen=True)
class Metabolism:
    """One enzyme's saturable handling of a compound.

    Attributes:
        enzyme: Name of the enzyme, used to decide who competes with whom. Any
            label works as long as it is used consistently across compounds.
        vmax_mg_per_h: Maximum metabolic rate once the enzyme is saturated.
            This is the ceiling that makes exposure non-linear in dose.
        km_mg_per_L: Unbound concentration giving half the maximum rate. Low Km
            means the enzyme saturates early.
    """

    enzyme: str
    vmax_mg_per_h: float
    km_mg_per_L: float

    def __post_init__(self) -> None:
        if not self.enzyme:
            raise ValueError("enzyme must be a non-empty name")
        if self.vmax_mg_per_h < 0:
            raise ValueError(f"vmax_mg_per_h must be non-negative, got {self.vmax_mg_per_h}")
        if self.km_mg_per_L <= 0:
            raise ValueError(f"km_mg_per_L must be positive, got {self.km_mg_per_L}")

    @property
    def intrinsic_clearance_L_per_h(self) -> float:
        """Clearance in the linear regime, where concentration is well below Km."""
        return self.vmax_mg_per_h / self.km_mg_per_L


@dataclass(frozen=True)
class Inhibition:
    """A compound's competitive inhibition of an enzyme it need not be a substrate of.

    Attributes:
        enzyme: The enzyme inhibited.
        ki_mg_per_L: Unbound inhibition constant. Lower means a stronger
            inhibitor; it is the concentration at which apparent Km doubles.
    """

    enzyme: str
    ki_mg_per_L: float

    def __post_init__(self) -> None:
        if not self.enzyme:
            raise ValueError("enzyme must be a non-empty name")
        if self.ki_mg_per_L <= 0:
            raise ValueError(f"ki_mg_per_L must be positive, got {self.ki_mg_per_L}")


def intrinsic_clearance(
    metabolism: tuple[Metabolism, ...], linear_clearance_L_per_h: float = 0.0
) -> float:
    """Total intrinsic clearance in the linear regime.

    Sums the low-concentration limit of every saturable pathway with any
    explicitly first-order clearance. Useful for comparing a saturable compound
    against a linear one, and for predicting what it will do at a low dose.

    Args:
        metabolism: The compound's saturable pathways.
        linear_clearance_L_per_h: Any additional first-order clearance.

    Returns:
        Intrinsic clearance in L/h, referenced to unbound concentration.
    """
    return linear_clearance_L_per_h + sum(
        pathway.intrinsic_clearance_L_per_h for pathway in metabolism
    )


def metabolic_rates(
    pathways: list[tuple[int, Metabolism]],
    inhibitors: dict[str, list[tuple[int, float]]],
    unbound: list[float],
) -> list[float]:
    """Metabolic rate for each compound, accounting for competition.

    Args:
        pathways: Each saturable pathway as ``(compound index, Metabolism)``. A
            compound may appear more than once if several enzymes clear it.
        inhibitors: For each enzyme, the ``(compound index, Ki)`` pairs that
            compete for it. Substrates appear here with their own Km.
        unbound: Unbound concentration of each compound at the enzyme, in mg/L,
            indexed as the compound indices above.

    Returns:
        Total metabolic rate per compound in mg/h, in compound index order.
    """
    rates = [0.0] * len(unbound)

    for index, pathway in pathways:
        concentration = unbound[index]
        if concentration <= 0.0 or pathway.vmax_mg_per_h == 0.0:
            continue

        # Everything else occupying the enzyme raises the apparent Km. A
        # compound does not compete with itself: its own occupancy is already
        # in the Michaelis-Menten denominator.
        occupancy = 0.0
        for other, ki in inhibitors.get(pathway.enzyme, ()):
            if other == index:
                continue
            if unbound[other] > 0.0:
                occupancy += unbound[other] / ki

        apparent_km = pathway.km_mg_per_L * (1.0 + occupancy)
        rates[index] += (
            pathway.vmax_mg_per_h * concentration / (apparent_km + concentration)
        )

    return rates
