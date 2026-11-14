"""Dose administration: bolus, infusion and oral regimens.

Three routes are supported, and they compose freely inside one `Regimen`:

* `IVBolus` -- the whole amount enters venous blood at an instant. The
  integration is restarted at that time with the amount added to the state,
  which is exact and avoids the stiff spike that approximating a bolus as a
  very short infusion would create.
* `IVInfusion` -- a constant rate into venous blood over a window.
* `OralDose` -- the amount enters a gut depot and is absorbed from there by a
  first-order process into the portal circulation, so it reaches the liver
  before the rest of the body. The depot is how first-pass metabolism arises
  naturally rather than being applied as a separate correction factor.

All amounts are in mg, all times in hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["IVBolus", "IVInfusion", "OralDose", "Dose", "Regimen", "repeated"]


@dataclass(frozen=True)
class IVBolus:
    """An instantaneous intravenous dose.

    Attributes:
        amount_mg: Dose amount.
        time_h: Time of administration.
    """

    amount_mg: float
    time_h: float = 0.0

    def __post_init__(self) -> None:
        _check_amount(self.amount_mg)
        _check_time(self.time_h)


@dataclass(frozen=True)
class IVInfusion:
    """A constant-rate intravenous infusion.

    Attributes:
        amount_mg: Total amount delivered over the infusion.
        duration_h: Length of the infusion. Must be positive.
        start_h: Time the infusion starts.
    """

    amount_mg: float
    duration_h: float
    start_h: float = 0.0

    def __post_init__(self) -> None:
        _check_amount(self.amount_mg)
        _check_time(self.start_h, name="start_h")
        if self.duration_h <= 0:
            raise ValueError(f"duration_h must be positive, got {self.duration_h}")

    @property
    def end_h(self) -> float:
        return self.start_h + self.duration_h

    @property
    def rate_mg_per_h(self) -> float:
        return self.amount_mg / self.duration_h

    def rate_at(self, t: float) -> float:
        """Infusion rate in mg/h at time ``t``, zero outside the window."""
        return self.rate_mg_per_h if self.start_h <= t < self.end_h else 0.0


@dataclass(frozen=True)
class OralDose:
    """An oral dose absorbed by a first-order process from a gut depot.

    Attributes:
        amount_mg: Dose amount placed in the gut depot.
        time_h: Time of administration.
        absorption_rate_const_per_h: First-order absorption rate constant (ka).
        bioavailable_fraction: Fraction of the dose that reaches the portal
            circulation, accounting for dissolution and gut wall losses but not
            hepatic first pass, which the liver compartment handles itself.
        lag_h: Delay between administration and the start of absorption.
    """

    amount_mg: float
    time_h: float = 0.0
    absorption_rate_const_per_h: float = 1.0
    bioavailable_fraction: float = 1.0
    lag_h: float = 0.0

    def __post_init__(self) -> None:
        _check_amount(self.amount_mg)
        _check_time(self.time_h)
        if self.absorption_rate_const_per_h <= 0:
            raise ValueError(
                "absorption_rate_const_per_h must be positive, "
                f"got {self.absorption_rate_const_per_h}"
            )
        if not 0.0 < self.bioavailable_fraction <= 1.0:
            raise ValueError(
                f"bioavailable_fraction must be in (0, 1], got {self.bioavailable_fraction}"
            )
        if self.lag_h < 0:
            raise ValueError(f"lag_h must be non-negative, got {self.lag_h}")

    @property
    def depot_entry_time_h(self) -> float:
        """Time the dose becomes available for absorption."""
        return self.time_h + self.lag_h

    @property
    def depot_amount_mg(self) -> float:
        """Amount entering the depot, after bioavailability losses."""
        return self.amount_mg * self.bioavailable_fraction


Dose = IVBolus | IVInfusion | OralDose


@dataclass(frozen=True)
class Regimen:
    """A collection of doses administered to one animal.

    Attributes:
        doses: The individual dose events, in any order.
    """

    doses: tuple[Dose, ...] = field(default_factory=tuple)

    def __init__(self, *doses: Dose | Regimen) -> None:
        flat: list[Dose] = []
        for dose in doses:
            if isinstance(dose, Regimen):
                flat.extend(dose.doses)
            else:
                flat.append(dose)
        object.__setattr__(self, "doses", tuple(flat))

    def __iter__(self):
        return iter(self.doses)

    def __len__(self) -> int:
        return len(self.doses)

    @property
    def total_amount_mg(self) -> float:
        """Total amount administered across every dose."""
        return sum(dose.amount_mg for dose in self.doses)

    def discontinuity_times(self) -> list[float]:
        """Times at which the input rate changes or a bolus lands.

        The solver must not step across these, so it integrates between them
        and is told about them explicitly.
        """
        times: set[float] = set()
        for dose in self.doses:
            if isinstance(dose, IVBolus):
                times.add(dose.time_h)
            elif isinstance(dose, IVInfusion):
                times.add(dose.start_h)
                times.add(dose.end_h)
            elif isinstance(dose, OralDose):
                times.add(dose.depot_entry_time_h)
        return sorted(times)

    def infusion_rate_at(self, t: float) -> float:
        """Combined infusion rate into venous blood at time ``t``, in mg/h."""
        return sum(
            dose.rate_at(t) for dose in self.doses if isinstance(dose, IVInfusion)
        )

    def boluses_at(self, t: float) -> float:
        """Total bolus amount landing exactly at time ``t``, in mg."""
        return sum(
            dose.amount_mg
            for dose in self.doses
            if isinstance(dose, IVBolus) and dose.time_h == t
        )

    def oral_doses_at(self, t: float) -> list[OralDose]:
        """Oral doses whose absorption begins exactly at time ``t``."""
        return [
            dose
            for dose in self.doses
            if isinstance(dose, OralDose) and dose.depot_entry_time_h == t
        ]

    def last_event_time_h(self) -> float:
        """Time of the final dosing event, or 0 for an empty regimen."""
        times = self.discontinuity_times()
        return times[-1] if times else 0.0


def repeated(dose: Dose, *, every_h: float, count: int) -> Regimen:
    """A regimen repeating one dose at a fixed interval.

    Args:
        dose: The dose to repeat. Its own time is used as the first occurrence.
        every_h: Interval between doses.
        count: Total number of doses, including the first.

    Returns:
        A `Regimen` holding ``count`` copies of the dose, shifted in time.

    Raises:
        ValueError: If ``every_h`` is not positive or ``count`` is below 1.
    """
    if every_h <= 0:
        raise ValueError(f"every_h must be positive, got {every_h}")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    doses: list[Dose] = []
    for i in range(count):
        shift = i * every_h
        if isinstance(dose, IVInfusion):
            doses.append(IVInfusion(dose.amount_mg, dose.duration_h, dose.start_h + shift))
        elif isinstance(dose, IVBolus):
            doses.append(IVBolus(dose.amount_mg, dose.time_h + shift))
        else:
            doses.append(
                OralDose(
                    dose.amount_mg,
                    dose.time_h + shift,
                    dose.absorption_rate_const_per_h,
                    dose.bioavailable_fraction,
                    dose.lag_h,
                )
            )
    return Regimen(*doses)


def _check_amount(amount_mg: float) -> None:
    if amount_mg < 0:
        raise ValueError(f"amount_mg must be non-negative, got {amount_mg}")


def _check_time(time_h: float, name: str = "time_h") -> None:
    if time_h < 0:
        raise ValueError(f"{name} must be non-negative, got {time_h}")
