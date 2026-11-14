"""Comparing a prediction against measured data.

Every other test in this package checks that the code does what the model says.
None of them checks that the model says anything true. That is what this module
is for: it puts a simulated profile beside concentrations somebody actually
measured, and reports how far apart they are.

Prediction or fit?
------------------
The distinction decides what a comparison is worth, and it is easy to blur.

A **prediction** parameterises the compound from sources independent of the
data -- measured protein binding, measured clearance, physicochemistry -- then
simulates and compares. Whatever comes out is the answer.

A **fit** adjusts parameters until the curves agree. A fit always looks good
and proves almost nothing on its own; it is a way of estimating parameters, not
of testing a model.

This module computes the same numbers either way, so it cannot tell which you
did. Record it yourself, in `ValidationResult.provenance`, and be honest about
it -- a reader has no way to tell from the plot.

The metrics
-----------
Pharmacokinetic predictions are compared as *ratios*, not differences, because
a 1 mg/L error means something different at a peak of 2 than at a trough of
0.05. So the standard measures are fold errors:

* **Average fold error** (AFE) measures bias -- whether the prediction runs
  systematically high or low. 1.0 is unbiased; 2.0 means it predicts twice the
  truth on average.
* **Absolute average fold error** (AAFE) measures precision, counting errors in
  either direction. It is never below 1 and never below AFE.
* **Fraction within two-fold** is the criterion the field conventionally treats
  as acceptable agreement for a PBPK prediction.

Two-fold is a weak bar. It is the convention because PBPK predictions carry
large parameter uncertainty, not because being wrong by 90% is good.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .results import Result

__all__ = ["ObservedProfile", "ValidationResult", "validate", "validate_many"]


@dataclass(frozen=True)
class ObservedProfile:
    """Concentrations someone measured.

    Attributes:
        time_h: Sample times, in hours.
        concentration: Measured concentration at each time, in mg/L.
        subject: Identifier for the individual or study arm.
        matrix: What was measured -- ``"plasma"``, ``"blood"``, or a tissue
            name. The prediction is compared against the same quantity.
        source: Where the data came from. Record it; a comparison against
            unattributed numbers is not evidence.
    """

    time_h: NDArray[np.float64]
    concentration: NDArray[np.float64]
    subject: str = "observed"
    matrix: str = "plasma"
    source: str = ""

    def __post_init__(self) -> None:
        time = np.asarray(self.time_h, dtype=float)
        values = np.asarray(self.concentration, dtype=float)
        if time.ndim != 1 or values.ndim != 1:
            raise ValueError("time_h and concentration must be one-dimensional")
        if time.size != values.size:
            raise ValueError(
                f"got {time.size} times but {values.size} concentrations"
            )
        if time.size == 0:
            raise ValueError("an observed profile needs at least one measurement")
        if np.any(time < 0):
            raise ValueError("sample times must be non-negative")
        if np.any(np.diff(time) < 0):
            raise ValueError("sample times must be in order")
        object.__setattr__(self, "time_h", time)
        object.__setattr__(self, "concentration", values)


@dataclass(frozen=True)
class ValidationResult:
    """How far a prediction fell from the measurements.

    Attributes:
        observed: The measured concentrations.
        predicted: The simulated concentration at each observed time.
        provenance: ``"prediction"`` when parameters came from sources
            independent of this data, ``"fit"`` when they were adjusted to it.
            Nothing can verify this; it is a label the caller is responsible
            for setting truthfully.
        excluded: How many measurements were dropped because they were zero or
            negative, where a fold error is undefined.
    """

    observed: ObservedProfile
    predicted: NDArray[np.float64]
    provenance: str = "prediction"
    excluded: int = 0
    _ratios: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    @property
    def n(self) -> int:
        """How many measurements the metrics are computed from."""
        return int(self._ratios.size)

    @property
    def fold_errors(self) -> NDArray[np.float64]:
        """Predicted divided by observed, at each usable measurement."""
        return self._ratios

    @property
    def average_fold_error(self) -> float:
        """Bias: the geometric mean of predicted over observed.

        Above 1 the prediction runs high, below 1 it runs low. A model that is
        wildly imprecise but symmetric can still score near 1 here, which is
        why this is always read beside AAFE.
        """
        if self.n == 0:
            return float("nan")
        return float(10.0 ** np.mean(np.log10(self._ratios)))

    @property
    def absolute_average_fold_error(self) -> float:
        """Precision: the geometric mean of the fold error in either direction."""
        if self.n == 0:
            return float("nan")
        return float(10.0 ** np.mean(np.abs(np.log10(self._ratios))))

    @property
    def fraction_within_two_fold(self) -> float:
        """Fraction of measurements predicted within a factor of two."""
        return self.fraction_within(2.0)

    def fraction_within(self, fold: float) -> float:
        """Fraction of measurements predicted within a given fold of the truth.

        Args:
            fold: The factor to allow, at least 1.

        Raises:
            ValueError: If ``fold`` is below 1.
        """
        if fold < 1.0:
            raise ValueError(f"fold must be at least 1, got {fold}")
        if self.n == 0:
            return float("nan")
        inside = (self._ratios >= 1.0 / fold) & (self._ratios <= fold)
        return float(np.mean(inside))

    @property
    def worst_fold_error(self) -> float:
        """The largest miss in either direction, as a fold."""
        if self.n == 0:
            return float("nan")
        return float(np.max(np.maximum(self._ratios, 1.0 / self._ratios)))

    def summary(self) -> dict[str, float | str | int]:
        """Headline numbers describing the agreement."""
        return {
            "subject": self.observed.subject,
            "provenance": self.provenance,
            "n": self.n,
            "excluded": self.excluded,
            "average_fold_error": self.average_fold_error,
            "absolute_average_fold_error": self.absolute_average_fold_error,
            "within_two_fold": self.fraction_within_two_fold,
            "worst_fold_error": self.worst_fold_error,
        }


def validate(
    result: Result,
    observed: ObservedProfile,
    *,
    provenance: str = "prediction",
) -> ValidationResult:
    """Compare a simulation against measured concentrations.

    The prediction is read at the observed sample times, interpolating linearly
    between output points, so the comparison is made where measurements
    actually exist rather than on a grid of the model's choosing.

    Args:
        result: The simulation to judge.
        observed: The measurements to judge it against.
        provenance: ``"prediction"`` or ``"fit"``. See the module docstring --
            this is a label you are responsible for setting truthfully.

    Returns:
        A `ValidationResult`.

    Raises:
        ValueError: If the observed times run past the end of the simulation,
            or provenance is not one of the two allowed labels.
    """
    if provenance not in ("prediction", "fit"):
        raise ValueError(
            f"provenance must be 'prediction' or 'fit', got {provenance!r}"
        )

    last = float(result.time_h[-1])
    if float(observed.time_h[-1]) > last + 1e-9:
        raise ValueError(
            f"observations run to {observed.time_h[-1]:g} h but the simulation "
            f"ends at {last:g} h; extend duration_h so every sample is covered"
        )

    matrix = observed.matrix
    if matrix == "plasma":
        curve = result.plasma_concentration()
    elif matrix == "blood":
        curve = result.blood_concentration()
    else:
        curve = result.tissue_concentration(matrix)

    predicted = np.interp(observed.time_h, result.time_h, curve)

    # A fold error is undefined where nothing was detected, so those points are
    # dropped from the metrics rather than silently distorting them.
    usable = (observed.concentration > 0) & (predicted > 0)
    ratios = predicted[usable] / observed.concentration[usable]

    return ValidationResult(
        observed=observed,
        predicted=predicted,
        provenance=provenance,
        excluded=int(np.sum(~usable)),
        _ratios=ratios,
    )


def validate_many(results: list[ValidationResult]) -> dict[str, float | int]:
    """Pool several per-subject comparisons into one set of metrics.

    Each measurement counts once, so subjects with more samples weigh more --
    which is usually what you want when asking how the model does overall.

    Args:
        results: The per-subject comparisons.

    Returns:
        Pooled metrics, plus how many subjects contributed.

    Raises:
        ValueError: If no results were given.
    """
    if not results:
        raise ValueError("no validation results to pool")

    ratios = np.concatenate([r.fold_errors for r in results if r.n])
    if ratios.size == 0:
        return {"subjects": len(results), "n": 0}

    logs = np.log10(ratios)
    return {
        "subjects": len(results),
        "n": int(ratios.size),
        "average_fold_error": float(10.0 ** np.mean(logs)),
        "absolute_average_fold_error": float(10.0 ** np.mean(np.abs(logs))),
        "within_two_fold": float(
            np.mean((ratios >= 0.5) & (ratios <= 2.0))
        ),
        "within_1_5_fold": float(
            np.mean((ratios >= 1 / 1.5) & (ratios <= 1.5))
        ),
        "worst_fold_error": float(np.max(np.maximum(ratios, 1.0 / ratios))),
    }
