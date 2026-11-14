"""The published-data check, run as a test so it cannot quietly rot.

This is the only test in the suite that compares the model against something
measured rather than against its own internal consistency. It needs the
`Theoph` dataset, which is downloaded rather than vendored, so it skips when
the data is not present instead of failing offline.

To fetch the data, run `python examples/validate_theophylline.py` once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vivosim import OralDose, Regimen, get_physiology, simulate
from vivosim.validation import ObservedProfile, validate, validate_many

EXAMPLES = Path(__file__).parent.parent / "examples"
DATA = EXAMPLES / "data" / "Theoph.csv"

if not DATA.exists():
    pytest.skip(
        "needs examples/data/Theoph.csv; run examples/validate_theophylline.py",
        allow_module_level=True,
    )

sys.path.insert(0, str(EXAMPLES))
from validate_theophylline import (  # noqa: E402
    ABSORPTION_RATE,
    BIOAVAILABILITY,
    load_theoph,
    theophylline,
)


@pytest.fixture(scope="module")
def comparisons():
    """Predict every subject and compare against what was measured."""
    compound = theophylline()
    results = []
    for entry in load_theoph(DATA).values():
        times = np.asarray(entry["time"], dtype=float)
        weight = float(entry["weight"])
        result = simulate(
            compound,
            get_physiology("human", body_weight_kg=weight),
            Regimen(
                OralDose(
                    amount_mg=float(entry["dose_mg_per_kg"]) * weight,
                    absorption_rate_const_per_h=ABSORPTION_RATE,
                    bioavailable_fraction=BIOAVAILABILITY,
                )
            ),
            duration_h=float(times[-1]) + 1.0,
            n_points=2001,
        )
        results.append(
            validate(
                result,
                ObservedProfile(
                    time_h=times,
                    concentration=np.asarray(entry["conc"], dtype=float),
                    matrix="plasma",
                ),
                provenance="prediction",
            )
        )
    return results


def test_the_dataset_is_the_published_one(comparisons) -> None:
    # Twelve subjects, eleven samples each; one per subject is the pre-dose
    # measurement, where a fold error against a predicted zero is undefined.
    assert len(comparisons) == 12
    assert sum(c.n + c.excluded for c in comparisons) == 132


def test_the_prediction_meets_the_conventional_acceptance_bar(comparisons) -> None:
    # The criterion the field uses for a PBPK prediction. Parameters were fixed
    # from the literature before the data was looked at and not adjusted after,
    # so this is a prediction rather than a fit -- which is the only reason the
    # number means anything.
    pooled = validate_many(comparisons)
    assert pooled["within_two_fold"] >= 0.85, pooled


def test_the_prediction_is_not_badly_biased(comparisons) -> None:
    pooled = validate_many(comparisons)
    assert 0.7 <= pooled["average_fold_error"] <= 1.4, pooled


def test_the_prediction_is_reasonably_precise(comparisons) -> None:
    pooled = validate_many(comparisons)
    assert pooled["absolute_average_fold_error"] <= 1.7, pooled


def test_the_error_is_concentrated_in_the_absorption_phase(comparisons) -> None:
    # A fixed literature absorption rate cannot capture how differently
    # individuals absorb, so the early samples miss hardest while distribution
    # and elimination are predicted well. Recording this keeps an honest
    # account of where the model is weak.
    early, late = [], []
    for comparison in comparisons:
        usable = (comparison.observed.concentration > 0) & (comparison.predicted > 0)
        times = comparison.observed.time_h[usable]
        folds = np.abs(np.log10(comparison.fold_errors))
        early.extend(folds[times < 1.0])
        late.extend(folds[times >= 4.0])
    assert np.median(early) > np.median(late)
