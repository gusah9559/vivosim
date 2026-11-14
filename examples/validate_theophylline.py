"""Predict a published theophylline study and see how close it lands.

This is the test that matters. Everything in the test suite checks that the
code does what the model says; this checks whether the model says anything
true, by predicting concentrations somebody actually measured in people.

The data
--------
The `Theoph` dataset: 12 subjects given a single oral dose of theophylline,
each sampled 11 times over 25 hours. It is distributed with R and is one of the
most reproduced datasets in pharmacokinetics, which is exactly why it is a good
benchmark -- anyone can check this result.

    Boeckmann AJ, Sheiner LB, Beal SL (1994). NONMEM Users Guide Part V.
    Distributed with R as `datasets::Theoph`.

It is downloaded rather than vendored, so this repository redistributes no
third-party data and the numbers used are visibly the published ones.

Prediction, not fit
-------------------
Every compound parameter below is a literature value for theophylline, fixed
before the data was looked at, and none is adjusted afterwards. The model is
then run per subject at that subject's own weight and dose. Whatever agreement
comes out is the result -- including if it is poor.

Run it with::

    python examples/validate_theophylline.py
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from vivosim import Compound, OralDose, Regimen, get_physiology, simulate
from vivosim.validation import ObservedProfile, validate, validate_many

THEOPH_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/"
    "csv/datasets/Theoph.csv"
)

# Literature parameters for theophylline, fixed in advance.
#
#   fraction unbound in plasma   ~0.6   (about 40% bound, mostly to albumin)
#   blood:plasma ratio           ~0.82  (little uptake into red cells)
#   tissue partitioning          ~0.7   (distributes close to total body water;
#                                        its volume of distribution is ~0.5 L/kg,
#                                        well below the ~0.6 L/kg of body water)
#   hepatic intrinsic clearance         (set so that the well-stirred model gives
#                                        the textbook ~2.8 L/h in a 70 kg adult;
#                                        theophylline is a low-extraction drug)
#   absorption rate constant     ~1.5/h (rapid and essentially complete from the
#                                        oral solution used in this study)
FRACTION_UNBOUND = 0.60
BLOOD_PLASMA_RATIO = 0.82
PARTITION_COEFFICIENT = 0.70
ABSORPTION_RATE = 1.5
BIOAVAILABILITY = 1.0

# A low-extraction compound's clearance is approximately CL_int * fu, so the
# intrinsic clearance implied by a whole-body clearance of 2.8 L/h is 2.8/fu.
# Referenced to unbound concentration, as the model expects.
TARGET_CLEARANCE_L_PER_H = 2.8
HEPATIC_CL_INT = TARGET_CLEARANCE_L_PER_H / FRACTION_UNBOUND


def theophylline() -> Compound:
    """Theophylline, parameterised from the literature and not from the data."""
    return Compound(
        name="theophylline",
        default_partition_coefficient=PARTITION_COEFFICIENT,
        blood_plasma_ratio=BLOOD_PLASMA_RATIO,
        fraction_unbound_plasma=FRACTION_UNBOUND,
        hepatic_intrinsic_clearance_L_per_h=HEPATIC_CL_INT,
        molecular_weight=180.16,
    )


def load_theoph(cache: Path) -> dict[str, dict[str, object]]:
    """Download the dataset if needed and group it by subject."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {THEOPH_URL}")
        urllib.request.urlopen(THEOPH_URL, timeout=60)  # noqa: S310 - fixed https URL
        with urllib.request.urlopen(THEOPH_URL, timeout=60) as response:  # noqa: S310
            cache.write_bytes(response.read())

    rows = list(csv.DictReader(cache.read_text(encoding="utf-8").splitlines()))
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"time": [], "conc": [], "weight": 0.0, "dose_mg_per_kg": 0.0}
    )
    for row in rows:
        subject = row["Subject"]
        entry = grouped[subject]
        entry["time"].append(float(row["Time"]))  # type: ignore[union-attr]
        entry["conc"].append(float(row["conc"]))  # type: ignore[union-attr]
        entry["weight"] = float(row["Wt"])
        entry["dose_mg_per_kg"] = float(row["Dose"])
    return grouped


def main() -> int:
    cache = Path(__file__).parent / "data" / "Theoph.csv"
    try:
        subjects = load_theoph(cache)
    except Exception as error:  # noqa: BLE001 - the script is useless without data
        print(f"could not obtain the dataset: {error}")
        return 1

    compound = theophylline()
    comparisons = []

    print()
    print("Theophylline, single oral dose - predicted against measured")
    print(f"  {len(subjects)} subjects, parameters fixed from the literature")
    print()
    print(f"  {'subject':>8} {'wt kg':>7} {'dose mg':>8} {'n':>3} "
          f"{'bias':>7} {'fold err':>9} {'<2x':>6}")

    for subject, entry in sorted(subjects.items(), key=lambda kv: int(kv[0])):
        times = np.asarray(entry["time"], dtype=float)
        weight = float(entry["weight"])  # type: ignore[arg-type]
        dose_mg = float(entry["dose_mg_per_kg"]) * weight  # type: ignore[arg-type]

        result = simulate(
            compound,
            get_physiology("human", body_weight_kg=weight),
            Regimen(
                OralDose(
                    amount_mg=dose_mg,
                    absorption_rate_const_per_h=ABSORPTION_RATE,
                    bioavailable_fraction=BIOAVAILABILITY,
                )
            ),
            duration_h=float(times[-1]) + 1.0,
            n_points=2001,
        )

        observed = ObservedProfile(
            time_h=times,
            concentration=np.asarray(entry["conc"], dtype=float),
            subject=f"subject {subject}",
            matrix="plasma",
            source="Theoph (Boeckmann, Sheiner & Beal 1994)",
        )
        comparison = validate(result, observed, provenance="prediction")
        comparisons.append(comparison)

        print(
            f"  {subject:>8} {weight:>7.1f} {dose_mg:>8.1f} {comparison.n:>3} "
            f"{comparison.average_fold_error:>7.2f} "
            f"{comparison.absolute_average_fold_error:>9.2f} "
            f"{comparison.fraction_within_two_fold:>5.0%}"
        )

    pooled = validate_many(comparisons)
    print()
    print("  pooled over every measurement")
    print(f"    measurements        {pooled['n']}")
    print(f"    bias (AFE)          {pooled['average_fold_error']:.2f}"
          "   (1.00 would be unbiased)")
    print(f"    precision (AAFE)    {pooled['absolute_average_fold_error']:.2f}"
          "   (1.00 would be exact)")
    print(f"    within 1.5-fold     {pooled['within_1_5_fold']:.0%}")
    print(f"    within 2-fold       {pooled['within_two_fold']:.0%}"
          "   (the conventional acceptance bar)")
    print(f"    worst single point  {pooled['worst_fold_error']:.2f}x")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
