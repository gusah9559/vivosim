"""Simulating a population rather than one animal."""

from __future__ import annotations

import numpy as np
import pytest

from vivosim import (
    RAT,
    Compound,
    IVBolus,
    Metabolism,
    Regimen,
    Variability,
    sample_population,
    simulate,
    simulate_population,
)

N = 40
SEED = 7


def low_extraction() -> Compound:
    """A compound the liver clears slowly, so enzyme capacity sets its fate."""
    return Compound(
        name="low extraction",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.1,
        hepatic_intrinsic_clearance_L_per_h=0.2,
    )


def high_extraction() -> Compound:
    """A compound cleared as fast as blood can deliver it to the liver."""
    return Compound(
        name="high extraction",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=1.0,
        hepatic_intrinsic_clearance_L_per_h=400.0,
    )


def population(compound: Compound | None = None, **kwargs):
    return simulate_population(
        compound if compound is not None else low_extraction(),
        RAT,
        Regimen(IVBolus(1.0)),
        n=kwargs.pop("n", N),
        seed=kwargs.pop("seed", SEED),
        duration_h=kwargs.pop("duration_h", 24),
        **kwargs,
    )


# -- sampling -------------------------------------------------------------


def test_variability_rejects_negative_spread() -> None:
    with pytest.raises(ValueError, match="hepatic_clearance must be non-negative"):
        Variability(hepatic_clearance=-0.1)


def test_a_population_needs_at_least_one_individual() -> None:
    with pytest.raises(ValueError, match="at least one individual"):
        sample_population(low_extraction(), RAT, 0)


def test_sampling_preserves_the_mean() -> None:
    # A lognormal draw placed naively sits below its own reference, which would
    # make every population quietly lighter and slower than the animal it was
    # built from. The shift that prevents that is worth pinning down.
    people = sample_population(low_extraction(), RAT, 4000, seed=SEED)
    weights = np.array([p.physiology.body_weight_kg for p in people])
    assert weights.mean() == pytest.approx(RAT.body_weight_kg, rel=0.02)

    clearances = np.array(
        [p.compound.hepatic_intrinsic_clearance_L_per_h for p in people]
    )
    reference = low_extraction().hepatic_intrinsic_clearance_L_per_h
    assert clearances.mean() == pytest.approx(reference, rel=0.03)


def test_sampled_values_stay_physical() -> None:
    people = sample_population(low_extraction(), RAT, 500, seed=SEED)
    for person in people:
        assert person.physiology.body_weight_kg > 0
        assert person.compound.hepatic_intrinsic_clearance_L_per_h >= 0
        # An unbound fraction is a proportion and a draw must not escape it.
        assert 0.0 < person.compound.fraction_unbound_plasma <= 1.0


def test_sampling_is_reproducible_from_a_seed() -> None:
    first = sample_population(low_extraction(), RAT, 20, seed=11)
    second = sample_population(low_extraction(), RAT, 20, seed=11)
    other = sample_population(low_extraction(), RAT, 20, seed=12)

    weights = lambda group: [p.physiology.body_weight_kg for p in group]  # noqa: E731
    assert weights(first) == weights(second)
    assert weights(first) != weights(other)


def test_scaling_a_saturable_pathway_moves_vmax_not_km() -> None:
    # Individuals differ in how much enzyme they express, not in the enzyme's
    # affinity for the substrate.
    compound = Compound(
        name="saturable",
        default_partition_coefficient=2.0,
        metabolism=(Metabolism(enzyme="CYP3A4", vmax_mg_per_h=2.0, km_mg_per_L=0.5),),
    )
    people = sample_population(compound, RAT, 50, seed=SEED)
    assert all(p.compound.metabolism[0].km_mg_per_L == 0.5 for p in people)
    assert len({p.compound.metabolism[0].vmax_mg_per_h for p in people}) > 1


def test_no_variability_gives_identical_individuals() -> None:
    people = sample_population(
        low_extraction(), RAT, 25, Variability.none(), seed=SEED
    )
    assert len({p.physiology.body_weight_kg for p in people}) == 1
    assert len({p.compound.fraction_unbound_plasma for p in people}) == 1


# -- the population result ------------------------------------------------


def test_a_population_without_variability_matches_one_animal() -> None:
    # The control that anchors everything: identical individuals must reproduce
    # the single-animal simulation exactly.
    result = population(variability=Variability.none(), n=5)
    reference_auc = result.reference.auc("plasma")
    assert np.allclose(result.auc, reference_auc, rtol=1e-9)
    assert np.allclose(result.concentrations[0], result.reference.plasma_concentration())


def test_every_individual_is_simulated_on_the_shared_grid() -> None:
    result = population()
    assert result.size == N
    assert result.concentrations.shape == (N, result.time_h.size)
    assert np.array_equal(result.time_h, result.reference.time_h)


def test_variability_spreads_the_exposure() -> None:
    result = population()
    assert result.auc.std() > 0
    assert result.fold_range() > 1.0


def test_the_percentile_band_is_ordered() -> None:
    low, median, high = population().percentile_band(5, 95)
    assert np.all(low <= median + 1e-12)
    assert np.all(median <= high + 1e-12)


def test_a_wider_band_contains_a_narrower_one() -> None:
    result = population()
    inner_low, _, inner_high = result.percentile_band(25, 75)
    outer_low, _, outer_high = result.percentile_band(5, 95)
    assert np.all(outer_low <= inner_low + 1e-12)
    assert np.all(outer_high >= inner_high - 1e-12)


def test_bad_percentiles_are_rejected() -> None:
    result = population(n=5)
    with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
        result.percentile_band(95, 5)


def test_more_variability_widens_the_distribution() -> None:
    tight = population(variability=Variability(hepatic_clearance=0.05)).fold_range()
    loose = population(variability=Variability(hepatic_clearance=0.8)).fold_range()
    assert loose > tight


# -- physiology the model should reproduce --------------------------------


def test_enzyme_variability_matters_less_for_a_high_extraction_compound() -> None:
    # A compound the liver clears as fast as blood arrives is limited by blood
    # flow, not by enzyme, so differences in enzyme expression are buffered.
    # One cleared slowly is limited by enzyme and inherits its full spread.
    # Reproducing this is a real check that clearance was wired through the
    # circulation rather than applied to the compound in isolation.
    spread = Variability(
        body_weight=0.0,
        cardiac_output=0.0,
        hepatic_clearance=0.5,
        renal_clearance=0.0,
        fraction_unbound=0.0,
        partition_coefficients=0.0,
    )
    low = population(low_extraction(), variability=spread).fold_range()
    high = population(high_extraction(), variability=spread).fold_range()
    assert low > high, (
        f"a low-extraction compound should inherit enzyme variability "
        f"({low:.2f}x) more than a high-extraction one ({high:.2f}x)"
    )


def test_a_heavier_individual_sees_a_fixed_dose_more_dilutely() -> None:
    spread = Variability(
        body_weight=0.3,
        cardiac_output=0.0,
        hepatic_clearance=0.0,
        renal_clearance=0.0,
        fraction_unbound=0.0,
        partition_coefficients=0.0,
    )
    result = population(variability=spread)
    weights = np.array([p.physiology.body_weight_kg for p in result.individuals])
    # The same milligrams spread through a bigger animal give lower exposure.
    assert np.corrcoef(weights, result.auc)[0, 1] < -0.8


# -- probabilities --------------------------------------------------------


def test_probability_above_counts_the_tail() -> None:
    result = population()
    median_cmax = float(np.median(result.cmax))
    assert result.probability_above(median_cmax) == pytest.approx(0.5, abs=0.1)
    assert result.probability_above(0.0) == 1.0
    assert result.probability_above(result.cmax.max() * 10) == 0.0


def test_probability_within_a_window_counts_the_middle() -> None:
    result = population()
    assert result.probability_auc_within(0.0, result.auc.max() * 2) == 1.0
    narrow = result.probability_auc_within(
        float(np.percentile(result.auc, 25)), float(np.percentile(result.auc, 75))
    )
    assert narrow == pytest.approx(0.5, abs=0.1)


def test_an_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="high > low"):
        population(n=5).probability_auc_within(10.0, 10.0)


def test_probability_auc_above_complements_the_window() -> None:
    result = population()
    median = float(np.median(result.auc))
    assert result.probability_auc_above(median) == pytest.approx(0.5, abs=0.1)


def test_summary_reports_the_spread() -> None:
    summary = population().summary()
    assert summary["n"] == N
    assert summary["auc_p5"] <= summary["auc_median"] <= summary["auc_p95"]
    assert summary["auc_fold_range"] >= 1.0


def test_an_individual_profile_is_a_real_simulation() -> None:
    # Each row must be the individual's own simulation, not a scaled copy of
    # the reference, so re-running one by hand has to reproduce it.
    result = population(n=3)
    person = result.individuals[1]
    direct = simulate(
        person.compound, person.physiology, Regimen(IVBolus(1.0)), times_h=result.time_h
    )
    assert np.allclose(result.concentrations[1], direct.plasma_concentration())
