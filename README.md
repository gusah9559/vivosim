# vivosim

Physiologically based pharmacokinetic (PBPK) simulation for preclinical species.

Given a compound's distribution and clearance parameters, an animal's
physiology and a dosing regimen, vivosim computes how much of the compound is
in the blood and in each tissue over time.

> **Scope.** This is a research and teaching tool for numerical experiments.
> It is not a clinical, veterinary or regulatory tool, it does not recommend
> doses, and its output is not evidence about any real compound. A PBPK model
> is only as good as the parameters it is given, and the built-in physiological
> values are literature means for a typical healthy animal. Nothing here
> replaces measured data or expert review.

## Why PBPK

A classical compartmental PK model fits a curve to blood samples. It describes
what happened, but its compartments have no anatomical meaning, so it cannot
answer where the compound went or extrapolate to a new dose route, a new
species or a new body weight.

A PBPK model instead builds the animal: real organ volumes, real blood flows,
one compartment per tissue, connected in the actual circulatory topology. The
parameters separate cleanly into *what the animal is* and *what the compound
is*, so changing one does not require refitting the other. That is what lets
the same compound be simulated in a mouse and then a rat, or moved from
intravenous to oral dosing, without starting over.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+, NumPy and SciPy.

## Use

```python
from vivosim import Compound, IVBolus, Regimen, get_physiology, simulate

compound = Compound(
    name="example",
    partition_coefficients={"adipose": 12.0, "muscle": 3.0, "brain": 0.4},
    default_partition_coefficient=1.0,
    blood_plasma_ratio=1.1,
    fraction_unbound_plasma=0.15,
    hepatic_intrinsic_clearance_L_per_h=2.0,
)

result = simulate(
    compound,
    get_physiology("rat", body_weight_kg=0.25),
    Regimen(IVBolus(amount_mg=1.0)),
    duration_h=24,
)

print(result.summary())
print(result.tissue_concentration("adipose")[:5])
result.to_csv("profile.csv")
```

Oral and infusion routes, and repeated dosing, compose in the same `Regimen`:

```python
from vivosim import IVInfusion, OralDose, Regimen, repeated

Regimen(
    IVInfusion(amount_mg=2.0, duration_h=1.0),
    OralDose(amount_mg=5.0, time_h=6.0, absorption_rate_const_per_h=0.9),
)

repeated(OralDose(amount_mg=5.0, absorption_rate_const_per_h=0.9), every_h=12, count=6)
```

## Starting from a molecular structure

The partition coefficients above do not have to be supplied by hand. They can
be predicted from the molecule itself, because a compound distributes into a
tissue by dissolving in that tissue's lipid and water — so lipophilicity,
ionization state and the known lipid content of each organ are enough to
compute where it goes.

```python
from vivosim import compound_from_properties, simulate, get_physiology, IVBolus, Regimen
from vivosim.structure import properties_from_smiles

properties = properties_from_smiles(
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen
    fraction_unbound_plasma=0.01,   # measured, not predicted
    pka=4.4,                        # measured, not predicted
)
compound = compound_from_properties(properties, hepatic_intrinsic_clearance_L_per_h=1.2)

result = simulate(compound, get_physiology("rat"), Regimen(IVBolus(1.0)), duration_h=24)
```

Structure reading needs RDKit, an optional dependency (`pip install rdkit`).
Without it the rest of the package works unchanged and you supply `Kp` values
directly.

### What is predicted, and what is measured

This distinction is the whole honesty of the method, so it is enforced in the
API rather than left to the docs — an ionizable structure given without a pKa
raises rather than silently defaulting to neutral.

| Quantity | Source |
|---|---|
| Molecular weight, polar surface area | Exact, from the structure |
| logP | Estimated from the structure (Crippen), ~1 log unit of scatter |
| Tissue partition coefficients | Predicted, from logP + pKa + tissue composition |
| pKa | **Measured.** Not predicted here; a wrong pKa distorts every Kp |
| Plasma protein binding | **Measured.** Not predictable to the needed accuracy |
| Metabolic clearance | **Measured.** Normally scaled from microsomes or hepatocytes |

What this buys you is *distribution*: where the compound goes and how long it
stays. That is the difference between a compound cleared in an hour and one
that sits in fat for a week, and it does fall out of structure alone.

What it does not buy you is *effect*. Nothing here predicts whether a compound
works, or what it does to an animal. Predicted partition coefficients are a
structured starting hypothesis to be replaced by measurement, not an answer.
The method is known to do poorly on compounds that are actively transported,
that bind tissue components specifically, or that are strong bases accumulating
in acidic phospholipid — for which the Rodgers & Rowland extension is the
better choice and is not implemented yet.

Check a prediction before building on it. The cheapest check is the volume of
distribution, which follows from the Kp values alone and is widely measured:

```python
from vivosim import volume_of_distribution_L_per_kg

volume_of_distribution_L_per_kg(compound, get_physiology("rat"))
```

## Model structure

Blood leaves the venous pool, crosses the lung, and becomes arterial blood that
perfuses every tissue. Most tissues drain back to the venous pool. Gut and
spleen carry portal flow and drain into the liver, so an orally absorbed
compound meets hepatic metabolism before it reaches the rest of the body —
first-pass extraction emerges from the topology rather than being applied as a
correction factor.

Each tissue is one well-mixed compartment at perfusion limitation:

```
dA_T/dt = Q_T * (C_art - C_T / Kp_blood_T)
```

Hepatic metabolism is driven by the unbound concentration in blood leaving the
liver, which makes the liver well-stirred within the circulation. Renal
clearance of unchanged compound is referenced to blood leaving the kidney.

Units are mg, L, L/h, mg/L and hours throughout.

### What the model does not do yet

Named so that nobody mistakes an absent mechanism for a prediction:

- Permeability-limited tissue uptake, for compounds that do not equilibrate
  instantly with their tissue.
- Saturable (Michaelis-Menten) metabolism, so exposure is strictly linear in
  dose.
- Active transport, enterohepatic recycling, biliary excretion.
- Metabolites. Only the parent compound is tracked.
- Drug-drug interactions.
- Partitioning of strong bases into acidic phospholipid (Rodgers & Rowland).
- pKa prediction from structure.
- Population variability. One simulation describes one animal.

## Validation

The test suite checks the engine against limits where the answer is known
independently of the implementation:

- **Conservation of mass** for every route and regimen. Amount in the body plus
  amount eliminated equals amount administered, to solver tolerance.
- **Equilibrium distribution.** With elimination switched off, every tissue
  must settle at exactly `Kp` times the plasma concentration.
- **Steady state under infusion** must match the well-stirred hepatic clearance
  `CL = Q_h * CL_int_u / (Q_h + CL_int_u)` computed by hand.
- **Flow and volume closure.** Tissue blood flows sum to cardiac output and
  volumes sum to body weight, both exactly.
- **First pass.** An oral dose gives lower exposure than the same intravenous
  dose when the liver clears the compound, and identical exposure when it does
  not.
- **Real molecules.** Computed molecular weights match the true values; logP
  ranks caffeine below ibuprofen below diazepam; ibuprofen, an ionized and
  heavily bound acid, is predicted to stay near plasma volume, as it measurably
  does, while neutral lipophilic diazepam distributes into tissue.

```bash
pytest
```

## Physiological parameters

Organ volumes and blood flows for rat and mouse are literature means from:

> Brown RP, Delp MD, Lindstedt SL, Rhomberg LR, Beliles RP (1997).
> "Physiological parameter values for physiologically based pharmacokinetic
> models." *Toxicology and Industrial Health* 13(4):407-484.

Each species includes a lumped `rest` compartment carrying the balance of body
weight and cardiac output, so that both close exactly; without it the named
tissues account for only about 90% of flow in the rat and the circulation would
not conserve mass.

Override anything with values from your own study:

```python
from vivosim import TissueSpec, get_physiology

phys = get_physiology("rat", 0.28).with_tissue("liver", TissueSpec(0.040, 0.019))
```

## License

Apache-2.0. See [LICENSE](LICENSE).
