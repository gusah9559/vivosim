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

## Saturable metabolism and drug-drug interactions

First-order clearance removes a fixed *fraction* of what is present, so
exposure is proportional to dose and two compounds given together never notice
each other. Real metabolism is done by a finite number of enzyme molecules.
Supply enough substrate and they saturate — and a second compound competing for
the same enzyme displaces the first.

Declare a saturable pathway instead of a clearance, and name the enzyme:

```python
from vivosim import Compound, Inhibition, Metabolism, simulate

victim = Compound(
    name="victim",
    fraction_unbound_plasma=0.5,
    metabolism=(Metabolism(enzyme="CYP3A4", vmax_mg_per_h=2.0, km_mg_per_L=0.5),),
)
```

Exposure then stops being proportional to dose, which is exactly the regime in
which a linear model is dangerous:

| Dose | AUC | AUC/dose |
|---|---|---|
| 0.1 mg | 1.64 | 16.4 |
| 1 mg | 16.7 | 16.7 |
| 10 mg | 206 | 20.6 |
| 100 mg | 6705 | 67.1 |

Below Km the compound behaves exactly like a first-order one with
`CL_int = Vmax/Km` — the test suite pins that limit — and above it, clearance
falls away.

Compounds simulated **together** compete for any enzyme they share:

```python
inhibitor = Compound(
    name="inhibitor",
    fraction_unbound_plasma=0.5,
    inhibits=(Inhibition(enzyme="CYP3A4", ki_mg_per_L=0.05),),
)

study = simulate([victim, inhibitor], physiology, [victim_doses, inhibitor_doses])

study["victim"].auc("plasma")     # 19.6, against 6.2 for the victim alone
study.interacting_enzymes()       # {'CYP3A4': ['victim', 'inhibitor']}
```

This cannot be reproduced by simulating each compound separately and overlaying
the results: the interaction is a property of the pair. A substrate competes
through its own Km — occupying the site is what matters, not whether the
occupier is turned over — and a pure inhibitor declares a Ki without being
metabolised. Compounds on *different* enzymes are verified to be unaffected by
each other, so the coupling exists only where it should.

Only **competitive** inhibition is modelled: apparent Km rises, Vmax does not,
and enough substrate still outruns the inhibitor. Non-competitive inhibition,
mechanism-based inactivation and enzyme induction behave differently and are
not implemented.

## Viewer

```python
from vivosim.viewer import render_html, show

render_html(result, "profile.html")   # a file you can mail, attach or commit
show(result)                          # opens it, natively if pywebview is installed
```

`render_html` writes **one self-contained HTML file**. The data is embedded in
it and nothing is fetched from a network, so it opens on any machine, works
offline, and can be sent to someone who has never installed vivosim. For a
scientific tool that matters more than a prettier window: a result that only
exists inside a running session is a result nobody else can check.

`show` opens that same file in a native window when
[`pywebview`](https://pywebview.flowrl.com/) is installed (`pip install
"vivosim[app]"`), so it behaves like a desktop application — one implementation,
two ways to open it.

The page carries:

- **Headline metrics** — peak plasma, time of peak, AUC, Vss, fraction
  eliminated, and the mass balance error, so a bad integration is visible rather
  than hidden.
- **Concentration over time**, log by default as PK convention expects, with a
  crosshair readout, selectable tissues and dose markers.
- **An anatomical view** where each organ is shaded by its concentration, with a
  time scrubber. Watching a dose cross the lung, load the liver and leave again
  tells you more about a compound in five seconds than a table does in five
  minutes. Shading is logarithmic so lightly loaded organs stay legible beside
  the most loaded one.
- **Small multiples** of every tissue on a shared scale.
- **A table view** of every plotted value, so nothing is visible only as colour.

Charts are drawn as inline SVG with no plotting library, which is what keeps the
file self-contained. Light and dark are both designed rather than inverted, and
the categorical palette is validated for colour-vision deficiency.

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
- Active transport, enterohepatic recycling, biliary excretion.
- Non-competitive inhibition, mechanism-based inactivation, enzyme induction.
- Metabolites. Only the parent compound is tracked.
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
- **The saturation limit.** Far below Km a saturable compound reproduces a
  first-order one with `CL_int = Vmax/Km` exactly, and above Km no more than
  `Vmax * duration` can be cleared however much is given.
- **Interaction specificity.** Compounds sharing an enzyme raise each other's
  exposure; compounds on different enzymes reproduce their solo profiles.
  Overwhelming inhibition converges on the no-metabolism case.
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
