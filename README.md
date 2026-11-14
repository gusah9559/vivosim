# vivosim

[![tests](https://github.com/gusah9559/vivosim/actions/workflows/ci.yml/badge.svg)](https://github.com/gusah9559/vivosim/actions/workflows/ci.yml)

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

## Quickstart

```bash
pip install -e .
vivosim demo interaction --open
```

That renders a worked example and opens it. To run your own study, write one
small text file — no Python needed:

```bash
vivosim new study.toml      # a commented starter file
vivosim run study.toml --open
```

`study.toml` looks like this. Units are milligrams, litres and hours
throughout:

```toml
[species]
name = "rat"                 # rat, mouse or human
body_weight_kg = 0.25
# condition = "hepatic impairment (moderate)"   # see: vivosim conditions

[simulation]
duration_h = 48
# population = 200           # simulate this many individuals instead of one
# seed = 42

[[compound]]
name = "drug A"
fraction_unbound_plasma = 0.2          # measured
default_partition_coefficient = 2.0    # or: smiles = "CC(C)Cc1ccc..."
hepatic_intrinsic_clearance_L_per_h = 2.0

  [[compound.dose]]
  route = "iv"               # iv, infusion or oral
  amount_mg = 1.0
```

Add a second `[[compound]]` and they are simulated **together**, competing for
any enzyme they share, and compared against their solo runs. Add
`population = 200` and you get the spread across individuals instead of one
animal. The page it renders is a single self-contained HTML file.

| Command | |
|---|---|
| `vivosim demo` | list the built-in examples |
| `vivosim new study.toml` | write a commented starter file |
| `vivosim run study.toml --open` | simulate and open the result |
| `vivosim conditions` | list the disease states |
| `vivosim validate` | check the model against published data |

Requires Python 3.10+, NumPy and SciPy. Reading molecular structures also needs
RDKit (`pip install "vivosim[chem]"`).

## From Python

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

`assess_interaction` makes the comparison the field actually asks for, running
each compound twice on the identical regimen and time grid -- once beside the
others, once by itself:

```python
from vivosim import assess_interaction

report = assess_interaction([victim, inhibitor], physiology, regimens, duration_h=72)

report["victim"].auc_ratio        # 3.15
report["victim"].classification   # 'moderate'
report.affected()                 # [victim] -- the inhibitor is unchanged
```

The ratio is banded by the thresholds regulators use for interaction strength
(>=5 strong, >=2 moderate, >=1.25 weak). Those bands describe clinical
significance in humans and this is a simulated rat; the vocabulary is borrowed
because it is how the field talks about magnitude, not because a number from
here is a finding about anything.

None of this can be reproduced by simulating each compound separately and
overlaying the results: the interaction is a property of the pair. A substrate competes
through its own Km — occupying the site is what matters, not whether the
occupier is turned over — and a pure inhibitor declares a Ki without being
metabolised. Compounds on *different* enzymes are verified to be unaffected by
each other, so the coupling exists only where it should.

Only **competitive** inhibition is modelled: apparent Km rises, Vmax does not,
and enough substrate still outruns the inhibitor. Non-competitive inhibition,
mechanism-based inactivation and enzyme induction behave differently and are
not implemented.

## Disease states

Every parameter set above describes a healthy animal, and real subjects often
are not. A damaged liver can leave several times the exposure from the same
dose — which is how a routine dose becomes an overdose.

```python
from vivosim import hepatic_impairment, obesity, renal_impairment

sick_compound, sick_body = hepatic_impairment("severe").apply(compound, physiology)
```

A condition changes both sides of the problem, because disease does not respect
that separation: hepatic impairment reduces liver blood flow *and* enzyme
capacity *and* raises the unbound fraction, and all three push exposure the
same way. Flows and volumes are rebalanced so the animal still conserves mass —
a condition that broke that would corrupt every number downstream, so it is
tested for every condition shipped.

These are *pharmacokinetic* states: a body altered by disease, moving a
compound around differently. Nothing here models the disease itself, its
progression, or whether a compound treats it. An impaired animal here is a
healthy animal with different numbers, not a sick one.

## Concentration to effect

```python
from vivosim import EmaxModel, effect_of

model = EmaxModel(
    baseline=0.0, e_max=100.0, ec50_mg_per_L=0.4,   # all measured, not predicted
    equilibration_rate_const_per_h=0.4,             # effect lags concentration
)
response = effect_of(result, model)

response.time_of_peak_h     # effect peaks after concentration does
response.saturation         # 0.97 — the response is already maxed out
response.time_above(50.0)   # hours spent above half-maximal
```

**Read this before using it.** Given *measured* parameters relating a
concentration to a *measured* response, this computes how that response rises
and falls. It can show that an effect peaks two hours after a dose, that a
second dose lands before the first wore off, or that doubling a dose barely
moves a response already near its ceiling.

It cannot tell you whether a compound works. Emax and EC50 are not predictable
from a molecule — they come from an experiment that already measured the effect
you are asking about. Invent them and the curve means nothing. And "effect"
here is one measured readout, not a disease: whether moving that readout cures
anything is a different question, and not one any simulation answers.

## A population, not one animal

Every result above describes one animal built from mean parameters. No such
animal exists. Individuals differ in body weight, organ size, plasma binding
and — most of all — in how much metabolising enzyme they express, which varies
several-fold. That is why the same dose can be uneventful in most animals and
an overdose in a few.

```python
from vivosim import Variability, simulate_population

population = simulate_population(
    compound, physiology, regimen,
    n=200, seed=42, duration_h=48,
    variability=Variability(hepatic_clearance=0.4, body_weight=0.15),
)

population.fold_range()                      # 1.6 — spread between the 5th and 95th percentile
population.percentile_band(5, 95)            # envelopes to plot behind the median
population.probability_above(5.0)            # fraction whose peak exceeds 5 mg/L
population.probability_auc_within(12, 25)    # fraction landing in a target window
```

Parameters are drawn from lognormal distributions — positive, right-skewed,
and what the field uses. Always pass a `seed` for anything you intend to
report.

**What the probabilities mean.** They are about *exposure*: whether a
concentration lands inside a window you supplied. A compound being present at a
given concentration is not the same as a compound having an effect, so none of
these is a probability that a drug works. Thresholds are yours to supply from
measured data — nothing here knows what is safe.

**What is approximated.** Parameters are sampled independently. Real ones are
correlated: a heavier animal has a bigger liver, and its enzymes move together.
Ignoring that widens some distributions and narrows others, so the percentiles
are an approximation of the spread rather than a calibrated one. The default
variability figures are generic starting points, not measurements of any real
population.

## Viewer

```python
from vivosim.viewer import render_html, show

render_html(result, "profile.html")   # a file you can mail, attach or commit
show(result)                          # opens it, natively if pywebview is installed

render_html(report, "interaction.html")   # a Study or an interaction report too
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

- **An interaction panel**, when given an `assess_interaction` report: each
  compound's exposure alone against together, the ratio and its band, and the
  two plasma curves overlaid — solid for together, dashed for alone, so the gap
  between them *is* the interaction.
- **A compound selector**, when several were simulated, switching every panel
  on the page between them.
- **A validation panel**, when given comparisons against measured data: the
  predicted-against-measured scatter with the line of perfect agreement and the
  two-fold band, and the same misses against sample time — which shows *when*
  the model goes wrong rather than only by how much. It states on the page
  whether the parameters were predicted or fitted, because that is the one
  thing a reader cannot tell by looking.
- **A population panel**, when given a population result: the percentile
  envelope with real individual profiles behind it, the exposure histogram, and
  the mean-parameter animal drawn against the population median — which is not
  the same curve, and seeing that is half the point of simulating a population.
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
- **A rotatable 3D arrangement** of the organs, shaded and lit by
  concentration. The organs are spheres, not meshes: a plausible-looking organ
  nobody measured would be a worse lie than an obvious abstraction.
- **Terms that explain themselves** on hover, because a viewer only a
  pharmacologist can read is not much use to whoever commissioned the model.

Charts are drawn as inline SVG with no plotting library, which is what keeps the
file self-contained. Light and dark are both designed rather than inverted, and
the categorical palette is validated for colour-vision deficiency.

Two skins ship: a plain one, and a **HUD** skin (the toggle in the header) with
a darker surface, monospaced readouts, corner brackets and glow. The series
colours are deliberately *identical* between them — a palette is safe because
of how its hues are spaced and ordered, so swapping in neon ones to suit a mood
would quietly break colour-vision separation. The character comes from
everything else, which costs nothing in legibility.

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
- Correlation between sampled population parameters.
- Disease *progression*, and whether any compound treats anything.
- Emax parameters from structure. The PK/PD layer interpolates between
  measurements; it does not predict them.
- Partitioning of strong bases into acidic phospholipid (Rodgers & Rowland).
- pKa prediction from structure.

## Validation

### Against measured data

Everything else in this suite checks that the code does what the model says.
This checks whether the model says anything true, by predicting concentrations
somebody actually measured.

**Theophylline, 12 subjects, single oral dose** ([`Theoph`][theoph], the
dataset distributed with R):

| | |
|---|---|
| Measurements | 120 (one pre-dose sample per subject excluded, where a fold error against a predicted zero is undefined) |
| Within 2-fold | **92%** |
| Within 1.5-fold | 63% |
| Bias (AFE) | 0.87 |
| Precision (AAFE) | 1.46 |
| Worst single point | 4.95x |

Every compound parameter was a literature value for theophylline, fixed before
the data was looked at and not adjusted afterwards — so this is a *prediction*,
not a fit. That distinction is what makes the number mean anything, and
`validate()` asks you to record which one you did, because a reader cannot tell
from the plot.

Two-fold is the conventional acceptance bar for a PBPK prediction. It is a weak
bar — it is the convention because these predictions carry large parameter
uncertainty, not because being wrong by 90% is good.

**Where it misses:** almost entirely in the first hour. A single literature
absorption rate was used for everyone, and absorption varies a great deal
between individuals. Median fold error by phase: 1.47x before 1 h, falling to
1.32x from 4 h on, once distribution and elimination govern the curve. The
model is better at where a compound goes than at how fast it gets in.

```bash
python examples/validate_theophylline.py
```

The same comparison can be rendered as a page:

```python
render_html(result, "validation.html", validations=comparisons)
```

The dataset is downloaded rather than vendored, so this repository
redistributes no third-party data.

[theoph]: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Theoph.html

### Against known limits

The rest of the suite checks the engine against cases where the answer follows
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
- **Population control.** A population simulated with zero variability
  reproduces the single-animal result exactly, and lognormal sampling preserves
  the mean rather than drifting below it.
- **Extraction-dependent variability.** Enzyme variability spreads exposure for
  a compound the liver clears slowly, and is buffered for one cleared as fast
  as blood delivers it — a known pharmacological consequence of putting
  clearance inside the circulation rather than applying it to the compound
  alone.
- **Conditions stay balanced.** Every disease state still sums flows to cardiac
  output and volumes to body weight, and still conserves mass.
- **Extraction-dependence, twice.** Hepatic impairment hits a slowly cleared
  compound hardest and spares a flow-limited one; renal impairment leaves a
  liver-cleared compound untouched. Neither was coded in — both follow from
  clearance sitting inside the circulation.
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

Organ volumes and blood flows for rat, mouse and human are literature means from:

> Brown RP, Delp MD, Lindstedt SL, Rhomberg LR, Beliles RP (1997).
> "Physiological parameter values for physiologically based pharmacokinetic
> models." *Toxicology and Industrial Health* 13(4):407-484.

`get_physiology("human")` gives a reference 70 kg adult, added so predictions
can be checked against the far larger body of published human data — its
derived figures (6.0 L/min cardiac output, 3.0 L plasma, 1.8 L liver) are
themselves asserted in the tests, so an arithmetic slip in the table cannot
pass unnoticed.

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
