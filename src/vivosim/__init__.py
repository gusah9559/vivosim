"""vivosim -- physiologically based pharmacokinetic simulation.

A perfusion-limited PBPK engine for preclinical species. Given a compound's
distribution and clearance parameters, an animal's physiology and a dosing
regimen, it computes the concentration-time profile in blood and in each
tissue.

Intended for research and education. Not a clinical or regulatory tool, and
not a substitute for measured data; see the README.

Example:
    >>> from vivosim import Compound, IVBolus, Regimen, get_physiology, simulate
    >>> compound = Compound(
    ...     name="example",
    ...     default_partition_coefficient=1.5,
    ...     fraction_unbound_plasma=0.1,
    ...     hepatic_intrinsic_clearance_L_per_h=2.0,
    ... )
    >>> result = simulate(
    ...     compound, get_physiology("rat"), Regimen(IVBolus(1.0)), duration_h=12
    ... )
    >>> round(result.auc("plasma"), 3) > 0
    True
"""

from .compound import Compound
from .conditions import (
    CONDITIONS,
    HEALTHY,
    Condition,
    hepatic_impairment,
    obesity,
    renal_impairment,
)
from .coupled import CoupledModel
from .dosing import Dose, IVBolus, IVInfusion, OralDose, Regimen, repeated
from .interaction import (
    CompoundInteraction,
    InteractionReport,
    assess_interaction,
    classify_ratio,
)
from .kinetics import Inhibition, Metabolism
from .model import PBPKModel
from .partitioning import (
    MolecularProperties,
    apparent_logd,
    compound_from_properties,
    predict_partition_coefficients,
    volume_of_distribution_L_per_kg,
)
from .pharmacodynamics import EffectProfile, EmaxModel, effect_of
from .physiology import HUMAN, MOUSE, RAT, SPECIES, Physiology, TissueSpec, get_physiology
from .population import (
    Individual,
    PopulationResult,
    Variability,
    sample_population,
    simulate_population,
)
from .results import Result, Study
from .simulate import simulate
from .tissue_composition import (
    PLASMA_COMPOSITION,
    RAT_TISSUE_COMPOSITION,
    TissueComposition,
)
from .validation import ObservedProfile, ValidationResult, validate, validate_many

__version__ = "0.0.1"

__all__ = [
    "HUMAN",
    "MOUSE",
    "PLASMA_COMPOSITION",
    "RAT",
    "RAT_TISSUE_COMPOSITION",
    "SPECIES",
    "CONDITIONS",
    "Compound",
    "Condition",
    "CompoundInteraction",
    "CoupledModel",
    "Dose",
    "EffectProfile",
    "EmaxModel",
    "Inhibition",
    "InteractionReport",
    "IVBolus",
    "IVInfusion",
    "Metabolism",
    "MolecularProperties",
    "ObservedProfile",
    "OralDose",
    "PBPKModel",
    "HEALTHY",
    "Individual",
    "Physiology",
    "PopulationResult",
    "Regimen",
    "Result",
    "Study",
    "TissueComposition",
    "TissueSpec",
    "ValidationResult",
    "Variability",
    "__version__",
    "apparent_logd",
    "assess_interaction",
    "classify_ratio",
    "effect_of",
    "compound_from_properties",
    "get_physiology",
    "hepatic_impairment",
    "obesity",
    "predict_partition_coefficients",
    "renal_impairment",
    "repeated",
    "sample_population",
    "simulate_population",
    "validate",
    "validate_many",
    "simulate",
    "volume_of_distribution_L_per_kg",
]
