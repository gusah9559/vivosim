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
from .dosing import Dose, IVBolus, IVInfusion, OralDose, Regimen, repeated
from .model import PBPKModel
from .physiology import MOUSE, RAT, SPECIES, Physiology, TissueSpec, get_physiology
from .results import Result
from .simulate import simulate

__version__ = "0.0.1"

__all__ = [
    "MOUSE",
    "RAT",
    "SPECIES",
    "Compound",
    "Dose",
    "IVBolus",
    "IVInfusion",
    "OralDose",
    "PBPKModel",
    "Physiology",
    "Regimen",
    "Result",
    "TissueSpec",
    "__version__",
    "get_physiology",
    "repeated",
    "simulate",
]
