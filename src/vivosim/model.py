"""The perfusion-limited PBPK model: state layout and derivatives.

Structure
---------
Blood leaves the venous pool, passes through the lung, and becomes arterial
blood. Arterial blood perfuses every tissue. Most tissues drain back to the
venous pool; gut and spleen drain into the liver, so an oral dose meets hepatic
metabolism before reaching the rest of the body.

Each tissue is a single well-mixed compartment whose exchange with blood is
limited by perfusion rather than by permeation. For a tissue ``T``::

    dA_T/dt = Q_T * (C_art - C_T / Kp_blood_T)

where ``A_T`` is the amount in the tissue, ``C_T = A_T / V_T`` the total tissue
concentration, and ``Kp_blood_T`` the tissue-to-blood partition coefficient.
The term ``C_T / Kp_blood_T`` is the concentration of the blood leaving the
tissue, which at perfusion limitation is in equilibrium with the tissue.

Elimination
-----------
Hepatic metabolism is driven by the unbound concentration in blood leaving the
liver, giving a well-stirred liver embedded in the circulation::

    rate = CL_int * fu_p * C_liver_out / BP

Renal clearance of unchanged drug is referenced to the blood concentration
leaving the kidney. Both are integrated as cumulative states so that mass
balance can be checked exactly.

Units
-----
Amounts in mg, volumes in L, flows in L/h, concentrations in mg/L, time in h.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .compound import Compound
from .dosing import OralDose, Regimen
from .physiology import Physiology

__all__ = ["PBPKModel"]


@dataclass
class PBPKModel:
    """A compound, an animal and a dosing regimen assembled into an ODE system.

    The model is built once and then integrated by `vivosim.simulate`. It owns
    the mapping between named compartments and state vector indices, which it
    exposes so that results can be labelled.

    Attributes:
        compound: The compound being simulated.
        physiology: The animal being simulated.
        regimen: The doses administered.
    """

    compound: Compound
    physiology: Physiology
    regimen: Regimen

    def __post_init__(self) -> None:
        phys = self.physiology

        # State layout: arterial, venous, each tissue, one depot per oral dose,
        # then two cumulative elimination accumulators.
        self.tissue_names: list[str] = list(phys.tissues)
        self._oral_doses: list[OralDose] = [
            dose for dose in self.regimen if isinstance(dose, OralDose)
        ]

        names = ["arterial", "venous", *self.tissue_names]
        names += [f"depot_{i}" for i in range(len(self._oral_doses))]
        names += ["eliminated_hepatic", "eliminated_renal"]
        self.state_names: list[str] = names
        self.index: dict[str, int] = {name: i for i, name in enumerate(names)}

        self.i_arterial = self.index["arterial"]
        self.i_venous = self.index["venous"]
        self.i_hepatic_out = self.index["eliminated_hepatic"]
        self.i_renal_out = self.index["eliminated_renal"]
        self._i_tissue: dict[str, int] = {name: self.index[name] for name in self.tissue_names}
        self._depot_indices: list[int] = [
            self.index[f"depot_{i}"] for i in range(len(self._oral_doses))
        ]

        self.i_lung = self._i_tissue[phys.lung_name]
        self.i_liver = self._i_tissue[phys.liver_name]

        # Precompute per-tissue constants as parallel arrays so the derivative
        # is pure vector arithmetic rather than dictionary lookups.
        order = self.tissue_names
        self._tissue_slots = np.array([self._i_tissue[n] for n in order], dtype=np.intp)
        self._volumes = np.array([phys.volume_L(n) for n in order], dtype=float)
        self._flows = np.array([phys.flow_L_per_h(n) for n in order], dtype=float)
        self._kp_blood = np.array([self.compound.kp_blood(n) for n in order], dtype=float)

        is_lung = np.array([n == phys.lung_name for n in order])
        is_liver = np.array([n == phys.liver_name for n in order])
        drains_liver = np.array([phys.tissues[n].drains_to == "liver" for n in order])

        # Position within the per-tissue arrays (not the state vector) of the
        # two compartments that need their own balance equations.
        self._lung_pos = int(np.flatnonzero(is_lung)[0])
        self._liver_pos = int(np.flatnonzero(is_liver)[0])

        # Tissues whose outflow reaches the venous pool directly: everything
        # except the lung (which feeds the arterial side), the portal tissues
        # (which feed the liver) and the liver itself (handled explicitly,
        # since its outflow carries the portal flow too).
        self._to_venous = ~is_lung & ~drains_liver & ~is_liver
        self._to_liver = drains_liver
        # Tissues perfused by arterial blood through the generic exchange term.
        # The lung draws on venous blood and the liver has a composite inflow,
        # so both are excluded here.
        self._arterially_perfused = ~is_lung & ~is_liver

        self.cardiac_output_L_per_h = phys.cardiac_output_L_per_h
        self.liver_outflow_L_per_h = phys.liver_inflow_L_per_h()
        self._v_arterial = phys.arterial_volume_L
        self._v_venous = phys.venous_volume_L

        self._i_kidney = self._i_tissue.get("kidney")
        self._kidney_pos = order.index("kidney") if "kidney" in phys.tissues else None

        # Clearance terms, folded into the unit conversions they need once.
        c = self.compound
        self._hepatic_cl = c.hepatic_intrinsic_clearance_L_per_h * (
            c.fraction_unbound_plasma / c.blood_plasma_ratio
        )
        self._renal_cl = c.renal_clearance_L_per_h

        self._ka = np.array(
            [d.absorption_rate_const_per_h for d in self._oral_doses], dtype=float
        )
        self._depot_slots = np.array(self._depot_indices, dtype=np.intp)

    # -- state construction ------------------------------------------------

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    def initial_state(self) -> NDArray[np.float64]:
        """A zero state vector.

        Doses are applied by the simulator at their scheduled times, including
        any at t=0, so the model itself starts empty.
        """
        return np.zeros(self.n_states, dtype=float)

    def oral_doses(self) -> list[tuple[int, OralDose]]:
        """Each oral dose paired with the state index of its depot."""
        return list(zip(self._depot_indices, self._oral_doses, strict=True))

    # -- derivatives -------------------------------------------------------

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Time derivative of the state vector.

        Args:
            t: Time in hours.
            y: Current state vector of amounts in mg.

        Returns:
            dy/dt in mg/h.
        """
        dy = np.zeros_like(y)

        # Concentration of the blood leaving each tissue. At perfusion
        # limitation it is in equilibrium with the tissue itself.
        c_out = y[self._tissue_slots] / (self._volumes * self._kp_blood)
        c_arterial = y[self.i_arterial] / self._v_arterial
        c_venous = y[self.i_venous] / self._v_venous

        qc = self.cardiac_output_L_per_h
        lung_out = c_out[self._lung_pos]
        liver_out = c_out[self._liver_pos]
        q_liver_total = self.liver_outflow_L_per_h
        q_hepatic_artery = self._flows[self._liver_pos]

        # Lung: takes the whole venous return, discharges into arterial blood.
        dy[self.i_lung] = qc * (c_venous - lung_out)
        dy[self.i_arterial] = qc * (lung_out - c_arterial)

        # Ordinary tissues: perfused by arterial blood.
        flux = np.where(
            self._arterially_perfused, self._flows * (c_arterial - c_out), 0.0
        )
        dy[self._tissue_slots] += flux

        # Liver: hepatic artery plus portal inflow in, total hepatic outflow
        # out, less what metabolism removes.
        portal_in = float(np.sum(self._flows[self._to_liver] * c_out[self._to_liver]))
        hepatic_rate = self._hepatic_cl * liver_out
        dy[self.i_liver] += (
            q_hepatic_artery * c_arterial + portal_in - q_liver_total * liver_out - hepatic_rate
        )
        dy[self.i_hepatic_out] = hepatic_rate

        # Kidney: clearance of unchanged drug.
        renal_rate = 0.0
        if self._kidney_pos is not None and self._renal_cl > 0.0:
            renal_rate = self._renal_cl * c_out[self._kidney_pos]
            dy[self._i_kidney] -= renal_rate
        dy[self.i_renal_out] = renal_rate

        # Venous pool: systemic return plus hepatic outflow, less what the
        # heart sends on to the lung, plus any running infusion.
        venous_in = float(np.sum(self._flows[self._to_venous] * c_out[self._to_venous]))
        venous_in += q_liver_total * liver_out
        dy[self.i_venous] = venous_in - qc * c_venous + self.regimen.infusion_rate_at(t)

        # First-order absorption from each gut depot into the portal blood,
        # which the liver sees first.
        if len(self._depot_slots):
            depot_amounts = y[self._depot_slots]
            absorbed = self._ka * depot_amounts
            dy[self._depot_slots] -= absorbed
            dy[self.i_liver] += float(np.sum(absorbed))

        return dy
