"""Simulation output: concentration-time profiles and derived PK metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import PBPKModel

__all__ = ["Result"]


@dataclass(frozen=True)
class Result:
    """The outcome of one simulation.

    Attributes:
        time_h: Output time grid, in hours.
        amounts: Amount in each compartment over time, in mg, shaped
            ``(n_states, n_times)`` and ordered as ``model.state_names``.
        model: The model that produced the result, retained so that volumes
            and labels are available for derived quantities.
    """

    time_h: NDArray[np.float64]
    amounts: NDArray[np.float64]
    model: PBPKModel

    # -- raw access --------------------------------------------------------

    def amount(self, compartment: str) -> NDArray[np.float64]:
        """Amount in one compartment over time, in mg."""
        try:
            return self.amounts[self.model.index[compartment]]
        except KeyError:
            available = ", ".join(self.model.state_names)
            raise KeyError(
                f"unknown compartment {compartment!r}; available: {available}"
            ) from None

    def tissue_concentration(self, tissue: str) -> NDArray[np.float64]:
        """Total concentration in one tissue over time, in mg/L."""
        if tissue not in self.model.physiology.tissues:
            available = ", ".join(self.model.physiology.tissues)
            raise KeyError(f"unknown tissue {tissue!r}; available: {available}")
        return self.amount(tissue) / self.model.physiology.volume_L(tissue)

    def blood_concentration(self, site: str = "venous") -> NDArray[np.float64]:
        """Whole blood concentration over time, in mg/L.

        Args:
            site: ``"venous"`` or ``"arterial"``. Venous blood is what a
                peripheral blood sample measures, so it is the default.
        """
        phys = self.model.physiology
        volume = {"venous": phys.venous_volume_L, "arterial": phys.arterial_volume_L}
        if site not in volume:
            raise KeyError(f"site must be 'venous' or 'arterial', got {site!r}")
        return self.amount(site) / volume[site]

    def plasma_concentration(self, site: str = "venous") -> NDArray[np.float64]:
        """Plasma concentration over time, in mg/L.

        Converted from whole blood using the compound's blood:plasma ratio.
        This is the quantity a bioanalytical assay on plasma reports, and so
        the one to compare against measured data.
        """
        return self.blood_concentration(site) / self.model.compound.blood_plasma_ratio

    # -- mass balance ------------------------------------------------------

    def total_in_body(self) -> NDArray[np.float64]:
        """Total amount still in the body over time, in mg."""
        model = self.model
        eliminated = {model.i_hepatic_out, model.i_renal_out}
        keep = [i for i in range(model.n_states) if i not in eliminated]
        return self.amounts[keep].sum(axis=0)

    def total_eliminated(self) -> NDArray[np.float64]:
        """Cumulative amount eliminated over time, in mg."""
        return self.amount("eliminated_hepatic") + self.amount("eliminated_renal")

    def mass_balance_error(self) -> float:
        """Largest absolute departure from conservation of mass, in mg.

        The amount in the body plus the amount eliminated must equal the
        amount administered so far. A well-converged integration keeps this at
        the level of the solver tolerance; a large value means the model or the
        tolerances need attention.
        """
        administered = self._cumulative_administered()
        accounted = self.total_in_body() + self.total_eliminated()
        return float(np.max(np.abs(accounted - administered)))

    def _cumulative_administered(self) -> NDArray[np.float64]:
        """Amount administered up to each output time, in mg."""
        from .dosing import IVBolus, IVInfusion, OralDose

        total = np.zeros_like(self.time_h)
        for dose in self.model.regimen:
            if isinstance(dose, IVBolus):
                total += np.where(self.time_h >= dose.time_h, dose.amount_mg, 0.0)
            elif isinstance(dose, IVInfusion):
                delivered = np.clip(self.time_h - dose.start_h, 0.0, dose.duration_h)
                total += delivered * dose.rate_mg_per_h
            elif isinstance(dose, OralDose):
                # Only the bioavailable part ever enters the model, so that is
                # what conservation applies to.
                total += np.where(
                    self.time_h >= dose.depot_entry_time_h, dose.depot_amount_mg, 0.0
                )
        return total

    # -- PK metrics --------------------------------------------------------

    def cmax(self, series: str = "plasma") -> float:
        """Peak concentration of a profile, in mg/L."""
        return float(np.max(self._series(series)))

    def tmax(self, series: str = "plasma") -> float:
        """Time of peak concentration, in hours."""
        return float(self.time_h[int(np.argmax(self._series(series)))])

    def auc(self, series: str = "plasma") -> float:
        """Area under the concentration-time curve over the simulated window.

        Computed by the trapezoidal rule on the output grid, so it reflects the
        grid's resolution. In mg*h/L.
        """
        return float(np.trapezoid(self._series(series), self.time_h))

    def summary(self, series: str = "plasma") -> dict[str, float]:
        """Cmax, Tmax and AUC for one profile, plus the mass balance error."""
        return {
            "cmax_mg_per_L": self.cmax(series),
            "tmax_h": self.tmax(series),
            "auc_mg_h_per_L": self.auc(series),
            "mass_balance_error_mg": self.mass_balance_error(),
        }

    def _series(self, series: str) -> NDArray[np.float64]:
        """Resolve a profile name to a concentration series.

        Accepts ``"plasma"``, ``"blood"``, or any tissue name.
        """
        if series == "plasma":
            return self.plasma_concentration()
        if series == "blood":
            return self.blood_concentration()
        return self.tissue_concentration(series)

    # -- export ------------------------------------------------------------

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Every profile as a flat mapping, convenient for export or plotting."""
        out: dict[str, NDArray[np.float64]] = {"time_h": self.time_h}
        out["plasma_mg_per_L"] = self.plasma_concentration()
        out["blood_mg_per_L"] = self.blood_concentration()
        for tissue in self.model.physiology.tissues:
            out[f"{tissue}_mg_per_L"] = self.tissue_concentration(tissue)
        return out

    def to_csv(self, path: str) -> None:
        """Write every profile to a CSV file with a header row."""
        data = self.to_dict()
        header = ",".join(data)
        matrix = np.column_stack(list(data.values()))
        np.savetxt(path, matrix, delimiter=",", header=header, comments="", fmt="%.8g")
