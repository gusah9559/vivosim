"""Running a PBPK model: piecewise integration across dosing events.

Dosing creates discontinuities that a smooth ODE solver cannot step across
correctly. A bolus is a jump in the state; the start and end of an infusion are
kinks in the input rate. So the time axis is split at every dosing event, each
segment is integrated on its own, and the discrete jumps are applied to the
state between segments. This keeps the solver inside a smooth problem
throughout, which is both more accurate and faster than letting it discover the
discontinuities by shrinking its step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .compound import Compound
from .dosing import Regimen
from .model import PBPKModel
from .physiology import Physiology
from .results import Result

__all__ = ["simulate"]


def simulate(
    compound: Compound,
    physiology: Physiology,
    regimen: Regimen,
    *,
    duration_h: float | None = None,
    n_points: int = 601,
    times_h: NDArray[np.float64] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "LSODA",
) -> Result:
    """Simulate a compound in an animal under a dosing regimen.

    Args:
        compound: The compound to simulate.
        physiology: The animal to simulate it in.
        regimen: The doses to administer.
        duration_h: Length of the simulation. Defaults to 24 hours past the
            last dosing event. Ignored when ``times_h`` is given.
        n_points: Number of points on the output grid. Ignored when
            ``times_h`` is given.
        times_h: An explicit output grid, in hours. Must be sorted and start at
            or after 0.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver, in mg. The default is tight
            because amounts in poorly perfused tissues are small, and a loose
            absolute tolerance quietly truncates them.
        method: A `scipy.integrate.solve_ivp` method. ``"LSODA"`` adapts
            between stiff and non-stiff regimes, which suits the wide spread of
            perfusion rates across tissues.

    Returns:
        A `Result` holding the amount in every compartment over the output grid.

    Raises:
        ValueError: If the output grid is invalid, or the integration fails.
    """
    model = PBPKModel(compound=compound, physiology=physiology, regimen=regimen)

    grid = _output_grid(regimen, duration_h, n_points, times_h)
    event_times = [t for t in regimen.discontinuity_times() if t < grid[-1]]

    # Segment boundaries: the start of the grid, every dosing event inside it,
    # and the end of the grid.
    boundaries = sorted({float(grid[0]), *event_times, float(grid[-1])})

    state = model.initial_state()
    collected = np.zeros((model.n_states, grid.size), dtype=float)

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = _apply_doses(model, state, start)

        # Output times in [start, end). A boundary point belongs to the segment
        # that begins there, by which time any dose landing on it has been
        # applied -- so a dosing time reports the state just after the dose,
        # which is what a sample taken "at" that time would reflect.
        in_segment = (grid >= start) & (grid < end)
        segment_times = grid[in_segment]

        if segment_times.size:
            collected[:, in_segment] = _integrate(
                model, state, start, end, segment_times, rtol, atol, method
            )

        state = _advance(model, state, start, end, rtol, atol, method)

    # The last grid point is always the final boundary, so no segment claimed
    # it; report it from the state carried there, after any dose at that instant.
    collected[:, -1] = _apply_doses(model, state, float(grid[-1]))

    return Result(time_h=grid, amounts=collected, model=model)


def _apply_doses(
    model: PBPKModel, state: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    """Add any bolus and oral doses scheduled exactly at ``t`` to the state."""
    regimen = model.regimen
    bolus = regimen.boluses_at(t)
    oral = [(slot, dose) for slot, dose in model.oral_doses() if dose.depot_entry_time_h == t]

    if not bolus and not oral:
        return state

    updated = state.copy()
    updated[model.i_venous] += bolus
    for slot, dose in oral:
        updated[slot] += dose.depot_amount_mg
    return updated


def _integrate(
    model: PBPKModel,
    state: NDArray[np.float64],
    start: float,
    end: float,
    output_times: NDArray[np.float64],
    rtol: float,
    atol: float,
    method: str,
) -> NDArray[np.float64]:
    """Integrate one segment and sample it at the requested times."""
    if start == end:
        return np.repeat(state[:, None], output_times.size, axis=1)

    solution = solve_ivp(
        model.rhs,
        (start, end),
        state,
        t_eval=output_times,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    if not solution.success:
        raise ValueError(f"integration failed on [{start}, {end}]: {solution.message}")
    return solution.y


def _advance(
    model: PBPKModel,
    state: NDArray[np.float64],
    start: float,
    end: float,
    rtol: float,
    atol: float,
    method: str,
) -> NDArray[np.float64]:
    """Carry the state from the start of a segment to its end."""
    if start == end:
        return state
    solution = solve_ivp(
        model.rhs, (start, end), state, rtol=rtol, atol=atol, method=method
    )
    if not solution.success:
        raise ValueError(f"integration failed on [{start}, {end}]: {solution.message}")
    return solution.y[:, -1]


def _output_grid(
    regimen: Regimen,
    duration_h: float | None,
    n_points: int,
    times_h: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Build and validate the output time grid."""
    if times_h is not None:
        grid = np.asarray(times_h, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("times_h must be a one-dimensional grid of at least 2 points")
        if grid[0] < 0:
            raise ValueError(f"times_h must start at or after 0, got {grid[0]}")
        if np.any(np.diff(grid) <= 0):
            raise ValueError("times_h must be strictly increasing")
        return grid

    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    total = duration_h if duration_h is not None else regimen.last_event_time_h() + 24.0
    if total <= 0:
        raise ValueError(f"duration_h must be positive, got {total}")
    return np.linspace(0.0, total, n_points)
