"""Running a PBPK model: piecewise integration across dosing events.

Dosing creates discontinuities that a smooth ODE solver cannot step across
correctly. A bolus is a jump in the state; the start and end of an infusion are
kinks in the input rate. So the time axis is split at every dosing event, each
segment is integrated on its own, and the discrete jumps are applied to the
state between segments. This keeps the solver inside a smooth problem
throughout, which is both more accurate and faster than letting it discover the
discontinuities by shrinking its step.

One compound or several may be simulated at once. Several are not the same as
several separate runs: compounds sharing a metabolising enzyme compete for it,
so their exposures depend on each other.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .compound import Compound
from .coupled import CoupledModel, build_coupled
from .dosing import Regimen
from .physiology import Physiology
from .results import Result, Study

__all__ = ["simulate"]


def simulate(
    compound: Compound | Sequence[Compound],
    physiology: Physiology,
    regimen: Regimen | Sequence[Regimen],
    *,
    duration_h: float | None = None,
    n_points: int = 601,
    times_h: NDArray[np.float64] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "LSODA",
) -> Result | Study:
    """Simulate one or several compounds in an animal under their regimens.

    Args:
        compound: A compound, or several to simulate together. Several
            compounds sharing a metabolising enzyme compete for it.
        physiology: The animal to simulate them in.
        regimen: The doses to administer. For several compounds, either one
            regimen per compound in the same order, or a single regimen applied
            to each.
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
        A `Result` when one compound was given, or a `Study` holding one
        `Result` per compound when several were.

    Raises:
        ValueError: If the compounds and regimens do not correspond, the output
            grid is invalid, or the integration fails.
    """
    compounds, regimens, single = _normalise(compound, regimen)
    system = build_coupled(compounds, physiology, regimens)

    grid = _output_grid(regimens, duration_h, n_points, times_h)
    boundaries = _segment_boundaries(regimens, grid)

    state = system.initial_state()
    collected = np.zeros((system.n_states, grid.size), dtype=float)

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = _apply_doses(system, state, start)

        # Output times in [start, end). A boundary point belongs to the segment
        # that begins there, by which time any dose landing on it has been
        # applied -- so a dosing time reports the state just after the dose,
        # which is what a sample taken "at" that time would reflect.
        in_segment = (grid >= start) & (grid < end)
        segment_times = grid[in_segment]

        if segment_times.size:
            collected[:, in_segment] = _integrate(
                system, state, start, end, segment_times, rtol, atol, method
            )

        state = _advance(system, state, start, end, rtol, atol, method)

    # The last grid point is always the final boundary, so no segment claimed
    # it; report it from the state carried there, after any dose at that instant.
    collected[:, -1] = _apply_doses(system, state, float(grid[-1]))

    results = [
        Result(time_h=grid, amounts=collected[system.block(index)], model=model)
        for index, model in enumerate(system.models)
    ]
    if single:
        return results[0]
    return Study(results=results, system=system)


def _normalise(
    compound: Compound | Sequence[Compound],
    regimen: Regimen | Sequence[Regimen],
) -> tuple[list[Compound], list[Regimen], bool]:
    """Resolve the flexible call signature into parallel lists."""
    single = isinstance(compound, Compound)
    compounds = [compound] if single else list(compound)
    if not compounds:
        raise ValueError("at least one compound is required")

    if isinstance(regimen, Regimen):
        regimens = [regimen] * len(compounds)
    else:
        regimens = list(regimen)
        if len(regimens) != len(compounds):
            raise ValueError(
                f"got {len(compounds)} compounds but {len(regimens)} regimens; "
                "pass one regimen per compound, or a single regimen for all"
            )
    return compounds, regimens, single


def _segment_boundaries(regimens: list[Regimen], grid: NDArray[np.float64]) -> list[float]:
    """Times the integration must stop at: every dosing event, plus the ends."""
    events: set[float] = set()
    for regimen in regimens:
        events.update(t for t in regimen.discontinuity_times() if t < grid[-1])
    return sorted({float(grid[0]), *events, float(grid[-1])})


def _apply_doses(
    system: CoupledModel, state: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    """Add any bolus and oral doses scheduled exactly at ``t`` to the state."""
    updated = state.copy()
    changed = False

    for index, model in enumerate(system.models):
        regimen = model.regimen
        bolus = regimen.boluses_at(t)
        oral = [
            (slot, dose)
            for slot, dose in model.oral_doses()
            if dose.depot_entry_time_h == t
        ]
        if not bolus and not oral:
            continue

        changed = True
        offset = system.block(index).start
        updated[offset + model.i_venous] += bolus
        for slot, dose in oral:
            updated[offset + slot] += dose.depot_amount_mg

    return updated if changed else state


def _integrate(
    system: CoupledModel,
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
        system.rhs,
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
    system: CoupledModel,
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
        system.rhs, (start, end), state, rtol=rtol, atol=atol, method=method
    )
    if not solution.success:
        raise ValueError(f"integration failed on [{start}, {end}]: {solution.message}")
    return solution.y[:, -1]


def _output_grid(
    regimens: list[Regimen],
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
    if duration_h is not None:
        total = duration_h
    else:
        last_event = max(regimen.last_event_time_h() for regimen in regimens)
        total = last_event + 24.0
    if total <= 0:
        raise ValueError(f"duration_h must be positive, got {total}")
    return np.linspace(0.0, total, n_points)
