"""An interactive viewer for a simulation result.

`render_html` writes a single self-contained HTML file: the data is embedded in
it, nothing is fetched from a network, and it opens on any machine with a
browser. That matters for a scientific tool -- a result you can mail to a
colleague, attach to a report or check into a repository is worth more than one
that only exists inside a running session.

`show` opens that file. If `pywebview` is installed it appears in a native
window, so it behaves like a desktop application; otherwise it falls back to the
default browser. Either way there is one implementation to maintain.

Example:
    >>> from vivosim import IVBolus, Regimen, get_physiology, simulate
    >>> from vivosim.viewer import render_html
    >>> result = simulate(compound, get_physiology("rat"), Regimen(IVBolus(1.0)))
    >>> render_html(result, "profile.html")          # doctest: +SKIP
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

from .dosing import IVBolus, IVInfusion, OralDose
from .interaction import InteractionReport
from .partitioning import volume_of_distribution_L_per_kg
from .results import Result, Study

__all__ = ["render_html", "show", "build_payload"]

_TEMPLATE = Path(__file__).with_name("templates") / "viewer.html"

# Decades of concentration the anatomical shading spans below the peak.
_ANATOMY_DECADES = 3

# Where each tissue sits in the schematic. Coordinates are in the anatomy
# SVG's own 440x300 viewBox. Discrete organs are placed anatomically; the bulk
# tissues -- muscle, fat, skin, bone and the lumped remainder -- have no single
# location, so they are drawn as a labelled row beneath the animal rather than
# pretending to a position they do not have.
_ORGAN_LAYOUT: dict[str, dict[str, Any]] = {
    "brain": {"shape": "circle", "cx": 86, "cy": 92, "r": 12},
    "lung": {"shape": "ellipse", "cx": 174, "cy": 80, "rx": 22, "ry": 14},
    "heart": {"shape": "ellipse", "cx": 172, "cy": 114, "rx": 12, "ry": 10},
    "liver": {"shape": "ellipse", "cx": 232, "cy": 92, "rx": 26, "ry": 21},
    "spleen": {"shape": "ellipse", "cx": 252, "cy": 131, "rx": 10, "ry": 6},
    "gut": {"shape": "ellipse", "cx": 296, "cy": 110, "rx": 28, "ry": 19},
    "kidney": {"shape": "ellipse", "cx": 302, "cy": 76, "rx": 11, "ry": 8},
}

_BULK_TISSUES = ("muscle", "adipose", "skin", "bone", "rest")
_BULK_Y = 230
_BULK_HEIGHT = 34
_BULK_WIDTH = 66
_BULK_GAP = 6
_BULK_START_X = 52


Displayable = Result | Study | InteractionReport


def build_payload(result: Displayable, title: str | None = None) -> dict[str, Any]:
    """Assemble everything the viewer page needs, as plain JSON-ready data.

    Separated from file writing so the payload can be inspected, tested, or
    embedded in a different page.

    Args:
        result: What to display -- one simulation, a `Study` of several
            compounds simulated together, or an `InteractionReport` comparing
            them against their solo runs.
        title: Heading for the page. Defaults to the compound's name, or a
            summary when several compounds are shown.

    Returns:
        A JSON-serialisable mapping.
    """
    report = result if isinstance(result, InteractionReport) else None
    if report is not None:
        results = [entry.together for entry in report]
    elif isinstance(result, Study):
        results = list(result)
    else:
        results = [result]

    physiology = results[0].model.physiology
    tissues = list(physiology.tissues)
    series_order = ["plasma", *tissues]

    blocks = [_compound_block(one, tissues) for one in results]
    if title is None:
        title = (
            blocks[0]["name"]
            if len(blocks) == 1
            else " + ".join(block["name"] for block in blocks)
        )

    return {
        "title": title,
        "subject": _describe_subject(physiology),
        "time": _clean(results[0].time_h),
        "tissueOrder": tissues,
        "labels": _labels(series_order),
        "anatomy": _anatomy_layout(tissues),
        "anatomyDecades": _ANATOMY_DECADES,
        "compounds": blocks,
        "interaction": _interaction_block(report),
        "footer": _footer(results),
    }


def _compound_block(result: Result, tissues: list[str]) -> dict[str, Any]:
    """Everything the page needs about one compound."""
    series: dict[str, list[float]] = {
        "plasma": _clean(result.plasma_concentration()),
        "blood": _clean(result.blood_concentration()),
    }
    for tissue in tissues:
        series[tissue] = _clean(result.tissue_concentration(tissue))

    default_visible = [
        name for name in ("plasma", "liver", "adipose", "muscle") if name in series
    ]

    tissue_matrix = (
        np.vstack([np.asarray(series[tissue]) for tissue in tissues])
        if tissues
        else np.zeros((1, len(result.time_h)))
    )
    anatomy_max = float(np.max(tissue_matrix))

    # Open the anatomical view where there is something to see. At t=0 an
    # intravenous dose is still in the vein and every organ reads zero, which
    # looks like a broken page rather than the start of a simulation.
    default_time_index = int(np.argmax(tissue_matrix.sum(axis=0)))

    return {
        "name": result.model.compound.name,
        "regimen": _describe_regimen(result),
        "series": series,
        "seriesOrder": ["plasma", *tissues],
        "defaultVisible": default_visible,
        "tiles": _tiles(result),
        "doseEvents": _dose_events(result),
        "anatomyMax": anatomy_max if anatomy_max > 0 else 1.0,
        "multiplesMax": anatomy_max if anatomy_max > 0 else 1.0,
        "defaultTimeIndex": default_time_index,
    }


def _interaction_block(report: InteractionReport | None) -> dict[str, Any] | None:
    """The alone-against-together comparison, when there is one."""
    if report is None:
        return None
    return {
        "enzymes": report.interacting_enzymes(),
        "compounds": [
            {
                "name": entry.name,
                "aucAlone": entry.auc_alone,
                "aucTogether": entry.auc_together,
                "aucRatio": entry.auc_ratio,
                "cmaxRatio": entry.cmax_ratio,
                "classification": entry.classification,
                "affected": entry.is_affected,
                "aloneSeries": _clean(entry.alone.plasma_concentration()),
            }
            for entry in report
        ],
    }


def render_html(result: Displayable, path: str | Path, *, title: str | None = None) -> Path:
    """Write a self-contained interactive viewer for a result.

    Args:
        result: What to display -- a `Result`, `Study` or `InteractionReport`.
        path: Where to write the HTML file.
        title: Heading for the page. Defaults to the compound's name.

    Returns:
        The path written.
    """
    payload = build_payload(result, title)
    template = _TEMPLATE.read_text(encoding="utf-8")

    heading = payload["title"]
    page = template.replace("__TITLE__", _escape(f"{heading} — vivosim"))
    # The payload goes inside a <script type="application/json"> block, so only
    # a literal "</script>" could break out of it.
    page = page.replace(
        "__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return destination


def show(
    result: Displayable,
    *,
    title: str | None = None,
    path: str | Path | None = None,
    native: bool = True,
    width: int = 1240,
    height: int = 900,
) -> Path:
    """Render a result and open it.

    Args:
        result: The simulation to display.
        title: Heading for the page.
        path: Where to write the HTML. Defaults to a temporary file that
            persists, so the viewer can be reopened or shared afterwards.
        native: Prefer a native window via `pywebview` when it is installed.
            Set False to always use the browser.
        width: Native window width.
        height: Native window height.

    Returns:
        The path to the rendered HTML file.
    """
    if path is None:
        import tempfile

        directory = Path(tempfile.mkdtemp(prefix="vivosim-"))
        path = directory / "viewer.html"

    rendered = render_html(result, path, title=title)
    heading = title if title is not None else build_payload(result, title)["title"]

    if native:
        try:
            import webview  # type: ignore[import-not-found]
        except ImportError:
            pass
        else:
            webview.create_window(
                f"{heading} — vivosim",
                str(rendered),
                width=width,
                height=height,
            )
            webview.start()
            return rendered

    webbrowser.open(rendered.resolve().as_uri())
    return rendered


# -- payload pieces -------------------------------------------------------


def _clean(values: Any) -> list[float]:
    """A finite float list, safe to embed as strict JSON."""
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in array]


def _labels(names: list[str]) -> dict[str, str]:
    overrides = {"plasma": "Plasma", "blood": "Whole blood", "gut": "Gut", "rest": "Other tissue"}
    return {name: overrides.get(name, name.capitalize()) for name in [*names, "blood"]}


def _describe_subject(physiology: Any) -> str:
    grams = physiology.body_weight_kg * 1000.0
    weight = f"{grams:.0f} g" if grams < 1000 else f"{physiology.body_weight_kg:.2f} kg"
    return (
        f"{physiology.species.capitalize()}, {weight} · cardiac output "
        f"{physiology.cardiac_output_L_per_h:.2f} L/h"
    )


def _describe_regimen(result: Result) -> str:
    doses = list(result.model.regimen)
    if not doses:
        return "No dose administered."

    parts: list[str] = []
    for dose in doses[:4]:
        if isinstance(dose, IVBolus):
            parts.append(f"{dose.amount_mg:g} mg IV bolus at {dose.time_h:g} h")
        elif isinstance(dose, IVInfusion):
            parts.append(
                f"{dose.amount_mg:g} mg IV over {dose.duration_h:g} h "
                f"from {dose.start_h:g} h"
            )
        elif isinstance(dose, OralDose):
            parts.append(
                f"{dose.amount_mg:g} mg oral at {dose.time_h:g} h "
                f"(ka {dose.absorption_rate_const_per_h:g}/h)"
            )
    if len(doses) > 4:
        parts.append(f"and {len(doses) - 4} more")

    total = result.model.regimen.total_amount_mg
    return " · ".join(parts) + f" — {total:g} mg total"


def _tiles(result: Result) -> list[dict[str, str]]:
    model = result.model
    vss = volume_of_distribution_L_per_kg(model.compound, model.physiology)
    error = result.mass_balance_error()
    eliminated = float(result.total_eliminated()[-1])
    administered = model.regimen.total_amount_mg
    fraction = (eliminated / administered * 100.0) if administered > 0 else 0.0

    return [
        {"label": "Peak plasma", "value": f"{result.cmax('plasma'):.3g}", "unit": "mg/L"},
        {"label": "Time of peak", "value": f"{result.tmax('plasma'):.3g}", "unit": "h"},
        {"label": "Plasma AUC", "value": f"{result.auc('plasma'):.3g}", "unit": "mg·h/L"},
        {"label": "Vss", "value": f"{vss:.3g}", "unit": "L/kg"},
        {"label": "Eliminated", "value": f"{fraction:.0f}", "unit": "%"},
        {"label": "Mass balance", "value": f"{error:.1e}", "unit": "mg"},
    ]


def _dose_events(result: Result) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for dose in result.model.regimen:
        if isinstance(dose, IVBolus):
            events.append({"time": dose.time_h, "label": "IV"})
        elif isinstance(dose, IVInfusion):
            events.append({"time": dose.start_h, "label": "inf"})
        elif isinstance(dose, OralDose):
            events.append({"time": dose.depot_entry_time_h, "label": "PO"})
    # Many repeated doses would turn the plot into a picket fence.
    return events if len(events) <= 12 else []


def _anatomy_layout(tissues: list[str]) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []

    for tissue in tissues:
        spec = _ORGAN_LAYOUT.get(tissue)
        if spec is not None:
            layout.append({"tissue": tissue, **spec})

    bulk = [tissue for tissue in tissues if tissue in _BULK_TISSUES]
    for position, tissue in enumerate(bulk):
        x = _BULK_START_X + position * (_BULK_WIDTH + _BULK_GAP)
        layout.append(
            {
                "tissue": tissue,
                "shape": "rect",
                "x": x,
                "y": _BULK_Y,
                "w": _BULK_WIDTH,
                "h": _BULK_HEIGHT,
                "labelX": x + _BULK_WIDTH / 2,
                "labelY": _BULK_Y + _BULK_HEIGHT + 13,
            }
        )

    # Anything the schematic has no place for still has to appear somewhere, or
    # the view would silently omit part of the body.
    placed = {entry["tissue"] for entry in layout}
    leftover = [tissue for tissue in tissues if tissue not in placed]
    for position, tissue in enumerate(leftover):
        x = _BULK_START_X + position * (_BULK_WIDTH + _BULK_GAP)
        layout.append(
            {
                "tissue": tissue,
                "shape": "rect",
                "x": x,
                "y": _BULK_Y + _BULK_HEIGHT + 26,
                "w": _BULK_WIDTH,
                "h": 22,
                "labelX": x + _BULK_WIDTH / 2,
                "labelY": _BULK_Y + _BULK_HEIGHT + 26 + 35,
            }
        )
    return layout


def _footer(results: list[Result]) -> str:
    parameters = " \u00b7 ".join(
        f"<strong>{_escape(r.model.compound.name)}</strong>: "
        f"f<sub>u,p</sub> {r.model.compound.fraction_unbound_plasma:g}, "
        f"blood:plasma {r.model.compound.blood_plasma_ratio:g}, "
        f"hepatic CL<sub>int</sub> "
        f"{r.model.compound.linear_hepatic_clearance_L_per_h:g} L/h, "
        f"renal CL {r.model.compound.renal_clearance_L_per_h:g} L/h"
        for r in results
    )
    return (
        "<strong>Research and teaching tool.</strong> This is a numerical "
        "experiment, not evidence about any real compound, and not a clinical, "
        "veterinary or regulatory tool. It does not recommend doses. Output is "
        "only as good as the parameters supplied. "
        + parameters + ". "
        
        "Physiological values are literature means for a typical healthy animal. "
        "Generated by vivosim."
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
