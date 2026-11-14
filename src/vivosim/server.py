"""A local console: pick compounds in a browser and run them.

A rendered page is a finished report -- you cannot change a dose inside it,
because the simulation it describes already happened. Running a *new* one needs
the solver, which means a process. So this serves a small control panel that
posts a study back to Python, runs it, and shows the result in place.

It binds to localhost only and runs on the standard library's HTTP server. It
is a development convenience for one person on one machine, not a deployment
target: there is no authentication, no sandboxing and no concurrency to speak
of. Do not expose it to a network.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .compound import Compound
from .conditions import CONDITIONS
from .dosing import IVBolus, IVInfusion, OralDose, Regimen
from .interaction import assess_interaction
from .library import LIBRARY
from .physiology import SPECIES, get_physiology
from .population import Variability, simulate_population
from .simulate import simulate
from .viewer import build_payload, render_html

__all__ = ["serve", "run_study", "build_app_state"]

_CONSOLE = Path(__file__).with_name("templates") / "console.html"
_MAX_BODY_BYTES = 256 * 1024


def build_app_state() -> dict[str, Any]:
    """Everything the console needs to render its controls."""
    return {
        "compounds": [
            {
                "key": entry.key,
                "label": entry.label,
                "note": entry.note,
                "measured": entry.measured,
                "dose": entry.suggested_dose_mg_per_kg,
            }
            for entry in LIBRARY.values()
        ],
        "species": [
            {
                "key": key,
                "label": key.capitalize(),
                "weight": physiology.body_weight_kg,
            }
            for key, physiology in SPECIES.items()
        ],
        "conditions": [
            {"key": key, "label": key, "note": condition.description}
            for key, condition in CONDITIONS.items()
        ],
        "routes": [
            {"key": "iv", "label": "IV bolus"},
            {"key": "infusion", "label": "IV infusion"},
            {"key": "oral", "label": "Oral"},
        ],
    }


def _regimen(spec: dict[str, Any], weight_kg: float) -> Regimen:
    """Build a regimen from the console's controls."""
    route = str(spec.get("route", "iv"))
    amount = float(spec.get("dose_mg_per_kg", 4.0)) * weight_kg
    if amount <= 0:
        raise ValueError("the dose must be greater than zero")

    if route == "iv":
        return Regimen(IVBolus(amount_mg=amount))
    if route == "infusion":
        duration = float(spec.get("infusion_h", 2.0))
        if duration <= 0:
            raise ValueError("the infusion duration must be greater than zero")
        return Regimen(IVInfusion(amount_mg=amount, duration_h=duration))
    if route == "oral":
        return Regimen(
            OralDose(
                amount_mg=amount,
                absorption_rate_const_per_h=float(spec.get("ka", 1.0)),
            )
        )
    raise ValueError(f"unknown route {route!r}")


def run_study(request: dict[str, Any]) -> tuple[Any, str]:
    """Run what the console asked for.

    Args:
        request: The console's study description.

    Returns:
        The result object and a title for the page.

    Raises:
        ValueError: If the request does not describe a runnable study.
        KeyError: If it names a compound, species or condition that is unknown.
    """
    selected = list(request.get("compounds", []))
    if not selected:
        raise ValueError("pick at least one compound")
    if len(selected) > 2:
        raise ValueError("pick at most two compounds")

    species_key = str(request.get("species", "rat"))
    weight = request.get("body_weight_kg")
    physiology = get_physiology(species_key, float(weight) if weight else None)

    compounds: list[Compound] = []
    regimens: list[Regimen] = []
    for spec in selected:
        key = str(spec.get("key"))
        if key not in LIBRARY:
            raise KeyError(f"no compound {key!r} in the library")
        compounds.append(LIBRARY[key].compound)
        regimens.append(_regimen(spec, physiology.body_weight_kg))

    condition_key = str(request.get("condition", "healthy"))
    if condition_key not in CONDITIONS:
        raise KeyError(f"no condition {condition_key!r}")
    condition = CONDITIONS[condition_key]
    if not condition.is_healthy:
        physiology = condition.apply_physiology(physiology)
        compounds = [condition.apply_compound(c) for c in compounds]

    duration = float(request.get("duration_h", 48.0))
    if duration <= 0:
        raise ValueError("the duration must be greater than zero")

    population = int(request.get("population", 0) or 0)
    title = " + ".join(c.name for c in compounds)

    if population:
        if len(compounds) > 1:
            raise ValueError(
                "a population run takes one compound; deselect one, or turn "
                "the population off to compare the two"
            )
        return (
            simulate_population(
                compounds[0],
                physiology,
                regimens[0],
                n=max(2, min(population, 500)),
                seed=int(request.get("seed", 42)),
                duration_h=duration,
                variability=Variability(
                    hepatic_clearance=float(request.get("variability", 0.4))
                ),
            ),
            title,
        )

    if len(compounds) > 1:
        return assess_interaction(compounds, physiology, regimens, duration_h=duration), title

    return simulate(compounds[0], physiology, regimens[0], duration_h=duration), title


class _Handler(BaseHTTPRequestHandler):
    """Routes for the console. One instance per request, as the stdlib does."""

    server_version = "vivosim"
    reports: dict[str, bytes] = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # The stdlib logs every request to stderr, which buries the one line
        # the user needs. Errors still surface through the responses.
        pass

    # -- helpers ----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8"
        )

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            page = _CONSOLE.read_text(encoding="utf-8").replace(
                "__STATE__", json.dumps(build_app_state())
            )
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path.startswith("/report/"):
            report = self.reports.get(path.rsplit("/", 1)[-1])
            if report is None:
                self._send(404, b"no such report", "text/plain; charset=utf-8")
                return
            self._send(200, report, "text/html; charset=utf-8")
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?", 1)[0] != "/api/run":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._json(400, {"error": "the request body was empty or too large"})
            return

        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            self._json(400, {"error": f"malformed request: {error}"})
            return

        try:
            result, title = run_study(request)
        except (ValueError, KeyError) as error:
            self._json(400, {"error": str(error).strip("'")})
            return
        except Exception as error:  # noqa: BLE001 - never kill the server
            self._json(500, {"error": f"the simulation failed: {error}"})
            return

        import tempfile
        import uuid

        token = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as directory:
            path = render_html(result, Path(directory) / "report.html", title=title)
            self.reports[token] = path.read_bytes()

        # Keep the last few reports only; a long session would otherwise hold
        # every page it ever rendered in memory.
        while len(self.reports) > 8:
            self.reports.pop(next(iter(self.reports)))

        payload = build_payload(result, title)
        headline = payload["compounds"][0]["tiles"]
        self._json(
            200,
            {
                "report": f"/report/{token}",
                "title": title,
                "tiles": headline,
                "interaction": payload.get("interaction"),
                "population": (
                    {
                        key: payload["population"][key]
                        for key in ("n", "aucMedian", "aucP5", "aucP95", "foldRange")
                    }
                    if payload.get("population")
                    else None
                ),
            },
        )


def serve(port: int = 8765, open_browser: bool = True) -> None:
    """Run the console until interrupted.

    Args:
        port: Port to listen on. Bound to localhost only.
        open_browser: Whether to open a browser at the console.
    """
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"vivosim console at {url}")
    print("press ctrl-c to stop")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
