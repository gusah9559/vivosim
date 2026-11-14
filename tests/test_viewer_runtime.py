"""The rendered page must actually run.

Every other viewer test inspects the payload or the file's text. None of them
loads the page, so a page that throws on startup -- and renders nothing at all
-- would pass all of them. This test opens the file in a real browser and
fails if anything throws or if the expected content never appears.

It needs a Chrome or Chromium binary and skips without one, so the suite still
runs anywhere. That makes it a safety net rather than a dependency.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from vivosim import RAT, Compound, IVBolus, OralDose, Regimen, simulate
from vivosim.viewer import render_html

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)

# Injected before the viewer's own script so that anything thrown during
# startup lands in the DOM, where --dump-dom can report it.
ERROR_HOOK = """<script>
window.addEventListener("error", function (e) {
  var box = document.createElement("pre");
  box.id = "startup-error";
  box.textContent = "ERROR: " + e.message + " @" + e.lineno + ":" + e.colno;
  (document.body || document.documentElement).appendChild(box);
});
window.addEventListener("unhandledrejection", function (e) {
  var box = document.createElement("pre");
  box.id = "startup-error";
  box.textContent = "REJECTION: " + e.reason;
  (document.body || document.documentElement).appendChild(box);
});
</script>
"""


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if os.path.isabs(candidate):
            if Path(candidate).exists():
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


@pytest.fixture(scope="module")
def chrome() -> str:
    binary = find_chrome()
    if binary is None:
        pytest.skip("needs Chrome or Chromium to load the rendered page")
    return binary


def render_demo(tmp_path: Path) -> Path:
    compound = Compound(
        name="Runtime check",
        partition_coefficients={"adipose": 9.0, "liver": 4.0, "brain": 0.6},
        default_partition_coefficient=1.5,
        blood_plasma_ratio=1.2,
        fraction_unbound_plasma=0.2,
        hepatic_intrinsic_clearance_L_per_h=2.0,
        renal_clearance_L_per_h=0.3,
    )
    regimen = Regimen(
        IVBolus(2.0), OralDose(5.0, time_h=6.0, absorption_rate_const_per_h=0.9)
    )
    result = simulate(compound, RAT, regimen, duration_h=36)
    return render_html(result, tmp_path / "viewer.html")


def load_dom(chrome: str, page: Path) -> str:
    instrumented = page.with_name("instrumented.html")
    instrumented.write_text(
        page.read_text(encoding="utf-8").replace("<body>", "<body>" + ERROR_HOOK, 1),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--virtual-time-budget=4000",
            "--dump-dom",
            instrumented.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout


def test_page_loads_without_throwing(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_demo(tmp_path))
    error = re.search(r'id="startup-error">(.*?)</pre>', dom, re.S)
    assert error is None, f"the viewer threw on load: {error.group(1) if error else ''}"


def test_page_draws_its_charts(chrome: str, tmp_path: Path) -> None:
    # A page that throws part-way through startup still renders its static
    # markup, so the shell alone proves nothing -- check the parts that only
    # exist if the scripts ran to completion.
    dom = load_dom(chrome, render_demo(tmp_path))

    assert dom.count("series-path") >= 4, "concentration curves were not drawn"
    assert "organ" in dom, "the anatomical view was not built"
    assert dom.count("<tr") > 10, "the tissue ranking table was not filled"
    assert "multiple" in dom, "small multiples were not built"


def test_page_populates_its_controls(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_demo(tmp_path))
    # One chip per selectable series, each built by script.
    assert dom.count('class="chip"') >= len(RAT.tissues)
    assert 'aria-pressed="true"' in dom


def test_tissue_names_reach_the_rendered_page(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_demo(tmp_path))
    for tissue in ("Liver", "Adipose", "Brain", "Kidney"):
        assert tissue in dom, tissue
