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


def render_interaction(tmp_path: Path) -> Path:
    """A victim and an inhibitor, with the alone-against-together comparison."""
    from vivosim import Inhibition, IVInfusion, Metabolism, assess_interaction

    victim = Compound(
        name="Victim",
        partition_coefficients={"adipose": 8.0, "liver": 4.0},
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        metabolism=(Metabolism(enzyme="CYP3A4", vmax_mg_per_h=2.0, km_mg_per_L=0.5),),
    )
    inhibitor = Compound(
        name="Inhibitor",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        inhibits=(Inhibition(enzyme="CYP3A4", ki_mg_per_L=0.05),),
    )
    report = assess_interaction(
        [victim, inhibitor],
        RAT,
        [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=48.0))],
        duration_h=48,
    )
    return render_html(report, tmp_path / "ddi.html")


def test_interaction_page_loads_without_throwing(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_interaction(tmp_path))
    error = re.search(r'id="startup-error">(.*?)</pre>', dom, re.S)
    assert error is None, f"the viewer threw on load: {error.group(1) if error else ''}"


def test_interaction_page_draws_the_comparison(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_interaction(tmp_path))
    # Match the drawn element, not the stylesheet rule that names the class:
    # the rule is in the template whether or not anything was plotted.
    assert 'class="solo-path"' in dom, "the alone-against-together chart was not drawn"
    # One band per compound. The exact band depends on the scenario, so assert
    # that a real one was reported rather than pinning a particular strength.
    assert dom.count('class="band"') >= 2
    assert re.search(r'class="band"[^>]*>(strong|moderate|weak)<', dom), (
        "the victim's interaction band was not reported"
    )
    assert 'data-affected="false"' in dom, "the unaffected compound was not marked"


def test_interaction_page_offers_a_compound_selector(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_interaction(tmp_path))
    assert "Victim" in dom and "Inhibitor" in dom
    # The selector is only revealed when there is more than one compound.
    assert 'id="compound-controls" hidden' not in dom


def test_single_compound_page_hides_the_interaction_card(chrome: str, tmp_path: Path) -> None:
    # Nothing was compared, so the comparison card must stay out of the way.
    dom = load_dom(chrome, render_demo(tmp_path))
    assert 'id="interaction-card" hidden' in dom
    assert 'id="compound-controls" hidden' in dom


def render_population(tmp_path: Path) -> Path:
    """A population of individuals on one dose."""
    from vivosim import Variability, simulate_population

    compound = Compound(
        name="Population drug",
        partition_coefficients={"adipose": 6.0, "liver": 4.0},
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.2,
        hepatic_intrinsic_clearance_L_per_h=0.4,
    )
    result = simulate_population(
        compound,
        RAT,
        Regimen(IVBolus(2.0)),
        n=30,
        seed=1,
        duration_h=36,
        variability=Variability(hepatic_clearance=0.5),
    )
    return render_html(result, tmp_path / "pop.html")


def test_population_page_loads_without_throwing(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_population(tmp_path))
    error = re.search(r'id="startup-error">(.*?)</pre>', dom, re.S)
    assert error is None, f"the viewer threw on load: {error.group(1) if error else ''}"


def test_population_page_draws_the_band_and_histogram(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_population(tmp_path))
    assert 'class="band-outer"' in dom, "the percentile band was not drawn"
    assert 'class="band-inner"' in dom
    assert 'class="median-path"' in dom, "the median was not drawn"
    assert 'class="trace"' in dom, "individual profiles were not drawn"
    assert 'class="bar"' in dom, "the exposure histogram was not drawn"


def test_pages_without_a_population_hide_the_card(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_demo(tmp_path))
    assert 'id="population-card" hidden' in dom


def test_the_3d_view_is_built(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_demo(tmp_path))
    assert 'class="orb"' in dom, "the 3D organ view was not built"
    assert 'class="ring"' in dom
    # Terms must be able to explain themselves to a reader who did not build
    # the model.
    assert 'class="label term"' in dom or 'class="term"' in dom


def render_validation(tmp_path: Path) -> Path:
    """A prediction drawn against measurements."""
    import numpy as np

    from vivosim.validation import ObservedProfile, validate

    compound = Compound(
        name="Validated",
        default_partition_coefficient=2.0,
        fraction_unbound_plasma=0.5,
        hepatic_intrinsic_clearance_L_per_h=2.0,
    )
    simulation = simulate(compound, RAT, Regimen(IVBolus(1.0)), duration_h=24)
    times = np.array([0.5, 2.0, 6.0, 12.0, 20.0])
    truth = np.interp(times, simulation.time_h, simulation.plasma_concentration())
    observed = ObservedProfile(
        time_h=times,
        concentration=truth * np.array([1.0, 1.3, 0.8, 3.0, 1.1]),
        subject="subject A",
        source="synthetic",
    )
    comparison = validate(simulation, observed, provenance="prediction")
    return render_html(simulation, tmp_path / "val.html", validations=[comparison])


def test_validation_page_loads_without_throwing(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_validation(tmp_path))
    error = re.search(r'id="startup-error">(.*?)</pre>', dom, re.S)
    assert error is None, f"the viewer threw on load: {error.group(1) if error else ''}"


def test_validation_page_draws_both_figures(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_validation(tmp_path))
    assert 'class="unity"' in dom, "the line of perfect agreement was not drawn"
    assert 'class="fold-bound"' in dom, "the two-fold band was not drawn"
    # One dot per usable measurement, across both figures.
    assert dom.count('class="obs-dot"') == 10
    assert "prediction rather than a fit" in dom


def test_pages_without_measurements_hide_the_card(chrome: str, tmp_path: Path) -> None:
    dom = load_dom(chrome, render_demo(tmp_path))
    assert 'id="validation-card" hidden' in dom
