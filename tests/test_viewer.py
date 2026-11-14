"""The viewer must carry the result faithfully and stand alone as a file."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from vivosim import (
    RAT,
    Compound,
    IVBolus,
    IVInfusion,
    OralDose,
    Regimen,
    repeated,
    simulate,
)
from vivosim.viewer import build_payload, render_html


def compound() -> Compound:
    return Compound(
        name="Example",
        partition_coefficients={"adipose": 9.0, "liver": 4.0},
        default_partition_coefficient=1.5,
        fraction_unbound_plasma=0.2,
        hepatic_intrinsic_clearance_L_per_h=1.5,
    )


def block(payload: dict, index: int = 0) -> dict:
    """The per-compound part of a payload, which the page reads for one compound."""
    return payload["compounds"][index]


def result(regimen: Regimen | None = None, **kwargs):
    return simulate(
        compound(), RAT, regimen if regimen is not None else Regimen(IVBolus(1.0)),
        duration_h=kwargs.pop("duration_h", 24), **kwargs
    )


# -- payload --------------------------------------------------------------


def test_payload_covers_every_tissue() -> None:
    payload = build_payload(result())
    # A tissue missing from the payload would simply vanish from the viewer.
    for tissue in RAT.tissues:
        assert tissue in block(payload)["series"]
        assert tissue in payload["tissueOrder"]
    assert "plasma" in block(payload)["series"]


def test_every_series_matches_the_time_grid() -> None:
    payload = build_payload(result())
    length = len(payload["time"])
    for name, values in block(payload)["series"].items():
        assert len(values) == length, name


def test_payload_values_match_the_result() -> None:
    simulation = result()
    payload = build_payload(simulation)
    assert block(payload)["series"]["plasma"] == pytest.approx(
        list(simulation.plasma_concentration())
    )
    assert block(payload)["series"]["liver"] == pytest.approx(
        list(simulation.tissue_concentration("liver"))
    )


def test_every_tissue_is_placed_in_the_anatomy() -> None:
    # An unplaced tissue would be silently absent from the anatomical view.
    payload = build_payload(result())
    placed = {organ["tissue"] for organ in payload["anatomy"]}
    assert placed == set(RAT.tissues)


def test_anatomy_shapes_are_all_drawable() -> None:
    payload = build_payload(result())
    for organ in payload["anatomy"]:
        shape = organ.get("shape")
        assert shape in ("circle", "ellipse", "rect"), organ
        required = {
            "circle": ("cx", "cy", "r"),
            "ellipse": ("cx", "cy", "rx", "ry"),
            "rect": ("x", "y", "w", "h"),
        }[shape]
        assert all(key in organ for key in required), organ


def test_default_time_index_lands_where_tissue_is_loaded() -> None:
    # Opening at t=0 would show an empty animal and read as a broken page.
    simulation = result()
    payload = build_payload(simulation)
    index = block(payload)["defaultTimeIndex"]
    assert 0 < index < len(payload["time"])
    assert simulation.tissue_concentration("liver")[index] > 0


def test_default_visible_series_are_real_series() -> None:
    payload = build_payload(result())
    visible = block(payload)["defaultVisible"]
    assert visible
    assert all(name in block(payload)["series"] for name in visible)
    # The colour palette has six slots, so the default must not exceed them.
    assert len(visible) <= 6


def test_payload_is_strict_json() -> None:
    # NaN and Infinity are not valid JSON; a single one would stop the page
    # parsing its own data.
    payload = build_payload(result())
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))
    assert round_tripped["title"] == "Example"


def test_title_overrides_the_compound_name() -> None:
    assert build_payload(result(), title="Study 7")["title"] == "Study 7"


def test_regimen_is_described_for_each_route() -> None:
    iv = block(build_payload(result(Regimen(IVBolus(2.0)))))["regimen"]
    assert "IV bolus" in iv and "2 mg" in iv

    oral = build_payload(result(Regimen(OralDose(5.0, absorption_rate_const_per_h=1.0))))
    assert "oral" in block(oral)["regimen"]

    infusion = build_payload(result(Regimen(IVInfusion(3.0, duration_h=2.0))))
    assert "over 2 h" in block(infusion)["regimen"]


def test_empty_regimen_is_described_without_crashing() -> None:
    assert "No dose" in block(build_payload(result(Regimen())))["regimen"]


def test_dose_markers_are_dropped_when_too_many() -> None:
    # Twenty dose lines would turn the plot into a picket fence.
    crowded = result(repeated(IVBolus(1.0), every_h=1.0, count=20), duration_h=24)
    assert block(build_payload(crowded))["doseEvents"] == []

    sparse = result(repeated(IVBolus(1.0), every_h=6.0, count=3), duration_h=24)
    assert len(block(build_payload(sparse))["doseEvents"]) == 3


def test_tiles_report_the_headline_metrics() -> None:
    payload = build_payload(result())
    labels = [tile["label"] for tile in block(payload)["tiles"]]
    assert "Peak plasma" in labels
    assert "Vss" in labels
    assert "Mass balance" in labels


def test_anatomy_scale_is_the_peak_tissue_concentration() -> None:
    simulation = result()
    payload = build_payload(simulation)
    peak = max(
        max(simulation.tissue_concentration(tissue)) for tissue in RAT.tissues
    )
    assert block(payload)["anatomyMax"] == pytest.approx(peak)


# -- rendered file --------------------------------------------------------


def test_rendered_page_is_self_contained(tmp_path) -> None:
    path = render_html(result(), tmp_path / "viewer.html")
    html = path.read_text(encoding="utf-8")

    # No placeholder survives, and nothing is fetched from a network: the file
    # has to work offline and when mailed to someone else.
    assert "__PAYLOAD__" not in html
    assert "__TITLE__" not in html
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)


def test_rendered_page_embeds_parsable_data(tmp_path) -> None:
    path = render_html(result(), tmp_path / "viewer.html")
    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>', html, re.S
    )
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert "plasma" in block(payload)["series"]
    assert len(payload["time"]) == len(block(payload)["series"]["plasma"])


def test_rendered_page_carries_the_scope_disclaimer(tmp_path) -> None:
    # The page travels on its own, so the statement of what it is must travel
    # with it rather than living only in the README.
    html = render_html(result(), tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert "not a clinical" in html
    assert "does not recommend doses" in html


def test_render_creates_missing_directories(tmp_path) -> None:
    path = render_html(result(), tmp_path / "nested" / "deep" / "viewer.html")
    assert path.exists()


def test_title_reaches_the_page(tmp_path) -> None:
    html = render_html(result(), tmp_path / "v.html", title="Study 7").read_text(
        encoding="utf-8"
    )
    assert "Study 7" in html


HOSTILE = '</script><img src=x onerror=alert(1)>"'


def test_a_hostile_name_cannot_close_the_data_block(tmp_path) -> None:
    # The name is embedded in a JSON script block, where the only escape is a
    # literal closing tag. The text may appear inside the block -- there it is
    # inert data -- but it must never be able to end it.
    simulation = simulate(
        compound().replace(name=HOSTILE), RAT, Regimen(IVBolus(1.0)), duration_h=4
    )
    html = render_html(simulation, tmp_path / "v.html").read_text(encoding="utf-8")

    payload_block = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>', html, re.S
    )
    assert payload_block is not None
    # The block ends at the template's own closing tag, so the whole payload
    # survives and parses -- the name did not terminate it early.
    assert json.loads(payload_block.group(1).replace("<\\/", "</"))["title"] == HOSTILE
    assert "</script>" not in payload_block.group(1)


def test_a_hostile_name_is_escaped_in_the_page_title(tmp_path) -> None:
    simulation = simulate(
        compound().replace(name=HOSTILE), RAT, Regimen(IVBolus(1.0)), duration_h=4
    )
    html = render_html(simulation, tmp_path / "v.html").read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    assert title is not None
    assert "<img" not in title.group(1)
    assert "&lt;" in title.group(1)


def test_markup_is_never_interpolated_from_label_data(tmp_path) -> None:
    # Tissue names are model data -- a caller can define their own -- and the
    # tooltip and table view build markup by string concatenation, so every
    # label interpolated into HTML must go through the escaping helper.
    html = render_html(result(), tmp_path / "v.html").read_text(encoding="utf-8")
    for fragment in re.findall(r'rows \+=.*?;', html, re.S):
        assert "esc(" in fragment, fragment
    assert 'html += "<th>" + esc(label(name))' in html


# -- several compounds ----------------------------------------------------


def interacting():
    """A victim and an inhibitor contesting one enzyme."""
    from vivosim import Inhibition, Metabolism

    victim = Compound(
        name="Victim",
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
    return victim, inhibitor


def study_result():
    from vivosim import IVInfusion, simulate

    victim, inhibitor = interacting()
    return simulate(
        [victim, inhibitor],
        RAT,
        [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=48.0))],
        duration_h=48,
    )


def interaction_result():
    from vivosim import IVInfusion, assess_interaction

    victim, inhibitor = interacting()
    return assess_interaction(
        [victim, inhibitor],
        RAT,
        [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=48.0))],
        duration_h=48,
    )


def test_a_study_gives_one_block_per_compound() -> None:
    payload = build_payload(study_result())
    assert [block["name"] for block in payload["compounds"]] == ["Victim", "Inhibitor"]
    for compound_block in payload["compounds"]:
        assert len(compound_block["series"]["plasma"]) == len(payload["time"])


def test_a_study_title_names_both_compounds() -> None:
    assert build_payload(study_result())["title"] == "Victim + Inhibitor"


def test_a_plain_study_carries_no_comparison() -> None:
    # Without solo runs there is nothing to compare against, so the page must
    # not claim an interaction it never measured.
    assert build_payload(study_result())["interaction"] is None


def test_an_interaction_report_carries_the_comparison() -> None:
    payload = build_payload(interaction_result())
    interaction = payload["interaction"]
    assert interaction is not None
    assert interaction["enzymes"] == {"CYP3A4": ["Victim", "Inhibitor"]}

    victim = interaction["compounds"][0]
    assert victim["name"] == "Victim"
    assert victim["aucTogether"] > victim["aucAlone"]
    assert victim["aucRatio"] > 1.0
    assert victim["affected"] is True
    assert victim["classification"] in ("weak", "moderate", "strong")
    # The solo profile has to travel with the page, or it cannot be plotted.
    assert len(victim["aloneSeries"]) == len(payload["time"])


def test_the_unaffected_compound_is_marked_unaffected() -> None:
    interaction = build_payload(interaction_result())["interaction"]
    inhibitor = interaction["compounds"][1]
    assert inhibitor["affected"] is False
    assert inhibitor["classification"] == "none"


def test_each_compound_keeps_its_own_dosing_description() -> None:
    payload = build_payload(study_result())
    assert "IV bolus" in payload["compounds"][0]["regimen"]
    assert "over 48 h" in payload["compounds"][1]["regimen"]


def test_the_footer_names_every_compound(tmp_path) -> None:
    html = render_html(interaction_result(), tmp_path / "v.html").read_text(
        encoding="utf-8"
    )
    assert "Victim" in html and "Inhibitor" in html


def test_multi_compound_payload_is_strict_json() -> None:
    payload = build_payload(interaction_result())
    assert json.loads(json.dumps(payload, allow_nan=False))["interaction"] is not None


# -- population -----------------------------------------------------------


def population_result():
    from vivosim import Variability, simulate_population

    return simulate_population(
        compound(),
        RAT,
        Regimen(IVBolus(1.0)),
        n=25,
        seed=5,
        duration_h=24,
        variability=Variability(hepatic_clearance=0.5),
    )


def test_a_single_result_carries_no_population() -> None:
    assert build_payload(result())["population"] is None


def test_a_population_payload_carries_its_bands() -> None:
    payload = build_payload(population_result())
    band = payload["population"]
    assert band is not None
    assert band["n"] == 25
    length = len(payload["time"])
    for key in ("low", "median", "high", "innerLow", "innerHigh", "reference"):
        assert len(band[key]) == length, key


def test_the_population_bands_are_ordered() -> None:
    band = build_payload(population_result())["population"]
    for low, inner_low, median, inner_high, high in zip(
        band["low"], band["innerLow"], band["median"], band["innerHigh"], band["high"],
        strict=True,
    ):
        assert low <= inner_low + 1e-12
        assert inner_low <= median + 1e-12
        assert median <= inner_high + 1e-12
        assert inner_high <= high + 1e-12


def test_population_traces_are_real_individuals() -> None:
    result = population_result()
    band = build_payload(result)["population"]
    # Every drawn trace has to match one of the simulated individuals, or the
    # chart would show curves nobody computed.
    rows = [list(row) for row in result.concentrations]
    for trace in band["traces"]:
        assert any(np.allclose(trace, row) for row in rows)


def test_the_population_histogram_accounts_for_everyone() -> None:
    result = population_result()
    band = build_payload(result)["population"]
    histogram = band["aucHistogram"]
    assert sum(histogram["counts"]) == result.size
    assert len(histogram["edges"]) == len(histogram["counts"]) + 1


def test_the_population_payload_reports_its_spread() -> None:
    band = build_payload(population_result())["population"]
    assert band["aucP5"] <= band["aucMedian"] <= band["aucP95"]
    assert band["foldRange"] >= 1.0


def test_a_population_renders_a_page(tmp_path) -> None:
    html = render_html(population_result(), tmp_path / "pop.html").read_text(
        encoding="utf-8"
    )
    assert "Across a population" in html
    assert "__PAYLOAD__" not in html
