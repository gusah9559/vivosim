"""The viewer must carry the result faithfully and stand alone as a file."""

from __future__ import annotations

import json
import re

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
        assert tissue in payload["series"]
        assert tissue in payload["tissueOrder"]
    assert "plasma" in payload["series"]


def test_every_series_matches_the_time_grid() -> None:
    payload = build_payload(result())
    length = len(payload["time"])
    for name, values in payload["series"].items():
        assert len(values) == length, name


def test_payload_values_match_the_result() -> None:
    simulation = result()
    payload = build_payload(simulation)
    assert payload["series"]["plasma"] == pytest.approx(
        list(simulation.plasma_concentration())
    )
    assert payload["series"]["liver"] == pytest.approx(
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
    index = payload["defaultTimeIndex"]
    assert 0 < index < len(payload["time"])
    assert simulation.tissue_concentration("liver")[index] > 0


def test_default_visible_series_are_real_series() -> None:
    payload = build_payload(result())
    assert payload["defaultVisible"]
    assert all(name in payload["series"] for name in payload["defaultVisible"])
    # The colour palette has six slots, so the default must not exceed them.
    assert len(payload["defaultVisible"]) <= 6


def test_payload_is_strict_json() -> None:
    # NaN and Infinity are not valid JSON; a single one would stop the page
    # parsing its own data.
    payload = build_payload(result())
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))
    assert round_tripped["compound"] == "Example"


def test_title_overrides_the_compound_name() -> None:
    assert build_payload(result(), title="Study 7")["compound"] == "Study 7"


def test_regimen_is_described_for_each_route() -> None:
    iv = build_payload(result(Regimen(IVBolus(2.0))))["regimen"]
    assert "IV bolus" in iv and "2 mg" in iv

    oral = build_payload(result(Regimen(OralDose(5.0, absorption_rate_const_per_h=1.0))))
    assert "oral" in oral["regimen"]

    infusion = build_payload(result(Regimen(IVInfusion(3.0, duration_h=2.0))))
    assert "over 2 h" in infusion["regimen"]


def test_empty_regimen_is_described_without_crashing() -> None:
    assert "No dose" in build_payload(result(Regimen()))["regimen"]


def test_dose_markers_are_dropped_when_too_many() -> None:
    # Twenty dose lines would turn the plot into a picket fence.
    crowded = result(repeated(IVBolus(1.0), every_h=1.0, count=20), duration_h=24)
    assert build_payload(crowded)["doseEvents"] == []

    sparse = result(repeated(IVBolus(1.0), every_h=6.0, count=3), duration_h=24)
    assert len(build_payload(sparse)["doseEvents"]) == 3


def test_tiles_report_the_headline_metrics() -> None:
    payload = build_payload(result())
    labels = [tile["label"] for tile in payload["tiles"]]
    assert "Peak plasma" in labels
    assert "Vss" in labels
    assert "Mass balance" in labels


def test_anatomy_scale_is_the_peak_tissue_concentration() -> None:
    simulation = result()
    payload = build_payload(simulation)
    peak = max(
        max(simulation.tissue_concentration(tissue)) for tissue in RAT.tissues
    )
    assert payload["anatomyMax"] == pytest.approx(peak)


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
    assert "plasma" in payload["series"]
    assert len(payload["time"]) == len(payload["series"]["plasma"])


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
    assert json.loads(payload_block.group(1).replace("<\\/", "</"))["compound"] == HOSTILE
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
