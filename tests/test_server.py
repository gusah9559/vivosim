"""The browser console: its library, its study builder, and its HTTP surface."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from vivosim import Study
from vivosim.library import LIBRARY, get_compound, names
from vivosim.population import PopulationResult
from vivosim.results import Result
from vivosim.server import _Handler, build_app_state, run_study

# -- the library ----------------------------------------------------------


def test_every_entry_is_simulatable() -> None:
    from vivosim import RAT, IVBolus, Regimen, simulate

    for key, entry in LIBRARY.items():
        result = simulate(entry.compound, RAT, Regimen(IVBolus(1.0)), duration_h=12)
        assert result.mass_balance_error() < 1e-6, key


def test_entries_carry_an_explanation() -> None:
    # A picker of unexplained names teaches nobody anything.
    for key, entry in LIBRARY.items():
        assert entry.note.strip(), key
        assert entry.label.strip(), key
        assert entry.suggested_dose_mg_per_kg > 0, key


def test_only_the_real_compound_claims_literature_parameters() -> None:
    # Invented numbers under a real drug's name would be a lie that looks like
    # data, so exactly one entry may be marked as measured.
    measured = [key for key, entry in LIBRARY.items() if entry.measured]
    assert measured == ["theophylline"]


def test_archetypes_behave_as_their_notes_claim() -> None:
    from vivosim import RAT, IVBolus, Regimen, simulate

    def run(key: str):
        return simulate(
            get_compound(key), RAT, Regimen(IVBolus(1.0)), duration_h=120
        )

    # "Floods into fat and leaves slowly": late in the run, when plasma has
    # largely cleared, fat must still be holding the compound. Comparing
    # against the plasma peak would prove nothing, since an intravenous dose
    # peaks instantly at a concentration no tissue ever sees.
    fatty = run("lipophilic")
    assert fatty.tissue_concentration("adipose")[-1] > 20 * fatty.plasma_concentration()[-1]

    # "Almost nothing reaches tissue": fat stays below plasma wherever there is
    # signal to compare. The far tail is excluded because both curves are at
    # 1e-13 there, where solver noise decides the ordering and means nothing.
    polar = run("polar_acid")
    plasma = polar.plasma_concentration()
    adipose = polar.tissue_concentration("adipose")
    # 1e-9 mg/L is four orders above the solver's residual noise here and far
    # below anything physically meaningful, so it separates signal from grass.
    signal = plasma > 1e-9
    assert signal.sum() > 50
    assert np.all(adipose[signal] < plasma[signal])

    brain = run("brain_penetrant")
    assert brain.tissue_concentration("brain").max() > brain.tissue_concentration(
        "muscle"
    ).max()


def test_unknown_library_key_is_reported_clearly() -> None:
    with pytest.raises(KeyError, match="no compound 'aspirin' in the library"):
        get_compound("aspirin")


def test_names_lists_everything() -> None:
    assert set(names()) == set(LIBRARY)


# -- the state the console renders from -----------------------------------


def test_app_state_offers_every_control() -> None:
    state = build_app_state()
    assert {c["key"] for c in state["compounds"]} == set(LIBRARY)
    assert {s["key"] for s in state["species"]} >= {"rat", "mouse", "human"}
    assert any(c["key"] == "healthy" for c in state["conditions"])
    assert {r["key"] for r in state["routes"]} == {"iv", "infusion", "oral"}


def test_app_state_is_json_serialisable() -> None:
    assert json.loads(json.dumps(build_app_state()))["routes"]


# -- building a study from the console's controls -------------------------


def test_one_compound_gives_a_plain_result() -> None:
    result, title = run_study(
        {"compounds": [{"key": "lipophilic", "route": "iv", "dose_mg_per_kg": 4}]}
    )
    assert isinstance(result, Result)
    assert title == "Lipophilic neutral"


def test_two_compounds_are_compared() -> None:
    result, title = run_study(
        {
            "compounds": [
                {"key": "victim", "route": "iv", "dose_mg_per_kg": 4},
                {"key": "inhibitor", "route": "infusion", "dose_mg_per_kg": 40,
                 "infusion_h": 48},
            ],
            "duration_h": 48,
        }
    )
    assert result["Interaction victim"].auc_ratio > 1.5
    assert "+" in title


def test_a_population_is_sampled_when_asked() -> None:
    result, _ = run_study(
        {
            "compounds": [{"key": "saturable", "route": "iv", "dose_mg_per_kg": 8}],
            "population": 12,
            "seed": 1,
            "duration_h": 24,
        }
    )
    assert isinstance(result, PopulationResult)
    assert result.size == 12


def test_the_population_size_is_capped() -> None:
    # An unbounded number from a browser field would hang the server.
    result, _ = run_study(
        {
            "compounds": [{"key": "renal", "route": "iv", "dose_mg_per_kg": 4}],
            "population": 100000,
            "duration_h": 4,
        }
    )
    assert result.size <= 500


def test_a_condition_is_applied() -> None:
    healthy, _ = run_study(
        {"compounds": [{"key": "saturable", "route": "iv", "dose_mg_per_kg": 4}],
         "duration_h": 48}
    )
    impaired, _ = run_study(
        {"compounds": [{"key": "saturable", "route": "iv", "dose_mg_per_kg": 4}],
         "condition": "hepatic impairment (severe)", "duration_h": 48}
    )
    assert impaired.auc("plasma") > healthy.auc("plasma")


def test_each_route_builds_its_own_regimen() -> None:
    for route, extra in (
        ("iv", {}),
        ("infusion", {"infusion_h": 3.0}),
        ("oral", {"ka": 0.8}),
    ):
        result, _ = run_study(
            {
                "compounds": [{"key": "renal", "route": route, "dose_mg_per_kg": 4,
                               **extra}],
                "duration_h": 24,
            }
        )
        assert result.cmax("plasma") > 0, route


def test_species_and_weight_are_honoured() -> None:
    result, _ = run_study(
        {
            "compounds": [{"key": "theophylline", "route": "oral",
                           "dose_mg_per_kg": 4, "ka": 1.5}],
            "species": "human",
            "body_weight_kg": 70.0,
            "duration_h": 24,
        }
    )
    assert result.model.physiology.species == "human"
    assert result.model.physiology.body_weight_kg == 70.0


# -- the refusals a user will hit -----------------------------------------


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"compounds": []}, "at least one compound"),
        (
            {"compounds": [{"key": "a"}, {"key": "b"}, {"key": "c"}]},
            "at most two compounds",
        ),
        (
            {"compounds": [{"key": "renal", "dose_mg_per_kg": 0}]},
            "dose must be greater than zero",
        ),
        (
            {"compounds": [{"key": "renal", "route": "patch"}]},
            "unknown route",
        ),
        (
            {"compounds": [{"key": "renal"}], "duration_h": 0},
            "duration must be greater than zero",
        ),
        (
            {
                "compounds": [{"key": "renal"}, {"key": "saturable"}],
                "population": 10,
            },
            "population run takes one compound",
        ),
    ],
)
def test_bad_requests_say_what_is_wrong(request_body: dict, message: str) -> None:
    with pytest.raises((ValueError, KeyError), match=message):
        run_study(request_body)


def test_unknown_names_are_refused() -> None:
    with pytest.raises(KeyError, match="no compound 'nope'"):
        run_study({"compounds": [{"key": "nope"}]})
    with pytest.raises(KeyError, match="no condition"):
        run_study({"compounds": [{"key": "renal"}], "condition": "cursed"})


# -- the HTTP surface -----------------------------------------------------


@pytest.fixture
def console():
    """A console server on an ephemeral port, torn down after the test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def post(base: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/run",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_console_page_is_served_with_its_state(console: str) -> None:
    with urllib.request.urlopen(f"{console}/", timeout=30) as response:  # noqa: S310
        page = response.read().decode()
    assert "__STATE__" not in page, "the state placeholder was never filled"
    assert "vivosim console" in page
    assert "Theophylline" in page


def test_a_run_returns_a_report_that_can_be_fetched(console: str) -> None:
    status, data = post(
        console,
        {
            "compounds": [{"key": "renal", "route": "iv", "dose_mg_per_kg": 4}],
            "duration_h": 12,
        },
    )
    assert status == 200, data
    assert data["tiles"]

    with urllib.request.urlopen(console + data["report"], timeout=30) as response:  # noqa: S310
        report = response.read().decode()
    assert "Concentration over time" in report


def test_an_interaction_run_reports_the_ratio(console: str) -> None:
    status, data = post(
        console,
        {
            "compounds": [
                {"key": "victim", "route": "iv", "dose_mg_per_kg": 4},
                {"key": "inhibitor", "route": "infusion", "dose_mg_per_kg": 40,
                 "infusion_h": 48},
            ],
            "duration_h": 48,
        },
    )
    assert status == 200, data
    assert data["interaction"] is not None
    assert max(c["aucRatio"] for c in data["interaction"]["compounds"]) > 1.5


def test_a_bad_request_is_a_400_with_a_readable_reason(console: str) -> None:
    status, data = post(console, {"compounds": []})
    assert status == 400
    assert "at least one compound" in data["error"]


def test_malformed_json_does_not_crash_the_server(console: str) -> None:
    request = urllib.request.Request(
        f"{console}/api/run", data=b"{not json", method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=30)  # noqa: S310
        raise AssertionError("expected a 400")
    except urllib.error.HTTPError as error:
        assert error.code == 400
        assert "malformed" in json.loads(error.read())["error"]

    # Still alive afterwards.
    with urllib.request.urlopen(f"{console}/", timeout=30) as response:  # noqa: S310
        assert response.status == 200


def test_an_unknown_path_is_a_404(console: str) -> None:
    for path in ("/nope", "/report/deadbeef"):
        try:
            urllib.request.urlopen(console + path, timeout=30)  # noqa: S310
            raise AssertionError(f"expected a 404 for {path}")
        except urllib.error.HTTPError as error:
            assert error.code == 404


def test_study_results_from_the_console_are_the_library_compounds() -> None:
    result, _ = run_study(
        {"compounds": [{"key": "theophylline", "route": "iv", "dose_mg_per_kg": 4}]}
    )
    assert not isinstance(result, Study)
    assert result.model.compound.name == "Theophylline"
