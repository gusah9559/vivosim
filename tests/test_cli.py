"""The command line, which is how someone uses this without writing Python."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vivosim.cli import STARTER, main

MINIMAL = """
[species]
name = "rat"
body_weight_kg = 0.25

[simulation]
duration_h = 12

[[compound]]
name = "drug A"
fraction_unbound_plasma = 0.2
default_partition_coefficient = 2.0
hepatic_intrinsic_clearance_L_per_h = 2.0

  [[compound.dose]]
  route = "iv"
  amount_mg = 1.0
"""


def write(tmp_path: Path, text: str, name: str = "study.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def payload_of(html: Path) -> dict:
    import re

    match = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>',
        html.read_text(encoding="utf-8"),
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</"))


# -- plumbing -------------------------------------------------------------


def test_no_arguments_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "vivosim" in capsys.readouterr().out


def test_demo_lists_the_examples(capsys) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    for name in ("single", "interaction", "population", "condition"):
        assert name in out


def test_conditions_are_listed_with_their_meaning(capsys) -> None:
    assert main(["conditions"]) == 0
    out = capsys.readouterr().out
    assert "hepatic impairment (moderate)" in out
    assert "Reduced hepatic blood flow" in out


# -- the starter file -----------------------------------------------------


def test_new_writes_a_file_that_actually_runs(tmp_path: Path, capsys) -> None:
    # A starter that does not run is worse than no starter.
    path = tmp_path / "study.toml"
    assert main(["new", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == STARTER

    output = tmp_path / "out.html"
    assert main(["run", str(path), "-o", str(output)]) == 0
    assert output.exists()
    capsys.readouterr()


def test_new_refuses_to_clobber_without_force(tmp_path: Path, capsys) -> None:
    path = write(tmp_path, "keep me")
    assert main(["new", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == "keep me"
    assert main(["new", str(path), "--force"]) == 0
    assert path.read_text(encoding="utf-8") == STARTER
    capsys.readouterr()


# -- running a study ------------------------------------------------------


def test_run_renders_beside_the_config_by_default(tmp_path: Path, capsys) -> None:
    config = write(tmp_path, MINIMAL)
    assert main(["run", str(config)]) == 0
    assert (tmp_path / "study.html").exists()
    capsys.readouterr()


def test_run_simulates_what_the_file_says(tmp_path: Path, capsys) -> None:
    config = write(tmp_path, MINIMAL)
    output = tmp_path / "out.html"
    assert main(["run", str(config), "-o", str(output)]) == 0
    payload = payload_of(output)
    assert payload["compounds"][0]["name"] == "drug A"
    assert payload["time"][-1] == pytest.approx(12.0)
    capsys.readouterr()


def test_two_compounds_are_compared_as_an_interaction(tmp_path: Path, capsys) -> None:
    config = write(
        tmp_path,
        MINIMAL
        + """
[[compound]]
name = "inhibitor"
fraction_unbound_plasma = 0.5
default_partition_coefficient = 2.0

  [[compound.inhibits]]
  enzyme = "CYP3A4"
  ki_mg_per_L = 0.05

  [[compound.dose]]
  route = "infusion"
  amount_mg = 20.0
  duration_h = 12.0
""",
    )
    output = tmp_path / "out.html"
    assert main(["run", str(config), "-o", str(output)]) == 0
    payload = payload_of(output)
    # Two compounds given together are compared alone against together.
    assert payload["interaction"] is not None
    assert len(payload["compounds"]) == 2
    capsys.readouterr()


def test_a_population_run_produces_a_band(tmp_path: Path, capsys) -> None:
    config = write(
        tmp_path,
        MINIMAL.replace(
            "duration_h = 12", "duration_h = 12\npopulation = 8\nseed = 3"
        ),
    )
    output = tmp_path / "out.html"
    assert main(["run", str(config), "-o", str(output)]) == 0
    assert payload_of(output)["population"]["n"] == 8
    capsys.readouterr()


def test_a_condition_changes_the_result(tmp_path: Path, capsys) -> None:
    healthy = tmp_path / "healthy.html"
    impaired = tmp_path / "impaired.html"
    main(["run", str(write(tmp_path, MINIMAL, "a.toml")), "-o", str(healthy)])
    main([
        "run",
        str(
            write(
                tmp_path,
                MINIMAL.replace(
                    'name = "rat"',
                    'name = "rat"\ncondition = "hepatic impairment (severe)"',
                ),
                "b.toml",
            )
        ),
        "-o",
        str(impaired),
    ])
    well = payload_of(healthy)["compounds"][0]["series"]["plasma"]
    unwell = payload_of(impaired)["compounds"][0]["series"]["plasma"]
    assert max(unwell) >= max(well)
    assert unwell[-1] > well[-1]
    capsys.readouterr()


def test_demos_render(tmp_path: Path, capsys) -> None:
    for which in ("single", "interaction", "condition"):
        output = tmp_path / f"{which}.html"
        assert main(["demo", which, "-o", str(output)]) == 0
        assert output.exists()
    capsys.readouterr()


# -- errors a person will actually hit ------------------------------------


def test_a_study_with_no_compound_says_so(tmp_path: Path, capsys) -> None:
    config = write(tmp_path, '[species]\nname = "rat"\n')
    assert main(["run", str(config)]) == 1
    assert "no [[compound]] block" in capsys.readouterr().err


def test_a_compound_with_no_dose_says_so(tmp_path: Path, capsys) -> None:
    config = write(tmp_path, '[[compound]]\nname = "x"\n')
    assert main(["run", str(config)]) == 1
    assert "no [[compound.dose]] block" in capsys.readouterr().err


def test_an_unknown_route_lists_the_real_ones(tmp_path: Path, capsys) -> None:
    config = write(
        tmp_path,
        '[[compound]]\nname = "x"\n\n  [[compound.dose]]\n  route = "injection"\n'
        "  amount_mg = 1.0\n",
    )
    assert main(["run", str(config)]) == 1
    error = capsys.readouterr().err
    assert "'iv', 'infusion' or 'oral'" in error


def test_an_unknown_condition_lists_the_real_ones(tmp_path: Path, capsys) -> None:
    config = write(
        tmp_path, MINIMAL.replace('name = "rat"', 'name = "rat"\ncondition = "flu"')
    )
    assert main(["run", str(config)]) == 1
    error = capsys.readouterr().err
    assert "unknown condition" in error
    assert "hepatic impairment (moderate)" in error


def test_a_missing_required_key_names_it(tmp_path: Path, capsys) -> None:
    config = write(
        tmp_path,
        '[[compound]]\nname = "x"\n\n  [[compound.dose]]\n  route = "infusion"\n'
        "  amount_mg = 1.0\n",
    )
    assert main(["run", str(config)]) == 1
    assert "duration_h" in capsys.readouterr().err


def test_broken_toml_is_reported_as_such(tmp_path: Path, capsys) -> None:
    config = write(tmp_path, "[species\nname = rat")
    assert main(["run", str(config)]) == 1
    assert "not valid TOML" in capsys.readouterr().err


def test_a_population_of_several_compounds_is_refused(tmp_path: Path, capsys) -> None:
    # Competition and population sampling both work; combining them is simply
    # not implemented, and saying so beats a confusing result.
    config = write(
        tmp_path,
        MINIMAL.replace("duration_h = 12", "duration_h = 12\npopulation = 5")
        + '\n[[compound]]\nname = "b"\n\n  [[compound.dose]]\n  route = "iv"\n'
        "  amount_mg = 1.0\n",
    )
    assert main(["run", str(config)]) == 1
    assert "population run takes one compound" in capsys.readouterr().err
