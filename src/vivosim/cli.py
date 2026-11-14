"""The command line: run a simulation without writing Python.

Describe the study in a small text file, run one command, get a page::

    vivosim run study.toml --open

The library underneath is the same one; this just spares you the import
boilerplate when all you want is to change a dose and look at the result.

    vivosim demo                 list the built-in examples
    vivosim demo interaction     run one of them
    vivosim new study.toml       write a commented starter file
    vivosim run study.toml       simulate and render
    vivosim validate             check the model against published data
    vivosim serve                open a browser console to pick and run
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Any

from . import __version__
from .compound import Compound
from .conditions import CONDITIONS
from .dosing import IVBolus, IVInfusion, OralDose, Regimen
from .interaction import assess_interaction
from .kinetics import Inhibition, Metabolism
from .physiology import get_physiology
from .population import Variability, simulate_population
from .simulate import simulate
from .viewer import render_html

__all__ = ["main"]

STARTER = '''# A vivosim study. Run it with:  vivosim run study.toml --open
#
# Units are milligrams, litres and hours throughout.

[species]
name = "rat"              # rat, mouse or human
body_weight_kg = 0.25
# condition = "hepatic impairment (moderate)"   # see: vivosim conditions

[simulation]
duration_h = 48
# population = 200        # simulate this many individuals instead of one
# seed = 42               # always set a seed if you will report the result

# One [[compound]] block per compound. Give two or more and they are simulated
# together, competing for any enzyme they share.
[[compound]]
name = "drug A"
fraction_unbound_plasma = 0.2       # measured
blood_plasma_ratio = 1.0            # measured
default_partition_coefficient = 2.0 # or set smiles = "..." to predict these
hepatic_intrinsic_clearance_L_per_h = 2.0

  [[compound.dose]]
  route = "iv"            # iv, infusion or oral
  amount_mg = 1.0
  time_h = 0.0

# A second compound that blocks the first one's enzyme:
#
# [[compound]]
# name = "inhibitor"
# fraction_unbound_plasma = 0.5
# default_partition_coefficient = 2.0
#
#   [[compound.inhibits]]
#   enzyme = "CYP3A4"
#   ki_mg_per_L = 0.05
#
#   [[compound.dose]]
#   route = "infusion"
#   amount_mg = 20.0
#   duration_h = 48.0
'''


class ConfigError(Exception):
    """A study file that cannot be read as written."""


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise ConfigError(
                "reading TOML on Python 3.10 needs the tomli package: "
                "pip install tomli"
            ) from None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where} is missing required key {key!r}")
    return mapping[key]


def _build_dose(spec: dict[str, Any], where: str):
    route = str(_require(spec, "route", where)).lower()
    amount = float(_require(spec, "amount_mg", where))

    if route == "iv":
        return IVBolus(amount_mg=amount, time_h=float(spec.get("time_h", 0.0)))
    if route == "infusion":
        return IVInfusion(
            amount_mg=amount,
            duration_h=float(_require(spec, "duration_h", where)),
            start_h=float(spec.get("time_h", spec.get("start_h", 0.0))),
        )
    if route == "oral":
        return OralDose(
            amount_mg=amount,
            time_h=float(spec.get("time_h", 0.0)),
            absorption_rate_const_per_h=float(
                spec.get("absorption_rate_const_per_h", 1.0)
            ),
            bioavailable_fraction=float(spec.get("bioavailable_fraction", 1.0)),
            lag_h=float(spec.get("lag_h", 0.0)),
        )
    raise ConfigError(f"{where}: route must be 'iv', 'infusion' or 'oral', got {route!r}")


def _build_compound(spec: dict[str, Any], index: int) -> tuple[Compound, Regimen]:
    where = f"compound {index + 1}"
    name = str(spec.get("name", f"compound {index + 1}"))

    metabolism = tuple(
        Metabolism(
            enzyme=str(_require(entry, "enzyme", f"{where} metabolism")),
            vmax_mg_per_h=float(_require(entry, "vmax_mg_per_h", f"{where} metabolism")),
            km_mg_per_L=float(_require(entry, "km_mg_per_L", f"{where} metabolism")),
        )
        for entry in spec.get("metabolism", [])
    )
    inhibits = tuple(
        Inhibition(
            enzyme=str(_require(entry, "enzyme", f"{where} inhibits")),
            ki_mg_per_L=float(_require(entry, "ki_mg_per_L", f"{where} inhibits")),
        )
        for entry in spec.get("inhibits", [])
    )

    smiles = spec.get("smiles")
    if smiles:
        from .partitioning import compound_from_properties
        from .structure import properties_from_smiles

        properties = properties_from_smiles(
            str(smiles),
            fraction_unbound_plasma=float(
                _require(spec, "fraction_unbound_plasma", where)
            ),
            name=name,
            pka=spec.get("pka"),
            ionization=spec.get("ionization"),
            blood_plasma_ratio=float(spec.get("blood_plasma_ratio", 1.0)),
            logp=spec.get("logp"),
        )
        compound = compound_from_properties(
            properties,
            hepatic_intrinsic_clearance_L_per_h=float(
                spec.get("hepatic_intrinsic_clearance_L_per_h", 0.0)
            ),
            renal_clearance_L_per_h=float(spec.get("renal_clearance_L_per_h", 0.0)),
        ).replace(metabolism=metabolism, inhibits=inhibits)
    else:
        compound = Compound(
            name=name,
            partition_coefficients=dict(spec.get("partition_coefficients", {})),
            default_partition_coefficient=float(
                spec.get("default_partition_coefficient", 1.0)
            ),
            blood_plasma_ratio=float(spec.get("blood_plasma_ratio", 1.0)),
            fraction_unbound_plasma=float(spec.get("fraction_unbound_plasma", 1.0)),
            hepatic_intrinsic_clearance_L_per_h=float(
                spec.get("hepatic_intrinsic_clearance_L_per_h", 0.0)
            ),
            renal_clearance_L_per_h=float(spec.get("renal_clearance_L_per_h", 0.0)),
            metabolism=metabolism,
            inhibits=inhibits,
            molecular_weight=spec.get("molecular_weight"),
        )

    doses = [
        _build_dose(entry, f"{where} dose {i + 1}")
        for i, entry in enumerate(spec.get("dose", []))
    ]
    if not doses:
        raise ConfigError(f"{where} has no [[compound.dose]] block")
    return compound, Regimen(*doses)


def _run_study(config: dict[str, Any]):
    species = config.get("species", {})
    physiology = get_physiology(
        str(species.get("name", "rat")), species.get("body_weight_kg")
    )

    compound_specs = config.get("compound", [])
    if not compound_specs:
        raise ConfigError("the study has no [[compound]] block")

    built = [_build_compound(spec, i) for i, spec in enumerate(compound_specs)]
    compounds = [c for c, _ in built]
    regimens = [r for _, r in built]

    condition_name = species.get("condition")
    if condition_name:
        if condition_name not in CONDITIONS:
            available = "\n  ".join(sorted(CONDITIONS))
            raise ConfigError(
                f"unknown condition {condition_name!r}. Available:\n  {available}"
            )
        condition = CONDITIONS[condition_name]
        physiology = condition.apply_physiology(physiology)
        compounds = [condition.apply_compound(c) for c in compounds]

    settings = config.get("simulation", {})
    duration = settings.get("duration_h")
    population_size = settings.get("population")

    if population_size:
        if len(compounds) > 1:
            raise ConfigError(
                "a population run takes one compound; remove the extra "
                "[[compound]] blocks or drop 'population'"
            )
        return simulate_population(
            compounds[0],
            physiology,
            regimens[0],
            n=int(population_size),
            seed=settings.get("seed"),
            duration_h=duration,
            variability=Variability(**settings.get("variability", {})),
        )

    if len(compounds) > 1:
        return assess_interaction(
            compounds, physiology, regimens, duration_h=duration
        )
    return simulate(compounds[0], physiology, regimens[0], duration_h=duration)


def _open(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


# -- subcommands -----------------------------------------------------------


def _cmd_new(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite")
        return 1
    path.write_text(STARTER, encoding="utf-8")
    print(f"wrote {path}")
    print(f"edit it, then:  vivosim run {path} --open")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load_toml(Path(args.config))
    result = _run_study(config)
    output = Path(args.output) if args.output else Path(args.config).with_suffix(".html")
    render_html(result, output, title=config.get("title"))
    print(f"wrote {output}")
    if args.open:
        _open(output)
    return 0


DEMOS = {
    "single": "one compound in a rat, read from its structure",
    "interaction": "two compounds competing for one enzyme",
    "population": "200 individuals on the same dose",
    "condition": "the same dose in a healthy and an impaired animal",
}


def _cmd_demo(args: argparse.Namespace) -> int:
    if not args.which:
        print("built-in examples:\n")
        for name, description in DEMOS.items():
            print(f"  {name:<12} {description}")
        print("\nrun one with:  vivosim demo interaction --open")
        return 0

    from .conditions import hepatic_impairment

    output = Path(args.output) if args.output else Path(f"vivosim-{args.which}.html")
    rat = get_physiology("rat")

    if args.which == "single":
        compound = Compound(
            name="Example",
            partition_coefficients={"adipose": 9.0, "liver": 4.0, "brain": 0.6},
            default_partition_coefficient=1.5,
            blood_plasma_ratio=1.1,
            fraction_unbound_plasma=0.2,
            hepatic_intrinsic_clearance_L_per_h=3.0,
        )
        result = simulate(
            compound,
            rat,
            Regimen(IVBolus(2.0), OralDose(5.0, time_h=6.0, absorption_rate_const_per_h=0.9)),
            duration_h=36,
        )
    elif args.which == "interaction":
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
        result = assess_interaction(
            [victim, inhibitor],
            rat,
            [Regimen(IVBolus(1.0)), Regimen(IVInfusion(20.0, duration_h=72.0))],
            duration_h=72,
        )
    elif args.which == "population":
        compound = Compound(
            name="Example",
            partition_coefficients={"adipose": 6.0, "liver": 4.0},
            default_partition_coefficient=2.0,
            fraction_unbound_plasma=0.2,
            hepatic_intrinsic_clearance_L_per_h=0.4,
        )
        result = simulate_population(
            compound, rat, Regimen(IVBolus(2.0)), n=200, seed=1, duration_h=36
        )
    elif args.which == "condition":
        compound = Compound(
            name="Example",
            default_partition_coefficient=2.0,
            fraction_unbound_plasma=0.2,
            hepatic_intrinsic_clearance_L_per_h=0.4,
        )
        sick_compound, sick_body = hepatic_impairment("severe").apply(compound, rat)
        healthy = simulate(compound, rat, Regimen(IVBolus(1.0)), duration_h=48)
        impaired = simulate(sick_compound, sick_body, Regimen(IVBolus(1.0)), duration_h=48)
        print(f"  healthy  AUC {healthy.auc('plasma'):.2f} mg*h/L")
        print(f"  impaired AUC {impaired.auc('plasma'):.2f} mg*h/L "
              f"({impaired.auc('plasma') / healthy.auc('plasma'):.2f}x)")
        result = impaired
    else:  # pragma: no cover - argparse restricts the choices
        raise ConfigError(f"unknown demo {args.which!r}")

    render_html(result, output)
    print(f"wrote {output}")
    if args.open:
        _open(output)
    return 0


def _cmd_conditions(args: argparse.Namespace) -> int:
    print("conditions available for [species].condition:\n")
    for name, condition in CONDITIONS.items():
        print(f"  {name}")
        print(f"      {condition.description}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve as run_server

    run_server(port=args.port, open_browser=not args.no_open)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    examples = Path(__file__).parent.parent.parent / "examples"
    script = examples / "validate_theophylline.py"
    if not script.exists():
        print(
            "the validation example ships with the source checkout, not the "
            "installed package; clone the repository to run it"
        )
        return 1
    sys.path.insert(0, str(examples))
    from validate_theophylline import main as validate_main  # type: ignore[import-not-found]

    return int(validate_main())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vivosim",
        description="Physiologically based pharmacokinetic simulation.",
        epilog="Start with:  vivosim demo",
    )
    parser.add_argument("--version", action="version", version=f"vivosim {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    new = subparsers.add_parser("new", help="write a commented starter study file")
    new.add_argument("path", nargs="?", default="study.toml")
    new.add_argument("--force", action="store_true", help="overwrite an existing file")
    new.set_defaults(func=_cmd_new)

    run = subparsers.add_parser("run", help="simulate a study file and render it")
    run.add_argument("config", help="path to a .toml study file")
    run.add_argument("-o", "--output", help="where to write the HTML")
    run.add_argument("--open", action="store_true", help="open the page when done")
    run.set_defaults(func=_cmd_run)

    demo = subparsers.add_parser("demo", help="run a built-in example")
    demo.add_argument("which", nargs="?", choices=sorted(DEMOS))
    demo.add_argument("-o", "--output", help="where to write the HTML")
    demo.add_argument("--open", action="store_true", help="open the page when done")
    demo.set_defaults(func=_cmd_demo)

    conditions = subparsers.add_parser("conditions", help="list the disease states")
    conditions.set_defaults(func=_cmd_conditions)

    serve = subparsers.add_parser(
        "serve", help="open a browser console to pick compounds and run them"
    )
    serve.add_argument("-p", "--port", type=int, default=8765)
    serve.add_argument(
        "--no-open", action="store_true", help="do not open a browser"
    )
    serve.set_defaults(func=_cmd_serve)

    validate = subparsers.add_parser(
        "validate", help="check the model against published data"
    )
    validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``vivosim`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
