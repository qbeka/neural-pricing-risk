"""One command line for every step of the project."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from asianrisk.analytic import conditional_lower_bound_call, moment_matched_call
from asianrisk.contract import Contract, Market
from asianrisk.demo import write_demo
from asianrisk.domain import load_domain, sample_dataset
from asianrisk.labels import generate_labels, save_dataset
from asianrisk.evaluate import git_commit, run_sealed_evaluation, sha256
from asianrisk.pipeline import (
    fit_differential,
    fit_grids,
    fit_networks,
    generate_datasets,
    select_simulation_units,
    write_books,
)
from asianrisk.reference import SimulationConfig, price_reference
from asianrisk.report import write_report

SEALED = ("test", "stress_near_expiry", "stress_kink")


def price(args) -> None:
    """Price one contract through every available route and print the records."""
    contract = Contract("cli", "X", args.strike, args.maturity, args.fixings)
    market = Market(args.spot, args.vol, args.rate)
    config = SimulationConfig(n_paths=2**args.log2_paths, seed=args.seed, control=args.control)
    result = price_reference(contract, market, config)
    output = {
        "contract": contract.to_record(),
        "market": market.to_record(),
        "reference": result.to_record(),
    }
    inputs = (args.spot, args.strike, args.vol, args.rate, args.maturity, args.fixings)
    for name, formula in (
        ("lower_bound", conditional_lower_bound_call),
        ("moment_matched", moment_matched_call),
    ):
        value, delta = formula(*inputs)
        output[name] = {"price": float(value), "delta": float(delta)}
    print(json.dumps(output, indent=2))


def require_freeze(freeze_path: Path, config_path: Path) -> None:
    """Refuse sealed work unless a real freeze exists for this exact config."""
    try:
        freeze = json.loads(freeze_path.read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"sealed set refused: no valid freeze at {freeze_path} ({error})")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if freeze.get("config_sha256") != digest:
        raise SystemExit(f"sealed set refused: {config_path} does not match the freeze")


def labels(args) -> None:
    """Generate labels for named datasets. Sealed sets need a matching freeze."""
    config = json.loads(Path(args.config).read_text())
    domain = load_domain(config)
    out = Path(args.out)
    for name in args.sets:
        if name in SEALED:
            require_freeze(Path(args.freeze), Path(args.config))
        spec = config["datasets"][name]
        points = sample_dataset(name, domain, config)
        values, timing = generate_labels(
            points, config["rate"], config["n_fixings"], 2 ** spec["log2_units"], spec["seed"]
        )
        save_dataset(out / f"{name}.npz", points, values, {"name": name, **spec, **timing})
        print(json.dumps({"name": name, "n_points": len(points), **timing}))


def fit(args) -> None:
    """Generate development data, then fit every grid and network. Safe to rerun."""
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    write_books(config, Path(args.books))
    generate_datasets(config, out)
    fit_grids(config, out)
    fit_networks(config, out)
    fit_differential(config, out)


def tune(args) -> None:
    """Pick the simulation size on the development book and save the ladder."""
    config = json.loads(Path(args.config).read_text())
    result = select_simulation_units(config, Path(args.out), Path(args.books))
    path = Path(args.out) / "simulation_ladder.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def freeze(args) -> None:
    """Record the hypothesis, workload, baselines, metrics and decision rule."""
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    out = Path(args.out)
    path = Path(args.freeze)
    if path.exists():
        raise SystemExit(f"{path} already exists: a freeze is never replaced")
    ladder = json.loads((out / "simulation_ladder.json").read_text())
    record = {
        **json.loads(Path(args.statement).read_text()),
        "commit": git_commit(),
        "config_sha256": sha256(config_path),
        "tolerance": config["tolerance"],
        "books": {
            name: sha256(Path(args.books) / f"book_{name}.json") for name in config["books"]
        },
        "simulation_log2_units": ladder["chosen_log2_units"],
        "simulation_ladder_sha256": sha256(out / "simulation_ladder.json"),
        "models": {p.name: sha256(p) for p in sorted((out / "models").glob("*.pt"))},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}")


def evaluate(args) -> None:
    """Label the sealed sets and take every measurement. Needs a matching freeze."""
    config_path = Path(args.config)
    freeze_path = Path(args.freeze)
    require_freeze(freeze_path, config_path)
    config = json.loads(config_path.read_text())
    out = Path(args.out)
    generate_datasets(config, out, SEALED)
    run_sealed_evaluation(
        config, config_path, out, Path(args.books), Path(args.run),
        json.loads(freeze_path.read_text()),
    )


def report(args) -> None:
    """Rebuild tables, plots and the verdict from a recorded run."""
    summary = write_report(Path(args.run))
    print(json.dumps(summary["verdict"], indent=2))


def demo(args) -> None:
    """Generate the offline demo page from a recorded run."""
    write_demo(Path(args.run), Path(args.target))
    print(f"wrote {args.target}")


def smoke(args) -> None:
    """Run every stage on a tiny config in a scratch directory. Takes a few minutes."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        shared = dict(config="configs/smoke.json", out=str(root), books=str(root / "books"))
        freeze_path = root / "freeze.json"
        run_dir = root / "run"
        fit(argparse.Namespace(**shared))
        tune(argparse.Namespace(**shared))
        freeze(argparse.Namespace(
            **shared, statement="configs/freeze_statement.json", freeze=str(freeze_path)
        ))
        evaluate(argparse.Namespace(**shared, freeze=str(freeze_path), run=str(run_dir)))
        report(argparse.Namespace(run=str(run_dir)))
        demo(argparse.Namespace(run=str(run_dir), target=str(root / "site" / "index.html")))
        page = root / "site" / "index.html"
        if "__DATA__" in page.read_text():
            raise SystemExit("demo page was not filled in")
    print("smoke run passed")


def main() -> None:
    parser = argparse.ArgumentParser(prog="asianrisk", description=__doc__)
    commands = parser.add_subparsers(required=True)

    price_command = commands.add_parser("price", help="price one contract")
    price_command.add_argument("--spot", type=float, default=100.0)
    price_command.add_argument("--strike", type=float, default=100.0)
    price_command.add_argument("--vol", type=float, default=0.20)
    price_command.add_argument("--rate", type=float, default=0.03)
    price_command.add_argument("--maturity", type=float, default=1.0)
    price_command.add_argument("--fixings", type=int, default=12)
    price_command.add_argument("--log2-paths", type=int, default=20)
    price_command.add_argument("--seed", type=int, default=1)
    price_command.add_argument("--control", default="conditional")
    price_command.set_defaults(handler=price)

    labels_command = commands.add_parser("labels", help="generate labels for named datasets")
    labels_command.add_argument("--config", default="configs/experiment.json")
    labels_command.add_argument("--out", default="artifacts/data")
    labels_command.add_argument("--freeze", default="results/freeze.json")
    labels_command.add_argument("--sets", nargs="+", default=["train", "validation", "calibration"])
    labels_command.set_defaults(handler=labels)

    for name, run, help_text in (
        ("fit", fit, "generate development data and fit every surface"),
        ("tune", tune, "pick the simulation size on the development book"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/experiment.json")
        command.add_argument("--out", default="artifacts")
        command.add_argument("--books", default="configs")
        command.set_defaults(handler=run)

    freeze_command = commands.add_parser("freeze", help="freeze the plan before sealed data")
    freeze_command.add_argument("--config", default="configs/experiment.json")
    freeze_command.add_argument("--out", default="artifacts")
    freeze_command.add_argument("--books", default="configs")
    freeze_command.add_argument("--statement", default="configs/freeze_statement.json")
    freeze_command.add_argument("--freeze", default="results/freeze.json")
    freeze_command.set_defaults(handler=freeze)

    evaluate_command = commands.add_parser("evaluate", help="run the sealed evaluation")
    evaluate_command.add_argument("--config", default="configs/experiment.json")
    evaluate_command.add_argument("--out", default="artifacts")
    evaluate_command.add_argument("--books", default="configs")
    evaluate_command.add_argument("--freeze", default="results/freeze.json")
    evaluate_command.add_argument("--run", default="results/run")
    evaluate_command.set_defaults(handler=evaluate)

    report_command = commands.add_parser("report", help="rebuild a report from a recorded run")
    report_command.add_argument("--run", default="results/run")
    report_command.set_defaults(handler=report)

    demo_command = commands.add_parser("demo", help="generate the offline demo page")
    demo_command.add_argument("--run", default="results/run")
    demo_command.add_argument("--target", default="site/index.html")
    demo_command.set_defaults(handler=demo)

    smoke_command = commands.add_parser("smoke", help="run every stage on a tiny config")
    smoke_command.set_defaults(handler=smoke)

    args = parser.parse_args()
    args.handler(args)
