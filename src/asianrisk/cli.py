"""One command line for every step of the project."""

import argparse
import hashlib
import json
from pathlib import Path

from asianrisk.analytic import conditional_lower_bound_call, moment_matched_call
from asianrisk.contract import Contract, Market
from asianrisk.domain import load_domain, sample_dataset
from asianrisk.labels import generate_labels, save_dataset
from asianrisk.reference import SimulationConfig, price_reference

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
    price_command.set_defaults(run=price)

    labels_command = commands.add_parser("labels", help="generate labels for named datasets")
    labels_command.add_argument("--config", default="configs/experiment.json")
    labels_command.add_argument("--out", default="artifacts/data")
    labels_command.add_argument("--freeze", default="results/freeze.json")
    labels_command.add_argument("--sets", nargs="+", default=["train", "validation", "calibration"])
    labels_command.set_defaults(run=labels)

    args = parser.parse_args()
    args.run(args)
