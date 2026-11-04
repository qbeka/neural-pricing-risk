"""One command line for every step of the project."""

import argparse
import json

from asianrisk.analytic import conditional_lower_bound_call, moment_matched_call
from asianrisk.contract import Contract, Market
from asianrisk.reference import SimulationConfig, price_reference


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

    args = parser.parse_args()
    args.run(args)
