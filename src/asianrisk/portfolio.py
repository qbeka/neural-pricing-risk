"""The frozen book and the shock grid.

A book is a list of contract records plus one market per underlying. Everything
is generated once from a seed and saved to JSON, and the benchmark reads only
the saved file, so the workload cannot drift between runs.

Under a shock each contract is still at inception. The shock changes the market
inputs and no time passes, which matches the contract definition.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asianrisk.contract import SCHEMA_VERSION, Contract, Market, Scenario

TENOR_MONTHS = (1, 2, 3, 4, 6, 9, 12, 15, 18, 21, 24, 30)
SPOT_SHIFTS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
VOL_SHIFTS = (-0.30, -0.15, 0.0, 0.15, 0.30, 0.50)


@dataclass(frozen=True)
class Book:
    book_id: str
    rate: float
    contracts: tuple[Contract, ...]
    markets: dict[str, Market]

    def to_record(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "book_id": self.book_id,
            "rate": self.rate,
            "markets": {name: market.to_record() for name, market in self.markets.items()},
            "contracts": [contract.to_record() for contract in self.contracts],
        }

    @classmethod
    def from_record(cls, record: dict) -> "Book":
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"expected schema {SCHEMA_VERSION}")
        return cls(
            book_id=record["book_id"],
            rate=record["rate"],
            contracts=tuple(Contract.from_record(c) for c in record["contracts"]),
            markets={k: Market.from_record(m) for k, m in record["markets"].items()},
        )


def generate_book(
    book_id: str, seed: int, rate: float, n_fixings: int, n_contracts: int, n_underlyings: int,
    bespoke_maturities: bool,
) -> Book:
    """Random book. Standard tenors share (vol, T) pairs. Bespoke ones rarely do.

    Bespoke maturities are whole days on a 360-day year, drawn over the same range.
    """
    rng = np.random.default_rng(seed)
    markets = {
        f"U{i + 1}": Market(
            spot=float(np.round(rng.uniform(40.0, 200.0), 2)),
            vol=float(np.round(rng.uniform(0.15, 0.45), 3)),
            rate=rate,
        )
        for i in range(n_underlyings)
    }
    names = list(markets)
    contracts = []
    for i in range(n_contracts):
        name = names[rng.integers(n_underlyings)]
        moneyness = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
        if bespoke_maturities:
            maturity = int(rng.integers(30, 901)) / 360.0
        else:
            maturity = TENOR_MONTHS[rng.integers(len(TENOR_MONTHS))] / 12.0
        contracts.append(
            Contract(
                contract_id=f"{book_id}-{i:04d}",
                underlying=name,
                strike=float(np.round(markets[name].spot / moneyness, 2)),
                maturity=maturity,
                n_fixings=n_fixings,
                quantity=float(rng.integers(1, 101) * rng.choice([-1, 1])),
            )
        )
    return Book(book_id, rate, tuple(contracts), markets)


def shock_grid() -> list[Scenario]:
    return [
        Scenario(f"spot{spot:+.0%}_vol{vol:+.0%}", spot, vol)
        for spot in SPOT_SHIFTS
        for vol in VOL_SHIFTS
    ]


def request_points(book: Book, scenario: Scenario) -> np.ndarray:
    """One (log_moneyness, vol, maturity) row per contract under a scenario."""
    rows = []
    for contract in book.contracts:
        market = scenario.apply(book.markets[contract.underlying])
        rows.append((np.log(market.spot / contract.strike), market.vol, contract.maturity))
    return np.array(rows)


def all_requests(book: Book, scenarios: list[Scenario]):
    """Stacked points for every (scenario, contract), plus strikes and quantities."""
    points = np.concatenate([request_points(book, scenario) for scenario in scenarios])
    strikes = np.tile([c.strike for c in book.contracts], len(scenarios))
    quantities = np.tile([c.quantity for c in book.contracts], len(scenarios))
    return points, strikes, quantities


def save_book(book: Book, path: Path) -> None:
    path.write_text(json.dumps(book.to_record(), indent=1) + "\n")


def load_book(path: Path) -> Book:
    return Book.from_record(json.loads(path.read_text()))
