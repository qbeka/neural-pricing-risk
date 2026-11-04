"""Versioned records for contracts, market states, shocks and pricing results.

Units used everywhere in this project:
    spot, strike     currency per share
    vol              annualized, decimal (0.20 means 20% per square root year)
    rate             continuously compounded per year, decimal
    maturity         years
    price            currency per share, today
    delta            currency of price per currency of spot (dimensionless)

Contract definition: a European-style Asian call seen at inception. There are
`n_fixings` equally spaced future fixings at t_i = i * maturity / n_fixings for
i = 1..n_fixings, so the last fixing is at maturity and no fixing is at time 0.
There is no fixing history. The payoff at maturity is max(A - strike, 0), where
A is the arithmetic mean of the fixings. The dividend yield is fixed at zero.
"""

from dataclasses import asdict, dataclass
from typing import Self

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _Record:
    def to_record(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_record(cls, record: dict) -> Self:
        fields = dict(record)
        version = fields.pop("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"expected schema {SCHEMA_VERSION}, got {version}")
        return cls(**fields)


@dataclass(frozen=True)
class Contract(_Record):
    contract_id: str
    underlying: str
    strike: float
    maturity: float
    n_fixings: int
    quantity: float = 1.0  # signed number of options held


@dataclass(frozen=True)
class Market(_Record):
    spot: float
    vol: float
    rate: float


@dataclass(frozen=True)
class Scenario(_Record):
    """A relative shock: spot * (1 + spot_shift) and vol * (1 + vol_shift)."""

    scenario_id: str
    spot_shift: float
    vol_shift: float

    def apply(self, market: Market) -> Market:
        return Market(
            spot=market.spot * (1 + self.spot_shift),
            vol=market.vol * (1 + self.vol_shift),
            rate=market.rate,
        )


@dataclass(frozen=True)
class PriceResult(_Record):
    price: float
    delta: float
    price_se: float
    delta_se: float
    route: str
    runtime_seconds: float
