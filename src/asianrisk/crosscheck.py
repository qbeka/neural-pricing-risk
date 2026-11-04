"""Independent comparison against QuantLib.

QuantLib works in dates, so times are matched exactly: a 360-day day count with
fixings every whole number of days gives t_i = i * T / n with no rounding.
The rate and vol curves use that same day count. The example in
`Examples/AsianOption/AsianOption.cpp` has a dividend yield, an Actual/365
calendar schedule and a fixing on the settlement date, and all three differ
from this contract, so its numbers are not comparable to ours.
"""

import QuantLib as ql

from asianrisk.contract import Contract, Market

_DAY_COUNT = ql.Actual360()
_TODAY = ql.Date(15, ql.January, 2026)


def _option_and_process(contract: Contract, market: Market, average):
    days = contract.maturity * 360 / contract.n_fixings
    if abs(days - round(days)) > 1e-9:
        raise ValueError("maturity * 360 / n_fixings must be a whole number of days")
    ql.Settings.instance().evaluationDate = _TODAY
    fixing_dates = [_TODAY + round(days) * i for i in range(1, contract.n_fixings + 1)]
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, contract.strike)
    exercise = ql.EuropeanExercise(fixing_dates[-1])
    running = 0.0 if average == ql.Average.Arithmetic else 1.0
    option = ql.DiscreteAveragingAsianOption(average, running, 0, fixing_dates, payoff, exercise)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(market.spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, 0.0, _DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, market.rate, _DAY_COUNT)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_TODAY, ql.NullCalendar(), market.vol, _DAY_COUNT)
        ),
    )
    return option, process


def quantlib_geometric_price(contract: Contract, market: Market) -> float:
    option, process = _option_and_process(contract, market, ql.Average.Geometric)
    option.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
    return option.NPV()


def quantlib_arithmetic_mc(
    contract: Contract, market: Market, n_paths: int, seed: int, variance_reduced: bool = False
):
    """QuantLib Monte Carlo, returning price and standard error.

    `variance_reduced` turns on QuantLib's own antithetic and geometric control.
    """
    option, process = _option_and_process(contract, market, ql.Average.Arithmetic)
    option.setPricingEngine(
        ql.MCDiscreteArithmeticAPEngine(
            process, "pseudorandom", requiredSamples=n_paths, seed=seed,
            antitheticVariate=variance_reduced, controlVariate=variance_reduced,
        )
    )
    return option.NPV(), option.errorEstimate()


def quantlib_turnbull_wakeman(contract: Contract, market: Market) -> float:
    option, process = _option_and_process(contract, market, ql.Average.Arithmetic)
    option.setPricingEngine(ql.TurnbullWakemanAsianEngine(process))
    return option.NPV()


def quantlib_finite_difference(
    contract: Contract, market: Market, t_grid: int = 400, x_grid: int = 400, a_grid: int = 200
) -> float:
    option, process = _option_and_process(contract, market, ql.Average.Arithmetic)
    option.setPricingEngine(ql.FdBlackScholesAsianEngine(process, t_grid, x_grid, a_grid))
    return option.NPV()
