"""Error summaries. All price errors are absolute and divided by strike."""

import numpy as np

BASIS_POINT = 1e-4


def error_summary(price, delta, reference_price, reference_delta) -> dict:
    """Median and 99th percentile absolute errors, with the denominator."""
    price_error = np.abs(np.asarray(price) - reference_price) / BASIS_POINT
    delta_error = np.abs(np.asarray(delta) - reference_delta)
    return {
        "n": int(len(price_error)),
        "price_median_bp": float(np.median(price_error)),
        "price_p99_bp": float(np.quantile(price_error, 0.99)),
        "price_max_bp": float(np.max(price_error)),
        "delta_median": float(np.median(delta_error)),
        "delta_p99": float(np.quantile(delta_error, 0.99)),
        "delta_max": float(np.max(delta_error)),
    }
