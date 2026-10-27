# Python/JavaScript parity check (later stage)

For one fixed JSON array of hold times, record the Python result from `src.features.macd` with `ql=0.6` and `qh=1.0`. The future `webapp/js/macd.js` must return the same value within a small floating-point tolerance.

The comparison must use the same array, quantile definition, inclusive corridor boundaries, pair mask, and absolute consecutive differences. It must never include the typed text itself.
