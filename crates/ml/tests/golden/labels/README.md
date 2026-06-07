# Breakout-labeler goldens

Python↔Rust parity vectors for `labeler::generate_labels_breakout` (the port of
`fks-full/src/ruby/src/ml/labeler.py::generate_labels_breakout`). Each `*.json`
is checked by the `python_goldens` test: it runs the Rust labeler on the case's
OHLC and asserts the per-bar labels **exactly** match the recorded Python output
(labels are integers, so parity is exact, not tolerance-based).

The test **skips cleanly when this directory has no `*.json`** (so CI stays
green until goldens are generated). In-sandbox correctness is otherwise covered
by the helper + scenario unit tests in `labeler.rs`.

## Generating goldens

Produced from the Python oracle by
[`tools/parity/record_labels.py`](../../../../tools/parity/record_labels.py),
run where `numpy` + `pandas` + the ruby `ml` package are available (no torch
needed — `labeler.py` is pure numpy/pandas):

```sh
python tools/parity/record_labels.py \
    --ruby-src /path/to/fks-full/src/ruby/src \
    --out crates/ml/tests/golden/labels
```

Then `cargo test -p jflow-ml labeler` verifies parity.

## JSON schema (one case per file)

```jsonc
{
  "high":  [...], "low": [...], "close": [...],  // float32 OHLC
  "expected": [ /* one i64 label per bar: 0 flat, 1 long, 2 short, 3 loss */ ]
}
```

Uses the labeler defaults, which match `BreakoutLabelConfig::default()`
(consolidation_bars 20, atr_mult 5.5, breakout_wait 40, tp/sl 1.5/1.0,
max_hold 90, atr_period 14).
