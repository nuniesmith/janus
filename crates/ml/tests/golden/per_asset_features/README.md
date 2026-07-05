# PerAssetCnn feature-pipeline goldens

Python↔Rust parity vectors for `features::per_asset_cnn::extract_features`
(the bit-faithful port of `fks/src/ruby/src/ml/features.py`). Each `*.json`
here is checked by the `python_goldens` test: it runs the Rust extractor on the
case's OHLCV + scalars and asserts the output matches the recorded **Python**
`extract_features` output within `2e-3`.

The test **skips cleanly when this directory has no `*.json`** (so CI stays
green until goldens are generated). In-sandbox correctness is otherwise covered
by the hand-computed helper unit tests and `extract_features_contract`.

## Generating goldens

Produced from the Python oracle by
[`tools/parity/record_features.py`](../../../../tools/parity/record_features.py),
run where `numpy` + `pandas` + the ruby `ml` package are available (no torch
needed — `features.py` is pure numpy/pandas):

```sh
python tools/parity/record_features.py \
    --ruby-src /path/to/fks/src/ruby/src \
    --out crates/ml/tests/golden/per_asset_features
```

Then `cargo test -p jflow-ml features::per_asset_cnn` verifies parity.

## JSON schema (one case per file)

```jsonc
{
  "open":[...], "high":[...], "low":[...], "close":[...], "volume":[...], // float32 OHLCV
  "imbalance": 0.8, "vol_pct": 0.5, "wave_ratio": 0.3,                    // live scalars
  "window": 60,
  "expected": [ /* N_FEATURES*window = 1200 floats, row-major channel*window+bar */ ]
}
```

Inputs are rounded to float32 by the recorder so both sides see identical bars.
The `expected` layout is channel-major (`channel*window + bar`), matching the
`(20, window)` tensor `PerAssetCnn` consumes.
