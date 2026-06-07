# PerAssetCNN parity goldens

This directory holds **PyTorch → burn parity golden vectors** for
`PerAssetCnn` (`crates/ml/src/models/per_asset_cnn.rs`). Each `*.json` here is
checked by the `pytorch_goldens` test, which:

1. builds a `PerAssetCnn` from the golden's `config`,
2. injects the golden's `weights` (burn-native layout), and
3. asserts the `burn` forward pass reproduces each case's `expected_logits`
   within `tolerance`.

The test **skips cleanly when this directory has no `*.json` files**, so CI
stays green until goldens are generated. (The architecture is already covered
unconditionally by the in-crate differential test `burn_matches_reference`,
which checks `burn` against an independent raw-`f32` reference.)

## Generating goldens

Goldens are produced from the *Python* champion (the parity oracle) by
[`tools/parity/record_perasset_cnn.py`](../../../../tools/parity/record_perasset_cnn.py),
run in an environment with `torch` and the ruby `ml` package available:

```sh
# champion parity (true 93.5% model)
python tools/parity/record_perasset_cnn.py \
    --ruby-src /path/to/fks-full/src/ruby/src \
    --checkpoint /path/to/fks-full/models/cnn_btc.pt \
    --out crates/ml/tests/golden/perasset_cnn/cnn_btc.json

# architecture parity only (no checkpoint required)
python tools/parity/record_perasset_cnn.py \
    --ruby-src /path/to/fks-full/src/ruby/src \
    --out crates/ml/tests/golden/perasset_cnn/seeded.json
```

Then `cargo test -p jflow-ml per_asset_cnn` verifies parity automatically.

## Golden JSON schema

```jsonc
{
  "config":  { "n_features": 20, "window": 60, "n_classes": 4, "embedding_dim": 64 },
  "weights": [ { "name": "encoder.block1.conv.weight", "shape": [64,20,3], "data": [ ... ] }, ... ],
  "cases":   [ { "input":           { "name": "input",           "shape": [2,20,60], "data": [...] },
                "expected_logits":  { "name": "expected_logits",  "shape": [2,4],     "data": [...] } } ],
  "tolerance": 1e-3,
  "source": "models/cnn_btc.pt"
}
```

Weights are in **burn-native layout** (`Linear` `[in,out]`, `Conv1d`
`[out,in,k]`, BatchNorm `gamma`/`beta`/`running_mean`/`running_var`); the
recorder transposes PyTorch `Linear` weights so the Rust loader needs no
per-layer special-casing.
