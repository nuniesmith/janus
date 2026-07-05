#!/usr/bin/env python3
"""Record Python→Rust parity goldens for the PerAssetCnn feature pipeline.

Runs the *Python* champion feature extractor (`ml.features.extract_features`,
the inference path — the parity oracle) on synthetic OHLCV and writes golden
JSON that the Rust port's differential test checks against:
`crates/ml/src/features/per_asset_cnn.rs::tests::python_goldens`.

Inputs are rounded to float32 first (as the Rust side and the Python extractor
both operate on float32 candles), so both sides see identical bars.

Usage
-----
    python tools/parity/record_features.py \
        --ruby-src /path/to/fks/src/ruby/src \
        --out crates/ml/tests/golden/per_asset_features

Requires: numpy + pandas + the ruby `ml` package importable from --ruby-src.
(No torch needed — features.py is pure numpy/pandas.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def gen_ohlcv(n: int, seed: int):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    vol = rng.uniform(500, 5000, n)
    # Round to float32 so Python and Rust operate on identical inputs.
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    return f32(open_), f32(high), f32(low), f32(close), f32(vol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruby-src", required=True, help="path to fks/src/ruby/src (so `ml.features` imports)")
    ap.add_argument("--out", required=True, help="output golden directory")
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--bars", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ruby_src).resolve()))
    from ml.features import extract_features  # noqa: E402

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for ci in range(args.cases):
        o, h, lo, c, v = gen_ohlcv(args.bars, args.seed + ci)
        df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v})
        imbalance = round(0.5 + ci * 0.3, 4)
        vol_pct = round(0.4 + ci * 0.1, 4)
        wave_ratio = round(-0.5 + ci * 0.4, 4)

        feats = extract_features(
            df, imbalance=imbalance, vol_pct=vol_pct, wave_ratio=wave_ratio, window=args.window
        )
        if feats is None or feats.shape != (20, args.window):
            raise SystemExit(f"extract_features returned unexpected result for case {ci}")

        case = {
            "open": o.tolist(),
            "high": h.tolist(),
            "low": lo.tolist(),
            "close": c.tolist(),
            "volume": v.tolist(),
            "imbalance": imbalance,
            "vol_pct": vol_pct,
            "wave_ratio": wave_ratio,
            "window": args.window,
            # row-major (channel-major) flatten — matches the Rust layout.
            "expected": feats.astype(np.float32).flatten().tolist(),
        }
        path = outdir / f"case_{ci}.json"
        path.write_text(json.dumps(case))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
