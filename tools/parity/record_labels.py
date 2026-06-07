#!/usr/bin/env python3
"""Record Python→Rust parity goldens for the breakout labeler.

Runs the *Python* champion labeler (`ml.labeler.generate_labels_breakout`, the
oracle) on synthetic OHLC and writes golden JSON the Rust port's differential
test checks against:
`crates/ml/src/labeler.rs::tests::python_goldens`.

Uses the Python defaults, which match `BreakoutLabelConfig::default()`. Inputs
are rounded to float32 so both sides see identical bars.

Usage
-----
    python tools/parity/record_labels.py \
        --ruby-src /path/to/fks-full/src/ruby/src \
        --out crates/ml/tests/golden/labels

Requires: numpy + pandas + the ruby `ml` package importable from --ruby-src.
(No torch needed — labeler.py is pure numpy/pandas.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def gen_ohlc(n: int, seed: int):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.002, n)))
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    return f32(high), f32(low), f32(close)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruby-src", required=True, help="path to fks-full/src/ruby/src (so `ml.labeler` imports)")
    ap.add_argument("--out", required=True, help="output golden directory")
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--bars", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ruby_src).resolve()))
    from ml.labeler import generate_labels_breakout  # noqa: E402

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for ci in range(args.cases):
        high, low, close = gen_ohlc(args.bars, args.seed + ci)
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(args.bars, dtype=np.float32),
            }
        )
        labels = generate_labels_breakout(df)  # Python defaults
        case = {
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
            "expected": [int(v) for v in labels.tolist()],
        }
        path = outdir / f"case_{ci}.json"
        path.write_text(json.dumps(case))
        print(f"wrote {path} ({len(labels)} bars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
