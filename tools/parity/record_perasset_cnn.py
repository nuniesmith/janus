#!/usr/bin/env python3
"""Record PyTorch→burn parity goldens for PerAssetCNN.

Part of the Python → Rust migration parity harness (see fks
`docs/architecture/RUST_MIGRATION.md`). This script loads the *Python* champion
`PerAssetCNN` (the parity oracle), runs a set of fixed inputs through it in
`eval()` mode, and writes a golden JSON that the Rust differential test at
`crates/ml/src/models/per_asset_cnn.rs::tests::pytorch_goldens` checks the
`burn` reimplementation against.

The weights are emitted in **burn-native layout** so the Rust loader is dumb:
  - `nn.Linear.weight`  [out, in]    → transposed to [in, out]
  - `nn.Conv1d.weight`  [out, in, k] → unchanged
  - `nn.BatchNorm1d`    weight/bias  → renamed gamma/beta; running_mean/var as-is

Usage
-----
    # against a trained champion checkpoint
    python tools/parity/record_perasset_cnn.py \
        --ruby-src /path/to/fks/src/ruby/src \
        --checkpoint /path/to/fks/models/cnn_btc.pt \
        --out crates/ml/tests/golden/perasset_cnn/cnn_btc.json

    # architecture-parity only (fresh seeded init, no checkpoint needed)
    python tools/parity/record_perasset_cnn.py \
        --ruby-src /path/to/fks/src/ruby/src \
        --out crates/ml/tests/golden/perasset_cnn/seeded.json

Requires: torch (+ the ruby `ml` package importable from --ruby-src).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _tensor(name: str, t: torch.Tensor) -> dict:
    return {"name": name, "shape": list(t.shape), "data": t.detach().cpu().flatten().tolist()}


def _state_dict_to_weightmap(sd: dict[str, torch.Tensor]) -> list[dict]:
    """Map a PerAssetCNN PyTorch state_dict to the burn-native WeightMap."""
    out: list[dict] = []
    for i in range(1, 5):
        out.append(_tensor(f"encoder.block{i}.conv.weight", sd[f"encoder.block{i}.conv.weight"]))
        out.append(_tensor(f"encoder.block{i}.bn.gamma", sd[f"encoder.block{i}.bn.weight"]))
        out.append(_tensor(f"encoder.block{i}.bn.beta", sd[f"encoder.block{i}.bn.bias"]))
        out.append(_tensor(f"encoder.block{i}.bn.running_mean", sd[f"encoder.block{i}.bn.running_mean"]))
        out.append(_tensor(f"encoder.block{i}.bn.running_var", sd[f"encoder.block{i}.bn.running_var"]))
        skip = f"encoder.block{i}.skip.weight"
        if skip in sd:  # blocks 1-3 (in != out)
            out.append(_tensor(skip, sd[skip]))
        # SE: nn.Linear weights are [out, in] → transpose to burn [in, out]
        out.append(_tensor(f"encoder.se{i}.fc1.weight", sd[f"encoder.se{i}.excite.0.weight"].t()))
        out.append(_tensor(f"encoder.se{i}.fc2.weight", sd[f"encoder.se{i}.excite.2.weight"].t()))
    out.append(_tensor("encoder.pool_proj.weight", sd["encoder.pool_proj.0.weight"].t()))
    out.append(_tensor("head.fc1.weight", sd["head.1.weight"].t()))
    out.append(_tensor("head.fc1.bias", sd["head.1.bias"]))
    out.append(_tensor("head.fc2.weight", sd["head.3.weight"].t()))
    out.append(_tensor("head.fc2.bias", sd["head.3.bias"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruby-src", required=True, help="path to fks/src/ruby/src (so `ml.model` imports)")
    ap.add_argument("--checkpoint", default=None, help="champion .pt; omit for a fresh seeded init")
    ap.add_argument("--out", required=True, help="output golden JSON path")
    ap.add_argument("--cases", type=int, default=8, help="number of input cases to record")
    ap.add_argument("--batch", type=int, default=2, help="batch size per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ruby_src).resolve()))
    from ml.model import N_CLASSES, N_FEATURES, PerAssetCNN  # noqa: E402

    torch.manual_seed(args.seed)

    if args.checkpoint:
        model = PerAssetCNN.load(args.checkpoint)
        cfg = {
            "n_features": model.n_features,
            "window": model.window,
            "n_classes": model.n_classes,
            "embedding_dim": model.embedding_dim,
        }
    else:
        model = PerAssetCNN()
        cfg = {"n_features": N_FEATURES, "window": 60, "n_classes": N_CLASSES, "embedding_dim": 64}
    model.eval()

    cases = []
    with torch.no_grad():
        for _ in range(args.cases):
            x = torch.randn(args.batch, cfg["n_features"], cfg["window"])
            logits = model(x)
            cases.append({
                "input": _tensor("input", x),
                "expected_logits": _tensor("expected_logits", logits),
            })

    golden = {
        "config": cfg,
        "weights": _state_dict_to_weightmap(model.state_dict()),
        "cases": cases,
        "tolerance": args.tolerance,
        "source": args.checkpoint or "fresh-seeded-init",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden))
    print(f"wrote {out} ({len(cases)} cases, source={golden['source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
