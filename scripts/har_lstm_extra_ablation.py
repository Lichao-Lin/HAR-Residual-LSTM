"""Independent extended ablation runner for the existing 96-day HAR-LSTM core.

All cases share identical forecast origins, training seeds and train/val/test
splits. Select cases before inspecting test results; quick mode is only a smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression


CASES = {
    "full": {},
    "no_layernorm": {"norm": False},
    "no_residual": {"residual": False},
    "ixic_only": {"features": "ixic"},
    "cross_asset_only": {"features": "cross"},
    "with_har_input": {"har": True},
    "lookback_48": {"T": 48},
    "lookback_252": {"T": 252},
}


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 3187, 9254])
    p.add_argument("--epochs", type=positive, default=90)
    p.add_argument("--hidden", type=positive, default=128)
    p.add_argument("--layers", type=positive, default=2)
    p.add_argument("--threads", type=positive, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quick-samples", type=positive, default=256)
    args = p.parse_args(argv)
    if args.quick and args.quick_samples < 64:
        p.error("--quick-samples must be at least 64 for nontrivial holdouts")
    if any(not 0 <= s < 2**32 for s in args.seeds):
        p.error("seeds must be in [0, 2**32)")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        p.error("duplicate seeds/cases would invalidate summaries")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable; use --device cpu or install CUDA-enabled PyTorch")
    candidates = [args.repo] if args.repo else [Path(__file__).resolve().parent.parent, Path.cwd()]
    root = next((r.resolve() for r in candidates if r and (r / "scripts/run_outline_complete.py").is_file()), None)
    if root is None:
        p.error("Supply --repo PATH or run from the HAR-Residual-LSTM checkout")
    sys.path.insert(0, str(root / "scripts"))
    core = importlib.import_module("run_outline_complete")
    if Path(core.__file__).resolve() != root / "scripts/run_outline_complete.py":
        p.error("A different checkout is already imported")
    core.DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.set_num_threads(args.threads)
    if args.data_dir:
        core.DATA = args.data_dir.resolve()
    _, assets, meta, x, h, y, tr, va, te = core.build()
    if args.quick:
        n = min(len(y), args.quick_samples)
        meta, x, h, y = meta.iloc[-n:].reset_index(drop=True), x[-n:], h[-n:], y[-n:]
        args.epochs, args.hidden, args.layers = min(args.epochs, 2), min(args.hidden, 16), 1
        # Keep the same seed list across cases; quick mode uses one seed.
        args.seeds = args.seeds[:1]
    tr, va, te = core.split_indices(len(y))
    if not all(np.isfinite(a).all() for a in (x, h, y)):
        p.error("Nonfinite model input or target")
    if any(b[0] - a[-1] - 1 < core.H - 1 for a, b in ((tr, va), (va, te))):
        p.error("The split gap is shorter than horizon - 1")
    hp = LinearRegression().fit(h[tr], y[tr]).predict(h)
    residual = y - hp
    cases = list(dict.fromkeys(["full", *args.cases]))
    predictions = {"Date": meta.Date.iloc[te].to_numpy(), "y": y[te], "HAR": hp[te]}
    rows = []
    ixic = assets.index("IXIC")
    for seed in args.seeds:
        for name in cases:
            config = {"T": 96, "norm": True, "har": False, "features": "all", "residual": True, **CASES[name]}
            selected = ([ixic] if config["features"] == "ixic" else
                        [j for j in range(len(assets)) if j != ixic] if config["features"] == "cross" else list(range(len(assets))))
            if not selected:
                p.error("No remaining cross-asset features")
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # no_residual changes only the target/additive baseline, not the head.
            pv, pt, params, epochs = core.train_seq(
                "lstm", x[:, :, selected], h, residual if config["residual"] else y,
                tr, va, te, T=config["T"], hidden=args.hidden, layers=args.layers,
                norm=config["norm"], har=config["har"], seed=seed, epochs=args.epochs,
            )
            alpha = None
            if config["residual"]:
                grid = np.linspace(0, 1.25, 126)
                alpha = float(grid[np.argmin([np.mean((y[va] - hp[va] - a * pv)**2) for a in grid])])
                pt = hp[te] + alpha * pt
            if not np.isfinite(pt).all():
                raise RuntimeError(f"Nonfinite predictions in {name}, seed {seed}")
            metrics = core.metric(y[te], pt)
            rows.append({"case": name, "seed": seed, "alpha": alpha, "parameters": params,
                         "epochs": epochs, "lookback": config["T"], "input_features": len(selected), **metrics})
            predictions[f"{name}_seed{seed}"] = pt
            print(json.dumps(rows[-1]), flush=True)
    frame = pd.DataFrame(rows)
    reference = frame[frame.case.eq("full")].set_index("seed").MSE
    frame["delta_MSE_vs_full_same_seed"] = frame.MSE - frame.seed.map(reference)
    tag = "quick" if args.quick else "full"
    output = (args.output_dir.resolve() if args.output_dir else root / "outputs/extra_ablation") / f"{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "per_seed.csv", index=False)
    frame.groupby("case")[["MSE", "MAE", "R2", "QLIKE", "delta_MSE_vs_full_same_seed"]].agg(["mean", "std"]).to_csv(output / "summary.csv")
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame([{"split": name, "n": len(ii), "start": str(meta.Date.iloc[ii[0]].date()), "end": str(meta.Date.iloc[ii[-1]].date())}
                  for name, ii in (("train", tr), ("validation", va), ("test", te))]).to_csv(output / "split_summary.csv", index=False)
    metadata = {"quick": args.quick, "research_result": not args.quick, "seeds": args.seeds, "cases": cases,
                "hidden": args.hidden, "layers": args.layers, "epochs": args.epochs,
                "device": str(core.DEVICE), "torch": torch.__version__, "gap": core.SPLIT_GAP, "horizon": core.H,
                "assets": assets, "HAR_uses_IXIC": True,
                "cross_asset_only_scope": "Only the residual sequence excludes IXIC; the HAR baseline still uses IXIC.",
                "input_sha256": {name: hashlib.sha256((core.DATA / name).read_bytes()).hexdigest() for name in ("returns.csv", "prices.csv")}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved local outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
