"""CLI for the existing HAR-LSTM models; independent of the other handoff files.

Place in the repository's scripts/ directory, or pass --repo from elsewhere.
Quick runs validate plumbing only; they are not comparable research results.
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


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, help="HAR-Residual-LSTM checkout")
    p.add_argument("--data-dir", type=Path, help="directory with returns.csv and prices.csv")
    p.add_argument("--output-dir", type=Path, help="parent output directory; each run gets a new subdirectory")
    p.add_argument("--model", choices=("har", "direct", "residual", "all"), default="residual")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=positive, default=90)
    p.add_argument("--hidden", type=positive, default=128)
    p.add_argument("--layers", type=positive, default=2)
    p.add_argument("--lookback", type=positive, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=positive, default=4)
    p.add_argument("--quick", action="store_true", help="at most 2 epochs, hidden <=16, 1 layer; one small chronological cohort")
    p.add_argument("--quick-samples", type=positive, default=256)
    p.add_argument("--dry-run", action="store_true", help="validate data and splits without training or writing outputs")
    return p


def load_core(repo):
    candidates = [repo] if repo else [Path(__file__).resolve().parent.parent, Path.cwd()]
    root = next((p.resolve() for p in candidates if p is not None and (p / "scripts/run_outline_complete.py").is_file()), None)
    if root is None:
        raise ValueError("Repository not found: supply --repo PATH or run from the checkout")
    sys.path.insert(0, str(root / "scripts"))
    core = importlib.import_module("run_outline_complete")
    if Path(core.__file__).resolve() != root / "scripts/run_outline_complete.py":
        raise ValueError("A different checkout is already imported in this Python process")
    return root, core


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Install a CUDA-capable PyTorch build or use --device cpu")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def prepare(core, args):
    if args.data_dir is not None:
        core.DATA = args.data_dir.resolve()
    for name in ("returns.csv", "prices.csv"):
        if not (core.DATA / name).is_file():
            raise ValueError(f"Required input missing: {name}")
    _, assets, meta, x, h, y, _, _, _ = core.build()
    if args.quick:
        n = min(len(y), args.quick_samples)
        meta, x, h, y = meta.iloc[-n:].reset_index(drop=True), x[-n:], h[-n:], y[-n:]
    tr, va, te = core.split_indices(len(y))
    if not all(np.isfinite(a).all() for a in (x, h, y)):
        raise ValueError("Nonfinite model input or target")
    for earlier, later in ((tr, va), (va, te)):
        if later[0] - earlier[-1] - 1 < core.H - 1:
            raise ValueError("Split gap is too short for the prediction horizon")
    return assets, meta, x, h, y, tr, va, te


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.quick and args.quick_samples < 64:
        p.error("--quick-samples must be at least 64 for nontrivial holdouts")
    if not 1 <= args.lookback <= 252:
        p.error("--lookback must be between 1 and 252")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be between 0 and 2**32 - 1")
    try:
        device = choose_device(args.device)
        root, core = load_core(args.repo)
        core.DEVICE = device
        torch.set_num_threads(args.threads)
        if args.quick:
            args.epochs, args.hidden, args.layers = min(args.epochs, 2), min(args.hidden, 16), 1
        assets, meta, x, h, y, tr, va, te = prepare(core, args)
    except ValueError as exc:
        p.error(str(exc))
    split = [{"split": name, "n": len(ii), "start": str(meta.Date.iloc[ii[0]].date()), "end": str(meta.Date.iloc[ii[-1]].date())}
             for name, ii in (("train", tr), ("validation", va), ("test", te))]
    print(json.dumps({"quick": args.quick, "device": str(device), "splits": split}, indent=2))
    if args.dry_run:
        return 0
    har = LinearRegression().fit(h[tr], y[tr])
    hp = har.predict(h)
    predictions, rows = {"HAR": hp[te]}, []
    kinds = ("direct", "residual") if args.model == "all" else (() if args.model == "har" else (args.model,))
    for kind in kinds:
        residual = kind == "residual"
        target = y - hp if residual else y
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        pv, pt, params, epochs = core.train_seq(
            "lstm", x, h, target, tr, va, te, T=args.lookback,
            hidden=args.hidden, layers=args.layers, norm=True,
            har=not residual, seed=args.seed, epochs=args.epochs,
        )
        alpha = 0.0
        if residual:
            grid = np.linspace(0, 1.25, 126)
            alpha = float(grid[np.argmin([np.mean((y[va] - hp[va] - a * pv)**2) for a in grid])])
            pt = hp[te] + alpha * pt
        if not np.isfinite(pt).all():
            raise RuntimeError(f"Nonfinite {kind} predictions")
        predictions[kind] = pt
        rows.append({"model": kind, "alpha": alpha, "parameters": params, "epochs": epochs, **core.metric(y[te], pt)})
    rows.insert(0, {"model": "HAR", **core.metric(y[te], hp[te])})
    tag = "quick" if args.quick else "full"
    output = (args.output_dir.resolve() if args.output_dir else root / "outputs/cli") / f"{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({"Date": meta.Date.iloc[te].to_numpy(), "y": y[te], **predictions}).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(split).to_csv(output / "split_summary.csv", index=False)
    metadata = {"quick": args.quick, "research_result": not args.quick, "device": str(device),
                "torch": torch.__version__, "seed": args.seed, "epochs": args.epochs,
                "hidden": args.hidden, "layers": args.layers, "lookback": args.lookback,
                "horizon": core.H, "gap": core.SPLIT_GAP, "assets": assets,
                "input_sha256": {name: hashlib.sha256((core.DATA / name).read_bytes()).hexdigest() for name in ("prices.csv", "returns.csv")}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved local outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
