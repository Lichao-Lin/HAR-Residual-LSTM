"""Batched CUDA/CPU residual-LSTM training with optional AMP and checkpoints.

Independent handoff file: requires only the existing HAR-Residual-LSTM checkout.
Default device is CUDA and fails explicitly when CUDA is unavailable. CPU is
supported for tests with --device cpu; existing requirements remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def make_loader(xs, hs, ii, batch_size, target=None, shuffle=False, pin=False, seed=42):
    tensors = [torch.from_numpy(xs[ii]), torch.from_numpy(hs[ii])]
    if target is not None:
        tensors.append(torch.from_numpy(target[ii]))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def predict(model, loader, device, mean, scale):
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            # Always evaluate in FP32, including when training uses AMP.
            values = model(batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True))
            parts.append(values.float().cpu().numpy())
    return np.concatenate(parts) * scale + mean


def predict_checkpoint(checkpoint, sequences, har_features, core, device="cpu", batch_size=128):
    """Inference from this script's checkpoint and already engineered raw arrays.

    sequences: (n, history>=lookback, features in checkpoint['assets'] order).
    har_features: (n, 3), columns har_d / har_w / har_m, prior to scaling.
    """
    device = torch.device(device)
    if sequences.shape[1] < checkpoint["lookback"] or sequences.shape[2] != len(checkpoint["assets"]):
        raise ValueError("Sequence history or asset count does not match checkpoint")
    if har_features.shape != (len(sequences), 3):
        raise ValueError("Expected three aligned HAR features per sequence")
    xs = ((sequences[:, -checkpoint["lookback"]:] - checkpoint["x_mean"].numpy()) / checkpoint["x_scale"].numpy()).astype(np.float32)
    hs = ((har_features - checkpoint["h_mean"].numpy()) / checkpoint["h_scale"].numpy()).astype(np.float32)
    model = core.SeqLSTM(len(checkpoint["assets"]), hidden=checkpoint["hidden"], layers=checkpoint["layers"], norm=True, har=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = make_loader(xs, hs, np.arange(len(xs)), batch_size, pin=device.type == "cuda")
    correction = predict(model, loader, device, checkpoint["target_mean"], checkpoint["target_scale"])
    hp = har_features @ checkpoint["har_coef"].numpy() + checkpoint["har_intercept"]
    return hp + checkpoint["alpha"] * correction


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="CUDA FP16 training only; off by default")
    p.add_argument("--epochs", type=positive, default=90)
    p.add_argument("--batch-size", type=positive, default=32)
    p.add_argument("--eval-batch-size", type=positive, default=128)
    p.add_argument("--hidden", type=positive, default=128)
    p.add_argument("--layers", type=positive, default=2)
    p.add_argument("--lookback", type=positive, default=96)
    p.add_argument("--patience", type=positive, default=10)
    p.add_argument("--threads", type=positive, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quick-samples", type=positive, default=256)
    args = p.parse_args(argv)
    if args.quick and args.quick_samples < 64:
        p.error("--quick-samples must be at least 64 for nontrivial holdouts")
    if not 1 <= args.lookback <= 252:
        p.error("--lookback must be in [1, 252]")
    if not 0 <= args.seed < 2**32 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error("Invalid seed or learning rate")
    wants_cuda = args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
    if wants_cuda and (not torch.cuda.is_available() or not 0 <= args.gpu_index < torch.cuda.device_count()):
        p.error("CUDA device unavailable. Install CUDA-enabled PyTorch or use --device cpu for testing")
    if args.amp and not wants_cuda:
        p.error("--amp requires CUDA; remove --amp for CPU training")
    device = torch.device(f"cuda:{args.gpu_index}" if wants_cuda else "cpu")
    candidates = [args.repo] if args.repo else [Path(__file__).resolve().parent.parent, Path.cwd()]
    root = next((r.resolve() for r in candidates if r and (r / "scripts/run_outline_complete.py").is_file()), None)
    if root is None:
        p.error("Supply --repo PATH or run from the HAR-Residual-LSTM checkout")
    sys.path.insert(0, str(root / "scripts"))
    core = importlib.import_module("run_outline_complete")
    if Path(core.__file__).resolve() != root / "scripts/run_outline_complete.py":
        p.error("A different checkout is already imported")
    if args.data_dir:
        core.DATA = args.data_dir.resolve()
    _, assets, meta, x, h, y, _, _, _ = core.build()
    if args.quick:
        n = min(len(y), args.quick_samples)
        meta, x, h, y = meta.iloc[-n:].reset_index(drop=True), x[-n:], h[-n:], y[-n:]
        args.epochs, args.hidden, args.layers = min(args.epochs, 2), min(args.hidden, 16), 1
    tr, va, te = core.split_indices(len(y))
    if any(b[0] - a[-1] - 1 < core.H - 1 for a, b in ((tr, va), (va, te))):
        p.error("Split gap too short for the forecast horizon")
    if not all(np.isfinite(a).all() for a in (x, h, y)):
        p.error("Nonfinite model input or target")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if wants_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    har = LinearRegression().fit(h[tr], y[tr])
    hp = har.predict(h)
    target = y - hp
    xs = x[:, -args.lookback:].copy()
    mu = xs[tr].reshape(-1, xs.shape[-1]).mean(0)
    sd = xs[tr].reshape(-1, xs.shape[-1]).std(0) + 1e-6
    xs = ((xs - mu) / sd).astype(np.float32)
    hsc = StandardScaler().fit(h[tr])
    hs = hsc.transform(h).astype(np.float32)
    ym, ys = float(target[tr].mean()), float(target[tr].std() + 1e-6)
    z = ((target - ym) / ys).astype(np.float32)
    train = make_loader(xs, hs, tr, args.batch_size, target=z, shuffle=True, pin=wants_cuda, seed=args.seed)
    validation = make_loader(xs, hs, va, args.eval_batch_size, pin=wants_cuda)
    test = make_loader(xs, hs, te, args.eval_batch_size, pin=wants_cuda)
    model = core.SeqLSTM(len(assets), hidden=args.hidden, layers=args.layers, norm=True, har=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    best, wait, best_state, history = float("inf"), 0, None, []
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for a, b, c in train:
            a, b, c = (t.to(device, non_blocking=wants_cuda) for t in (a, b, c))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                loss = ((model(a, b) - c)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; try FP32 or check input scaling")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(c)
            count += len(c)
        scheduler.step()
        pv = predict(model, validation, device, ym, ys)
        val_loss = float(np.mean(((pv - target[va]) / ys)**2))
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total / count, "validation_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best - 1e-5:
            best, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best_state)
    pv = predict(model, validation, device, ym, ys)
    grid = np.linspace(0, 1.25, 126)
    alpha = float(grid[np.argmin([np.mean((y[va] - hp[va] - a * pv)**2) for a in grid])])
    prediction = hp[te] + alpha * predict(model, test, device, ym, ys)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite test predictions")
    tag = "quick" if args.quick else "full"
    output = (args.output_dir.resolve() if args.output_dir else root / "outputs/gpu_training") / f"{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = {"format_version": 1, "state_dict": best_state, "assets": assets,
                  "hidden": args.hidden, "layers": args.layers, "lookback": args.lookback,
                  "horizon": core.H, "gap": core.SPLIT_GAP, "alpha": alpha,
                  "x_mean": torch.from_numpy(mu), "x_scale": torch.from_numpy(sd),
                  "h_mean": torch.from_numpy(hsc.mean_), "h_scale": torch.from_numpy(hsc.scale_),
                  "target_mean": ym, "target_scale": ys,
                  "har_coef": torch.from_numpy(har.coef_), "har_intercept": float(har.intercept_),
                  "quick": args.quick}
    checkpoint_path = output / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    repeated = predict_checkpoint(restored, x[te], h[te], core, device, args.eval_batch_size)
    if not np.allclose(prediction, repeated, atol=1e-5, rtol=1e-4):
        raise RuntimeError("Checkpoint round-trip predictions do not agree")
    pd.DataFrame(history).to_csv(output / "training.csv", index=False)
    pd.DataFrame({"Date": meta.Date.iloc[te].to_numpy(), "y": y[te], "HAR": hp[te], "prediction": prediction}).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame([{"split": name, "n": len(ii), "start": str(meta.Date.iloc[ii[0]].date()), "end": str(meta.Date.iloc[ii[-1]].date())}
                  for name, ii in (("train", tr), ("validation", va), ("test", te))]).to_csv(output / "split_summary.csv", index=False)
    metadata = {"quick": args.quick, "research_result": not args.quick, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if wants_cuda else None,
                "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda, "amp": args.amp,
                "seed": args.seed, "epochs_run": len(history), "batch_size": args.batch_size,
                "learning_rate": args.lr, "hidden": args.hidden, "layers": args.layers, "lookback": args.lookback,
                "alpha": alpha, "horizon": core.H, "gap": core.SPLIT_GAP,
                "metrics": core.metric(y[te], prediction), "HAR_metrics": core.metric(y[te], hp[te]),
                "checkpoint_roundtrip_passed": True,
                "input_sha256": {name: hashlib.sha256((core.DATA / name).read_bytes()).hexdigest() for name in ("returns.csv", "prices.csv")}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved local outputs and checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
