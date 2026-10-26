from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import torch
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from temporal_split import SPLIT_GAP, split_indices

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / "outputs"
ART = OUT / "artifacts"
for p in (OUT, ART):
    p.mkdir(parents=True, exist_ok=True)

SEED = 2026
HORIZON = 5
LOOKBACK = 60
TARGET = "IXIC"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_data():
    d = pd.read_csv(DATA / "returns.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    # VIX is intentionally excluded from every model, table, and interpretation.
    assets = [c for c in d.columns if c not in ("Date", "VIX")]
    r = d[assets].astype(float)
    # Past and future 5-day realized variance are non-overlapping at forecast origin t.
    past_rv = r[TARGET].pow(2).rolling(HORIZON).sum()
    future_rv = sum(r[TARGET].shift(-k).pow(2) for k in range(1, HORIZON + 1))
    log_past = np.log(past_rv.clip(lower=1e-12))
    frame = pd.DataFrame({
        "Date": d["Date"], "future_var": future_rv,
        "y": np.log(future_rv.clip(lower=1e-12)),
        "har_d": log_past,
        "har_w": log_past.rolling(5).mean(),
        "har_m": log_past.rolling(22).mean(),
    })
    # Sequence inputs are signed returns plus absolute returns; both are known at origin t.
    seq = np.concatenate([r.to_numpy(), np.abs(r.to_numpy())], axis=1)
    rows, seqs = [], []
    for t in range(max(LOOKBACK - 1, 26), len(d) - HORIZON):
        if frame.loc[t, ["y", "har_d", "har_w", "har_m"]].isna().any():
            continue
        rows.append(t)
        seqs.append(seq[t - LOOKBACK + 1:t + 1])
    meta = frame.loc[rows].reset_index(drop=True)
    Xseq = np.asarray(seqs, dtype=np.float32)
    Xhar = meta[["har_d", "har_w", "har_m"]].to_numpy(np.float32)
    y = meta["y"].to_numpy(np.float32)
    true_var = meta["future_var"].to_numpy(np.float64)
    return d, assets, meta, Xseq, Xhar, y, true_var


def metrics(y, pred):
    v, pv = np.exp(y), np.exp(np.clip(pred, -30, 10))
    return {
        "MSE_log": mean_squared_error(y, pred),
        "MAE_log": mean_absolute_error(y, pred),
        "R2_log": r2_score(y, pred),
        "QLIKE": float(np.mean(v / pv + np.log(pv))),
    }


def dm_test(y, pa, pb, loss="mse", lag=4):
    if loss == "mse":
        d = (y - pa) ** 2 - (y - pb) ** 2
    else:
        v = np.exp(y)
        va, vb = np.exp(np.clip(pa, -30, 10)), np.exp(np.clip(pb, -30, 10))
        d = (v / va + np.log(va)) - (v / vb + np.log(vb))
    x = np.ones((len(d), 1))
    fit = sm.OLS(d, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(fit.tvalues[0]), float(fit.pvalues[0]), float(d.mean())


def residual_shrinkage(y_val, har_val, residual_val):
    """Choose correction strength on validation only; test labels never enter."""
    grid = np.linspace(0.0, 1.25, 126)
    losses = [mean_squared_error(y_val, har_val + a * residual_val) for a in grid]
    return float(grid[int(np.argmin(losses))])


class LSTMReg(nn.Module):
    def __init__(self, in_dim, har_dim=3, hidden=32, layers=1, layernorm=True, use_har=True):
        super().__init__()
        self.use_har = use_har
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=.1 if layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden) if layernorm else nn.Identity()
        self.head = nn.Sequential(nn.Linear(hidden + (har_dim if use_har else 0), 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 1))
    def forward(self, xs, xh):
        h = self.norm(self.lstm(xs)[0][:, -1])
        if self.use_har:
            h = torch.cat([h, xh], dim=1)
        return self.head(h).squeeze(1)


def train_lstm(Xs, Xh, target, tr, va, te, *, hidden=32, layers=1, layernorm=True,
               use_har=True, lookback=LOOKBACK, seed=SEED, epochs=70):
    seed_all(seed)
    Xs = Xs[:, -lookback:].copy()
    # Per-channel scaling fitted only on training sequences.
    mu = Xs[tr].reshape(-1, Xs.shape[-1]).mean(0)
    sd = Xs[tr].reshape(-1, Xs.shape[-1]).std(0) + 1e-6
    Xs = (Xs - mu) / sd
    hsc = StandardScaler().fit(Xh[tr])
    Xh2 = hsc.transform(Xh).astype(np.float32)
    tmu, tsd = float(target[tr].mean()), float(target[tr].std() + 1e-6)
    z = ((target - tmu) / tsd).astype(np.float32)
    model = LSTMReg(Xs.shape[-1], hidden=hidden, layers=layers, layernorm=layernorm, use_har=use_har).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xs[tr]), torch.from_numpy(Xh2[tr]), torch.from_numpy(z[tr])), batch_size=128, shuffle=True)
    best, wait, best_state = math.inf, 0, None
    for _ in range(epochs):
        model.train()
        for a, b, c in loader:
            a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(a, b), c)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xs[va]).to(DEVICE), torch.from_numpy(Xh2[va]).to(DEVICE)).cpu().numpy()
        vl = float(np.mean((pv - z[va]) ** 2))
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 10: break
    model.load_state_dict(best_state)
    model.eval()
    def pred(ix):
        with torch.no_grad():
            p = model(torch.from_numpy(Xs[ix]).to(DEVICE), torch.from_numpy(Xh2[ix]).to(DEVICE)).cpu().numpy()
        return p * tsd + tmu
    return model, pred(te), pred(va), {"mu": mu, "sd": sd, "hsc": hsc, "tmu": tmu, "tsd": tsd, "Xs": Xs, "Xh": Xh2}


def engineered_features(Xseq, Xhar, assets):
    # Economically interpretable summaries over the sequence for tree baselines and SHAP.
    ret = Xseq[:, :, :len(assets)]
    cols, arr = [], []
    for j, a in enumerate(assets):
        for w in (1, 5, 22, 60):
            arr.append(ret[:, -w:, j].mean(1)); cols.append(f"{a}_ret{w}")
            arr.append(np.sqrt(np.mean(ret[:, -w:, j] ** 2, axis=1))); cols.append(f"{a}_vol{w}")
    for j, c in enumerate(("HAR_d", "HAR_w", "HAR_m")):
        arr.append(Xhar[:, j]); cols.append(c)
    return np.column_stack(arr), cols


def main():
    started = time.time()
    d, assets, meta, Xseq, Xhar, y, true_var = make_data()
    tr, va, te = split_indices(len(y))
    pd.DataFrame({"split": ["train", "validation", "test"], "n": [len(tr), len(va), len(te)],
                  "start": [meta.Date.iloc[x[0]] for x in (tr, va, te)], "end": [meta.Date.iloc[x[-1]] for x in (tr, va, te)]}).to_csv(ART / "split_summary.csv", index=False)

    preds, rows = {}, []
    har = LinearRegression().fit(Xhar[tr], y[tr])
    preds["HAR-RV"] = har.predict(Xhar[te])
    Xeng, eng_cols = engineered_features(Xseq, Xhar, assets)
    ridge = Ridge(alpha=10).fit(Xeng[tr], y[tr]); preds["Ridge"] = ridge.predict(Xeng[te])
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=.6, n_jobs=-1, random_state=SEED).fit(Xeng[tr], y[tr])
    preds["Random Forest"] = rf.predict(Xeng[te])
    gbm = lgb.LGBMRegressor(n_estimators=1200, learning_rate=.015, num_leaves=15, max_depth=5,
                            subsample=.8, colsample_bytree=.8, reg_lambda=2, random_state=SEED, verbosity=-1)
    gbm.fit(Xeng[tr], y[tr], eval_set=[(Xeng[va], y[va])], callbacks=[lgb.early_stopping(80, verbose=False)])
    preds["LightGBM"] = gbm.predict(Xeng[te])

    # Direct LSTM predicts y. Residual LSTM predicts the out-of-sample HAR residual using fixed train HAR.
    _, pdir, _, _ = train_lstm(Xseq, Xhar, y, tr, va, te, seed=SEED)
    preds["Direct LSTM"] = pdir
    har_all = har.predict(Xhar)
    resid = y - har_all
    best_model, pres, vres, pack = train_lstm(Xseq, Xhar, resid, tr, va, te, seed=SEED)
    alpha = residual_shrinkage(y[va], har_all[va], vres)
    preds["HAR + Residual LSTM"] = har_all[te] + alpha * pres

    for name, p in preds.items():
        z = metrics(y[te], p); z["Model"] = name; z["Delta_R2_vs_HAR"] = z["R2_log"] - metrics(y[te], preds["HAR-RV"])["R2_log"]
        rows.append(z)
    perf = pd.DataFrame(rows).set_index("Model").sort_values("R2_log", ascending=False)
    perf.to_csv(ART / "main_performance.csv")

    # Ablations and capacity/window sensitivity.
    abl = []
    configs = [
        ("完整模型", dict()), ("无LayerNorm", dict(layernorm=False)),
        ("仅1层LSTM", dict(layers=1)), ("2层LSTM", dict(layers=2)),
        ("hidden=16", dict(hidden=16)), ("hidden=64", dict(hidden=64)),
        ("lookback=20", dict(lookback=20)), ("lookback=40", dict(lookback=40)),
        ("仅跨资产序列(无HAR输入)", dict(use_har=False)),
    ]
    for name, kw in configs:
        if name == "完整模型": p = preds["HAR + Residual LSTM"]
        else:
            _, rr, vr, _ = train_lstm(Xseq, Xhar, resid, tr, va, te, seed=SEED, **kw)
            aa = residual_shrinkage(y[va], har_all[va], vr)
            p = har_all[te] + aa * rr
        z = metrics(y[te], p); z.update({"Configuration": name, "Delta_R2_vs_HAR": z["R2_log"] - metrics(y[te], preds["HAR-RV"])["R2_log"]})
        abl.append(z)
    pd.DataFrame(abl).set_index("Configuration").to_csv(ART / "ablation.csv")

    # Seed robustness for the proposed model.
    seed_rows = []
    for s in (2024, 2025, 2026, 2027, 2028):
        _, rr, vr, _ = train_lstm(Xseq, Xhar, resid, tr, va, te, seed=s)
        aa = residual_shrinkage(y[va], har_all[va], vr)
        z = metrics(y[te], har_all[te] + aa * rr); z["seed"] = s; z["alpha"] = aa; seed_rows.append(z)
    pd.DataFrame(seed_rows).to_csv(ART / "seed_robustness.csv", index=False)

    # DM tests: negative statistic means first-listed model has lower loss.
    dm = []
    for other in ("HAR-RV", "LightGBM", "Direct LSTM"):
        for loss in ("mse", "qlike"):
            t, p, md = dm_test(y[te], preds["HAR + Residual LSTM"], preds[other], loss)
            dm.append({"Comparison": f"Proposed vs {other}", "Loss": loss.upper(), "DM_stat": t, "p_value": p, "mean_loss_diff": md})
    pd.DataFrame(dm).to_csv(ART / "dm_tests.csv", index=False)

    # Permutation importance on residual LSTM, grouped by asset channels.
    base = metrics(y[te], preds["HAR + Residual LSTM"])["MSE_log"]
    imp = []
    rng = np.random.default_rng(SEED)
    for j, a in enumerate(assets):
        vals = []
        for _ in range(8):
            xp = Xseq.copy(); order = rng.permutation(te)
            xp[te, :, j] = xp[order, :, j]; xp[te, :, j + len(assets)] = xp[order, :, j + len(assets)]
            # Reuse model and train scalers.
            xs = (xp[:, -LOOKBACK:] - pack["mu"]) / pack["sd"]
            with torch.no_grad():
                rr = best_model(torch.from_numpy(xs[te].astype(np.float32)).to(DEVICE), torch.from_numpy(pack["Xh"][te]).to(DEVICE)).cpu().numpy()
            pp = har_all[te] + alpha * (rr * pack["tsd"] + pack["tmu"])
            vals.append(mean_squared_error(y[te], pp) - base)
        imp.append({"Asset": a, "Permutation_MSE_increase": np.mean(vals), "SD": np.std(vals)})
    impdf = pd.DataFrame(imp).sort_values("Permutation_MSE_increase", ascending=False)
    impdf.to_csv(ART / "permutation_importance.csv", index=False)

    # TreeSHAP gives feature-attribution evidence independent of the LSTM permutation analysis.
    sample_ix = np.linspace(te[0], te[-1], min(500, len(te)), dtype=int)
    explainer = shap.TreeExplainer(gbm)
    sv = explainer.shap_values(Xeng[sample_ix])
    shap_feature = pd.DataFrame({"feature": eng_cols, "mean_abs_shap": np.abs(sv).mean(0)})
    shap_feature["asset"] = shap_feature.feature.str.split("_").str[0]
    shap_feature.groupby("asset", as_index=False).mean(numeric_only=True).sort_values("mean_abs_shap", ascending=False).to_csv(ART / "shap_asset_importance.csv", index=False)
    shap_feature.sort_values("mean_abs_shap", ascending=False).to_csv(ART / "shap_feature_importance.csv", index=False)

    # Regime analysis based on target realized variance quartiles, predetermined at each origin.
    test_meta = meta.iloc[te].copy(); test_meta["past_rv"] = np.exp(test_meta["har_d"])
    q25, q75 = np.quantile(test_meta.past_rv, [.25, .75])
    test_meta["regime"] = np.where(test_meta.past_rv >= q75, "高波动", np.where(test_meta.past_rv <= q25, "低波动", "中波动"))
    test_meta["har_error"] = (y[te] - preds["HAR-RV"]) ** 2
    test_meta["proposed_error"] = (y[te] - preds["HAR + Residual LSTM"]) ** 2
    test_meta["correction"] = preds["HAR + Residual LSTM"] - preds["HAR-RV"]
    regime = test_meta.groupby("regime").agg(n=("Date", "size"), HAR_MSE=("har_error", "mean"), Proposed_MSE=("proposed_error", "mean"), Mean_abs_correction=("correction", lambda x: np.mean(np.abs(x))))
    regime["MSE_reduction_pct"] = 100 * (regime.HAR_MSE - regime.Proposed_MSE) / regime.HAR_MSE
    regime.to_csv(ART / "regime_analysis.csv")

    predout = pd.DataFrame({"Date": meta.Date.iloc[te].to_numpy(), "y_true_logvar": y[te], **{k: v for k, v in preds.items()}})
    predout.to_csv(ART / "test_predictions.csv", index=False)

    # Lead-lag incremental F-tests with HAR controls and 1-5 lagged absolute cross-asset returns.
    pvals = []
    allret = d.set_index("Date")[assets]
    tmp = meta.set_index("Date")
    for a in [x for x in assets if x != TARGET]:
        zz = pd.DataFrame({"y": tmp.y, "d": tmp.har_d, "w": tmp.har_w, "m": tmp.har_m})
        ar = allret[a].abs()
        for lag in range(0, 5): zz[f"x{lag+1}"] = ar.shift(lag).reindex(zz.index)
        zz = zz.dropna(); cut = int(len(zz) * .85); zt = zz.iloc[:cut]
        restricted = sm.OLS(zt.y, sm.add_constant(zt[["d", "w", "m"]])).fit()
        unrestricted = sm.OLS(zt.y, sm.add_constant(zt.drop(columns="y"))).fit()
        f, p, _ = unrestricted.compare_f_test(restricted)
        pvals.append({"Asset": a, "F_stat": f, "p_value": p, "neglog10p": -np.log10(max(p, 1e-300))})
    pd.DataFrame(pvals).sort_values("p_value").to_csv(ART / "incremental_f_tests.csv", index=False)

    env = {"python": os.sys.version, "torch": torch.__version__, "device": str(DEVICE), "cuda": torch.cuda.is_available(),
           "n_observations": len(y), "assets": assets, "horizon": HORIZON, "lookback": LOOKBACK,
           "split_gap_observations": SPLIT_GAP,
           "residual_shrinkage_alpha": alpha, "runtime_seconds": time.time() - started}
    (ART / "run_metadata.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(perf.to_string())
    print("\nDM\n", pd.DataFrame(dm).to_string(index=False))
    print("\nRegime\n", regime.to_string())
    print(json.dumps(env, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
