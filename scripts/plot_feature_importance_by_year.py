from pathlib import Path

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "15_features_yearly_importance"
OUT.mkdir(parents=True, exist_ok=True)

import run_outline_complete as core


raw, assets, meta, X, Hx, y, tr, va, te = core.build()
xe, names = core.eng(X, Hx, assets, 96)

# A single fixed, full-sample model is used only for retrospective explanation.
model = lgb.LGBMRegressor(
    n_estimators=900,
    learning_rate=0.02,
    num_leaves=15,
    max_depth=5,
    reg_lambda=3,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)
model.fit(xe, y)
shap_values = np.asarray(shap.TreeExplainer(model).shap_values(xe))

years = meta.Date.dt.year.to_numpy()
unique_years = np.arange(years.min(), years.max() + 1)
engineered_year = pd.DataFrame(index=names, columns=unique_years, dtype=float)
for year in unique_years:
    mask = years == year
    engineered_year[year] = np.abs(shap_values[mask]).mean(axis=0)

# Aggregate lagged/rolling derivatives back to the 15 original market features.
# HAR state variables are intentionally excluded because they are not original inputs.
asset_year = pd.DataFrame(index=assets, columns=unique_years, dtype=float)
for asset in assets:
    rows = [name for name in names if name.startswith(asset + "_")]
    asset_year.loc[asset] = engineered_year.loc[rows].sum(axis=0)
asset_year.index.name = "feature"

# Annual shares make cross-year composition comparable despite changes in SHAP scale.
asset_share = asset_year.div(asset_year.sum(axis=0), axis=1) * 100
asset_year.to_csv(OUT / "15_features_mean_abs_shap_by_year.csv", encoding="utf-8-sig")
asset_share.to_csv(OUT / "15_features_importance_share_by_year.csv", encoding="utf-8-sig")

overall = pd.DataFrame(
    {
        "feature": assets,
        "mean_abs_shap": asset_year.mean(axis=1).reindex(assets).to_numpy(),
        "mean_share_percent": asset_share.mean(axis=1).reindex(assets).to_numpy(),
    }
).sort_values("mean_abs_shap", ascending=False)
overall.to_csv(OUT / "15_features_importance_overall.csv", index=False, encoding="utf-8-sig")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def export(fig, stem):
    paths = [
        (OUT / f"{stem}.svg", {}),
        (OUT / f"{stem}.pdf", {}),
        (OUT / f"{stem}.png", {"dpi": 300}),
        (OUT / f"{stem}.tiff", {"dpi": 600}),
    ]
    for path, kwargs in paths:
        fig.savefig(path, bbox_inches="tight", **kwargs)


# Main figure: exactly 15 original features x all available years.
order = overall["feature"].tolist()
heat = asset_share.loc[order]
fig, ax = plt.subplots(figsize=(7.1, 5.2))
sns.heatmap(
    heat,
    ax=ax,
    cmap="YlGnBu",
    linewidths=0.35,
    linecolor="white",
    cbar_kws={"label": "年度特征重要性占比（%）", "shrink": 0.78},
)
ax.set_xlabel("年份")
ax.set_ylabel("原始特征（不含 VIX）")
ax.set_title("15 个原始特征的重要性随年份变化", fontweight="bold", pad=10)
fig.tight_layout()
export(fig, "15_features_importance_heatmap_by_year")
plt.close(fig)

# Companion figure: one panel per feature, avoiding an unreadable 15-line spaghetti plot.
fig, axes = plt.subplots(5, 3, figsize=(7.1, 8.8), sharex=True)
trend_color = "#2E78B7"
for ax, feature in zip(axes.flat, order):
    values = asset_share.loc[feature]
    ax.plot(unique_years, values, color=trend_color, lw=1.25, marker="o", markersize=2.0)
    ax.fill_between(unique_years, 0, values, color=trend_color, alpha=0.10)
    ax.set_title(feature, loc="left", fontweight="bold", fontsize=8.5)
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
    ax.tick_params(labelsize=6.5)
for ax in axes[-1]:
    ax.set_xticks(unique_years[::3])
    ax.set_xlabel("年份")
fig.supylabel("年度特征重要性占比（%）", x=0.015, fontsize=8)
fig.suptitle("各原始特征重要性的年度变化", y=0.995, fontweight="bold", fontsize=10)
fig.tight_layout(rect=(0.025, 0, 1, 0.98))
export(fig, "15_features_importance_trends_by_year")
plt.close(fig)

note = """# 年度特征重要性说明

- 图表严格展示 15 个原始市场特征：IXIC、ES、NQ、IRX、FVX、TNX、DX、JPY、CNY、CL、NG、GC、SI、HG、ZC；VIX 未进入模型或图表。
- 模型内部使用各原始特征的滞后与滚动派生量；图表将同一原始特征的全部派生量的绝对 TreeSHAP 值求和，重新汇总为 15 项。
- 为便于比较不同年份的重要性构成，主图使用每年归一化后的重要性占比；CSV 同时提供未归一化的年度平均绝对 SHAP。
- 这是固定全样本 LightGBM 模型的回顾性解释结果，不是回测，也不应表述为跨年份样本外稳健性检验。
- 可用年份为 2006—2024 年（此前样本用于形成 96 日输入窗口）。
"""
(OUT / "年度特征重要性说明.md").write_text(note, encoding="utf-8")


print("features", len(assets), assets)
print("years", unique_years[0], unique_years[-1], "observations", len(y))
print(overall.to_string(index=False))
