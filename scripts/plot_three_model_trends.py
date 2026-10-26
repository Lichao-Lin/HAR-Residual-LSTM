from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs' / 'artifacts_complete' / 'ensemble_predictions.csv'
OUT = ROOT / 'outputs' / 'model_trend_figures'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE, parse_dates=['Date']).sort_values('Date')
assert len(df) == 709, f'Expected 709 frozen-test observations, found {len(df)}'
models = [
    ('HAR + Residual LSTM', '本文主模型', '#2E78B7'),
    ('HAR-RV', 'HAR-RV', '#777777'),
    ('Direct LSTM', '直接 LSTM', '#E69F00'),
]

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

ymin = min(df['y'].min(), *(df[c].min() for c, _, _ in models))
ymax = max(df['y'].max(), *(df[c].max() for c, _, _ in models))
pad = 0.045 * (ymax - ymin)
ylim = (ymin - pad, ymax + pad)

def style_axis(ax, ylabel=True):
    ax.set_ylim(*ylim)
    ax.grid(axis='y', color='#E7E7E7', linewidth=0.55, zorder=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if ylabel:
        ax.set_ylabel('未来5日对数实现方差')
    ax.margins(x=0)

def draw(ax, column, label, color, panel=None):
    ax.plot(df['Date'], df['y'], color='#202020', linewidth=0.80, alpha=0.78,
            label='真实值', zorder=2)
    ax.plot(df['Date'], df[column], color=color, linewidth=1.25, alpha=0.98,
            label=label, zorder=3)
    r2 = r2_score(df['y'], df[column])
    mse = mean_squared_error(df['y'], df[column])
    prefix = f'{panel}  ' if panel else ''
    ax.set_title(f'{prefix}{label}预测走势', loc='left', fontweight='bold', fontsize=9)
    ax.text(0.985, 0.94, f'R² = {r2:.3f}\nMSE = {mse:.3f}\nn = {len(df)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color='#333333', linespacing=1.35)
    ax.legend(loc='lower left', ncol=2, handlelength=2.8)
    style_axis(ax)

def export(fig, stem):
    paths=[]
    for ext, kw in [('png', {'dpi': 300}), ('pdf', {}), ('svg', {}), ('tiff', {'dpi': 600})]:
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, bbox_inches='tight', **kw)

        paths.append(path)
    return paths

# Combined comparison: exact same geometry and axes for all panels.
fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.2), sharex=True, sharey=True)
for ax, (col, label, color), panel in zip(axes, models, ['a', 'b', 'c']):
    draw(ax, col, label, color, panel)
axes[-1].set_xlabel('测试日期')
fig.suptitle('主模型、HAR 与 LSTM 的冻结测试集预测走势', fontsize=11, fontweight='bold', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.15)
export(fig, 'combined_three_model_trends')
plt.close(fig)

# Three standalone panels with identical dimensions and limits.
stems = ['main_model_vs_truth', 'har_vs_truth', 'direct_lstm_vs_truth']
for (col, label, color), stem in zip(models, stems):
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    draw(ax, col, label, color)
    ax.set_xlabel('测试日期')
    fig.tight_layout()
    export(fig, stem)
    plt.close(fig)

# Traceable source data and metric table.
df[['Date', 'y', *[x[0] for x in models]]].to_csv(OUT / 'trend_source_data.csv', index=False, encoding='utf-8-sig')

metric_rows=[]
for col,label,_ in models:
    metric_rows.append({'模型':label,'样本量':len(df),'R2':r2_score(df.y,df[col]),'MSE':mean_squared_error(df.y,df[col]),'纵轴下限':ylim[0],'纵轴上限':ylim[1]})
pd.DataFrame(metric_rows).to_csv(OUT/'trend_metrics.csv',index=False,encoding='utf-8-sig')

print(pd.DataFrame(metric_rows).to_string(index=False))
print(f'common_ylim={ylim}')
