from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'yearly_feature_importance'
OUT.mkdir(parents=True,exist_ok=True)
import run_outline_complete as core

raw,assets,meta,X,Hx,y,tr,va,te=core.build()
xe,names=core.eng(X,Hx,assets,96)

# A fixed retrospective explanation model. This is not a walk-forward backtest.
model=lgb.LGBMRegressor(n_estimators=900,learning_rate=.02,num_leaves=15,max_depth=5,
                        reg_lambda=3,subsample=.9,colsample_bytree=.9,
                        random_state=42,verbosity=-1,n_jobs=-1)
model.fit(xe,y)
sv=np.asarray(shap.TreeExplainer(model).shap_values(xe))
years=meta.Date.dt.year.to_numpy()
unique_years=np.arange(years.min(),years.max()+1)

feature_year=pd.DataFrame(index=names,columns=unique_years,dtype=float)
for yr in unique_years:
    mask=years==yr
    feature_year[yr]=np.abs(sv[mask]).mean(axis=0) if mask.any() else np.nan
feature_year.index.name='feature'
feature_year.to_csv(OUT/'all_feature_importance_by_year.csv',encoding='utf-8-sig')

def group_name(feature):
    if feature.startswith('HAR_'): return 'HAR状态'
    return feature.split('_',1)[0]
groups=pd.Series([group_name(x) for x in names],index=names,name='asset_group')
asset_year=feature_year.assign(asset_group=groups).groupby('asset_group').sum(numeric_only=True)
asset_year.to_csv(OUT/'asset_importance_by_year.csv',encoding='utf-8-sig')

summary=pd.DataFrame({'feature':names,'overall_mean_abs_shap':np.abs(sv).mean(axis=0),'group':[group_name(x) for x in names]}).sort_values('overall_mean_abs_shap',ascending=False)
summary.to_csv(OUT/'feature_importance_overall.csv',index=False,encoding='utf-8-sig')

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':8,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#2E78B7'; palette=['#2E78B7','#E69F00','#6F8F72','#8B6DAF','#B65F5F','#4E9A9A','#A17C5B','#7A869A']

def export(fig,stem):
    for ext,kw in [('png',{'dpi':300}),('pdf',{}),('svg',{}),('tiff',{'dpi':600})]:
        p=OUT/f'{stem}.{ext}'; fig.savefig(p,bbox_inches='tight',**kw)

# Main-text heatmap: top 30 features, every available year.
top30=summary.head(30).feature.tolist(); hm=feature_year.loc[top30]
fig,ax=plt.subplots(figsize=(7.1,7.3)); sns.heatmap(hm,ax=ax,cmap='Blues',linewidths=.25,cbar_kws={'label':'年度平均 |TreeSHAP|','shrink':.72})
ax.set_xlabel('年份'); ax.set_ylabel('特征'); ax.set_title('主要特征的重要性随年份变化（固定解释模型）',fontweight='bold'); fig.tight_layout(); export(fig,'top30_feature_importance_by_year'); plt.close(fig)

# Asset-group heatmap: summed absolute SHAP across all engineered features of each asset.
order=asset_year.mean(1).sort_values(ascending=False).index; ah=asset_year.loc[order]
fig,ax=plt.subplots(figsize=(7.1,4.8)); sns.heatmap(ah,ax=ax,cmap='YlGnBu',linewidths=.3,cbar_kws={'label':'资产组年度 Σ|TreeSHAP|','shrink':.75})
ax.set_xlabel('年份'); ax.set_ylabel('资产或状态组'); ax.set_title('资产级重要性随年份变化',fontweight='bold'); fig.tight_layout(); export(fig,'asset_importance_heatmap_by_year'); plt.close(fig)

# Top-eight asset trajectories, normalized within each year to show composition changes.
shares=asset_year.div(asset_year.sum(axis=0),axis=1); top8=asset_year.mean(1).nlargest(8).index
fig,ax=plt.subplots(figsize=(7.1,3.9))
for color,g in zip(palette,top8): ax.plot(unique_years,100*shares.loc[g],marker='o',markersize=2.6,lw=1.15,label=g,color=color)
ax.set_xticks(unique_years[::2]); ax.set_xlabel('年份'); ax.set_ylabel('年度总重要性占比（%）'); ax.set_title('主要资产的重要性构成变化',fontweight='bold'); ax.grid(axis='y',color='#E5E5E5',lw=.55); ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.18)); fig.tight_layout(); export(fig,'top8_asset_importance_trends'); plt.close(fig)

# Complete all-feature heatmap for appendix. Large height is intentional: every row remains readable.
all_order=summary.feature.tolist(); full=feature_year.loc[all_order]
fig,ax=plt.subplots(figsize=(8.3,27)); sns.heatmap(full,ax=ax,cmap='Blues',linewidths=.08,cbar_kws={'label':'年度平均 |TreeSHAP|','shrink':.22,'pad':.015})
ax.set_xlabel('年份'); ax.set_ylabel('全部工程特征'); ax.set_title('全部特征的重要性随年份变化（附录全图）',fontweight='bold',pad=10); ax.tick_params(axis='y',labelsize=5.5); fig.tight_layout(); export(fig,'all_features_importance_by_year_full'); plt.close(fig)

note='''# 年度特征重要性说明\n\n- 图中使用固定的全样本 LightGBM 解释模型，对每个观测计算 TreeSHAP，再按年份汇总平均绝对 SHAP。\n- 这是回顾性特征使用模式分析，不是年度回测，也不用于证明跨年份样本外稳健性。\n- 数据覆盖模型可构造完整 96 日输入后的 2006—2024 年；VIX 未进入模型。\n- `all_features_importance_by_year_full` 展示全部工程特征；主文建议使用 top30、资产级热力图或 top8 曲线。\n- 资产级重要性为该资产全部收益与波动摘要特征的绝对 SHAP 之和；年度占比按每年全部资产组归一化。\n'''
(OUT/'年度特征重要性说明.md').write_text(note,encoding='utf-8')

print('years',unique_years[0],unique_years[-1],'n_features',len(names),'n_obs',len(y))
print(summary.head(15).to_string(index=False))
