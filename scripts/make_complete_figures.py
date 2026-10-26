from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'outputs'/'artifacts_complete'; FIG=ROOT/'outputs'/'figures_complete'; FIG.mkdir(parents=True, exist_ok=True)
import run_outline_complete as core
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
BLUE='#2E78B7'; ORANGE='#EAA700'; GREY='#999999'; PALE='#B7C3CF'; RED='#B64040'
def save(fig,name):
    for ext,kw in [('png',{'dpi':300}),('pdf',{}),('svg',{}),('tiff',{'dpi':600})]: fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',**kw)
    plt.close(fig)

# Correlation heatmap: descriptive dependence.
ret=pd.read_csv(core.DATA/'returns.csv').drop(columns=['Date','VIX']); corr=ret.corr(); corr.to_csv(ART/'return_correlation_matrix.csv')
fig,ax=plt.subplots(figsize=(7.1,5.8)); sns.heatmap(corr,ax=ax,cmap='vlag',center=0,vmin=-1,vmax=1,square=True,linewidths=.25,cbar_kws={'label':'Pearson相关系数','shrink':.75}); ax.set_title('跨资产日收益相关性'); fig.tight_layout(); save(fig,'figureS1_correlation_heatmap')

# Full Granger heatmap with stars.
p=pd.read_csv(ART/'granger_pvalue_matrix.csv',index_col=0); val=-np.log10(p.clip(lower=1e-50)); annot=np.empty(p.shape,dtype=object)
for i in range(p.shape[0]):
    for j in range(p.shape[1]): annot[i,j]='' if i==j else ('**' if p.iloc[i,j]<.01 else ('*' if p.iloc[i,j]<.05 else ''))
fig,ax=plt.subplots(figsize=(7.1,6.1)); sns.heatmap(val,ax=ax,cmap='Blues',vmin=0,vmax=min(12,float(np.nanmax(val.replace(np.inf,np.nan)))),square=True,linewidths=.25,annot=annot,fmt='',cbar_kws={'label':'−log10(p)','shrink':.75}); ax.set_xlabel('源资产（Granger原因）'); ax.set_ylabel('目标资产'); ax.set_title('VAR-Wald Granger因果检验（* p<0.05，** p<0.01）'); fig.tight_layout(); save(fig,'figure1_granger_heatmap')

# IRF curves.
ir=pd.read_csv(ART/'var_irf_ixic.csv'); fig,ax=plt.subplots(figsize=(7.1,3.8))
for s,z in ir.groupby('source'): ax.plot(z.horizon,z.response,lw=1.3,label=s)
ax.axhline(0,c='#555',lw=.7); ax.set_xlabel('冲击后交易日'); ax.set_ylabel('IXIC标准化收益响应'); ax.set_title('IXIC对主要跨资产一标准差正交冲击的VAR脉冲响应'); ax.legend(ncol=3); fig.tight_layout(); save(fig,'figureS2_var_irf')

# Main model comparison with seed SD.
allr=pd.read_csv(ART/'all_model_seed_results.csv'); sm=allr.groupby('Model').agg(R2=('R2','mean'),R2sd=('R2','std'),MSE=('MSE','mean'),MSEsd=('MSE','std')).reset_index().sort_values('R2')
sm[['R2sd','MSEsd']]=sm[['R2sd','MSEsd']].fillna(0); colors=[BLUE if x=='HAR + Residual LSTM' else (ORANGE if x=='iTransformer' else GREY if x=='HAR-RV' else PALE) for x in sm.Model]
fig,axs=plt.subplots(1,2,figsize=(7.1,3.8)); axs[0].barh(sm.Model,sm.R2,xerr=sm.R2sd,color=colors); axs[0].set_xlabel('样本外 R²（五种子均值±SD）'); axs[0].set_title('a  预测解释度',loc='left',fontweight='bold'); axs[1].barh(sm.Model,sm.MSE,xerr=sm.MSEsd,color=colors); axs[1].set_xlabel('对数方差MSE（五种子均值±SD）'); axs[1].set_title('b  预测误差',loc='left',fontweight='bold'); fig.tight_layout(); save(fig,'figure3_complete_model_comparison')

# Predictions and residual correction mechanism with TNX.
pred=pd.read_csv(ART/'ensemble_predictions.csv',parse_dates=['Date']); z=pred.iloc[-260:]
fig,ax=plt.subplots(figsize=(7.1,3.5)); ax.plot(z.Date,z.y,c='#222',lw=1,label='真实值'); ax.plot(z.Date,z['HAR-RV'],c=GREY,lw=1,label='HAR-RV'); ax.plot(z.Date,z['HAR + Residual LSTM'],c=BLUE,lw=1.2,label='本文方法'); ax.set_ylabel('未来5日对数实现方差'); ax.set_title('冻结测试集预测轨迹（五种子集成）'); ax.legend(ncol=3); fig.tight_layout(); save(fig,'figure4_prediction_trajectory')
mech=pd.read_csv(ART/'mechanism_timeseries.csv',parse_dates=['Date']); hi=mech.regime.eq('高波动').to_numpy();
fig,axs=plt.subplots(2,1,figsize=(7.1,4.6),sharex=True,gridspec_kw={'height_ratios':[1.25,1]}); axs[0].plot(mech.Date,mech.correction,c=BLUE,lw=.8); axs[0].axhline(0,c='#555',lw=.6); axs[0].fill_between(mech.Date,0,1,where=hi,color='#EAA700',alpha=.12,transform=axs[0].get_xaxis_transform()); axs[0].set_ylabel('残差修正量'); axs[0].set_title('a  Residual LSTM修正',loc='left',fontweight='bold')
axs[1].plot(mech.Date,mech.TNX_level,c='#6B6B6B',lw=.9); axs[1].fill_between(mech.Date,0,1,where=hi,color='#EAA700',alpha=.12,transform=axs[1].get_xaxis_transform()); axs[1].set_ylabel('TNX水平'); axs[1].set_title('b  同期10年期美债收益率',loc='left',fontweight='bold'); axs[1].set_xlabel('测试日期'); fig.tight_layout(); save(fig,'figure5_correction_tnx_mechanism')

# Exact ablation and yearly stability.
ab=pd.read_csv(ART/'outline_ablation.csv').sort_values('R2'); fig,axs=plt.subplots(1,2,figsize=(7.1,4.2)); axs[0].barh(ab.Configuration,ab.R2,color=np.where(ab.Configuration.eq('完整模型'),BLUE,PALE)); axs[0].axvline(allr.loc[allr.Model.eq('HAR-RV'),'R2'].mean(),ls='--',c=GREY,lw=1); axs[0].set_xlabel('样本外 R²'); axs[0].set_title('a  大纲指定完整消融',loc='left',fontweight='bold')
yr=pd.read_csv(ART/'yearly_robustness.csv'); axs[1].bar(yr.year,yr.Delta_R2,color=np.where(yr.Delta_R2>=0,BLUE,RED)); axs[1].axhline(0,c='#555',lw=.7); axs[1].set_xlabel('测试年份'); axs[1].set_ylabel('ΔR² vs HAR'); axs[1].set_title('b  跨年份稳健性',loc='left',fontweight='bold'); fig.tight_layout(); save(fig,'figure6_ablation_yearly')

# Nonlinear residual predictability and regime table visualization.
nl=json.loads((ART/'nonlinear_residual_test.json').read_text()); rg=pd.read_csv(ART/'regime_complete.csv'); rg['reduction']=100*(rg.HAR_MSE-rg.Proposed_MSE)/rg.HAR_MSE
fig,axs=plt.subplots(1,2,figsize=(7.1,3.2)); axs[0].bar(['零预测器','LightGBM\n跨资产残差模型'],[nl['MSE_zero'],nl['MSE_model']],color=[GREY,BLUE]); axs[0].set_ylabel('HAR残差MSE'); axs[0].set_title(f"a  非线性残差检验（R²={nl['Residual_R2']:.3f}, p={nl['p']:.3f}）",loc='left',fontweight='bold')
order=['低波动','中波动','高波动']; q=rg.set_index('regime').reindex(order); axs[1].bar(order,q.reduction,color=np.where(q.reduction>=0,BLUE,RED)); axs[1].axhline(0,c='#555',lw=.7); axs[1].set_ylabel('相对HAR的MSE下降（%）'); axs[1].set_title('b  市场状态异质性',loc='left',fontweight='bold'); fig.tight_layout(); save(fig,'figureS3_nonlinear_regime')

# Complete-sample LightGBM SHAP.
raw,assets,meta,X,Hx,y,tr,va,te=core.build(); xe,names=core.eng(X,Hx,assets,96)
gb=lgb.LGBMRegressor(n_estimators=1200,learning_rate=.015,num_leaves=15,max_depth=5,reg_lambda=2,random_state=42,verbosity=-1).fit(xe[tr],y[tr],eval_set=[(xe[va],y[va])],callbacks=[lgb.early_stopping(80,verbose=False)])
ii=np.linspace(te[0],te[-1],min(500,len(te)),dtype=int); sv=shap.TreeExplainer(gb).shap_values(xe[ii]); sf=pd.DataFrame({'feature':names,'mean_abs_shap':np.abs(sv).mean(0)}).sort_values('mean_abs_shap',ascending=False); sf['asset']=sf.feature.str.split('_').str[0]; sf.to_csv(ART/'shap_complete_features.csv',index=False); sf.groupby('asset',as_index=False).mean(numeric_only=True).sort_values('mean_abs_shap',ascending=False).to_csv(ART/'shap_complete_assets.csv',index=False)
q=sf.head(20).sort_values('mean_abs_shap'); fig,ax=plt.subplots(figsize=(7.1,4.5)); ax.barh(q.feature,q.mean_abs_shap,color=ORANGE); ax.set_xlabel('平均|TreeSHAP值|'); ax.set_title('统一母样本上的LightGBM特征重要性'); fig.tight_layout(); save(fig,'figure7_shap_importance')
print('complete figures generated')
