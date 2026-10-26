from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'outputs'/'artifacts'; FIG=ROOT/'outputs'/'figures'
d=pd.read_csv(ROOT/'data'/'returns.csv', parse_dates=['Date']).drop(columns=['VIX'])
rows=[]
for c in d.columns[1:]:
    x=d[c].to_numpy(float)
    rows.append({'资产':c,'均值':x.mean(),'标准差':x.std(ddof=1),'偏度':stats.skew(x,bias=False),
                 '超额峰度':stats.kurtosis(x,bias=False),'ADF p值':adfuller(x,autolag='AIC')[1],
                 'JB p值':stats.jarque_bera(x).pvalue,'样本量':len(x)})
pd.DataFrame(rows).to_csv(ART/'descriptive_statistics.csv',index=False,encoding='utf-8-sig')

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':8,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.1,3.5)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
def box(x,y,w,h,text,fc,ec):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.08',facecolor=fc,edgecolor=ec,linewidth=1.1)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8); return p
def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,color='#555',lw=1))
box(.3,4.25,2.1,.8,'目标市场历史波动\nRV(d), RV(w), RV(m)','#E7E7E7','#777')
box(3.0,4.25,1.5,.8,'HAR-RV\nOLS','#E7E7E7','#777'); arrow(2.4,4.65,3.0,4.65)
box(5.1,4.25,1.8,.8,'基线预测\ny_HAR','#E7E7E7','#777'); arrow(4.5,4.65,5.1,4.65)
box(.3,1.25,2.1,.8,'15个市场序列\n60日收益与绝对收益','#DDECF7','#2673B8')
box(3.0,1.25,1.5,.8,'LSTM\n32维隐藏层','#DDECF7','#2673B8'); arrow(2.4,1.65,3.0,1.65)
box(5.1,1.25,1.8,.8,'LayerNorm\n线性预测头','#DDECF7','#2673B8'); arrow(4.5,1.65,5.1,1.65)
box(7.55,1.25,1.4,.8,'残差修正\ne_hat','#DDECF7','#2673B8'); arrow(6.9,1.65,7.55,1.65)
box(7.55,4.25,1.4,.8,'最终预测\ny_HAR + αe_hat','#F8E2DF','#B33A3A'); arrow(6.9,4.65,7.55,4.65); arrow(8.25,2.05,8.25,4.25)
ax.text(5,5.75,'两阶段 HAR + Residual LSTM 框架',ha='center',fontsize=11,fontweight='bold')
ax.text(5,.35,'α 仅由验证集确定；测试集在模型与校准参数冻结后使用一次',ha='center',color='#555')
fig.tight_layout()
for ext,kw in [('png',{'dpi':300}),('pdf',{}),('svg',{}),('tiff',{'dpi':600})]: fig.savefig(FIG/f'figure2_architecture.{ext}',bbox_inches='tight',**kw)
plt.close(fig)
print('supporting outputs complete')
