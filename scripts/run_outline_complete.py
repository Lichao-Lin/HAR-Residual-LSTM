from __future__ import annotations
import json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]; DATA = ROOT / 'data'
OUT=ROOT/'outputs'; ART=OUT/'artifacts_complete'; ART.mkdir(parents=True, exist_ok=True); FIG=OUT/'figures_complete'; FIG.mkdir(parents=True, exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(min(8,os.cpu_count() or 1))
ASSET_EXCLUDE={'Date','VIX'}; H=5; MAX_T=252; DEFAULT_T=96

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def build():
    raw=pd.read_csv(DATA/'returns.csv',parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    assets=[c for c in raw.columns if c not in ASSET_EXCLUDE]; r=raw[assets].astype(float)
    past=r.IXIC.pow(2).rolling(H).sum(); future=sum(r.IXIC.shift(-k).pow(2) for k in range(1,H+1))
    lp=np.log(past.clip(lower=1e-12))
    meta=pd.DataFrame({'Date':raw.Date,'y':np.log(future.clip(lower=1e-12)),'future_var':future,
                       'har_d':lp,'har_w':lp.rolling(5).mean(),'har_m':lp.rolling(22).mean(),'TNX_level':pd.read_csv(DATA/'prices.csv').TNX})
    ix=[]; seq=[]
    for t in range(MAX_T-1,len(raw)-H):
        if not meta.loc[t,['y','har_d','har_w','har_m']].isna().any():
            ix.append(t); seq.append(r.iloc[t-MAX_T+1:t+1].to_numpy(np.float32))
    meta=meta.loc[ix].reset_index(drop=True); X=np.asarray(seq); Hx=meta[['har_d','har_w','har_m']].to_numpy(np.float32); y=meta.y.to_numpy(np.float32)
    n=len(y); a=int(.70*n); b=int(.85*n); tr=np.arange(a); va=np.arange(a,b); te=np.arange(b,n)
    return raw,assets,meta,X,Hx,y,tr,va,te

def metric(y,p):
    v=np.exp(y); pv=np.exp(np.clip(p,-30,10))
    return {'MSE':mean_squared_error(y,p),'MAE':mean_absolute_error(y,p),'R2':r2_score(y,p),'QLIKE':float(np.mean(v/pv+np.log(pv)))}

def dm(y,a,b,kind='mse'):
    if kind=='mse': d=(y-a)**2-(y-b)**2
    else:
        v=np.exp(y); av=np.exp(np.clip(a,-30,10)); bv=np.exp(np.clip(b,-30,10)); d=v/av+np.log(av)-(v/bv+np.log(bv))
    fit=sm.OLS(d,np.ones((len(d),1))).fit(cov_type='HAC',cov_kwds={'maxlags':4})
    return float(fit.tvalues[0]),float(fit.pvalues[0]),float(d.mean())

class SeqLSTM(nn.Module):
    def __init__(self,n,hidden=128,layers=2,norm=True,har=True):
        super().__init__(); self.har=har
        self.rnn=nn.LSTM(n,hidden,layers,batch_first=True,dropout=.1 if layers>1 else 0)
        self.norm=nn.LayerNorm(hidden) if norm else nn.Identity(); self.out=nn.Linear(hidden+(3 if har else 0),1)
    def forward(self,x,h):
        z=self.norm(self.rnn(x)[0][:,-1]); z=torch.cat([z,h],1) if self.har else z; return self.out(z).squeeze(1)

class ITransformer(nn.Module):
    def __init__(self,n,T,d=64,layers=2,heads=4):
        super().__init__(); self.embed=nn.Linear(T,d)
        block=nn.TransformerEncoderLayer(d,heads,dim_feedforward=128,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(block,layers); self.head=nn.Sequential(nn.LayerNorm(d+3),nn.Linear(d+3,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,h):
        z=self.enc(self.embed(x.transpose(1,2))).mean(1); return self.head(torch.cat([z,h],1)).squeeze(1)

def train_seq(kind,X,Hx,target,tr,va,te,T=96,hidden=128,layers=2,norm=True,har=True,seed=42,epochs=90):
    seed_all(seed); xs=X[:,-T:].copy(); mu=xs[tr].reshape(-1,xs.shape[-1]).mean(0); sd=xs[tr].reshape(-1,xs.shape[-1]).std(0)+1e-6; xs=(xs-mu)/sd
    hs=StandardScaler().fit(Hx[tr]); hh=hs.transform(Hx).astype(np.float32); ym=float(target[tr].mean()); ys=float(target[tr].std()+1e-6); z=((target-ym)/ys).astype(np.float32)
    model=ITransformer(xs.shape[-1],T) if kind=='itransformer' else SeqLSTM(xs.shape[-1],hidden,layers,norm,har)
    model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    dl=DataLoader(TensorDataset(torch.from_numpy(xs[tr]),torch.from_numpy(hh[tr]),torch.from_numpy(z[tr])),batch_size=32,shuffle=True)
    best=1e99; state=None; wait=0
    for ep in range(epochs):
        model.train()
        for a,b,c in dl:
            a,b,c=a.to(DEVICE),b.to(DEVICE),c.to(DEVICE); opt.zero_grad(set_to_none=True); loss=((model(a,b)-c)**2).mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        sched.step(); model.eval()
        with torch.no_grad(): pv=model(torch.from_numpy(xs[va]).to(DEVICE),torch.from_numpy(hh[va]).to(DEVICE)).cpu().numpy()
        vl=float(np.mean((pv-z[va])**2))
        if vl<best-1e-5: best=vl; wait=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=10: break
    model.load_state_dict(state); model.eval()
    def pred(ii):
        with torch.no_grad(): q=model(torch.from_numpy(xs[ii]).to(DEVICE),torch.from_numpy(hh[ii]).to(DEVICE)).cpu().numpy()
        return q*ys+ym
    return pred(va),pred(te),sum(p.numel() for p in model.parameters()),ep+1

def eng(X,Hx,assets,T=96):
    x=X[:,-T:]; out=[]; names=[]
    for j,a in enumerate(assets):
        for w in (1,5,22,60,96):
            if w<=T:
                out += [x[:,-w:,j].mean(1),np.sqrt(np.mean(x[:,-w:,j]**2,1))]; names += [f'{a}_ret{w}',f'{a}_vol{w}']
    for j,n in enumerate(['HAR_d','HAR_w','HAR_m']): out.append(Hx[:,j]); names.append(n)
    return np.column_stack(out),names

def main():
    t0=time.time(); raw,assets,meta,X,Hx,y,tr,va,te=build(); har=LinearRegression().fit(Hx[tr],y[tr]); hp=har.predict(Hx); residual=y-hp
    seeds=[42,123,2024,3187,9254]; predictions={'HAR-RV':[hp[te]]*len(seeds)}; param_rows=[]
    # Common classical baselines.
    xe,names=eng(X,Hx,assets,DEFAULT_T)
    ridge=Ridge(10).fit(xe[tr],y[tr]); rf=RandomForestRegressor(n_estimators=500,min_samples_leaf=5,max_features=.6,n_jobs=-1,random_state=42).fit(xe[tr],y[tr])
    gb=lgb.LGBMRegressor(n_estimators=1200,learning_rate=.015,num_leaves=15,max_depth=5,reg_lambda=2,random_state=42,verbosity=-1).fit(xe[tr],y[tr],eval_set=[(xe[va],y[va])],callbacks=[lgb.early_stopping(80,verbose=False)])
    for n,p in [('Ridge',ridge.predict(xe[te])),('Random Forest',rf.predict(xe[te])),('LightGBM',gb.predict(xe[te]))]: predictions[n]=[p]*len(seeds)
    # Five-seed deep models.
    for s in seeds:
        vv,pp,np_,ep=train_seq('lstm',X,Hx,residual,tr,va,te,T=96,hidden=128,layers=2,norm=True,har=False,seed=s)
        alpha=float(np.linspace(0,1.25,126)[np.argmin([mean_squared_error(y[va],hp[va]+a*vv) for a in np.linspace(0,1.25,126)])])
        predictions.setdefault('HAR + Residual LSTM',[]).append(hp[te]+alpha*pp); param_rows.append({'model':'Residual LSTM','seed':s,'params':np_,'epochs':ep,'alpha':alpha})
        _,pp,np_,ep=train_seq('lstm',X,Hx,y,tr,va,te,T=96,hidden=128,layers=2,norm=True,har=True,seed=s); predictions.setdefault('Direct LSTM',[]).append(pp); param_rows.append({'model':'Direct LSTM','seed':s,'params':np_,'epochs':ep})
        _,pp,np_,ep=train_seq('itransformer',X,Hx,y,tr,va,te,T=96,seed=s); predictions.setdefault('iTransformer',[]).append(pp); param_rows.append({'model':'iTransformer','seed':s,'params':np_,'epochs':ep})
    rows=[]
    for n,ps in predictions.items():
        for k,p in enumerate(ps): rows.append({'Model':n,'seed':seeds[k],**metric(y[te],p)})
    pd.DataFrame(rows).to_csv(ART/'all_model_seed_results.csv',index=False)
    summary=pd.DataFrame(rows).groupby('Model')[['MSE','MAE','R2','QLIKE']].agg(['mean','std']); summary.to_csv(ART/'all_model_summary.csv')
    pd.DataFrame(param_rows).to_csv(ART/'deep_model_training.csv',index=False)
    # Use seed-ensemble averages for formal paired comparisons.
    ensemble={n:np.mean(np.stack(ps),0) for n,ps in predictions.items()}
    pd.DataFrame({'Date':meta.Date.iloc[te].to_numpy(),'y':y[te],**ensemble}).to_csv(ART/'ensemble_predictions.csv',index=False)
    dmrows=[]
    for other in ['HAR-RV','Direct LSTM','iTransformer','LightGBM','Random Forest']:
        for loss in ['mse','qlike']:
            z=dm(y[te],ensemble['HAR + Residual LSTM'],ensemble[other],loss); dmrows.append({'comparison':f'Proposed vs {other}','loss':loss,**dict(zip(['DM','p','mean_diff'],z))})
    pd.DataFrame(dmrows).to_csv(ART/'dm_all_models.csv',index=False)
    # Exact outline ablations, single fixed seed; common 252-history mother sample.
    configs=[('完整模型',dict(T=96,hidden=128,layers=2,norm=True,har=False)),('无LayerNorm',dict(T=96,hidden=128,layers=2,norm=False,har=False)),
             ('hidden=64',dict(T=96,hidden=64,layers=2,norm=True,har=False)),('hidden=256',dict(T=96,hidden=256,layers=2,norm=True,har=False)),
             ('layers=1',dict(T=96,hidden=128,layers=1,norm=True,har=False)),('layers=3',dict(T=96,hidden=128,layers=3,norm=True,har=False)),
             ('T=48',dict(T=48,hidden=128,layers=2,norm=True,har=False)),('T=252',dict(T=252,hidden=128,layers=2,norm=True,har=False))]
    ab=[]
    for name,kw in configs:
        vv,pp,np_,ep=train_seq('lstm',X,Hx,residual,tr,va,te,seed=42,**kw); grid=np.linspace(0,1.25,126); alpha=float(grid[np.argmin([mean_squared_error(y[va],hp[va]+a*vv) for a in grid])]); pred=hp[te]+alpha*pp
        ab.append({'Configuration':name,**metric(y[te],pred),'Delta_R2_vs_HAR':r2_score(y[te],pred)-r2_score(y[te],hp[te]),'params':np_,'epochs':ep,'alpha':alpha})
    # w/o HAR means direct LSTM with returns only; w/o residual means direct LSTM + HAR input.
    for name,harin in [('w/o HAR features',False),('w/o Residual Learning',True)]:
        _,p,np_,ep=train_seq('lstm',X,Hx,y,tr,va,te,T=96,hidden=128,layers=2,norm=True,har=harin,seed=42)
        ab.append({'Configuration':name,**metric(y[te],p),'Delta_R2_vs_HAR':r2_score(y[te],p)-r2_score(y[te],hp[te]),'params':np_,'epochs':ep,'alpha':np.nan})
    pd.DataFrame(ab).to_csv(ART/'outline_ablation.csv',index=False)
    # Full 15x15 Granger matrix (VAR-based Wald causality, lag selected by BIC up to 5 on train+val only).
    vr=raw[['Date']+assets].set_index('Date').loc[:meta.Date.iloc[va[-1]],assets]
    vsc=(vr-vr.mean())/vr.std(); order=max(1,VAR(vsc).select_order(5).selected_orders.get('bic') or 1); vf=VAR(vsc).fit(order)
    gp=pd.DataFrame(np.ones((len(assets),len(assets))),index=assets,columns=assets); gs=pd.DataFrame(np.zeros_like(gp),index=assets,columns=assets)
    for target in assets:
        for source in assets:
            if target!=source:
                test=vf.test_causality(target,[source],kind='wald'); gp.loc[target,source]=test.pvalue; gs.loc[target,source]=test.test_statistic
    gp.to_csv(ART/'granger_pvalue_matrix.csv'); gs.to_csv(ART/'granger_stat_matrix.csv'); (ART/'var_order.json').write_text(json.dumps({'BIC_lag':order}),encoding='utf-8')
    # IRF: IXIC response to one-SD generalized ordering-dependent shocks, reported transparently as orthogonalized VAR IRF.
    irf=vf.irf(20).orth_irfs; ti=assets.index('IXIC'); top=gp.loc['IXIC'].drop('IXIC').nsmallest(6).index
    ir=[]
    for src in top:
        sj=assets.index(src)
        for h in range(21): ir.append({'source':src,'target':'IXIC','horizon':h,'response':irf[h,ti,sj]})
    pd.DataFrame(ir).to_csv(ART/'var_irf_ixic.csv',index=False)
    # Nonlinear predictability of HAR residual: LightGBM vs zero on same engineered cross-asset set.
    xcross=xe[:,[i for i,n in enumerate(names) if not n.startswith('HAR') and not n.startswith('IXIC')]]
    nl=lgb.LGBMRegressor(n_estimators=800,learning_rate=.015,num_leaves=7,max_depth=3,reg_lambda=4,random_state=42,verbosity=-1).fit(xcross[tr],residual[tr],eval_set=[(xcross[va],residual[va])],callbacks=[lgb.early_stopping(60,verbose=False)])
    nr=nl.predict(xcross[te]); zero=np.zeros_like(nr); t,p,md=dm(residual[te],nr,zero,'mse')
    (ART/'nonlinear_residual_test.json').write_text(json.dumps({'Residual_R2':r2_score(residual[te],nr),'MSE_model':mean_squared_error(residual[te],nr),'MSE_zero':mean_squared_error(residual[te],zero),'DM':t,'p':p,'mean_diff':md},indent=2),encoding='utf-8')
    # Regime and rolling-year results using ensemble proposed.
    mm=meta.iloc[te].copy(); mm['HAR']=ensemble['HAR-RV']; mm['Proposed']=ensemble['HAR + Residual LSTM']; mm['truth']=y[te]; mm['correction']=mm.Proposed-mm.HAR; q=np.quantile(np.exp(mm.har_d),[.25,.75]); mm['regime']=np.where(np.exp(mm.har_d)>=q[1],'高波动',np.where(np.exp(mm.har_d)<=q[0],'低波动','中波动'))
    rr=[]
    for g,z in mm.groupby('regime'): rr.append({'regime':g,'n':len(z),'HAR_MSE':mean_squared_error(z.truth,z.HAR),'Proposed_MSE':mean_squared_error(z.truth,z.Proposed),'mean_correction':z.correction.mean(),'mean_abs_correction':z.correction.abs().mean()})
    pd.DataFrame(rr).to_csv(ART/'regime_complete.csv',index=False); mm.to_csv(ART/'mechanism_timeseries.csv',index=False)
    yr=[]
    for year,z in mm.groupby(mm.Date.dt.year): yr.append({'year':year,'n':len(z),'HAR_MSE':mean_squared_error(z.truth,z.HAR),'Proposed_MSE':mean_squared_error(z.truth,z.Proposed),'Delta_R2':r2_score(z.truth,z.Proposed)-r2_score(z.truth,z.HAR)})
    pd.DataFrame(yr).to_csv(ART/'yearly_robustness.csv',index=False)
    metaout={'device':str(DEVICE),'assets':assets,'VIX_used':False,'max_history':MAX_T,'default_history':DEFAULT_T,'n':len(y),'train_n':len(tr),'val_n':len(va),'test_n':len(te),'runtime_s':time.time()-t0}
    (ART/'metadata_complete.json').write_text(json.dumps(metaout,ensure_ascii=False,indent=2),encoding='utf-8'); print(summary); print(pd.DataFrame(dmrows)); print(pd.DataFrame(ab)); print(metaout)

if __name__=='__main__': main()
