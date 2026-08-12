import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path('/Users/maxleah/Documents/Codex/2026-08-12/gen')
DATA = ROOT/'work'/'cmapss'
OUT = ROOT/'work'/'cmapss_results'
OUT.mkdir(parents=True, exist_ok=True)
COLS = ['unit','cycle','op1','op2','op3'] + [f's{i}' for i in range(1,22)]

def load(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=COLS, engine='python')

def ts_slope(y):
    y=np.asarray(y,float); n=len(y); x=np.arange(n,dtype=float)
    pairs=np.triu_indices(n,1)
    if len(pairs[0])>1000:
        rng=np.random.default_rng(20260812+n)
        take=rng.choice(len(pairs[0]),size=1000,replace=False)
        ii,jj=pairs[0][take],pairs[1][take]
    else:
        ii,jj=pairs
    return float(np.median((y[jj]-y[ii])/(x[jj]-x[ii])))

def slopes(y):
    y=np.asarray(y,float); x=np.arange(len(y),dtype=float)
    ols=float(np.polyfit(x,y,1)[0]); ts=ts_slope(y)
    w=max(8,int(len(y)*.6)); starts=np.linspace(0,len(y)-w,5).round().astype(int)
    local=np.array([ts_slope(y[st:st+w]) for st in starts])
    return ols,ts,float(np.median(local)),float(np.quantile(local,.1)),float(np.quantile(local,.9))

def robust_scales(train):
    sc={}
    for s in [f's{i}' for i in range(1,22)]:
        v=train[s].to_numpy(float); med=np.median(v); mad=np.median(np.abs(v-med))*1.4826
        sc[s]=mad if mad>1e-12 else max(np.std(v),1.0)
    return sc

def feature_table(df, cutoff, scales):
    rows=[]
    for unit,g in df.groupby('unit'):
        g=g.sort_values('cycle'); L=int(g.cycle.max())
        if L<cutoff+8: continue
        h=g[g.cycle<=cutoff]
        row={'unit':int(unit),'life':L,'rul_at_cutoff':L-cutoff}
        for s,scale in scales.items():
            o,t,c,lo,hi=slopes(h[s].to_numpy()/scale)
            row[f'{s}_ols']=o; row[f'{s}_ts']=t; row[f'{s}_si']=c; row[f'{s}_width']=hi-lo
        rows.append(row)
    return pd.DataFrame(rows)

def evaluate(df, cutoff, seed=20260812):
    rng=np.random.default_rng(seed); units=df.unit.unique(); rng.shuffle(units); fold=np.array_split(units,3); records=[]
    for k,test_units in enumerate(fold):
        train_units=set(units)-set(test_units); tr=df[df.unit.isin(train_units)]; te=df[df.unit.isin(test_units)]
        scales=robust_scales(tr); Ftr=feature_table(tr,cutoff,scales); Fte=feature_table(te,cutoff,scales)
        sensors=[f's{i}' for i in range(1,22)]
        for method in ['ols','ts','si']:
            assoc=[]
            for s in sensors:
                x=np.abs(Ftr[f'{s}_{method}']); y=-Ftr.rul_at_cutoff
                assoc.append((s, spearmanr(x,y).statistic if x.nunique()>1 else 0.0))
            order=[s for s,_ in sorted(assoc,key=lambda z:abs(z[1]),reverse=True)]
            for topk in [3,5,10]:
                top=order[:topk]
                score=np.column_stack([np.abs(Fte[f'{s}_{method}']) for s in top]).mean(axis=1)
                rho=spearmanr(score,-Fte.rul_at_cutoff).statistic
                tau=kendalltau(score,-Fte.rul_at_cutoff).statistic
                widths=np.column_stack([Fte[f'{s}_width'] for s in top]).mean() if method=='si' else np.nan
                records.append({'cutoff':cutoff,'fold':k+1,'method':method,'topk':topk,'spearman':rho,'kendall':tau,'mean_struct_width':widths,'selected':';'.join(top)})
    out=pd.DataFrame(records); out.to_csv(OUT/f'cmapss_fd001_cutoff_{cutoff}_fold_results.csv',index=False)
    return out

def main():
    train=load(DATA/'train_FD001.txt'); test=load(DATA/'test_FD001.txt')
    allres=[]
    for cutoff in [50,80]: allres.append(evaluate(train,cutoff))
    res=pd.concat(allres,ignore_index=True)
    summary=res.groupby(['cutoff','method','topk'],as_index=False).agg(spearman_mean=('spearman','mean'),spearman_sd=('spearman','std'),kendall_mean=('kendall','mean'),mean_struct_width=('mean_struct_width','mean'))
    summary.to_csv(OUT/'cmapss_fd001_summary.csv',index=False)
    meta={'dataset':'NASA C-MAPSS FD001','source_primary':'https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data','mirror_used':'https://github.com/huster123/c-mapss-full-dataset-','train_units':int(train.unit.nunique()),'cutoffs':[50,80],'folds':3,'seed':20260812,'scale':'fleet MAD from training fold','methods':['OLS','global Theil-Sen','METS-SI center=local Theil-Sen median; width=Q90-Q10'],'note':'This is a sensor-selection proxy task, not an RUL predictor.'}
    for p in [DATA/'train_FD001.txt',DATA/'test_FD001.txt',DATA/'RUL_FD001.txt']:
        meta[p.name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'cmapss_fd001_metadata.json').write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__': main()

