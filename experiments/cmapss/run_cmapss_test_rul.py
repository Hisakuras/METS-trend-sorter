import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
from run_cmapss_validation import load, robust_scales, slopes

ROOT=Path('/Users/maxleah/Documents/Codex/2026-08-12/gen'); DATA=ROOT/'work/cmapss'; OUT=ROOT/'work/cmapss_results'
SENS=[f's{i}' for i in range(1,22)]

def full_features(df, scales):
    rows=[]
    for unit,g in df.groupby('unit'):
        g=g.sort_values('cycle'); row={'unit':int(unit),'life':int(g.cycle.max())}
        for s in SENS:
            o,t,c,lo,hi=slopes(g[s].to_numpy()/scales[s])
            row[f'{s}_ols']=o; row[f'{s}_ts']=t; row[f'{s}_si']=t; row[f'{s}_width']=hi-lo
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    tr=load(DATA/'train_FD001.txt'); te=load(DATA/'test_FD001.txt'); rul=np.loadtxt(DATA/'RUL_FD001.txt')
    rng=np.random.default_rng(20260812); units=tr.unit.unique(); rng.shuffle(units); folds=np.array_split(units,3); allrows=[]
    for fold,test_units in enumerate(folds,1):
        train_units=set(units)-set(test_units); a=tr[tr.unit.isin(train_units)]; scales=robust_scales(a)
        F=full_features(a,scales); T=full_features(te,scales)
        for method in ['ols','ts','si']:
            assoc=[]
            for s in SENS:
                v=spearmanr(np.abs(F[f'{s}_{method}']),-F.life).statistic
                assoc.append((s,0 if np.isnan(v) else abs(v)))
            order=[s for s,_ in sorted(assoc,key=lambda z:z[1],reverse=True)]
            for topk in [3,5,10]:
                top=order[:topk]; score=np.column_stack([np.abs(T[f'{s}_{method}']) for s in top]).mean(1)
                # higher score should correspond to lower RUL
                allrows.append({'fold':fold,'method':method,'topk':topk,'spearman':spearmanr(score,-rul).statistic,'kendall':kendalltau(score,-rul).statistic,'mean_width':np.mean([T[f'{s}_width'].mean() for s in top]) if method=='si' else np.nan,'selected':';'.join(top)})
    res=pd.DataFrame(allrows); res.to_csv(OUT/'cmapss_fd001_test_rul_fold_results.csv',index=False)
    summ=res.groupby(['method','topk'],as_index=False).agg(spearman_mean=('spearman','mean'),spearman_sd=('spearman','std'),kendall_mean=('kendall','mean'),mean_width=('mean_width','mean'))
    summ.to_csv(OUT/'cmapss_fd001_test_rul_summary.csv',index=False); print(summ.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__': main()

