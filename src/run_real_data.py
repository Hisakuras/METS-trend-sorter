#!/usr/bin/env python3
"""METS 真实数据集全面验证：4 数据集 × 4 数据特性"""
import json
import numpy as np
from scipy.stats import spearmanr, norm

import statsmodels.api as sm

import sys
sys.path.insert(0, "/tmp")
from mets_v5 import METS_rank, METS_single


def theil_sen_slope(arr):
    n = len(arr)
    if n < 2: return 0.0
    i, j = np.triu_indices(n, k=1)
    if len(i) > 40000:
        idx = np.random.default_rng(0).choice(len(i), 40000, replace=False)
        i, j = i[idx], j[idx]
    return float(np.median((arr[j] - arr[i]) / (j - i + 1e-9)))


def ols_slope(arr):
    return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])


def mk_trend_test(arr):
    """Mann-Kendall 趋势检验 p 值"""
    n = len(arr)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(arr[j] - arr[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if var_s == 0: return 1.0
    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
    return 2 * norm.sf(abs(z))


def evaluate(name, series_dict, out):
    names = list(series_dict.keys())
    arrays = [series_dict[nm] for nm in names]
    n_series = len(arrays)
    lengths = [len(a) for a in arrays]

    ts_s = np.array([theil_sen_slope(a) for a in arrays])
    ols_s = np.array([ols_slope(a) for a in arrays])
    _, mets_s, mets_sig = METS_rank(arrays)

    rho_mets_ts = spearmanr(mets_s, ts_s)[0]
    rho_ols_ts = spearmanr(ols_s, ts_s)[0]
    rho_mets_ols = spearmanr(mets_s, ols_s)[0]

    ts_sign = np.sign(ts_s)
    mets_dir = float(np.mean(np.sign(mets_s) == ts_sign)) if n_series > 1 else np.nan
    ols_dir = float(np.mean(np.sign(ols_s) == ts_sign)) if n_series > 1 else np.nan

    mk_ps = np.array([mk_trend_test(a) for a in arrays])
    mk_sig = mk_ps < 0.05
    mets_sig_count = int(np.sum(mets_sig))
    agreement = float(np.mean(mets_sig == mk_sig)) if n_series > 1 else np.nan

    print(f"\n=== {name} ({n_series} 序列, 长度 {min(lengths)}-{max(lengths)}) ===")
    print(f"  vs Theil-Sen: METS rho={rho_mets_ts:.3f} | OLS rho={rho_ols_ts:.3f}")
    print(f"  vs OLS: METS rho={rho_mets_ols:.3f}")
    print(f"  方向一致: METS {mets_dir*100:.1f}% | OLS {ols_dir*100:.1f}%")
    print(f"  MK显著 {mk_sig.sum()}/{n_series} | METS显著 {mets_sig_count}/{n_series} | 一致率 {agreement*100:.1f}%")

    out[name] = {
        "n_series": n_series, "lengths": [int(l) for l in lengths],
        "rho_mets_ts": round(rho_mets_ts, 3), "rho_ols_ts": round(rho_ols_ts, 3),
        "rho_mets_ols": round(rho_mets_ols, 3),
        "mets_direction": round(mets_dir, 3), "ols_direction": round(ols_dir, 3),
        "mk_significant": int(mk_sig.sum()), "mets_significant": mets_sig_count,
        "agreement": round(agreement, 3),
        "series_names": names,
        "mets_scores": [round(float(x), 4) for x in mets_s],
        "ols_slopes": [round(float(x), 4) for x in ols_s],
        "ts_slopes": [round(float(x), 4) for x in ts_s],
        "mk_sig_flags": [bool(x) for x in mk_sig],
        "mets_sig_flags": [bool(x) for x in mets_sig],
    }
    return out


if __name__ == "__main__":
    results = {}

    # 1. macrodata 经济指标（线性主导）
    md = sm.datasets.macrodata.load_pandas().data
    econ = {}
    for col in ["realgdp", "realcons", "realinv", "realgovt", "realdpi",
                "cpi", "m1", "tbilrate", "unemp", "pop", "infl", "realint"]:
        econ[col] = md[col].dropna().values.astype(float)
    results = evaluate("macrodata(经济)", econ, results)

    # 2. elnino 逐月海温（季节周期）
    en = sm.datasets.elnino.load_pandas().data
    sea = {m: en[m].dropna().values.astype(float)
           for m in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]}
    results = evaluate("elnino(海温)", sea, results)

    # 3. co2 年度切片（强趋势）：每年一个序列
    co2 = sm.datasets.co2.load_pandas().data["co2"].dropna()
    years = sorted(set(co2.index.year))
    co2_annual = {}
    for y in years:
        vals = co2[co2.index.year == y].values.astype(float)
        if len(vals) >= 40:  # 需要足够长度
            co2_annual[str(y)] = vals
    results = evaluate("co2年度切片(强趋势)", co2_annual, results)

    # 4. sunspots 周期窗口（纯周期）：每 11 年一个窗口
    sp = sm.datasets.sunspots.load_pandas().data
    years_all = sp["YEAR"].values.astype(int)
    activity = sp["SUNACTIVITY"].values.astype(float)
    windows = {}
    wstart = years_all[0]
    while wstart + 11 <= years_all[-1]:
        mask = (years_all >= wstart) & (years_all < wstart + 11)
        if mask.sum() >= 8:
            windows[f"{wstart}-{wstart+10}"] = activity[mask]
        wstart += 11
    results = evaluate("sunspots窗口(周期)", windows, results)

    # 5. danish_data 丹麦宏观经济（对数尺度变量）
    dd = sm.datasets.danish_data.load_pandas().data
    dk = {}
    for col in ["lrm", "lry", "lpy", "ibo", "ide"]:
        dk[col] = dd[col].dropna().values.astype(float)
    results = evaluate("danish(丹麦经济)", dk, results)

    # 6. grunfeld 公司面板（11 家公司 × 20 年投资额）
    gf = sm.datasets.grunfeld.load_pandas().data
    firms = {}
    for firm_name in gf["firm"].unique():
        sub = gf[gf["firm"] == firm_name].sort_values("year")
        vals = sub["invest"].dropna().values.astype(float)
        if len(vals) >= 15:
            firms[str(firm_name)] = vals
    results = evaluate("grunfeld(公司投资)", firms, results)



    with open("/Users/maxleah/Desktop/mets_figs/real_data_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存 real_data_results.json")
