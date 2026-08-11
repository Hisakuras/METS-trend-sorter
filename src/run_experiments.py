"""METS v3 完整实验：真实数据 + 图表"""
import json
import os
import time
import numpy as np
from scipy.stats import spearmanr

# matplotlib 修复
import matplotlib
import matplotlib.font_manager as fm
import pathlib

def _safe_macos_fonts():
    paths = [pathlib.Path("/System/Library/Fonts"),
             pathlib.Path("/System/Library/Fonts/Supplemental"),
             pathlib.Path("/Library/Fonts"),
             pathlib.Path.home() / "Library/Fonts"]
    found = []
    for p in paths:
        if p.exists():
            found.extend(p.glob("*.ttf")); found.extend(p.glob("*.ttc")); found.extend(p.glob("*.otf"))
    return found

fm._get_macos_fonts = _safe_macos_fonts
try: del fm.fontManager
except Exception: pass
fm.fontManager = fm.FontManager()
for fp in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, "/tmp")
from mets_v3 import METS_rank, METS_single

OUT = "/Users/maxleah/Desktop/mets_figs/"
os.makedirs(OUT, exist_ok=True)


def ols_slope(arr):
    n = len(arr)
    return np.polyfit(np.arange(n), arr, 1)[0]


def theil_sen_slope(arr):
    n = len(arr)
    if n < 2: return 0.0
    i, j = np.triu_indices(n, k=1)
    if len(i) > 40000:
        idx = np.random.default_rng(0).choice(len(i), 40000, replace=False)
        i, j = i[idx], j[idx]
    return float(np.median((arr[j] - arr[i]) / (j - i + 1e-9)))


def base_extreme_slope(arr):
    """Base-Extreme：原分位数极值（对照旧版）"""
    z = (arr - arr.mean()) / (arr.std() + 1e-9)
    n = len(z)
    y_max = np.percentile(z, 95); y_min = np.percentile(z, 5)
    i_max = np.mean(np.where(z >= y_max)[0]) if np.any(z >= y_max) else np.argmax(z)
    i_min = np.mean(np.where(z <= y_min)[0]) if np.any(z <= y_min) else np.argmin(z)
    return np.sign(i_max - i_min) * (y_max - y_min) / (n ** 0.7)


def gen_linear(n, k, scale, noise_ratio=0.15):
    t = np.arange(n) / (n - 1)
    return scale * (1 + 0.5 * k * t) + np.random.default_rng().normal(0, noise_ratio * scale, n)


def add_salt_pepper(arr, ratio, sigma):
    arr = arr.copy(); n = len(arr)
    n_out = int(n * ratio)
    if n_out > 0:
        idx = np.random.default_rng().choice(n, n_out, replace=False)
        arr[idx] = np.random.default_rng().choice([-1, 1], n_out) * 10 * sigma
    return arr


# 实验1：噪声鲁棒性
def exp1():
    rng = np.random.default_rng(42)
    results = {"noise_ratio": [], "OLS": [], "TheilSen": [], "BaseExtreme": [], "METS": []}
    for ratio in [0.0, 0.1, 0.2, 0.3]:
        slopes, ols_s, ts_s, be_s, arrays = [], [], [], [], []
        for _ in range(80):
            n = int(rng.integers(15, 60))
            k = rng.uniform(-2, 2)
            scale = rng.uniform(10, 1000)
            arr = add_salt_pepper(gen_linear(n, k, scale), ratio, 0.15 * scale)
            slopes.append(k); arrays.append(arr)
        slopes = np.array(slopes)
        for a in arrays:
            ols_s.append(ols_slope(a)); ts_s.append(theil_sen_slope(a)); be_s.append(base_extreme_slope(a))
        _, mets_s, _ = METS_rank(arrays, B=500, seed=1)
        results["noise_ratio"].append(ratio)
        results["OLS"].append(round(spearmanr(ols_s, slopes)[0], 3))
        results["TheilSen"].append(round(spearmanr(ts_s, slopes)[0], 3))
        results["BaseExtreme"].append(round(spearmanr(be_s, slopes)[0], 3))
        results["METS"].append(round(spearmanr(mets_s, slopes)[0], 3))
    return results


# 实验2：无效趋势剔除
def exp2():
    rng = np.random.default_rng(7)
    arrays, slopes = [], []
    for _ in range(100):
        n = int(rng.integers(15, 60))
        k = rng.uniform(-2, 2)
        scale = rng.uniform(10, 1000)
        arrays.append(gen_linear(n, k, scale)); slopes.append(k)
    for _ in range(50):
        n = int(rng.integers(15, 60))
        arrays.append(rng.normal(0, 100, n)); slopes.append(0.0)
    slopes = np.array(slopes)
    results = {}
    ps, scores_all, Qs = [], [], []
    for a in arrays:
        s, Q, p = METS_single(a, B=500, seed=1)
        scores_all.append(s); Qs.append(Q); ps.append(p)
    ps = np.array(ps); scores_all = np.array(scores_all); Qs = np.array(Qs)
    scores_no_fdr = np.where(ps < 0.05, scores_all, 0.2 * scores_all)
    scores_no_fdr[Qs < 0.3] = 0
    noise_mask = slopes == 0
    all_med_no_fdr = np.median(np.abs(scores_no_fdr)) if np.any(scores_no_fdr) else 1
    results["METS_noFDR_误排前50%"] = round(float(np.mean(
        np.argsort(np.argsort(scores_no_fdr))[noise_mask] >= len(arrays) / 2)), 3)
    results["METS_noFDR_假阳性率"] = round(float(np.mean(ps[noise_mask] < 0.05)), 3)
    _, scores_fdr, sig = METS_rank(arrays, B=500, seed=1)
    results["METS_FDR_误排前50%"] = round(float(np.mean(
        np.argsort(np.argsort(scores_fdr))[noise_mask] >= len(arrays) / 2)), 3)
    results["METS_FDR_假阳性率"] = round(float(np.mean(sig[noise_mask])), 3)
    ols_s = np.array([ols_slope(a) for a in arrays])
    ols_rank = np.argsort(np.argsort(ols_s))
    results["OLS_误排前50%"] = round(float(np.mean(ols_rank[noise_mask] >= len(arrays) / 2)), 3)
    return results


# 实验3：非线性形态
def exp3():
    rng = np.random.default_rng(11)
    forms = {"MonoUp": [], "MonoDown": [], "Vshape": [], "InvVshape": []}
    for _ in range(40):
        n = int(rng.integers(20, 50))
        t = np.linspace(0, 1, n)
        forms["MonoUp"].append(100 * t + rng.normal(0, 3, n))
        forms["MonoDown"].append(100 * (1 - t) + rng.normal(0, 3, n))
        v = np.concatenate([np.linspace(0, 1, n // 2), np.linspace(1, 0, n - n // 2)])
        forms["Vshape"].append(100 * v + rng.normal(0, 3, n))
        forms["InvVshape"].append(100 * (1 - v) + rng.normal(0, 3, n))
    results = {"form": [], "OLS": [], "TheilSen": [], "METS": []}
    all_arr, all_form = [], []
    for f, arrs in forms.items():
        all_arr.extend(arrs); all_form.extend([f] * len(arrs))
    ols_s = np.array([ols_slope(a) for a in all_arr])
    ts_s = np.array([theil_sen_slope(a) for a in all_arr])
    _, mets_s, _ = METS_rank(all_arr, B=500, seed=1)
    for f in forms:
        idx = [i for i, x in enumerate(all_form) if x == f]
        results["form"].append(f)
        results["OLS"].append(int(np.mean(np.argsort(np.argsort(ols_s))[idx])))
        results["TheilSen"].append(int(np.mean(np.argsort(np.argsort(ts_s))[idx])))
        results["METS"].append(int(np.mean(np.argsort(np.argsort(mets_s))[idx])))
    return results


# 实验4：非线性单调趋势
def exp4():
    rng = np.random.default_rng(19)
    results = {"form": [], "OLS": [], "TheilSen": [], "METS": []}
    for form in ["linear", "power", "exp", "log"]:
        slopes, arrs = [], []
        for _ in range(50):
            n = int(rng.integers(20, 60))
            k = rng.uniform(1, 3) * rng.choice([-1, 1])
            t = np.linspace(0, 1, n)
            if form == "linear": a = 100 * (1 + 0.5 * k * t)
            elif form == "power": a = 100 * (1 + 0.3 * k * t ** 1.5)
            elif form == "exp": a = 100 * np.exp(0.3 * k * t)
            else: a = 100 * (1 + 0.3 * k * np.log1p(3 * t))
            arrs.append(a + rng.normal(0, 8, n)); slopes.append(k)
        slopes = np.array(slopes)
        ols_s = [ols_slope(a) for a in arrs]
        ts_s = [theil_sen_slope(a) for a in arrs]
        _, mets_s, _ = METS_rank(arrs, B=500, seed=1)
        results["form"].append(form)
        results["OLS"].append(round(spearmanr(ols_s, slopes)[0], 3))
        results["TheilSen"].append(round(spearmanr(ts_s, slopes)[0], 3))
        results["METS"].append(round(spearmanr(mets_s, slopes)[0], 3))
    return results


# 实验5：计算效率
def exp5():
    rng = np.random.default_rng(23)
    arrays = []
    for _ in range(200):
        n = 100
        k = rng.uniform(-2, 2); scale = rng.uniform(10, 1000)
        arrays.append(gen_linear(n, k, scale))
    timing = {}
    t0 = time.perf_counter()
    for a in arrays: ols_slope(a)
    timing["OLS"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    for a in arrays: theil_sen_slope(a)
    timing["TheilSen"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    for a in arrays: base_extreme_slope(a)
    timing["BaseExtreme"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    METS_rank(arrays, B=1000, seed=1)
    timing["METS(B=1000)"] = round((time.perf_counter() - t0) * 1000, 1)
    return timing


# 实验6：消融
def exp6():
    rng = np.random.default_rng(29)
    slopes, arrays = [], []
    for _ in range(100):
        n = int(rng.integers(15, 60))
        k = rng.uniform(-2, 2)
        scale = rng.uniform(10, 1000)
        arr = add_salt_pepper(gen_linear(n, k, scale), 0.2, 0.15 * scale)
        slopes.append(k); arrays.append(arr)
    slopes = np.array(slopes)
    results = {}
    _, s_full, _ = METS_rank(arrays, B=500, seed=1)
    results["完整METS"] = round(spearmanr(s_full, slopes)[0], 3)
    s_no_norm = np.array([base_extreme_slope(a) for a in arrays])
    results["移除归一化"] = round(spearmanr(s_no_norm, slopes)[0], 3)
    ps = [METS_single(a, B=500, seed=1)[2] for a in arrays]
    s_no_perm = np.array([s if p < 0.05 else s / 0.2 for s, p in zip(s_full, ps)])
    results["移除显著性惩罚"] = round(spearmanr(s_no_perm, slopes)[0], 3)
    return results


# 实验7：参数敏感性
def exp7():
    rng = np.random.default_rng(31)
    slopes, arrays = [], []
    for _ in range(100):
        n = int(rng.integers(15, 60))
        k = rng.uniform(-2, 2); scale = rng.uniform(10, 1000)
        arrays.append(gen_linear(n, k, scale)); slopes.append(k)
    slopes = np.array(slopes)
    results = {"param": [], "value": [], "rho": []}
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        _, s, _ = METS_rank(arrays, alpha=alpha, B=200, seed=1)
        results["param"].append("alpha"); results["value"].append(alpha)
        results["rho"].append(round(spearmanr(s, slopes)[0], 3))
    for B in [100, 300, 1000]:
        _, s, _ = METS_rank(arrays, B=B, seed=1)
        results["param"].append("B"); results["value"].append(B)
        results["rho"].append(round(spearmanr(s, slopes)[0], 3))
    return results


def make_figs(e1, e4, e6, e7):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = e1["noise_ratio"]
    for key, marker, color in [("OLS", "o", "#d62728"), ("TheilSen", "^", "#9467bd"),
                                ("BaseExtreme", "s", "#ff7f0e"), ("METS", "D", "#1f77b4")]:
        ax.plot(x, e1[key], marker=marker, label=key, linewidth=2, markersize=6)
    ax.set_xlabel("椒盐噪声比例"); ax.set_ylabel("Spearman 秩相关系数")
    ax.set_title("实验1：噪声鲁棒性对比（真实数据）")
    ax.set_xticks(x); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}fig1_noise_robustness.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(e4["form"])); w = 0.25
    for i, key in enumerate(["OLS", "TheilSen", "METS"]):
        ax.bar(x + (i - 1) * w, e4[key], w, label=key)
    ax.set_xticks(x); ax.set_xticklabels(["线性", "幂律", "指数", "对数"])
    ax.set_ylabel("Spearman 秩相关系数"); ax.set_title("实验4：非线性单调趋势排序能力")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0.5, 1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}fig2_nonlinear_trend.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(e6.keys()); vals = list(e6.values())
    colors = ["#1f77b4"] + ["#d62728"] * (len(labels) - 1)
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Spearman 秩相关系数"); ax.set_title("实验6：消融实验（噪声20%）")
    ax.set_ylim(0.3, 1.0); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}fig3_ablation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = [v for p, v in zip(e7["param"], e7["value"]) if p == "alpha"]
    rhos_a = [r for p, r in zip(e7["param"], e7["rho"]) if p == "alpha"]
    ax.plot(alphas, rhos_a, "o-", label="α（长度惩罚）")
    ax.set_xlabel("α 取值"); ax.set_ylabel("Spearman 秩相关系数")
    ax.set_title("实验7：参数敏感性（α）")
    ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(0.6, 1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}fig4_sensitivity.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    print("=== 实验1：噪声鲁棒性 ===")
    e1 = exp1(); print(json.dumps(e1, ensure_ascii=False))
    print("\n=== 实验2：无效趋势剔除 ===")
    e2 = exp2(); print(json.dumps(e2, ensure_ascii=False))
    print("\n=== 实验3：非线性形态 ===")
    e3 = exp3(); print(json.dumps(e3, ensure_ascii=False))
    print("\n=== 实验4：非线性单调趋势 ===")
    e4 = exp4(); print(json.dumps(e4, ensure_ascii=False))
    print("\n=== 实验5：计算效率 ===")
    e5 = exp5(); print(json.dumps(e5, ensure_ascii=False))
    print("\n=== 实验6：消融 ===")
    e6 = exp6(); print(json.dumps(e6, ensure_ascii=False))
    print("\n=== 实验7：参数敏感性 ===")
    e7 = exp7(); print(json.dumps(e7, ensure_ascii=False))
    make_figs(e1, e4, e6, e7)
    print(f"\n图表已保存至 {OUT}")
    with open(f"{OUT}experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({"exp1": e1, "exp2": e2, "exp3": e3, "exp4": e4,
                   "exp5": e5, "exp6": e6, "exp7": e7}, f, ensure_ascii=False, indent=2)
    print("结果已保存 experiment_results.json")
