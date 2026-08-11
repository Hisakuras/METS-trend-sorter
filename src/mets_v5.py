"""METS v5：性能优化版（np.argsort 秩 + 向量化 Spearman + 缓存时间轴）

目标：完整流程 O(N·n log n) 但常数极小，接近 OLS 量级
"""
import numpy as np
from scipy.stats import t as t_dist

# 时间轴缓存：{n: arange(n)}
_TIME_CACHE = {}


def _arange(n):
    if n not in _TIME_CACHE:
        _TIME_CACHE[n] = np.arange(n, dtype=float)
    return _TIME_CACHE[n]


def _rank_pos_diff(arr):
    """秩空间极值位置差（np.argsort 实现）"""
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")  # 稳定排序 → 秩
    r = np.empty(n, dtype=float)
    r[order] = np.arange(n, dtype=float)  # 秩 0..n-1
    hi_thr = 0.95 * (n - 1)
    lo_thr = 0.05 * (n - 1)
    i_hi = np.mean(np.where(r >= hi_thr)[0]) if np.any(r >= hi_thr) else n - 1
    i_lo = np.mean(np.where(r <= lo_thr)[0]) if np.any(r <= lo_thr) else 0
    return i_hi - i_lo


def _spearman_p(time_arr, arr):
    """Spearman 秩相关渐近 p 值（向量化，O(n log n)）"""
    n = len(arr)
    rt = np.argsort(time_arr, kind="mergesort")
    rt_rank = np.empty(n); rt_rank[rt] = np.arange(n)
    ra = np.argsort(arr, kind="mergesort")
    ra_rank = np.empty(n); ra_rank[ra] = np.arange(n)
    d = rt_rank - ra_rank
    rho = 1 - 6 * np.sum(d * d) / (n * (n * n - 1))
    if abs(rho) >= 1.0 - 1e-12:
        return 0.0
    t_stat = rho * np.sqrt((n - 2) / (1 - rho * rho))
    return 2 * t_dist.sf(abs(t_stat), df=n - 2)


def METS_single(arr, alpha=0.7, q_thresh=0.3):
    """METS v5 单序列（O(n log n)）"""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0, 0.0, 1.0

    mu = np.mean(arr)
    sigma = np.std(arr)
    if sigma < 1e-12:
        return 0.0, 0.0, 1.0
    z = (arr - mu) / sigma

    # 1. 秩空间极值（抗噪）
    order = np.argsort(arr, kind="mergesort")
    r = np.empty(n, dtype=float)
    r[order] = np.arange(n, dtype=float)
    hi_thr = 0.95 * (n - 1); lo_thr = 0.05 * (n - 1)
    i_hi = np.mean(np.where(r >= hi_thr)[0]) if np.any(r >= hi_thr) else n - 1
    i_lo = np.mean(np.where(r <= lo_thr)[0]) if np.any(r <= lo_thr) else 0
    y_max = np.percentile(z, 95); y_min = np.percentile(z, 5)
    if abs(y_max - y_min) < 1e-9:
        return 0.0, 0.0, 1.0

    # 2. 基础斜率 + 均值修正
    s_base = np.sign(i_hi - i_lo) * (y_max - y_min) / (n ** alpha)
    mid = (y_max + y_min) / 2
    beta = np.clip((0.0 - mid) / (y_max - y_min + 1e-9), -0.5, 0.5)
    s_adj = s_base * (1 + 0.3 * beta)

    # 3. 质量评估
    diffs = np.diff(z)
    pos = np.sum(diffs > 0); neg = np.sum(diffs < 0)
    mcr = max(pos, neg) / (n - 1)
    if i_hi != i_lo:
        slope_line = (y_max - y_min) / (i_hi - i_lo + 1e-9)
        intercept = y_min - slope_line * i_lo
        y_hat = slope_line * np.arange(n) + intercept
        ss_res = np.sum((z - y_hat) ** 2)
        ss_tot = np.sum((z - mu) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-9)
    else:
        r2 = 0.0
    Q = np.clip(0.5 * mcr + 0.5 * r2, 0.0, 1.0)
    if Q < q_thresh:
        return 0.0, Q, 1.0

    # 4. 多尺度
    split = n // 3
    if split < 1:
        s_multi = s_adj
    else:
        segs = [z[:split], z[split:2*split], z[2*split:]]
        s_l = _rank_pos_diff(segs[0])
        s_m = _rank_pos_diff(segs[1])
        s_r = _rank_pos_diff(segs[2])
        s_multi = np.sign(s_adj) * np.sqrt(abs(s_adj * s_m) + 1e-9)
        if s_l * s_r < 0:
            s_multi *= 0.5

    # 5. 渐近显著性
    p = _spearman_p(_arange(n), arr)

    return s_multi, Q, p


def METS_rank(arrays, alpha=0.7, q=0.05, q_thresh=0.3):
    """METS v5 批量排序 + BH 校正"""
    results = [METS_single(a, alpha=alpha, q_thresh=q_thresh) for a in arrays]
    scores = np.array([r[0] for r in results])
    Qs = np.array([r[1] for r in results])
    ps = np.array([r[2] for r in results])

    n = len(ps)
    order = np.argsort(ps)
    sorted_ps = ps[order]
    thresh = (np.arange(1, n + 1) / n) * q
    significant = np.zeros(n, dtype=bool)
    k_max = -1
    for k in range(n):
        if sorted_ps[k] <= thresh[k]:
            k_max = k
    if k_max >= 0:
        significant[order[:k_max + 1]] = True

    final_scores = np.where(significant, scores, 0.2 * scores)
    final_scores[Qs < q_thresh] = 0.0
    return np.argsort(final_scores), final_scores, significant


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    print('=== 方向正确率（v5）===')
    for ratio in [0.0, 0.1, 0.2, 0.3]:
        ok, tot = 0, 60
        for _ in range(tot):
            n = int(rng.integers(15, 60))
            k = rng.uniform(0.5, 2.0)
            t = np.arange(n) / (n - 1)
            arr = 100 * (1 + 0.5 * k * t) + rng.normal(0, 15, n)
            n_out = int(n * ratio)
            if n_out > 0:
                idx = rng.choice(n, n_out, replace=False)
                arr[idx] = rng.choice([-1, 1], n_out) * 150
            s, Q, p = METS_single(arr)
            if np.sign(s) > 0:
                ok += 1
        print(f'噪声{ratio:.0%}: 方向正确 {ok/tot*100:.1f}%')

    arrays = []
    for _ in range(200):
        n = 100
        k = rng.uniform(-2, 2); scale = rng.uniform(10, 1000)
        t = np.arange(n) / (n - 1)
        arrays.append(scale * (1 + 0.5 * k * t) + rng.normal(0, 0.15 * scale, n))
    t0 = time.perf_counter()
    METS_rank(arrays)
    print(f'\n速度 (200序列, n=100): {(time.perf_counter()-t0)*1000:.1f}ms')
