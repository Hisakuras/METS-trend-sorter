import numpy as np
import sys
sys.path.insert(0, '/tmp')
from mets_final import METS_rank
from scipy.stats import spearmanr

rng = np.random.default_rng(42)
true_slopes, arrays = [], []

for _ in range(100):
    n = int(rng.integers(10, 60))
    k = rng.uniform(-2, 2)
    scale = rng.uniform(10, 1000)
    t = np.arange(n) / (n - 1)          # 归一化时间 [0,1]
    # 线性趋势：终点偏移 = k * scale（相对幅度可控）
    signal = scale * (1 + 0.5 * k * t)
    noise_std = 0.15 * scale            # SNR 适中
    arr = signal + rng.normal(0, noise_std, n)
    true_slopes.append(k)
    arrays.append(arr)

order, scores, sig = METS_rank(arrays, B=500, seed=1)
rho, p = spearmanr(scores, true_slopes)
print(f'Spearman rho (排序分 vs 真实斜率): {rho:.4f}  (p={p:.2e})')

mask = np.abs(true_slopes) > 0.2
correct = np.sum(np.sign(scores[mask]) == np.sign(np.array(true_slopes)[mask])) / mask.sum()
print(f'方向正确率(斜率>0.2): {correct*100:.1f}%')

strong = np.abs(true_slopes) > 1.0
weak = np.abs(true_slopes) < 0.3
if weak.sum() > 0 and strong.sum() > 0:
    print(f'强趋势平均|得分|: {np.abs(scores[strong]).mean():.4f} ({strong.sum()}条)')
    print(f'弱趋势平均|得分|: {np.abs(scores[weak]).mean():.4f} ({weak.sum()}条)')
    print(f'弱趋势显著比例: {sig[weak].mean()*100:.1f}% (应低)')
    print(f'强趋势显著比例: {sig[strong].mean()*100:.1f}% (应高)')
