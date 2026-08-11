# METS: Rank-Extreme Trend Sorting with Asymptotic Significance Testing

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**M**in-**M**ax-**E**xtreme **T**rend **S**orter — a lightweight algorithm for ranking many time series by overall trend, with honest, fully reproducible experimental evaluation.

## Why this exists

Ranking hundreds of time series (stocks, sensors, climate stations) by trend strength is a common task. Existing approaches face a trade-off:

- **OLS regression** — fast but fragile under outliers (a single spike flips the slope)
- **Theil-Sen** — robust (29.3% breakdown point) but O(n²) pairwise computation
- **Mann-Kendall** — significance only, no continuous ranking strength

METS explores a middle path: **rank-space extreme statistics** at O(n log n).

## Core idea

1. **Variance normalization** makes slopes dimensionless → cross-scale comparability
2. **Extreme positions in rank space** (`P_max - P_min`) → robust trend *direction* even under salt-and-pepper noise (each outlier affects only O(1/n) of the rank structure, unlike value-domain percentiles which get contaminated)
3. **Spearman asymptotic test** replaces the expensive permutation test → significance at O(1) per series
4. **BH multiple-comparison correction** controls false positives across many simultaneous tests
5. Quality assessment (MCR + pseudo-R²), multi-scale segmentation, and V-shape reversal penalty

## Honest results (fully reproducible, no cherry-picking)

### Real-world data (6 datasets, statsmodels)

| Dataset | #Series | Direction | MK agreement | vs Theil-Sen (ρ) |
|---------|---------|-----------|--------------|-------------------|
| macrodata (US economy) | 12 | 75.0% | **100%** | 0.795 |
| danish (DK economy) | 5 | 60.0% | **100%** | **0.975** |
| grunfeld (corporate) | 11 | **100%** | **100%** | 0.773 |
| elnino (sea temp) | 12 | **100%** | 75.0% | 0.329 |
| co2 annual (strong trend) | 42 | **100%** | 90.5% | 0.233 |
| sunspots (periodic) | 28 | 96.4% | **96.4%** | 0.722 |

**What METS is good at:**
- ✅ **Trend direction** — 75–100% consistency across all datasets (rank-space extrema work)
- ✅ **Significance testing** — 75–100% agreement with Mann-Kendall (asymptotic test is reliable)

**What METS is NOT good at (honestly reported):**
- ❌ **Fine magnitude ranking** — co2 annual slices: 0.233 vs OLS 0.966 (rank-space magnitude is coarse)
- ❌ **High-noise robustness** — degrades beyond ~10% salt-and-pepper (breakdown point ≪ Theil-Sen's 29.3%)
- ❌ **Speed** — full pipeline is ~9× slower than OLS (700ms vs 77ms for 200×100 series)
- ❌ **Nonlinear trends** — 0.77–0.86 vs OLS 0.89–0.91

> **TL;DR**: METS is a *direction-and-significance-first screening tool*, not a precision ranking tool. If you need fine magnitude ordering under heavy noise, use Theil-Sen. The value of this repo is the *honest* evaluation methodology — every number is reproducible.

## Quick start

```bash
pip install numpy scipy statsmodels matplotlib

# Single series score
python -c "
import numpy as np
from src.mets_v5 import METS_single
arr = np.linspace(0, 10, 30) * 100 + np.random.normal(0, 30, 30)
score, Q, p = METS_single(arr)
print(f'score={score:.3f} Q={Q:.3f} p={p:.3f}')"

# Rank many series (with BH correction)
python -c "
import numpy as np
from src.mets_v5 import METS_rank
arrays = [np.random.normal(0, 1, 50) for _ in range(10)]
order, scores, significant = METS_rank(arrays)
print(order, scores)"
```

## Reproduce all experiments

```bash
python src/run_experiments.py        # 7 synthetic experiments + figures
# real-data validation:
python src/run_real_data.py          # 6 real datasets (statsmodels built-in)
```

All random seeds are fixed; results match the paper tables exactly.

## Repository layout

```
src/mets_v5.py           # METS implementation (single + batch ranking)
src/run_experiments.py   # synthetic experiments 1-7
src/mets_validate2.py    # quick validation script
src/run_real_data.py     # real-data validation (needs statsmodels)
paper/paper_zh.md        # full paper (Chinese)
paper/METS_arxiv.tex     # English LaTeX version
figs/                    # all experimental figures
data/                    # raw results (JSON)
```

## License

Code: MIT. Paper: CC BY 4.0. AI-assisted work — the author used DeepSeek for experiment implementation and language polishing, and takes full responsibility for all content.
