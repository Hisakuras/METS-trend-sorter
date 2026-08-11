# Reddit 分享帖文案（英文）

---

## 帖子 1：r/MachineLearning（标题：展示项目 / 讨论方法论）

**Title:**
[P] METS — a trend ranking algorithm with honest, fully reproducible benchmarks (including where it FAILS)

**Body:**

I built a lightweight time-series trend ranking algorithm (O(n log n)) that ranks hundreds of series by trend strength using rank-space extreme statistics — direction from rank positions (robust to salt-and-pepper noise), significance from a Spearman asymptotic test with BH multiple-comparison correction.

But here's the part I actually want to share: **the honest benchmark**. I evaluated it against OLS and Theil-Sen on 6 real datasets (US/DK macroeconomics, Grunfeld corporate panel, sea surface temp, CO2, sunspots) plus synthetic noise experiments.

**What works:**
- Direction consistency: 75–100% across all 6 real datasets
- Significance agreement with Mann-Kendall: 75–100% (perfect on 3 datasets)

**What does NOT work (reported honestly in the README):**
- Fine magnitude ranking: 0.233 vs OLS 0.966 on CO2 annual slices
- High-noise robustness: breakdown point ≪ Theil-Sen's 29.3%
- Speed: full pipeline ~9× slower than OLS

TL;DR: it's a direction-and-significance-first screening tool, NOT a precision ranking tool. The repo is fully reproducible — every number in the tables comes from running the scripts, seeds fixed, no cherry-picking. I think the reproducibility methodology matters more than the algorithm itself.

Repo: https://github.com/Hisakuras/METS-trend-sorter

Happy to discuss the design choices (rank-space extrema vs value-domain percentiles, asymptotic test vs permutation test) — especially criticism on where the approach is wrong.

---

## 帖子 2：r/datasets（标题：数据集/基准）

**Title:**
[P] Reproducible benchmark harness for time-series trend ranking — 6 real datasets, fixed seeds, all scripts public

**Body:**

I made a fully reproducible benchmark for time-series trend ranking algorithms: OLS, Theil-Sen, and a new rank-extreme method (METS) evaluated on 6 real datasets from statsmodels:

- macrodata (US economy, 12 indicators, 1959–2009)
- danish_data (DK economy, 5 log-scale variables)
- grunfeld (11 firms × 20 years corporate investment)
- elnino (12 monthly sea-surface temperature series, 1950–2010)
- co2 (42 annual slices of CO2 concentration since 1958)
- sunspots (28 eleven-year cycle windows since 1700)

All experiments run with fixed random seeds; results are stored as JSON in the repo. You can regenerate every table with two commands. The README includes a honest limitations section — including where the proposed method loses to Theil-Sen.

I think this is useful as a starting point if you're evaluating trend/ranking methods on real data and want a clean, citable baseline setup.

Repo: https://github.com/Hisakuras/METS-trend-sorter

---

## 发帖建议

| Subreddit | 用哪版 | 注意 |
|-----------|--------|------|
| r/MachineLearning | 帖子 1 | 需要 [P] 前缀（项目）；新账号发帖可能需 karma，先在子版块评论活跃几天 |
| r/datasets | 帖子 2 | 门槛较低，适合数据基准话题 |
| r/statistics | 帖子 1 改编 | 强调显著性检验/MK 对照，受众是统计学家 |

## 补充动作（可选，涨星效果好）

1. **GitHub Discussions 开帖**：repo 里开启 Discussions 方便讨论
2. **Twitter/X 发一条**：附 README 截图 + repo 链接
3. **LinkedIn**：写"我做了个算法并诚实公布了它的失败"——这种内容反而传播好
