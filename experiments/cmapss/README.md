# NASA C-MAPSS FD001 验证

## 数据来源

主要来源：NASA Open Data 的 [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)。

本次运行使用公开镜像中的 FD001 文件，以避免把 NASA 原始数据重新提交到仓库：

`https://github.com/huster123/c-mapss-full-dataset-/tree/master/Data`

运行 `python download_cmapss.py` 下载 `train_FD001.txt`、`test_FD001.txt` 和 `RUL_FD001.txt`，脚本会输出文件大小和 SHA-256。

## 实验任务

这是“早期退化传感器筛选”任务，不是 RUL 预测任务。每台发动机在第 50 或第 80 个周期截断，用训练折内传感器 MAD 标准化，按发动机做 3 折交叉验证。方法包括 OLS、全局 Theil–Sen 和 METS-SI。METS-SI 的结构宽度为 5 个重叠局部窗口斜率的 Q90−Q10。

测试目标是：使用训练折中选出的 Top-K 传感器趋势强度，对测试发动机的剩余寿命风险进行排序。指标是 Spearman、Kendall 和结构区间宽度。

## 运行

```bash
python download_cmapss.py
python run_cmapss_validation.py
```

结果写入 `cmapss_results/`。当前结果显示，METS-SI 在早期 50 周期窗口的 Top-3 和 Top-10 筛选上优于 OLS，但在 80 周期窗口的 Top-10 上低于 Theil–Sen。因此，不得据此声称 METS-SI 全面优于 Theil–Sen。

