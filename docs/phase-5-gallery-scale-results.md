# Phase 5B 小 Gallery 稳定性复核

## 结论

本实验在查看独立 Evaluation 前，把跨质量 Calibration 已冻结的部署候选原样重放到 3、5、10、25、50、100 人 Gallery。每档使用 10 个固定种子身份子集，并覆盖自然/混合 Gallery 与自然/XQLFW Probe 四个主要场景。

```text
聚合：Mean Prototype
规则：最高人员分数阈值
阈值：0.5557855367660522
数据分区：只使用 Calibration
参数重选：无
```

240 个结果全部满足 `FPIR_valid≤1%`。3～50 人的四场景 Unknown FPIR 均为 0；100 人场景的最高观测 FPIR 为 0.0814%。因此从 4,599 人完整 Gallery 标定的最高分阈值迁移到课设的少量身份库时没有出现风险放大。

## 为什么小库需要单独检查

最高分阈值、候选分差和 NAC 对候选身份数量的敏感性不同：

- 最高分阈值只看最佳候选的绝对相似度；Gallery 变小通常使 Unknown 能碰到的最高分下降；
- 候选分差依赖第一、第二名，候选越少，第二名可能明显变弱；
- NAC 的分母直接由前 K 个身份组成，小 Gallery 中实际使用 `min(K, 身份数)`，分数分布会变化。

本轮最终选中的是最高分阈值，但仍执行规模复核，以证明现场 3 人以上 Gallery 与大规模数据标定之间的迁移方向。最终 Demo Gallery 不参与这个实验，也不会被拿来重新选阈值。

## 汇总结果

| Gallery 身份 | 四场景最大 FPIR | 自然库→自然 Probe TPIR 均值 | 混合库→自然 Probe TPIR 均值 | 自然库→XQLFW Probe TPIR 均值 | 混合库→XQLFW Probe TPIR 均值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0% | 95.98% | 96.46% | 69.76% | 73.26% |
| 5 | 0% | 96.30% | 96.43% | 69.31% | 71.42% |
| 10 | 0% | 97.27% | 97.16% | 68.31% | 70.68% |
| 25 | 0% | 97.00% | 96.55% | 67.93% | 70.60% |
| 50 | 0% | 97.08% | 96.33% | 67.79% | 70.09% |
| 100 | 0.0814% | 97.25% | 96.73% | 67.04% | 69.50% |

身份数少时，某次随机子集包含的 Probe 数量和身份难度差异较大，因此 3 人档单次 TPIR 范围较宽；这正是不能用“本人 + 两个公开身份”直接标定参数的原因。完整逐次结果保存在本地 `data/experiments/phase5b/gallery_scale_report.json`。

## 复现

```bash
.venv/bin/python scripts/evaluate_gallery_scale.py \
  --data-dir ./data \
  --protocol ./data/experiments/phase4/protocol.json \
  --natural-cache ./data/logs/cache/lfw_raw.sqlite \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite \
  --calibration-report ./data/experiments/phase5b/calibration_report.json \
  --gallery-sizes 3 5 10 25 50 100 \
  --repeats 10 \
  --seed 2026 \
  --target-fpir 0.01
```

脚本只读取两个 embedding 缓存和 Calibration 报告，不初始化 InsightFace，也不读取 Evaluation 标签。完整算法对比、独立 Evaluation 和置信区间见 [Phase 5B 联合标定结果](phase-5b-cross-quality-results.md)。
