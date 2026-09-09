# Phase 5 XQLFW 跨质量实验结果

## 结论

本实验已覆盖 XQLFW 官方 6,000 对及其引用的 7,263 张去重图片。使用 InsightFace `buffalo_l` 的原始 embedding、官方 10 折协议和数据集随附逐图质量分，不使用桌面应用的启发式质量门或初始阈值。

在两种图像域都成功提取的同一批 5,871 对上：

| 图像域 | 验证准确率 |
| --- | ---: |
| 原始 LFW deep-funneled | 98.48% |
| XQLFW 跨质量变体 | 94.14% |
| 变化 | **-4.34 个百分点** |

跨质量退化主要压低同一人的相似度：同人 Pair 均值从 `0.6562` 降到 `0.3968`，下降 `0.2594`；异人 Pair 均值只从 `0.0052` 变为 `0.0070`。因此不能用“所有相似度整体平移”解释质量影响，也不能把普通 LFW 阈值直接复用到跨质量图片。

## 协议与实现

- pairs：官方 `xqlfw_pairs.txt`，10 折，每折原始设计为 300 对同人和 300 对异人；
- 质量：官方 `xqlfw_scores.txt`，只用于结果分桶，不参与 embedding 加权或拒绝；
- 阈值：每轮只用其余 9 折的实际相似度断点选择准确率最高的阈值，同效时优先更少异人误接收，再应用到当前 1 折；
- 主体脸：数据集已有身份标签，检测到多个候选时选择面积最大的主体脸；
- 缓存：`RawEmbeddingCache` 不包含质量门或验证阈值，评测阶段不初始化人脸模型；
- 后端：ONNX Runtime session 优先选择 `CUDAExecutionProvider`，并保留
  `CPUExecutionProvider` 作为不支持算子的回退；首次 7,263 张提取用时约 4 分钟。

该 1:1 verification 实验回答“同一模型面对跨质量图片时验证性能如何变化”，不比较本项目 1:N 的人员聚合方法，也不直接设置桌面应用阈值。

## 全量覆盖

| 项目 | 原始 LFW | XQLFW |
| --- | ---: | ---: |
| 协议去重图片 | 7,263 | 7,263 |
| 有效 embedding | 7,239 | 7,200 |
| 模型 FTE | 24 | 63 |
| 官方 Pair | 6,000 | 6,000 |
| 有效 Pair | 5,961 | 5,894 |
| 同人有效 Pair | 2,976 | 2,940 |
| 异人有效 Pair | 2,985 | 2,954 |
| 同人正确 | 2,886 | 2,661 |
| 异人正确 | 2,984 | 2,889 |
| 各自有效集准确率 | 98.47% | 94.16% |
| 10 折准确率标准差 | 0.48% | 0.97% |
| 平均特征提取耗时 | 32.81 ms/图 | 32.29 ms/图 |

两边有效集略有不同，因此主要退化结论使用 5,871 个共同有效 Pair，而不是直接相减两组不同分母的准确率。

## 推理子阶段优化实验

原始 `FaceAnalysis` 在每张图上除检测和识别外，还执行了本项目未使用的性别年龄、
二维关键点和三维关键点模型。当前实现通过 `allowed_modules` 只执行
`detection` 与 `recognition`；检测器自身输出的五点关键点仍用于识别模型对齐，
所以没有删掉人脸对齐步骤。

为避免用不同样本或缓存命中制造虚假加速，优化后使用独立空缓存重新处理完全相同的
7,263 张 XQLFW 图片，并与优化前缓存逐条连接比较：

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| 有效 embedding 平均耗时 | 32.293 ms/图 | 23.896 ms/图 | **-26.0%** |
| 完整提取墙钟时间 | 239.93 s | 179.45 s | **-25.2%** |
| 有效 embedding | 7,200 | 7,200 | 0 |
| 模型 FTE | 63 | 63 | 0 |
| 官方有效 Pair | 5,894 | 5,894 | 0 |
| 10 折验证准确率 | 94.1636% | 94.1636% | 0 |

两份缓存的 7,263 条状态、文件哈希、人脸数量、质量指标及 embedding BLOB
逐项比较均为零差异。这证明该改动减少了未使用模型的逐图计算，没有改变当前检测、
对齐、embedding 或验证结果。这里测得的是**人脸模型推理子阶段**，不是包含摄像头
取帧、数据库检索、日志和 UI 刷新的完整 E2E；E2E 仍需在桌面应用链路单独计时。

## 共同 Pair 的错误变化

| 状态变化 | Pair 数 |
| --- | ---: |
| 两边都正确 | 5,510 |
| 原始正确、XQLFW 错误 | 272 |
| 原始错误、XQLFW 正确 | 17 |
| 两边都错误 | 72 |

XQLFW 使 272 个原本正确的 Pair 变错，同时有 17 个原本错误的 Pair 变对，净减少 255 个正确结果。这比只展示最终准确率更直接地说明退化发生在同一批样本上。

## 相似度分布变化

| Pair 类型 | 原始均值 | XQLFW 均值 | 均值变化 | 原始中位数 | XQLFW 中位数 | 中位数变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同人 | 0.6562 | 0.3968 | -0.2594 | 0.6785 | 0.4167 | -0.2618 |
| 异人 | 0.0052 | 0.0070 | +0.0018 | 0.0046 | 0.0050 | +0.0003 |

XQLFW 的 10 个训练折阈值位于 `0.1331～0.1416`，原始 LFW 同协议阈值位于 `0.2262～0.2326`。阈值变化来自同人分数分布压缩，不能解释为模型突然把异人都判得更相似。

## 官方质量分的分桶证据

以一对图片中的较低官方质量分分桶：

| 最低质量分 | 共同 Pair | 原始准确率 | XQLFW 准确率 | 变化 |
| --- | ---: | ---: | ---: | ---: |
| `[0.00, 0.40)` | 1,962 | 99.03% | 93.88% | -5.15 pp |
| `[0.40, 0.60)` | 3,719 | 98.17% | 94.19% | -3.98 pp |
| `[0.60, 0.80)` | 190 | 98.95% | 95.79% | -3.16 pp |

以两张图片官方质量分的绝对差分桶：

| 质量分差 | 共同 Pair | 原始准确率 | XQLFW 准确率 | 变化 |
| --- | ---: | ---: | ---: | ---: |
| `[0.10, 0.20)` | 777 | 99.10% | 95.75% | -3.35 pp |
| `[0.20, 0.30)` | 1,763 | 98.81% | 93.70% | -5.10 pp |
| `[0.30, 0.50)` | 2,953 | 98.07% | 94.11% | -3.96 pp |
| `[0.50, 1.00]` | 378 | 98.94% | 93.12% | -5.82 pp |

最低质量更低、质量差更大时总体下降更明显，但分桶结果不是严格单调函数，不能据此发明一个线性质量权重。它支持的结论是“跨质量确实造成可测退化”，而不是“当前启发式 quality_score 已经可以修复退化”。

## 当前边界与后续用途

- XQLFW 证明了单帧低质量会压低同人相似度，但没有证明多帧择优一定提高多少；多帧扩展仍需摄像头同条件 A/B 复测。
- 本实验不包含活体攻击、极暗环境、严重运动模糊恢复、极端遮挡或支付级安全。
- XQLFW 的 1:1 折次阈值不能写入 Phase 4 的 1:N 开放集配置。
- 官方质量分仅用于解释结果，不取代 Phase 3 已否定的启发式质量加权。

## 本地产物与复现

本地产物均位于被 Git 忽略的 `data/`：

- `data/logs/cache/xqlfw_raw.sqlite`；
- `data/experiments/phase5/xqlfw_raw_extraction_manifest.json`；
- `data/experiments/phase5/xqlfw_evaluation_report.json`；
- `data/experiments/phase5/lfw_xqlfw_pairs_baseline_report.json`；
- `data/experiments/phase5/xqlfw_domain_comparison.json`。
- `data/logs/cache/xqlfw_raw_optimized.sqlite`；
- `data/experiments/phase5/xqlfw_optimized_extraction_manifest.json`；
- `data/experiments/phase5/xqlfw_optimized_evaluation_report.json`。

```bash
.venv/bin/python scripts/extract_xqlfw_raw_embeddings.py --data-dir ./data
.venv/bin/python scripts/evaluate_xqlfw.py --data-dir ./data
.venv/bin/python scripts/evaluate_xqlfw.py \
  --data-dir ./data \
  --dataset-dir ./data/datasets/lfw_funneled \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --cache-dataset-id lfw-natural-v1 \
  --report-output ./data/experiments/phase5/lfw_xqlfw_pairs_baseline_report.json
.venv/bin/python scripts/compare_xqlfw_domains.py
```
