# 数据集评测记录

本文记录各数据集在项目中的职责、已完成产物和限制。单元测试通过数不代表识别效果，小型 pilot 也不替代完整实验。

## 1. 数据集职责

| 数据集 | 项目用途 | 已有规模 | 不负责回答的问题 |
| --- | --- | --- | --- |
| LFW deep-funneled | Phase 2 固定开放集 1:N Gallery/Probe、保存无阈值分数；Phase 3 质量退化实验；Phase 4 联合标定 | 13,233 张图片 | 不代表桌面摄像头的设备、距离和现场光照域 |
| XQLFW | Phase 5A 真实跨质量 1:1 验证；Phase 5B 与 LFW 同路径构造跨质量开放集 1:N 场景 | 官方 6,000 对涉及 7,263 张；完整变体目录对应 LFW 13,233 张 | 不单独产生一套 XQLFW 专用部署阈值 |
| QMUL-SurvFace | Phase 5 远距离、小脸和监控域覆盖压力分析 | Gallery 60,294，Mated Probe 60,423，Unmated Probe 121,736 | 不等价于普通 UVC 摄像头，不用于降低桌面质量门 |
| 外置摄像头开发样本 | Phase 4 后复核桌面域，Phase 5 多帧与 E2E 实验 | 按场景现场采集 | 不与最终 Demo Gallery 混用，不替代公开数据集的固定协议 |

本地位置：

- LFW：`data/datasets/lfw_funneled/`；
- XQLFW：`data/datasets/xqlfw/`；
- QMUL-SurvFace：`data/datasets/qmul-survface/QMUL-SurvFace/`。

数据来源：

- [LFW deep-funneled 下载源](https://ndownloader.figshare.com/files/5976015)
- [XQLFW 官方下载页](https://martlgap.github.io/xqlfw/pages/download.html)
- [QMUL-SurvFace 项目页](https://github.com/QMUL-SurvFace/qmul-survface.github.io)

数据集、模型、缓存和逐样本结果全部位于被 Git 忽略的 `data/`。

## 2. 当前执行状态

| 任务 | 覆盖 | 状态 | 本地产物 |
| --- | --- | --- | --- |
| LFW pilot | 7 张有效 Probe | 已完成；仅作真实模型链路烟雾测试 | `data/logs/phase2_algorithm_baseline.json` |
| LFW 固定阈值历史诊断 | Gallery 7,490、Probe 5,743 | 已完成；结果依赖初始阈值，不作最终方法结论 | `data/logs/lfw_full_algorithm_baseline.json` |
| LFW Phase 2 无阈值结果 | 有效 Gallery 4,735；有效 Probe 3,842；六方法 23,052 条 | 已完成；身份互斥分区和拒绝项已保存 | `data/experiments/phase2/` |
| LFW Phase 4 自然原始结果 | 有效 Gallery 7,465；有效 Probe 5,720；六方法 34,320 条 | 已完成；无质量预筛，已完成联合标定和一次独立 Evaluation | `data/experiments/phase4/` |
| LFW/XQLFW 跨质量开放集联合标定 | 六场景、六种聚合、四类接收规则 | 已完成；主工作点选择 Mean Prototype + 最高分阈值 `0.555786` | `data/experiments/phase5b/` |
| 小 Gallery 稳定性复核 | 3/5/10/25/50/100 人，每档 10 次、四场景 | 已完成 Calibration-only 固定参数重放；240 个结果均满足 1% FPIR | `data/experiments/phase5b/gallery_scale_report.json` |
| 跨质量独立 Evaluation | 六场景、三个预先冻结参考工作点 | 已执行一次；主规则最差 TPIR 66.30%、最坏 FPIR 1.43%，含身份 bootstrap 95% 区间 | `data/experiments/phase5b/evaluation_report.json` |
| XQLFW 全量跨质量实验 | 7,263 张、6,000 对；有效 7,200 张、5,894 对 | 已完成 CUDA 原始提取、官方 10 折验证及与原始 LFW 的 5,871 个共同 Pair 对比 | `data/experiments/phase5/` |
| XQLFW 推理子阶段优化 | 同一 7,263 张图片、独立空缓存 | 只执行检测与识别后，有效 embedding 平均耗时下降 26.0%；embedding 与验证结果不变 | `data/experiments/phase5/xqlfw_optimized_*` |
| QMUL 官方协议 | 全部官方 MAT 标签和目录 | 已核验 | `data/logs/qmul_survface_protocol.json` |
| QMUL 默认门压力预检 | 60,294 张 Gallery | `min_face_size=112` 下无有效 Gallery；作为域限制证据，正式分析在 Phase 5 | 缓存位于 `data/logs/cache/qmul_survface.sqlite`，没有有效识别报告 |
| QMUL 全量原始压力实验 | 242,453 张；436 张有效 embedding | 已完成无质量门原始提取；70/3,000 个 Gallery 身份可用，Mated 仅 4 张可评分且 Rank-1 为 0 | `data/experiments/phase5/qmul_*` |

LFW Phase 4 原始提取使用实际 `CUDAExecutionProvider`，13,233 张中 13,185 张获得 embedding、48 张 FTE。再次读取同一缓存时 13,233 张全部命中且模型推理为 0；聚合或判定参数变化不再触发 InsightFace。

## 3. Phase 2 可复现命令

### 3.1 生成覆盖全图的源协议

```bash
.venv/bin/python scripts/prepare_lfw.py --data-dir ./data --full \
  --known-fraction 0.8 --enrollment-per-identity 5 --seed 2026 \
  --output ./data/datasets/lfw_full_open_set_protocol.json
```

### 3.2 首次提取 embedding

只有模型、检测、对齐或质量规则变化导致缓存不可复用时，才运行：

```bash
.venv/bin/python scripts/evaluate_lfw.py --data-dir ./data --stream \
  --protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --min-face-size 80 \
  --cache-path ./data/logs/cache/lfw.sqlite \
  --report-output ./data/logs/lfw_full_algorithm_baseline.json
```

评测专用 `min_face_size=80` 不会写回应用正式配置。

### 3.3 从缓存导出固定分区和无阈值分数

```bash
.venv/bin/python scripts/export_lfw_decision_scores.py \
  --data-dir ./data \
  --source-protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --cache-path ./data/logs/cache/lfw.sqlite \
  --output-dir ./data/experiments/phase2 \
  --min-face-size 80
```

缓存只有一个批次时脚本自动选择；有多个批次时必须用 `--cache-extraction-id` 明确指定。脚本不会导入 `FaceEngine`，缓存缺失或图片 SHA-256 改变时直接失败，不会悄悄重新推理。

### 3.4 Phase 4 自然 LFW 与联合标定

```bash
.venv/bin/python scripts/extract_lfw_raw_embeddings.py --data-dir ./data
.venv/bin/python scripts/export_lfw_decision_scores.py \
  --data-dir ./data \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --output-dir ./data/experiments/phase4
.venv/bin/python scripts/calibrate_thresholds.py \
  --scores ./data/experiments/phase4/decision_scores.sqlite
.venv/bin/python scripts/evaluate_selected_operating_point.py \
  --scores ./data/experiments/phase4/decision_scores.sqlite \
  --target-fpir 0.003
```

前两个 Phase 2 入口保留为历史链路；正式聚合结论以 [Phase 4 结果](phase-4-results.md) 为准。

## 4. Phase 5 评测入口

XQLFW：

```bash
.venv/bin/python scripts/extract_xqlfw_raw_embeddings.py --data-dir ./data
.venv/bin/python scripts/evaluate_xqlfw.py --data-dir ./data
.venv/bin/python scripts/compare_xqlfw_domains.py
```

QMUL-SurvFace：

```bash
.venv/bin/python scripts/prepare_qmul.py \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace \
  --output ./data/logs/qmul_survface_protocol.json

.venv/bin/python scripts/extract_qmul_raw_embeddings.py \
  --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace

.venv/bin/python scripts/evaluate_qmul.py \
  --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace \
  --cache-path ./data/logs/cache/qmul_survface_raw.sqlite \
  --transfer-policy ./data/experiments/phase4/final_evaluation.json
```

正式结果见 [Phase 5 XQLFW 结果](phase-5-xqlfw-results.md)、
[Phase 5 QMUL 结果](phase-5-qmul-results.md)和
[Phase 5B 联合标定结果](phase-5b-cross-quality-results.md)。QMUL 的有效分母极小，因此不能把
LFW 工作点产生的“FPIR=0”解释成鲁棒性准确率；它同时发生 TPIR=0 的全拒绝。

Phase 5B 跨质量开放集：

```bash
.venv/bin/python scripts/extract_xqlfw_raw_embeddings.py \
  --data-dir ./data \
  --protocol ./data/experiments/phase4/protocol.json \
  --cache-path ./data/logs/cache/xqlfw_full_cuda.sqlite \
  --manifest ./data/experiments/phase5b/xqlfw_full_cuda_manifest.json
.venv/bin/python scripts/export_cross_quality_scores.py --data-dir ./data \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite
.venv/bin/python scripts/calibrate_cross_quality.py \
  --scores ./data/experiments/phase5b/decision_scores.sqlite
.venv/bin/python scripts/evaluate_gallery_scale.py --data-dir ./data \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite
.venv/bin/python scripts/evaluate_cross_quality_policy.py \
  --scores ./data/experiments/phase5b/decision_scores.sqlite \
  --calibration-report ./data/experiments/phase5b/calibration_report.json
```

## 5. 结果解释规则

1. Gallery、Known Probe、Unknown Probe、Calibration、Evaluation 和最终 Demo Gallery 角色不得混用。
2. Calibration/Evaluation 在质量过滤前按来源身份划分，避免只挑选模型成功样本。
3. 无阈值 Rank-1 只评价 Known 第一候选排序；Unknown FPIR 必须在指定阈值工作点统计。
4. 聚合方法改变会改变分数分布，匹配阈值和候选分差必须与方法联合选择。
5. XQLFW 官方 Pair 是 1:1 verification；Phase 5B 只利用其与 LFW 同身份、同路径的图像变体构造 1:N 跨质量场景。QMUL 仍只作为监控域压力集。
6. 所有正式结论必须记录模型、质量规则、协议哈希、拒绝数量、有效分母和运行环境。
