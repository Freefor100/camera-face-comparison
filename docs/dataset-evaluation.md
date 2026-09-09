# 数据集评测记录

本文记录各数据集在项目中的职责、已完成产物和限制。单元测试通过数不代表识别效果，小型 pilot 也不替代完整实验。

## 1. 数据集职责

| 数据集 | 项目用途 | 已有规模 | 不负责回答的问题 |
| --- | --- | --- | --- |
| LFW deep-funneled | Phase 2 固定开放集 1:N Gallery/Probe、保存无阈值分数；Phase 3 质量退化实验；Phase 4 联合标定 | 13,233 张图片 | 不代表桌面摄像头的设备、距离和现场光照域 |
| XQLFW | Phase 5 真实跨质量 1:1 验证压力分析 | 官方 6,000 对，涉及 7,263 张去重图片 | 不用于比较 1:N 人员聚合，也不直接决定桌面阈值 |
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
| XQLFW 全量跨质量实验 | 7,263 张、6,000 对；有效 7,200 张、5,894 对 | 已完成 CUDA 原始提取、官方 10 折验证及与原始 LFW 的 5,871 个共同 Pair 对比 | `data/experiments/phase5/` |
| QMUL 官方协议 | 全部官方 MAT 标签和目录 | 已核验 | `data/logs/qmul_survface_protocol.json` |
| QMUL 默认门压力预检 | 60,294 张 Gallery | `min_face_size=112` 下无有效 Gallery；作为域限制证据，正式分析在 Phase 5 | 缓存位于 `data/logs/cache/qmul_survface.sqlite`，没有有效识别报告 |

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

## 4. Phase 5 预留入口

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

.venv/bin/python scripts/evaluate_qmul.py \
  --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace \
  --cache-path ./data/logs/cache/qmul_survface.sqlite \
  --report-output ./data/logs/qmul_survface_evaluation_report.json
```

XQLFW 正式结果见 [Phase 5 XQLFW 结果](phase-5-xqlfw-results.md)。QMUL 正式分析仍必须先根据数据域定义可解释的模型覆盖，不能把“几乎全部拒绝”后的极小有效分母当成鲁棒性准确率。

## 5. 结果解释规则

1. Gallery、Known Probe、Unknown Probe、Calibration、Evaluation 和最终 Demo Gallery 角色不得混用。
2. Calibration/Evaluation 在质量过滤前按来源身份划分，避免只挑选模型成功样本。
3. 无阈值 Rank-1 只评价 Known 第一候选排序；Unknown FPIR 必须在指定阈值工作点统计。
4. 聚合方法改变会改变分数分布，匹配阈值和候选分差必须与方法联合选择。
5. XQLFW 是 1:1 verification；QMUL 是监控域开放集压力集；两者不直接改写桌面应用参数。
6. 所有正式结论必须记录模型、质量规则、协议哈希、拒绝数量、有效分母和运行环境。
