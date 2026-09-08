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
| XQLFW 全量预检 | 7,263 张、6,000 对 | 已完成 CUDA 链路预检；仅 2,296 张图片、4 对通过现有质量门，正式分析在 Phase 5 | `data/logs/xqlfw_full_evaluation_report.json` |
| QMUL 官方协议 | 全部官方 MAT 标签和目录 | 已核验 | `data/logs/qmul_survface_protocol.json` |
| QMUL 默认门压力预检 | 60,294 张 Gallery | `min_face_size=112` 下无有效 Gallery；作为域限制证据，正式分析在 Phase 5 | 缓存位于 `data/logs/cache/qmul_survface.sqlite`，没有有效识别报告 |

LFW 全量 embedding 原运行使用实际 `CUDAExecutionProvider`。Phase 2 无阈值导出只读取 8,577 条有效和 4,656 条拒绝缓存记录，没有重新运行 InsightFace。

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

## 4. Phase 5 预留入口

XQLFW：

```bash
.venv/bin/python scripts/evaluate_xqlfw.py \
  --data-dir ./data \
  --dataset-dir ./data/datasets/xqlfw/lfw_original_imgs_min_qual0.85variant11 \
  --pairs ./data/datasets/xqlfw/xqlfw_pairs.txt \
  --min-face-size 80 \
  --cache-path ./data/logs/cache/xqlfw.sqlite \
  --report-output ./data/logs/xqlfw_full_evaluation_report.json
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

这些入口已存在，但正式 Phase 5 必须先根据数据域定义可解释的质量覆盖分析；不能把“几乎全部拒绝”后的极小有效分母当成鲁棒性准确率。

## 5. 结果解释规则

1. Gallery、Known Probe、Unknown Probe、Calibration、Evaluation 和最终 Demo Gallery 角色不得混用。
2. Calibration/Evaluation 在质量过滤前按来源身份划分，避免只挑选模型成功样本。
3. 无阈值 Rank-1 只评价 Known 第一候选排序；Unknown FPIR 必须在指定阈值工作点统计。
4. 聚合方法改变会改变分数分布，匹配阈值和候选分差必须与方法联合选择。
5. XQLFW 是 1:1 verification；QMUL 是监控域开放集压力集；两者不直接改写桌面应用参数。
6. 所有正式结论必须记录模型、质量规则、协议哈希、拒绝数量、有效分母和运行环境。
