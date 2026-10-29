# 数据集实验与参数依据

这里保存数据集协议、特征提取缓存、人员聚合、接收规则比较和性能对照代码。桌面程序的安装与操作见[项目说明](../README.md)。

## 目录与运行准备

- `face_evaluation/`：数据集协议、候选分数、参数搜索和统计模块。
- `commands/`：数据准备、特征提取、效果评价和性能实验命令。
- `tests/`：实验协议与统计公式的自动化测试。
- 实验结论保存在 `docs/experiments/`，图片、缓存和原始产物保存在 `data/`。

先按项目说明安装 CPU 或 GPU 后端，再增加开发与实验依赖：

```bash
.venv/bin/python -m pip install -e ".[dev,evaluation]"
```

以下命令从项目根目录执行，使用现有本地模型和相应数据集。数据集实验在 Linux 上执行；Windows 的解释器路径为 `.\.venv\Scripts\python.exe`。

## 数据集准备

提交包保留实验协议、XQLFW 图片配对与质量标签、QMUL-SurvFace 的 MAT 标签，以及已有实验分数和报告；不包含原始图片、下载归档和特征提取缓存。模型也需先按项目说明准备。查看已有结果无需下载图片或执行模型。

- LFW：运行下方源协议命令时加入 `--download`，脚本会下载并校验图片，解压到 `data/datasets/lfw_funneled/`。
- XQLFW：从[官方下载页](https://martlgap.github.io/xqlfw/pages/download.html)下载质量变化图片，解压后使图片目录为 `data/datasets/xqlfw/lfw_original_imgs_min_qual0.85variant11/`。包中保留 `xqlfw_pairs.txt` 和 `xqlfw_scores.txt`。
- QMUL-SurvFace：从[项目页](https://github.com/QMUL-SurvFace/qmul-survface.github.io)获取数据，解压到 `data/datasets/qmul-survface/QMUL-SurvFace/`，保留官方目录结构。

数据来源、规模和用途见[数据集说明](../docs/experiments/dataset-evaluation.md)。下方提取命令会重建缓存；候选分数和已有报告仍保留在 `data/experiments/`。如需保留多次实验结果，使用不同的输出目录。

## 结果与开发记录

| 内容 | 记录 |
| --- | --- |
| 数据集组成与协议 | [数据集说明](../docs/experiments/dataset-evaluation.md) |
| 质量指标与拒绝代价 | [质量实验](../docs/experiments/phase-3-results.md) |
| 自然图像上的早期规则比较 | [自然 LFW 实验](../docs/experiments/phase-4-results.md) |
| 最终人员聚合与跨质量接收规则 | [跨质量比较](../docs/experiments/phase-5b-cross-quality-results.md) |
| 矩阵检索和多帧方法 | [矩阵与多帧实验](../docs/experiments/phase-5c-results.md) |
| 采集、检测与特征提取的时间优化 | [五帧流程对照](../docs/experiments/phase-5d-results.md) |
| 各阶段任务和方案变化 | [开发阶段记录](../docs/experiments/development-history.md) |

参数选择集用于比较方法与选择阈值，评价集按待识别身份与其分开。评价划分延续了早期实验，早期结果曾用于开发分析；最终部署参数由参数选择集确定，冻结后再统计评价结果。最终演示标准库不参与选参。

## 复现命令

实验保存原始特征，再生成尚未应用接收阈值的候选分数，随后比较聚合方法与判定规则。调整判定参数时可以直接复用候选结果，不必重复执行人脸模型。

LFW 全量协议与自然图片原始缓存：

```bash
.venv/bin/python -m experiments.commands.prepare_lfw --data-dir ./data --download --full \
  --known-fraction 0.8 --enrollment-per-identity 5 --seed 2026 \
  --output ./data/datasets/lfw_full_open_set_protocol.json
.venv/bin/python -m experiments.commands.extract_lfw_raw_embeddings --data-dir ./data
.venv/bin/python -m experiments.commands.export_lfw_decision_scores \
  --data-dir ./data \
  --source-protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --output-dir ./data/experiments/phase4
```

跨质量实验使用完整 XQLFW 同路径变体，先在参数选择集和小型标准库中比较候选方案，再固定参数并统计评价集结果：

```bash
.venv/bin/python -m experiments.commands.extract_xqlfw_raw_embeddings --data-dir ./data \
  --protocol ./data/experiments/phase4/protocol.json \
  --cache-path ./data/logs/cache/xqlfw_full_cuda.sqlite \
  --manifest ./data/experiments/phase5b/xqlfw_full_cuda_manifest.json
.venv/bin/python -m experiments.commands.export_cross_quality_scores --data-dir ./data \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite
.venv/bin/python -m experiments.commands.calibrate_cross_quality \
  --scores ./data/experiments/phase5b/decision_scores.sqlite
.venv/bin/python -m experiments.commands.evaluate_gallery_scale --data-dir ./data \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite
.venv/bin/python -m experiments.commands.evaluate_cross_quality_policy \
  --scores ./data/experiments/phase5b/decision_scores.sqlite \
  --calibration-report ./data/experiments/phase5b/calibration_report.json
```

XQLFW 按官方 6,000 对图片进行一对一验证；QMUL-SurvFace 用于监控小脸场景的模型适用性分析：

```bash
.venv/bin/python -m experiments.commands.extract_xqlfw_raw_embeddings --data-dir ./data
.venv/bin/python -m experiments.commands.evaluate_xqlfw --data-dir ./data
.venv/bin/python -m experiments.commands.compare_xqlfw_domains
.venv/bin/python -m experiments.commands.prepare_qmul --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace
.venv/bin/python -m experiments.commands.extract_qmul_raw_embeddings --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace
```

## 输出文件与重复运行

各命令的缓存和报告路径可通过参数指定。原始特征缓存按数据集、提取配置和图片路径复用；图片内容变化时更新对应项。JSON 报告先写临时文件，再替换指定目标，重复写入同一路径会覆盖旧报告，需要保留不同实验时应使用不同输出位置。

`data/logs/cache/` 是实验特征缓存，`data/experiments/` 保存各阶段分数和结果；桌面识别记录另存于 `data/face_library.sqlite`，不由上述实验报告代替。原始实验结果保留各轮输入、配置和版本信息，历史参数按对应轮次解释。
