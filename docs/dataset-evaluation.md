# 数据集评测记录

本文记录各数据集在项目中的职责、已完成产物和限制。单元测试通过数不代表识别效果，小规模试运行也不替代完整实验。
数据角色和指标定义见 [术语与内部命名](术语与内部命名.md)。

## 1. 数据集职责

| 数据集 | 项目用途 | 已有规模 | 不负责回答的问题 |
| --- | --- | --- | --- |
| LFW deep-funneled | 阶段 2 固定开放集 1:N 标准库/待识别图片、保存未应用接收规则的候选分数；阶段 3 质量退化实验；阶段 4 联合标定 | 13,233 张图片 | 不代表桌面摄像头的设备、距离和现场光照域 |
| XQLFW | 阶段 5A 真实跨质量 1:1 验证；阶段 5B 与 LFW 同路径构造跨质量开放集 1:N 场景 | 官方 6,000 对涉及 7,263 张；完整变体目录对应 LFW 13,233 张 | 不单独产生一套 XQLFW 专用部署阈值 |
| QMUL-SurvFace | 阶段 5 远距离、小脸和监控域覆盖压力分析 | 标准库 60,294，同源待识别图片 60,423，非同源待识别图片 121,736 | 不等价于普通 UVC 摄像头，不用于设置桌面识别参数 |
| 外置摄像头开发样本 | 阶段 4 后复核桌面域，阶段 5 多帧与端到端实验 | 按场景现场采集 | 不与最终演示标准库混用，不替代公开数据集的固定协议 |

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
| LFW 小规模试运行 | 7 张有效待识别图片 | 已完成；仅作真实模型链路连通性测试 | `data/logs/phase2_algorithm_baseline.json` |
| LFW 固定阈值历史诊断 | 标准库 7,490、待识别图片 5,743 | 已完成；结果依赖初始阈值，不作最终方法结论 | `data/logs/lfw_full_algorithm_baseline.json` |
| LFW 阶段 2 连续候选分数 | 有效标准库 4,735；有效待识别图片 3,842；六种聚合方法 23,052 条 | 已完成；未应用接收阈值，身份互斥分区和拒绝项已保存 | `data/experiments/phase2/` |
| LFW 阶段 4 自然原始结果 | 有效标准库 7,465；有效待识别图片 5,720；六方法 34,320 条 | 已完成；无质量预筛，已完成联合标定和一次独立验证集 | `data/experiments/phase4/` |
| LFW/XQLFW 跨质量开放集联合标定 | 六场景、六种聚合、四类接收规则 | 已完成；在 1% 目标下选择人员平均特征向量 + 最高分阈值 `0.555786` | `data/experiments/phase5b/` |
| 小型标准库稳定性复核 | 3/5/10/25/50/100 人，每档 10 次、四场景 | 已完成固定参数、只使用参数选择集的重放；240 个结果均满足 1% 未知人员误接收率 | `data/experiments/phase5b/gallery_scale_report.json` |
| 跨质量独立验证集 | 六场景、三个预先固定的参考目标 | 已执行一次；主规则最差已登记人员正确接收率 66.30%、最坏未知人员误接收率 1.43%，含按来源身份重复抽样得到的 95% 置信区间 | `data/experiments/phase5b/evaluation_report.json` |
| XQLFW 全量跨质量实验 | 7,263 张、6,000 对；有效 7,200 张、5,894 对 | 已完成 CUDA 原始提取、官方 10 折验证及与原始 LFW 的 5,871 个共同图片对比较 | `data/experiments/phase5/` |
| XQLFW 推理子阶段优化 | 同一 7,263 张图片、独立空缓存 | 只执行检测与识别后，有效人脸特征向量平均耗时下降 26.0%；人脸特征向量与验证结果不变 | `data/experiments/phase5/xqlfw_optimized_*` |
| QMUL 官方协议 | 全部官方 MAT 标签和目录 | 已核验 | `data/logs/qmul_survface_protocol.json` |
| QMUL 旧尺寸规则压力预检 | 60,294 张标准库 | 旧 112 px 拒绝条件下无有效标准库；只作为“预筛会掩盖模型覆盖”的历史证据 | `data/logs/cache/qmul_survface.sqlite` |
| QMUL 全量原始压力实验 | 242,453 张；436 张有效人脸特征向量 | 已完成不使用质量拒绝规则的原始提取；70/3,000 个标准库身份可用，同源待识别图片仅 4 张可评分且第一候选均错误 | `data/experiments/phase5/qmul_*` |
| 阶段 5C 完整链路性能 | 自然 LFW，3/10/100/1,000/4,588 个身份，每档 30 张待识别图片 | 已完成内存人员平均特征矩阵与旧逐人路径的同条件对照；4,588 人完整链路中位数 566.103→288.430 ms，候选和接受结果一致 | `data/experiments/phase5c/runtime-performance/` |
| 阶段 5C 多帧鲁棒性 | 阶段 3 已缓存数据；5 类模拟短序列；身份互斥参数选择集/独立验证集 | 已完成单帧、清晰度最高帧、特征最一致帧和有效帧平均特征比较；独立集最差正确接收率 82.86%→100%，未知误接收率未上升，清晰度最高帧已接入摄像头 | `data/experiments/phase5c/multiframe/` |
| 阶段 5C 生产服务回放 | 自然 LFW，4,588 个身份，10 条同身份序列；单帧和五帧各 10 次 | 已用真实 `RecognitionService` 记录模型、质量、矩阵检索、判定和日志耗时；当前环境 CUDA 回退 CPU，真实摄像头取帧和 CUDA 速度留给阶段 6 | `data/experiments/phase5c/production-runtime/` |

LFW 阶段 4 原始提取使用实际 `CUDAExecutionProvider`，13,233 张中 13,185 张获得人脸特征向量、48 张人脸特征提取失败。再次读取同一缓存时 13,233 张全部命中且模型推理为 0；聚合或判定参数变化不再触发 InsightFace。

## 3. LFW 当前可复现入口

### 3.1 生成覆盖全图的源协议

```bash
.venv/bin/python scripts/prepare_lfw.py --data-dir ./data --full \
  --known-fraction 0.8 --enrollment-per-identity 5 --seed 2026 \
  --output ./data/datasets/lfw_full_open_set_protocol.json
```

### 3.2 提取与缓存原始人脸特征向量

```bash
.venv/bin/python scripts/extract_lfw_raw_embeddings.py --data-dir ./data
```

原始缓存不执行基于数值指标的质量预筛；只有模型、检测、对齐或图片内容变化才需要重新提取。

### 3.3 从缓存导出固定分区和连续候选分数

```bash
.venv/bin/python scripts/export_lfw_decision_scores.py \
  --data-dir ./data \
  --source-protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --output-dir ./data/experiments/phase4
```

缓存只有一个批次时脚本自动选择；有多个批次时必须用 `--cache-extraction-id` 明确指定。脚本不会初始化 `FaceEngine`，缓存缺失或图片 SHA-256 改变时直接失败。自然 LFW 的旧扫描器已被阶段 5B 四规则联合标定替代。

## 4. 阶段 5 评测入口

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
```

正式结果见 [阶段 5 XQLFW 结果](phase-5-xqlfw-results.md)、
[阶段 5 QMUL 结果](phase-5-qmul-results.md)和
[阶段 5B 联合标定结果](phase-5b-cross-quality-results.md)。阶段 5C 的完整链路、多帧和界面验收见
[阶段 5C 结果](phase-5c-results.md)。QMUL 的有效分母极小；历史机械阈值迁移报告只证明发生全拒绝，当前代码只保留原始覆盖提取入口，不把它当作桌面部署评测器。

阶段 5B 跨质量开放集：

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

1. 标准库、已登记人员图片、未登记人员图片、参数选择集、独立验证集和最终演示标准库角色不得混用。
2. 参数选择集/独立验证集在质量过滤前按来源身份划分，避免只挑选模型成功样本。
3. 尚未应用接收规则时的第一候选身份正确率只评价候选排序；未登记人员误接收率必须在应用指定接收阈值后统计。
4. 聚合方法改变会改变分数分布，匹配阈值和候选分差必须与方法联合选择。
5. XQLFW 官方图片对用于一对一验证；阶段 5B 只利用其与 LFW 同身份、同路径的图像变体构造一对多跨质量场景。QMUL 仍只作为监控域压力集。
6. 所有正式结论必须记录模型、质量规则、协议哈希、拒绝数量、有效分母和运行环境。
