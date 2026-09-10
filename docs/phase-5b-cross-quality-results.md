# Phase 5B 开放集识别规则与跨质量联合标定结果

## 1. 最终结论

本阶段使用自然 LFW 与其同身份、同路径的 XQLFW 低质量变体，统一比较六种人员聚合和四类开放集接收规则。实验冻结且当前桌面应用已经接入的部署方案为：

```text
人员聚合：Mean Prototype
接收规则：score_threshold（只检查最高人员分数）
最低分数：minimum_score = 0.5557855367660522
候选分差：不参与判定，只保留为结果解释字段
质量总分/等级：不参与判定
```

这不是 InsightFace 官方阈值，也不是凭经验手填的数值。它来自固定 Calibration 分区，在四个主要场景分别满足有效 Unknown `FPIR≤1%` 后，使最差场景端到端 Known TPIR 最大的真实分数断点。

最终没有部署仅候选分差、双条件或 NAC，原因是实验收益不足：

- Mean Prototype + NAC-32 的 Calibration 最差 TPIR 为 67.23%，只比最高分阈值的 66.84% 高 0.39 个百分点，低于实验前规定的 1 个百分点进入门槛，而且平均 TPIR 略低；
- Mean Prototype + `score_and_gap` 自动选到 `minimum_gap=0`，与仅最高分阈值完全等价，说明候选分差没有提供额外约束；
- Mean Prototype + 仅候选分差的最差 TPIR 为 62.03%，低于最高分阈值；
- 复杂规则没有提供足够收益时，保留更容易解释、对小 Gallery 更稳定的最高分阈值。

独立 Evaluation 没有用于再次调参。主规则的最坏有效 Unknown FPIR 从 Calibration 的 0.98% 漂移到 Evaluation 的 1.43%，说明 1% 是开发集工作目标而不是对任意数据的保证；但严格 0.3% 规则会把跨质量最差端到端 TPIR 从约 66% 降到 14%～28%，不适合作为本课设的默认交互式识别策略。

## 2. 数据、模型与实验隔离

- 模型：InsightFace `buffalo_l`，512 维 L2 归一化 embedding；
- 推理：优先 `CUDAExecutionProvider`，本轮 XQLFW 补提取确认实际使用 RTX 3050 Laptop GPU；
- 自然域：LFW deep-funneled 13,233 张，13,185 张得到 embedding，48 张 FTE；
- 低质量域：XQLFW 完整同路径变体 13,233 张，13,113 张得到 embedding，120 张 FTE；
- Gallery：沿用固定 LFW 开放集协议，共 7,490 张、4,599 个身份；
- Probe：5,743 张，在质量处理前按来源身份拆成 Calibration 与 Evaluation；
- Calibration/Evaluation 的 Known 身份和 Unknown 来源身份互斥；
- 最终 Demo Gallery、摄像头照片、QMUL-SurvFace 均未参与选参。

XQLFW 完整补提取的运行摘要为：

| 项目 | 数量 |
| --- | ---: |
| 协议路径 | 13,233 |
| 有效 embedding | 13,113 |
| FTE | 120 |
| 已有缓存命中 | 7,263 |
| 本轮 CUDA 推理 | 5,970 |

无阈值分数库共写入 205,344 条“场景 × Probe × 聚合方法”排序记录，保存每张 Probe 的前 32 个候选身份和分数；另记录 234 次 Probe FTE 与 260 次 Gallery FTE。分数库不保存 `minimum_score`、`minimum_gap` 或 NAC 概率阈值，因此改变接收参数不需要重新运行 InsightFace。

## 3. 六个跨质量场景

四个主要场景共同选择同一参数：

| Gallery | Probe | 角色 |
| --- | --- | --- |
| 自然 LFW | 自然 LFW | 基础场景 |
| 自然 LFW | XQLFW | Probe 跨质量 |
| 自然与 XQLFW 固定混合 | 自然 LFW | 入库质量不齐 |
| 自然与 XQLFW 固定混合 | XQLFW | 主要鲁棒场景 |

另有两个全 XQLFW Gallery 场景，只用于报告极端质量域变化，不参加选参。混合 Gallery 使用种子 `2026` 和路径哈希固定；多样本身份尽量交替使用自然与 XQLFW 样本，因此不是每次运行随机更换图片。

`FPIR_valid` 以成功得到 embedding 的 Unknown 为分母，避免把检测失败伪装成正确拒识；`TPIR_e2e` 以协议全部 Known 为分母，FTE 计为未识别。

## 4. 聚合方法与接收规则的公平比较

下表全部来自同一个 Calibration、同一 Gallery/Probe、同一模型和同一 `FPIR≤1%` 约束。表中的 TPIR 是四个主要场景中最差的端到端 TPIR。

| 人员聚合 | 接收规则 | 标定参数 | 最差 TPIR | 平均 TPIR | 最坏 FPIR |
| --- | --- | --- | ---: | ---: | ---: |
| Single | 最高分阈值 | `T=0.538348` | 46.83% | 66.92% | 0.98% |
| Single | 仅候选分差 | `G=0.294606` | 42.93% | 64.21% | 0.98% |
| Single | 最高分+分差 | `T=0.538348, G=0` | 46.83% | 66.92% | 0.98% |
| Max | 最高分阈值 | `T=0.662166` | 46.45% | 64.50% | 0.98% |
| Max | 仅候选分差 | `G=0.391077` | 45.91% | 64.61% | 0.98% |
| Max | 最高分+分差 | `T=0.635404, G=0.053868` | 51.41% | 69.42% | 0.98% |
| **Mean Prototype** | **最高分阈值** | **`T=0.555786`** | **66.84%** | **82.52%** | **0.98%** |
| Mean Prototype | 仅候选分差 | `G=0.332412` | 62.03% | 78.34% | 0.98% |
| Mean Prototype | 最高分+分差 | `T=0.555786, G=0` | 66.84% | 82.52% | 0.98% |
| Top-K Mean, K=2 | 最高分阈值 | `T=0.555509` | 61.80% | 78.27% | 0.98% |
| Top-K Mean, K=2 | 仅候选分差 | `G=0.307245` | 58.90% | 76.60% | 0.98% |
| Top-K Mean, K=2 | 最高分+分差 | `T=0.555509, G=0` | 61.80% | 78.27% | 0.98% |
| Top-K Mean, K=3 | 最高分阈值 | `T=0.555772` | 58.67% | 76.26% | 0.98% |
| Top-K Mean, K=3 | 仅候选分差 | `G=0.301401` | 57.60% | 75.82% | 0.98% |
| Top-K Mean, K=3 | 最高分+分差 | `T=0.555772, G=0` | 58.67% | 76.26% | 0.98% |
| Top-K Mean, K=5 | 最高分阈值 | `T=0.555442` | 45.61% | 65.87% | 0.98% |
| Top-K Mean, K=5 | 仅候选分差 | `G=0.299017` | 45.84% | 66.86% | 0.98% |
| Top-K Mean, K=5 | 最高分+分差 | `T=0.521355, G=0.289137` | 46.60% | 67.48% | 0.98% |

### 4.1 NAC 对照

NAC 只在 Mean Prototype 上比较。它对前 K 个身份分数做局部 softmax，得到第一候选在局部邻域中的相对占比；它主要改变 Unknown 接收置信度，不改变第一候选是谁。

| NAC 邻居数 | 概率阈值 | 最差 TPIR | 平均 TPIR | 最坏 FPIR | 相对简单基线的最差 TPIR 增益 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.582346 | 62.03% | 78.34% | 0.98% | -4.82 pp |
| 4 | 0.318015 | 64.94% | 80.71% | 0.98% | -1.91 pp |
| 8 | 0.167445 | 66.92% | 81.97% | 0.98% | +0.08 pp |
| 16 | 0.086966 | 67.15% | 82.35% | 0.98% | +0.31 pp |
| 32 | 0.044744 | 67.23% | 82.51% | 0.98% | +0.39 pp |

NAC-32 是按最差 TPIR 排序的原始第一名，但 0.39 pp 小于预先规定的 1 pp 复杂规则收益门槛，且平均 TPIR 没有超过简单基线，因此最终选择没有被事后偏好改变。

### 4.2 为什么 Mean Prototype 胜出

- Single 只固定使用每人的第一张参考图，无法利用其他样本覆盖姿态与画质变化；
- Max 容易被某一张偶然高相似样本拉高，Unknown 的最高分也随 Gallery 样本数增加，因此达到同一 FPIR 时阈值更高；
- Top-K Mean 仍直接聚合样本分数，K 增大后会把不匹配姿态或低质量样本的低分带入人员分数；
- Mean Prototype 先在 embedding 空间形成一个身份中心并重新归一化，本实验中对自然和跨质量 Probe 的最差场景最稳定。

这只是当前数据和 `buffalo_l` 下的系统级结论，不应推广成所有人脸模型、数据集和采样方式的普遍定律。

## 5. 三个 FPIR 工作点

| Calibration 目标 | 选中方案 | 参数 | Calibration 最差 TPIR | Calibration 最坏 FPIR | Evaluation 最差 TPIR | Evaluation 最坏 FPIR |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0.3% | Mean Prototype + 仅分差 | `G=0.507386` | 27.73% | 0.244% | 13.81% | 0.254% |
| **1%（当前部署）** | **Mean Prototype + 最高分阈值** | **`T=0.555786`** | **66.84%** | **0.977%** | **66.30%** | **1.430%** |
| 10% | Mean Prototype + 最高分/分差 | `T=0.311742, G=0.039057` | 88.39% | 9.943% | 88.53% | 14.419% |

0.3% 规则安全目标更严格，但交互式应用中大量正常 Known 会被拒绝；10% 规则明显增加 Unknown 误接收。1% 是本项目在当前实验范围内的折中目标。Evaluation 对 1% 和 10% 都出现超目标，表明开发集工作点需要报告不确定性，不能写成“误识率保证”。

## 6. 主规则的 Calibration 结果

| 场景 | 协议 Known | 有效 Known / Unknown | Rank-1 | 正确接收 / 错误身份接收 | Unknown 误接收 | TPIR e2e | FPIR valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 自然库→自然 Probe | 1,309 | 1,301 / 1,228 | 1,286 | 1,273 / 10 | 12 | 97.25% | 0.977% |
| 自然库→XQLFW Probe | 1,309 | 1,292 / 1,227 | 1,211 | 875 / 5 | 4 | 66.84% | 0.326% |
| 混合库→自然 Probe | 1,309 | 1,301 / 1,228 | 1,286 | 1,266 / 9 | 11 | 96.72% | 0.896% |
| 混合库→XQLFW Probe | 1,309 | 1,292 / 1,227 | 1,187 | 907 / 5 | 6 | 69.29% | 0.489% |

混合质量入库没有让系统全面失效：对 XQLFW Probe，混合库比纯自然库的 TPIR 高 2.45 pp；对自然 Probe 则低 0.53 pp。这说明“库中照片质量不齐”需要通过固定协议衡量，不应简单把低质量参考图一律删除或赋予未经验证的质量权重。

## 7. 一次性独立 Evaluation

下表使用冻结的 `T=0.5557855`，没有搜索新参数。括号内为按来源身份 bootstrap 2,000 次得到的 95% 区间。

| 场景 | 有效 Known / Unknown | 正确接收 / 错误身份接收 | Unknown 误接收 | TPIR e2e（95% CI） | FPIR valid（95% CI） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自然库→自然 Probe | 2,002 / 1,189 | 1,971 / 15 | 17 | 98.26%（97.38%～98.91%） | 1.43%（0.40%～3.07%） |
| 自然库→XQLFW Probe | 1,990 / 1,179 | 1,330 / 9 | 14 | 66.30%（62.95%～69.11%） | 1.19%（0.39%～2.24%） |
| 混合库→自然 Probe | 2,002 / 1,189 | 1,969 / 12 | 15 | 98.16%（97.22%～98.76%） | 1.26%（0.28%～2.87%） |
| 混合库→XQLFW Probe | 1,990 / 1,179 | 1,365 / 9 | 14 | 68.05%（65.07%～71.32%） | 1.19%（0.42%～2.17%） |
| 全 XQLFW 库→自然 Probe（压力） | 2,002 / 1,189 | 1,920 / 10 | 5 | 95.71%（92.18%～97.41%） | 0.42%（0.09%～0.87%） |
| 全 XQLFW 库→XQLFW Probe（压力） | 1,990 / 1,179 | 1,393 / 8 | 6 | 69.44%（65.08%～72.39%） | 0.51%（0.17%～0.98%） |

FPIR 区间较宽，是因为 Unknown 错误集中在少数身份，按身份而不是按图片抽样后会保留这种相关性。报告因此同时保存点估计、计数和区间，不用单一百分比夸大稳定性。

## 8. 小 Gallery 稳定性

在查看 Evaluation 前，冻结主规则先在 Calibration 身份中对 3/5/10/25/50/100 人 Gallery 各做 10 次固定种子重放，共 240 个“规模 × 重复 × 场景”结果。阈值始终保持 `0.5557855`，没有用每个小库重新标定。

| Gallery 身份数 | 四场景最大 FPIR | 自然 Probe TPIR e2e 典型范围 | XQLFW Probe TPIR e2e 典型范围 |
| ---: | ---: | ---: | ---: |
| 3 | 0% | 95.98%～96.46%（均值） | 69.76%～73.26%（均值） |
| 5 | 0% | 96.30%～96.43% | 69.31%～71.42% |
| 10 | 0% | 97.16%～97.27% | 68.31%～70.68% |
| 25 | 0% | 96.55%～97.00% | 67.93%～70.60% |
| 50 | 0% | 96.33%～97.08% | 67.79%～70.09% |
| 100 | 0.081% | 96.73%～97.25% | 67.04%～69.50% |

全规模均满足 1% FPIR。随着 Gallery 变小，Unknown 的最高候选分数通常下降，因此从 4,599 人全库标定的阈值迁移到课设的少量身份库时更保守。3/5 人 TPIR 的单次范围较宽，来源是抽到的身份图片数量和难度不同，不应拿最终三个人自行重新调阈值。

## 9. 对质量处理的结论

本阶段不把启发式 `quality_score` 用作拒绝或加权。Phase 3 已证明旧质量门会拒绝大量本可正确识别的图片；本阶段又看到混合质量 Gallery 对低质量 Probe 有时反而有帮助。因此运行时改为：

- 损坏图片、无人脸、多人脸和 embedding 失败继续阻断；
- 尺寸、模糊、亮度、对比度和检测置信度保留原始测量与调整提示；
- 不使用启发式总分、质量等级、质量加权 Top-K 或分层阈值；
- 一人多图用 Mean Prototype 表示，不因为某张图的启发式质量低就自动丢弃。

这不代表低质量没有影响。XQLFW Probe 使自然 Gallery 的端到端 TPIR 从 98.26% 降到 66.30%，说明跨质量仍是系统局限；本项目的改进是用跨质量数据参与系统级规则选择，而不是声称攻克了底层低质量人脸表征问题。

## 10. 运行时接入复核

提交 `0a34600` 已把桌面服务替换为同一 Mean Prototype 和最高分阈值，并重建空的开发
配置/数据库。除自动化行为测试外，还用本地真实模型和临时标准库完成一次不参与调参的
图片烟雾复核：

| 临时 Gallery / Probe | 结果 | 第一候选分数 |
| --- | --- | ---: |
| Obama 2009 + 2016 入库；Obama 2012 未入库图片 | Known，命中 Barack Obama | 0.8196 |
| 同一临时 Gallery；Donald Trump 2017 | Unknown，低于冻结阈值 | -0.0442 |

该进程中 CUDA 设备/运行库不可见，实际走 CPU fallback；这与全量选参 manifest 中已确认
的 CUDA 推理不冲突，也验证了部署规则不依赖 GPU 才能运行。烟雾库位于临时目录，运行
结束后删除；最终 Demo Gallery 仍为空，不参与任何调参。

## 11. 复现命令与本地产物

```bash
.venv/bin/python scripts/extract_xqlfw_raw_embeddings.py \
  --data-dir ./data \
  --protocol ./data/experiments/phase4/protocol.json \
  --cache-path ./data/logs/cache/xqlfw_full_cuda.sqlite \
  --manifest ./data/experiments/phase5b/xqlfw_full_cuda_manifest.json

.venv/bin/python scripts/export_cross_quality_scores.py \
  --data-dir ./data \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite

.venv/bin/python scripts/calibrate_cross_quality.py \
  --scores ./data/experiments/phase5b/decision_scores.sqlite

.venv/bin/python scripts/evaluate_gallery_scale.py \
  --data-dir ./data \
  --xqlfw-cache ./data/logs/cache/xqlfw_full_cuda.sqlite

.venv/bin/python scripts/evaluate_cross_quality_policy.py \
  --scores ./data/experiments/phase5b/decision_scores.sqlite \
  --calibration-report ./data/experiments/phase5b/calibration_report.json
```

本地产物位于被 Git 忽略的 `data/experiments/phase5b/`：

- `xqlfw_full_cuda_manifest.json`：完整 XQLFW 提取覆盖和实际 CUDA 后端；
- `decision_scores.sqlite`：六场景无阈值前 32 候选排序；
- `calibration_report.json`：69 个“聚合/规则/目标”工作点和三个冻结候选；
- `gallery_scale_report.json`：240 次 Calibration-only 小库重放；
- `evaluation_report.json`：一次性 Evaluation 与按身份 95% 置信区间；
- `manifest.json`：协议、缓存和分数库哈希。

这些数据文件不提交 Git；本文保存可审计的实验设计、关键计数、结果、选择依据和局限性，供课程报告复用。
