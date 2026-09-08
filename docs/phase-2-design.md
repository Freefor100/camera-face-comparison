# Phase 2：固定评测数据与无阈值分数

## 1. 阶段目标

Phase 2 不再用当前默认阈值直接评判哪种方法最好。本阶段只固定后续实验输入，并保存不依赖 `match_threshold` 和 `min_score_gap` 的连续分数：

```text
固定 LFW Gallery 与全部 Probe
  → 在质量过滤前按来源身份划分 Calibration/Evaluation
  → 复用已经完成的 embedding 缓存
  → 对每张有效 Probe 计算六种人员聚合结果
  → 保存第一候选、第二候选与候选分差
```

这样 Phase 4 改变匹配阈值或候选分差时只需查询本地 SQLite，不必再次运行 InsightFace。XQLFW 和 QMUL-SurvFace 不参与本阶段的聚合选择，它们保留到 Phase 5 做跨质量和监控小脸压力分析。

## 2. 固定数据协议

源协议为 `data/datasets/lfw_full_open_set_protocol.json`：

- 覆盖 LFW 全部 13,233 张图片；
- Gallery 7,490 张，Probe 5,743 张；
- Gallery 身份的留出图片为 Known Probe；
- 未进入 Gallery 的身份图片为 Unknown Probe。

`split_lfw_protocol()` 在质量过滤前完成第二层划分：

- Gallery 在 Calibration 和 Evaluation 中完全相同，保证候选身份数一致；
- Known Probe 按 `expected_person_id` 分组；
- Unknown Probe 按图片首级目录恢复真实来源身份；
- Known 与 Unknown 分别以固定种子 `2026` 洗牌，50% 来源身份进入 Calibration，其余进入 Evaluation；
- 同一 Probe 来源身份不会跨越两个分区。

生成的 `data/experiments/phase2/protocol.json` 保存源协议 SHA-256、随机种子、分区比例、图片路径、真实来源身份和数据角色。该文件位于 Git 忽略的 `data/`，不会把数据集路径清单提交到源码仓库。

## 3. embedding 缓存与判定参数解耦

`EvaluationEmbeddingCache` 继续按数据集标识、缓存批次标识、相对路径和图片 SHA-256 恢复有效 embedding 或拒绝原因。新代码把标识拆成：

- `embedding_extraction_id`：模型、检测、质量测量和质量过滤参数；
- `decision_policy_id`：聚合配置、匹配阈值、候选分差及质量层判定参数。

只有会改变有效图片或 embedding 的参数才能使 embedding 缓存失效。改变 `match_threshold`、`min_score_gap` 或 `top_k` 不会使 embedding 缓存失效。

本地已完成缓存按实际提取配置重标为当前 `embedding_extraction_id`：LFW/XQLFW 的 80 像素评测配置为 `buffalo_l:a9e5cd6dc7584a66`，QMUL 的 112 像素配置为 `buffalo_l:6d44572e8a00da16`。这是开发数据的一次直接重建动作，代码中没有迁移、旧字段别名或回退解析。

## 4. 六种人员聚合候选

所有方法使用同一有效 Gallery 和同一 Probe embedding。`quality_score` 被保存，但不参与本阶段主聚合。

| 内部名称 | K | 人员分数定义 |
| --- | ---: | --- |
| `single` | 0 | 每个身份固定使用协议顺序中的第一张有效参考样本 |
| `max` | 0 | 该身份全部参考样本相似度的最大值 |
| `mean_prototype` | 0 | 样本向量平均后重新 L2 归一化，再与 Probe 计算相似度 |
| `top_k_mean` | 2 | 最高 2 个样本相似度的普通平均 |
| `top_k_mean` | 3 | 最高 3 个样本相似度的普通平均 |
| `top_k_mean` | 5 | 最高 5 个样本相似度的普通平均 |

样本数少于 K 时使用该身份的全部有效样本。K=1 与 Max 完全重复，因此不另设候选。历史报告中的“质量加权 Top-K”保留为历史诊断，不属于这六组主候选。

## 5. 无阈值分数库

`data/experiments/phase2/decision_scores.sqlite` 包含：

- 运行编号、协议哈希和实际缓存批次标识；
- `calibration` / `evaluation` 分区；
- Probe 路径、Known 真实身份或 Unknown 来源身份；
- 聚合方法及 K；
- 第一候选身份与 `top_score`；
- 第二候选身份与 `second_score`；
- `score_gap = top_score - second_score`；
- 第一候选是否等于 Known 真实身份；
- Probe 的启发式质量分、质量层级、五项原始质量指标；
- 评分耗时；
- Gallery/Probe 的质量拒绝路径和原因。

表结构刻意不含 `match_threshold` 和 `min_score_gap`。`calibration_summary.json` 与 `evaluation_summary.json` 只汇总覆盖数、无阈值 Rank-1、连续分数分位点和评分耗时，不执行 Unknown 接收判定。当前评分耗时是一次批处理中共同计算六组候选的平均每 Probe 成本，不用于比较单个聚合方法的速度；方法级 E2E 计时留给实际部署方案的性能实验。

## 6. 实际产物与覆盖

执行命令：

```bash
.venv/bin/python scripts/export_lfw_decision_scores.py \
  --data-dir ./data \
  --source-protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --cache-path ./data/logs/cache/lfw.sqlite \
  --output-dir ./data/experiments/phase2 \
  --min-face-size 80
```

实际生成：

```text
data/experiments/phase2/
├── protocol.json
├── decision_scores.sqlite
├── calibration_summary.json
├── evaluation_summary.json
└── manifest.json
```

覆盖结果：

- Gallery：7,490 张，其中 4,735 张有效、2,755 张拒绝；
- Probe：5,743 张，其中 3,842 张有效、1,901 张拒绝；
- 六种聚合结果：`3,842 × 6 = 23,052` 条；
- Calibration 有效 Probe 1,696 张，Evaluation 有效 Probe 2,146 张；
- Known 来源身份分区各 127 个，Unknown 来源身份分区各 575 个，交集均为 0。

分区的有效图片数不必相等：划分单位是质量过滤前的身份，不是过滤后成功图片数量。

## 7. 本阶段不作出的结论

- 无阈值 Rank-1 只能检验第一候选排序，不能代表开放集 Unknown 拒识效果；
- 当前固定阈值全量报告只保留为历史诊断，不用于选择最终方法；
- 本阶段不写回正式阈值，不修改应用默认聚合，不建立最终 Demo Gallery；
- 本阶段不根据 XQLFW 或 QMUL 结果降低桌面应用质量门；
- `quality_score` 尚未证明能预测识别错误，因此不用于主候选加权。

下一依赖阶段是 Phase 3：先验证并冻结质量门与入库拒绝规则。质量门冻结后，Phase 4 才在同一批有效数据上联合选择聚合方法、匹配阈值和候选分差。
