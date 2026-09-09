# Phase 4 聚合方法与开放集判定联合标定结果

## 结论

本阶段已完成自然 LFW 全量原始特征重建、六种人员聚合比较、开放集工作点精确扫描，以及一次独立 Evaluation。所有聚合方法使用同一 Gallery、同一 Probe、同一 InsightFace `buffalo_l` embedding，不使用启发式质量分预筛或加权。

在预先指定的主工作点 `FPIR ≤ 0.3%` 下，Calibration 选出的候选方案是：

```text
人员聚合：Mean Prototype（同一人员的样本向量求均值后重新 L2 归一化）
开放集判定：score_gap >= 0.5078019499778748
match_threshold：-1.0（余弦分数合法下界，在该工作点等价于未启用绝对分数门）
```

该方案在 Calibration 上的 FPIR/TPIR 为 `0.2443% / 61.49%`，锁定参数后在独立 Evaluation 上为 `0.2523% / 48.45%`，仍满足 0.3% FPIR 目标。Evaluation 的 Known Rank-1 为 `1974/2002 = 98.60%`，说明大部分 Known 的第一候选身份正确；较低 TPIR 主要来自严格拒识规则主动拒绝，而不是第一名大量排错。

这只是 LFW 4,599 身份 Gallery 下的算法候选，不直接写入桌面应用配置。候选分差取决于候选人数和数据域，最终部署前仍需使用非最终演示人员的摄像头开发数据复核；最终 Demo Gallery 不参与调参。

## 原始特征与分数覆盖

CUDA 全量提取运行时间为 2026-09-09 18:53:47 至 19:01:14（Asia/Shanghai）：

| 项目 | 数量 |
| --- | ---: |
| LFW 图片总数 | 13,233 |
| 得到 512 维 embedding | 13,185 |
| 模型未检测到人脸 | 48 |
| 仅检测到一张脸 | 10,949 |
| 检测到多张脸并选取最大主体脸 | 2,236 |
| Gallery 原始/有效 | 7,490 / 7,465 |
| Probe 原始/有效 | 5,743 / 5,720 |
| 六方法连续分数记录 | 34,320 |

有身份标签的 LFW 图片允许从多张检测结果中选取面积最大的主体脸；桌面应用仍保留多人脸拒绝业务规则。第二次读取同一缓存覆盖全部 13,233 张图片，`cache_hit_total=13,233`、`inference_total=0`，证明改变质量或判定规则无需重新运行 InsightFace。

本地产物位于被 Git 忽略的 `data/experiments/phase4/` 和 `data/logs/cache/lfw_raw.sqlite`：

- `raw_extraction_manifest.json`：实际 CUDA 提取记录；
- `raw_cache_reuse_manifest.json`：零推理缓存复用记录；
- `decision_scores.sqlite`：不含阈值的逐 Probe 六方法分数；
- `calibration_report.json`：12 组“聚合 × 判定”变体在三个 FPIR 目标下的精确工作点；
- `final_evaluation.json`：主工作点锁定后的一次 Evaluation 结果。

## Calibration 与 Evaluation

Calibration 是只用于选择聚合方法和判定参数的开发分区；Evaluation 是选择完成后才读取一次的独立分区，用来检查选择能否泛化。二者共享同一 Gallery，但 Known Probe 身份和 Unknown 来源身份均按身份互斥，不允许同一 Probe 身份跨分区。

| 分区 | 原始 Probe | 有效 Known | 有效 Unknown | 模型失败 |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 2,544 | 1,301 | 1,228 | 15 |
| Evaluation | 3,199 | 2,002 | 1,189 | 8 |

Gallery 包含 4,599 个候选身份和 7,490 张参考图片。Calibration/Evaluation 分别包含 127 个 Known Probe 身份和 575 个 Unknown 来源身份；两分区身份互斥。

## 六种聚合和两类判定规则

下表只使用 Calibration，比较主工作点 `FPIR ≤ 0.3%`。`仅阈值` 检查最高人员分数；`阈值+分差` 还允许检查第一名与第二名的候选分差。`TPIR` 分母包含所有有效 Known，不把被拒绝的 Known 从分母删除。

| 聚合方法 | 判定 | 匹配阈值 | 最小候选分差 | Rank-1 | TPIR | FPIR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Single | 仅阈值 | 0.6883 | 0 | 98.62% | 42.35% | 0.244% |
| Single | 阈值+分差 | 0.6883 | 0 | 98.62% | 42.35% | 0.244% |
| Max | 仅阈值 | 0.8222 | 0 | 98.77% | 7.23% | 0.244% |
| Max | 阈值+分差 | 0.8222 | 0.1494 | 98.77% | 7.23% | 0.244% |
| Mean Prototype | 仅阈值 | 0.7815 | 0 | 98.85% | 50.27% | 0.244% |
| **Mean Prototype** | **阈值+分差** | **-1** | **0.5078** | **98.85%** | **61.49%** | **0.244%** |
| Top-K Mean, K=2 | 仅阈值 | 0.7272 | 0 | 98.85% | 45.43% | 0.244% |
| Top-K Mean, K=2 | 阈值+分差 | 0.7272 | 0 | 98.85% | 45.43% | 0.244% |
| Top-K Mean, K=3 | 仅阈值 | 0.7272 | 0 | 98.85% | 37.59% | 0.244% |
| Top-K Mean, K=3 | 阈值+分差 | 0.7272 | 0 | 98.85% | 37.59% | 0.244% |
| Top-K Mean, K=5 | 仅阈值 | 0.7273 | 0 | 98.62% | 19.45% | 0.244% |
| Top-K Mean, K=5 | 阈值+分差 | 0.7273 | 0 | 98.62% | 19.45% | 0.244% |

Mean Prototype 的候选分差规则比同方法仅阈值多接收 `800 - 654 = 146` 个正确 Known，而 Unknown 误接收仍为 3 个，因此本实验不能删除候选分差。其他方法在该目标下没有获得分差收益。

工作点灵敏度也必须保留：Mean Prototype 在 `FPIR≤1%` 时只用阈值即可达到 97.85% TPIR；在 Calibration 观测 `FPIR=0%` 时 TPIR 只有 0.85%。这表明阈值不存在脱离安全目标的“通用最优值”。

## 独立 Evaluation

主工作点的参数在查看 Evaluation 标签前已由统一规则锁定：先满足目标 FPIR，再最大化 TPIR；同效时优先不使用候选分差，随后优先模板紧凑、检索开销较低的方法。

| 指标 | Calibration | Evaluation |
| --- | ---: | ---: |
| Known 有效数 | 1,301 | 2,002 |
| Known Rank-1 | 1,286（98.85%） | 1,974（98.60%） |
| 正确接收 Known | 800 | 970 |
| 错误接收 Known | 2 | 4 |
| Unknown 有效数 | 1,228 | 1,189 |
| 错误接收 Unknown | 3 | 3 |
| TPIR | 61.49% | 48.45% |
| FNIR | 38.51% | 51.55% |
| FPIR | 0.244% | 0.252% |

Evaluation 达到预设 FPIR 目标，但 TPIR 比 Calibration 下降 13.04 个百分点。这不是继续读取 Evaluation 反复调参的理由，而是需要在后续摄像头域复核中保留的泛化风险。

## 当前边界与交接

- `Mean Prototype + score_gap` 是 LFW 大 Gallery 条件下的候选规则，不是已经冻结的桌面部署参数。
- `score_gap=0.5078` 对 Gallery 候选身份数量和人员组成敏感；最终只有少量身份的 Demo Gallery 不能直接声称拥有相同 FPIR/TPIR。
- Phase 3 已否定启发式 `quality_score` 的硬拒绝、加权和质量分层阈值用途；当前桌面应用尚未删除这套旧部署策略，待摄像头开发域复核后一次性替换。
- XQLFW 和 QMUL-SurvFace 不参与本轮方法选择或阈值标定，分别留给真实跨质量分析和监控小脸压力测试。

## 复现命令

```bash
.venv/bin/python scripts/extract_lfw_raw_embeddings.py --data-dir ./data
.venv/bin/python scripts/export_lfw_decision_scores.py \
  --data-dir ./data \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --output-dir ./data/experiments/phase4
.venv/bin/python scripts/calibrate_thresholds.py \
  --scores ./data/experiments/phase4/decision_scores.sqlite \
  --output ./data/experiments/phase4/calibration_report.json
.venv/bin/python scripts/evaluate_selected_operating_point.py \
  --scores ./data/experiments/phase4/decision_scores.sqlite \
  --target-fpir 0.003 \
  --output ./data/experiments/phase4/final_evaluation.json
```

图片、embedding、SQLite、JSON 和 manifest 均不进入 Git；本文提交实验协议、关键数字、结论和局限性。
