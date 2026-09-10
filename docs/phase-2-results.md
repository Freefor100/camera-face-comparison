# Phase 2：固定评测数据与无阈值结果

> 本阶段产物用于 Phase 3 质量验证和 Phase 4 联合标定。它不使用当前阈值选最终方法，也不修改应用配置。

> 后续状态：Phase 3 已证明本阶段质量预筛过严，Phase 4 已重建自然 LFW 原始缓存并完成联合标定。本页保留为历史诊断，正式算法结果见 [Phase 4 结果](phase-4-results.md)。

## 1. 无阈值结果产物

运行日期：2026-09-08。源数据为本地 LFW 13,233 张图片，模型为已经完成缓存的 InsightFace `buffalo_l`。本轮只读取缓存，没有重新初始化或运行 InsightFace。

本地产物位于 `data/experiments/phase2/`：

- `protocol.json`：固定 Calibration/Evaluation 分区；
- `decision_scores.sqlite`：逐 Probe、逐方法的第一/第二候选连续分数；
- `calibration_summary.json`：Calibration 的无阈值覆盖和 Rank-1 摘要；
- `evaluation_summary.json`：Evaluation 的无阈值覆盖和 Rank-1 摘要；
- `manifest.json`：协议、缓存、质量配置、代码版本和覆盖统计。

这些文件包含数据集路径和实验结果，被 Git 忽略。

## 2. 覆盖与分区

| 项目 | 原始数量 | 有效数量 | 质量拒绝 |
| --- | ---: | ---: | ---: |
| Gallery 图片 | 7,490 | 4,735 | 2,755 |
| Probe 图片 | 5,743 | 3,842 | 1,901 |

有效 Probe 的六种聚合结果共 `3,842 × 6 = 23,052` 条。分区情况：

| 分区 | 有效 Probe | 有效 Known | 有效 Unknown | 拒绝 Probe |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 1,696 | 917 | 779 | 848 |
| Evaluation | 2,146 | 1,364 | 782 | 1,053 |

划分发生在质量过滤前，且以身份为单位。Known 来源身份在两个分区各 127 个，Unknown 来源身份各 575 个；两类身份的跨分区交集均为 0。有效图片数不相等是身份图片数量及质量拒绝造成的正常结果。

## 3. 六种方法的无阈值 Rank-1

下表只检查“真实 Known 身份是否排在第一名”，不执行 Unknown 接收，因此不能据此选择开放集最终方法。

| 方法 | Calibration Rank-1 | Evaluation Rank-1 |
| --- | ---: | ---: |
| Single | 913/917（99.56%） | 1363/1364（99.93%） |
| Max | 914/917（99.67%） | 1363/1364（99.93%） |
| Mean Prototype | 914/917（99.67%） | 1363/1364（99.93%） |
| Top-K Mean, K=2 | 914/917（99.67%） | 1363/1364（99.93%） |
| Top-K Mean, K=3 | 914/917（99.67%） | 1363/1364（99.93%） |
| Top-K Mean, K=5 | 914/917（99.67%） | 1363/1364（99.93%） |

结论仅限于：除 Single 在 Calibration 少命中 1 张外，这批有效 Known Probe 的第一名排序差异很小。Unknown 是否会被误接收取决于各方法自身的 `top_score` 分布和后续工作点，必须在 Phase 4 联合标定后判断。Evaluation 在选定最终方案前不会用于反复调参。

## 4. 历史固定阈值诊断

早前 `data/logs/lfw_full_algorithm_baseline.json` 在未完成身份互斥 Calibration 的情况下，直接使用初始阈值和质量层策略比较四种方法。该结果只保留为“为什么需要重构评测链路”的历史证据：

| 历史方法 | Known 正确 / 2,281 | Unknown 拒识 / 1,561 | FPIR | FNIR |
| --- | ---: | ---: | ---: | ---: |
| Single | 2,015 | 1,558 | 0.19% | 11.66% |
| Max | 2,188 | 1,556 | 0.32% | 4.08% |
| Mean Prototype | 2,226 | 1,555 | 0.38% | 2.41% |
| 历史质量加权 Top-K | 2,113 | 1,557 | 0.26% | 7.37% |

这些数字随阈值变化，且各方法的分数尺度不同，不能用一套初始阈值公平决定最终方法。历史质量加权 Top-K 也不属于当前六种普通聚合候选。

早期 7 张有效 Probe 的 pilot 同样只作为链路烟雾测试：Single 为 Known 2/4，Max 与 Mean Prototype 为 4/4，历史质量加权 Top-K 为 3/4；Unknown 均为 3/3。该样本量不承担算法结论。

## 5. 历史结论和后续完成状态

Phase 2 已完成“固定数据 + 缓存 embedding + 保存无阈值候选分数”，解决了反复运行模型和阈值变化污染方法比较的问题。

本阶段结束时确定的依赖顺序及后续完成情况为：

1. Phase 3 已验证人脸尺寸、模糊、亮度、对比度和旧启发式质量总分，结论是删除无依据硬门；
2. Phase 4 已重建 13,185 条无质量预筛的自然 LFW embedding；
3. Phase 5B 已在 Calibration 上联合比较六种聚合、四类规则和 NAC；
4. 当前部署已选择 Mean Prototype + 最高分阈值 `0.5557855`，并只执行一次独立 Evaluation。

XQLFW 和 QMUL-SurvFace 的正式结果已经分别记录在 Phase 5/5B 文档；本页继续只承担历史问题证据。
