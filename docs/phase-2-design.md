# Phase 2：固定评测数据与无阈值分数（历史阶段）

## 1. 阶段目标

本阶段第一次把“模型特征提取”和“开放集接收判定”拆开，不再用一个主观初始阈值
直接比较聚合方法：

```text
固定 LFW Gallery 与全部 Probe
  → 在图片过滤前按来源身份划分 Calibration/Evaluation
  → 复用 embedding 缓存
  → 对每张有效 Probe 计算六种人员聚合结果
  → 保存第一候选、第二候选和候选分差，不保存接收参数
```

XQLFW 和 QMUL-SurvFace 当时不参与聚合选择，后续分别用于跨质量与监控小脸压力分析。

## 2. 固定协议

源协议覆盖 LFW 13,233 张图片：Gallery 7,490 张，Probe 5,743 张。Gallery 身份的留出
图片是 Known Probe，完全不进入 Gallery 的身份图片是 Unknown Probe。

`split_lfw_protocol()` 在图片过滤前按身份划分：

- Gallery 在 Calibration 与 Evaluation 中相同；
- Known 按 Gallery 身份分组；
- Unknown 按图片首级目录的来源身份分组；
- 固定种子 `2026`，50% 来源身份进入 Calibration，其余进入 Evaluation；
- 同一 Known 或 Unknown 来源身份不跨分区。

协议保存源协议 SHA-256、随机种子、比例、角色与全部相对路径。

## 3. 六种聚合候选

| 内部名称 | K | 人员分数定义 |
| --- | ---: | --- |
| `single` | 0 | 每个身份固定使用第一张有效参考样本 |
| `max` | 0 | 该身份全部参考样本相似度的最大值 |
| `mean_prototype` | 0 | 样本向量平均、重新归一化后与 Probe 点积 |
| `top_k_mean` | 2 | 最高 2 个样本相似度的普通平均 |
| `top_k_mean` | 3 | 最高 3 个样本相似度的普通平均 |
| `top_k_mean` | 5 | 最高 5 个样本相似度的普通平均 |

K=1 与 Max 重复。每人样本少于 K 时使用全部有效样本。

## 4. 历史产物

`data/experiments/phase2/decision_scores.sqlite` 当时保存：

- Calibration/Evaluation 分区和 Probe 路径；
- Known 真实身份或 Unknown 来源身份；
- 聚合方法和 K；
- 第一/第二候选身份、连续分数和候选分差；
- Known 第一候选是否正确；
- 当时缓存携带的质量观测与拒绝原因；
- 评分耗时、协议哈希和代码版本。

实际覆盖为 Gallery 4,735 张有效、2,755 张拒绝；Probe 3,842 张有效、1,901 张拒绝；
六种聚合共 23,052 条连续分数。Calibration 有效 Probe 1,696 张，Evaluation 2,146 张。

## 5. 本阶段发现的问题

无阈值 Rank-1 只能说明第一候选排序是否正确，不能说明 Unknown 是否会被接收。更重要的
是，当时缓存先执行启发式质量预筛，导致 1,901 张 Probe 没有 embedding，分母受到旧
质量规则污染。因此 Phase 2 结果只保留为发现问题的历史证据，不能作为最终方法结论。

Phase 3 随后用 10,108 次单因素实验否定旧质量硬门；Phase 4 建立不含质量预筛的
`RawEmbeddingCache` 并重新提取 13,185/13,233 张自然 LFW 图片；Phase 5B 再用自然
LFW/XQLFW 六场景完成最终联合标定。当前可执行命令以 README 为准。
