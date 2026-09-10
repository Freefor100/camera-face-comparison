# Phase 5B 跨质量开放集规则联合标定

## 目标

在不重新训练 InsightFace 的前提下，用自然 LFW、XQLFW Probe 和混合质量 Gallery 共同选择一套可部署的人员聚合、开放集接收规则和参数。主目标为各主要 Calibration 场景的有效 Unknown `FPIR ≤ 1%`，随后最大化最差场景端到端 Known TPIR。

## 固定数据

- 复用 Phase 4 的 LFW Gallery、Known/Unknown Probe 和身份互斥 Calibration/Evaluation。
- XQLFW 使用与 LFW 相同的 13,233 条相对路径；只补提取缺失 embedding，不改变身份与分区。
- 混合 Gallery 使用种子 `2026`。多样本身份按路径哈希排序后交替使用自然与 XQLFW 图片；单样本身份按身份哈希奇偶固定选择域。
- 前四个主要场景为自然/自然、自然/XQLFW、混合/自然、混合/XQLFW；全 XQLFW Gallery 配自然或 XQLFW Probe 只作压力测试。

## 候选算法

人员聚合比较 Single、Max、Mean Prototype 和 Top-K Mean；Top-K 分别取 `2/3/5`。每个聚合保存前 32 个身份候选，不含任何判定参数。

接收规则必须作为互斥类型报告：

1. `score_threshold`：最高人员分数达到下限；
2. `score_gap`：第一、第二候选分差达到下限；
3. `score_and_gap`：同时满足最高分和候选分差；
4. `nac`：Mean Prototype 的前 `k` 个身份分数经过局部 softmax 后，第一候选概率达到下限，`k=2/4/8/16/32`。

所有参数只扫描 Calibration 中真实出现的分数断点。禁止使用 `-1` 等哨兵值模拟另一种规则，也不使用 Evaluation 标签选参。

## 评价与选择

- `FPIR_valid` 以成功获得 embedding 的 Unknown 为分母，负责约束开放集风险。
- `TPIR_valid` 只评价有效 Known；`TPIR_e2e` 以协议中全部 Known 为分母，模型失败计为未识别。
- 同时记录 Rank-1、FTE、正确接收、错误身份接收、Unknown 错误接收、耗时和按来源身份 bootstrap 的 95% 置信区间。
- 主工作点为四个主要场景分别满足 `FPIR_valid≤1%`；另报告 `0.3%` 与 `10%`。
- 满足 FPIR 后先最大化最差场景 `TPIR_e2e`，再最大化平均 `TPIR_e2e`。
- NAC 或候选分差只有在最差场景 TPIR 比 Mean Prototype + 最高分阈值至少提高 1 个百分点，并通过 Calibration 身份中的小 Gallery 稳定性复核时才可部署；否则选择简单基线。
- 最终候选冻结后在 Evaluation 只运行一次，Evaluation 超出目标只能记录泛化风险，不能回头改参数。

## 系统交接

实验完成后，应用只接入选中的一套规则。无人脸、多人脸、图片损坏和 embedding 失败继续阻断；人脸尺寸、模糊、亮度和对比度仅记录并提示。删除启发式 `quality_score`、质量等级、质量加权 Top-K 和分层识别阈值，不兼容旧配置和开发数据库。

