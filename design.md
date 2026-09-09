# 人脸比对系统当前实现说明

本文只记录当前代码已经实现的行为，不描述下一阶段方案。

## 1. 系统边界

当前程序是离线运行的开放集 1:N 人脸识别桌面应用。输入一张摄像头帧或本地图片后，程序在本地人员库中寻找候选人；匹配分数或候选分差不满足当前配置时输出“未知人员”。

程序不针对已录入人员重新训练分类器。InsightFace `buffalo_l` 是固定的预训练模型；新增人员和追加样本只会增加图片、embedding 和人员记录。

当前支持三种图片来源：

- `camera`：摄像头当前帧；
- `file`：用户选择的本地图片；
- `dataset`：LFW 评测图片。

三种来源最终都使用同一个 `FaceEngine` 提取人脸和 embedding。

## 2. 当前人脸处理链路

```text
BGR 图片
   ↓
InsightFace FaceAnalysis.get(frame)
   ├─ 人脸检测
   ├─ 五点关键点
   ├─ InsightFace 内部对齐和识别模型输入变换
   └─ embedding
   ↓
只保留检测分数达到门槛的检测结果
   ↓
必须恰好一张可信人脸
   ↓
脸部尺寸和拉普拉斯清晰度检查
   ↓
embedding L2 归一化
   ↓
亮度、对比度等质量评估
```

当前模型从 `data/models/buffalo_l/` 加载。`FaceAnalysis` 只执行 `detection` 与
`recognition` 模块，不逐图执行应用没有读取的性别年龄、二维关键点和三维关键点模型。
Linux NVIDIA 环境由 ONNX Runtime 优先使用 CUDA，并保留 CPU 作为不支持算子的回退；
程序在模型准备后检查 InsightFace session 的实际 provider，并将后端写入评测报告。
本地检测和识别模型分别为 `det_10g.onnx` 与 `w600k_r50.onnx`。

应用代码没有单独实现对齐器，也不保存对齐后的 112×112 人脸。InsightFace 的识别模型适配器在提取 embedding 时调用五点对齐和模型输入归一化；应用层只接收边界框、检测分数、关键点和最终 embedding。

## 3. 当前识别流程

`RecognitionService.compare_input()` 当前按以下顺序执行：

1. 检查 SQLite、外键、样本图片哈希和 embedding 哈希。
2. 从输入图片提取人脸；无人脸、多人脸、脸过小或模糊时返回 `invalid`。
3. 计算亮度、对比度、脸尺寸、检测分数和清晰度，得到 `high`、`medium` 或 `reject` 质量等级。
4. `reject` 不进入身份比对，返回 `invalid` 和具体质量原因。
5. 读取标准库中的全部人员和全部样本 embedding。
6. Query 与每张参考样本计算余弦相似度。
7. 每个人按相似度选择 Top-K 样本，再按参考样本质量做加权平均。
8. 对人员分数排序，应用当前质量等级的匹配阈值和候选分差。
9. 写入一条识别日志并返回结果。

当前实现中的 `invalid` 包含输入质量不合格和标准库完整性失败；`unknown` 表示图片已成功进入身份比对，但没有通过开放集规则。

## 4. 当前多样本打分算法

设 Query embedding 为 \(q\)，某人的第 \(j\) 个样本 embedding 为 \(e_{ij}\)。两者在比对前都会做 L2 归一化：

\[
s_{ij}=q^Te_{ij}
\]

当前代码不是先生成固定身份模板。它为每个 Query 单独完成以下计算：

1. 对人员 \(i\) 的所有样本分数 \(s_{ij}\) 从高到低排序；
2. 选择前 `top_k` 个分数；
3. 使用样本入库时记录的质量分数计算权重：

\[
w_{ij}=0.5+0.5\times clamp(Q_{ij},0,1)
\]

4. 得到该人员的当前 Query 分数：

\[
S_i=\frac{\sum_jw_{ij}s_{ij}}{\sum_jw_{ij}}
\]

每个当前格式的入库样本都必须保存数值型 `quality_score`，取值范围为 0–1；缺失或越界会使本次识别返回数据错误。新数据目录默认 `top_k=3`。

这个人员分数依赖当前 Query。当前 SQLite 没有身份模板表，内存中也没有固定的 Mean Prototype 或 Quality-aware Prototype。

## 5. 当前开放集决策

把所有人员分数从高到低排列，最高分为 \(S_1\)，第二高分为 \(S_2\)。当前规则是：

```text
S1 < match_threshold
    → unknown / score_below_threshold

存在第二候选，并且 S1 - S2 < min_score_gap
    → unknown / score_gap_below_minimum

其余情况
    → matched / 第一候选人员
```

库中没有人员时返回 `unknown / empty_face_library`。只有一个人员时没有第二候选，因此只检查匹配阈值。

全新数据目录生成的默认配置为：

| 探针质量 | `match_threshold` | `min_score_gap` |
| --- | ---: | ---: |
| high | 0.50 | 0.05 |
| medium | 0.60 | 0.08 |

这些值是初始运行参数，不是公开数据集标定结果。配置加载器要求当前结构完整存在，不对旧字段或缺失字段做兼容解析。

## 6. 当前质量规则

硬性质量检查使用以下配置：

- 最低检测分数；
- 最小脸部像素尺寸；
- 最小拉普拉斯方差；
- 最低和最高亮度；
- 最低对比度。

通过硬检查后，程序把五个归一化指标组合为 0–1 质量分数：

```text
0.25 × detection
+ 0.25 × face_size
+ 0.25 × sharpness
+ 0.15 × exposure
+ 0.10 × contrast
```

该分数是当前项目自定义的工程指标，不是 InsightFace 模型输出，也不是专用 FIQA 模型分数。它同时用于探针质量分级和参考样本加权。

## 7. 当前入库与标准库存储

当前入库流程支持本地图片和摄像头帧，不要求固定动作、姿态或顺序：

```text
一组输入图片
   ↓
逐张执行单脸检查、质量检查和 embedding 提取
   ↓
图片先写入 data/faces/.staging/
   ↓
全部成功后移动到 data/faces/<person_id>/
   ↓
一个 SQLite 事务写入人员和所有样本
```

UI 创建人员时至少要有一张合格图片，人员和首批样本在同一次数据库事务中写入。Repository 不提供创建空人员的接口，也没有人员生命周期状态。创建成功的人员立即参与识别；同一人员可以继续追加任意数量的合格样本。

当前代码中没有固定姿态录入会话或五步录入入口。

SQLite 当前包含：

- `persons`：人员名称和创建时间；
- `face_samples`：图片相对路径、`float32` embedding BLOB、维度、质量、来源和哈希；
- `recognition_logs`：决策、候选分数、耗时和原因。

SQLite 开启外键、WAL、5 秒 busy timeout 和 `BEGIN IMMEDIATE` 写事务。

## 8. 当前完整性检查

入库时分别记录：

- 样本图片文件的 SHA-256；
- embedding `float32` 字节的 SHA-256。

识别前，程序运行 SQLite `integrity_check`、`foreign_key_check`，并检查图片是否存在、图片哈希是否一致、embedding 哈希是否一致。任一检查失败时，本次识别停止并返回 `library_integrity_failed:<kind>`。

该机制用于发现文件丢失、误覆盖和局部数据库损坏。它不检查摄像头帧，不参与相似度计算，也不能防御能够同时修改图片、向量和哈希值的攻击者。

## 9. 当前摄像头和线程行为

摄像头由 OpenCV 按设备索引打开：Linux 优先 V4L2，Windows 优先 DirectShow，macOS 优先 AVFoundation，失败后回退通用后端。Linux 的索引 0 通常对应 `/dev/video0`，但代码没有写死设备路径。

预览在 `CameraWorker` 中持续读取，UI 接收复制后的帧。识别在单独的 `RecognitionWorker` 中执行。停止预览时，当前代码清除最后一帧、检测框和待比较帧。

本地图片识别与摄像头抓拍调用同一个 `RecognitionService`。待测图片不会自动加入标准库。

## 10. 当前 LFW 评测

当前项目可以显式下载 LFW deep-funneled，生成固定的 `lfw-open-set-v1` 协议：

- 已知身份的一部分图片用于 Gallery；
- 同身份的其他图片作为 Known Probe；
- 完全不进入 Gallery 的身份作为 Unknown Probe。

`scripts/evaluate_lfw.py` 使用与应用相同的 `FaceEngine` 和质量规则提取真实 embedding；`EvaluationEmbeddingCache` 按图片 SHA-256 保存有效向量或拒绝原因。该缓存保存的是经过特定质量策略筛选后的结果，因此缓存键同时包含三类标识：

- `embedding_extraction_id`：模型、检测输入、对齐和 embedding 归一化实现；
- `quality_policy_id`：检测置信度、尺寸、清晰度、亮度、对比度和启发式分层规则；
- `decision_policy_id`：聚合、匹配阈值、候选分差和质量层判定配置。

质量门、匹配阈值、候选分差或 K 变化都不会改变 `embedding_extraction_id`。改变质量策略会进入新的质量策略缓存分区。

评测构建 Gallery 时，只要某个身份至少有 1 张图片成功提取 embedding，该身份就会进入 Gallery；其余失败图片会单独记录为 enrollment rejection。协议生成器可以为每个身份分配多张图片，但这不是激活门槛。

`split_lfw_protocol()` 在质量过滤前按来源身份把 Probe 固定为 `calibration` 和 `evaluation`。Gallery 保持相同；Known 使用真实 Gallery 身份分组，Unknown 从图片首级目录恢复来源身份，同一来源身份不会跨分区。分区协议保存源协议哈希、种子、分区比例和全部路径。

Phase 4 新增 `scripts/extract_lfw_raw_embeddings.py` 和 `RawEmbeddingCache`。原始缓存主键只包含数据集、模型提取版本、相对路径和图片 SHA-256；记录主体脸 embedding、五项原始测量、检测数量、耗时或 FTE，不包含质量策略。LFW 有身份标签且主体明确，因此检测到多张脸时选择面积最大的主体脸；这个规则只用于数据集，不改变桌面应用的多人脸拒绝。

`scripts/export_lfw_decision_scores.py` 是原始缓存的 cache-only 导出入口，不导入或初始化 `FaceEngine`。它从已有缓存生成六种人员聚合：Single、Max、Mean Prototype、Top-K Mean K=2/3/5。Top-K Mean 是普通平均，不读取质量权重。每张有效 Probe 保存第一候选、`top_score`、第二候选、`second_score`、`score_gap`、真实标签、原始质量指标和评分耗时；模型 FTE 另表保存原因。分数表不保存质量等级、匹配阈值或最小候选分差。

本机 Phase 2 历史 `decision_scores.sqlite` 由质量筛选缓存生成：4,735 张有效 Gallery、3,842 张有效 Probe、1,901 张 Probe 拒绝，六种方法共 23,052 条分数记录。Phase 4 的 CUDA 原始提取覆盖 13,233 张 LFW：13,185 张得到 embedding、48 张 FTE；其中 2,236 张检测到多个候选但按主体脸规则保留。新分数库包含 7,465 张有效 Gallery、5,720 张有效 Probe和 34,320 条六方法记录，使用独立目录，防止历史诊断覆盖正式实验输入。

历史流式报告仍会在给定固定阈值下输出 FPIR/FNIR，但这类结果被标记为历史诊断。项目还已有 XQLFW 官方 pairs 解析、QMUL-SurvFace 官方 MAT 协议解析和相应评测入口；它们不会被 LFW cache-only 导出调用。

## 11. 当前开放集工作点扫描器

`scripts/calibrate_thresholds.py` 直接读取 `decision_scores.sqlite`，只查询 `split='calibration'`，不在参数选择期间读取 Evaluation。它对每种人员聚合分别比较：

- 只使用 `match_threshold`；
- 同时使用 `match_threshold` 和 `min_score_gap`。

候选值来自 Calibration 中实际出现的 `top_score` 和 `score_gap` 断点，并补充余弦分数 `[-1, 1]` 与分差 `[0, 2]` 的合法边界。二维规则先建立“最高分断点 × 候选分差断点”的离散计数，再用后缀累计一次得到所有组合的接收数量，不按 0.01 网格近似，也不需要反向传播。

扫描器分别输出 FPIR 不超过 1%、0.3% 和观测 0% 的工作点。每个工作点记录匹配阈值、是否启用候选分差、Known Rank-1、TPIR、FNIR、Unknown 误接收数和 FPIR；若闭区间内没有组合达到目标，会明确标记 `meets_target=false`。

`select_best_operating_point()` 只接收 Calibration 报告，按“满足 FPIR、最大 TPIR、不使用分差、模板紧凑和检索开销”顺序选出唯一候选。`scripts/evaluate_selected_operating_point.py` 在选择完成后才调用 `evaluate_operating_point()` 读取 Evaluation，并把选择和评估一起原子写入 JSON。

本机自然 LFW 实验在主目标 `FPIR≤0.3%` 下选出 Mean Prototype 与候选分差规则：Calibration 为 FPIR 0.244%、TPIR 61.49%，独立 Evaluation 为 FPIR 0.252%、TPIR 48.45%、Rank-1 98.60%。这是 4,599 身份 LFW Gallery 下的候选；当前桌面应用仍使用原有质量加权 Top-K 和初始质量层参数，尚未因该结果修改。应用接入必须先复核少量身份 Gallery 和摄像头域偏移。

## 12. 当前 XQLFW 跨质量评测

`scripts/extract_xqlfw_raw_embeddings.py` 只提取官方 6,000 对引用的 7,263 张去重图片，使用与 LFW 相同的 `RawEmbeddingCache` 和最大主体脸规则，不应用桌面质量门。`scripts/evaluate_xqlfw.py` 是 cache-only 入口，读取数据集随附的逐图质量分；每一折只用其他九折的实际相似度断点选择验证阈值，再在当前折统计结果。它保存每个 Pair 的相似度、折次阈值、预测、最低质量和质量分差。

`scripts/compare_xqlfw_domains.py` 按折次和图片路径连接原始 LFW 与 XQLFW 报告，只在两边都成功提取的共同 Pair 上比较准确率、错误转移和相似度变化。当前全量结果为 XQLFW 有效图片 7,200/7,263、有效 Pair 5,894/6,000、10 折准确率 94.16%；共同 5,871 对中，原始 LFW 为 98.48%，XQLFW 为 94.14%。该结果只描述 1:1 跨质量退化，不修改 1:N 应用阈值。

同一批 XQLFW 图片在只执行检测和识别模块后，有效 embedding 平均提取耗时从
32.293 ms 降至 23.896 ms；两份缓存的检测状态、人脸数量、质量指标和 embedding
逐条一致，官方验证准确率不变。该测量仅覆盖模型推理子阶段，不代表应用 E2E 耗时。

## 13. 当前 QMUL-SurvFace 监控域压力评测

`scripts/extract_qmul_raw_embeddings.py` 使用 QMUL 官方 MAT 标签和三个图片目录，
把 Gallery、Mated Probe 与 Unmated Probe 共 242,453 张图片写入独立
`RawEmbeddingCache`。它不应用桌面数值质量门；每 100 张提交一次，支持中断恢复。

`scripts/evaluate_qmul.py` 是 cache-only 入口。它为每个至少有一张有效参考图的身份建立
归一化 Mean Prototype，保存有效 Probe 的第一、第二候选和候选分差，并分别统计模型
FTE、Mated 身份无有效 Gallery、Rank-1 和 Unmated 分数分布。Phase 4 的 LFW 工作点
只被机械应用为跨域迁移诊断，不使用 QMUL 标签重新搜索参数。

当前全量结果为 436/242,453 张产生 embedding；Gallery 74/60,294 张、70/3,000
个身份可用，Mated 只有 4/60,423 张可评分且 Rank-1 为 0，Unmated 有
264/121,736 张可评分。LFW 候选分差门使 268 张可评分 Probe 全部被拒。该结果描述
监控小脸域不受当前模型覆盖的系统边界，不修改桌面应用阈值。

## 14. 当前小 Gallery 规模评测

`scripts/evaluate_gallery_scale.py` 只读取 Phase 4 自然 LFW 原始缓存。它从身份互斥的
Calibration/Evaluation Known 池分别选择 3、5、10、25、50、100 个身份，每档使用
10 个确定性重复；每次只在 Calibration 比较单阈值和阈值加候选分差，再把唯一工作点
应用到 Evaluation。Phase 4 的完整 Gallery 工作点同时作为固定迁移对照。

60 组迁移对照的 Evaluation Unknown FPIR 均为 0；完整 Gallery 工作点的平均 TPIR
从 3 人的 98.63% 随规模增加下降到 100 人的 83.06%。独立小规模标定有 56/60 次
选择单一匹配阈值，3 人档的阈值和 TPIR 方差明显较大。该结果只说明自然 LFW 域内的
Gallery 规模效应，不替代真实摄像头域复核，也不修改当前应用配置。
