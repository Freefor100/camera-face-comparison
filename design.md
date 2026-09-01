# 人脸比对系统当前实现说明

本文只记录当前代码已经实现的行为，不描述下一阶段方案。

## 1. 系统边界

当前程序是离线运行的开放集 1:N 人脸识别桌面应用。输入一张摄像头帧或本地图片后，程序在本地人员库中寻找候选人；匹配分数或候选差距不满足当前配置时输出“未知人员”。

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

当前模型从 `data/models/buffalo_l/` 加载，ONNX Runtime 固定使用 `CPUExecutionProvider`。本地模型目录包含 `det_10g.onnx`、`w600k_r50.onnx` 等文件。

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
8. 对人员分数排序，应用当前质量等级的匹配阈值和候选差距。
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

存在第二候选，并且 S1 - S2 < min_margin
    → unknown / candidate_gap_below_minimum

其余情况
    → matched / 第一候选人员
```

库中没有人员时返回 `unknown / empty_face_library`。只有一个人员时没有第二候选，因此只检查匹配阈值。

全新数据目录生成的默认配置为：

| 探针质量 | `match_threshold` | `min_margin` |
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

当前协议没有 Calibration/Test 两级身份划分。`scripts/evaluate_lfw.py` 使用与应用相同的 `FaceEngine` 和质量规则提取真实 embedding，然后比较：

- baseline：每个人的最高样本分数，使用全局阈值且不使用 margin；
- optimized：当前 Top-K 质量加权分数，使用质量分层阈值和 margin。

评测构建 Gallery 时，只要某个身份至少有 1 张图片成功提取 embedding，该身份就会进入 Gallery；其余失败图片会单独记录为 enrollment rejection。协议生成器可以为每个身份分配多张图片，但这不是激活门槛。

本机已运行的小型 pilot 包含 3 个入库身份和 3 个未知身份。入库阶段有 3 张图片被拒绝，探针阶段有 5 张图片被拒绝，最终只有 4 张 Known Probe 和 3 张 Unknown Probe 进入打分。Max 基线在这 7 张图片上得到已知 4/4、未知 3/3；当前 Top-K 质量加权策略得到已知 3/4、未知 3/3。

当前评测报告把 `probe_rejections` 单独列出，但 FPIR、FNIR 和 Rank-1 的分母只包含成功进入打分的 Probe。因此这些结果是“成功提取 embedding 后的条件识别结果”，不是包含检测和质量失败的端到端结果。

## 11. 当前阈值校准器

`scripts/calibrate_thresholds.py` 读取已预先生成的人员分数 JSONL，在 `0.30–0.80` 的匹配阈值和 `0.00–0.20` 的 margin 之间按 0.01 枚举。

当前选择顺序是：

1. 未知人员误接收数量最少；
2. 已知人员正确识别数量最多；
3. 若仍相同，选择更高的匹配阈值；
4. 若仍相同，选择更大的 margin。

脚本可以分别把结果写入 high 或 medium 配置，但当前仓库没有一份独立 Calibration 数据集产生的正式校准结果。
