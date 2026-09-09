# Phase 5 QMUL-SurvFace 监控小脸压力结果

## 结论

本实验完整覆盖 QMUL-SurvFace 官方开放集身份识别协议的 242,453 张图片，
不抽样、不应用桌面程序的尺寸、清晰度、亮度或启发式质量门。结果显示，当前
InsightFace `buffalo_l` 检测与识别链路无法直接覆盖该数据集的远距离监控小脸域：

- 242,453 张中只有 436 张产生 embedding，模型 FTE 为 242,017 张，整体有效率
  仅 `0.1798%`；
- 3,000 个 Gallery 身份中只有 70 个身份至少保留一张 embedding；
- 60,423 张 Mated Probe 中只有 98 张检测成功，且仅 4 张同时拥有可用的同身份
  Gallery，4 张 Rank-1 均错误；
- 121,736 张 Unmated Probe 中只有 264 张可评分。

因此，本实验不能支持“当前系统适用于监控小脸”的结论。相反，它以完整分母证明：
专门的小脸检测器、监控域适配、跟踪融合或针对性训练属于另一套研究与工程任务，
不应塞入本课程设计 TODO。桌面 UVC 摄像头仍需使用接近实际展示条件的数据复核。

## 协议和实现

- Gallery：60,294 张、3,000 个身份；
- Mated Probe：60,423 张，身份存在于官方 Gallery；
- Unmated Probe：121,736 张，身份不在 Gallery；
- 主体脸规则：数据集已有目标身份标签，多候选时选择面积最大的检测结果；
- 模型策略：只执行检测和识别模块，不执行性别年龄及额外关键点模型；
- 质量策略：无，只有检测器未产生主体脸时记录模型 FTE；
- 缓存：按数据集、embedding 提取版本、相对路径和图片 SHA-256 保存成功或 FTE；
- 推理后端：ONNX Runtime session 优先 `CUDAExecutionProvider`，并保留 CPU 回退；
- 推理耗时：完整墙钟 `3,872.75 s`（64 分 33 秒），缓存内平均模型调用
  `15.512 ms/图`。

这里的 FTE（Failure To Enroll/Extract）表示模型没有检测到可提取主体脸，不是
`min_face_size=112` 等项目质量门造成的拒绝。旧质量门压力预检与本实验使用不同缓存，
不会进入本次分母。

## 全量模型覆盖

| 角色 | 官方图片 | 有效 embedding | 模型 FTE | 有效率 |
| --- | ---: | ---: | ---: | ---: |
| Gallery | 60,294 | 74 | 60,220 | 0.1227% |
| Mated Probe | 60,423 | 98 | 60,325 | 0.1622% |
| Unmated Probe | 121,736 | 264 | 121,472 | 0.2169% |
| 合计 | 242,453 | 436 | 242,017 | 0.1798% |

Gallery 的 74 张有效图片只覆盖 70/3,000 个身份，即 `2.33%`。98 张有效 Mated
Probe 中，94 张的官方身份没有任何有效 Gallery，因此不能进入 1:N 检索；最终只有
4 张可评分，占全部 Mated 的 `0.00662%`。

## 无阈值 Mean Prototype 结果

Phase 4 已在自然 LFW 上完成六种聚合方法的公平联合比较，并选出 Mean Prototype。
QMUL 的目的不是重复方法选择，所以只为每个有效身份建立归一化均值原型，并保存每张
可评分 Probe 的第一候选、第二候选、最高分和候选分差。

| 指标 | Mated | Unmated |
| --- | ---: | ---: |
| 可评分 Probe | 4 | 264 |
| Rank-1 正确 | 0 | 不适用 |
| 最高分中位数 | 0.4747 | 0.4959 |
| 最高分范围 | 0.4259～0.5627 | 0.1694～0.8097 |
| 候选分差中位数 | 0.0352 | 0.0344 |
| 候选分差最大值 | 0.0616 | 0.1638 |
| Mean Prototype 检索耗时 | 0.0094 ms/Probe | 0.0094 ms/Probe |

4 张 Mated 的分母过小，`0/4` 不能被包装成稳定识别率；它只说明在当前少量可评分
交集上也没有正确 Rank-1。Unmated 最高分可以高于 Mated，进一步说明不能把自然 LFW
的分数分布解释为监控域分布。

## LFW 工作点跨域迁移诊断

Phase 4 的 LFW 主工作点为：

```text
Mean Prototype
match_threshold = -1.0
min_score_gap = 0.50780195
```

该工作点机械应用到 QMUL 的 268 张可评分 Probe 后，Known 接受、Known 错误接受和
Unknown 错误接受均为 0。原因不是系统在 QMUL 达到理想 FPIR，而是所有候选分差都
低于 `0.5078`，系统发生全拒绝，TPIR 同样为 0。该结果证明 LFW 工作点不能跨域直接
部署，也不能用“FPIR=0”掩盖检测覆盖和全拒绝问题。

QMUL 标签没有用于搜索新阈值或候选分差；本阶段不在压力测试集上反向调参。

## 系统边界

QMUL-SurvFace 与本项目的外置桌面摄像头展示存在明显域差异。下面这些方向已有研究，
但需要新模型、训练数据或完整跟踪系统，不列入课程 TODO：

- 监控小脸专用检测与识别模型；
- 超分辨率或人脸复原与身份保持联合训练；
- 视频跟踪、跨帧对齐和多帧特征融合；
- 监控域到桌面/网页人脸域的域适配；
- 面向该域重新训练并在独立协议上标定工作点。

本课程设计继续解决可验证的桌面域问题：单帧偶发模糊/曝光波动、多帧择优、少量身份
Gallery 的工作点复核和完整应用 E2E 耗时。

## 本地产物与复现

本地产物都位于 Git 忽略的 `data/`：

- `data/logs/cache/qmul_survface_raw.sqlite`：242,453 条原始成功/FTE 缓存；
- `data/experiments/phase5/qmul_raw_extraction_manifest.json`；
- `data/experiments/phase5/qmul_decision_scores.sqlite`：268 条无阈值候选分数；
- `data/experiments/phase5/qmul_pressure_report.json`。

```bash
.venv/bin/python scripts/extract_qmul_raw_embeddings.py \
  --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace

.venv/bin/python scripts/evaluate_qmul.py \
  --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace \
  --cache-path ./data/logs/cache/qmul_survface_raw.sqlite \
  --transfer-policy ./data/experiments/phase4/final_evaluation.json
```

第二条命令只读取缓存、协议和 Phase 4 工作点，不导入或初始化 `FaceEngine`。
