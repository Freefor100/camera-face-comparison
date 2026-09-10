# 摄像头人脸比对系统

离线、跨平台的开放集 1:N 人脸识别课程设计项目。程序从外置摄像头或本地图片读取人脸，在本地标准库中检索已录入人员；证据不足时输出“未知人员”，不会强行给出姓名。

当前版本先保证主链路可运行，再按依赖顺序完成质量验证、开放集规则标定、鲁棒性扩展和最终展示。当前代码的准确流程见 [design.md](design.md)，任务书技术要求见 [任务书要求提取](docs/任务书要求提取.md)，需求分析见 [需求分析](docs/需求分析.md)，术语见 [术语与内部命名](docs/术语与内部命名.md)，当前跨质量 1:N 实验契约见 [Phase 5B 计划](docs/phase-5b-cross-quality-open-set-plan.md)，论文与标准依据见 [开放集识别与质量评估调研](docs/技术调研-开放集识别规则与质量评估.md)。

## 项目目标

- 调用外置摄像头，显示实时画面并支持抓拍。
- 建立不少于 3 个身份的标准人脸库。
- 对输入图片完成人脸检测、特征提取和开放集 1:N 识别。
- 匹配成功时显示姓名和相似度，不满足规则时显示“未知人员”。
- 通过 UI 从摄像头或本地图片新增人员、追加样本。
- 保留“发现问题—实施优化—同条件复测”的量化过程，按开发计划完成可展示、可量化的鲁棒性和端到端时间优化。

## 任务书完成度

| 任务书要求 | 当前状态 | 完成度判断 |
| --- | --- | --- |
| 外置摄像头实时采集与拍照 | 已实现跨平台设备扫描、预览线程、停止清屏和抓拍入口 | **Phase 1 真实硬件链路已通过** |
| 不少于 3 个身份的标准库 | 已实现人员、图片和 embedding 的持久化；尚未建立最终 Demo Gallery | **代码已实现；最终数据待 Phase 6** |
| 姓名/未知人员判别 | 已实现单脸检查、1:N 打分、阈值和候选分差拒识 | **LFW 联合标定已完成；桌面域参数待复核** |
| 系统 UI | 已实现识别页、标准库页、状态和异常反馈 | **Phase 1 功能链路已通过；视觉收尾在 Phase 5** |
| 标准库动态扩容 | 已实现本地图片和当前画面新增、追加及重启恢复 | **Phase 1 完整链路已通过** |
| 扩展优化功能 | 已完成未使用模型裁剪及全量同条件性能复测；多帧择优和完整 E2E 优化仍待实现 | **时间优化已部分完成；鲁棒性扩展待做** |
| 优化前后对比证据 | 已完成质量实验、六方法联合标定、XQLFW/QMUL 压力实验和推理耗时 A/B；桌面多帧尚未复测 | **Phase 2～5 已形成阶段证据** |

“代码已实现”只说明相应路径存在且自动化测试通过，不等于已经完成真实摄像头、正式数据集或现场条件验收。

## 当前实现

- 以 Python 源码项目运行，不制作 EXE；推荐 Python 3.11–3.13。
- PySide6 提供桌面 UI，OpenCV 负责摄像头和图片读写。
- InsightFace `buffalo_l` 与 ONNX Runtime 提取人脸 embedding；Linux 上自动优先使用 CUDA，实际 CUDA session 不可用时回退 CPU。模型从本地 `data/models/` 加载。
- 摄像头帧和本地图片使用同一套检测、质量检查、特征提取和识别服务。
- 创建人员时必须同时提供至少一张有效图片；人员和首批样本原子写入后立即参与识别。没有固定姿态、动作或图片数量要求。
- 同一人员可以继续追加多张样本。当前查询时与库中所有人员的所有样本比对，再按人员进行 Top-K 质量加权聚合。
- SQLite 保存人员、embedding、质量元数据和识别日志；样本原图保存在 `data/faces/`。
- 图片与向量分别记录 SHA-256，用于发现文件缺失、误覆盖或向量 BLOB 被改写。
- 默认阈值和当前质量加权聚合只是运行中的初始策略，尚未经过质量规则冻结和联合标定，不能当作最终最优方案。
- Phase 2 的 3,842 张有效 Probe 结果保留为“质量预筛为何会污染实验”的历史诊断。Phase 4 已重建与质量策略无关的 13,185 张自然 LFW embedding，并保存 5,720 张有效 Probe 的六种无阈值结果。
- Phase 3 已完成 10,108 条正式单因素质量测量。结果证明当前启发式质量门严重过严，`quality_score` 不应作为主算法的拒绝、加权或分层阈值依据；部署摄像头硬门仍需在真实摄像头域复核。
- Phase 4 在自然 LFW 严格工作点选出的实际规则是 `Mean Prototype + 仅候选分差`，并在独立 Evaluation 达到 `FPIR=0.252%`、`TPIR=48.45%`；旧报告中的最高分阈值 `-1` 只是关闭条件的哨兵值。该结果不是最终部署规则。
- Phase 5 的 XQLFW 全量实验已完成：共同有效 5,871 对上，原始 LFW 到跨质量变体的 10 折验证准确率由 98.48% 降至 94.14%，同人相似度均值下降 0.2594。
- 人脸引擎已停止执行未使用的性别年龄和额外关键点模型；同一批 7,263 张 XQLFW 图片上，有效 embedding 平均耗时由 32.293 ms 降至 23.896 ms，结果逐字节不变。该结果是推理子阶段优化，不等同于 E2E 优化。
- Phase 5 的 QMUL-SurvFace 全量压力实验已完成：242,453 张监控图片中仅 436 张产生 embedding，3,000 个 Gallery 身份仅 70 个可用；这被记录为监控小脸域限制，不用于修改桌面阈值。
- Phase 5 已完成 3～100 人小 Gallery 的 60 组 LFW 实验：完整 Gallery 工作点迁移后 Unknown FPIR 均为 0，风险已收窄到摄像头域；直接用 3 个 Known 身份重新标定的方差过大，不采用。

## 阶段 TODO

### Phase 0：录入规则清理（已完成）

- [x] 人员记录与至少一张有效样本原子创建，创建成功后立即参与识别。
- [x] 删除固定姿态流程和多张样本激活门槛。
- [x] 删除空人员状态字段、旧数据库迁移和对应旧接口、测试、文档。
- [x] 配置文件只接受当前完整结构，不解析旧字段或为缺失字段回退。
- [x] 修正任务书提取和当前实现说明。

### Phase 1：基础链路验收（已完成）

验收过程见 [Phase 1 基础链路验收记录](docs/phase-1-validation.md)，跨阶段问题由 [已知问题台账](docs/known-issues.md) 传递和关闭。

- [x] 建立可随时删除的 Development Gallery，不导入最终展示人员。
- [x] 验证外置摄像头扫描、连续采集、停止清屏、重开和摄像头来源录入。
- [x] 验证单张/多张本地图片录入、摄像头来源录入和动态追加。
- [x] 验证 Known、Unknown、低质量输入、多人脸和系统错误提示。
- [x] 验证关闭数据库后重开，人员、样本、完整性状态和识别链路仍然正常。
- [x] 记录问题清单；这些临时身份和结果不进入最终验收数据。

### Phase 2：固定评测数据和无阈值结果（已完成）

- [x] 使用 LFW 全量源协议固定 Gallery、Known Probe 和 Unknown Probe。
- [x] 在质量过滤前按身份固定 Calibration/Evaluation；同一 Known 或 Unknown 来源身份不跨分区。
- [x] 拆分 `embedding_extraction_id`、`quality_policy_id` 与 `decision_policy_id`，质量门和判定参数不再伪装成模型配置变化。
- [x] 从现有 8,577 条有效缓存生成 3,842 张有效 Probe 的无阈值结果，不重新运行 InsightFace。
- [x] 保存 Single、Max、Mean Prototype、Top-K Mean K=2/3/5，共 23,052 条第一/第二候选分数。
- [x] 保存 1,901 条 Probe 与 2,755 条 Gallery 拒绝记录、协议哈希、质量配置和代码版本。
- [x] 将固定阈值全量结果降级为历史诊断，不据此选择最终方法。
- [x] 固化开放集识别、模板聚合、FIQA 和大规模检索的技术调研及项目简化边界。

### Phase 3：质量门与入库拒绝规则验证

- [x] 将原始质量测量与质量策略判定拆开，保持应用现有行为不变。
- [x] 固定 Calibration 中全部 116 个有效 Known 身份和种子选择的 300 个 Unknown 身份；协议不含 Evaluation，身份和图片均无交集或缺失。
- [x] 分别改变 Probe 和入库参考图，保存可恢复的原始 embedding 与质量指标。
- [x] 单因素测试人脸尺寸、模糊、亮度和对比度，不制造全因素笛卡尔组合。
- [x] 记录 FTE、质量拒绝率、同人相似度、Rank-1、Unknown 最高候选分数分布和低质量参考图影响；最终 FPIR 留到 Phase 4 工作点统计。
- [x] 从 Baseline 实际指标断点比较硬门候选值，量化“错误减少量”和“额外拒绝量”。
- [x] 检查启发式 `quality_score` 的错误分桶及 Error-versus-Reject 曲线，确认其拒绝代价远大于收益。
- [x] 冻结算法结论：无证据的软质量拒绝、质量加权和质量分层阈值不进入最终主算法；桌面摄像头硬门留到 Phase 5 实拍复核。

### Phase 4：聚合方法与开放集判定联合标定

- [x] 在 CUDA 上重建与启发式质量策略无关的 13,185 张自然 LFW embedding；数据集有标签时选择主体脸，只保留 48 张模型 FTE。
- [x] 生成 5,720 张有效 Probe、34,320 条六方法无阈值分数，并验证 13,233 张缓存复读不触发模型推理。
- [x] 在相同 Calibration 上比较六种聚合方法，各自包含“仅匹配阈值”和“匹配阈值 + 候选分差”，共 12 组。
- [x] 使用实际 `top_score` / `score_gap` 断点精确扫描，不使用反向传播或固定网格。
- [x] 报告 `FPIR ≤ 1%`、主工作点 `FPIR ≤ 0.3%` 和 Calibration 观测 `FPIR = 0%`。
- [x] 按预定规则选出 `Mean Prototype + score_gap`，并只在独立 Evaluation 执行一次最终统计。
- [x] Evaluation 得到 Rank-1 98.60%、FPIR 0.252%、TPIR 48.45%；完整分母、参数和局限见 Phase 4 结果。
- [ ] 使用非最终演示人员的摄像头开发样本复核桌面域；该项随多帧扩展在 Phase 5 完成。

### Phase 5：鲁棒性扩展、耗时与 UI 收尾

- [x] 使用 XQLFW 官方 6,000 对完成真实跨质量分析，并与原始 LFW 的同协议、共同有效 Pair 比较。
- [x] 删除逐图执行的未使用模型，并用独立空缓存全量复测：有效 embedding 平均耗时下降 26.0%，7,263 条结果零差异。
- [x] 完整处理 QMUL-SurvFace 242,453 张官方图片，记录原始模型覆盖、FTE、可评分分母和 LFW 工作点全拒绝现象，并将监控小脸域列为系统限制。
- [x] 使用自然 LFW 做 3/5/10/25/50/100 人 Gallery、每档 10 次身份互斥复测，确认 LFW 域内缩小 Gallery 不会使完整 Gallery 工作点产生误接收。
- [ ] **Phase 5B：**补齐 XQLFW 13,233 张同路径 embedding，构造自然、跨质量 Probe、混合质量 Gallery 和全低质量压力场景。
- [ ] **Phase 5B：**显式比较最高分阈值、仅候选分差、两者联合和 NAC；主目标改为四个主要场景各自 `FPIR_valid≤1%`，并最大化最差场景 `TPIR_e2e`。
- [ ] **Phase 5B：**在 Calibration 内完成小 Gallery 稳定性检查，冻结一套统一聚合、规则和参数后只运行一次 Evaluation。
- [ ] 使用非最终演示人员建立临时摄像头开发集，检查剩余的桌面摄像头域偏移。
- [ ] 实现“短时间多帧采集 + 质量择优”，以单帧为基线做相同人员、场景、方法和阈值的复测。
- [ ] 记录输入、检测、特征、检索、判定、日志和 UI 的分阶段及 E2E 耗时，只优化实测瓶颈。
- [ ] 根据 Phase 3 结论删除数值质量硬门、启发式软质量加权和质量分层识别阈值；只保留输入有效性阻断，其他质量指标用于提示。
- [ ] 完成 UI 视觉、状态反馈、操作说明、故障排查和跨平台收尾。
- [ ] 冻结模型、质量门、聚合方法和判定参数。

### Phase 6：最终 Demo Gallery 与验收

- [ ] 参数冻结后再导入“本人 + 少量公开身份”。
- [ ] Gallery 至少包含 3 个身份；具体数量在本阶段根据展示需要决定。
- [ ] 本人站到摄像头前演示 Known，同学临时测试 Unknown。
- [ ] 演示动态新增身份、追加样本、重启恢复和离线运行。
- [ ] 最终 Demo Gallery 不参与前面的阈值标定或鲁棒性调参。

LFW 用于 Phase 2 固定分数、Phase 3 质量验证和 Phase 4 联合标定；XQLFW 与 QMUL 主要用于 Phase 5 鲁棒性和极端域压力分析，不直接决定桌面应用阈值。最终 Demo Gallery 要等模型、质量规则、聚合和阈值冻结后再建立。

## 可实现优化与系统局限性

扩展功能是课程加分项。本项目开发计划包含两个方向：一是“短时间窗口多帧采集 + 质量择优”，二是面向完整识别链路的端到端时间优化。前者的目标是减少单帧偶发模糊、眨眼或曝光波动导致的拒识；后者以实际端到端耗时为依据定位瓶颈并实施小范围优化。两者都必须保留优化前基线、同条件复测结果、收益和代价。多帧质量择优不声称解决极端暗光、严重模糊、强遮挡或极端侧脸等信息已经丢失的情况；时间优化也不以牺牲识别结果或异常处理为代价。这类质量评估和多样本聚合方法有明确的研究依据，例如 [Quality Aware Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Quality_Aware_Network_CVPR_2017_paper.html)、[SER-FIQ](https://openaccess.thecvf.com/content_CVPR_2020/html/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_CVPR_2020_paper.html) 和 [MagFace](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html)。课设只采用可解释的工程简化，不复现这些论文的训练方法或效果。

下面的问题不放进 TODO，也不承诺在课设中解决；最终报告会把它们作为系统局限性，并保留失败样例：

- 极暗环境中脸部信号已经丢失，或严重运动模糊、强遮挡、极端侧脸导致关键身份信息不可见。
- 远距离小脸、低分辨率监控画面和摄像头跨域泛化。QMUL-SurvFace 针对的正是更困难的监控小脸场景，与桌面 UVC 摄像头展示条件不同。
- 支付级安全，包括活体检测、照片/屏幕重放攻击防护、传感器可信链和攻击者模型。本项目的 SHA-256 只发现误删、误替换和局部损坏，不是安全认证机制。
- 对所有相机、肤色、年龄、光照和姿态给出统一准确率保证。当前自定义质量分数是启发式指标，不是经过大规模人群与设备标定的 FIQA 模型。

多帧择优可以降低偶发眨眼、轻度模糊和单帧曝光波动的影响，但不能恢复已经丢失的图像信息。XQLFW 可用于观察跨质量退化，[QMUL-SurvFace](https://arxiv.org/abs/1804.09691) 可用于压力测试；它们都不能替代与实际摄像头条件接近的 Calibration 和最终现场验收。

## 安装

Windows、Linux、macOS 均使用独立虚拟环境。必须明确选择 CPU 或 GPU 后端，不能同时安装两个 ONNX Runtime 包。

Linux/macOS 使用 CPU：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cpu,dev,evaluation]"
```

Linux（NVIDIA GPU）使用 GPU：

```bash
python -m pip install -e ".[gpu,dev,evaluation]"
```

程序启动时会检查 InsightFace session 的实际 provider；评测 JSON 的 `execution_backend` 字段是最终依据，不以 `nvidia-smi` 或 provider 列表单独判断。

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[cpu,dev,evaluation]"
```

在能联网的开发机器准备一次模型：

```bash
python scripts/prepare_models.py --data-dir ./data
```

脚本把模型放入可搬运的 `data/models/`。应用启动和正常演示只读取本地模型，不主动联网；演示机器需要复制源码和完整 `data/` 目录。

## 启动与操作

```bash
python -m camera_face_comparison --data-dir ./data
```

### 实时比对

1. 刷新设备并选择摄像头，然后启动预览。
2. 点击“抓拍并比对”，或点击“选择本地图片”。
3. 查看姓名/未知人员、相似度、判定原因和处理耗时。
4. 停止预览后，程序清空当前帧和最后显示画面。

### 标准人脸库

- “从本地图片新增人员”：输入姓名后选择一张或多张图片。
- “从当前画面新增人员”：把当前摄像头帧作为首张样本。
- 第一张通过单脸和质量检查的图片保存成功后，人员立即参与识别。
- “为选中人员导入图片”或“添加当前画面”可以继续追加样本。
- 多张样本可改善覆盖范围，但不是录入门槛，也不要求固定动作。

## 当前识别规则

系统不做原图像素级比较。输入人脸和标准库样本都被转换为 L2 归一化特征向量，余弦相似度为：

```text
sim(q, e) = q · e
```

当前实现对每个人员执行以下操作：

1. Query 与该人员的所有样本分别计算相似度。
2. 选择分数最高的 Top-K 个样本；样本少于 K 时使用全部样本。
3. 按入库质量分数加权，得到该人员的候选分数。
4. 对所有人员排序，检查最佳分数阈值和第一/第二候选分差。

只有两项都满足时才输出姓名：

```text
最佳人员分数 >= 当前探针质量等级的 match_threshold
最佳人员分数 - 第二人员分数 >= 当前探针质量等级的 min_score_gap
```

只有一个候选人员时没有第二名，只检查匹配阈值。无人脸、多人脸、脸过小、明显模糊或质量等级为 `reject` 时不会进入身份打分。

## 数据目录与一致性检查

```text
data/
├─ config.toml                    # 识别阈值和质量规则
├─ face_library.sqlite            # 人员、向量、质量元数据、识别日志
├─ faces/<person-id>/             # 已入库样本图片
├─ demo-candidates/<person-id>/   # 最终演示候选，当前不进入数据库
├─ models/buffalo_l/              # 离线 ONNX 模型
├─ datasets/                      # 开发数据集、协议（可选）
├─ experiments/phase2/            # 历史质量预筛分数
├─ experiments/phase3/            # 单因素质量实验
├─ experiments/phase4/            # 自然 LFW 分数、联合标定与独立评估
└─ logs/                          # 评分记录和实验报告
```

SQLite 开启外键、WAL、busy timeout 和短写事务。人员与一批样本的创建使用同一个数据库事务；图片先写入 staging 目录，全部成功后再进入正式目录。项目不兼容旧数据库和旧配置；结构变化后删除开发用 `face_library.sqlite` 与过期 `config.toml`，由程序按当前结构重新创建。

SHA-256 检查针对已入库参考图片和 SQLite 中的 embedding BLOB，不检查摄像头当前帧，也不参与相似度计算。它用于发现误删、误替换和局部数据损坏，不用于抵御能够同时改写图片、向量和哈希值的攻击者。

`data/` 被 Git 忽略，可能包含人脸图片、模型和实验数据。

## 当前数据集评测入口

现有脚本保留小型 LFW pilot、固定阈值报告和 Phase 2 质量预筛结果作为历史诊断。Phase 4 的正式算法比较使用自然 LFW 原始缓存和无阈值 SQLite；Phase 5 的 XQLFW 与 QMUL-SurvFace 均使用无质量门原始缓存。所有产物均在被 Git 忽略的 `data/` 下。

```bash
python scripts/prepare_lfw.py --data-dir ./data --download \
  --known-identities 3 --unknown-identities 3 \
  --enrollment-per-identity 5 --probes-per-identity 2

python scripts/evaluate_lfw.py --data-dir ./data --min-face-size 80
```

全量 LFW 协议覆盖本地 LFW 的每张图片：

```bash
python scripts/prepare_lfw.py --data-dir ./data --full \
  --known-fraction 0.8 --enrollment-per-identity 5 --seed 2026 \
  --output ./data/datasets/lfw_full_open_set_protocol.json
python scripts/evaluate_lfw.py --data-dir ./data --stream \
  --protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --min-face-size 80 \
  --cache-path ./data/logs/cache/lfw.sqlite \
  --report-output ./data/logs/lfw_full_algorithm_baseline.json

python scripts/extract_lfw_raw_embeddings.py --data-dir ./data
python scripts/export_lfw_decision_scores.py \
  --data-dir ./data \
  --source-protocol ./data/datasets/lfw_full_open_set_protocol.json \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --output-dir ./data/experiments/phase4
python scripts/calibrate_thresholds.py \
  --scores ./data/experiments/phase4/decision_scores.sqlite
python scripts/evaluate_selected_operating_point.py \
  --scores ./data/experiments/phase4/decision_scores.sqlite \
  --target-fpir 0.003
python scripts/evaluate_gallery_scale.py --data-dir ./data \
  --gallery-sizes 3 5 10 25 50 100 --repeats 10
```

XQLFW 使用官方 6,000 对、原始无质量门缓存和 10 折阈值；QMUL 使用官方 Gallery、Mated Probe 和 Unmated Probe：

```bash
python scripts/extract_xqlfw_raw_embeddings.py --data-dir ./data
python scripts/evaluate_xqlfw.py --data-dir ./data
python scripts/compare_xqlfw_domains.py
python scripts/prepare_qmul.py --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace
python scripts/extract_qmul_raw_embeddings.py --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace
python scripts/evaluate_qmul.py --data-dir ./data \
  --dataset-root ./data/datasets/qmul-survface/QMUL-SurvFace \
  --cache-path ./data/logs/cache/qmul_survface_raw.sqlite \
  --transfer-policy ./data/experiments/phase4/final_evaluation.json
```

这些结果必须分别记录图片覆盖数、模型失败数、有效分母、协议和耗时。LFW 算法候选已经完成独立评估，但桌面摄像头参数和鲁棒性扩展仍要等 Phase 5 实验后冻结。

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m ruff check .
python -m compileall -q src scripts tests
```

自动化测试不访问网络、个人照片、本地模型或真实摄像头。外置摄像头、公开数据集和现场 UI 仍需按对应阶段手工验收。

## 摄像头故障排查

- Linux：`ls /dev/video*` 查看系统识别的设备，并确认当前用户有视频设备权限。程序把设备索引交给 OpenCV，Linux 优先使用 V4L2；不自行读取 `/dev/video*` 字节流，也不把设备路径写死为 `/dev/video0`。节点存在不代表一定能提供采集画面，例如同一 UVC 设备可能同时暴露视频节点和非采集节点。
- Windows：在隐私设置中允许桌面应用访问摄像头；程序优先 DirectShow，失败时回退 OpenCV 通用后端。
- macOS：在“隐私与安全性”中授予终端或 Python 摄像头权限；程序优先 AVFoundation。
- 模型缺失：在有网络的机器运行 `scripts/prepare_models.py`，再复制 `data/models/`。
