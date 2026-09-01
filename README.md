# 摄像头人脸比对系统

离线、跨平台的开放集 1:N 人脸识别课程设计项目。程序从外置摄像头或本地图片读取人脸，在本地标准库中检索已录入人员；证据不足时输出“未知人员”，不会强行给出姓名。

当前版本先保证主链路可运行，再按阶段完成算法比较、数据标定、鲁棒性优化和最终展示。当前代码的准确流程见 [design.md](design.md)，任务书技术要求见 [docs/任务书要求提取.md](docs/任务书要求提取.md)。

## 项目目标

- 调用外置摄像头，显示实时画面并支持抓拍。
- 建立不少于 3 个身份的标准人脸库。
- 对输入图片完成人脸检测、特征提取和开放集 1:N 识别。
- 匹配成功时显示姓名和相似度，不满足规则时显示“未知人员”。
- 通过 UI 从摄像头或本地图片新增人员、追加样本。
- 保留“发现问题—实施优化—同条件复测”的量化过程，完成至少一项可展示、可量化的扩展功能。

## 任务书完成度

| 任务书要求 | 当前状态 | 完成度判断 |
| --- | --- | --- |
| 外置摄像头实时采集与拍照 | 已实现跨平台设备扫描、预览线程、停止清屏和抓拍入口 | **代码已实现；需要真实硬件验证** |
| 不少于 3 个身份的标准库 | 已实现人员、图片和 embedding 的持久化；尚未建立最终 Demo Gallery | **代码已实现；最终数据待 Phase 6** |
| 姓名/未知人员判别 | 已实现单脸检查、1:N 打分、阈值和候选差距拒识 | **代码已实现；需要数据标定** |
| 系统 UI | 已实现识别页、标准库页、状态和异常反馈 | **代码已实现；需要真实硬件与交互验收** |
| 标准库动态扩容 | 已实现本地图片和当前画面新增、追加及重启恢复 | **代码已实现；需要完整链路验收** |
| 至少一项扩展功能 | 计划优先实现“短时间窗口多帧采集 + 质量择优”，并与单帧基线同条件比较 | **尚未完成扩展** |
| 优化前后对比证据 | 已有实验脚本和 LFW 小样本烟雾测试，但没有正式标定/独立评测结果 | **需要数据标定与鲁棒性实验** |

“代码已实现”只说明相应路径存在且自动化测试通过，不等于已经完成真实摄像头、正式数据集或现场条件验收。

## 当前实现

- 以 Python 源码项目运行，不制作 EXE；推荐 Python 3.11–3.13。
- PySide6 提供桌面 UI，OpenCV 负责摄像头和图片读写。
- InsightFace `buffalo_l` 与 ONNX Runtime CPU 提取人脸 embedding，模型从本地 `data/models/` 加载。
- 摄像头帧和本地图片使用同一套检测、质量检查、特征提取和识别服务。
- 创建人员时必须同时提供至少一张有效图片；人员和首批样本原子写入后立即参与识别。没有固定姿态、动作或图片数量要求。
- 同一人员可以继续追加多张样本。当前查询时与库中所有人员的所有样本比对，再按人员进行 Top-K 质量加权聚合。
- SQLite 保存人员、embedding、质量元数据和识别日志；样本原图保存在 `data/faces/`。
- 图片与向量分别记录 SHA-256，用于发现文件缺失、误覆盖或向量 BLOB 被改写。
- 默认阈值只是初始运行参数，尚未使用独立 Calibration/Evaluation 分区标定，不能当作最终最优参数。
- 当前 LFW pilot 只证明数据集下载、协议生成、真实模型提取和指标导出链路能运行，不作为正式效果结论。

## 阶段 TODO

### Phase 0：录入规则清理（已完成）

- [x] 人员记录与至少一张有效样本原子创建，创建成功后立即参与识别。
- [x] 删除固定姿态流程和多张样本激活门槛。
- [x] 删除空人员状态字段、旧数据库迁移和对应旧接口、测试、文档。
- [x] 配置文件只接受当前完整结构，不解析旧字段或为缺失字段回退。
- [x] 修正任务书提取和当前实现说明。

### Phase 1：基础链路验收

- [ ] 建立可随时删除的 Development Gallery，不导入最终展示人员。
- [ ] 验证外置摄像头扫描、预览、停止清屏和抓拍录入。
- [ ] 验证单张/多张本地图片录入、当前画面录入和动态追加。
- [ ] 验证 Known、Unknown、低质量输入、多人脸和系统错误提示。
- [ ] 验证关闭程序后重启，人员和样本仍可正常读取和识别。
- [ ] 记录问题清单；这些临时身份和结果不进入最终验收数据。

### Phase 2：算法基线

- [ ] 统一 Gallery、Known Probe、Unknown Probe 的数据流和结果格式。
- [ ] 实现并比较 Single、Max、Mean Prototype 和当前 Top-K 聚合。
- [ ] 明确区分 Unknown、Low Quality、FTE/FTA 和系统错误。
- [ ] 固定可回放的输入与指标，避免不同算法使用不同样本。

### Phase 3：数据集调参

- [ ] 先用与桌面摄像头条件接近的开发图片建立 Gallery、Known Probe、Unknown Probe 和 Calibration 分区。
- [ ] 用 Calibration 标定身份匹配阈值及确有必要的候选参数。
- [ ] 使用身份隔离、未参与调参的 Open-set Evaluation 分区复测。
- [ ] 输出 TPIR、FPIR、FNIR、Rank-1、FTE、FTA 和端到端耗时。
- [ ] 保存协议、参数、逐样本结果和可复现命令。
- [ ] LFW 继续用于链路和算法烟雾测试；XQLFW 只补充跨质量诊断；QMUL-SurvFace 只作为极端监控场景压力测试，不直接决定摄像头场景阈值。

### Phase 4：鲁棒性优化

- [ ] 在正常光、可恢复的暗光、轻中度侧脸、轻度运动模糊和距离变化下建立可重复的摄像头测试样本；复杂背景主要检查检测框和多人脸处理。
- [ ] 先完成一项主扩展：在短时间窗口内采集多帧，通过现有质量指标选择最佳帧，再进入识别。
- [ ] 用单帧抓拍作为基线，在相同人员、场景和阈值下比较成功率、拒识率与耗时。
- [ ] 时间允许时，再比较 Mean Prototype、当前 Top-K 和简单 Quality-aware Prototype；没有完整实验就不加入最终功能声明。
- [ ] 保存逐样本结果、失败样例、优化收益和时间代价，形成任务书要求的“优化前—优化后”证据。

### Phase 5：参数冻结与 UI 收尾

- [ ] 冻结模型版本、人员表示、质量规则、阈值和运行配置。
- [ ] 完成 UI 视觉、状态反馈和错误提示收尾。
- [ ] 补齐操作说明、数据目录复制、故障排查和跨平台检查。
- [ ] 冻结后不再使用最终 Demo Gallery 反向调参。

### Phase 6：最终 Demo Gallery 与验收

- [ ] 参数冻结后再导入“本人 + 少量公开身份”。
- [ ] Gallery 至少包含 3 个身份；具体数量在本阶段根据展示需要决定。
- [ ] 本人站到摄像头前演示 Known，同学临时测试 Unknown。
- [ ] 演示动态新增身份、追加样本、重启恢复和离线运行。
- [ ] 最终 Demo Gallery 不参与前面的阈值标定或鲁棒性调参。

公开数据集只用于开发阶段的算法选择、调参和独立评测。最终 Demo Gallery 要等算法、阈值和鲁棒性方案冻结后再建立，不能提前混入开发数据或调参过程。

## 可实现优化与系统局限性

扩展功能是课程加分项，不是可有可无的装饰。本项目会优先做能在现有 CPU 桌面应用中完整实现、复测和解释的优化：输入质量门控、多帧质量择优，以及时间允许时的简单质量感知模板。这类方法有明确的研究依据，例如多图质量加权聚合的 [Quality Aware Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Quality_Aware_Network_CVPR_2017_paper.html)，以及用于估计人脸图像可识别性的 [SER-FIQ](https://openaccess.thecvf.com/content_CVPR_2020/html/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_CVPR_2020_paper.html) 和 [MagFace](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html)。课设实现采用可解释的工程简化，不声称复现这些论文的训练方法或效果。

下面的问题不放进 TODO，也不承诺在课设中解决；最终报告会把它们作为系统局限性，并保留失败样例：

- 极暗环境中脸部信号已经丢失，或严重运动模糊、强遮挡、极端侧脸导致关键身份信息不可见。
- 远距离小脸、低分辨率监控画面和摄像头跨域泛化。QMUL-SurvFace 针对的正是更困难的监控小脸场景，与桌面 UVC 摄像头展示条件不同。
- 支付级安全，包括活体检测、照片/屏幕重放攻击防护、传感器可信链和攻击者模型。本项目的 SHA-256 只发现误删、误替换和局部损坏，不是安全认证机制。
- 对所有相机、肤色、年龄、光照和姿态给出统一准确率保证。当前自定义质量分数是启发式指标，不是经过大规模人群与设备标定的 FIQA 模型。

多帧择优可以降低偶发眨眼、轻度模糊和单帧曝光波动的影响，但不能恢复已经丢失的图像信息。XQLFW 可用于观察跨质量退化，[QMUL-SurvFace](https://arxiv.org/abs/1804.09691) 可用于压力测试；它们都不能替代与实际摄像头条件接近的 Calibration 和最终现场验收。

## 安装

Windows、Linux、macOS 均使用独立虚拟环境。Linux/macOS 示例：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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
4. 对所有人员排序，检查最佳分数阈值和第一/第二候选差距。

只有两项都满足时才输出姓名：

```text
最佳人员分数 >= 当前探针质量等级的 match_threshold
最佳人员分数 - 第二人员分数 >= 当前探针质量等级的 min_margin
```

只有一个候选人员时没有第二名，只检查匹配阈值。无人脸、多人脸、脸过小、明显模糊或质量等级为 `reject` 时不会进入身份打分。

## 数据目录与一致性检查

```text
data/
├─ config.toml                    # 识别阈值和质量规则
├─ face_library.sqlite            # 人员、向量、质量元数据、识别日志
├─ faces/<person-id>/             # 已入库样本图片
├─ models/buffalo_l/              # 离线 ONNX 模型
├─ datasets/                      # 开发数据集、协议（可选）
└─ logs/                          # 评分记录和实验报告
```

SQLite 开启外键、WAL、busy timeout 和短写事务。人员与一批样本的创建使用同一个数据库事务；图片先写入 staging 目录，全部成功后再进入正式目录。项目不兼容旧数据库和旧配置；结构变化后删除开发用 `face_library.sqlite` 与过期 `config.toml`，由程序按当前结构重新创建。

SHA-256 检查针对已入库参考图片和 SQLite 中的 embedding BLOB，不检查摄像头当前帧，也不参与相似度计算。它用于发现误删、误替换和局部数据损坏，不用于抵御能够同时改写图片、向量和哈希值的攻击者。

`data/` 被 Git 忽略，可能包含人脸图片、模型和实验数据。

## 当前 LFW 烟雾测试

现有脚本可下载 LFW、生成一个小型开放集协议并验证真实模型链路：

```bash
python scripts/prepare_lfw.py --data-dir ./data --download \
  --known-identities 3 --unknown-identities 3 \
  --enrollment-per-identity 5 --probes-per-identity 2

python scripts/evaluate_lfw.py --data-dir ./data --min-face-size 80
```

输出包括逐 Probe 分数和汇总指标。该 pilot 没有独立的 Calibration/Open-set Evaluation 身份划分，样本量也不足，因此只作为链路烟雾测试。正式阈值和鲁棒性结论必须等 Phase 3、Phase 4 完成后再给出。

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m ruff check .
python -m compileall -q src scripts tests
```

自动化测试不访问网络、个人照片、本地模型或真实摄像头。外置摄像头、公开数据集和现场 UI 仍需按对应阶段手工验收。

## 摄像头故障排查

- Linux：`ls /dev/video*` 查看系统识别的设备，并确认当前用户有视频设备权限。程序通过 OpenCV 设备索引打开，Linux 优先 V4L2，不写死 `/dev/video0`。
- Windows：在隐私设置中允许桌面应用访问摄像头；程序优先 DirectShow，失败时回退 OpenCV 通用后端。
- macOS：在“隐私与安全性”中授予终端或 Python 摄像头权限；程序优先 AVFoundation。
- 模型缺失：在有网络的机器运行 `scripts/prepare_models.py`，再复制 `data/models/`。
