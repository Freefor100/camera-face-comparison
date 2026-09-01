# 摄像头人脸比对系统

离线、跨平台的开放集 1:N 人脸识别课程设计项目。程序可从外置摄像头或本地图片读取人脸，在本地标准库中识别已录入人员；证据不够时输出“未知人员”，不会强行猜测姓名。

设计逻辑、数据边界、完整性校验和真实 LFW pilot 结果见 [design.md](design.md)。

## 功能

- PySide6 桌面界面：摄像头预览、抓拍识别和选择本地图片识别。
- OpenCV 摄像头适配：Linux 优先 V4L2，Windows 优先 DirectShow，macOS 优先 AVFoundation；界面通过设备索引扫描，不把 `/dev/video0` 写死。
- InsightFace `buffalo_l` 本地 ONNX 模型 + ONNX Runtime CPU 推理；日常演示不需要网络。
- 开放集 1:N 比对：归一化向量余弦相似度、每人 Top-K 质量加权聚合、质量分级阈值和第一/第二候选差距拒识。
- 人员库动态扩容：从本地多选图片或当前摄像头画面新增、追加样本；没有固定动作或固定顺序要求。
- 草稿状态：人员达到默认 3 张合格样本才变为 `active` 并参与识别，避免单张偶然样本直接入库。
- 数据完整性：样本图和向量 BLOB 分别保存 SHA-256；发现文件替换、缺失或向量改写时停止信任该库。
- 可复现实验：可下载 LFW，生成固定开放集协议，导出 FPIR、FNIR、Rank-1、拒识率和端到端耗时。

## 安装

推荐 Python 3.11–3.13，Windows、Linux、macOS 均可运行。

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

在能联网的开发机器准备一次模型；它会下载到可搬运的 `data/models/`：

```bash
python scripts/prepare_models.py --data-dir ./data
```

演示机器只需复制源码和整个 `data/` 目录。模型加载前会关闭 Albumentations 的在线版本检查，因此正常启动不会因网络不可用而失败。

## 启动与操作

```bash
python -m camera_face_comparison --data-dir ./data
```

### 实时比对

1. 点击“刷新设备”，选择外置摄像头，启动预览。
2. 点击“抓拍并比对”，或点击“选择本地图片”。两种输入走同一套识别规则。
3. 结果显示姓名、相似度、处理耗时；分数或候选差距不够时显示未知人员和原因。
4. 点击“停止预览”会清空画面与最后一帧，避免误把旧画面当作实时视频。

### 标准人脸库

- “从本地图片新增人员”：输入姓名后可多选图片。每张都必须检测到单张、质量合格的人脸。
- “从当前画面新增人员”：把当前画面作为一张样本创建人员；样本不足时显示草稿状态。
- “为选中人员导入图片”或“添加当前画面”：为已有人员追加任意数量的样本。
- 人员达到 `data/config.toml` 中的 `min_active_samples`（默认 3）后自动激活。姿态提示可以作为采集建议，但程序不要求五个固定动作。

## 识别规则

待测图片不会与标准库原图做像素级比较。系统把单张合格人脸转为 L2 归一化特征向量 `q`，与每张参考样本 `e` 计算余弦相似度：

```text
sim(q, e) = q · e
```

每个人先取最好的 Top-K 样本（默认 3），再按样本入库质量加权。只有同时满足下列条件才输出姓名：

```text
最佳人员分数 >= 当前探针质量等级的 match_threshold
最佳人员分数 - 第二人员分数 >= 当前探针质量等级的 min_margin
```

图片过暗、过曝、模糊、对比度低、脸太小、无人脸或可信人脸超过一张，会在比对前拒绝。高质量与中等质量探针使用不同阈值；阈值位于 `data/config.toml`，不是写死在算法中。

## 数据目录与完整性

```text
data/
├─ config.toml                    # 阈值、质量门槛、激活样本数
├─ face_library.sqlite            # 人员、向量、质量元数据、识别日志
├─ faces/<person-id>/             # 已入库样本图片
├─ models/buffalo_l/              # 离线 ONNX 模型
├─ datasets/                      # LFW 归档、图片与协议（可选）
└─ logs/                          # LFW 评分记录和实验报告
```

SQLite 开启外键、WAL 和短写事务。每张已入库图片和每个 `float32` 向量 BLOB 都有独立 SHA-256。检查的是标准库参考资料，不会给摄像头当前帧做哈希；目的是发现样本图被外部覆盖、图片缺失或数据库向量被改写。它不是防御拥有完整磁盘控制权的攻击者。

`data/` 被 Git 忽略，里面可能含有人脸图片、模型和实验数据。

## LFW 真实评测

下载和评测都需要显式运行，应用本身不会自动下载数据。

```bash
python scripts/prepare_lfw.py --data-dir ./data --download \
  --known-identities 3 --unknown-identities 3 \
  --enrollment-per-identity 5 --probes-per-identity 2

python scripts/evaluate_lfw.py --data-dir ./data --min-face-size 80
```

第一条命令下载 LFW funneled 归档，支持断点续传，并以 scikit-learn 公布的 SHA-256 校验后再解压。它生成 `data/datasets/lfw_open_set_protocol.json`：已知人员的模板图和探针图分离，未知人员从不进入图库。

第二条命令使用本地模型逐张提取真实向量，输出：

- `data/logs/lfw_scores.jsonl`：可回放的逐探针分数、样本质量和耗时；
- `data/logs/lfw_evaluation_report.json`：基础版与优化版的 FPIR、FNIR、Rank-1、未知拒识率、误识数和平均耗时。

`--min-face-size 80` 只适用于 250×250 的 LFW 评测图，不会修改摄像头运行的 `config.toml`。当前仓库已跑过一次小样本真实 pilot；结果、拒绝图片数量和限制写在 [design.md](design.md#7-开源数据集实验)。小样本上没有观察到优化优于基线的差异，不能据此宣称提升。

若需要校准阈值，应从最终评测集之外的留出集导出 JSONL，再分别校准高质量和中等质量探针：

```bash
python scripts/calibrate_thresholds.py --data-dir ./data \
  --scores ./data/calibration_high.jsonl --quality-tier high

python scripts/calibrate_thresholds.py --data-dir ./data \
  --scores ./data/calibration_medium.jsonl --quality-tier medium
```

使用同一份已保存评分记录重复生成基线/优化版对比：

```bash
python scripts/evaluate_experiment.py \
  --data-dir ./data --scores ./data/logs/lfw_scores.jsonl
```

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m ruff check .
```

自动化测试不访问真实摄像头、网络、个人照片或本地模型。LFW 脚本和摄像头流程需要按上面的真实命令手工验收。

## 摄像头故障排查

- Linux：`ls /dev/video*` 查看系统识别的设备，并确认当前用户有视频设备权限。程序仍通过 OpenCV 索引打开设备。
- Windows：在系统隐私设置中允许桌面应用访问摄像头，关闭占用摄像头的软件。
- macOS：在“隐私与安全性”中授予终端或 Python 摄像头权限。
- 模型缺失：在有网络的机器运行 `prepare_models.py`，然后复制 `data/models/`。
