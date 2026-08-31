# 摄像头人脸比对系统

基于外置摄像头的离线人脸比对课程设计项目。程序以 Python 源码形式跨平台运行：实时预览摄像头画面，采集标准人脸库，识别已录入人员并拒识未知人员。

## 功能

- 扫描并选择外置摄像头；Linux 使用 V4L2、Windows 优先 DirectShow、macOS 优先 AVFoundation，均通过 OpenCV 设备索引访问。
- 实时预览，点击“抓拍并比对”后显示人脸框、姓名/未知人员、相似度和处理耗时。
- 标准库至少支持三名人员；新增人员必须完成正脸、左转、右转、抬头、低头五步采样。
- SQLite 保存人员、样本元数据、特征向量和识别日志；样本图片独立存放，重启后仍可使用。
- 使用 InsightFace `buffalo_l` 的本地模型和 ONNX Runtime CPU 推理；演示时不依赖网络。
- 使用 Top-2 样本平均余弦相似度、匹配阈值和第一/第二候选差距判定，降低单样本偶然高分和未知人员误认。

## 运行环境

- Python 3.11（推荐；项目代码兼容 Python 3.11–3.13）。
- 64 位 Windows、Linux 或 macOS。
- 一台可被系统识别的外置 UVC 摄像头。
- 首次准备模型时需要网络；正常演示无需联网。

## 安装与离线模型准备

在项目根目录执行：

```bash
python3.11 -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

安装项目与开发依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

在有网络的机器上下载模型到本地 `data/models/`：

```bash
python scripts/prepare_models.py --data-dir ./data
```

将整个 `data/` 文件夹与源码一起复制到演示机器；不要在演示机器首次下载模型。

## 启动

```bash
python -m camera_face_comparison --data-dir ./data
```

应用启动后：

1. 在“实时比对”页刷新并选择摄像头，启动预览。
2. 在“标准人脸库”页新增至少三名人员；每人完成五种姿态采样。
3. 如需扩充某位已有人员，在“标准人脸库”选中该人员，点击“为选中人员追加样本”。
4. 回到实时比对页，单击“抓拍并比对”。
5. 已录入人员会显示姓名；分数不足或候选差距不足时显示“未知人员”。

## 数据目录

`--data-dir` 指定所有本地数据的位置：

```text
data/
├─ config.toml             # 阈值和样本质量规则
├─ face_library.sqlite     # 人员、特征向量与识别日志
├─ models/buffalo_l/       # 离线模型
├─ faces/<person-id>/      # 标准库样本图片
└─ logs/                   # 预留的导出目录
```

`data/` 已被 Git 忽略，因为它包含个人脸部数据和模型文件。请只在获得参与者同意的前提下采集并妥善保管这些数据。

## 人脸比对规则

待测图像不会和数据库原图做像素级比较。系统会先检测和校验单张人脸，再把人脸转成归一化特征向量。待测向量会与标准库全部样本向量计算余弦相似度；每名人员取最高两张样本分数的均值。只有同时满足以下条件才输出姓名：

```text
最佳人员分数 >= match_threshold
最佳人员分数 - 第二人员分数 >= min_margin
```

否则输出“未知人员”。默认阈值仅用于首次调试，最终应使用测试数据校准。

## 优化实验与阈值校准

保留基础版和优化版对同一测试集的结果。建议测试三名已知人员的正常光照、侧脸、暗光和眼镜/遮挡场景，并加入至少三名未知人员。

将每次测试的人员级分数保存为 JSONL，例如：

```json
{"expected_person_id":"<alice-id>","person_scores":{"<alice-id>":0.72,"<bob-id>":0.45}}
{"expected_person_id":null,"person_scores":{"<alice-id>":0.52,"<bob-id>":0.49}}
```

然后运行：

```bash
python scripts/calibrate_thresholds.py \
  --data-dir ./data \
  --scores ./data/calibration_scores.jsonl
```

脚本优先选择未知人员误认更少的阈值组合，再提高已知人员正确识别数，并将结果写入 `data/config.toml`。

使用同一份测试记录比较优化前后效果（该记录不含人脸图片，可单独保存）：

```json
{"expected_person_id":"<alice-id>","sample_scores":{"<alice-id>":[0.72,0.70],"<bob-id>":[0.45,0.42]},"latency_ms":118}
{"expected_person_id":null,"sample_scores":{"<alice-id>":[0.52,0.30],"<bob-id>":[0.49,0.27]},"latency_ms":123}
```

```bash
python scripts/evaluate_experiment.py \
  --data-dir ./data \
  --scores ./data/experiment_scores.jsonl
```

它会将基础版（单样本最高分）与优化版（Top-2 均值和候选差距）的已知人员正确数、未知人员拒识率、误识次数和平均耗时写入 `data/logs/optimization_report.json`，可直接作为课程设计的优化过程证据。

## 测试

```bash
python -m pytest
```

自动化测试不依赖真实摄像头、网络、模型文件或个人照片。真实摄像头、离线模型加载和界面流程需要按上面的手工步骤验收。

## 常见问题

- **Linux 未发现摄像头：** 检查 `ls /dev/video*` 是否列出设备，并确认当前用户拥有视频设备访问权限；应用仍以设备索引扫描，而不是把 `/dev/video0` 写死。
- **Windows 未发现摄像头：** 检查系统隐私设置是否允许桌面应用访问摄像头，并关闭正在占用摄像头的软件。
- **macOS 无法打开摄像头：** 在系统“隐私与安全性”中授予终端或 Python 摄像头权限。
- **启动提示模型缺失：** 在有网络的开发机器执行 `prepare_models.py`，再复制 `data/models/`。
