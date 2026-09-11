# 阶段 1 基础链路验收记录

## 状态

**已完成。** 真实摄像头连续采集、停止与重开，真实模型 LFW 回放，公开图片已登记人员/未知人员，摄像头样本录入与追加，异常输入、完整性检查和重启恢复均已通过。阶段 1 发现的两个用户界面功能缺陷已经分别修复并提交。

阶段 1 只验证现有主链路并修复可稳定复现的功能缺陷。识别率、聚合算法、质量拒绝规则和
阈值选择传递到 [已知问题台账](known-issues.md)，不在本阶段用少量样本临时调整。

## 环境与起点

| 项目 | 值 |
| --- | --- |
| 阶段 1 起点 commit | `104a4b9` (`refactor: remove legacy enrollment compatibility`) |
| 阶段 1 功能修复 | `8cad5fd`（显示候选分差）、`a5cc738`（预览期间禁止刷新设备） |
| 操作系统 | Linux `7.2.2-zen1-1-zen` x86_64 |
| Python | 3.13.9 |
| OpenCV | 4.14.0 |
| InsightFace | 0.7.3，`buffalo_l` 本地模型 |
| ONNX Runtime | 1.29.0，`CPUExecutionProvider` |
| PySide6 | 6.11.2 |
| 数据目录 | `./data`，被 Git 忽略，仅用于开发验收 |

## 阶段 0 里程碑验证

提交 `104a4b9` 前已执行：

| 命令 | 结果 |
| --- | --- |
| `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` | 43 passed，0 failed |
| `.venv/bin/python -m ruff check .` | All checks passed |
| `.venv/bin/python -m compileall -q src scripts tests` | 通过 |
| `git diff --check` | 通过 |

该提交不包含 `data/`、模型、LFW 图片或 `/tmp` 结果。

## 真实模型 LFW 回放

本阶段当时使用小规模试运行入口生成下列原始结果。该入口后来被无质量预筛的 LFW 全量
原始缓存与跨质量联合标定取代，当前可执行命令以 README 为准；本节只保留阶段 1
发生过的链路证据及文件哈希。

原始结果保存在被 Git 忽略的 `data/logs/`：

| 文件 | SHA-256 |
| --- | --- |
| `phase1_lfw_report.json` | `ab41823261e4c9795b22886463a31697e395fe653e89774cdcbc89911324479e` |
| `phase1_lfw_scores.jsonl` | `88fc88e625b5404eb18ec554407dbd53c99a46bc774ebcd7ea4c1e82fc4877ae` |

有效待识别图片共 7 个：已登记人员 4 个，未知人员 3 个。

| 策略 | 已登记人员正确 | 未知人员拒识 | 未知人员误接收率 | 已登记人员未正确识别率 | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最高样本分数基线 | 4/4 | 3/3 | 0.00 | 0.00 | 1070.89 ms |
| 当时的质量加权前 K 个样本分数平均 | 3/4 | 3/3 | 0.00 | 0.25 | 1070.89 ms |

拒绝样本包括多人脸和模糊输入。该回放使用评测专用 `min_face_size_px=80`；应用默认值 112 会拒绝全部 250×250 LFW 样本。两项结论分别登记为 `ALG-001` 与 `DATA-001`，本阶段不改聚合或阈值。

## 公开图片预检

所有图片只保存在 `data/phase1-inputs/`，不提交 Git。预检使用当时的默认配置、同一个
`FaceEngine`、同一套质量拒绝规则和同一个 `recognize_embedding()`，没有修改阈值。

| 用途 | 文件 | 真实模型结果 |
| --- | --- | --- |
| Obama 录入 | `barack_obama_2009.jpg` | `high`，质量分 0.8918，通过 |
| Obama 未入库已登记人员图片 | `barack_obama_2012_original.jpg` | `high`，与 2009 样本相似度约 0.759，通过当前判定 |
| Obama 后续追加 | `barack_obama_2016.jpg` | `high`，与 2009 样本相似度约 0.752，通过当前判定 |
| 未登记人员图片 | `donald_trump_2017.jpg` | `high`，仅含 Obama 的标准库下分数约 -0.059，判为未知人员 |

图片来源：2009 年 [Barack Obama 官方肖像](https://commons.wikimedia.org/wiki/File:Official_portrait_of_Barack_Obama.jpg)、2012 年 [Barack Obama 官方肖像原图](https://commons.wikimedia.org/wiki/File:President_Barack_Obama.jpg)、2016 年 [Barack Obama 肖像](https://commons.wikimedia.org/wiki/File:Barack_Obama_in_October_2016.jpg)、2017 年 [Donald Trump 官方肖像](https://commons.wikimedia.org/wiki/File:Donald_Trump_official_portrait.jpg)。

原计划候选中的 Obama 2012 PNG、Angela Merkel 与 Joe Biden 图片被当前门控判为模糊，另一张 Angela Merkel 候选检测到多人脸，因此没有把这些失败候选冒充已登记人员/未知人员正例，也没有为通过预检而降低门槛。

## 真实摄像头验收

宿主存在 `/dev/video0` 和 `/dev/video1`。通过项目 `CameraService` 使用 OpenCV/V4L2 扫描后，以 `Camera 0` 完成连续采集：

| 项目 | 结果 |
| --- | --- |
| 连续采集 | 15.07 秒、446 帧 |
| 帧尺寸 | 640×480、BGR 三通道 |
| 画面变化 | 下采样均值变化范围 15.58，证明不是重复静止缓冲帧 |
| 停止与重开 | `close()` 后重新 `open(0)`，首帧仍为 640×480 |
| 停止清屏 | 离屏用户界面行为测试验证当前帧、检测框与 Pixmap 被清除 |

验收中复现了“预览运行时点击刷新，已占用的 `video0` 扫描失败并清空设备列表”。修复后，预览期间设备下拉框和刷新按钮禁用，逻辑入口也不会重复扫描；停止预览后恢复。详见 `UI-002`。

## 真实模型主链路回放

回放使用用户实际摄像头录入并保存的 `liu` 样本作为 Camera 输入，使用公开高分辨率肖像作为本地图片输入，在临时数据库和临时人脸目录中运行；正式开发数据库、模型与阈值未被修改。原始报告保存在 `data/logs/phase1_mainflow_report.json`，SHA-256 为 `ff6f7bcccd06cfd2bb2824dd5047bfef1c32161b626541eb67038ebf3c005075`。

| ID | 验收条件 | 状态 | 证据/结果 |
| --- | --- | --- | --- |
| `P1-UI-01` | 刷新设备后至少出现一个可采集节点 | 通过 | 扫描得到 `Camera 0/1`；`Camera 0` 连续提供 446 帧。 |
| `P1-UI-02` | 预览连续运行至少 15 秒；停止后清除最后一帧；再次启动可恢复 | 通过 | 真实设备连续读取与重开通过；停止清屏由离屏用户界面测试验证。 |
| `P1-UI-03` | 用当前画面创建 `Phase1_Camera_User`，再追加一帧，样本数为 2 | 通过 | 使用真实摄像头已保存帧按 Camera 输入回放，创建 1 张、追加 1 张，最终 2 张。 |
| `P1-UI-04` | 抓拍本人得到已登记人员，并显示检测框、相似度、候选分差和耗时 | 通过 | `Phase1_Camera_User` 得分 1.000、第二候选 0.073、返回有效 bbox，耗时 717.90 ms；结果文本行为测试覆盖候选分差和耗时。 |
| `P1-UI-05` | 用 Obama 2009 创建人员；2012 原图得到已登记人员 | 通过 | 2012 未入库待识别图片命中 `Barack_Obama`，得分 0.759、第二候选 0.074。 |
| `P1-UI-06` | Donald Trump 2017 在非空开发标准库中得到未知人员 | 通过 | 最佳分 -0.005、第二候选 -0.059，原因 `score_below_threshold`。 |
| `P1-UI-07` | 一次追加 Obama 2012 原图和 2016 图片，样本数变为 3 | 通过 | 一次追加返回 2，Obama 最终样本数 3。 |
| `P1-UI-08` | `Adrien_Brody_0002.jpg` 提示多人脸；原始 LFW 单人脸提示脸过小；非图片文件提示打开失败 | 通过 | 原因依次为 `multiple_faces`、`face_size_below_minimum`、`could not decode image`。 |
| `P1-UI-09` | 关闭并重启后，人员、样本、完整性状态及已登记人员/未知人员链路仍可用 | 通过 | 重开后样本数仍为 Camera 2、Obama 3；完整性正常；Obama 再识别得分 0.844。 |

`P1-UI-08` 记录的是阶段 1 当时的历史行为。后续质量实验表明，不宜仅凭固定脸部尺寸或模糊阈值阻断普通桌面输入；当前版本已改为仅拒绝无法解码、无人脸、多人脸和无效特征向量，其他质量指标只提供调整提示。

## 阶段 1 最终验证

| 命令 | 结果 |
| --- | --- |
| `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` | 44 passed，0 failed |
| `.venv/bin/python -m ruff check .` | All checks passed |
| `.venv/bin/python -m compileall -q src scripts tests` | 通过 |
| `git diff --check` | 文档提交前通过 |

## 阶段结论

阶段 1 基础链路验收完成。该结论当时只表示摄像头、输入、录入、识别、拒绝、持久化和
用户界面状态链路可运行，不把当时的聚合、质量规则或阈值写成最终效果。`ALG-001` 后续
已由跨质量联合标定关闭，`DATA-001` 已通过删除没有实验证据支持的质量拒绝门槛关闭。开发标准库
仅用于本阶段，最终演示标准库仍在运行时和扩展功能固定后建立。
