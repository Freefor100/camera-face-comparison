# 已知问题台账

本文件负责把验收阶段发现的问题传递给后续阶段。开始一个阶段前，先读取“处理阶段”包含该阶段且状态未关闭的记录；只有达到“关闭条件”并补充可复现的“关闭证据”后，才能关闭问题。

状态只使用以下值：

- `待处理`：问题已复现，尚未达到关闭条件。
- `已接受`：确认是当前数据域或项目范围限制，不通过临时调参掩盖。
- `已关闭`：关闭条件已经满足，并已记录复测证据。

| ID | 状态 | 首次发现阶段 | 复现证据 | 影响 | 处理阶段 | 关闭条件 | 关闭证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ALG-001` | 待处理 | Phase 1 | 在 commit `104a4b9` 上执行 `python scripts/evaluate_lfw.py --data-dir ./data --min-face-size 80`。相同的 7 个有效 Probe 中，Max 基线 Known 为 4/4、Unknown 为 3/3；当前 Top-K/质量加权策略 Known 为 3/4、Unknown 为 3/3。原始报告：`data/logs/phase1_lfw_report.json`，SHA-256 `ab41823261e4c9795b22886463a31697e395fe653e89774cdcbc89911324479e`。 | 当前聚合策略在这个小样本回放中多拒绝了一个 Known；样本量不足，不能据此改阈值或宣称 Max 更优。 | Phase 2 比较聚合算法；Phase 3 使用独立 Calibration 标定阈值并复测。 | 在固定协议和相同有效 Probe 上完成 Single、Max、Mean Prototype、当前 Top-K 的同条件比较，确定默认聚合方法；再用与 Evaluation 身份隔离的数据标定阈值，并在独立 Open-set Evaluation 上记录 FPIR/FNIR/Rank-1。 | — |
| `DATA-001` | 已接受 | Phase 1 | 250×250 LFW 图片在应用默认 `min_face_size_px=112` 下没有可用 Gallery；改用评测参数 `--min-face-size 80` 后真实模型链路可运行。 | LFW 小图不能直接使用桌面摄像头正式质量门槛；若强行降低默认门槛，会把数据集适配混入应用策略。 | Phase 1 接受限制；Phase 3 继续使用显式评测配置。 | 应用默认门槛不因 LFW 临时降低；所有 LFW 结果明确记录评测专用 `min_face_size_px=80`，不把结果表述为默认应用效果。 | 本台账与 [Phase 1 验收记录](phase-1-validation.md) 已区分应用配置和评测配置。 |
| `UI-001` | 已关闭 | Phase 1 | `RecognitionResult` 已包含 `runner_up_score`，但旧 `MainWindow.on_recognition_result()` 只显示最高相似度和耗时，无法观察候选差距。 | UI 丢失服务层已经计算出的判定证据，不满足主识别页验收条件。 | Phase 1 | Matched/Unknown 均显示候选差距；没有第二候选时显示 `--`；针对性 UI 行为测试通过。 | commit `8cad5fd`；`tests/test_ui.py::test_recognition_result_shows_candidate_gap` 通过。 |
| `UI-002` | 已关闭 | Phase 1 | 用户在预览运行时点击“刷新设备”，已占用的 `/dev/video0` 被再次扫描，OpenCV 打开失败后设备下拉列表变空；终端保留 V4L2 打开失败证据。 | 预览仍在运行但列表与状态失真，停止后用户无法直接按原选择重启。 | Phase 1 | 预览运行时禁用刷新和设备选择；逻辑入口不执行二次扫描且保留列表；停止后控件恢复。 | commit `a5cc738`；`tests/test_ui.py::test_main_window_shows_library_and_updates_camera_controls` 通过，最终全量 44 passed。 |

## 登记规则

- 崩溃、线程/按钮状态错误、摄像头打开或停止错误、图片保存失败、重启丢失、UI 与服务结果不一致，作为 Phase 1 功能缺陷处理。
- 误拒、误识、聚合效果、质量门宽严和阈值选择，登记为 `ALG-*` 或 `DATA-*`，不在 Phase 1 的少量样本上临时调参。
- 新记录必须包含稳定复现证据、影响、明确处理阶段和可验证关闭条件；“看起来好了”不能作为关闭证据。
