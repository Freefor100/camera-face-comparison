# Phase 3～4 质量实验与联合标定实施计划

> **执行方式：** 当前任务使用测试驱动逐项实施；每个里程碑形成独立提交。实验数据、缓存和报告保存在 `data/`，不进入 Git。

**目标：** 先用 Calibration 身份验证数值质量门和入库拒绝规则，再用冻结后的有效数据联合选择人员聚合方法、匹配阈值及候选分差。

**架构：** 把人脸模型产生的 embedding/原始质量测量、质量规则和开放集判定规则拆成三层。Phase 3 只使用 Calibration 身份和可恢复实验缓存；Phase 4 的精确扫描器可以提前实现，但最终选择只允许在质量规则冻结后运行，并只在独立 Evaluation 上验证一次。

**技术栈：** Python 3.11+、InsightFace、ONNX Runtime CUDA/CPU、OpenCV、NumPy、SQLite、pytest。

**依据：** `README.md` 的 Phase 3～4 路线图、`docs/phase-2-design.md`、`docs/phase-2-results.md` 和 `docs/技术调研-开放集识别规则与质量评估.md`。

## 全局约束

- 不读取 Evaluation 来选择质量门、聚合方法或识别参数。
- LFW Calibration 中使用全部 116 个具有有效 Probe 的 Known 身份；Unknown 从 397 个有效来源身份中按种子 `2026` 固定选择 300 个。
- 质量实验只做单因素退化，不做全因素笛卡尔积。
- `quality_score` 先作为观测量，不参与人员聚合或样本加权。
- Phase 3 不用未冻结的匹配阈值报告最终 FPIR；只保存 Unknown 最高候选分数分布，FPIR 在 Phase 4 工作点上计算。
- 代码不兼容旧缓存、旧配置和旧接口；实验数据结构变化时重建 `data/experiments/phase3/`。
- 解释性注释和 docstring 使用中文。

---

### Task 1：拆分原始质量测量与质量规则

**Files:**
- Modify: `src/camera_face_comparison/image_input.py`
- Modify: `src/camera_face_comparison/evaluation_cache.py`
- Modify: `tests/test_quality.py`
- Modify: `tests/test_evaluation_cache.py`

**Interfaces:**
- Produces: `measure_quality(frame, observation) -> dict[str, float]`
- Produces: `apply_quality_policy(metrics, settings) -> QualityProfile`
- Keeps: `assess_quality(frame, observation, settings) -> QualityProfile` as the composition used by the application.
- Produces: `quality_policy_id(settings) -> str` independent of `embedding_extraction_id(settings)`.

- [x] 写失败测试，证明相同图片测量值不随质量阈值变化，并证明质量策略可以对同一测量值给出不同接收结果。
- [x] 运行 `pytest tests/test_quality.py tests/test_evaluation_cache.py -v`，确认因新接口不存在而失败。
- [x] 实现测量/策略拆分，使 `assess_quality()` 保持现有应用行为。
- [x] 从 `embedding_extraction_id` 删除质量阈值，新增独立 `quality_policy_id`。
- [x] 运行受影响测试并提交：`refactor: separate face quality measurement from policy`。

### Task 2：固定 Phase 3 实验协议

**Files:**
- Create: `src/camera_face_comparison/quality_experiment.py`
- Create: `scripts/prepare_quality_experiment.py`
- Create: `tests/test_quality_experiment.py`

**Interfaces:**
- Produces: `QualityExperimentProtocol`，保存 Known Probe、Unknown Probe、每个 Known 身份的参考图候选和随机种子。
- Produces: `build_quality_experiment_protocol(split_protocol, decision_scores_path, unknown_count=300, seed=2026)`。
- Produces: `write_quality_experiment_protocol()` / `read_quality_experiment_protocol()`。

- [ ] 写失败测试，使用小型协议和分数库验证只读取 Calibration、Known/Unknown 身份不重叠、固定种子可复现。
- [ ] 运行 `pytest tests/test_quality_experiment.py -v`，确认因模块不存在而失败。
- [ ] 实现协议构建，Known 每个身份固定选择一张已通过 Phase 2 的 Probe；Unknown 每个来源身份最多选择一张。
- [ ] CLI 生成 `data/experiments/phase3/protocol.json` 并记录源协议哈希。
- [ ] 运行针对性测试和真实协议生成命令，核对 Known=116、Unknown=300。
- [ ] 提交：`feat: add fixed quality experiment protocol`。

### Task 3：实现单因素退化和可恢复原始测量缓存

**Files:**
- Modify: `src/camera_face_comparison/quality_experiment.py`
- Create: `scripts/run_quality_experiment.py`
- Modify: `tests/test_quality_experiment.py`

**Interfaces:**
- Produces: `DegradationSpec(kind, level)`。
- Produces: `apply_degradation(frame, spec, baseline_bbox) -> np.ndarray`。
- Produces: `select_primary_face(faces) -> FaceObservation`，仅用于已知单主体数据集，按人脸面积、中心距离和检测分数确定主脸。
- Produces: `QualityExperimentStore`，以图片路径、文件 SHA-256、退化条件和 `embedding_extraction_id` 为键保存 embedding、原始指标、检测数量、耗时或 FTE 原因。

- [ ] 写失败测试，分别验证高斯模糊、亮度、对比度和固定画布人脸尺寸退化只改变指定因素。
- [ ] 写失败测试，验证缓存重开后不重复调用 fake engine，图片哈希或提取标识变化时失效。
- [ ] 实现退化条件：人脸目标尺寸 `160/112/96/80/64/48 px`，高斯模糊 `σ=0/1/2/3/4`，亮度倍率 `1.0/0.75/0.50/0.35/1.25/1.50`，对比度倍率 `1.0/0.75/0.50/0.25`；重复基线只运行一次。
- [ ] 实现按批提交、进度输出和中断恢复；运行时记录实际 CUDA/CPU provider。
- [ ] 运行针对性测试并提交：`feat: add recoverable face quality degradation experiment`。

### Task 4：生成质量证据和冻结建议

**Files:**
- Create: `src/camera_face_comparison/quality_analysis.py`
- Create: `scripts/analyze_quality_experiment.py`
- Create: `tests/test_quality_analysis.py`
- Create: `docs/phase-3-results.md`
- Modify: `README.md`
- Modify: `docs/known-issues.md`

**Interfaces:**
- Produces: `analyze_probe_quality()` 和 `analyze_gallery_quality()`。
- Produces: 每条件的 FTE、当前质量规则拒绝率、Known Rank-1、同人分数、Unknown top-score 分位点。
- Produces: 每个原始指标和 `quality_score` 的 Error-versus-Reject 表，以及清洁输入保留率为 90%/95%/99% 的候选门。

- [ ] 写失败测试，以手工分数验证 Rank-1、分位点和 Error-versus-Reject 计算。
- [ ] 实现分析函数和原子 JSON 报告写入。
- [ ] 在 CUDA 上运行真实实验；若中断，从 SQLite 缓存恢复。
- [ ] 生成 `data/experiments/phase3/report.json` 和 `docs/phase-3-results.md`，文档只写实际数字。
- [ ] 根据证据关闭或保留 `QUAL-001/QUAL-002`；不能稳定预测错误的数值指标不作为硬拒绝或加权依据。
- [ ] 运行针对性测试、Ruff、compileall 和 diff 检查后提交：`docs: record phase 3 quality evidence`。

### Task 5：实现 Phase 4 精确断点扫描器

**Files:**
- Replace: `src/camera_face_comparison/calibration.py`
- Replace: `scripts/calibrate_thresholds.py`
- Modify: `tests/test_calibration.py`
- Modify: `tests/test_calibration_script.py`

**Interfaces:**
- Produces: `calibrate_method(rows, target_fpir, use_score_gap) -> OperatingPoint`。
- Produces: `compare_methods(decision_scores_path, split='calibration') -> CalibrationReport`。
- Produces: `evaluate_operating_point(decision_scores_path, selected, split='evaluation') -> EvaluationReport`。

- [ ] 写失败测试，验证仅阈值扫描、阈值加候选分差扫描、无方案达到目标 FPIR 以及同效时优先简单规则。
- [ ] 使用实际 `top_score` 和 `score_gap` 离散断点；二维规则通过反向累计计数扫描，不构造“候选组合 × Probe”的三维数组。
- [ ] 对六种聚合分别报告 `FPIR≤1%`、`FPIR≤0.3%`、Calibration 观测 `FPIR=0%`，同时保留仅阈值和阈值+候选分差两组。
- [ ] 禁止脚本在选择阶段查询 `split='evaluation'`；只有显式评估命令可以读取 Evaluation。
- [ ] 运行针对性测试并提交：`feat: add exact open-set operating-point calibration`。

### Task 6：质量冻结后的联合标定与应用接入

**Files:**
- Modify: `src/camera_face_comparison/config.py`
- Modify: `src/camera_face_comparison/recognition.py`
- Modify: `src/camera_face_comparison/ui/main_window.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_recognition.py`
- Modify: `tests/test_ui.py`
- Create: `docs/phase-4-results.md`
- Modify: `README.md`
- Modify: `design.md`
- Modify: `docs/known-issues.md`

**Interfaces:**
- Consumes: Phase 3 冻结的 `quality_policy_id` 和重新导出的无阈值分数。
- Consumes: Phase 4 选出的 `aggregation_method/top_k/match_threshold/use_score_gap/min_score_gap`。
- Produces: 应用配置和运行识别服务使用同一套冻结规则。

- [ ] 若质量规则改变，重新生成 LFW embedding/无阈值分数；未改变则复用 Phase 2 分数库。
- [ ] 在 Calibration 选择主工作点，在 Evaluation 只执行一次最终统计。
- [ ] 先写失败测试，再把选定聚合与规则接入应用；删除未被选择的部署旧逻辑和质量分层阈值。
- [ ] 记录 Phase 4 实际 FPIR、FNIR、TPIR、Rank-1、FTE、FTA 和有效分母。
- [ ] 全量运行 pytest、Ruff、compileall、`git diff --check`，提交：`feat: apply calibrated open-set recognition policy`。

## 完成判定

- Phase 3 协议不含任何 Evaluation Probe。
- 真实实验报告记录实际 provider、样本数、退化条件、FTE、质量拒绝、Rank-1 和连续分数分布。
- Phase 4 选择过程只读取 Calibration，最终报告才读取 Evaluation。
- 应用最终配置来自实验结果，不保留无依据的初始质量分层或软质量加权。
- 每个功能测试都有过失败阶段；真实数据实验不塞进 pytest。
