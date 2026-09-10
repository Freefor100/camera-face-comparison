# 人脸比对系统当前实现说明

本文只描述当前代码已经实现的行为。后续工作见 `README.md`，实验结论见
`docs/phase-5b-cross-quality-results.md`。

## 1. 系统定位

程序是离线桌面端开放集 1:N 人脸识别系统：输入一张摄像头帧或本地图片，先在
标准库的所有身份中找第一候选，再用冻结的开放集规则决定输出姓名还是“未知人员”。
它不是原图 1:1 像素比对，也不会因新增人员而重新训练 InsightFace。

运行时由以下模块组成：

- PySide6：界面、状态反馈和工作线程；
- OpenCV：跨平台摄像头、图片解码和图片保存；
- InsightFace `buffalo_l`：人脸检测、五点对齐和 512 维 embedding；
- ONNX Runtime：优先 CUDA，实际 CUDA session 不可用时回退 CPU；
- SQLite：人员、样本 embedding、原始质量指标和识别日志。

## 2. 摄像头与图片输入

摄像头由 OpenCV 按整数索引打开。Linux 优先 V4L2，Windows 优先 DirectShow，
macOS 优先 AVFoundation，平台后端失败时再使用 OpenCV 通用后端。Linux 索引 0
通常对应 `/dev/video0`，但程序不直接读取设备节点，也不写死某个路径。

预览由 `CameraWorker` 持续读取。每帧复制后交给 UI，抓拍识别再复制一次，避免后台
推理读取不断变化的缓冲区。预览运行时禁止刷新设备；停止后释放设备并清除当前帧、
检测框和最后画面。本地图片通过 `ImageInput.from_file()` 解码，和摄像头帧进入同一个
识别服务。

## 3. 人脸检测与特征提取

```text
BGR 图片
  → InsightFace 人脸检测
  → 恰好一张人脸？
       否：invalid（无人脸或多人脸）
       是：InsightFace 五点对齐与识别模型
  → embedding 有效且非零？
       否：invalid
       是：L2 归一化为单位向量
```

`FaceAnalysis` 只加载 `detection` 和 `recognition`，不运行应用没有使用的性别年龄与
额外关键点模型。该裁剪在 XQLFW 7,263 张图片的独立空缓存 A/B 中使有效 embedding
平均推理耗时从 32.293 ms 降到 23.896 ms，缓存状态、指标和 embedding 无差异。

当前不按检测分、脸部尺寸、清晰度、亮度或对比度拒绝一张已经产生有效单脸
embedding 的图片。原因是 Phase 3 的 10,108 次单因素实验表明，旧启发式质量门会
大量拒绝仍能正确识别的图片。程序仍记录五项原始指标，并把异常值转换为不阻断流程的
拍摄建议。

## 4. 标准库录入

创建人员支持一张或多张摄像头帧/本地图片，不要求固定姿态、动作或数量。每张输入必须
产生唯一且有效的人脸 embedding；任一输入失败时，本批录入整体失败。

```text
输入图片
  → 唯一人脸与 embedding 校验
  → 记录原始质量指标
  → 图片写入 data/faces/.staging/
  → 全部图片成功后移动到 data/faces/<person_id>/
  → 一个 SQLite 事务写入人员和全部样本
```

第一张有效样本写入后，人员立即参与识别。已有人员可继续原子追加多张样本。代码没有
空人员、人员生命周期、五姿态会话或旧数据库迁移逻辑。

## 5. 人员级表示与余弦相似度

设人员 (i) 有单位样本向量 (e_{i1},\ldots,e_{im})。当前固定使用 Mean Prototype：

\[
p_i=\operatorname{norm}\left(\frac{1}{m}\sum_{j=1}^{m}e_{ij}\right)
\]

输入单位向量 (q) 与该人员的分数为：

\[
S_i=q^Tp_i
\]

所以同一人员无论有一张还是多张样本，都只形成一个归一化身份原型；识别时 Query 与
每个身份原型各做一次点积。原始图片、质量指标和样本数量不直接进入分数加权。

## 6. 开放集接收规则

所有人员分数按降序排列。当前唯一部署规则为：

```text
第一候选分数 >= 0.5557855367660522
    → matched / 输出第一候选姓名
否则
    → unknown / score_below_threshold
```

配置名称为 `minimum_score`。第一候选、第二候选及二者的 `score_gap` 仍写入日志并显示，
但候选分差不参与当前接收判定。库为空时返回 `unknown / empty_face_library`。

该方案来自 LFW/XQLFW 四个主要跨质量场景的 Calibration 联合标定，不是主观默认值。
实验比较了 Single、Max、Mean Prototype、Top-K Mean K=2/3/5，以及最高分阈值、
仅候选分差、双条件和 NAC。主目标 `FPIR_valid ≤ 1%` 下，Mean Prototype + 最高分
阈值获得最高的最差场景端到端 TPIR；NAC-32 只增加 0.39 个百分点，未达到预先规定的
1 个百分点复杂规则收益门槛。

## 7. 识别服务顺序

`RecognitionService.compare_input()` 当前按以下顺序执行：

1. 检查 SQLite 完整性、外键、样本图片哈希和 embedding 哈希；
2. 提取唯一人脸并归一化 embedding；
3. 测量检测分、脸部尺寸、清晰度、亮度和对比度，生成非阻断提示；
4. 读取全部标准库 embedding，按身份建立 Mean Prototype；
5. 计算全部身份分数并应用 `minimum_score`；
6. 写入一次识别日志；
7. 把同一个结果对象交给 UI。

`invalid` 表示图片解码、人脸数量、embedding 或标准库完整性失败；`unknown` 表示已经
完成有效 1:N 检索，但最高身份分数未达到接收阈值。两者不能混为一类。

## 8. SQLite 与简单完整性检查

当前数据库表：

- `persons(id, display_name, created_at)`；
- `face_samples(..., quality_metrics_json, source_type, image_sha256, embedding_sha256)`；
- `recognition_logs(..., top_score, second_score, score_gap, acceptance_score,
  acceptance_rule, latency_ms, reason)`。

SQLite 开启外键、WAL、5 秒 busy timeout 和 `BEGIN IMMEDIATE` 短写事务。SQLite 自身
负责并发写锁；应用不另外实现文件锁。入库时记录图片文件和 embedding 字节的 SHA-256，
识别前检查文件缺失、误替换和 BLOB 损坏。该机制不是密码学防篡改：能同时改写数据和
哈希的攻击者不在本课设威胁模型内。

## 9. 实验基础设施

当前保留三类可复用实验组件：

- `RawEmbeddingCache`：按数据集、模型提取版本、相对路径和文件 SHA-256 保存原始
  embedding、指标、FTE 与耗时；
- 无阈值分数导出：固定 Gallery/Probe 后保存多个聚合方法的候选身份与连续分数；
- 联合标定与评估：只用 Calibration 的实际分数断点选参，再把冻结参数应用于一次
  Evaluation；支持按身份 bootstrap 95% 置信区间和小 Gallery 重放。

XQLFW 官方 Pair 入口仍用于 1:1 跨质量验证；LFW/XQLFW 同路径变体用于开放集 1:N
联合标定。QMUL-SurvFace 的原始提取入口保留用于监控小脸域覆盖统计，但监控域旧判定器
已移除，避免把桌面阈值机械迁移成可部署结论。

## 10. 当前明确边界

- 不含活体检测、照片/屏幕攻击防护或支付级可信链；
- 不解决极暗、严重模糊、强遮挡、极端侧脸和远距离监控小脸；
- 不重新训练 InsightFace，不实现 LDA、EVM、学习式模板聚合或专用 FIQA；
- 质量提示阈值只是操作建议，不是识别准确率保证；
- 多帧质量择优、完整 E2E 分阶段耗时优化、UI 收尾和最终 Demo Gallery 尚未实现。
