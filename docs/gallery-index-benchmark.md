# Gallery 内存索引 CPU/CUDA 基准实验

## 1. 实验问题

当前桌面识别服务每次查询都会从 SQLite 读取样本、按人员重新分组、重新计算 Mean
Prototype，并在 Python 中逐人计算分数和完整排序。本实验只回答三个问题：

1. 把人员原型预先构造成连续 `P×512` 矩阵能否加速候选检索；
2. 单 Query 场景是否值得把检索放到 CUDA；
3. 是否值得在 SQLite 中额外持久化 Mean Prototype。

实验不修改人脸模型、聚合公式、阈值或最终 Evaluation，也不把图片解码、人脸推理、
完整性扫描和日志写入时间混入检索数字。

## 2. 数据与环境

- 数据：Phase 4 固定的自然 LFW 协议；协议 Gallery 为 4,599 个身份、7,490 张图片；
- 有效数据：4,588 个身份、7,465 个 Gallery embedding、5,720 个有效 Probe embedding；
- Query：固定种子 `2026` 选择 200 个不同的真实 LFW Probe；
- 当前实现因重建开销较大，每个规模抽取其中 30 个 Query 计时；
- 向量：InsightFace `buffalo_l` 的 512 维 `float32` L2 单位向量；
- CPU：Intel Core i5-12500H；NumPy 2.5.2；OpenBLAS 0.3.34，默认 16 线程；
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4,096 MiB；驱动 610.57.04；
- CUDA 路径：ONNX Runtime 1.29 的 `MatMul + TopK`，原型矩阵作为 initializer 常驻
  GPU，每个 Query 由当前 NumPy 接口从 CPU 上传，只把前两名分数和索引传回 CPU。

每个实现先预热 10 次，再逐 Query 同步计时。CPU 矩阵另外测试单 BLAS 线程；表中采用
实际更快的默认线程结果。CUDA 使用 ONNX Runtime profile 复核，`MatMul` 和 `TopK`
两个节点均实际分配给 `CUDAExecutionProvider`，不是 provider 列表存在但节点回退 CPU。

## 3. 被比较的实现

| 实现 | 每次 Query 执行的工作 |
| --- | --- |
| 当前生产路径 | 重新归一化全部样本、按身份求 Mean Prototype、逐人点积、完整排序和阈值判定 |
| 缓存原型 + Python 循环 | 原型只构建一次，但仍逐行调用 Python 点积并取前两名 |
| CPU 连续矩阵 | `prototype_matrix @ query`，再用部分排序取前两名 |
| CUDA ONNX | GPU `MatMul + TopK`；包含单 Query 输入上传和 top-2 输出下载 |

CPU 矩阵与当前生产路径在每个规模抽查 10 个 Query，共 50 次；top-1 全部一致，top-2
分数最大绝对误差为 `1.1921e-7`。CUDA 与 CPU 的首个 Query top-1 全部一致，最大分数
误差不超过 `5.9605e-8`。这些误差来自 `float32` 运算顺序，不影响当前阈值判定。

## 4. 单 Query 延迟结果

单位均为毫秒。加速比使用当前生产路径 p50 除以 CPU 矩阵 p50。

| 身份数 | 样本数 | 当前 p50 / p95 | 缓存原型 Python p50 | CPU 矩阵 p50 / p95 | CUDA p50 / p95 | CPU 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6 | 0.0425 / 0.0579 | 0.0084 | 0.0061 / 0.0066 | 0.0581 / 0.0633 | 6.94× |
| 10 | 20 | 0.1203 / 0.1452 | 0.0128 | 0.0063 / 0.0067 | 0.0453 / 0.0902 | 19.03× |
| 100 | 165 | 1.1215 / 1.1630 | 0.0663 | 0.0087 / 0.0092 | 0.0469 / 0.0911 | 128.81× |
| 1,000 | 1,642 | 11.4751 / 11.6804 | 0.6398 | 0.0206 / 0.0341 | 0.0884 / 0.1436 | 558.37× |
| 4,588 | 7,465 | 53.7852 / 55.7222 | 3.2198 | 0.0528 / 0.1165 | 0.1321 / 0.2871 | 1,018.93× |

完整 LFW 规模的人员原型矩阵占 `9,396,224` 字节，约 8.96 MiB，只占 4 GiB 显存的
约 0.22%，因此显存容量不是问题。CUDA 较慢的原因是单 Query 工作量较小，同时必须
承担输入上传、输出下载和 kernel 调度；GPU 的并行计算没有抵消这些固定成本。完整规模
CUDA p50 是最快 CPU 的 2.50 倍，即延迟更高而不是获得加速。

CUDA 首个 session 初始化为 306.32 ms，运行时已经初始化后，其余不同规模 session
构建为 2.25～8.05 ms。ONNX Runtime 还提示环境未安装 `libcufft.so.12`；本实验算子不
依赖 FFT，profile 已确认 `MatMul/TopK` 在 CUDA 执行，但不应把这解释为完整 CUDA
运行库已经无缺项。

## 5. 启动构建与持久化结果

完整有效 Gallery 的实测启动工作为：

| 操作 | 耗时 |
| --- | ---: |
| 从 SQLite 缓存读取 7,465 个 Gallery embedding | 29.84 ms |
| 计算 4,588 个 Mean Prototype 并构建连续矩阵 | 61.58 ms |
| 保存预计算 `.npy` 矩阵 | 2.20 ms |
| 同一进程、受操作系统页缓存影响的 `.npy` 加载 | 1.89 ms |

从样本恢复完整矩阵合计约 91.42 ms。预计算矩阵确实能缩短这一次启动工作，但 1.89 ms
是同一进程页缓存结果，不是严格冷启动数据。即使按该数字计算，节省的也只是启动时约
90 ms，却要引入样本—原型一致性、聚合版本、模型版本和事务更新问题。因此当前规模
不把 Mean Prototype 作为第二份事实写入 SQLite；启动时从原始 embedding 重建更简单、
可靠。

## 6. 结论与下一步边界

1. **CPU 连续矩阵方案成立。**它保持精确分数和候选不变，并在大 Gallery 中消除重复
   原型计算、Python 循环和完整排序；不需要 Faiss、红黑树或近似索引。
2. **当前单 Query 不采用 CUDA 检索。**本机所有测试规模下默认 OpenBLAS CPU 均更快；
   CUDA 继续用于耗时远高于检索的人脸检测和 embedding 推理。
3. **当前不持久化人员原型。**完整 LFW 规模启动重建不足 0.1 秒，持久化收益不足以抵消
   派生数据一致性复杂度。
4. **最终少量身份 Demo 的绝对收益很小。**3～10 人只节省约 0.04～0.11 ms，不能作为
   端到端时间优化成果；完整 LFW 规模则能节省约 53.73 ms，说明它是明确的可扩展性优化。
5. 若进入生产实现，应新增内存 `GalleryIndex`：启动时完成一次完整性检查并构建矩阵，
   识别时只执行精确矩阵检索；人员创建或追加样本成功后，仅重建受影响人员的原型。该
   改动必须再做包含人脸推理、完整性策略和日志的 E2E A/B，不能直接用本实验的检索微
   基准替代系统总耗时结论。

## 7. 复现与产物

```bash
.venv/bin/python scripts/benchmark_gallery_index.py
```

本地原始报告和预计算矩阵位于被 Git 忽略的目录：

- `data/experiments/gallery-index/benchmark_report.json`
- `data/experiments/gallery-index/lfw_mean_prototypes.npy`

脚本和报告均固定协议 SHA-256、原始 embedding 缓存 SHA-256、数据范围、硬件、BLAS
线程数、CUDA 节点 provider、延迟分布、矩阵内存和数值一致性证据。
