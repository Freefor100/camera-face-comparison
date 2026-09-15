# 阶段 6：最终演示标准库与本地验收

## 1. 建库结果

参数、人员平均特征向量、最高分阈值、质量规则和界面已经冻结后，使用
`data/demo-candidates/` 建立最终演示标准库。建库命令为：

```bash
.venv/bin/python scripts/prepare_demo_gallery.py \
  --data-dir ./data \
  --candidates-dir ./data/demo-candidates \
  --manifest ./data/experiments/phase6/demo-gallery-manifest.json
```

实际建立了 6 个身份、18 张样本：

| 身份 | 样本数 |
| --- | ---: |
| Angela Merkel | 1 |
| Barack Obama | 4 |
| Donald Trump | 1 |
| 景甜 | 4 |
| 孙宇晨 | 5 |
| 张继科 | 3 |

每张图片都先通过真实 `FaceEngine` 的唯一人脸和特征提取预检，再交给正式
`EnrollmentService` 写入。Angela Merkel 的另一张候选图片检测到多人脸，没有纳入标准库；
其余 18 张均产生有效特征。图片中的“人脸偏小”或“清晰度较低”只作为提示，不阻断有效单脸。

图片复制到 `data/faces/<person_id>/`，SQLite 只保存人员、样本路径、特征向量和哈希；最终演示
数据库和图片均在 `data/`，不提交 Git。该标准库建立在参数冻结之后，不参与之前的数据集调参、
聚合方法选择或阈值标定。

## 2. 自动验收结果

验收命令：

```bash
.venv/bin/python scripts/validate_demo_gallery.py \
  --data-dir ./data \
  --known ./data/demo-candidates/barack_obama/barack_obama_2012_original.jpg \
  --unknown ./data/phase1-inputs/joe_biden_official.jpg \
  --expected-known-name "Barack Obama" \
  --output ./data/experiments/phase6/final-acceptance-report.json
```

结果：

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 初始标准库完整性 | 通过 | SQLite、图片哈希和特征向量哈希检查正常 |
| Known 本地图片 | 通过 | Obama → `Barack Obama`，最高分 `0.952704` |
| Unknown 本地图片 | 通过 | Joe Biden → `unknown`，最高分 `0.112275`，原因 `score_below_threshold` |
| 重启后恢复 | 通过 | 仍为 6 个身份、18 张样本，完整性正常 |

本次自动验收使用 CPU 回退，模型初始化时 CUDA provider 无法连接当前执行环境的驱动；因此
报告中的耗时不是 CUDA 结论。详细分数、分差和耗时见
`data/experiments/phase6/final-acceptance-report.json`，候选预检见
`data/experiments/phase6/demo-gallery-manifest.json`。

## 3. 尚未完成的现场验收

自动验收不能替代真实桌面环境中的以下操作：

1. 用用户实际 Python 环境确认模型后端为 CUDA，或记录 CPU 回退原因；
2. 刷新并选择 `/dev/video0` 或 `/dev/video1` 对应的 OpenCV 摄像头索引；
3. 连续预览、停止后画面清除、再次启动预览；
4. 本人站在摄像头前完成五帧 Known 识别；
5. 其他同学站到摄像头前验证 Unknown；
6. 追加一张样本、新增一个身份并确认立即参与识别；
7. 离线重启应用后再次完成摄像头识别。

这些项目属于真实硬件和现场状态验收，不再用于修改阈值或聚合方法。
