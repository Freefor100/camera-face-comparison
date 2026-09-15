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

自动建库脚本最初建立了 6 个身份、18 张样本：

| 身份 | 样本数 |
| --- | ---: |
| Angela Merkel | 1 |
| Barack Obama | 4 |
| Donald Trump | 1 |
| 景甜 | 4 |
| 孙宇晨 | 5 |
| 张继科 | 3 |

此后操作者通过图形界面新增了 `liu` 的 1 张样本；截至阶段 5D 文档同步时，本地数据库实际为
7 个身份、19 张样本。该手工新增身份证明动态扩容入口已被继续使用，但不改写下方最初自动验收
报告中的 6 个身份、18 张样本分母。

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

## 3. 真实摄像头验收

操作者已经在本机真实摄像头下完成人员录入、本人已登记识别和其他人员未知拒识，确认多帧采集、
标准库检索、最高分阈值判定和界面反馈可以完成。操作者同时确认当前模型后端和摄像头设备工作
正常。界面显示的“识别用时”用于现场观察即可；界面无法准确测量自身最终显示到屏幕上的耗时，
因此不把像素渲染时间作为单独验收项。

已经完成：

1. 使用真实摄像头完成人员录入；
2. 本人站在摄像头前完成已登记人员识别；
3. 其他人员站在摄像头前完成未知人员拒识；
4. 确认模型后端和摄像头设备正常。

阶段 6 的技术验收至此完成。上课前按展示顺序再运行一次属于普通演示准备，不是未完成的开发项，
也不再用于修改阈值或聚合方法。
