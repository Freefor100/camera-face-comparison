# Phase 5 小 Gallery 规模实验结果

## 结论

本实验使用 Phase 4 已完成的自然 LFW 原始 embedding，独立考察 Gallery 身份数量从
4,599 缩小到 3、5、10、25、50、100 时，候选分差和开放集工作点是否失效。每个规模
使用 10 个固定种子重复；Calibration 与 Evaluation 的 Known/Unknown 来源身份互斥，
最终演示照片和摄像头图片均未参与。

结果把原 `ALG-003` 风险拆成了两部分：

1. **LFW 域内的 Gallery 缩小没有造成候选分差失控。**Phase 4 完整 Gallery 工作点
   `match_threshold=-1.0`、`min_score_gap=0.5078` 迁移到 60 个小 Gallery 运行后，
   Evaluation Unknown FPIR 全部为 0；平均 TPIR 在 3 人时为 98.63%，100 人时为
   83.06%。Gallery 越小，第二候选通常越弱，分差规则反而更容易接受正确 Known。
2. **直接用极少 Known 身份重新标定不稳定。**3 人 Gallery 的 10 次 Calibration
   阈值范围为 `0.197～0.768`，Evaluation TPIR 均值 86.88%、标准差 15.21%。因此
   最终 3 人或少量身份演示库不能既作为调参数据又作为验收数据。

这不能关闭摄像头域问题。LFW 内部迁移成功只说明 Gallery 数量本身不是当前主要风险；
真实相机、距离、曝光和现场画质造成的域偏移仍必须使用非最终演示人员的开发数据复核。

## 实验协议

- 输入缓存：Phase 4 `lfw-natural-v1` 原始缓存；
- 有效 Gallery：7,465/7,490 张，形成 4,588 个有效身份原型；
- 有效 Probe：5,720/5,743 张；
- 可抽样 Known 身份：Calibration 127 个、Evaluation 127 个，彼此不重叠；
- 原始 Unknown：Calibration 1,228 张、Evaluation 1,189 张；
- 聚合：Mean Prototype，与 Phase 4 最终候选一致；
- 规模：3、5、10、25、50、100 个身份；
- 重复：每档 10 次，固定主种子 2026；
- 目标工作点：Calibration FPIR 不超过 0.3%；
- 每次独立标定同时比较“仅匹配阈值”和“匹配阈值 + 候选分差”；
- 完整 Gallery 工作点只做迁移对照，不参与小 Gallery 参数搜索。

每次重复分别打乱 Calibration 和 Evaluation 的 Known 身份池，选择相同数量但身份互斥
的 Gallery。参数只在 Calibration 实际分数断点上精确扫描，再一次性应用到该重复的
Evaluation。Unknown 始终来自原协议对应分区，不从成功样本中挑选。

## 每个规模的结果

| Gallery 身份 | 分差规则被选次数 | 独立标定阈值均值 | Evaluation Rank-1 均值 | 独立标定 TPIR 均值 | 独立标定 FPIR 均值 | 完整 Gallery 工作点迁移 TPIR | 迁移 FPIR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0/10 | 0.636 | 99.81% | 86.88% | 0.0168% | 98.63% | 0% |
| 5 | 1/10 | 0.565 | 98.86% | 96.31% | 0.0421% | 97.61% | 0% |
| 10 | 0/10 | 0.492 | 98.24% | 97.74% | 0% | 96.92% | 0% |
| 25 | 0/10 | 0.432 | 98.46% | 98.31% | 0% | 96.09% | 0% |
| 50 | 2/10 | 0.338 | 98.53% | 98.42% | 0.0421% | 92.01% | 0% |
| 100 | 1/10 | 0.370 | 98.71% | 98.61% | 0% | 83.06% | 0% |

60 次独立标定中，只有 4 次选择候选分差；56 次在目标 FPIR 下由单一匹配阈值获得
更高或相同 TPIR。这与 Phase 4 的 4,599 身份结果不矛盾：候选数量增大后，最佳候选
与第二候选更接近，分差才更可能提供额外区分。

个别独立 Evaluation 会因身份更换超过 Calibration 目标：5 人和 50 人档的最高
Evaluation FPIR 分别为 0.4205% 和 0.1682%。报告保留每个重复的具体 Gallery 身份、
Calibration 参数和 Evaluation 计数，不只保存均值。

## 对部署设计的影响

- 保留 Mean Prototype 作为当前应用接入候选；
- 不因为最终 Demo Gallery 只有少量身份，就使用最终演示身份重新标定；
- LFW 完整 Gallery 工作点在小 Gallery 上是保守且可用的候选，不立即写入应用配置；
- 下一步只需重点检查摄像头域是否显著改变同人分数和候选分差；
- 若摄像头域出现偏移，使用独立的临时开发身份调整部署参数，最终 Demo Gallery 仍只验收。

## 本地产物与复现

完整 60 次逐运行结果位于 Git 忽略的：

- `data/experiments/phase5/gallery_scale_report.json`。

```bash
.venv/bin/python scripts/evaluate_gallery_scale.py \
  --data-dir ./data \
  --protocol ./data/experiments/phase4/protocol.json \
  --cache-path ./data/logs/cache/lfw_raw.sqlite \
  --policy ./data/experiments/phase4/final_evaluation.json \
  --gallery-sizes 3 5 10 25 50 100 \
  --repeats 10 \
  --seed 2026 \
  --target-fpir 0.003
```

该脚本只读取缓存，不初始化 InsightFace，也不产生新的 embedding。
