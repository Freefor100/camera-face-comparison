# 领域词汇

## Identity（身份）

标准库中一个可被识别的人。一个身份可以关联多张参考样本。

## Gallery（标准库）

参与一次 1:N 搜索的身份及其参考数据集合。Gallery 中的身份数量是 N。

## Reference Sample（参考样本）

已明确归属于某个身份、用于构造或计算该身份匹配分数的人脸样本。

## Probe（探针）

本次待识别的人脸输入。Probe 可以来自摄像头、本地图片或评测数据集，不会因为被识别而自动成为参考样本。

## Known Probe（已知探针）

真实身份存在于当前 Gallery 的 Probe。

## Unknown Probe（未知探针）

真实身份不存在于当前 Gallery 的 Probe。

## Open-set Identification（开放集识别）

在 Gallery 中搜索最可能身份，同时允许把证据不足的 Probe 判定为 Unknown 的 1:N 识别任务。

## Embedding（特征向量）

预训练人脸编码器从一张对齐人脸中提取的数值表示。向量相似度用于计算人脸匹配证据。

## Person Score（人员分数）

一个 Probe 与一个 Gallery 身份之间的最终匹配分数。它不是某一张图片的原始像素差异。

## Matched（已匹配）

Probe 成功通过开放集接受规则，并被分配给 Gallery 中某个身份。

## Unknown（未知人员）

Probe 已完成有效特征提取和 Gallery 搜索，但没有足够证据分配给任何已知身份。

## Invalid / Low Quality（输入无效或质量不足）

Probe 未进入有效身份判断，例如无人脸、多人脸、图像质量不足或标准库不可用。它与 Unknown 不是同一种结果。
