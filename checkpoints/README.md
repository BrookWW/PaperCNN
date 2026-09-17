# 已训练模型 / Trained checkpoint

`papercnn.pt` 来自真实训练记录 [`20260917_220224_060004`](../experiments/20260917_220224_060004/summary.json) 的第 6 轮，包含 5,955 个模型参数、预处理尺寸、类别顺序及固定训练/验证清单。导出时仅把原机器的数据目录改为 `data`；模型权重逐项检查保持一致。

`papercnn.pt` comes from epoch 6 of the recorded run [`20260917_220224_060004`](../experiments/20260917_220224_060004/summary.json). It contains 5,955 model parameters, preprocessing dimensions, label order, and the fixed training/validation split. Export only replaces the original machine's data directory with `data`; every model tensor was checked for exact equality.

| 指标 / Metric | 值 / Value |
|---|---:|
| 验证样本 / Validation samples | 600 |
| 验证准确率 / Validation accuracy | 76.50% |
| 验证 macro-F1 / Validation macro-F1 | 0.7632 |
| 独立测试评估 / Held-out test evaluation | 未进行 / Not performed |

从仓库根目录预测自己的首页图片，无需训练数据：

Predict your own first-page image from the repository root, without the training dataset:

```bash
python predict.py --checkpoint checkpoints/papercnn.pt --image /path/to/page.png
```

在原始数据集上重新验证时，用 `--data-root /path/to/arxiv` 指定数据位置。输出的 softmax 分数未经概率校准，不能当作可靠的置信度保证。

To re-evaluate on the original validation dataset, pass `--data-root /path/to/arxiv --split val`. The softmax scores are uncalibrated and do not guarantee reliable confidence estimates.
