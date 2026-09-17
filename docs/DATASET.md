# 数据格式 / Dataset guide

[返回 README / Back to README](../README.md) · [中文](#中文) · [English](#english)

## 中文

本仓库不提供论文图片、原始 CSV 或下载脚本。训练和评估前，需要自行提供已经整理好的论文首页 PNG 图片与清单。以下格式对应已记录的实验；仓库中 `experiments/*/split.json` 记录了当时使用的训练/验证相对路径，但它不能替代原始数据。

### 目录与清单

设置 `ARXIV_DATA_ROOT` 为以下目录，或传入 `--data-root`：

```text
/path/to/arxiv/
├── train.csv
├── test.csv
├── manifest.csv
└── images/
    ├── math/
    │   └── <arxiv_id>v1.png
    ├── physics/
    │   └── <arxiv_id>v1.png
    └── cs/
        └── <arxiv_id>v1.png
```

`train.csv` 和 `test.csv` 至少包含以下两列。示例行仅说明格式，不是随仓库附带的图片：

```csv
image_path,label
images/math/2609.09152v2.png,math
images/physics/2609.00002v1.png,physics
images/cs/2609.00003v1.png,cs
```

| 字段 | 要求 |
|---|---|
| `image_path` | 相对数据根目录的 PNG 路径，例如 `images/math/2609.09152v2.png`；文件必须存在 |
| `label` | 严格使用 `math`、`physics` 或 `cs` |

图片路径必须位于 `images/` 下，并至少包含子目录和文件名；不接受绝对路径、`..`、指向目录之外的符号链接或以 `._` 开头的 macOS 附属文件。文件名用于识别论文：程序去掉扩展名及末尾版本号 `vN`，检查单个清单内是否重复引用同一论文。请使用带版本的 arXiv ID 命名图片，旧式 ID 如含 `/`，需先转换为唯一且一致的安全文件名。

完整的 `check.py --part data` 还需要 `manifest.csv`，至少包含：

```csv
image_path,label,split,arxiv_id
images/math/2609.09152v2.png,math,train,2609.09152
images/physics/2609.00002v1.png,physics,train,2609.00002
images/cs/2609.00003v1.png,cs,test,2609.00003
```

| 字段 | 要求 |
|---|---|
| `image_path` | 与 `train.csv` 和 `test.csv` 中的路径完全一致，无重复 |
| `label` | 与对应清单中的标签一致 |
| `split` | `train` 或 `test`；验证集属于原始 `train` 池，不写 `val` |
| `arxiv_id` | 同一论文的稳定标识，推荐不含版本号；不同版本必须共用同一标识 |

`manifest.csv` 必须精确覆盖两个清单的并集。完整检查会确认训练池与测试集没有相同的 `arxiv_id`。训练本身只读取 `train.csv`；`test.csv` 和 `manifest.csv` 是完整数据检查所需，测试评估还会读取 `test.csv`。

### 样本量与固定划分

| 清单或划分 | 每类数量 | 总数 |
|---|---:|---:|
| 原始 `train.csv` | 1,000 | 3,000 |
| 其中用于训练 | 800 | 2,400 |
| 其中用于验证 | 200 | 600 |
| `test.csv` | 300 | 900 |

划分按 `math → physics → cs` 顺序处理类别，各类别先按图片路径排序，再使用 NumPy 的种子 `231` 生成随机排列。每类前 200 个样本作为验证集，其余用于训练。DataLoader 使用独立的 PyTorch 随机生成器控制训练批次顺序。

完整检查针对上述数量设计；自定义更小或不平衡的数据集不会通过这一检查。训练加载器允许每类多于 `VAL_PER_CLASS` 个样本；如果有意改变数据规模，需要相应修改 `config.py` 和数据检查预期，并重新报告划分与结果。

### 预处理和检查边界

1. 转为灰度图。
2. 使用 LANCZOS 等比缩放并白色补边到宽 96、高 128。
3. 转为 `float32`，计算 `1 - pixel / 255`。
4. 返回形状为 `(1, 128, 96)`、范围为 `[0, 1]` 的张量。

原始图片不会被修改。没有随机图像增强，也没有从全体数据估计归一化统计量。

```bash
export ARXIV_DATA_ROOT="/path/to/arxiv"
python check.py --part data --data-root "$ARXIV_DATA_ROOT"
```

检查确认数量、标签、路径、文件存在性、清单一致性、训练/验证互斥，以及训练池/测试集的论文标识互斥。图片内容只抽查每类一张训练图；通过检查不表示所有文件均可解码，也不表示跨来源或近重复页面已被全面排除。`split.json` 保存确切的训练/验证清单、划分参数及 `train.csv` 的 SHA-256，便于追踪所使用的数据版本。

请在分发图片前确认原论文及数据来源的授权条件。当前仓库没有提供关于外部图片分发权利的证明或授权。

---

## English

This repository does not distribute paper images, original CSVs, or a download script. Supply first-page PNGs and manifests before training or evaluation. The schema below matches the recorded experiments. Files under `experiments/*/split.json` preserve historical train/validation paths but do not replace the original dataset.

### Directory layout and manifests

Set `ARXIV_DATA_ROOT` to the directory below, or pass it with `--data-root`:

```text
/path/to/arxiv/
├── train.csv
├── test.csv
├── manifest.csv
└── images/
    ├── math/<arxiv_id>v1.png
    ├── physics/<arxiv_id>v1.png
    └── cs/<arxiv_id>v1.png
```

Both `train.csv` and `test.csv` require at least these columns. The following rows illustrate the format; their images are not bundled:

```csv
image_path,label
images/math/2609.09152v2.png,math
images/physics/2609.00002v1.png,physics
images/cs/2609.00003v1.png,cs
```

| Field | Requirement |
|---|---|
| `image_path` | An existing PNG path relative to the dataset root, such as `images/math/2609.09152v2.png` |
| `label` | Exactly one of `math`, `physics`, or `cs` |

Paths must stay inside `images/` and include a subdirectory and filename. Absolute paths, `..`, links escaping the image directory, and macOS `._` sidecar entries are rejected. Filenames identify papers: removing the extension and a trailing version suffix `vN` must yield a unique paper identifier within each CSV. Use versioned arXiv IDs as filenames. If an older identifier contains `/`, first convert it to a consistent, unique, filesystem-safe name.

The full `check.py --part data` check also requires `manifest.csv` with at least:

```csv
image_path,label,split,arxiv_id
images/math/2609.09152v2.png,math,train,2609.09152
images/physics/2609.00002v1.png,physics,train,2609.00002
images/cs/2609.00003v1.png,cs,test,2609.00003
```

| Field | Requirement |
|---|---|
| `image_path` | Exactly matches an entry in `train.csv` or `test.csv`, with no duplicates |
| `label` | Matches the label in its source CSV |
| `split` | `train` or `test`; validation images belong to the original `train` pool, so do not use `val` |
| `arxiv_id` | A stable paper identifier, preferably without the version suffix; all versions of a paper must share it |

`manifest.csv` must cover the exact union of both CSVs. The full check verifies that the training pool and test set have no overlapping `arxiv_id` values. Training itself reads only `train.csv`; the full data check needs `test.csv` and `manifest.csv`, while test evaluation also reads `test.csv`.

### Counts and deterministic splitting

| Manifest or split | Per class | Total |
|---|---:|---:|
| Original `train.csv` | 1,000 | 3,000 |
| Used for training | 800 | 2,400 |
| Used for validation | 200 | 600 |
| `test.csv` | 300 | 900 |

The splitter processes labels in the order `math → physics → cs`. Within each label, it sorts image paths and applies NumPy permutations using seed `231`. The first 200 records become validation data and the remainder become training data. A separate PyTorch generator controls shuffled training batches.

The full checker enforces the counts above. Smaller or unbalanced custom datasets will fail that check. The training loader accepts more than `VAL_PER_CLASS` samples per class; intentional changes to dataset size require corresponding configuration and check expectations, and should be accompanied by a fresh report of splits and results.

### Preprocessing and scope of checks

1. Convert to grayscale.
2. Resize with LANCZOS, preserving aspect ratio, and pad with white to width 96 and height 128.
3. Convert to `float32` and compute `1 - pixel / 255`.
4. Return a tensor of shape `(1, 128, 96)` with values in `[0, 1]`.

Source images are never modified. There is no random image augmentation or dataset-wide learned normalization.

```bash
export ARXIV_DATA_ROOT="/path/to/arxiv"
python check.py --part data --data-root "$ARXIV_DATA_ROOT"
```

The check verifies counts, labels, paths, file existence, manifest consistency, disjoint train/validation splits, and disjoint paper identifiers between the training pool and test set. It decodes only one training image per class. A passing result does not establish that every image is decodable or that all near-duplicate pages and cross-source overlaps have been excluded. Saved `split.json` files contain the exact training and validation records, split settings, and the SHA-256 hash of `train.csv` for data tracking.

Before redistributing images, verify the rights associated with the papers and your data source. This repository supplies no evidence of, or permission for, redistribution of external images.
