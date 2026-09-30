"""论文首页数据：读取 CSV、固定分层划分与按需加载图片。
First-page data: CSV validation, deterministic stratified splitting, and lazy loading.
"""

import csv
import hashlib
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset


LABELS = ("math", "physics", "cs")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数 / must be a positive integer")


def normalize_arxiv_id(arxiv_id):
    """去掉首尾空白和末尾版本号，得到用于去重的论文 ID。
    Strip surrounding whitespace and a trailing version suffix for deduplication.
    """
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def paper_identity(image_path):
    """从图片文件名提取不带版本号的论文 ID。
    Extract a version-independent paper ID from an image filename.
    """
    return normalize_arxiv_id(Path(image_path).stem)


def read_rows(root, csv_name="train.csv"):
    """读取清单和文件信息，返回 image_path / label 字典列表，不打开图片。
    Read manifest and file metadata into image_path / label dictionaries without decoding images.

    图片按带版本的 arXiv ID 命名；去掉末尾 vN 后检查重复论文。
    Filenames use versioned arXiv IDs; removing trailing vN detects duplicate papers.
    返回的路径始终相对于数据根目录。
    Returned paths remain relative to the dataset root.
    """
    root = Path(root).expanduser().resolve()
    csv_path = (root / csv_name).resolve()
    if not csv_path.is_relative_to(root):
        raise ValueError("CSV 必须位于数据根目录内 / CSV must be within the dataset root")
    images_root = (root / "images").resolve()
    rows, seen_paths, seen_papers = [], set(), set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV 必须包含 image_path 和 label 两列 / "
                "CSV must contain image_path and label columns"
            )
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("image_path") or "").strip()
            label = (row.get("label") or "").strip()
            relative = Path(name)
            if label not in LABEL_TO_INDEX:
                raise ValueError(
                    f"第 {line_number} 行包含未知标签 / Unknown label on line {line_number}: {label!r}"
                )
            if (
                not name or relative.is_absolute() or ".." in relative.parts
                or len(relative.parts) < 3 or relative.parts[0] != "images"
                or any(part.startswith("._") for part in relative.parts)
                or relative.suffix.lower() != ".png"
            ):
                raise ValueError(
                    f"第 {line_number} 行不是正式图片的相对路径 / "
                    f"Line {line_number} must reference a regular PNG with a relative images/ path"
                )
            image_path = (root / relative).resolve()
            if not image_path.is_relative_to(images_root):
                raise ValueError(
                    f"第 {line_number} 行的图片指向 images 目录之外 / "
                    f"Image on line {line_number} resolves outside the images directory"
                )
            if not image_path.is_file():
                raise FileNotFoundError(f"清单图片不存在 / Manifest image does not exist: {image_path}")
            paper_id = paper_identity(relative)
            if image_path in seen_paths:
                raise ValueError(
                    f"第 {line_number} 行重复引用同一路径 / Duplicate image path on line {line_number}"
                )
            if paper_id in seen_papers:
                raise ValueError(
                    f"第 {line_number} 行重复引用同一论文或其另一版本 / "
                    f"Duplicate paper or alternate version on line {line_number}"
                )
            seen_paths.add(image_path)
            seen_papers.add(paper_id)
            rows.append({"image_path": relative.as_posix(), "label": label})
    if not rows:
        raise ValueError("CSV 中没有样本 / CSV contains no samples")
    return rows


def split_train_val(rows, seed=231, val_per_class=200):
    """仅划分传入的训练清单；每类先按路径排序，再按固定种子打乱。
    Split only the supplied training rows, sorting each class by path before seeded shuffling.

    按 LABELS 顺序处理，每类前 val_per_class 条用于验证，其余用于拟合。
    Process classes in LABELS order; use the first val_per_class rows for validation and the rest for fitting.
    相同训练清单和种子会得到相同划分。
    The same training manifest and seed produce the same split.
    """
    _positive_int(val_per_class, "val_per_class")
    if any(row["label"] not in LABEL_TO_INDEX for row in rows):
        raise ValueError("划分数据中包含未知标签 / Split data contains an unknown label")
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for label in LABELS:
        group = sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: row["image_path"],
        )
        if len(group) <= val_per_class:
            raise ValueError(
                f"{label} 样本不足：划出验证集后还需要训练样本 / "
                f"Not enough {label} samples to retain training rows after validation splitting"
            )
        order = rng.permutation(len(group))
        val.extend(dict(group[index]) for index in order[:val_per_class])
        fit.extend(dict(group[index]) for index in order[val_per_class:])
    return fit, val


ABLATION_MODES = ("none", "top-strip", "body-only")


def ablation_settings(mode="none"):
    """Return the complete, versioned geometry of a deterministic cue ablation.

    Coordinates refer to the original grayscale page, before resize/padding.
    These geometric regions do not detect or identify semantic content.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown cue ablation: {mode!r}; choose from {ABLATION_MODES}")
    settings = {"mode": mode, "version": 1, "stage": "before_resize_and_padding"}
    if mode == "top-strip":
        settings.update(top_fraction=0.25, fill_gray=255, rounding="ceil")
    elif mode == "body-only":
        settings.update(top_fraction=0.25, bottom_fraction=0.90,
                        horizontal_region="full_width", rounding="floor", min_height_rows=1)
    return settings


def apply_ablation(image, mode="none"):
    """Mask/crop a grayscale PIL page without mutating the input image."""
    settings = ablation_settings(mode)  # Validate even for direct callers.
    width, height = image.size
    if mode == "top-strip":
        image = image.copy()
        image.paste(255, (0, 0, width, math.ceil(height * settings["top_fraction"])))
    elif mode == "body-only":
        top = math.floor(height * settings["top_fraction"])
        bottom = max(top + settings["min_height_rows"], math.floor(height * settings["bottom_fraction"]))
        image = image.crop((0, top, width, bottom))
    return image


def load_image(path, image_size=(128, 96), ablation="none"):
    """image_size 为 (高, 宽)，输出 float32 张量，形状为 (1, 高, 宽)。
    Accept image_size as (height, width); return a float32 tensor of shape (1, height, width).

    等比缩放并补白，将灰度转换为白底 0、黑墨 1。
    Resize with aspect ratio preserved, pad with white, and map white to 0 and black ink to 1.
    变换不依赖学习到的统计量；每次读取一致，且不修改原图。
    The transform uses no learned statistics, is deterministic, and leaves source images unchanged.
    Optional cue masking/cropping acts on the original page before resizing and padding.
    """
    ablation_settings(ablation)
    if len(image_size) != 2:
        raise ValueError("image_size 必须是 (高, 宽) / image_size must be (height, width)")
    height, width = image_size
    _positive_int(height, "图像高度 / image height")
    _positive_int(width, "图像宽度 / image width")
    with Image.open(path) as image:
        image = ImageOps.pad(
            apply_ablation(image.convert("L"), ablation), (width, height),
            method=Image.Resampling.LANCZOS, color=255,
        )
        gray = np.asarray(image, dtype=np.float32)
    ink = np.float32(1.0) - gray / np.float32(255.0)
    return torch.from_numpy(ink).unsqueeze(0)


class PageDataset(Dataset):
    """仅在取样时打开图片；rows 来自 read_rows 或其划分。
    Open images on demand; rows come from read_rows or its train/validation split.
    """

    def __init__(self, root, rows, image_size=(128, 96), ablation="none"):
        self.root = Path(root).expanduser().resolve()
        self.rows = [dict(row) for row in rows]
        self.image_size = tuple(image_size)
        ablation_settings(ablation)
        self.ablation = ablation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_image(self.root / row["image_path"], self.image_size, self.ablation)
        target = LABEL_TO_INDEX[row["label"]]
        return image, target


def make_loaders(
    root, batch_size=16, image_size=(128, 96), split_seed=231,
    val_per_class=200, num_workers=0, ablation="none",
):
    """构造训练/验证 DataLoader，并返回可保存到 JSON 的划分信息。
    Build train/validation DataLoaders and return JSON-serializable split metadata.

    只读取 train.csv，不读取 test.csv 或测试图片。
    Read only train.csv, without accessing test.csv or test images.
    训练顺序使用独立的 torch.Generator；验证顺序固定，不预缓存图片。
    Use a dedicated torch.Generator for training order, fixed validation order, and no image precaching.
    """
    _positive_int(batch_size, "batch_size")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers 必须是非负整数 / num_workers must be a nonnegative integer")
    root = Path(root).expanduser().resolve()
    rows = read_rows(root, "train.csv")
    train_rows, val_rows = split_train_val(rows, split_seed, val_per_class)
    generator = torch.Generator().manual_seed(split_seed)
    train_loader = DataLoader(
        PageDataset(root, train_rows, image_size, ablation), batch_size=batch_size,
        shuffle=True, generator=generator, num_workers=num_workers,
    )
    val_loader = DataLoader(
        PageDataset(root, val_rows, image_size, ablation), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    split_info = {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "split_seed": int(split_seed),
        "val_per_class": int(val_per_class),
        "train_csv_sha256": hashlib.sha256((root / "train.csv").read_bytes()).hexdigest(),
    }
    return train_loader, val_loader, split_info
