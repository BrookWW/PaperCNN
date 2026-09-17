"""路径、数据划分与训练的默认配置；模型结构见 model.py。
Default paths, data split, and training settings; see model.py for the architecture.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# 用 ARXIV_DATA_ROOT 指定外部数据集；默认读取项目内的 data 目录。
# Set ARXIV_DATA_ROOT for an external dataset; default to the project's data directory.
DATA_ROOT = Path(os.environ.get("ARXIV_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
RUNS_DIR = PROJECT_ROOT / "runs"

# (高度, 宽度)，保留论文页的纵向比例。 / (Height, width), preserving a portrait layout.
IMAGE_SIZE = (128, 96)
# 固定训练/验证划分；调参时保持不变。 / Keep the train/validation split fixed while tuning.
SPLIT_SEED = 231
# 每类 1000 张时，800 张训练、200 张验证。 / With 1,000 pages per class: 800 train, 200 validation.
VAL_PER_CLASS = 200
# 模型初始化种子；批次顺序由 SPLIT_SEED 控制。 / Initialization seed; SPLIT_SEED controls batch order.
MODEL_SEED = 1718

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-3
# 只作用于权重，偏置不衰减。 / Apply weight decay to weights, excluding biases.
WEIGHT_DECAY = 3e-4
# 可按硬件改为 "mps" 或 "cuda"。 / Use "mps" or "cuda" when supported by your hardware.
DEVICE = "cpu"
# 默认单进程加载，减少跨平台启动开销。 / Default to in-process loading for portable startup.
NUM_WORKERS = 0
