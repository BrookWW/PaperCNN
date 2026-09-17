"""使用已训练模型预测论文首页或评估验证集、测试集。

Predict a paper's first page or evaluate a saved validation split or test set.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data import LABELS, PageDataset, load_image, paper_identity, read_rows
from model import PaperCNN
from train import evaluate, resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="模型检查点路径 / Model checkpoint path")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", type=Path, help="单张首页图片 / One first-page image to classify")
    target.add_argument("--split", choices=("val", "test"),
                        help="模型选择后再评估 test / Evaluate test only after model selection")
    parser.add_argument("--data-root", type=Path,
                        help="覆盖检查点中的数据根目录 / Override the dataset root stored in the checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=config.DEVICE,
                        help="计算设备 / Compute device")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location=device)
    if checkpoint["labels"] != list(LABELS) or checkpoint["num_classes"] != len(LABELS):
        raise ValueError("模型类别顺序与本项目不一致 / Checkpoint class order does not match this project")
    model = PaperCNN(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_size = tuple(checkpoint["image_size"])
    if args.image:
        image = load_image(args.image, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(image), dim=1)[0].cpu()
        report = {"image": str(args.image), "predicted_label": LABELS[probabilities.argmax().item()],
                  "probabilities": dict(zip(LABELS, probabilities.tolist()))}
    else:
        root = args.data_root or Path(checkpoint["data_root"])
        if args.split == "val":
            rows = checkpoint["split_info"]["val_rows"]
        else:
            rows = read_rows(root, "test.csv")
            development = checkpoint["split_info"]["train_rows"] + checkpoint["split_info"]["val_rows"]
            development_paths = {r["image_path"] for r in development}
            if development_paths & {r["image_path"] for r in rows}:
                raise ValueError("测试集与训练/验证图片路径存在重叠 / Test image paths overlap with training or validation")
            # 忽略版本号，避免同一论文跨划分泄漏。 / Ignore version suffixes to prevent paper-level leakage.
            development_ids = {paper_identity(r["image_path"]) for r in development}
            test_ids = {paper_identity(r["image_path"]) for r in rows}
            if development_ids & test_ids:
                raise ValueError("测试集与训练/验证包含同一论文或其另一版本 / Test papers overlap with training or validation, possibly across versions")
        loader = DataLoader(PageDataset(root, rows, image_size), batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)
        report = {"split": args.split, "label_order": list(LABELS), **evaluate(model, loader, device)}
        print("混淆矩阵：行是真实类别，列是预测类别；顺序 math、physics、cs。 / "
              "Confusion matrix: rows are true labels, columns are predictions; order: math, physics, cs.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
