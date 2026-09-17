"""训练 PaperCNN，记录指标并保存最佳检查点。

Train PaperCNN, record metrics, and save the best checkpoint.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

import config
from data import LABELS, make_loaders
from model import PaperCNN, train_batch


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请选择 cpu 或可用的 mps / CUDA is unavailable; choose cpu or an available mps device")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用，请选择 cpu / MPS is unavailable; choose cpu")
    return torch.device(name)


def evaluate(model, loader, device):
    """评估完整划分，不更新参数。 / Evaluate an entire split without updating parameters."""
    model.eval()
    loss_sum, count = 0.0, 0
    confusion = torch.zeros(len(LABELS), len(LABELS), dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            loss_sum += nn.functional.cross_entropy(scores, labels).item() * len(labels)
            predictions = scores.argmax(dim=1)
            indices = (labels * len(LABELS) + predictions).cpu()
            confusion += torch.bincount(indices, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
            count += len(labels)
    if count == 0:
        raise ValueError("评估数据为空 / Evaluation data is empty")
    true_positives = confusion.diag().float()
    denominators = confusion.sum(0) + confusion.sum(1)
    f1 = 2 * true_positives / denominators.clamp_min(1)
    return {"loss": loss_sum / count, "accuracy": true_positives.sum().item() / count,
            "macro_f1": f1.mean().item(), "num_examples": count,
            "confusion_matrix": confusion.tolist()}


def train_one_epoch(model, loader, optimizer, device):
    """逐批训练并汇总更新前指标。 / Train each batch and aggregate pre-update metrics."""
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss, scores = train_batch(model, images, labels, optimizer)
        loss_sum += loss.item() * len(labels)
        correct += (scores.argmax(1) == labels).sum().item()
        count += len(labels)
    if count == 0:
        raise ValueError("训练数据为空 / Training data is empty")
    return {"loss": loss_sum / count, "accuracy": correct / count, "num_examples": count}


def make_optimizer(model, learning_rate, weight_decay):
    # 仅衰减权重；偏置单独配置。 / Apply weight decay to weights only, excluding biases.
    weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
    biases = [p for name, p in model.named_parameters() if name.endswith("bias")]
    return torch.optim.Adam([
        {"params": weights, "weight_decay": weight_decay},
        {"params": biases, "weight_decay": 0.0},
    ], lr=learning_rate)


def save_curves(history, output):
    os.environ.setdefault("MPLCONFIGDIR", str(config.PROJECT_ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "val"):
        for ax, key in zip(axes, ("loss", "accuracy")):
            ax.plot([h["epoch"] for h in history], [h[split][key] for h in history], label=split)
            ax.set(xlabel="Epoch", ylabel=key)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT,
                        help="数据根目录 / Dataset root directory")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="训练轮数 / Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="每批样本数 / Examples per batch")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Adam 学习率 / Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY,
                        help="权重衰减系数 / Weight decay coefficient")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=config.DEVICE,
                        help="计算设备 / Compute device")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs、batch-size、lr 必须是正数 / epochs, batch-size, and lr must be positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay 必须是非负数 / weight-decay must be nonnegative")

    # 训练前检查模型与单批更新。 / Validate the model and batch update before training.
    from check import run
    for part in ("model", "step"):
        result = run(part)
        if result["status"] != "PASS":
            print("模型或训练步骤检查未通过，请修复后重试。 / Model or training-step checks did not pass; fix the reported issue before training.")
            return 2 if result["status"] == "PENDING" else 1
    device = resolve_device(args.device)
    random.seed(config.MODEL_SEED)
    np.random.seed(config.MODEL_SEED)
    torch.manual_seed(config.MODEL_SEED)
    loaders = make_loaders(args.data_root, args.batch_size, config.IMAGE_SIZE,
                           config.SPLIT_SEED, config.VAL_PER_CLASS, config.NUM_WORKERS)
    train_loader, val_loader, split_info = loaders
    model = PaperCNN(num_classes=len(LABELS)).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    output = config.RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    settings = {**vars(args), "data_root": str(args.data_root.resolve()),
                "image_size": list(config.IMAGE_SIZE), "labels": list(LABELS),
                "split_seed": config.SPLIT_SEED, "model_seed": config.MODEL_SEED,
                "train_count": len(train_loader.dataset), "val_count": len(val_loader.dataset),
                "torch_version": torch.__version__}
    (output / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2) + "\n")
    print(f"训练 / Train: {len(train_loader.dataset)}；验证 / Validation: {len(val_loader.dataset)}；"
          f"输出 / Output: {output}", flush=True)
    history, best_accuracy, best_epoch = [], -1.0, None
    probe_images = next(iter(val_loader))[0].to(device)
    best_scores = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"轮次 / Epoch {epoch}/{args.epochs}: 训练 / train loss={train_metrics['loss']:.4f}, "
              f"accuracy={train_metrics['accuracy']:.1%}; 验证 / val loss={val_metrics['loss']:.4f}, "
              f"accuracy={val_metrics['accuracy']:.1%}, macro-F1={val_metrics['macro_f1']:.3f}", flush=True)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val_metrics["accuracy"], epoch
            with torch.no_grad():
                best_scores = model(probe_images).detach().cpu().clone()
            torch.save({"model_state_dict": model.state_dict(), "num_classes": len(LABELS),
                        "labels": list(LABELS), "image_size": list(config.IMAGE_SIZE),
                        "preprocessing": "grayscale; aspect-preserving white padding; 1 - pixel/255",
                        "split_info": split_info, "data_root": settings["data_root"],
                        "epoch": epoch, "val_accuracy": best_accuracy}, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")

    # 从磁盘恢复并核对同一批预测。 / Reload from disk and verify predictions on the same batch.
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    restored = PaperCNN(num_classes=checkpoint["num_classes"]).to(device)
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(probe_images).cpu(), best_scores, rtol=1e-5, atol=1e-6)
    summary = {"best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
               "restored_val": evaluate(restored, val_loader, device),
               "reload_logits_match": True, "test_set_used": False,
               "training_metrics": "each batch before parameter update; loss is mean cross entropy"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    save_curves(history, output)
    print(f"完成 / Complete: 最佳验证准确率 / best validation accuracy {best_accuracy:.1%}，"
          f"轮次 / epoch {best_epoch}；检查点恢复验证通过 / checkpoint reload verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
