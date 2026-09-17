"""检查数据完整性、模型结构和单批训练步骤。

Check dataset integrity, model structure, and the single-batch training step.
"""
import argparse
from collections import Counter
import csv
from pathlib import Path

import config


def check_data(data_root):
    import torch
    from data import LABELS, PageDataset, normalize_arxiv_id, paper_identity, read_rows, split_train_val

    root = Path(data_root).expanduser().resolve()
    train = read_rows(root, "train.csv")
    # 仅读测试集元数据，不打开测试图片。 / Read test metadata without opening test images.
    test = read_rows(root, "test.csv")
    fit, val = split_train_val(train, config.SPLIT_SEED, config.VAL_PER_CLASS)
    assert (fit, val) == split_train_val(train, config.SPLIT_SEED, config.VAL_PER_CLASS), "相同种子必须得到相同划分 / The same seed must reproduce the split"
    paths = []
    for name, rows, per_class in (("训练池 / Training pool", train, 1000), ("拟合 / Fit", fit, 1000 - config.VAL_PER_CLASS), ("验证 / Validation", val, config.VAL_PER_CLASS), ("测试 / Test", test, 300)):
        assert Counter(row["label"] for row in rows) == Counter({label: per_class for label in LABELS}), f"{name}: 每类应为 {per_class} 张 / Expected {per_class} images per class"
        normalized = [(root / row["image_path"]).resolve() for row in rows]
        assert len(set(normalized)) == len(rows), f"{name}: 存在重复图片路径 / Duplicate image paths"
        assert all(path.is_relative_to(root.resolve()) and path.is_file() for path in normalized), f"{name}: 图片路径越界或文件缺失 / Image paths escape the root or files are missing"
        paths.append(set(normalized))
    train_paths, fit_paths, val_paths, test_paths = paths
    assert not fit_paths & val_paths and fit_paths | val_paths == train_paths, "拟合与验证应互斥并完整覆盖 train.csv / Fit and validation must partition train.csv without overlap"
    assert not train_paths & test_paths, "训练与测试图片路径有交集 / Training and test image paths overlap"
    train_ids = {paper_identity(row["image_path"]) for row in train}
    test_ids = {paper_identity(row["image_path"]) for row in test}
    assert not train_ids & test_ids, "训练与测试包含同一论文的不同版本 / Training and test contain versions of the same paper"
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["image_path"]: row for row in manifest_rows}
    assert len(manifest) == len(manifest_rows), "manifest 中有重复路径 / Duplicate paths in manifest"
    assert set(manifest) == {row["image_path"] for row in train + test}, "manifest 与两个 CSV 的图片清单不一致 / Manifest and split CSV image lists differ"
    paper_ids = {"train": set(), "test": set()}
    for split, rows in (("train", train), ("test", test)):
        for row in rows:
            meta = manifest[row["image_path"]]
            assert (meta["label"], meta["split"]) == (row["label"], split), "manifest 类别或划分不一致 / Manifest label or split does not match"
            paper_ids[split].add(normalize_arxiv_id(meta["arxiv_id"]))
    assert not paper_ids["train"] & paper_ids["test"], "同一论文出现在训练集与测试集 / The same paper appears in training and test"
    samples = [next(row for row in fit if row["label"] == label) for label in LABELS]
    dataset = PageDataset(root, samples, config.IMAGE_SIZE)
    assert len(dataset) == len(LABELS)
    for index, label in enumerate(LABELS):
        image, target = dataset[index]
        assert isinstance(image, torch.Tensor) and image.dtype == torch.float32, "图片应为 float32 Tensor / Images must be float32 tensors"
        assert tuple(image.shape) == (1, *config.IMAGE_SIZE), "图片形状应为 (1, 高, 宽) / Image shape must be (1, height, width)"
        assert torch.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1, "图片必须有限且归一化到 [0, 1] / Images must be finite and normalized to [0, 1]"
        assert int(target) == index, f"{label} 的类别编号应为 {index} / Class index for {label} must be {index}"
    return (f"每类 / Per class: fit {1000 - config.VAL_PER_CLASS}, val {config.VAL_PER_CLASS}, test 300；"
            "划分无交集；每类抽查 1 张训练图，未读测试图片 / "
            "No split overlap; checked one training image per class without opening test images")


def check_model():
    import torch
    from torch import nn
    from model import PaperCNN

    torch.manual_seed(231)
    model = PaperCNN(num_classes=3)
    for name, in_channels, out_channels in (("conv1", 1, 16), ("conv2", 16, 32)):
        layer = getattr(model, name)
        assert isinstance(layer, nn.Conv2d), f"{name} 应为 Conv2d / {name} must be Conv2d"
        assert (layer.in_channels, layer.out_channels, layer.kernel_size, layer.padding) == (in_channels, out_channels, (3, 3), (1, 1)), f"{name} 的通道数、卷积核或 padding 不符 / Incorrect channels, kernel size, or padding in {name}"
    assert isinstance(model.pool, nn.MaxPool2d) and model.pool.kernel_size in (2, (2, 2)), "pool 应为 2×2 MaxPool2d / pool must be 2×2 MaxPool2d"
    assert isinstance(model.readout, nn.AdaptiveAvgPool2d) and model.readout.output_size == (4, 3), "readout 应为自适应平均池化到 (4, 3) / readout must adaptively average-pool to (4, 3)"
    assert isinstance(model.fc, nn.Linear) and (model.fc.in_features, model.fc.out_features) == (32 * 4 * 3, 3), "fc 输入应为 384，输出应为 3 / fc must map 384 input features to 3 outputs"
    parameters = dict(model.named_parameters())
    assert set(parameters) == {f"{layer}.{kind}" for layer in ("conv1", "conv2", "fc") for kind in ("weight", "bias")}, "模型应注册这三层的共 6 组参数 / Register all six weight and bias tensors across the three layers"
    readout_shapes, pool_shapes, fc_scores = [], [], []
    hooks = [
        model.readout.register_forward_hook(lambda module, inputs, output: readout_shapes.append(tuple(output.shape))),
        model.pool.register_forward_hook(lambda module, inputs, output: pool_shapes.append(tuple(output.shape))),
        model.fc.register_forward_hook(lambda module, inputs, output: fc_scores.append(output.detach().clone())),
    ]
    try:
        for height, width in ((128, 96), (64, 48)):
            scores = model(torch.randn(2, 1, height, width))
            assert scores.shape == (2, 3) and scores.dtype == torch.float32 and torch.isfinite(scores).all(), "输出应为有限的 float32 (N, 3) logits / Output must be finite float32 logits of shape (N, 3)"
            assert pool_shapes[-2:] == [(2, 16, height // 2, width // 2), (2, 32, height // 4, width // 4)], "每次前向应调用两次 pool，各将高宽减半 / Each forward pass must pool twice, halving both spatial dimensions"
            assert fc_scores, "forward 必须调用 fc / forward must call fc"
            torch.testing.assert_close(scores, fc_scores[-1], msg="输出应等于 fc 的原始 logits，不另加 softmax / ReLU / Return raw fc logits without an extra softmax or ReLU")
        assert readout_shapes == [(2, 32, 4, 3)] * 2, "forward 必须调用 readout，固定展平前尺寸 / forward must use readout to fix the shape before flattening"
        assert len(pool_shapes) == 4 and len(fc_scores) == 2, "两次前向应调用 4 次 pool、2 次 fc / Two forward passes must call pool four times and fc twice"
    finally:
        for hook in hooks:
            hook.remove()
    torch.nn.functional.cross_entropy(scores, torch.tensor([0, 2])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values()), "存在未连接到 loss 的参数或非有限梯度 / Parameters are disconnected from loss or have nonfinite gradients"
    before = {name: p.detach().clone() for name, p in parameters.items()}
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert any(not torch.equal(p, before[name]) for name, p in parameters.items()), "梯度没有带来参数更新 / Gradients did not update any parameters"
    return ("6 组参数注册正确；两种输入尺寸可前向、反向与更新；输出原始 logits / "
            "Six parameter tensors registered; forward, backward, and updates verified at two input sizes; raw logits returned")


def check_step():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from model import train_batch

    torch.manual_seed(232)
    fresh_grads, score_grads = {}, []

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 3)
            self.calls = 0

        def forward(self, images):
            scores = self.fc(images.flatten(1))
            self.scores = scores.detach().clone()
            scores.register_hook(lambda grad: score_grads.append(grad.detach().clone()))
            self.calls += 1
            return scores

    model = Probe()
    model.train()
    for name, parameter in model.named_parameters():
        parameter.grad = torch.full_like(parameter, 123.0)
        parameter.register_hook(lambda grad, name=name: fresh_grads.__setitem__(name, grad.detach().clone()))

    class AuditedSGD(torch.optim.SGD):
        def __init__(self):
            super().__init__(model.parameters(), lr=0.1)
            self.steps = 0

        def step(self, closure=None):
            assert len(fresh_grads) == 2, "step 前必须完成当前 batch 的 backward / Complete the current batch backward pass before step"
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, "step 前梯度为空 / Missing gradients before step"
                torch.testing.assert_close(parameter.grad, fresh_grads[name], msg="旧梯度未清零或清零时机不正确 / Stale gradients were not cleared at the correct time")
            self.steps += 1
            fresh_grads.clear()
            return super().step(closure)

    optimizer = AuditedSGD()
    for step, batch_size in enumerate((3, 2), start=1):
        images, labels = torch.randn(batch_size, 1, 4, 4), torch.arange(batch_size)
        before = [p.detach().clone() for p in model.parameters()]
        loss, scores = train_batch(model, images, labels, optimizer)
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0 and not loss.requires_grad, "loss 应为已 detach 的零维 Tensor / Return a detached scalar loss tensor"
        assert isinstance(scores, torch.Tensor) and not scores.requires_grad and scores.grad_fn is None, "scores 应已 detach / Return detached scores"
        assert optimizer.steps == step and model.calls == step, "每个 batch 应恰好进行一次前向和一次更新 / Each batch must perform exactly one forward pass and one update"
        torch.testing.assert_close(scores, model.scores, msg="应返回本 batch 更新前的 logits / Return this batch's pre-update logits")
        torch.testing.assert_close(loss, F.cross_entropy(model.scores, labels), msg="loss 应为平均交叉熵 / Loss must be mean cross entropy")
        # 独立核对 logits 的交叉熵导数。 / Independently verify the cross-entropy gradient with respect to logits.
        expected = (scores.softmax(1) - F.one_hot(labels, 3)) / batch_size
        assert len(score_grads) == step, "每批应恰好反向传播一次 / Each batch must perform exactly one backward pass"
        torch.testing.assert_close(score_grads[-1], expected, msg="backward 的目标必须是平均交叉熵 / Differentiate the mean cross-entropy loss")
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before)), "batch 未更新参数 / The batch did not update parameters"
    return ("连续两批完成清零、交叉熵、反向传播与更新；返回值正确且已 detach / "
            "Two batches verified: zero gradients, cross entropy, backward, step, and correct detached outputs")


def run(part="all", *, data_root=config.DATA_ROOT, verbose=True):
    """返回检查状态和详细结果。 / Return check status and detailed results."""
    available = {"data": lambda: check_data(data_root), "model": check_model, "step": check_step}
    if part not in (*available, "all"):
        raise ValueError("part 必须是 data/model/step/all / part must be data/model/step/all")
    results = []
    for name in (available if part == "all" else (part,)):
        try:
            message = available[name]()
            status = "PASS"
        except NotImplementedError as error:
            status, message = "PENDING", str(error) or "所需实现不可用 / Required implementation is unavailable"
        except Exception as error:
            status, message = "FAIL", f"{type(error).__name__}: {error}"
        results.append({"part": name, "status": status, "message": message})
        if verbose:
            print(f"[{status}] {name}：{message}")
    status = "FAIL" if any(r["status"] == "FAIL" for r in results) else "PENDING" if any(r["status"] == "PENDING" for r in results) else "PASS"
    return {"status": status, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("data", "model", "step", "all"), default="all",
                        help="检查范围 / Checks to run")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT,
                        help="数据根目录 / Dataset root directory")
    arguments = parser.parse_args()
    result = run(arguments.part, data_root=arguments.data_root)
    raise SystemExit({"PASS": 0, "FAIL": 1, "PENDING": 2}[result["status"]])
