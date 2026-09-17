"""不依赖真实论文的流程回归测试。
Pipeline regression tests that use synthetic pages instead of real papers.
"""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch

import config
from data import LABELS, load_image, read_rows, split_train_val
from model import PaperCNN
import predict
import train


def write_csv(path, rows):
    """写入最小数据清单。 / Write a minimal dataset manifest."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "label"))
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root):
    """生成三类不同条纹的合成页，供流程测试使用。
    Generate three classes of striped synthetic pages for pipeline testing.
    """
    rows = []
    for label_index, label in enumerate(LABELS):
        directory = root / "images" / label
        directory.mkdir(parents=True)
        for index in range(3):
            relative = f"images/{label}/2601.{label_index * 10 + index:05d}v1.png"
            image = Image.new("L", (24, 32), 255)
            draw = ImageDraw.Draw(image)
            x = 2 + label_index * 6
            draw.rectangle((x, 4, x + 3, 24 + index), fill=0)
            image.save(root / relative)
            rows.append({"image_path": relative, "label": label})
    write_csv(root / "train.csv", rows)
    return rows


class DataTests(unittest.TestCase):
    def test_split_is_disjoint_and_independent_of_csv_order(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = make_fixture(Path(directory))
            fit, val = split_train_val(rows, val_per_class=1)
            self.assertEqual((fit, val), split_train_val(list(reversed(rows)), val_per_class=1))
            fit_paths = {row["image_path"] for row in fit}
            val_paths = {row["image_path"] for row in val}
            self.assertFalse(fit_paths & val_paths)
            self.assertEqual(len(fit_paths | val_paths), 9)

    def test_preprocessing_keeps_white_padding_and_ink_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            Image.new("L", (8, 16), 0).save(path)
            before = path.read_bytes()
            tensor = load_image(path, (16, 16))
            self.assertEqual(tuple(tensor.shape), (1, 16, 16))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.equal(tensor[0, :, :4], torch.zeros(16, 4)))
            self.assertTrue(torch.equal(tensor[0, :, 4:12], torch.ones(16, 8)))
            self.assertEqual(before, path.read_bytes())

    def test_manifest_rejects_path_escape_and_duplicate_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = make_fixture(root)
            write_csv(root / "invalid.csv", [{"image_path": "../escape.png", "label": "math"}])
            with self.assertRaises(ValueError):
                read_rows(root, "invalid.csv")
            alternate = rows[0]["image_path"].replace("v1.png", "v2.png")
            (root / alternate).write_bytes((root / rows[0]["image_path"]).read_bytes())
            write_csv(root / "duplicate.csv", [rows[0], {"image_path": alternate, "label": "math"}])
            with self.assertRaises(ValueError):
                read_rows(root, "duplicate.csv")


class PipelineTests(unittest.TestCase):
    def test_train_save_reload_predict_and_reject_test_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = make_fixture(root)
            with patch.multiple(config, RUNS_DIR=root / "runs", VAL_PER_CLASS=1, IMAGE_SIZE=(32, 24)):
                with redirect_stdout(io.StringIO()):
                    status = train.main(["--data-root", str(root), "--epochs", "1", "--batch-size", "3"])
                self.assertEqual(status, 0)
                output = next((root / "runs").iterdir())
                checkpoint = output / "best.pt"
                summary = json.loads((output / "summary.json").read_text())
                self.assertTrue(summary["reload_logits_match"])
                self.assertFalse(summary["test_set_used"])
                self.assertEqual(summary["restored_val"]["num_examples"], 3)
                self.assertTrue((output / "curves.png").is_file())
                with redirect_stdout(io.StringIO()):
                    report = predict.main(["--checkpoint", str(checkpoint), "--image", str(root / rows[0]["image_path"])])
                    validation = predict.main(["--checkpoint", str(checkpoint), "--data-root", str(root), "--split", "val"])
                self.assertIn(report["predicted_label"], LABELS)
                self.assertAlmostEqual(sum(report["probabilities"].values()), 1.0, places=6)
                self.assertEqual(validation["confusion_matrix"], summary["restored_val"]["confusion_matrix"])

                # 不同版本的同一论文必须在读取测试像素前被拒绝。
                # Reject alternate versions of development papers before reading test pixels.
                alternate = rows[0]["image_path"].replace("v1.png", "v2.png")
                (root / alternate).write_bytes((root / rows[0]["image_path"]).read_bytes())
                write_csv(root / "test.csv", [{"image_path": alternate, "label": "math"}])
                with self.assertRaisesRegex(ValueError, "overlap|appears|paper|版本|论文"):
                    predict.main(["--checkpoint", str(checkpoint), "--data-root", str(root), "--split", "test"])

    def test_published_checkpoint_has_expected_model_and_provenance(self):
        path = Path(__file__).resolve().parents[1] / "checkpoints" / "papercnn.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = PaperCNN(num_classes=checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 5955)
        self.assertEqual(checkpoint["labels"], list(LABELS))
        self.assertEqual(checkpoint["data_root"], "data")
        self.assertEqual(checkpoint["epoch"], 6)
        self.assertAlmostEqual(checkpoint["val_accuracy"], 0.765)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 1, 128, 96)).shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
