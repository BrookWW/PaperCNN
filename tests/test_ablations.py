"""Synthetic cue-ablation regression tests; no paper-classification evidence."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch

import config
from model import PaperCNN
from data import (ABLATION_MODES, PageDataset, ablation_settings, apply_ablation,
                  load_image, make_loaders)
import predict
import train
from test_pipeline import make_fixture, write_csv


class AblationTests(unittest.TestCase):
    def test_original_pixel_geometry_and_no_mutation(self):
        # Odd height distinguishes ceil masking from floor cropping.
        pixels = np.arange(7 * 11, dtype=np.uint8).reshape(11, 7)
        image = Image.fromarray(pixels)
        masked = np.asarray(apply_ablation(image, "top-strip"))
        np.testing.assert_array_equal(masked[:3], np.full((3, 7), 255))
        np.testing.assert_array_equal(masked[3:], pixels[3:])
        np.testing.assert_array_equal(np.asarray(apply_ablation(image, "body-only")), pixels[2:9])
        np.testing.assert_array_equal(np.asarray(image), pixels)
        for mode in ABLATION_MODES:
            self.assertEqual(apply_ablation(Image.new("L", (1, 1)), mode).size, (1, 1))

    def test_loading_matches_transform_before_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            pixels = np.arange(7 * 11, dtype=np.uint8).reshape(11, 7)
            Image.fromarray(pixels).save(path)
            original_bytes = path.read_bytes()
            masked = pixels.copy()
            masked[:3] = 255
            for mode, expected_pixels in (("none", pixels), ("top-strip", masked),
                                          ("body-only", pixels[2:9])):
                with self.subTest(mode=mode):
                    expected = ImageOps.pad(Image.fromarray(expected_pixels), (16, 16),
                                            method=Image.Resampling.LANCZOS, color=255)
                    expected = 1 - np.asarray(expected, dtype=np.float32) / np.float32(255)
                    actual = load_image(path, (16, 16), mode)
                    torch.testing.assert_close(actual[0], torch.from_numpy(expected), rtol=0, atol=0)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(tuple(actual.shape), (1, 16, 16))
                    self.assertGreaterEqual(actual.min().item(), 0)
                    self.assertLessEqual(actual.max().item(), 1)
                    self.assertTrue(torch.equal(actual, load_image(path, (16, 16), mode)))
            self.assertEqual(path.read_bytes(), original_bytes)
            with self.assertRaises(ValueError):
                load_image(path, ablation="unknown")
            with self.assertRaises(ValueError):
                PageDataset(directory, [], ablation="unknown")

    def test_loaders_share_transform_without_changing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root)
            baseline_split = None
            for mode in ABLATION_MODES:
                fit, val, split = make_loaders(root, image_size=(32, 24),
                                              val_per_class=1, ablation=mode)
                if baseline_split is None:
                    baseline_split = split
                self.assertEqual(split, baseline_split)
                for loader in (fit, val):
                    dataset = loader.dataset
                    self.assertEqual(dataset.ablation, mode)
                    path = root / dataset.rows[0]["image_path"]
                    self.assertTrue(torch.equal(dataset[0][0], load_image(path, (32, 24), mode)))

    def test_training_prediction_metadata_inheritance_and_overrides(self):
        for mode in ("top-strip", "body-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = make_fixture(root)
                with patch.multiple(config, RUNS_DIR=root / "runs", VAL_PER_CLASS=1, IMAGE_SIZE=(32, 24)):
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(train.main(["--data-root", str(root), "--epochs", "1",
                                                     "--batch-size", "3", "--ablation", mode]), 0)
                    output = next((root / "runs").iterdir())
                    checkpoint_path = output / "best.pt"
                    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
                    summary = json.loads((output / "summary.json").read_text())
                    for record in (checkpoint, summary, json.loads((output / "config.json").read_text())):
                        self.assertEqual(record["ablation"], ablation_settings(mode))
                    base = ["--checkpoint", str(checkpoint_path)]
                    image_args = ["--image", str(root / rows[0]["image_path"])]
                    with redirect_stdout(io.StringIO()):
                        report = predict.main(base + image_args)
                        explicit = predict.main(base + image_args + ["--ablation", mode])
                        val = predict.main(base + ["--split", "val"])
                        override = predict.main(base + image_args + ["--ablation", "none",
                                                "--output-json", str(root / "probe.json")])
                    expected_model = PaperCNN(num_classes=len(checkpoint["labels"]))
                    expected_model.load_state_dict(checkpoint["model_state_dict"])
                    expected_model.eval()
                    for result, effective_mode in ((report, mode), (override, "none")):
                        tensor = load_image(root / rows[0]["image_path"], (32, 24), effective_mode)
                        with torch.no_grad():
                            expected = torch.softmax(expected_model(tensor.unsqueeze(0)), dim=1)[0]
                        torch.testing.assert_close(torch.tensor(list(result["probabilities"].values())),
                                                   expected, rtol=0, atol=0)
                    self.assertEqual(report["probabilities"], explicit["probabilities"])
                    self.assertEqual(report["ablation"], ablation_settings(mode))
                    self.assertFalse(report["ablation_override"])
                    self.assertTrue(report["preprocessing_matches_training"])
                    self.assertEqual(val["confusion_matrix"], summary["restored_val"]["confusion_matrix"])
                    self.assertTrue(override["ablation_override"])
                    self.assertFalse(override["preprocessing_matches_training"])
                    self.assertEqual(override["training_ablation"], ablation_settings(mode))
                    self.assertEqual(override["ablation"], ablation_settings())
                    self.assertEqual(json.loads((root / "probe.json").read_text()), override)
                    # Disjoint synthetic test split uses the same inherited transform.
                    relative = "images/math/2601.99999v1.png"
                    (root / relative).write_bytes((root / rows[0]["image_path"]).read_bytes())
                    write_csv(root / "test.csv", [{"image_path": relative, "label": "math"}])
                    with redirect_stdout(io.StringIO()):
                        test = predict.main(base + ["--split", "test"])
                    self.assertEqual(test["ablation"], ablation_settings(mode))
                    self.assertEqual(test["num_examples"], 1)
                    # Old checkpoints must default to the original preprocessing.
                    del checkpoint["ablation"]
                    torch.save(checkpoint, root / "legacy.pt")
                    with redirect_stdout(io.StringIO()):
                        legacy = predict.main(["--checkpoint", str(root / "legacy.pt")] + image_args)
                    self.assertEqual(legacy["ablation"], ablation_settings())
                    checkpoint["ablation"] = {**ablation_settings(mode), "version": 999}
                    torch.save(checkpoint, root / "unsupported.pt")
                    with self.assertRaisesRegex(ValueError, "ablation metadata"):
                        predict.main(["--checkpoint", str(root / "unsupported.pt")] + image_args)

    def test_cli_rejects_unknown_modes(self):
        for entry, args in ((train.main, []), (predict.main, ["--checkpoint", "unused", "--image", "unused"])):
            with self.subTest(entry=entry.__module__), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entry(args + ["--ablation", "unknown"])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
