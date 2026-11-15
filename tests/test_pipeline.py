import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from bccd_ui.exports import export_csv, export_json, export_png, export_yolo
from bccd_ui.inference import Detector, read_image
from bccd_ui.rendering import render_detections
from bccd_ui.results import count_cells, normalize_detections


class ResultContractTests(unittest.TestCase):
    def test_filtered_boxes_counts_render_and_exports_agree(self):
        rows = [[-10, 5, 50, 25, .7, 0], [10, 10, 40, 30, .69, 0],
                [20, 20, 60, 40, .95, 1], [80, 40, 120, 60, .9, 2],
                [50, 20, 40, 30, .9, 0]]
        detections = normalize_detections(rows, 100, 50, .7, [0, 2])
        self.assertEqual(count_cells(detections), {"Platelet": 1, "RBC": 0, "WBC": 1})
        self.assertEqual(detections[0].bbox_xyxy, (0., 5, 50, 25))
        self.assertEqual(detections[0].bbox_xywhn, (.25, .3, .5, .4))
        report = json.loads(export_json(detections, {"width": 100, "height": 50}))
        csv_rows = list(csv.DictReader(StringIO(export_csv(detections).decode("utf-8-sig"))))
        yolo = export_yolo(detections).decode().splitlines()
        self.assertEqual(sum(report["counts"].values()), len(report["detections"]))
        self.assertEqual(len(csv_rows), len(yolo))
        self.assertEqual(yolo[0], "0 0.250000 0.300000 0.500000 0.400000")
        image = Image.new("RGB", (100, 50), "white")
        preview = render_detections(image, detections)
        self.assertEqual(image.getpixel((0, 5)), (255, 255, 255))
        self.assertNotEqual(preview.tobytes(), image.tobytes())
        self.assertEqual(Image.open(BytesIO(export_png(preview))).size, image.size)

    def test_empty_results_are_valid_downloads(self):
        image = Image.new("RGB", (20, 20), "white")
        self.assertEqual(render_detections(image, []).tobytes(), image.tobytes())
        self.assertEqual(sum(count_cells([]).values()), 0)
        self.assertEqual(export_yolo([]), b"")
        self.assertEqual(len(export_csv([]).decode("utf-8-sig").splitlines()), 1)
        self.assertEqual(json.loads(export_json([], {}))["detections"], [])

    def test_invalid_model_output_is_rejected(self):
        for row in ([0, 0, 10, 10, float("nan"), 0], [0, 0, 10, 10, .8, 8],
                    [0, 0, 10, 10, 1.2, 0]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                normalize_detections([row], 100, 100, .25, [0, 1, 2])

    def test_upload_validation_and_rgb_conversion(self):
        output = BytesIO()
        Image.new("RGBA", (16, 12), (1, 2, 3, 255)).save(output, format="PNG")
        image = read_image(output.getvalue())
        self.assertEqual((image.mode, image.size), ("RGB", (16, 12)))
        for data in (b"", b"not a real image"):
            with self.assertRaises(ValueError):
                read_image(data)
        with patch("bccd_ui.inference.MAX_FILE_BYTES", 1), self.assertRaises(ValueError):
            read_image(output.getvalue())
        with patch("bccd_ui.inference.MAX_IMAGE_PIXELS", 10), self.assertRaises(ValueError):
            read_image(output.getvalue())

    def test_missing_model_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "best.pt"):
            Detector(Path("tests/missing-checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
