from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from bccd_ui.config import ROOT


class AppFlowTests(unittest.TestCase):
    def test_sample_inference_and_stale_results(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        app.radio(key="source").set_value("Dùng ảnh mẫu").run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 3)
        self.assertGreater(sum(int(metric.value) for metric in app.metric), 0)
        self.assertEqual(len(app.get("download_button")), 4)
        app.slider(key="confidence").set_value(.5).run()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(any("đã thay đổi" in item.value for item in app.info))
        app.multiselect(key="classes").set_value([]).run()
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.exception), 0)

    def test_evaluation_and_missing_report(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        app.radio(key="page").set_value("Hiệu năng model").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([m.value for m in app.metric], ["96.52%", "91.51%", "97.39%", "90.40%"])
        with patch("bccd_ui.config.EVALUATION_DIR", Path("tests/no-report")):
            app = AppTest.from_file(str(ROOT / "app.py")).run()
            app.radio(key="page").set_value("Hiệu năng model").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.warning), 1)


if __name__ == "__main__":
    unittest.main()
