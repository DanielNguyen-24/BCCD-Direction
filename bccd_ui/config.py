from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs/bccd/yolov8n_v3/weights/best.pt"
EVALUATION_DIR = ROOT / "runs/evaluation/yolov8n_v3_test"
SAMPLE_DIR = ROOT / "dataset_v3/test/images"
CLASS_NAMES = {0: "Platelet", 1: "RBC", 2: "WBC"}
DISPLAY_NAMES = {0: "Tiểu cầu", 1: "Hồng cầu", 2: "Bạch cầu"}
COLORS = {0: "#b45309", 1: "#dc2626", 2: "#6d28d9"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
