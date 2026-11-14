
from pathlib import Path
import multiprocessing
import json

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT / "dataset_v3" / "data.yaml"

MODEL_PATH = (
    ROOT / "runs" / "bccd"
    / "yolov8n_v3" / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "runs" / "evaluation"
RUN_NAME = "yolov8n_v3_test"


def main():
    # 1. Kiểm tra file
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy data.yaml: {DATA_YAML}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy best.pt: {MODEL_PATH}"
        )

    print("=" * 55)
    print("BCCD - YOLOv8n V3 EVALUATION")
    print("=" * 55)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print("Split: test")
    print("Image size: 640")
    print("=" * 55)

    # 2. Load model
    model = YOLO(str(MODEL_PATH))

    # 3. Đánh giá trên tập test
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True
    )

    # 4. Tổng hợp metrics
    summary = {
        "model": "YOLOv8n V3",
        "dataset": "BCCD V3",
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_mAP50_95": {}
    }

    print("\n========== OVERALL RESULTS ==========")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")

    print("\n========== PER-CLASS RESULTS ==========")

    for class_id, class_map in enumerate(metrics.box.maps):
        class_name = model.names[class_id]

        summary["per_class_mAP50_95"][class_name] = (
            float(class_map)
        )

        print(f"{class_name}: {class_map:.4f}")

    # 5. Lưu JSON kết quả
    save_dir = Path(metrics.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_dir / "evaluation_summary.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== COMPLETED ==========")
    print(f"Results saved to: {save_dir}")
    print(f"Summary JSON: {report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
