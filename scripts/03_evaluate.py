
from ultralytics import YOLO
from pathlib import Path
import torch

MODEL_PATH = Path(
    r"C:\Users\DanielNguyen\runs\detect\runs\bccd\yolov8n_initial-4\weights\best.pt"
)

DATA_YAML = Path("initial_dataset_v2/data.yaml")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_PATH)
    print("Dataset:", DATA_YAML)

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        device=0,
        workers=0
    )

    print("\n===== EVALUATION RESULTS =====")
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\n===== PER-CLASS mAP50-95 =====")
    for i, value in enumerate(metrics.box.maps):
        print(f"Class {i}: {value:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
