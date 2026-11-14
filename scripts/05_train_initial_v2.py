
from ultralytics import YOLO
from pathlib import Path
import torch
import multiprocessing


def main():
    DATA_YAML = "initial_dataset_v2/data.yaml"

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    print("GPU:", torch.cuda.get_device_name(0))

    # Check dataset
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Cannot find {DATA_YAML}")

    # Load pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        patience=20,
        amp=True,
        cache=False,
        project="runs/bccd",
        name="yolov8n_initial",
        save=True,
        plots=True,
        seed=42
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
