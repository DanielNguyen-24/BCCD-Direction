from io import BytesIO
import os
from threading import Lock
from time import perf_counter
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CLASS_NAMES, MAX_FILE_BYTES, MAX_IMAGE_PIXELS
from .results import normalize_detections

# Keep the JPEG/PNG decoder before Ultralytics patches PIL.Image.open to
# attempt HEIF plugin installation on unidentified input.
_open_image = Image.open


def read_image(data):
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError("Chọn ảnh JPG/PNG có dung lượng tối đa 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with _open_image(BytesIO(data), formats=("JPEG", "PNG")) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Chỉ hỗ trợ nội dung ảnh JPEG hoặc PNG.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Ảnh vượt quá giới hạn 20 triệu pixel.")
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Không đọc được ảnh. Hãy chọn file JPG/PNG hợp lệ.") from exc


class Detector:
    def __init__(self, model_path):
        if not model_path.is_file():
            raise FileNotFoundError("Thiếu best.pt. Đặt trọng số tại runs/bccd/yolov8n_v3/weights/best.pt.")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        if self.model.names != CLASS_NAMES:
            raise ValueError("Checkpoint phải có mapping 0=Platelet, 1=RBC, 2=WBC.")
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.lock = Lock()

    def predict(self, image, threshold, iou, class_ids):
        started = perf_counter()
        with self.lock:
            result = self.model.predict(
                source=image, imgsz=640, conf=threshold, iou=iou,
                device=self.device, verbose=False, max_det=1000,
            )[0]
            rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
        detections = normalize_detections(rows, *image.size, threshold, class_ids)
        return detections, round((perf_counter() - started) * 1000, 1)
