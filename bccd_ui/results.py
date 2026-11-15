from dataclasses import asdict, dataclass
import math

from .config import CLASS_NAMES


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xywhn: tuple[float, float, float, float]

    def to_dict(self):
        return {"class_name": CLASS_NAMES[self.class_id], **asdict(self)}


def normalize_detections(rows, width, height, threshold, class_ids):
    """Validate and filter once; every view/export consumes these same boxes."""
    if width <= 0 or height <= 0:
        raise ValueError("Kích thước ảnh không hợp lệ.")
    detections = []
    for x1, y1, x2, y2, confidence, raw_class in rows:
        values = (x1, y1, x2, y2, confidence, raw_class)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Model trả về tọa độ hoặc confidence không hợp lệ.")
        class_id = int(raw_class)
        if raw_class != class_id or class_id not in CLASS_NAMES:
            raise ValueError("Model trả về lớp tế bào không được hỗ trợ.")
        if not 0 <= confidence <= 1:
            raise ValueError("Model trả về confidence ngoài khoảng 0–1.")
        if confidence < threshold or class_id not in class_ids:
            continue
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(
            class_id, float(confidence), (x1, y1, x2, y2),
            ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
             (x2 - x1) / width, (y2 - y1) / height),
        ))
    return detections


def count_cells(detections):
    return {name: sum(d.class_id == cid for d in detections)
            for cid, name in CLASS_NAMES.items()}
