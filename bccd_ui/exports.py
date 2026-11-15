import csv
from io import BytesIO, StringIO
import json

from .config import CLASS_NAMES
from .results import count_cells


def export_csv(detections):
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    for d in detections:
        writer.writerow([d.class_id, CLASS_NAMES[d.class_id], d.confidence, *d.bbox_xyxy])
    return output.getvalue().encode("utf-8-sig")


def export_json(detections, metadata):
    return json.dumps({"metadata": metadata, "counts": count_cells(detections),
                       "detections": [d.to_dict() for d in detections]},
                      ensure_ascii=False, indent=2).encode("utf-8")


def export_yolo(detections):
    return "".join(f"{d.class_id} " + " ".join(f"{v:.6f}" for v in d.bbox_xywhn) + "\n"
                   for d in detections).encode("utf-8")


def export_png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
