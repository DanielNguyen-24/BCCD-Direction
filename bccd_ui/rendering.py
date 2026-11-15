from PIL import ImageDraw, ImageFont

from .config import CLASS_NAMES, COLORS


def render_detections(image, detections):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default(size=max(12, min(22, image.width // 45)))
    stroke = max(2, image.width // 400)
    for detection in detections:
        box = detection.bbox_xyxy
        color = COLORS[detection.class_id]
        draw.rectangle(box, outline=color, width=stroke)
        label = f"{CLASS_NAMES[detection.class_id]} {detection.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = right - left + 8, bottom - top + 8
        x = max(0, min(box[0], image.width - label_width))
        y = max(0, box[1] - label_height)
        draw.rectangle((x, y, x + label_width, y + label_height), fill=color)
        draw.text((x + 4, y + 4 - top), label, fill="white", font=font)
    return preview
