
from pathlib import Path
import csv
import shutil
import multiprocessing

import cv2
from ultralytics import YOLO


# ==================================================
# 1. ĐƯỜNG DẪN
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    r"C:\Users\DanielNguyen\runs\detect\runs\bccd"
    r"\yolov8n_initial-4\weights\best.pt"
)

SOURCE_DIR = ROOT / "raw_dataset" / "images"

OUTPUT_DIR = ROOT / "auto_label_dataset" / "pilot_2250_v2"

IMAGE_OUTPUT = OUTPUT_DIR / "images"
LABEL_OUTPUT = OUTPUT_DIR / "labels"
PREDICTION_OUTPUT = OUTPUT_DIR / "predictions"


# ==================================================
# 2. CẤU HÌNH
# ==================================================

IMAGE_SIZE = 640
CONFIDENCE = 0.10
PLATELET_CONFIDENCE = 0.70
OTHER_CONFIDENCE = 0.10

DEVICE = 0
MAX_IMAGES = 2250

EXPECTED_NAMES = {
    0: "Platelet",
    1: "RBC",
    2: "WBC"
}


def main():

    # ==================================================
    # 3. KIỂM TRA MODEL VÀ DỮ LIỆU
    # ==================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH}"
        )

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {SOURCE_DIR}"
        )

    model = YOLO(str(MODEL_PATH))

    actual_names = {
        int(k): str(v)
        for k, v in model.names.items()
    }

    print("Model classes:", actual_names)

    if actual_names != EXPECTED_NAMES:
        raise ValueError(
            f"Class mapping không khớp!\n"
            f"Model: {actual_names}\n"
            f"Expected: {EXPECTED_NAMES}"
        )

    # ==================================================
    # 4. TẠO THƯ MỤC ĐẦU RA
    # ==================================================

    for folder in [
        IMAGE_OUTPUT,
        LABEL_OUTPUT,
        PREDICTION_OUTPUT
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # 5. LẤY DANH SÁCH ẢNH
    # ==================================================

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_files = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_files:
        raise RuntimeError("Không tìm thấy ảnh đầu vào.")

    selected_images = image_files[:MAX_IMAGES]

    print(f"Tổng số ảnh: {len(image_files)}")
    print(f"Số ảnh xử lý: {len(selected_images)}")

    # ==================================================
    # 6. KHỞI TẠO BÁO CÁO
    # ==================================================

    report_rows = []
    platelet_review_rows = []

    # ==================================================
    # 7. AUTO-LABEL TỪNG ẢNH
    # ==================================================

    for index, image_path in enumerate(selected_images, start=1):

        print(
            f"\n[{index}/{len(selected_images)}] "
            f"Đang xử lý: {image_path.name}"
        )

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # Sao chép ảnh gốc
        shutil.copy2(
            image_path,
            IMAGE_OUTPUT / image_path.name
        )

        # Tạo đường dẫn nhãn
        label_path = LABEL_OUTPUT / f"{image_path.stem}.txt"

        class_counts = {
            "Platelet": 0,
            "RBC": 0,
            "WBC": 0
        }

        detection_count = 0
        confidences = []

        # ==============================================
        # 8. GHI NHÃN YOLO
        # ==============================================

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as label_file:

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes.xywhn.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()

                for box, class_id, confidence in zip(
                    boxes, classes, scores
                ):

                    x_center, y_center, width, height = box
                    confidence = float(confidence)

                    # Ghi nhận tất cả dự đoán Platelet
                    if class_id == 0:

                        kept = confidence >= PLATELET_CONFIDENCE

                        platelet_review_rows.append({
                            "image": image_path.name,
                            "confidence": round(confidence, 4),
                            "x_center": round(float(x_center), 6),
                            "y_center": round(float(y_center), 6),
                            "width": round(float(width), 6),
                            "height": round(float(height), 6),
                            "kept": kept
                        })

                    # Lọc Platelet confidence thấp
                    if (
                        class_id == 0
                        and confidence < PLATELET_CONFIDENCE
                    ):
                        continue

                    # Lọc RBC và WBC
                    if (
                        class_id != 0
                        and confidence < OTHER_CONFIDENCE
                    ):
                        continue

                    # Ghi nhãn theo định dạng YOLO
                    label_file.write(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}\n"
                    )

                    class_name = EXPECTED_NAMES[class_id]

                    class_counts[class_name] += 1
                    detection_count += 1
                    confidences.append(confidence)

        # ==============================================
        # 9. LƯU ẢNH DỰ ĐOÁN
        # ==============================================

        preview = result.plot()

        prediction_path = (
            PREDICTION_OUTPUT / image_path.name
        )

        if not cv2.imwrite(str(prediction_path), preview):
            raise RuntimeError(
                f"Không thể lưu ảnh: {prediction_path}"
            )

        # ==============================================
        # 10. THỐNG KÊ TỪNG ẢNH
        # ==============================================

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        report_rows.append({
            "image": image_path.name,
            "detections": detection_count,
            "Platelet": class_counts["Platelet"],
            "RBC": class_counts["RBC"],
            "WBC": class_counts["WBC"],
            "average_confidence": round(avg_confidence, 4)
        })

        print(
            f"Detections: {detection_count} | "
            f"Platelet: {class_counts['Platelet']} | "
            f"RBC: {class_counts['RBC']} | "
            f"WBC: {class_counts['WBC']}"
        )

    # ==================================================
    # 11. XUẤT REPORT.CSV
    # ==================================================

    report_path = OUTPUT_DIR / "report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        fieldnames = [
            "image",
            "detections",
            "Platelet",
            "RBC",
            "WBC",
            "average_confidence"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # ==================================================
    # 12. XUẤT PLATELET_REVIEW.CSV
    # ==================================================

    platelet_report_path = (
        OUTPUT_DIR / "platelet_review.csv"
    )

    with open(
        platelet_report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        fieldnames = [
            "image",
            "confidence",
            "x_center",
            "y_center",
            "width",
            "height",
            "kept"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(platelet_review_rows)

    # ==================================================
    # 13. THÔNG BÁO HOÀN THÀNH
    # ==================================================

    print("\n========== HOÀN THÀNH ==========")
    print(f"Ảnh đã xử lý: {len(selected_images)}")
    print(f"Thư mục kết quả: {OUTPUT_DIR}")
    print(f"Báo cáo tổng hợp: {report_path}")
    print(f"Báo cáo Platelet: {platelet_report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
