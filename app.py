from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bccd_ui.config import (CLASS_NAMES, DISPLAY_NAMES, EVALUATION_DIR,
                            MODEL_PATH, SAMPLE_DIR)
from bccd_ui.exports import export_csv, export_json, export_png, export_yolo
from bccd_ui.inference import Detector, read_image
from bccd_ui.rendering import render_detections
from bccd_ui.results import count_cells


st.set_page_config(page_title="BCCD · Phân tích tế bào", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1440px; padding-top: 2rem;}
h1, h2, h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: white; border: 1px solid #dbe7e9;
border-radius: 12px; padding: 16px;}
.eyebrow {color: #0f766e; font-weight: 700; letter-spacing: .12em; font-size: .8rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_detector(model_path, modified_ns):
    return Detector(Path(model_path))


def show_evaluation():
    st.markdown('<div class="eyebrow">MODEL REPORT</div>', unsafe_allow_html=True)
    st.title("Hiệu năng model")
    st.write("Kết quả đánh giá đã lưu của YOLOv8n V3 trên tập test.")
    try:
        summary = json.loads((EVALUATION_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        fields = [("Precision", "precision"), ("Recall", "recall"),
                  ("mAP@50", "mAP50"), ("mAP@50–95", "mAP50_95")]
        for column, (label, key) in zip(st.columns(4), fields):
            column.metric(label, f"{float(summary[key]):.2%}")
        st.subheader("Kết quả theo loại tế bào")
        per_class = summary["per_class_mAP50_95"]
        table = pd.DataFrame([{"Loại tế bào": f"{DISPLAY_NAMES[cid]} ({name})",
                               "mAP@50–95": f"{float(per_class[name]):.2%}"}
                              for cid, name in CLASS_NAMES.items()])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Model: {summary['model']} · Dataset: {summary['dataset']} · Split: {summary['split']}")
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("Chưa đọc được báo cáo evaluation_summary.json hoặc báo cáo thiếu dữ liệu.")
    st.info("Đây là metrics từ báo cáo đã lưu, không phải độ chính xác của ảnh vừa tải lên. "
            "Cấu hình training lưu tên dataset_v2, trong khi evaluation dùng dataset_v3; "
            "nguồn dữ liệu huấn luyện checkpoint cần được xác minh thêm.")
    titles = {"confusion_matrix_normalized.png": "Ma trận nhầm lẫn chuẩn hóa",
              "BoxPR_curve.png": "Đường cong Precision–Recall",
              "BoxF1_curve.png": "F1 theo ngưỡng confidence"}
    for filename, title in titles.items():
        with st.expander(title, expanded=filename.startswith("confusion")):
            path = EVALUATION_DIR / filename
            if path.is_file():
                st.image(str(path), width="stretch")
            else:
                st.caption("Chưa có biểu đồ này trong thư mục đánh giá.")
    st.caption("Nguồn: runs/evaluation/yolov8n_v3_test/ · Không tự chạy lại evaluation khi mở trang.")


def show_analysis():
    st.markdown('<div class="eyebrow">BLOOD CELL DETECTION</div>', unsafe_allow_html=True)
    st.title("Khám phá từng tế bào")
    st.write("Tải ảnh hiển vi, nhận diện và đếm hồng cầu, bạch cầu, tiểu cầu trong một lần phân tích.")

    with st.sidebar:
        st.divider()
        st.subheader("Cấu hình nhận diện")
        threshold = st.slider("Ngưỡng confidence", 0.05, 0.95, 0.25, 0.05, key="confidence",
                              help="Ngưỡng cao giữ lại ít dự đoán hơn. Giá trị 0,25 là điểm bắt đầu thử nghiệm.")
        selected = st.multiselect("Loại tế bào cần hiển thị", list(CLASS_NAMES),
                                  default=list(CLASS_NAMES), key="classes",
                                  format_func=lambda cid: f"{DISPLAY_NAMES[cid]} ({CLASS_NAMES[cid]})")
        with st.expander("Nâng cao"):
            iou = st.slider("Ngưỡng IoU (NMS)", 0.10, 0.90, 0.70, 0.05,
                            help="Điều chỉnh bước loại bỏ các bounding box trùng nhau.")
        st.caption("YOLOv8n V3 · kích thước suy luận 640 px · CPU/GPU tự động")

    source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Dùng ảnh mẫu"], horizontal=True, key="source")
    data, image_name = None, None
    if source == "Tải ảnh lên":
        uploaded = st.file_uploader("Chọn ảnh hiển vi", type=["jpg", "jpeg", "png"],
                                    help="Tối đa 10 MB và 20 triệu pixel. Ảnh xử lý tại máy chạy ứng dụng.")
        if uploaded is not None:
            data, image_name = uploaded.getvalue(), uploaded.name
    else:
        samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:8]
        if samples:
            sample = st.selectbox("Chọn ảnh mẫu từ tập test", samples,
                                  format_func=lambda p: p.name)
            try:
                data, image_name = sample.read_bytes(), sample.name
            except OSError:
                st.error("Không đọc được ảnh mẫu. Hãy chọn ảnh khác.")
        else:
            st.info("Chưa có ảnh mẫu. Bạn có thể chuyển sang tải ảnh lên.")

    if data is None:
        st.info("Chọn ảnh hoặc dùng ảnh mẫu để bắt đầu phân tích.")
        st.caption("Tiểu cầu · Platelet / Hồng cầu · RBC / Bạch cầu · WBC")
        return
    try:
        image = read_image(data)
    except ValueError as exc:
        st.error(str(exc))
        return

    signature = hashlib.sha256(data).hexdigest(), threshold, iou, tuple(sorted(selected))
    signature += (MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else None,)
    stored = st.session_state.get("analysis")
    current = stored if stored and stored["signature"] == signature else None
    if stored and current is None:
        st.info("Ảnh hoặc cấu hình đã thay đổi. Bấm Phân tích ảnh để cập nhật kết quả.")

    if not selected:
        st.warning("Chọn ít nhất một loại tế bào để phân tích.")
    if st.button("Phân tích ảnh", type="primary", disabled=not selected, icon="🔬"):
        try:
            with st.spinner("Đang nạp model và phân tích ảnh…"):
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError("Thiếu best.pt tại runs/bccd/yolov8n_v3/weights/best.pt.")
                detector = get_detector(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
                detections, elapsed_ms = detector.predict(image, threshold, iou, selected)
                metadata = {"model": "YOLOv8n V3", "checkpoint": "runs/bccd/yolov8n_v3/weights/best.pt",
                            "image_name": image_name, "width": image.width, "height": image.height,
                            "confidence_threshold": threshold, "iou_threshold": iou,
                            "class_ids": sorted(selected), "image_size": 640, "max_detections": 1000,
                            "device": detector.device, "elapsed_ms": elapsed_ms,
                            "created_at": datetime.now(timezone.utc).isoformat()}
                current = {"signature": signature, "detections": detections, "metadata": metadata,
                           "preview": render_detections(image, detections)}
                st.session_state["analysis"] = current
        except Exception as exc:
            st.error(f"Không thể phân tích ảnh: {exc}")
            return

    original, annotated = st.columns(2)
    with original:
        st.subheader("Ảnh gốc")
        st.image(image, width="stretch")
        st.caption(f"{image_name} · {image.width} × {image.height} px")
    with annotated:
        st.subheader("Kết quả nhận diện")
        if current:
            st.image(current["preview"], width="stretch")
            metadata = current["metadata"]
            device = "CPU" if metadata["device"] == "cpu" else "GPU CUDA"
            st.caption(f"{device} · {metadata['elapsed_ms']:,.0f} ms · confidence ≥ {threshold:.0%}")
        else:
            st.info("Kết quả có bounding box sẽ xuất hiện sau khi phân tích.")
    if current is None:
        return

    detections = current["detections"]
    counts = count_cells(detections)
    for column, (cid, name) in zip(st.columns(3), CLASS_NAMES.items()):
        column.metric(f"{DISPLAY_NAMES[cid]} · {name}", counts[name])
    st.caption(f"Tổng: {len(detections)} tế bào thuộc các lớp đã chọn. Bounding box, số đếm và file xuất dùng cùng kết quả.")
    if not detections:
        st.warning("Không có tế bào thỏa điều kiện. Bạn có thể giảm ngưỡng confidence hoặc chọn ảnh khác.")
    if len(detections) >= 1000:
        st.warning("Đã đạt giới hạn 1.000 detection; ảnh có thể còn đối tượng chưa được hiển thị.")

    with st.expander("Chi tiết từng detection", expanded=True):
        records = [{"Loại": CLASS_NAMES[d.class_id], "Confidence": d.confidence,
                    **dict(zip(["x1", "y1", "x2", "y2"], d.bbox_xyxy))} for d in detections]
        st.dataframe(pd.DataFrame(records, columns=["Loại", "Confidence", "x1", "y1", "x2", "y2"]),
                     hide_index=True, width="stretch",
                     column_config={"Confidence": st.column_config.NumberColumn(format="%.3f")})
    st.subheader("Tải kết quả")
    stem = Path(image_name).stem
    downloads = [("Ảnh PNG", export_png(current["preview"]), f"{stem}_detected.png", "image/png"),
                 ("Bảng CSV", export_csv(detections), f"{stem}_detections.csv", "text/csv"),
                 ("Báo cáo JSON", export_json(detections, current["metadata"]), f"{stem}_report.json", "application/json"),
                 ("Nhãn YOLO", export_yolo(detections), f"{stem}.txt", "text/plain")]
    for column, (label, content, filename, mime) in zip(st.columns(4), downloads):
        column.download_button(label, content, file_name=filename, mime=mime, width="stretch", on_click="ignore")


with st.sidebar:
    st.title("🔬 BCCD")
    st.caption("Blood Cell Detection")
    page = st.radio("Điều hướng", ["Phân tích ảnh", "Hiệu năng model"], label_visibility="collapsed", key="page")
    st.caption("Không lưu ảnh tải lên vào dataset. Kết quả được giữ trong phiên làm việc.")

if page == "Phân tích ảnh":
    show_analysis()
else:
    show_evaluation()
