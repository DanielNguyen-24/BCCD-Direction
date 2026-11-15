# BCCD · Blood Cell Detection

Ứng dụng local nhận diện **tiểu cầu (Platelet), hồng cầu (RBC), bạch cầu (WBC)** trong ảnh hiển vi bằng YOLOv8n V3.

## Chạy giao diện

Mở terminal tại thư mục gốc repo. Môi trường đã kiểm tra: Python 3.13.16, inference CPU trên Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Nếu môi trường hiện tại đã có dependency:

```powershell
python -m streamlit run app.py
```

Truy cập **http://127.0.0.1:8501**. Dừng bằng `Ctrl+C` trong terminal. Ứng dụng chỉ lắng nghe trên máy local theo `.streamlit/config.toml`.

Các phiên bản trong `requirements-ui.txt` là phiên bản đã có trên máy phát triển. Nếu cài mới trên nền tảng khác, cần kiểm chứng lại khả năng cung cấp wheel và tương thích PyTorch/Ultralytics. CUDA không bắt buộc; ứng dụng tự chọn CPU khi CUDA không khả dụng.

## Sử dụng

1. Chọn **Tải ảnh lên** hoặc **Dùng ảnh mẫu**. Chấp nhận nội dung JPEG/PNG, tối đa 10 MB, 20 triệu pixel. Có tối đa 8 ảnh mẫu từ `dataset_v3/test/images/`.
2. Chọn confidence và loại tế bào ở thanh bên. Ngưỡng 0,25 là mặc định thử nghiệm, chưa được tối ưu riêng trên validation. IoU nằm trong phần Nâng cao.
3. Bấm **Phân tích ảnh**. Xem ảnh gốc/kết quả, số lượng theo lớp và tọa độ pixel từng detection. Bấm phân tích lại sau khi đổi ảnh hoặc cấu hình.
4. Tải ảnh PNG, bảng CSV, báo cáo JSON hoặc nhãn YOLO TXT. Tất cả dùng cùng danh sách detection sau lọc. YOLO TXT có 5 cột: `class_id x_center y_center width height`, tọa độ chuẩn hóa 0–1, không có confidence.
5. Mở **Hiệu năng model** để xem metrics và biểu đồ evaluation đã lưu. Đây là kết quả tập test, không phải confidence của ảnh đang phân tích.

Ảnh được xử lý tại máy chạy ứng dụng, không ghi vào dataset. Kết quả giữ trong phiên, chưa có lịch sử lâu dài. Model được cache giữa các lần tương tác và khóa khi suy luận để tránh gọi đồng thời trên cùng object. Lần chạy đầu bao gồm thời gian nạp model; thời gian hiển thị đo lời gọi predict, có thể gồm khởi tạo/warm-up và chờ khóa.

## Tệp cần có

- `runs/bccd/yolov8n_v3/weights/best.pt`: checkpoint nhận diện BCCD. Mapping bắt buộc: **0=Platelet, 1=RBC, 2=WBC**. Không thay bằng `yolov8n.pt` gốc vì model đó được dùng để khởi tạo training.
- `dataset_v3/test/images/`: ảnh mẫu tùy chọn; thiếu vẫn tải ảnh riêng được.
- `runs/evaluation/yolov8n_v3_test/`: JSON và biểu đồ; thiếu có thông báo trong UI.

Các đường dẫn trong ứng dụng được tính từ thư mục repo. Pipeline training cũ vẫn nằm trong `scripts/`; UI không chạy các script này. Cấu hình training đã lưu ghi dataset_v2 trong khi evaluation dùng dataset_v3; nguồn training của checkpoint cần xác minh thêm. Xem [báo cáo khảo sát và kế hoạch](docs/UI_PLAN.md).

## Kiểm tra

```powershell
python -X utf8 -m unittest discover -s tests -v
```

Bao gồm kiểm tra lọc box, đếm, export tọa độ YOLO, kết quả rỗng, ảnh lỗi, thiếu model và luồng Streamlit qua `AppTest`. Bài kiểm tra UI dùng checkpoint và một ảnh mẫu thật; cần có hai artifact đó. Không chạy lại training hay evaluation.

## Cấu trúc phần UI

```text
app.py                  Giao diện Streamlit và trạng thái phiên
bccd_ui/config.py       Đường dẫn, giới hạn ảnh, mapping lớp
bccd_ui/inference.py    Đọc ảnh, nạp model, chọn CPU/CUDA, predict
bccd_ui/results.py      Chuẩn hóa, lọc và đếm detection
bccd_ui/rendering.py    Vẽ box từ kết quả sau lọc
bccd_ui/exports.py      PNG, CSV, JSON và YOLO TXT
tests/                  Kiểm tra service và luồng UI
```

Bản đầu hỗ trợ một ảnh mỗi lần và xem báo cáo model. Thống kê dataset, batch inference, biên tập nhãn và training từ UI là các bước mở rộng sau.
