# Báo cáo code và kế hoạch UI BCCD

Ngày khảo sát: 15/11/2026. Nhánh làm việc: `UI`.

## 1. Phạm vi khảo sát

Đã đọc toàn bộ mã nguồn trong `scripts/`, file `check_total_of_cell.py`, ba file cấu hình dataset, cấu hình training và báo cáo evaluation được lưu trong `runs/`. Đã kiểm đếm ảnh, file nhãn và số bounding box của dataset V3; kiểm tra sự tồn tại của trọng số và phiên bản package bằng metadata.

Chưa chạy training, evaluation hoặc inference thực tế. Các chỉ số bên dưới lấy từ artifact đã lưu, chưa được tái xác nhận trên máy hiện tại. File báo cáo Word không thuộc phạm vi đọc code này.

## 2. Hiện trạng dự án

Đây là pipeline Python nhận diện tế bào máu bằng YOLOv8n, có ba lớp theo thứ tự:

| ID | Tên model | Tên hiển thị đề xuất |
| --- | --- | --- |
| 0 | Platelet | Tiểu cầu |
| 1 | RBC | Hồng cầu |
| 2 | WBC | Bạch cầu |

Luồng hiện có: ảnh gốc → lấy mẫu để gán nhãn → huấn luyện model ban đầu → tự gán nhãn dữ liệu → dataset V3 → huấn luyện → đánh giá.

| File/thư mục | Vai trò và nhận xét |
| --- | --- |
| `scripts/01_sample_images.py` | Lấy ngẫu nhiên 150 ảnh, seed 42; sao chép ảnh và ghi báo cáo nguồn. Có xử lý ngay ở cấp module. |
| `scripts/02_train_initial.py` | Train YOLOv8n trên initial_dataset; yêu cầu CUDA. |
| `scripts/03_evaluate.py` | Đánh giá model ban đầu; đường dẫn trọng số tuyệt đối theo máy cũ. |
| `scripts/04_auto_label.py` | Predict hàng loạt, lọc confidence riêng cho Platelet, xuất nhãn YOLO, preview và CSV. |
| `scripts/05_train_initial_v2.py` | Train trên initial_dataset_v2, cấu trúc tương tự bản đầu. |
| `scripts/06_train_v3` | File không có đuôi `.py`; nội dung chứa hai khối train, một khối dùng dataset_v2 và một khối dùng dataset_v3. |
| `scripts/07_evaluate_v3.py` | Đánh giá tập test bằng best.pt và xuất JSON tổng hợp. |
| `scripts/check_gpu.py` | In thông tin PyTorch và CUDA. |
| `check_total_of_cell.py` | Đếm số nhãn từng loại theo split. Có xử lý ngay ở cấp module. |
| `runs/bccd/yolov8n_v3/` | best.pt, last.pt, cấu hình, CSV huấn luyện và biểu đồ. |
| `runs/evaluation/yolov8n_v3_test/` | Metrics JSON, confusion matrix, đường cong và ảnh minh họa. |

Chưa thấy UI, API, README, khai báo dependency, cấu hình ứng dụng hoặc bộ test trong repo. Logic hiện được viết chủ yếu thành các chương trình độc lập, chưa có service inference dùng chung.

### Dữ liệu kiểm đếm trực tiếp

| Dataset | Train | Valid | Test | Tổng ảnh |
| --- | ---: | ---: | ---: | ---: |
| initial_dataset | 105 | 23 | 22 | 150 |
| initial_dataset_v2 | 105 | 23 | 22 | 150 |
| dataset_v3 | 1.575 | 338 | 337 | 2.250 |

Mỗi split có số file nhãn `.txt` bằng số ảnh. Kiểm đếm này chưa kiểm tra nội dung từng ảnh, trùng lặp giữa các split hoặc chất lượng nhãn.

| Split V3 | Tiểu cầu | Hồng cầu | Bạch cầu | Tổng bounding box |
| --- | ---: | ---: | ---: | ---: |
| Train | 1.219 | 28.190 | 827 | 30.236 |
| Valid | 240 | 6.104 | 188 | 6.532 |
| Test | 262 | 6.138 | 183 | 6.583 |
| Tổng | 1.721 | 40.432 | 1.198 | 43.351 |

### Kết quả đã lưu

Nguồn: `runs/evaluation/yolov8n_v3_test/evaluation_summary.json`.

| Chỉ số | Giá trị |
| --- | ---: |
| Precision | 96,52% |
| Recall | 91,51% |
| mAP@50 | 97,39% |
| mAP@50–95 | 90,40% |
| mAP@50–95 Platelet | 85,45% |
| mAP@50–95 RBC | 91,49% |
| mAP@50–95 WBC | 94,26% |

Tiểu cầu có mAP@50–95 thấp nhất trong ba lớp theo báo cáo này. UI cần cho phép xem riêng loại tế bào và confidence để hỗ trợ kiểm tra kết quả.

Có `runs/bccd/yolov8n_v3/weights/best.pt` và `last.pt`, mỗi file khoảng 6,51 MB. `yolov8n.pt` tại thư mục gốc được code dùng để khởi tạo training; UI nhận diện BCCD phải trỏ đến trọng số đã huấn luyện `best.pt`.

Môi trường kiểm tra: Python 3.13.16; metadata ghi nhận ultralytics 8.4.208, torch 2.15.0, streamlit 1.67.0; chưa thấy fastapi. Việc có package không xác nhận import thành công, tương thích checkpoint hoặc CUDA hoạt động.

## 3. Các điểm cần xử lý khi tích hợp

1. **Đường dẫn phụ thuộc máy cũ:** scripts 03 và 04 dùng đường dẫn `C:\Users\DanielNguyen\...`. Cần một cấu hình đường dẫn dựa trên thư mục gốc repo và thông báo rõ khi thiếu model.
2. **Script train V3 bị ghép hai chương trình:** `06_train_v3` có hai khối `main()` và hai entry point. Nếu chạy trực tiếp, khối đầu có thể lỗi vì không có dataset_v2; nếu khối đầu chạy xong thì còn có thể tiếp tục train khối sau. Cần tách và chuẩn hóa trước khi dùng lại luồng này.
3. **Tên dataset trong artifact chưa thống nhất:** `runs/bccd/yolov8n_v3/args.yaml` ghi dataset_v2 trên máy cũ, còn script evaluation dùng dataset_v3. Chưa đủ bằng chứng để kết luận nguồn dữ liệu training của checkpoint. UI phải hiển thị đây là kết quả đã lưu và ghi rõ nguồn, không tự khẳng định toàn bộ provenance đã được xác minh.
4. **Preview và nhãn auto-label có thể lệch:** code lọc Platelet dưới 0,70 khỏi nhãn và thống kê, nhưng gọi `result.plot()` trên kết quả chưa lọc. Khi tích hợp phải dùng cùng một tập detection sau lọc cho ảnh, số đếm, bảng và file xuất.
5. **Phụ thuộc GPU:** nhiều script yêu cầu CUDA hoặc cố định device=0. Inference UI nên tự chọn CUDA khi khả dụng và hỗ trợ CPU; kiểm tra inference thực tế trước khi hứa về tốc độ.
6. **Cấu hình dataset:** YAML dùng đường dẫn `../train/images`, `../valid/images`, `../test/images` và không khai báo `path`. Cần xác minh cách phiên bản Ultralytics hiện tại phân giải các đường dẫn trước khi chạy train/evaluation; chưa kết luận cấu hình này chạy đúng tại workspace mới.
7. **Khả năng tái lập:** bổ sung dependency đã kiểm chứng, hướng dẫn chạy và cấu hình. Tránh import script có side effect từ UI.
8. **Ngưỡng confidence:** 0,10/0,70 là cấu hình auto-label hiện có; conf=0,001 trong evaluation phục vụ tính metrics. Không dùng các giá trị này làm mặc định UI một cách máy móc; cần chọn ngưỡng suy luận dựa trên tập validation.

## 4. Hướng UI đề xuất

Giả định mục tiêu bản đầu là ứng dụng local phục vụ demo/đồ án, một người thao tác chính, giao diện tiếng Việt. Nếu mục tiêu chuyển thành dịch vụ nhiều người dùng hoặc thiết kế frontend tùy biến sâu, cần đánh giá lại kiến trúc.

Đề xuất **Streamlit + service Python dùng Ultralytics**, vì repo và môi trường hiện đều dựa trên Python. Streamlit có sẵn upload, download và cache tài nguyên model; giảm phần tích hợp cần làm cho bản demo. Đây là lựa chọn đề xuất, chưa triển khai.

Tài liệu chính thức đã đối chiếu:

- Upload: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Download: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- Cache model: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource

### Màn hình và bố cục

| Màn hình | Nội dung | Ưu tiên |
| --- | --- | --- |
| Phân tích ảnh | Tải ảnh/chọn ảnh mẫu, cấu hình, ảnh gốc và ảnh kết quả, số đếm, bảng detection, tải kết quả | MVP |
| Hiệu năng model | Chỉ số JSON đã lưu, kết quả theo lớp, confusion matrix và đường cong; ghi rõ nguồn và split test | MVP |
| Tổng quan dữ liệu | Số ảnh theo split, phân bố nhãn ba loại, vài ảnh minh họa | Sau MVP |
| Phân tích hàng loạt | Nhiều ảnh, tiến độ, tổng hợp CSV và ZIP kết quả | Sau MVP |

Màn hình Phân tích ảnh: thanh bên có điều hướng và phần cấu hình thu gọn; vùng chính có tải ảnh, nút Phân tích, hai cột ảnh gốc/kết quả, ba thẻ số đếm, bảng kết quả và nút tải xuống. Màn hình hẹp chuyển thành một cột. Dùng nền sáng, điểm nhấn xanh teal; màu bounding box cố định theo lớp và luôn có tên lớp đi kèm.

Luồng chính: chọn ảnh → xem ảnh gốc → chọn ngưỡng/lớp cần xem → Phân tích → so sánh ảnh → xem số đếm và detection → tải PNG/CSV/JSON hoặc nhãn YOLO.

Các trạng thái cần có: chưa chọn ảnh; đang nạp model; đang phân tích; thành công; không có detection; ảnh không đọc được; vượt giới hạn file/pixel; thiếu checkpoint; model sai mapping lớp; lỗi suy luận. Khi thay ảnh hoặc tham số, đánh dấu kết quả cũ và yêu cầu phân tích lại, tránh trình bày kết quả cũ như kết quả hiện tại.

## 5. Kiến trúc dự kiến

```text
app.py                       # Entry point Streamlit
bccd_ui/
  config.py                  # Đường dẫn và cấu hình
  inference.py               # Nạp model, chọn device, predict
  results.py                 # Chuẩn hóa và lọc detection, đếm theo lớp
  rendering.py               # Vẽ box, nhãn và confidence
  exports.py                 # CSV, JSON, PNG, YOLO TXT
  dataset.py                 # Thống kê dataset theo yêu cầu
  views/
    analysis.py
    evaluation.py
tests/                       # Kiểm tra logic quan trọng
requirements-ui.txt
README.md
```

UI gọi service inference, không gọi các script training/evaluation làm side effect. Cache model để tránh nạp lại theo mỗi thao tác; quản lý kết quả theo phiên. Nếu model dùng chung giữa phiên, cần kiểm soát truy cập predict phù hợp thay vì mặc định object model an toàn khi chạy đồng thời.

Detection chuẩn hóa gồm class_id, class_name, confidence, bbox_xyxy theo pixel và bbox_xywhn chuẩn hóa. Cùng danh sách sau lọc cấp dữ liệu cho renderer, số đếm, bảng và exports. JSON xuất kèm tên model, tham số, kích thước ảnh và thời gian xử lý; CSV UTF-8 tương thích tiếng Việt; YOLO TXT giữ đúng mapping 0/1/2.

MVP xử lý trong phiên và xuất theo yêu cầu; lưu lịch sử lâu dài chỉ bổ sung khi có nhu cầu cụ thể. Training, biên tập nhãn bằng tay, đăng nhập và triển khai public nằm ngoài MVP.

## 6. Kế hoạch triển khai và nghiệm thu

| Bước | Công việc | Điều kiện hoàn thành |
| --- | --- | --- |
| 1. Kiểm chứng nền tảng | Import package, nạp best.pt, xác minh model.names, chạy một ảnh bằng CPU/CUDA sẵn có; chuẩn hóa cấu hình và dependency | Một ảnh trả được detection đúng schema; thiếu model có lỗi rõ ràng; ghi nhận thiết bị và thời gian thực tế |
| 2. Service nhận diện | Tách predict, lọc theo lớp/ngưỡng, đếm, render và exports thành hàm dùng chung | Ảnh, bảng, tổng số và file xuất dùng cùng tập box; tọa độ được kiểm tra hợp lệ |
| 3. UI phân tích ảnh | Dựng bố cục, ảnh mẫu, upload JPG/PNG, nút chạy, trạng thái và downloads | Hoàn thành luồng từ upload đến tải kết quả trên màn hình rộng và hẹp; thay input không gây kết quả cũ bị hiểu nhầm |
| 4. Trang hiệu năng | Đọc JSON, hiển thị metrics theo lớp và các biểu đồ có sẵn | Số liệu đúng artifact; tách rõ metrics tập test với confidence của từng detection; thiếu artifact không làm sập app |
| 5. Hoàn thiện | Kiểm tra luồng lỗi, đối chiếu exports, viết README và lệnh chạy | Người khác cài dependency và chạy được bằng hướng dẫn; nhánh UI có thay đổi dễ review |
| 6. Mở rộng | Thống kê dataset, xử lý nhiều ảnh, ngưỡng riêng từng lớp nếu cần | Có tiến độ, giới hạn tài nguyên và báo lỗi từng ảnh; không ghi đè dữ liệu gốc |

Thứ tự nên làm: kiểm chứng best.pt → service inference → màn hình Phân tích ảnh → trang Hiệu năng → thống kê và batch. Bước 1 là điểm kiểm chứng quan trọng nhất vì chưa thực thi checkpoint trên máy này.

Kiểm thử tập trung vào mapping lớp, confidence tại biên ngưỡng, trường hợp không có box, tính nhất quán số đếm/ảnh/export, chuyển tọa độ YOLO và xử lý ảnh lỗi. Smoke test một ảnh thật; không yêu cầu train lại model để nghiệm thu UI. Kiểm tra cấu hình dataset và script train là công việc riêng nếu mở rộng sang quản lý training.

## 7. Trạng thái tại thời điểm khảo sát

Chỉ bổ sung tài liệu này. Chưa sửa pipeline hiện có, cài package, xây UI hoặc chạy lại model. File Word chưa được theo dõi trong Git đã tồn tại trước khảo sát và được giữ nguyên.

## 8. Bản UI đầu tiên đã triển khai

Theo yêu cầu triển khai tiếp theo, đã bổ sung ứng dụng Streamlit local trong `app.py` và package `bccd_ui/`:

- Tải ảnh JPEG/PNG hoặc chọn ảnh mẫu, giới hạn 10 MB và 20 triệu pixel.
- Chỉnh confidence, IoU và các lớp cần hiển thị; bấm phân tích để chạy model.
- Xem ảnh gốc/kết quả, số đếm và bảng detection. Thay input sẽ ẩn kết quả cũ cho đến khi phân tích lại.
- Xuất PNG, CSV, JSON và YOLO TXT từ cùng một danh sách box sau lọc.
- Xem metrics và biểu đồ evaluation đã lưu; có xử lý artifact bị thiếu.
- Cache model và dùng khóa cho predict; tự chọn CPU/CUDA. Không tự cài plugin khi giải mã ảnh lỗi.
- Có theme, cấu hình local-only, requirements và README hướng dẫn sử dụng.

Kiểm chứng tại máy hiện tại: checkpoint có mapping đúng 0=Platelet, 1=RBC, 2=WBC; CUDA không khả dụng, inference CPU chạy thành công. Bộ kiểm tra gồm 7 test đã đạt, trong đó AppTest chạy nhận diện ảnh mẫu bằng model thật, kiểm tra kết quả cũ bị ẩn khi đổi ngưỡng và trạng thái khi thiếu báo cáo. Chưa chạy lại evaluation/training, chưa kiểm chứng cài mới dependency trong môi trường sạch hoặc kiểm tra trực quan trên trình duyệt. Các script training/auto-label cũ chưa thay đổi; các vấn đề mục 3 còn áp dụng cho chúng.

Cấu trúc triển khai gọn hơn dự kiến: hai trang nằm trong `app.py`; service đã được tách riêng. Các phần dataset dashboard và batch vẫn ở giai đoạn sau MVP.

Server local đã khởi động thành công tại `http://127.0.0.1:8501`, trang chính trả HTTP 200 và endpoint `/_stcore/health` trả `ok`. Đã dừng server sau kiểm tra; chạy lại bằng lệnh trong README khi sử dụng.
