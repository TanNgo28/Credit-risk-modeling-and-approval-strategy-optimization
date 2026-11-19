# Decision Science: tài liệu dự án

Bộ tài liệu này mô tả hai bài toán (Challenge 1 và Challenge 2) và thư viện `pydo` đã dùng để giải Challenge 2.
Mỗi tài liệu đứng độc lập và có thể đọc riêng.

| Tài liệu | Nội dung | Đọc khi |
|---|---|---|
| [challenge_1_pd_model.md](challenge_1_pd_model.md) | Mô hình xác suất vỡ nợ (PD): kiểm tra dữ liệu, độ ổn định theo thời gian, mô hình, đánh giá, bàn giao | Cần hiểu hoặc bảo vệ mô hình PD |
| [challenge_2_portfolio_optimization.md](challenge_2_portfolio_optimization.md) | Chính sách duyệt/từ chối tối ưu: kinh tế từng khoản vay, 7 ràng buộc, chẩn đoán, chưng cất cây quyết định, tập kiểm thử, khuyến nghị | Cần hiểu hoặc bảo vệ chính sách |
| [pydo.md](pydo.md) | Tài liệu thư viện `pydo`: khái niệm, ngôn ngữ luật, công thức toán, API, kiểm thử, giới hạn | Cần dùng, mở rộng hoặc kiểm tra công cụ |

## Thứ tự và phụ thuộc

```
data/training_data.csv.gz
        │
        ▼
notebooks/01_pd_model.ipynb  ──►  results/pd_predictions.csv  ──►  notebooks/02_portfolio_optimization.ipynb (dùng src/pydo)
                                                                          │
                          notebooks/functions/*.py, results/models/*, results/optimal_policy.csv, results/constrained_output.csv
```

Notebook 02 chỉ chạy được sau notebook 01, vì nó đọc `results/pd_predictions.csv`.

## Cấu trúc thư mục

| Đường dẫn | Vai trò |
|---|---|
| `notebooks/01_pd_model.ipynb` | Notebook Challenge 1 (đã chạy, có output) |
| `notebooks/02_portfolio_optimization.ipynb` | Notebook Challenge 2 (đã chạy, có output) |
| `notebooks/functions/flatten.py`, `economics.py` | Hàm kinh tế của từng khoản vay (do notebook 02 ghi ra) |
| `src/pydo/` | Thư viện tối ưu quyết định PyDO |
| `tests/` | 43 test của `pydo` |
| `data/training_data.csv.gz` | Dữ liệu gốc: 200.000 đơn, 61 cột (không kèm repo, xem `data/README.md`) |
| `results/pd_predictions.csv` | Kết quả Challenge 1: `ID_APPLICATION, PD` cho 17.976 đơn tháng 5/2015 (không đưa lên git) |
| `results/optimal_policy.csv` | Kết quả Challenge 2: 32 luật của chính sách cuối cùng và khối KPI, trạng thái từng ràng buộc |
| `results/constrained_output.csv` | Kết quả Challenge 2: quyết định của solver cho từng đơn (không đưa lên git) |
| `results/models/clx_approval_tree.json`, `.sql` | Chính sách cuối cùng dưới dạng luật (JSON và biểu thức SQL CASE) |
| `docs/` | Thư mục này |

## Dữ liệu

Repo **không kèm dữ liệu đầu vào** (`data/training_data.csv.gz`) và các file kết quả theo từng đơn (`results/pd_predictions.csv`, `results/constrained_output.csv`). Muốn chạy lại, đặt file dữ liệu vào `data/` rồi chạy notebook 01 trước, notebook 02 sau; hai file kết quả sẽ được tạo lại.

## Môi trường

- Kernel `pydo-project` dùng môi trường ảo `.venv` của dự án. Cài bằng `pip install -r requirements.txt` (cài luôn `pydo` ở chế độ chỉnh sửa được):
  numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, lightgbm 4.7.0, shap 0.51.0, scipy 1.17.1, matplotlib 3.11.2.
- Notebook được chạy không giao diện bằng `nbclient` (kernel `pydo-project`, thư mục làm việc là `notebooks/`, vì đường dẫn trong notebook tính từ đó), nên output đã nằm sẵn trong file.
- Thời gian chạy tham khảo: notebook 01 khoảng 5 đến 6 phút, notebook 02 khoảng 10 đến 11 phút trên toàn bộ danh mục.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Chạy bằng Python của môi trường ảo ở trên. 43 test, khoảng 6 giây. Chi tiết ở phần "Kiểm thử" của [pydo.md](pydo.md).

## Lưu ý về PyDO

PyDO (Pythonic Decision Optimizer, gói Python `pydo`) là một thư viện nhỏ, tự chứa, được xây trong dự án này. Nó dùng các khái niệm phổ biến của tối ưu quyết định (danh mục, biến quyết định, giả định, hàm kinh tế, ràng buộc, kịch bản) và không phụ thuộc vào công cụ thương mại hay nội bộ nào.

## Quy ước chung của các tài liệu

- Tiền tệ là VND, "bn" là tỷ (10^9), "m" là triệu (10^6).
- "Kỳ vọng" là giá trị tính theo PD, không dùng kết quả thực tế. Chỉ phần tập kiểm thử (Challenge 2, mục 9) mới dùng `TARGET`.
- Mọi con số trong tài liệu lấy từ output của notebook trong thư mục này. Khi có suy luận chưa được kiểm chứng bằng phép thử riêng, tài liệu ghi rõ.
