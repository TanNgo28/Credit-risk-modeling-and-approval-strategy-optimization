# Challenge 1: mô hình PD (Probability of Default)

Notebook: `notebooks/01_pd_model.ipynb`. Kết quả bàn giao: `results/pd_predictions.csv`.

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Bài toán và luật chơi](#2-bài-toán-và-luật-chơi)
3. [Dữ liệu và cách chia](#3-dữ-liệu-và-cách-chia)
4. [Kiểm tra dữ liệu](#4-kiểm-tra-dữ-liệu)
5. [Độ ổn định theo thời gian](#5-độ-ổn-định-theo-thời-gian)
6. [Xử lý dữ liệu và feature engineering](#6-xử-lý-dữ-liệu-và-feature-engineering)
7. [Các mô hình](#7-các-mô-hình)
8. [Đánh giá](#8-đánh-giá)
9. [Chọn mô hình](#9-chọn-mô-hình)
10. [Giải thích mô hình](#10-giải-thích-mô-hình)
11. [Bàn giao `pd_predictions.csv`](#11-bàn-giao-pd_predictionscsv)
12. [Nhật ký quyết định](#12-nhật-ký-quyết-định)
13. [Hạn chế và việc tiếp theo](#13-hạn-chế-và-việc-tiếp-theo)
14. [Tái lập kết quả](#14-tái-lập-kết-quả)

---

## 1. Tóm tắt

- Mô hình được chọn: **LightGBM**, hiệu chuẩn bằng Platt scaling trên tập validation.
- Trên holdout (tháng 5/2015, chấm điểm một lần): **Gini 0,701, AUC 0,851, KS 0,534**.
- PD trung bình 0,3605 so với bad rate thực tế 0,3629.
- Phát hiện quan trọng nhất không nằm ở thuật toán mà ở dữ liệu: ba feature có vấn đề theo thời gian, trong đó `CATEGORICAL_9` **đảo chiều quan hệ với target từ 19/04/2015** nhưng phân phối không đổi, nên PSI không phát hiện được. Một phép chia ngẫu nhiên cũng không phát hiện được.
- Hai feature bị loại (`CATEGORICAL_9`, `CATEGORICAL_7`), một feature được xử lý (`NUMERICAL_10`).
- Có một tương tác ẩn (`NUMERICAL_32` × `CATEGORICAL_10`) mà mô hình tuyến tính không thấy; thêm một feature tương tác đưa logistic regression lên sát LightGBM.

| | Gini valid | Gini holdout | KS holdout | Gap overfit | Gap drift |
|---|---|---|---|---|---|
| A: Logistic regression (WOE) | 0,602 | 0,625 | 0,467 | 0,006 | −0,024 |
| A+: A cộng 1 feature tương tác | 0,671 | 0,694 | 0,527 | 0,005 | −0,023 |
| **B: LightGBM (được chọn)** | **0,683** | **0,701** | **0,534** | **0,043** | **−0,018** |

Gap drift âm của cả ba mô hình **không phải dấu hiệu của điều gì bất thường**: nó là hiệu ứng cơ cấu của tập validation (mục 8.4). Trên các dòng validation có `NUMERICAL_10` thật, Gini LightGBM là 0,701, bằng đúng Gini holdout.

## 2. Bài toán và luật chơi

Ước lượng xác suất vỡ nợ (`TARGET = 1`) cho từng đơn vay CLX, rồi giao điểm số cho Challenge 2.

Luật của đề (đều được tuân thủ):

1. Holdout là tháng 5/2015. Không dùng cho huấn luyện, chọn feature, thống kê encoding hay chỉnh tham số.
2. Phải huấn luyện ít nhất hai thuật toán, một cái đủ đơn giản để giải thích cho người không chuyên.
3. Không dùng 5 cột hợp đồng làm feature (`AMOUNT_VND`, `TENOR_MONTHS`, `INTEREST_RATE`, `MONTHLY_INSTALLMENT`, `PRODUCT_GROUP`): chúng mang quyết định của chính sách cũ.
4. Sản phẩm: notebook chạy được từ đầu đến cuối, và `pd_predictions.csv` cho mọi đơn tháng 5.

Cách tuân thủ luật 1: mọi quyết định về feature (giữ, loại, sửa) chỉ dùng dữ liệu development (trước 2015-05). Dữ liệu tháng 5 chỉ được dùng để tính PSI (không cần target) và chấm điểm cuối cùng. Phần kiểm tra bad rate theo tháng chỉ để mô tả.

## 3. Dữ liệu và cách chia

`data/training_data.csv.gz`: 200.000 dòng, 61 cột.

| Nhóm | Cột |
|---|---|
| Khóa, thời gian | `ID_APPLICATION` (không trùng), `TIME` (20/06/2014 đến 31/05/2015) |
| Feature số | `NUMERICAL_0` đến `NUMERICAL_41` (42 cột) |
| Feature phân loại | `CATEGORICAL_0` đến `CATEGORICAL_10` (11 cột) |
| Target | `TARGET` |
| Cột hợp đồng (bị cấm ở đây, dùng ở Challenge 2) | `AMOUNT_VND`, `TENOR_MONTHS`, `INTEREST_RATE`, `MONTHLY_INSTALLMENT`, `PRODUCT_GROUP` |

Bad rate tổng thể 35,9%, ổn định khoảng 36% mỗi tháng. Mỗi tháng có khoảng 18.000 đơn (tháng 6/2014 chỉ có từ ngày 20, tháng 2/2015 ngắn).

**Cách chia (bất đối xứng có chủ ý):**

| Phần | Kỳ | Cách chia | Số dòng | Bad rate | Dùng để |
|---|---|---|---|---|---|
| Train | 6/2014 đến 4/2015 | 80% ngẫu nhiên, phân tầng theo `TARGET` | 145.619 | 0,3585 | Huấn luyện mô hình, bin WOE, mọi bộ mã hóa |
| Validation | 6/2014 đến 4/2015 | 20% còn lại | 36.405 | 0,3585 | Siêu tham số, chọn mô hình, hiệu chuẩn |
| **Holdout** | **5/2015** | Cả tháng | 17.976 | 0,3629 | Chấm điểm một lần ở cuối |

Lý do: train và validation cùng kỳ nên khoảng cách train→validation chỉ đo **overfit**; holdout ngoài thời gian nên khoảng cách validation→holdout đo **độ ổn định**. Hai chẩn đoán này không bị lẫn vào nhau. Có assert không ID nào nằm ở hai phần.

Hệ quả cần nhớ: validation cùng kỳ với train nên **không thể cảnh báo trôi dữ liệu**. Việc đó là của phần 5.

## 4. Kiểm tra dữ liệu

Chỉ dùng dữ liệu development cho mọi thống kê liên quan đến target.

**Missing (6/53 feature có missing):**

| Feature | Tỷ lệ missing | Bad rate khi missing / khi có | Kết luận |
|---|---|---|---|
| `NUMERICAL_40` | 29,9% | 45,2% / 31,8% | Missing **có thông tin**, ổn định mọi tháng |
| `CATEGORICAL_2..5` | 39,9% (cùng thiếu đồng thời) | 42,2% / 31,6% | Missing **có thông tin**, ổn định mọi tháng |
| `NUMERICAL_10` | 30,9% | 35,8% / 35,9% | Missing **không có thông tin**: thiếu hoàn toàn ba tháng 11/2014 đến 01/2015 (và một phần tháng 10, tháng 2). Đây là sự cố nguồn dữ liệu |

**Feature phân loại:** 2 đến 5 level mỗi cột, không có level hiếm (nhỏ nhất khoảng 20%). Ngoại lệ: level `A` của `CATEGORICAL_7` có ở development nhưng **không có ở tháng 5**.

**Feature số:** 42 cột, gần chuẩn tắc, không có hằng số, không có đỉnh đột biến, không có giá trị bất khả thi, không có cặp gần trùng (tương quan lớn nhất 0,65 giữa `NUMERICAL_10` và `NUMERICAL_41`). `NUMERICAL_1` tập trung quanh 5, `NUMERICAL_40` có độ lệch chuẩn gấp đôi.
**38 trong 42 feature số gần như không có tín hiệu đơn biến** (|Gini| < 0,03). Chỉ `N10` (0,34), `N41` (0,22), `N40` (0,16), `N30` (0,16) có tín hiệu rõ. Phần còn lại hoặc là nhiễu, hoặc mang tín hiệu phi tuyến hay tương tác (mục 7.2).

## 5. Độ ổn định theo thời gian

Đề cảnh báo trước rằng feature có thể không ổn định theo thời gian. Một feature không ổn định trông rất tốt khi kiểm tra chéo rồi hỏng khi vận hành.

### 5.1 Quy tắc (viết ra trước khi áp dụng)

| # | Phép thử | Ngưỡng | Hành động |
|---|---|---|---|
| R1 | PSI giữa development (đã loại các tháng sự cố) và tháng 5 | > 0,25 | **Loại** |
| R2 | Độ lệch quan hệ theo tuần: bad rate theo bin của một tuần so với cùng bin ở các tuần còn lại | > 0,10 ở một tuần bất kỳ | **Loại** |
| R3 | Feature thiếu hoàn toàn (≥ 90%) trong cả tháng ở development | | **Sửa**: giữ, loại các tháng đó khỏi nền PSI, điền giá trị |
| R4 | PSI 0,10 đến 0,25 hoặc độ lệch 0,05 đến 0,10 | | Theo dõi (giữ) |

Độ lệch quan hệ (R2): chia feature thành 5 bin phân vị (cộng một bin `NA`), so sánh bad rate bin của từng tuần với bad rate bin của tất cả các tuần còn lại (leave-one-week-out), lấy trung bình theo số dòng; chỉ xét tuần có từ 1.000 dòng. Mức nhiễu nền của các feature bình thường là khoảng 0,025.
R2 tồn tại vì PSI chỉ nhìn phân phối của feature và hoàn toàn bỏ qua trường hợp **quan hệ với target đổi mà phân phối không đổi**.

### 5.2 Kết quả

| Feature | PSI (so với tháng 5) | Độ lệch tuần tối đa | Quyết định |
|---|---|---|---|
| `CATEGORICAL_9` | 0,000 | **0,358** | **Loại** (R2) |
| `CATEGORICAL_7` | 6,48 | 0,139 | **Loại** (R1 và R2) |
| `NUMERICAL_10` | 4,02 (thô); 0,247 sau khi loại tháng sự cố | 0,022 | **Sửa** (R3), giữ |
| 50 feature còn lại | < 0,01 | ≤ 0,04 | Giữ |

**`CATEGORICAL_9`:** quan hệ với target **đảo chiều đúng một lần**. Tính theo ngày, dấu của chênh lệch bad rate giữa hai level đảo từ **19/04/2015** và giữ nguyên đến hết dữ liệu development: level `AA` từ khoảng 0,55 xuống khoảng 0,18, level `BB` từ khoảng 0,18 lên khoảng 0,54. Tỷ trọng hai level vẫn 50/50. Đây cũng là feature mạnh nhất (IV 0,58), nên nó đúng là loại feature gây hỏng mô hình một cách âm thầm.

**`CATEGORICAL_7`:** level `A` (bad rate khoảng 50%) biến mất từ giữa tháng 4 (tỷ trọng giảm từ khoảng 50% xuống 30% trong tháng 4 rồi về 0% ở tháng 5) và các khách hàng đó nằm vào `B`/`C`. Bad rate của `B` tăng từ 0,29 lên 0,39 và của `C` từ 0,13 lên 0,32. Không thể phục hồi level cũ từ dữ liệu.

**Quyết định loại thay vì "sửa" `CATEGORICAL_9`** (đổi chỗ hai level sau 19/04): dữ liệu development không chứng minh được rằng chiều đảo giữ nguyên sang tháng 5. Nếu đoán sai, tín hiệu mạnh nhất sẽ bị đảo ngược. Đây là một đánh đổi có chủ ý: mất feature mạnh nhất để đổi lấy sự an toàn. Việc hỏi chủ dữ liệu là bước đầu tiên đề xuất (mục 13).

### 5.3 Một chia ngẫu nhiên sẽ không phát hiện được

Thí nghiệm ngay trong dữ liệu development (không dùng tháng 5): huấn luyện trên các ngày trước 19/04 (175.126 dòng), kiểm tra trên các ngày từ 19/04 đến 30/04 (6.898 dòng).

| Bộ feature | Gini chia ngẫu nhiên 80/20 | Gini sau điểm gãy |
|---|---|---|
| Cả 53 feature | 0,884 | **0,243** |
| 51 feature ổn định | 0,679 | 0,702 |

Với bộ đầy đủ, chia ngẫu nhiên báo Gini 0,88 và không có dấu hiệu bất thường, trong khi ngoài thời gian chỉ còn 0,24. Với bộ ổn định, hai con số gần nhau. Chính bước sàng lọc ở mục 5.1, không phải điểm số validation, mới bảo vệ được mô hình.

## 6. Xử lý dữ liệu và feature engineering

Mọi phép biến đổi chỉ học trên train rồi áp dụng nguyên xi cho validation và holdout.

### 6.1 Điền `NUMERICAL_10` (median)

NaN của `NUMERICAL_10` được điền bằng **median của các giá trị quan sát được trong train** (−0,0007). Điền 56.251 dòng development (30,9%): 45.058 ở train, 11.193 ở validation; holdout không có NaN. Bad rate của các dòng được điền (0,357) bằng bad rate của dòng có giá trị thật (0,359): thiếu dữ liệu không mang thông tin. Giá trị gốc được giữ ở cột `NUMERICAL_10_RAW`. Không thêm cờ `MISS_N10`, vì cờ đó chỉ mã hóa tháng nào bị sự cố.

**Điền median không khôi phục thông tin.** Các dòng được điền đều mang cùng một giá trị, nên với khoảng 31% dòng development, feature mạnh nhất (IV 0,26) không còn phân biệt được khách hàng nào. Một phương án đã được thử trước đó là điền bằng hồi quy theo `NUMERICAL_41` (R² khoảng 0,42, tức chỉ khôi phục một phần thông tin): Gini validation cao hơn khoảng 0,006 cho logistic và 0,009 cho LightGBM so với median, nhưng vẫn không bằng khi có giá trị thật. Phương án này đã được thay bằng median theo yêu cầu đơn giản hóa. Hệ quả định lượng xem mục 8.2 và 8.4.

### 6.2 Cờ missing

LightGBM xử lý NaN trực tiếp. Hai cờ tường minh được thêm cho missing **có thông tin**: `MISS_N40` (`NUMERICAL_40` thiếu) và `MISS_CAT2_5` (khối `CATEGORICAL_2..5` thiếu). Với mô hình logistic, missing được biểu diễn bằng bin `NA` riêng của WOE (tương đương cờ).

### 6.3 Mã hóa WOE

`WOE = ln( (bad_i / tổng bad) / (good_i / tổng good) )`, làm mượt bằng `eps = 0,5`. IV = tổng (tỷ trọng bad − tỷ trọng good) × WOE.

- Biến phân loại: mỗi level là một bin. Level có tỷ trọng dưới 1% gộp thành `OTHER`. Level chưa từng thấy khi chấm điểm nhận WOE bằng 0 (trung tính). Trong dữ liệu này không có level hiếm.
- Biến số: coarse classing (mục 6.4), cộng một bin `NA` riêng nếu có missing.

IV (train, 53 feature): `CATEGORICAL_9` 0,576 (vùng "đáng ngờ", đã loại), `CATEGORICAL_7` 0,487 (đã loại), `NUMERICAL_10` 0,265, `CATEGORICAL_1` 0,174, `NUMERICAL_1` 0,156, `NUMERICAL_41` 0,153, `CATEGORICAL_6` 0,140, `NUMERICAL_40` 0,128. 39 trong 53 feature có IV < 0,02.

### 6.4 Coarse classing cho biến số

Hàm `coarse_edges`, học trên train:

1. Bắt đầu từ 20 bin phân vị.
2. Gộp lặp lại hai bin **liền kề** có tỷ lệ good/bad giống nhau nhất (chi-square nhỏ nhất), cho đến khi mọi cặp liền kề khác biệt có ý nghĩa (p < 0,001; chặt hơn 0,05 vì có khoảng 19 phép thử mỗi biến trên 145.000 dòng) **và** còn tối đa **10 bin**.
3. Bin có dưới 5% số dòng được gộp vào bin liền kề giống nhất.
4. Ràng buộc hình dạng: WOE chỉ được đổi chiều tối đa một lần (đơn điệu, hình chữ U hoặc chữ U ngược). **Không ép đơn điệu**, để giữ dạng chữ U của `NUMERICAL_1`.
5. Biến không có tín hiệu có ý nghĩa thống kê sẽ gộp thành một bin.

Kết quả: 36 trong 42 biến số gộp về một bin (không có tín hiệu). 5 biến có tín hiệu giữ 8 đến 10 bin (`N10` 10, `N41` 10, `N1` 10, `N30` 9, `N40` 8). `NUMERICAL_1` là chữ U (rủi ro thấp nhất quanh 4,3 đến 5,3, tăng ở cả hai phía), IV tăng từ 0,131 lên 0,156 so với 10 bin phân vị cố định. Với scorecard: Gini validation 0,594 lên 0,602 so với 10 bin cố định, tổng số bin của các biến số được chọn giảm từ 48 xuống 47.

Ghi chú về trần 10 bin: các biến thể (trần 8, trần 10, không trần) chênh nhau khoảng 0,001 Gini, nằm trong nhiễu; trần 10 là quyết định về độ dễ đọc.

## 7. Các mô hình

### 7.1 Mô hình A: logistic regression trên WOE

- Chọn biến: feature ổn định, IV ≥ 0,02 và tương quan WOE đôi một < 0,70 (giữ biến IV cao hơn khi trùng). Còn **12 biến**; VIF lớn nhất 2,11.
- `LogisticRegression` của scikit-learn, siêu tham số `C` chọn trên validation trong {0,001; 0,01; 0,1; 1; 10}: chọn `C = 10`.
- Tất cả hệ số **dương** (đúng dấu kỳ vọng: WOE cao hơn nghĩa là rủi ro cao hơn).
- Đây là scorecard cộng tính: tổng điểm theo bin, đọc được bin theo bin.

### 7.2 Mô hình A+: thêm một feature tương tác

LightGBM xếp `NUMERICAL_32` cao theo gain nhưng IV của `NUMERICAL_32` và `CATEGORICAL_10` đều gần 0. Nguyên nhân là một **hiệu ứng chéo**: ở level `HNPAK` của `CATEGORICAL_10`, bad rate giảm từ 54% xuống 19% khi `NUMERICAL_32` tăng; ở level `NFAYV` thì tăng theo chiều ngược lại. Trung bình hai level triệt tiêu nhau nên tín hiệu đơn biến bằng 0.

Feature `INT_N32_x_C10 = NUMERICAL_32 × dấu`, trong đó dấu học trên train theo từng level (`HNPAK` −1, `NFAYV` +1). IV của feature mới 0,259 (so với 0,0008 của `NUMERICAL_32` riêng lẻ). Nó thu hẹp khoảng cách giữa logistic regression và LightGBM từ 0,082 xuống 0,012 Gini trên validation (khoảng 85%).

Kiểm tra ổn định cho feature tương tác (cùng phép thử R2 của mục 5): độ lệch tuần tối đa **0,023** (bằng mức nhiễu nền), tương quan bên trong từng level giữ đúng dấu ở cả 45 tuần. Phép thử chỉ bao phủ các tuần development, không nói gì về tháng 5.

Lưu ý về cách phát hiện: tương tác được nhìn ra từ LightGBM và SHAP trên dữ liệu train, không rà soát có hệ thống. Vì tôi đã thấy LightGBM hơn logistic trên validation trước khi quyết định thêm nó, con số validation của A+ không hoàn toàn độc lập với quyết định này. Holdout thì không bị ảnh hưởng.

### 7.3 Mô hình B: LightGBM

- 53 feature: 51 feature ổn định cộng hai cờ missing. Biến phân loại dùng xử lý gốc của LightGBM, danh sách level cố định theo train.
- Mục tiêu `binary`, `learning_rate` 0,05, bagging 0,8, dừng sớm 100 vòng.
- **Không dùng** `scale_pos_weight` hay `is_unbalance` (bad rate 36%, không thật sự mất cân bằng, và re-weight sẽ phá hiệu chuẩn mà Challenge 2 cần). **Không ràng buộc đơn điệu** (feature ẩn danh, có biến phi đơn điệu rõ như `NUMERICAL_1`).
- Lưới 5 cấu hình, chọn bằng quy tắc tiết kiệm: trong các cấu hình có Gini validation cách tốt nhất không quá 0,005, chọn cấu hình overfit ít nhất.

| num_leaves | min_data_in_leaf | feature_fraction | lambda_l2 | vòng | Gini train | Gini valid | gap |
|---|---|---|---|---|---|---|---|
| 7 | 300 | 0,8 | 10 | 949 | 0,706 | 0,680 | 0,026 |
| **15** | **200** | **0,8** | **10** | **671** | **0,726** | **0,683** | **0,043** |
| 31 | 200 | 0,8 | 1 | 747 | 0,784 | 0,684 | 0,101 |
| 31 | 500 | 0,5 | 10 | 1078 | 0,803 | 0,687 | 0,116 |
| 63 | 500 | 0,5 | 30 | 1050 | 0,870 | 0,685 | 0,185 |

Cấu hình tốt nhất theo Gini validation thuần túy (hàng 4) hơn 0,003 nhưng overfit gấp ba.

## 8. Đánh giá

### 8.1 Bảng chính (holdout tháng 5, chấm điểm một lần)

| | Logistic (A) | Logistic + tương tác (A+) | LightGBM (B) |
|---|---|---|---|
| AUC holdout | 0,813 | 0,847 | 0,851 |
| Gini train / valid / holdout | 0,607 / 0,602 / 0,625 | 0,676 / 0,671 / 0,694 | 0,726 / 0,683 / 0,701 |
| KS holdout | 0,467 | 0,527 | 0,534 |
| Gap overfit (train − valid) | 0,006 | 0,005 | 0,043 |
| Gap drift (valid − holdout) | −0,024 | −0,023 | −0,018 |
| Sai số calibration (Platt) | 0,009 | 0,009 | 0,011 |
| PD trung bình (Platt) | 0,3608 | 0,3601 | 0,3605 |

Bad rate thực tế của holdout là 0,3629; của validation là 0,3585.

### 8.2 Đọc hai gap

- **Overfit (train→validation):** không đáng kể cho hai mô hình tuyến tính (0,005 đến 0,006); LightGBM 0,043 (sau quy tắc tiết kiệm; cấu hình lớn nhất của lưới là 0,185).
- **Drift (validation→holdout) âm ở mọi mô hình:** holdout tốt hơn validation, nghĩa là không có dấu hiệu mất ổn định. Gini theo tháng của LightGBM dao động 0,63 đến 0,73 và không có xu hướng giảm.
- **Gap âm là hiệu ứng cơ cấu của tập validation, đã được kiểm chứng (mục 8.4).** Khoảng 31% dòng validation (11/2014 đến 01/2015 và hai tháng kề) chỉ mang **giá trị median được điền** của `NUMERICAL_10` thay cho giá trị thật, còn mọi đơn tháng 5 có giá trị thật. Trên các dòng validation có giá trị thật, Gini bằng đúng Gini holdout (0,701 so với 0,701): gap drift trên dữ liệu so sánh được là khoảng 0.

### 8.3 Calibration và lift (LightGBM, holdout)

- Platt scaling giảm sai số calibration từ 0,0139 xuống 0,0105.
- Sai lệch lớn nhất theo decile là khoảng 2,0 điểm phần trăm (decile 5). **Điểm yếu:** decile an toàn nhất dự báo 1,96% trong khi thực tế 0,67% (mô hình thận trọng ở nhóm này). Decile rủi ro cao nhất dự báo 89,3% so với thực tế 89,0%.
- Bảng lift: decile rủi ro cao nhất có bad rate 89,0% (lift 2,45) và bắt được 24,5% tổng số khoản vỡ nợ; top 3 decile bắt 60,7%; top 5 decile bắt 83,4%; decile thấp nhất chỉ 0,67% vỡ nợ.

### 8.4 Phép thử: Gini theo nguồn gốc của `NUMERICAL_10`

Giả thuyết: gap drift âm là do khoảng 31% dòng validation chỉ có giá trị median được điền của `NUMERICAL_10`, còn mọi dòng holdout có giá trị thật. Nếu đúng, trên các dòng validation có giá trị **thật**, Gini phải bằng Gini holdout. Phép thử nằm ở mục 7.6 của notebook (khoảng tin cậy 95% bằng bootstrap 300 lần; không huấn luyện lại mô hình, trừ phép đối chứng).

| Gini | Số dòng | LightGBM | Logistic | Logistic + tương tác |
|---|---|---|---|---|
| Validation, `NUMERICAL_10` **thật** | 25.212 (69,3%) | **0,701** [0,693; 0,710] | 0,624 | 0,689 |
| Validation, `NUMERICAL_10` **được điền** | 11.193 (30,7%) | 0,641 [0,625; 0,656] | 0,547 | 0,628 |
| Validation, toàn bộ | 36.405 | 0,683 | 0,602 | 0,671 |
| **Holdout** (mọi dòng thật) | 17.976 | **0,701** | 0,625 | 0,694 |

- **Kết quả:** trên dòng có giá trị thật, Gini validation và holdout khớp tới chữ số thập phân thứ ba (0,7013 so với 0,7014). Gap drift thực chất bằng 0; con số −0,018 trong bảng chính là trung bình của một nhóm 0,701 và một nhóm 0,641.
- **Cùng tháng** (loại ảnh hưởng mùa vụ): tháng 10/2014 và 02/2015 là hai tháng duy nhất có cả hai loại dòng. Gộp lại, nhóm thật 0,685 [0,664; 0,704] so với nhóm điền 0,596 [0,522; 0,663]; hai khoảng không chồng nhau. Riêng tháng 02 chỉ có 117 dòng điền nên không kết luận được.
- **Đối chứng:** LightGBM huấn luyện **không có** `NUMERICAL_10` cho Gini 0,646 trên dòng thật, 0,645 trên dòng được điền, 0,649 ở holdout: chênh lệch giữa hai nhóm biến mất. Vậy chênh lệch đến từ chính `NUMERICAL_10`, và feature này đáng khoảng 0,055 Gini (0,701 so với 0,646).
- **Hệ quả:** (1) trên dữ liệu đầy đủ, mô hình hoạt động như nhau ở giai đoạn phát triển và ở tháng 5; không có drift. (2) Tập validation **đánh giá thấp** mô hình vì trộn vào các dòng mà feature mạnh nhất không còn thông tin. (3) Nếu sự cố nguồn dữ liệu lặp lại, Gini trên các đơn bị ảnh hưởng sẽ vào khoảng 0,64; đó là lý do cảnh báo tỷ lệ thiếu theo từng cột nằm trong kế hoạch giám sát.
- **Giới hạn:** nhóm được điền có 11.193 dòng nên khoảng tin cậy rộng khoảng ±0,015. Bằng chứng chắc nhất là sự trùng khớp chính xác với holdout và phép đối chứng.

## 9. Chọn mô hình

Quy tắc (đặt trước khi nhìn holdout): **chọn mô hình đơn giản nhất có Gini validation cách mô hình tốt nhất không quá 0,01**. Gini validation: A 0,602; A+ 0,671; B 0,683. A+ kém B 0,012 nên không đạt, vì vậy chọn **B (LightGBM)**.

Không chọn theo Gini holdout cao nhất: ở đây mô hình cao nhất cũng là B, nhưng quyết định dựa vào validation để holdout giữ nguyên vai trò kiểm tra ngoài thời gian. Phương án dự phòng: A+ (cộng tính, overfit rất nhỏ, calibration tốt, kém B 0,008 Gini trên holdout) là đối thủ phù hợp nếu hội đồng rủi ro yêu cầu scorecard thuần.

## 10. Giải thích mô hình

**Gain so với SHAP** (LightGBM): hai bảng xếp hạng cùng đứng đầu bởi `CATEGORICAL_1`, `CATEGORICAL_6`, `NUMERICAL_10`, `NUMERICAL_40`, `NUMERICAL_1` nhưng thứ tự khác nhau. Gain tính số lần chia nên đề cao biến phải chia nhiều lần (ví dụ `NUMERICAL_10` hạng 1 theo gain nhưng hạng 3 theo SHAP; `NUMERICAL_1` hạng 2 và 5). `NUMERICAL_32` xếp cao theo gain ở các lần chạy nhưng không nằm trong top 12 theo SHAP: hiệu ứng của nó chỉ tồn tại khi đi cùng `CATEGORICAL_10` và đổi dấu giữa hai level.

**Các yếu tố chính** (tên cột đã được ẩn danh nên chỉ mô tả hình dạng):

1. `CATEGORICAL_1`: một nhóm (`DVSOM`) an toàn rõ rệt (bad rate 23%) so với hai nhóm còn lại (41 đến 44%).
2. `CATEGORICAL_6`: level `A` rủi ro nhất (45%), `C` an toàn nhất (25%).
3. `NUMERICAL_10`: rủi ro tăng gần tuyến tính; decile thấp nhất khoảng 14% vỡ nợ, cao nhất khoảng 62%. Biên độ lớn nhất (khoảng ±2 log-odds).
4. `NUMERICAL_40`: rủi ro tăng gần tuyến tính, và **thiếu giá trị là một tín hiệu rủi ro** (45% so với 32%).
5. `NUMERICAL_1`: hình chữ U quanh 5; rủi ro tăng mạnh khi lệch xa 5 ở cả hai phía.
6. `CATEGORICAL_8`, `NUMERICAL_30`, `NUMERICAL_41`, khối `CATEGORICAL_2..5` (thiếu giá trị làm tăng rủi ro).
7. Tương tác `NUMERICAL_32` × `CATEGORICAL_10` (hiệu ứng chéo).

Hình dạng trơn, hoặc đơn điệu hoặc chữ U sạch; không thấy đường răng cưa là dấu hiệu của overfit. Hai ca cá nhân (một ca PD rất thấp, một ca rất cao) được giải thích bằng đóng góp log-odds của từng feature, dạng "lý do chính" của một thông báo từ chối.

## 11. Bàn giao `pd_predictions.csv`

| Thuộc tính | Giá trị |
|---|---|
| Số dòng | 17.976 (mọi đơn tháng 5, không lọc) |
| Cột | `ID_APPLICATION`, `PD` |
| Mô hình | LightGBM đã hiệu chuẩn Platt trên validation |
| PD trung bình | 0,3605 (bad rate thực tế tháng 5: 0,3629; lệch −0,0024, tức thấp 0,24 điểm phần trăm) |
| Khoảng PD | 0,0005 đến 0,996 |

Kiểm tra tự động trong notebook: không trùng ID, không NaN, PD trong [0, 1], đủ số dòng, tập ID bằng tập ID tháng 5.
Hiệu chuẩn trên validation chỉnh được **mức** PD nhưng không sửa được trôi dữ liệu giữa giai đoạn development và tháng 5. Bằng chứng ngoài thời gian duy nhất là bản đối chiếu trên holdout ở trên. Phần lệch nhỏ còn lại (0,24 điểm) khớp với việc validation có bad rate thấp hơn tháng 5 chút (0,3585 so với 0,3629).

## 12. Nhật ký quyết định

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| `CATEGORICAL_9` | Loại, không sửa | Không chứng minh được chiều đảo giữ ở tháng 5; sai thì đảo ngược tín hiệu mạnh nhất |
| `CATEGORICAL_7` | Loại | Level `A` biến mất, không phục hồi được |
| `NUMERICAL_10` | Giữ, điền median | Sự cố nguồn dữ liệu, tháng 5 bình thường; median là cách điền đơn giản nhất (theo yêu cầu) |
| Ngưỡng loại feature | PSI 0,25; độ lệch 0,10 | Quy ước chuẩn, chưa tinh chỉnh |
| Binning | Coarse classing, p < 0,001, ≤ 10 bin, ≥ 5% | Đọc được, giữ chữ U của `NUMERICAL_1` |
| Mô hình thứ ba | A+ (một tương tác) | Kiểm tra xem scorecard có theo kịp boosting khi có đúng thông tin |
| Chọn mô hình | Đơn giản nhất trong 0,01 Gini (validation) | Quy tắc đặt trước, không nhìn holdout |
| Hiệu chuẩn | Platt trên validation | Giữ nguyên thứ hạng (Gini không đổi), 2 tham số |
| Overfit LightGBM | Quy tắc tiết kiệm (≤ 0,005 Gini cách tốt nhất) | Cấu hình tốt nhất tuyệt đối overfit gấp ba để lợi 0,003 Gini |

## 13. Hạn chế và việc tiếp theo

**Hạn chế**

- **Chỉ một tháng holdout**, nên "không có drift" là một quan sát, không phải bảo đảm.
- **Điền median không khôi phục thông tin** (mục 6.1). Hệ quả đã đo ở mục 8.4: trên các dòng chỉ có giá trị điền, Gini LightGBM là 0,64 so với 0,70 ở dòng có giá trị thật, nên nếu sự cố nguồn dữ liệu lặp lại thì chất lượng trên các đơn bị ảnh hưởng sẽ giảm tương ứng.
- **Gini holdout (khoảng 0,70) thấp hơn nhiều so với bộ đầy đủ trông có vẻ đạt** (0,88 khi chia ngẫu nhiên), vì đã loại `CATEGORICAL_9`.
- **Tương tác được tìm bằng tay** từ LightGBM và SHAP trên train, chỉ một tương tác; có thể còn tương tác khác. Số validation của A+ không hoàn toàn độc lập với quyết định này.
- **Decile an toàn nhất bị dự báo thận trọng** (1,96% so với 0,67%).
- Ngưỡng PSI/độ lệch, tham số coarse classing và lưới LightGBM nhỏ đều là quy ước hoặc lựa chọn của người làm, chưa tối ưu.

**Với thêm thời gian**

1. Hỏi chủ dữ liệu về `CATEGORICAL_9` và `CATEGORICAL_7`. Nếu 19/04 là một lần đổi mã nhãn được xác nhận thì sửa (đổi chỗ hai level sau ngày đó) và lấy lại feature mạnh nhất.
2. Tìm tương tác có hệ thống (SHAP interaction values hoặc quét cặp biến, có kiểm soát so sánh nhiều lần).
3. Khoảng tin cậy cho Gini (bootstrap), ràng buộc đơn điệu cho các biến đơn điệu.
4. Hiệu chuẩn isotonic ở vùng PD thấp nếu Challenge 2 nhạy với nó.
5. Giám sát khi vận hành: độ lệch quan hệ theo tuần và PSI cho mọi feature (kể cả feature tương tác); cảnh báo tỷ lệ thiếu của từng cột (như sự cố `NUMERICAL_10`); cảnh báo khi xuất hiện hoặc mất một level; Gini và PD trung bình so với thực tế hằng tháng; giữ A+ làm mô hình đối chứng.

## 14. Tái lập kết quả

1. Đặt `training_data.csv.gz` trong `data/` (repo không kèm file này, xem `data/README.md`).
2. Mở `notebooks/01_pd_model.ipynb` với kernel `pydo-project` (thư mục làm việc là `notebooks/`), chạy toàn bộ từ trên xuống (khoảng 5 đến 6 phút). Mọi seed đã cố định (chia dữ liệu 42, LightGBM 42).
3. Notebook ghi `results/pd_predictions.csv`.

Cấu trúc notebook: 0 môi trường và nạp dữ liệu; 1 kiểm tra dữ liệu; 2 độ ổn định; 3 chia dữ liệu (3.1 điền `NUMERICAL_10`); 4 feature engineering (4.2 coarse classing); 5 mô hình A; 6 mô hình B (6.5 và 6.6 tương tác, 6.7 kiểm tra ổn định); 7 đánh giá (7.1 chấm holdout một lần, 7.5 chọn mô hình, 7.6 phép thử Gini theo nguồn gốc của `NUMERICAL_10`); 8 giải thích; 9 bàn giao; 10 kết luận.
