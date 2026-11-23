# Challenge 2: tối ưu chính sách duyệt/từ chối

- Notebook: `notebooks/02_portfolio_optimization.ipynb`.
- Công cụ: thư viện `pydo` (xem [pydo.md](pydo.md)).
- Đầu vào: `results/pd_predictions.csv` từ Challenge 1.
- Kết quả bàn giao (trong `results/`): `optimal_policy.csv`, `constrained_output.csv`, `models/clx_approval_tree.json`, `models/clx_approval_tree.sql`.

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Bài toán](#2-bài-toán)
3. [Đầu vào và các cột dẫn xuất](#3-đầu-vào-và-các-cột-dẫn-xuất)
4. [Kinh tế của một khoản vay](#4-kinh-tế-của-một-khoản-vay)
5. [Mô hình hóa: biến, ràng buộc, ba flow](#5-mô-hình-hóa-biến-ràng-buộc-ba-flow)
6. [Kết quả của ba flow](#6-kết-quả-của-ba-flow)
7. [Chẩn đoán](#7-chẩn-đoán)
8. [Chưng cất thành chính sách dạng luật](#8-chưng-cất-thành-chính-sách-dạng-luật)
9. [Tập kiểm thử bằng kết quả thực tế](#9-tập-kiểm-thử-bằng-kết-quả-thực-tế)
10. [Sản phẩm bàn giao](#10-sản-phẩm-bàn-giao)
11. [Khuyến nghị](#11-khuyến-nghị)
12. [Nhật ký quyết định và giả định](#12-nhật-ký-quyết-định-và-giả-định)
13. [Hạn chế và việc tiếp theo](#13-hạn-chế-và-việc-tiếp-theo)
14. [Tái lập kết quả](#14-tái-lập-kết-quả)

---

## 1. Tóm tắt

- Bài toán: quyết định duyệt hay từ chối từng đơn trong 17.976 đơn tháng 5/2015, tối đa hóa lợi nhuận kỳ vọng và thỏa 7 ràng buộc kinh doanh.
- **Chính sách đề xuất: cây 32 luật** (theo dải rủi ro, kỳ hạn, lãi suất, ticket), giữ **97,35%** nghiệm tối ưu cấp tài khoản, thỏa cả 7 ràng buộc.
- Lợi nhuận kỳ vọng so với duyệt hết (BAU): **59,31 tỷ so với 11,98 tỷ (×5,0)**. Thực tế (tập kiểm thử): **58,28 tỷ so với 10,57 tỷ (×5,5)**.
- Hai ràng buộc chi phối: **C4 (kỳ hạn dài)** và **C5 (micro ≥ 12%)**. Nên cân nhắc nới C4.
- Giá trị của mô hình PD (Challenge 1) trong bài toán này: **10,4 tỷ VND** lợi nhuận kỳ vọng (17%).
- Công thức kinh tế là giả định tổng hợp của đề, nên lợi nhuận là con số mô phỏng.

| Chính sách | Lợi nhuận kỳ vọng | Rule vi phạm (trên 42) |
|---|---|---|
| BAU (duyệt hết) | 11,98 tỷ | 2 |
| Ngưỡng PD đơn (PD ≤ 0,362) | 54,94 tỷ | 1 |
| Trần không ràng buộc | 61,18 tỷ | 4 |
| Tối ưu cấp tài khoản | 60,92 tỷ | 0 |
| **Cây 32 luật** | **59,31 tỷ** | **0** |

## 2. Bài toán

- **Quyết định:** với mỗi đơn, duyệt (1) hay từ chối (0). Một biến quyết định `APPROVE_ACTION` ∈ {0, 1}, hai phương án mỗi đơn.

- **Mục tiêu:** tối đa hóa tổng lợi nhuận kỳ vọng của danh mục được duyệt, tính theo PD.

**Bảy ràng buộc (ngưỡng cố định của đề):**

| # | Ràng buộc | Giá trị |
|---|---|---|
| C1 | Tỷ lệ rủi ro = khối lượng vỡ nợ kỳ vọng / tổng số tiền giải ngân | ≤ 22% |
| C2 | Tỷ lệ duyệt = số đơn duyệt / tổng số đơn | ≥ 45% |
| C3 | Tổng giải ngân | ≤ 350 tỷ VND |
| C4 | Tỷ trọng số tiền giải ngân có kỳ hạn ≥ 36 tháng | ≤ 45% |
| C5 | Tỷ trọng **số khoản** duyệt thuộc `CLX_MICRO` | ≥ 12% |
| C6 | Tỷ trọng **số khoản** duyệt có lãi suất > 40% | ≤ 20% |
| C7 | Tỷ lệ duyệt giảm dần (hoặc giữ nguyên) khi rủi ro tăng, trong từng nhóm sản phẩm | — |

**Luật chơi của đề:**

- Tối ưu theo `PD`; `TARGET` chỉ dùng ở tập kiểm thử cuối.
- Giữ nguyên các ngưỡng của đề để đạt khả thi.
- Ghi rõ các tham số hợp đồng và lợi nhuận là số tổng hợp của đề.

## 3. Đầu vào và các cột dẫn xuất

- 17.976 đơn tháng 5/2015 ghép với `pd_predictions.csv` (mỗi đơn có đúng 1 dòng PD).
- Các cột hợp đồng dùng ở đây (Challenge 1 cấm dùng các cột này): `AMOUNT_VND` (5 triệu đến 250 triệu), `TENOR_MONTHS` (3 đến 60), `INTEREST_RATE` (0,16 đến 0,50), `MONTHLY_INSTALLMENT`, `PRODUCT_GROUP`.
- Kiểm tra dữ liệu: mọi lãi suất lớn hơn 0; `MONTHLY_INSTALLMENT` khớp công thức annuity với lãi suất/12 và được làm tròn tới đồng.
- Nếu duyệt hết, tổng giải ngân là 574,4 tỷ (so với ngân sách 350 tỷ).
- Cơ cấu sản phẩm: `CLX_SMALL` 9.777 đơn, `CLX_MID` 4.257, `CLX_MICRO` 3.619, `CLX_LARGE` 323.

**Các cột thô được tạo thêm** (đây là các cột duy nhất dùng làm điều kiện `WHERE` của ràng buộc, vì tập đơn mà ràng buộc áp dụng phải xác định trước khi chạy solver):

| Cột | Định nghĩa |
|---|---|
| `RISK_BAND` | 10 dải PD theo ngưỡng cố định: [0; 0,05), [0,05; 0,10), [0,10; 0,15), [0,15; 0,20), [0,20; 0,25), [0,25; 0,32), [0,32; 0,40), [0,40; 0,50), [0,50; 0,65), [0,65; 1,01) |
| `TENOR_BUCKET` | `T_SHORT` (≤ 11 tháng), `T_MID` (12 đến 35), `T_LONG` (≥ 36) |
| `HIGH_IR_FLAG` | 1 nếu lãi suất > 40% |
| `AMOUNT_M` | `AMOUNT_VND` ÷ 10⁶ (đơn vị triệu VND) |
| `APPROVE_ACTION_BAU` | Cột toàn 1: chính sách "duyệt hết" cho flow `simulation` |

**Vì sao dùng ngưỡng PD cố định thay vì chia theo phân vị**

- Chia theo phân vị: mỗi dải chứa cùng số đơn (ví dụ dải cao nhất là 10% đơn rủi ro nhất), nên ranh giới PD của dải thay đổi theo các đơn của từng tháng.
  - Ví dụ: tháng 5, 10% đơn rủi ro nhất bắt đầu từ PD 0,81. Nếu tháng sau có nhiều đơn tốt hơn, ranh giới này có thể tụt xuống 0,70 (số minh họa).
- Hệ quả: cùng một luật, ví dụ "từ chối dải cao nhất", từ chối các đơn có PD khác nhau mỗi tháng. Khi tỷ lệ duyệt của dải thay đổi, người theo dõi không phân biệt được nguyên nhân là chính sách hay ranh giới dịch chuyển.
- Ngưỡng cố định: dải "PD từ 0,20 đến 0,25" luôn mang cùng một ý nghĩa, nên so sánh được giữa các tháng.

**Vì sao chọn các ngưỡng này** (mỗi dải là một khoảng giá trị PD)

- Dưới 25%: mỗi dải rộng 5 điểm phần trăm PD (0 đến 0,05; 0,05 đến 0,10; …; 0,20 đến 0,25). Đây là vùng quyết định duyệt hay từ chối ảnh hưởng lợi nhuận nhiều nhất nên cần chia mịn.
- Từ 0,25 trở lên: dải rộng hơn (0,25 đến 0,32; 0,32 đến 0,40; 0,40 đến 0,50; 0,50 đến 0,65; từ 0,65). PD càng cao thì gần như khoản vay nào cũng lỗ, nên chia thô là đủ.
- PD hòa vốn (`PD*`) của một đơn là mức PD tại đó lợi nhuận kỳ vọng bằng 0. Trung vị của `PD*` trên 17.976 đơn khoảng 24%, nên các dải 0,20 đến 0,25 và 0,25 đến 0,32 nằm ngay quanh mức này.

**Số đơn trong mỗi dải**

- Dải nhỏ nhất là dải 4 (PD từ 0,20 đến 0,25): 1.090 đơn, 6,1% tổng số đơn. Đủ lớn để đọc kết quả.
- Bảng sản phẩm × dải có 40 ô (4 sản phẩm × 10 dải). 7 ô có dưới 30 đơn, đều thuộc `CLX_LARGE`, ở các dải 2 đến 8 (PD từ 0,10 đến 0,65), với 17 đến 29 đơn mỗi ô. Luật C7 giữ các ô này; kết quả ở đó kém ổn định.

## 4. Kinh tế của một khoản vay

- Hàm tính nằm ở:
  - `notebooks/functions/flatten.py`: đọc trường thô và giả định vào các thuộc tính có tiền tố.
  - `notebooks/functions/economics.py`: bốn bước `amortisation`, `profit_good`, `profit_bad`, `metrics`.
- Quy ước tiền tố: `p_` trường thô, `a_` hành động, `f_` giả định (fact), `v_` giá trị tính được.

### 4.1 Giả định (số tổng hợp của đề)

| Tham số | Giá trị |
|---|---|
| Chi phí vốn | 9,0%/năm trên dư nợ bình quân |
| Chi phí thu hút khách | 300.000 VND mỗi khoản duyệt |
| Chi phí phục vụ | 25.000 VND mỗi khoản mỗi tháng |
| Thời điểm vỡ nợ | sau `0,35 × kỳ hạn` kỳ trả |
| Thu hồi | 20% dư nợ gốc tại thời điểm vỡ nợ |

### 4.2 Công thức

Khoản vay annuity: mỗi tháng khách trả cùng một số tiền. Mỗi công thức có phần giải thích ký hiệu ngay bên dưới; tên trong ngoặc là tên cột hoặc biến tương ứng trong mã.

#### Bước 1. Lịch trả nợ

```text
(1)  r = i / 12
```

- `i`: lãi suất năm (`INTEREST_RATE`, `p_ir`); trong dữ liệu `i` từ 0,16 đến 0,50.
- `r`: lãi suất tháng (`v_monthly_rate`).

```text
(2)  B(m) = A·(1+r)^m − I·((1+r)^m − 1) / r
```

- `m`: số kỳ (tháng) đã trả, từ 0 đến `n`.
- `B(m)`: dư nợ gốc còn lại sau khi trả `m` kỳ (`v_balance_fn`). `B(0) = A` và `B(n)` bằng 0.
- `A`: số tiền vay, tức số tiền giải ngân, VND (`AMOUNT_VND`, `p_amount`).
- `n`: kỳ hạn, số tháng (`TENOR_MONTHS`, `p_tenor`).
- `I`: số tiền trả mỗi tháng, làm tròn tới đồng (`MONTHLY_INSTALLMENT`, `p_installment`).
- `I` làm tròn tới đồng nên `B(n)` lệch vài đồng quanh 0; mã lấy `B(m) = 0` khi kết quả âm.
- Tiền gốc được trả dồn về cuối kỳ: dư nợ giảm chậm ở đầu kỳ và nhanh ở cuối kỳ.

#### Bước 2. Lợi nhuận nếu khoản vay trả đủ

```text
(3)  B_avg_G = [ B(0) + B(1) + … + B(n) ] / (n + 1)
```

- `B_avg_G`: dư nợ gốc bình quân suốt kỳ hạn (`v_avg_outstanding_good`).
- `B_avg_G` lớn hơn nhiều so với `A/2`, vì tiền gốc trả dồn về cuối kỳ.

```text
(4)  Int_G = I·n − A
```

- `Int_G`: tiền lãi thu được khi trả đủ (`v_interest_income`). Bằng tổng tiền khách trả (`I·n`) trừ tiền gốc (`A`).

```text
(5)  Fund_G = c · B_avg_G · n/12
```

- `Fund_G`: chi phí vốn khi trả đủ (`v_funding_cost_good`).
- `c`: chi phí vốn mỗi năm, tính trên dư nợ bình quân; giả định 9,0% (`cost_of_funds`).
- `n/12`: số năm của khoản vay.

```text
(6)  Serv_G = s · n
```

- `Serv_G`: chi phí phục vụ khi trả đủ (`v_servicing_good`).
- `s`: chi phí phục vụ mỗi khoản mỗi tháng; giả định 25.000 VND (`servicing_cost`).

```text
(7)  G = Int_G − Fund_G − Serv_G − a
```

- `G`: lợi nhuận nếu khoản vay trả đủ ("good", `v_profit_good`).
- `a`: chi phí thu hút khách, trả một lần cho mỗi khoản duyệt; giả định 300.000 VND (`acquisition_cost`).

#### Bước 3. Lợi nhuận nếu khoản vay vỡ nợ

```text
(8)  m_d = θ · n,  làm tròn tới số nguyên gần nhất, tối thiểu là 1
```

- `θ`: vỡ nợ xảy ra sau bao nhiêu phần của kỳ hạn; giả định 0,35 (`default_at`).
- `m_d`: số kỳ khách đã trả trước khi ngừng trả ("d" là default, `v_months_to_default`).
- Ví dụ: kỳ hạn `n = 36` thì `θ·n = 12,6` và `m_d = 13`.

```text
(9)  B_d = A·(1+r)^m_d − I·((1+r)^m_d − 1) / r
```

- `B_d`: dư nợ gốc còn lại tại lúc vỡ nợ, tức số tiền khách còn nợ (`v_outstanding_at_default`).

```text
(10)  B_avg_D = [ B(0) + B(1) + … + B(m_d) ] / (m_d + 1)
```

- `B_avg_D`: dư nợ gốc bình quân trong `m_d` kỳ đầu (`v_avg_outstanding_bad`).

```text
(11)  Prin_D = A − B_d
```

- `Prin_D`: tiền gốc đã trả trước khi vỡ nợ.

```text
(12)  Int_D = I·m_d − Prin_D
```

- `Int_D`: tiền lãi thu được trước khi vỡ nợ (`v_interest_collected`).
- `I·m_d` là tổng tiền khách đã trả; trừ `Prin_D` vì mỗi kỳ trả gồm một phần gốc và một phần lãi.

```text
(13)  Fund_D = c · B_avg_D · m_d/12
(14)  Serv_D = s · m_d
```

- `Fund_D`, `Serv_D`: chi phí vốn và chi phí phục vụ khi vỡ nợ (`v_funding_cost_bad`, `v_servicing_bad`).
- Hai chi phí này chạy trong `m_d` tháng.

```text
(15)  Loss_D = (1 − ρ) · B_d
```

- `Loss_D`: lỗ tín dụng, tức phần dư nợ bị mất sau thu hồi (`v_credit_loss`).
- `ρ`: tỷ lệ thu hồi trên dư nợ tại lúc vỡ nợ; giả định 20% (`recovery_rate`).
- Lỗ tính trên dư nợ còn lại `B_d`.

```text
(16)  D = Int_D − Fund_D − Serv_D − Loss_D − a
```

- `D`: lợi nhuận nếu khoản vay vỡ nợ ("default", `v_profit_bad`); là số âm.
- Chi phí thu hút `a` trả cả khi khoản vay vỡ nợ.

#### Bước 4. Lợi nhuận kỳ vọng và điểm hòa vốn

```text
(17)  E = (1 − PD)·G + PD·D
```

- `PD`: xác suất vỡ nợ từ Challenge 1 (`PD`, `p_pd`).
- `E`: lợi nhuận kỳ vọng của khoản vay nếu được duyệt. Là trung bình của `G` và `D` với trọng số xác suất trả đủ (`1 − PD`) và xác suất vỡ nợ (`PD`).

```text
(18)  E > 0  ⇔  PD < PD* = G / (G − D)        (vì G > 0 > D)
```

- `PD*`: PD hòa vốn, là PD cao nhất mà khoản vay còn có lợi nhuận kỳ vọng dương (cột `breakeven_pd` trong notebook).

#### Bước 5. Biến đưa vào ràng buộc

```text
(19)  disbursed_m = x · A / 10⁶
(20)  exp_bad_m   = x · PD · A / 10⁶
(21)  exp_profit  = x · E
```

- `x`: quyết định, 1 là duyệt, 0 là từ chối (`APPROVE_ACTION`, `a_approve_action`).
- `disbursed_m`: số tiền giải ngân, triệu VND (`v_disbursed_amount_m`). Dùng cho C1, C3, C4.
- `exp_bad_m`: khối lượng vỡ nợ kỳ vọng, triệu VND (`v_exp_bad_volume_m`). Dùng để báo cáo.
- `exp_profit`: lợi nhuận kỳ vọng, VND (`v_exp_profit`). Đây là mục tiêu của bài toán.
- Đơn bị từ chối (`x = 0`) có mọi giá trị bằng 0, nên các phép tổng hợp của ràng buộc chỉ cộng các đơn được duyệt.
- Ba biến phụ trong mã:
  - `v_approved` (= `x`): dùng cho C2, C5, C6, C7.
  - `v_pd_disbursed` (= `PD`): đầu vào của trung bình có trọng số ở C1, trọng số là `disbursed_m`.
  - `v_disbursed_amount` (= `x·A`, VND) và `v_exp_profit_m` (= `x·E/10⁶`, triệu): dùng cho thí nghiệm điều kiện số học (so sánh đơn vị VND với triệu).

### 4.3 Kiểm tra độc lập

- Năm hợp đồng (nhỏ/lớn, ngắn/dài, lãi cao) được tính lại bằng lịch trả nợ **từng tháng**: `B(m) = B(m−1)·(1+r) − I`.
- Kết quả khớp pipeline ở mọi thành phần (lãi, dư nợ bình quân, chi phí vốn, lỗ khi vỡ nợ, `G`, `D`).
- Chênh lệch lớn nhất 34,7 đồng trên khoản 250 triệu: `MONTHLY_INSTALLMENT` làm tròn tới đồng nên lịch trả còn dư vài đồng ở kỳ cuối.
- Ngưỡng chấp nhận: chênh lệch ≤ 1 đồng mỗi tháng kỳ hạn.

### 4.4 Kinh tế cho thấy gì

- Điểm hòa vốn: `PD* = G / (G − D)`.
- **44,4%** đơn có PD < PD* (lợi nhuận kỳ vọng dương); PD* trung vị 24,1%.

| Sản phẩm | Số đơn | PD* trung vị | PD trung vị | Tỷ lệ đơn có lãi |
|---|---|---|---|---|
| `CLX_LARGE` | 323 | 0,457 | 0,153 | 81% |
| `CLX_MID` | 4.257 | 0,387 | 0,210 | 68% |
| `CLX_SMALL` | 9.777 | 0,236 | 0,294 | 44% |
| `CLX_MICRO` | 3.619 | 0,075 | 0,395 | **16%** |

**Vì sao khoản micro kỳ hạn ngắn không sinh lời**

- Xét `CLX_MICRO` kỳ hạn ≤ 6 tháng (2.092 khoản):
  - Khoản vay trung vị 10,4 triệu; tiền lãi trung vị cả kỳ chỉ 0,75 triệu.
  - Chi phí cố định (thu hút + phục vụ) trung vị 0,38 triệu, bằng 50% tiền lãi.
  - Khi vỡ nợ (sau 35% kỳ hạn), khoản micro ngắn trong năm hợp đồng kiểm tra còn khoảng 70% gốc chưa trả. Khoản lỗ gấp nhiều lần tiền lãi của một khoản trả đủ.
- Xét `CLX_MICRO` kỳ hạn ≤ 11 tháng (`T_SHORT`, 2.660 khoản): `PD*` trung vị là 4,8%, nên chỉ 8% khoản có lợi nhuận kỳ vọng dương.
- Hệ quả: C5 buộc duyệt ít nhất 12% số khoản là micro, tức cho vay đúng nhóm có kinh tế kém nhất.

## 5. Mô hình hóa: biến, ràng buộc, ba flow

### 5.1 Hệ thống ràng buộc (viết bằng ngôn ngữ luật của `pydo`)

Ký hiệu:

- `k`: chỉ số đơn; `N = 17.976`: tổng số đơn.
- `x_k`: quyết định; `A_k`: số tiền vay; `PD_k`: xác suất vỡ nợ.
- `T_long`: tập đơn có `TENOR_BUCKET = T_LONG`.
- `M`: tập đơn có `PRODUCT_GROUP = CLX_MICRO`.
- `H`: tập đơn có `HIGH_IR_FLAG = 1`.
- `S_{p,b}`: tập đơn của sản phẩm `p` thuộc dải rủi ro `b`.
- `Σ` không ghi tập là tổng trên mọi đơn.

| # | Dạng toán | Luật `pydo` (rút gọn) |
|---|---|---|
| C1 | `Σ x_k·PD_k·A_k / Σ x_k·A_k ≤ 0,22` | `WEIGHTED_AVG(v_pd_disbursed, v_disbursed_amount_m) <= 0.22` |
| C2 | `Σ x_k / N ≥ 0,45` | `AVG(v_approved) >= 0.45` |
| C3 | `Σ x_k·A_k / N ≤ 350.000 triệu / N` (viết theo đầu đơn để giữ nguyên độ chặt khi chạy mẫu) | `AVG(v_disbursed_amount_m) <= 350000 / 17976` |
| C4 | `Σ_{k∈T_long} x_k·A_k ≤ 0,45 · Σ x_k·A_k` | `SUM(v_disbursed_amount_m) WHERE (TENOR_BUCKET = 'T_LONG') <= SUM(v_disbursed_amount_m) * 0.45` |
| C5 | `Σ_{k∈M} x_k ≥ 0,12 · Σ x_k` | `SUM(v_approved) WHERE (PRODUCT_GROUP = 'CLX_MICRO') >= SUM(v_approved) * 0.12` |
| C6 | `Σ_{k∈H} x_k ≤ 0,20 · Σ x_k` | `SUM(v_approved) WHERE (HIGH_IR_FLAG = 1) <= SUM(v_approved) * 0.20` |
| C7 | Với mỗi sản phẩm `p` và mỗi cặp dải liền kề `(b, b+1)`: `mean_{k∈S_{p,b}} x_k ≥ mean_{k∈S_{p,b+1}} x_k` | `AVG(v_approved) WHERE (sản phẩm, dải b) >= AVG(v_approved) WHERE (sản phẩm, dải b+1)` |

- Tổng cộng 42 luật: C1 đến C6 và 36 luật C7 (4 sản phẩm × 9 cặp dải liền kề).
- C7 chia theo sản phẩm. Chia thêm theo kỳ hạn làm tăng số luật và để lại nhiều ô chỉ vài đơn.
- Cặp dải có một ô rỗng bị bỏ; ô thưa nhưng có đơn được giữ.

### 5.2 Vì sao một số luật an toàn khi chạy mẫu

- C1 (tỷ số hai tổng), C2 (trung bình), C4, C5, C6 (tỷ trọng, viết dạng `SUM(...) <= SUM(...) * k`) và C7 (tỷ lệ duyệt theo ô) đều là **tỷ lệ**, nên có nghĩa như nhau trên mẫu 25% và trên toàn bộ.
- Chỉ C3 là số tuyệt đối (tổng giải ngân). C3 được viết theo đầu đơn (`AVG(...) <= 350.000/N_full`) để giữ nguyên độ chặt khi chạy mẫu 25%; viết theo tổng thì giới hạn lỏng gấp 4 lần.

### 5.3 Ba flow và baseline

| Flow | Ý nghĩa |
|---|---|
| `simulation` | Đánh giá một chính sách cố định (ở đây: duyệt hết, đọc từ cột `APPROVE_ACTION_BAU`) |
| `unconstrained` | Phương án tốt nhất của từng đơn khi chưa áp luật: duyệt đúng các khoản có lợi nhuận kỳ vọng dương (trần) |
| `constrained` | Tối ưu dưới 42 luật: bài toán nguyên nhị phân giải bằng HiGHS, một biến nhị phân mỗi đơn |

- Thứ tự kỳ vọng: `simulation ≤ constrained ≤ unconstrained`; notebook kiểm tra thứ tự này bằng `assert`.

## 6. Kết quả của ba flow

| | BAU | Trần | Tối ưu có ràng buộc |
|---|---|---|---|
| Lợi nhuận kỳ vọng | 11,98 tỷ | 61,18 tỷ | 60,89 tỷ (60,92 tỷ khi giải tới gap 0,01%) |
| Tỷ lệ rủi ro (C1 ≤ 22%) | 32,9% | 14,6% | 14,4% |
| Tỷ lệ duyệt (C2 ≥ 45%) | 100% | 44,4% | 48,5% |
| Giải ngân (C3 ≤ 350 tỷ) | 574,4 tỷ | 324,5 tỷ | 335,3 tỷ |
| Kỳ hạn dài (C4 ≤ 45%) | 33,3% | 47,9% | 45,0% |
| Micro (C5 ≥ 12%) | 20,1% | 7,2% | 12,0% |
| Lãi cao (C6 ≤ 20%) | 18,0% | 7,7% | 8,5% |

- **BAU** vi phạm C1 và C3.
- **Trần** vi phạm 4 luật: C2 (44,4% < 45%), C4 (47,9% > 45%), C5 (7,2% < 12%) và một luật C7 của `CLX_MICRO` (dải 4 sang 5). Danh sách này là lý do cần mô hình có ràng buộc.
- **Tối ưu có ràng buộc:**
  - Giữ **99,5% của trần**, nhỏ hơn trần 0,29 tỷ (0,47%).
  - Solver: tối ưu chứng minh được, gap 0,05% (yêu cầu 1%), khoảng 32 giây, một nút duy nhất (nới lỏng LP đã nguyên).
- **Mẫu 25% so với toàn bộ:**
  - Lợi nhuận quy đổi lệch dưới 2% so với chạy toàn bộ (giữa +0,9% và +1,8% ở các lần chạy).
  - Tỷ lệ duyệt tương đương; mọi luật đều thỏa; kết luận không đổi.
  - Số cuối cùng luôn lấy từ chạy toàn bộ.

## 7. Chẩn đoán

Các phép quét dùng `Scenario.with_constraints` (đổi luật và giữ nguyên kinh tế đã tính):

- Số tiêu đề lấy từ chạy toàn bộ.
- Các phép quét nhiều lần (đường biên, chi phí từng luật riêng, bisection) chạy trên mẫu 25% rồi quy đổi; kết quả cao hơn chạy toàn bộ khoảng 1,8%, nên chỉ đọc hình dạng.

### 7.1 Điều kiện số học

Cùng một mô hình ở ba cách chọn đơn vị:

- (A) mọi đại lượng bằng VND thô; (B) số tiền giải ngân theo triệu, mục tiêu theo VND (bản dùng cho kết quả chính); (C) cả hai theo triệu.
- Dải hệ số: 1,1e12 (A), 4,8e11 (B), **1,1e6 (C)**; chỉ (C) dưới ngưỡng cảnh báo 1e9.
- Quyết định **giống nhau** ở cả ba (khác dưới 0,05% số đơn, đúng bằng dung sai solver); lợi nhuận như nhau; thời gian giải xấp xỉ nhau (khoảng 2 giây trên mẫu).
- Kết luận: điều kiện số học kém ở (A) và (B) cho kết quả như nhau ở quy mô này, vì dung sai của solver là tuyệt đối. Nên dùng (C) (`v_exp_profit_m`).

### 7.2 Ràng buộc nào chi phối, ràng buộc nào còn dư

- **Binding:** C4 (150.748 triệu so với giới hạn 150.871 triệu) và C5 (1.046 khoản so với tối thiểu 1.045,9).
- **Còn dư địa, mỗi luật vì một lý do khác nhau:**
  - C1: rủi ro 14,4% so với 22% (7,6 điểm dư). Mục tiêu lợi nhuận vốn đã chọn khoản PD thấp (vì một vụ vỡ nợ lỗ gấp nhiều lần lãi của khoản trả đủ); chính trần không ràng buộc đã có rủi ro 14,6%.
  - C3: 335 so với 350 tỷ. Trần không ràng buộc chỉ giải ngân 324 tỷ.
  - C6: khoản lãi cao chỉ 8,5% số khoản duyệt so với 20%.
  - C2: 48,5% so với 45%. Theo lợi nhuận thuần, tỷ lệ duyệt là 44,4%; C5 ép duyệt thêm micro và nâng tỷ lệ lên 48,5% (+4 điểm).
- **C7** (36 luật):
  - 4 luật mang cờ binding chỉ vì cả hai ô cùng duyệt 100% hoặc 0%.
  - Số luật binding thật là C4 và C5.

### 7.3 Profit bridge

- BAU **11,98 tỷ** → trần **61,18 tỷ** (+49,20 tỷ, +411%, ×5,1) → tối ưu có ràng buộc **60,89 tỷ** (−0,29 tỷ).
- Gần như toàn bộ giá trị đến từ chính sách tín dụng: bỏ các khoản làm mất tiền.
- Bảy luật kinh doanh tốn 0,47% của trần.

### 7.4 Giá bóng (shadow price; giải lại toàn bộ, một ngưỡng đổi một bước)

| Luật | Nới một bước | Siết một bước |
|---|---|---|
| **C4** (±1 điểm phần trăm) | **+0,100 tỷ** | **−0,164 tỷ** |
| **C5** (±1 điểm) | **+0,030 tỷ** | **−0,047 tỷ** |
| C1 (1 điểm), C2 (1 điểm), C3 (10 tỷ), C6 (1 điểm) | 0 | 0 |

Con số cho người ra quyết định:

- Một điểm phần trăm khẩu vị rủi ro (C1) hiện đáng giá 0 đồng.
- Luật tốn tiền là trần kỳ hạn dài (C4): khoảng 0,10 tỷ mỗi điểm.
- Siết tốn nhiều hơn nới: hình dạng thường gặp, vì những hoán đổi tốt nhất được làm trước.

### 7.5 Đường biên hiệu quả (mẫu 25%, quy đổi)

- **Giới hạn rủi ro (C1):** vô nghiệm từ 8% trở xuống, 56,7 tỷ ở 10%, 60,9 tỷ ở 12%, 61,9 tỷ ở 14%, phẳng khoảng 62,0 tỷ từ 16%. Mức 22% nằm sâu trong vùng phẳng.
- **Ngân sách (C3):** phẳng từ khoảng 325 tỷ, 60,7 tỷ ở 300 tỷ, 57,7 tỷ ở 275 tỷ, 53,2 tỷ ở 250 tỷ.
- **Ngưỡng duyệt tối thiểu (C2):** phẳng đến 50%, rồi 61,1 tỷ ở 55%, 58,2 tỷ ở 60%, 53,4 tỷ ở 65%, 45,9 tỷ ở 70%.

### 7.6 Giá của C5 và C6, và của từng luật chạy riêng

- C6 tốn 0 vì còn dư địa.
- **C5 tốn 0,055 tỷ VND** (0,09% lợi nhuận): ép thêm **286 khoản micro** (761 lên 1.047), tức khoảng **0,19 triệu VND lợi nhuận kỳ vọng mất mỗi khoản thêm**.
- Mỗi luật chạy riêng (mẫu, quy đổi): C4 0,40 tỷ, C5 0,09 tỷ, C7 0,03 tỷ, C2 0,004 tỷ, C1/C3/C6 bằng 0.
  - Tổng các luật riêng lẻ 0,52 tỷ so với 0,44 tỷ khi chạy cùng nhau: các luật chồng lấn, nên chi phí chung nhỏ hơn tổng chi phí riêng lẻ.
- **Lập luận về C5:**
  - Ủng hộ trả giá: rẻ; giữ cấp tín dụng cho nhóm chiếm 20% đơn; khách micro có thể thành khách lớn sau này (giá trị này mô hình chưa tính).
  - Phản đối: đây đúng là nhóm có kinh tế tệ nhất, và mô hình PD kém chính xác nhất ở đó.
  - Kết luận: giữ C5 vì hóa đơn nhỏ và cố định; luật nên đàm phán lại là C4, đắt gấp khoảng 3,5 lần mỗi điểm (0,164 so với 0,047 tỷ khi siết một điểm).

### 7.7 Căng thẳng giữa C1 và C2

- Đề dự đoán C1 (rủi ro ≤ 22%) và C2 (duyệt ≥ 45%) mâu thuẫn; kết quả cho thấy hai luật cùng thỏa với dư địa lớn:
  - Giữ duyệt ≥ 45% và các luật còn lại: rủi ro thấp nhất đạt được là **8,7%** (giới hạn 22%).
  - Giữ rủi ro ≤ 22%: tỷ lệ duyệt cao nhất đạt được là **76,5%** (sàn 45%).
- Đây là kết quả bisection trên mẫu, sai số khoảng 0,1 điểm.

### 7.8 Giá trị của mô hình PD

- Làm nhiễu logit của PD (độ lệch chuẩn 1,39): Gini giảm từ 0,701 xuống 0,551 (`TARGET` chỉ dùng để **đo** Gini; quyết định dùng PD).
- Tối ưu lại với PD nhiễu, rồi đánh giá chính sách đó bằng PD tốt:

| | PD của Challenge 1 | PD nhiễu |
|---|---|---|
| Gini | 0,701 | 0,551 |
| Lợi nhuận mà optimizer tin | 60,92 tỷ | 62,14 tỷ |
| **Lợi nhuận kỳ vọng thật (PD tốt)** | **60,92 tỷ** | **50,52 tỷ** |
| Tỷ lệ duyệt | 48,5% | 46,5% |
| Rủi ro (PD tốt) | 14,4% | 17,8% |
| Rule binding (C1 đến C6) | C4, C5 | C4, C5 |

- **Giá trị của mô hình tốt: 10,40 tỷ VND lợi nhuận kỳ vọng (17,1% của tối ưu)**, khoảng 125 tỷ mỗi năm nếu dòng đơn hàng tháng lặp lại (giả định).
- Optimizer dùng PD nhiễu còn tự ảo tưởng thêm **11,6 tỷ** (lời nguyền của người thắng cuộc: chọn đúng các khoản có nhiễu trông đẹp).
- **Tập luật binding giữ nguyên (C4, C5):**
  - Đề kỳ vọng mô hình tốt hơn đẩy luật rủi ro ra khỏi biên.
  - C1 còn dư với mô hình tốt (14,4%) và còn dư với mô hình yếu (17,8% tính bằng PD tốt, dưới 22%).
  - Giá trị của mô hình nằm ở lợi nhuận.
  - Giả định: PD của Challenge 1 gần sự thật hơn PD nhiễu; tập kiểm thử bằng kết quả thực kiểm tra giả định này.

### 7.9 Độ nhạy của giả định lợi nhuận

| Giả định | Lợi nhuận (theo chính nó) | Tỷ lệ duyệt | Quyết định đổi | Regret theo giả định gốc |
|---|---|---|---|---|
| Gốc: thu hồi 20%, vỡ nợ tại 35% kỳ hạn | 60,92 tỷ | 48,5% | — | — |
| Thu hồi 10% | 57,51 tỷ | 46,3% | 2,3% | 0,21 tỷ |
| Thu hồi 30% | 64,83 tỷ | 50,6% | 2,2% | 0,21 tỷ |
| Vỡ nợ tại 25% kỳ hạn | 56,14 tỷ | 45,3% | 3,3% | 0,44 tỷ |
| Vỡ nợ tại 50% kỳ hạn | 70,83 tỷ | 50,2% | 4,4% | 0,80 tỷ |

**Chính sách ổn định, mức lợi nhuận thay đổi theo giả định:**

- Đổi thu hồi 10 điểm phần trăm: đổi khoảng 2% quyết định, regret 0,21 tỷ (0,3% lợi nhuận).
- Thời điểm vỡ nợ nhạy hơn: tới 4,4% quyết định, regret 0,80 tỷ (1,3%); đây là giả định nên làm chắc trước.
- Tập luật binding luôn có C4 và C5; C3 (ngân sách) thêm vào khi giả định làm khoản vay sinh lời hơn.

## 8. Chưng cất thành chính sách dạng luật

- 17.976 quyết định riêng lẻ chưa là chính sách:
  - Khó triển khai, kiểm toán và giải thích.
  - Khó áp dụng cho đơn tháng sau.

### 8.1 Phương pháp (`pydo.DecisionTree`)

1. Trồng CART trên **các cột thô** (`RISK_BAND`, `PRODUCT_GROUP`, `TENOR_BUCKET`, `AMOUNT_M`, `INTEREST_RATE`) để dự đoán quyết định của nghiệm tối ưu cấp tài khoản.
   - Mỗi đơn có trọng số bằng mức đáng giá của quyết định (|lợi nhuận nếu duyệt − nếu từ chối|), để cây dùng các lần chia vào nơi có tiền.
2. Chỉ giữ **phân hoạch** của cây (mỗi lá là một nhóm đơn).
3. **Tối ưu lại một hành động chung cho mỗi lá** với **đúng bộ ràng buộc cũ**:
   - Đây là một bài MILP nhỏ, mỗi biến là một lá.
   - Kết quả: tập luật nhỏ thỏa mọi ràng buộc, hoặc báo lỗi rõ "phân hoạch quá thô: cần cây sâu hơn".

### 8.2 Chọn cỡ cây

| Cây | Số lá | Lợi nhuận kỳ vọng | Giữ lại so với tối ưu cấp tài khoản |
|---|---|---|---|
| độ sâu 4, ngưỡng chia mặc định của template | 11 | 53,40 tỷ | 87,7% |
| độ sâu 6, ngưỡng chia mặc định | 17 | 55,73 tỷ | 91,5% |
| độ sâu 5, ngưỡng chia 0 | 27 | 55,22 tỷ | 90,6% |
| **độ sâu 6, ngưỡng chia 0 (được chọn)** | **32** | **59,31 tỷ** | **97,35%** |

- Ngưỡng chia mặc định của template (`min_gini_gain = 0,0001`):
  - Ngăn CART chia theo `PRODUCT_GROUP`, đúng chiều mà C5 và C7 được viết.
  - Các lá khó định hình để thỏa luật với chi phí thấp; chi phí giữ ở mức 9 đến 12%.
- Đặt ngưỡng bằng 0: chia theo sản phẩm xuất hiện, chi phí xuống 2,65%.
- Quy tắc chọn cỡ, đặt trước: **cây khả thi nhỏ nhất có chi phí tối đa 3%**.
- Chính sách là một **bảng** 32 luật. Cây 17 lá dễ đọc hơn nhưng tốn 5,2 tỷ (8,5%).
- Dùng `PD` liên tục thay cho dải: 97,42% so với 97,35%. Kết quả ngang nhau, và luật dải cố định dễ triển khai và giám sát hơn.

### 8.3 Chi phí của việc đơn giản hóa

**Đơn giản hóa 17.976 quyết định thành 32 luật tốn 1,61 tỷ VND, tức 2,65% nghiệm tối ưu cấp tài khoản** (59,31 tỷ so với 60,92 tỷ).

- Hành động đa số của mỗi lá:
  - Giữ 59,66 tỷ (97,9%) nhưng vi phạm 3 trong 42 luật.
  - Tối ưu lại hành động theo lá tốn thêm 0,35 tỷ và thỏa cả 42 luật.
- Mất mát **phân tán**: lá tệ nhất chiếm 15% tổng mất mát, năm lá tệ nhất khoảng 54%. Cây sâu hơn chỉ thu hồi được một phần nhỏ của mất mát ở mỗi nhánh.
- **Khái quát tốt:**
  - Đồng thuận với quyết định cấp tài khoản 92,9% (tập train của cây) và 92,1% (tập test); lợi nhuận giữ lại 97,6% và 96,9%.
  - Chỉ tháng 5 có PD, nên kiểm tra bằng chia ngẫu nhiên theo đơn.

### 8.4 Cây dùng và hướng của các biến

- Cây dùng cả 5 biến.
- SHAP trung bình tuyệt đối: `RISK_BAND` 0,37 (áp đảo), kỳ hạn (`T_SHORT`) 0,13, `INTEREST_RATE` 0,03, `AMOUNT_M` 0,003, `PRODUCT_GROUP` 0,0005.

**Hướng của từng biến** (khớp với kinh tế khoản vay):

- Rủi ro càng cao càng bị từ chối (tương quan duyệt/dải −0,83).
- Kỳ hạn ngắn bị từ chối nhiều (tỷ lệ duyệt: 29% `T_SHORT`, 57% `T_MID`, 81% `T_LONG`): hiệu ứng chi phí cố định.
- Khoản lớn được duyệt nhiều hơn (+0,23): chi phí cố định được pha loãng.
- Lãi suất:
  - Tương quan thô với duyệt là −0,35: trông ngược nhưng bị nhiễu (lãi cao đi kèm PD cao).
  - **Trong cùng dải rủi ro và kỳ hạn**, các lá duyệt khi lãi trên ngưỡng và từ chối khi dưới: lãi bù đắp rủi ro đúng như mong đợi.
- `PRODUCT_GROUP` có trọng số SHAP gần 0 nhưng cần cho phân hoạch (C5 và C7 viết theo sản phẩm); thông tin sản phẩm đã nằm trong kỳ hạn, ticket và lãi.
- Mọi biến đẩy đúng hướng kinh tế, nên dữ liệu và mô hình lợi nhuận nhất quán.

### 8.5 Chính sách cuối cùng chấm bằng cùng một bộ đánh giá

- Cây thỏa **cả 42 luật**:
  - Duyệt 52,2%; rủi ro 14,6%; giải ngân **347,6 tỷ** (giới hạn 350); kỳ hạn dài **44,6%** (giới hạn 45%); micro 12,9%; lãi cao 8,5%.
  - Cây duyệt nhiều hơn nghiệm cấp tài khoản (52,2% so với 48,5%) và dùng gần hết ngân sách vì phải quyết định cả lá: khả thi nhưng ít dư địa hơn.
- So với **ngưỡng PD đơn tốt nhất** (PD ≤ 0,362, chỉ dùng PD):
  - Lợi nhuận 54,94 tỷ (90% tối ưu) và vi phạm luật ngân sách (354 tỷ).
  - Biết sản phẩm, kỳ hạn và lãi đáng khoảng 4,4 tỷ so với một ngưỡng điểm đơn thuần.

## 9. Tập kiểm thử bằng kết quả thực tế

- Chỉ ở đây `TARGET` được dùng.
- Với mỗi khoản được duyệt, lợi nhuận thực tế là `G` nếu `TARGET = 0` (trả đủ) và `D` nếu `TARGET = 1` (vỡ nợ).

| | Kỳ vọng | Thực hiện | Chênh |
|---|---|---|---|
| BAU (duyệt hết) | 11,98 tỷ | 10,57 tỷ | −1,41 tỷ (−11,8%) |
| Trần không ràng buộc | 61,18 tỷ | 60,33 tỷ | −0,85 tỷ (−1,4%) |
| Tối ưu cấp tài khoản | 60,92 tỷ | 60,30 tỷ | −0,63 tỷ (−1,0%) |
| **Cây 32 luật** | **59,31 tỷ** | **58,28 tỷ** | **−1,03 tỷ (−1,7%)** |

- Chính sách cây thu thực tế **58,28 tỷ, gấp 5,5 lần BAU**.
- Tỷ lệ rủi ro thực tế **15,0%** (kỳ vọng 14,6%, giới hạn 22%): thỏa luật rủi ro ngoài đời thực.

**Quy kết chênh lệch** (tách thành hai nguồn):

- **Mức PD:**
  - Cách làm: nhân mọi PD với **một hệ số chung** để PD trung bình bằng tỷ lệ vỡ nợ thực tế của sổ được duyệt (giữ nguyên thứ hạng).
  - Lợi nhuận kỳ vọng đổi **−0,58 tỷ** với nghiệm cấp tài khoản (PD trung bình 13,6% so với 13,9% thực tế).
  - Lợi nhuận kỳ vọng đổi **−1,40 tỷ** với cây (14,2% so với 14,7%).
  - Kết luận: mô hình Challenge 1 hơi lạc quan 0,2 đến 0,5 điểm phần trăm.
- **Thứ hạng, cơ cấu, ngẫu nhiên:** phần còn lại là **−0,04 tỷ** và **+0,37 tỷ**; phần này là nhiễu.
- **Có phải ngẫu nhiên?**
  - Mô phỏng Monte Carlo (vẽ `TARGET` từ chính PD của ta): 1 độ lệch chuẩn của lợi nhuận thực khoảng **1,1 tỷ**.
  - Lợi nhuận thực nằm ở phân vị 31% (cấp tài khoản) và 21% (cây).
  - Các chênh lệch (0,5 và 0,9 độ lệch chuẩn) phù hợp với PD đúng và may rủi bình thường.
  - Dữ liệu một tháng chưa đủ để tách PD thấp 0,3 điểm khỏi nhiễu.
- **Giả định lợi nhuận:**
  - Làm mức lợi nhuận của cùng chính sách đổi nhiều (kỳ vọng 53,9 đến 68,3 tỷ, thực hiện 52,7 đến 67,5 tỷ).
  - Chênh lệch giữa kỳ vọng và thực hiện giữ ở −0,8 đến −1,2 tỷ.
  - Bất định về giả định đổi **mức** lợi nhuận; sai số PD đổi **chênh lệch**; cái đầu lớn hơn.

## 10. Sản phẩm bàn giao

### `optimal_policy.csv`

- Một file, 81 dòng, 17 cột.
- Cột `record_type` phân biệt ba khối: `leaf_rule` (32 dòng), `portfolio_kpi`, và `constraint` (một dòng cho mỗi luật trong 42 luật).

| Cột | Ý nghĩa |
|---|---|
| `record_type`, `name` | Loại dòng và tên (`leaf 0`, tên KPI, tên luật) |
| `rule_description` | Luật dạng chữ (với `leaf_rule`) hoặc luật ràng buộc (với `constraint`) |
| `RISK_BAND`, `PRODUCT_GROUP`, `TENOR_BUCKET` | Điều kiện của lá trên ba chiều này (`any` nếu lá bao mọi giá trị) |
| `other_conditions` | Điều kiện trên `AMOUNT_M` và `INTEREST_RATE` |
| `action` | `approve` hoặc `reject` |
| `n_apps`, `approved_n` | Số đơn trong lá và số đơn được duyệt |
| `approved_volume_m`, `exp_bad_volume_m` | Số tiền giải ngân và khối lượng vỡ nợ kỳ vọng (triệu VND) |
| `exp_profit` | Lợi nhuận kỳ vọng (VND) |
| `value`, `limit`, `slack`, `satisfied` | Với `portfolio_kpi` và `constraint`: giá trị, giới hạn, độ dư và trạng thái |

### `constrained_output.csv`

- Export cấp tài khoản của solver (17.976 dòng): quyết định và các biến đầu ra cho từng đơn.

### `models/clx_approval_tree.json` và `.sql`

- Chính sách cuối cùng dạng luật.
- JSON: đọc độc lập với scikit-learn.
- SQL: biểu thức `CASE` cho bộ máy luật nói SQL.
- Định dạng trao đổi cây quyết định chuyên dụng (`.fjdt`) chưa được `pydo` hỗ trợ; SQL thay thế.

### Kiểm tra ràng buộc trên chính các file đã lưu

- Bộ kiểm tra **viết độc lập với solver** (chỉ biết bảy luật của đề), làm ba việc:
  - Đọc lại `constrained_output.csv`.
  - Nạp lại cây từ JSON rồi áp cho toàn bộ 17.976 đơn.
  - Đối chiếu `optimal_policy.csv` (tổng số đơn khớp, số duyệt khớp, mọi dòng ràng buộc thỏa).
- Cả ba thỏa đủ 7 luật.
- `pydo` chưa hỗ trợ `project.save(...)` (lưu cả dự án thành zip).

## 11. Khuyến nghị

1. **Chính sách:**
   - Duyệt đơn từ các dải rủi ro thấp.
   - Chọn lọc hơn ở kỳ hạn ngắn (chi phí cố định ăn hết biên lợi nhuận) và ở khoản rủi ro cao, trừ khi lãi suất bù đắp.
   - Giữ ít nhất 12% số khoản duyệt là micro; nằm trong ngân sách 350 tỷ và trần 45% kỳ hạn dài.
   - Áp dụng bằng **32 luật**: duyệt khoảng **52%** đơn, giải ngân **348 tỷ VND**, rủi ro kỳ vọng **14,6%**, thỏa cả bảy luật.
2. **Giá trị so với duyệt hết:**
   - **+47,3 tỷ VND lợi nhuận kỳ vọng** (59,31 so với 11,98 tỷ, ×5,0).
   - **+47,7 tỷ thực hiện** (58,28 so với 10,57 tỷ, ×5,5).
   - Gần như toàn bộ đến từ việc bỏ các khoản làm mất tiền; nghiệm tối ưu cấp tài khoản sẽ thêm 1,6 tỷ nữa.
3. **Luật nên cân nhắc nới: C4 (trần kỳ hạn dài):**
   - Mỗi điểm phần trăm giới hạn đáng khoảng 0,10 tỷ (siết một điểm mất 0,16 tỷ).
   - Chính sách trần nằm ở 47,9% so với trần 45%; nới tới khoảng 48% thu hồi gần hết 0,29 tỷ mà bảy luật tốn.
   - **C5 là bảo hiểm rẻ:** 0,055 tỷ tổng cộng, khoảng 0,19 triệu mỗi khoản micro thêm.
   - **C1, C2, C3, C6 hiện không tốn gì:** có thể hạ giới hạn rủi ro từ 22% xuống khoảng 16% mà giữ nguyên lợi nhuận. Mục tiêu lợi nhuận mới là thứ bảo vệ danh mục; luật rủi ro hiện chưa tác động.
4. **Chi phí triển khai (chưng cất):**
   - 1,61 tỷ VND, 2,65% của tối ưu cấp tài khoản.
   - Kỳ vọng 59,31 so với 60,92 tỷ; thực hiện 58,28 so với 60,30 tỷ, trong khi một độ lệch chuẩn ngẫu nhiên là 1,1 tỷ.
   - Cây 17 luật tốn 5,2 tỷ (8,5%): giá của một trang ngắn hơn.
5. **Điểm yếu lớn nhất:**
   - (a) Con số lợi nhuận dựa trên giả định kinh tế tổng hợp: chính sách ổn định (đổi 2 đến 4% quyết định) nhưng mức lợi nhuận thì không (khoảng 54 đến 68 tỷ cho cùng chính sách); thời điểm vỡ nợ cần làm chắc trước.
   - (b) Một tháng và một mô hình PD: PD hơi lạc quan 0,2 đến 0,5 điểm lẫn trong may rủi, và cây chưa được kiểm tra ngoài thời gian.
   - (c) Ít dư địa: cây dùng 347,6 trên 350 tỷ và 44,6% trên 45%. Một thay đổi cơ cấu đơn có thể đẩy nó vượt luật, nên cần tối ưu lại và kiểm tra lại hằng tháng.
   - (d) Các ô thưa: luật C7 của `CLX_LARGE` dựa trên ô 17 đến 29 đơn; decile PD an toàn nhất bị dự báo 2,0% so với 0,7% thực, nên khoản an toàn nhất bị định giá thấp chút.
   - (e) Công cụ là `pydo` tự xây: được kiểm tra bằng vét cạn và bộ kiểm tra độc lập nhưng chưa được đối chiếu với một công cụ tối ưu khác. Nên tái tạo nghiệm tối ưu cấp tài khoản trên một công cụ độc lập trước khi đưa vào vận hành.

## 12. Nhật ký quyết định và giả định

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Công cụ | `pydo` tự xây | Kiểm soát toàn bộ logic tối ưu và kiểm thử độc lập được |
| Ngưỡng ràng buộc | Cố định theo đề | Đề yêu cầu giữ nguyên |
| C3 | Viết theo đầu đơn | Bất biến khi chạy mẫu |
| C7 | Theo sản phẩm, bỏ cặp có ô rỗng, giữ ô thưa | Tránh nhân quá nhiều luật; mọi dải đều có chốt đơn điệu |
| Dải PD | 10 dải ngưỡng cố định | Giám sát được, ổn định theo tháng |
| Đơn vị | Số tiền giải ngân theo triệu | Thu hẹp dải hệ số của ma trận (so sánh VND với triệu) |
| Gap solver | 1% cho kết quả chính, 0,01% cho các phép quét | Kết quả chính theo đề; phép quét cần sai số nhỏ hơn hiệu ứng |
| Giá trị mô hình PD | Đánh giá bằng kỳ vọng theo PD tốt | `TARGET` chỉ dùng ở tập kiểm thử |
| Cỡ cây | Nhỏ nhất có chi phí ≤ 3% | Quy tắc đặt trước |
| Ngưỡng chia của cây | 0 (đưa vào lưới) | Mặc định của template chặn chia theo sản phẩm |
| Định dạng triển khai | JSON và SQL | `pydo` chưa có exporter `.fjdt` |

## 13. Hạn chế và việc tiếp theo

**Hạn chế**

- Kinh tế là số tổng hợp; một tháng dữ liệu; PD của một mô hình.
- Dữ liệu PD chỉ có tháng 5, nên cây chỉ được kiểm tra trên mẫu cùng thời gian.
- Cây dùng gần hết dư địa của C3 và C4.
- Nghiệm mới được kiểm tra bằng `pydo` và bộ kiểm tra độc lập; cần đối chiếu với một công cụ tối ưu khác.
- Các phép quét nhiều lần chạy trên mẫu 25% và cao hơn bản đầy đủ khoảng 1,8%.
- Thử nghiệm thu hồi và thời điểm vỡ nợ đổi giả định cho toàn bộ sổ, chưa chia theo sản phẩm.

**Với thêm thời gian**

1. Làm chắc thời điểm vỡ nợ và thu hồi từ dữ liệu thực.
2. Tối ưu lại và kiểm tra ràng buộc hằng tháng; thêm biên an toàn cho C3 và C4.
3. Đối chiếu nghiệm với một công cụ tối ưu khác.
4. Kiểm tra cây trên tháng kế tiếp khi có PD.
5. Thử luật C7 chia theo cả kỳ hạn, với mẫu đủ lớn.
6. Chia lợi nhuận vỡ nợ theo sản phẩm (thu hồi khác nhau).

## 14. Tái lập kết quả

1. Chạy notebook 01 để có `results/pd_predictions.csv`.
2. Mở `notebooks/02_portfolio_optimization.ipynb`:
   - Kernel: `pydo-project`.
   - Thư mục làm việc: `notebooks/`.
   - `pip install -r requirements.txt` cài `pydo` ở chế độ chỉnh sửa (notebook `import pydo`).
3. Chạy toàn bộ từ trên xuống (khoảng 11 phút trên toàn bộ danh mục).
   - Notebook tạo: `notebooks/functions/`, `results/models/`, `results/constrained_output.csv`, `results/optimal_policy.csv`.
4. Kiểm tra thư viện: `python -m unittest discover -s tests`.

**Cấu trúc notebook:**

- 0: môi trường.
- 1: danh mục.
- 2: kinh tế (2b: kiểm tra độc lập).
- 3: dựng dự án.
- 4: BAU.
- 5: trần.
- 6: mô hình có ràng buộc.
- 7: chẩn đoán cơ bản.
- 7b: trả lời năm câu hỏi (điều kiện số học, binding, profit bridge, giá bóng, đường biên, chi phí C5 và C6, căng thẳng C1 và C2, giá trị mô hình PD, độ nhạy).
- 8: chưng cất.
- 9: tập kiểm thử.
- 10: sản phẩm bàn giao.
- 11: khuyến nghị.
