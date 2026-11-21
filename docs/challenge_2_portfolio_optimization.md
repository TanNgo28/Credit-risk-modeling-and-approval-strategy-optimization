# Challenge 2: tối ưu chính sách duyệt/từ chối

Notebook: `notebooks/02_portfolio_optimization.ipynb`. Công cụ: thư viện `pydo` (xem [pydo.md](pydo.md)).
Đầu vào: `results/pd_predictions.csv` từ Challenge 1. Kết quả bàn giao (trong `results/`): `optimal_policy.csv`, `constrained_output.csv`, `models/clx_approval_tree.json`, `models/clx_approval_tree.sql`.

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Bài toán](#2-bài-toán)
3. [Đầu vào và các cột dẫn xuất](#3-đầu-vào-và-các-cột-dẫn-xuất)
4. [Kinh tế của một khoản vay](#4-kinh-tế-của-một-khoản-vay)
5. [Mô hình hóa: biến, ràng buộc, ba flow](#5-mô-hình-hóa-biến-ràng-buộc-ba-flow)
6. [Kết quả của ba flow](#6-kết-quả-của-ba-flow)
7. [Chẩn đoán](#7-chẩn-đoán)
8. [Chưng cất thành chính sách dạng luật](#8-chưng-cất-thành-chính-sách-dạng-luật)
9. [Tập kiểm thử bằng kết quả thực tế](#9-tập kiểm thử-bằng-kết-quả-thực-tế)
10. [Sản phẩm bàn giao](#10-sản-phẩm-bàn-giao)
11. [Khuyến nghị](#11-khuyến-nghị)
12. [Nhật ký quyết định và giả định](#12-nhật-ký-quyết-định-và-giả-định)
13. [Hạn chế và việc tiếp theo](#13-hạn-chế-và-việc-tiếp-theo)
14. [Tái lập kết quả](#14-tái-lập-kết-quả)

---

## 1. Tóm tắt

- Quyết định: duyệt hay từ chối từng đơn trong 17.976 đơn tháng 5/2015, để tối đa hóa lợi nhuận kỳ vọng dưới 7 ràng buộc kinh doanh.
- **Chính sách đề xuất: cây 32 luật** (theo dải rủi ro, kỳ hạn, lãi suất, ticket), giữ **97,35%** nghiệm tối ưu cấp tài khoản, thỏa cả 7 ràng buộc.
- Lợi nhuận kỳ vọng so với duyệt hết (BAU): **59,31 tỷ so với 11,98 tỷ (×5,0)**. Thực tế (tập kiểm thử): **58,28 tỷ so với 10,57 tỷ (×5,5)**.
- Hai ràng buộc thực sự chi phối: **C4 (kỳ hạn dài)** và **C5 (micro ≥ 12%)**. Rule nên cân nhắc nới là C4.
- Giá trị của mô hình PD (Challenge 1) trong bài toán này: **10,4 tỷ VND** lợi nhuận kỳ vọng (17%).
- Công thức kinh tế là giả định tổng hợp của đề, không phải số liệu của một danh mục thật.

| Chính sách | Lợi nhuận kỳ vọng | Rule vi phạm (trên 42) |
|---|---|---|
| BAU (duyệt hết) | 11,98 tỷ | 2 |
| Ngưỡng PD đơn (PD ≤ 0,362) | 54,94 tỷ | 1 |
| Trần không ràng buộc | 61,18 tỷ | 4 |
| Tối ưu cấp tài khoản | 60,92 tỷ | 0 |
| **Cây 32 luật** | **59,31 tỷ** | **0** |

## 2. Bài toán

**Quyết định:** với mỗi đơn, duyệt (1) hay từ chối (0). Một biến quyết định `APPROVE_ACTION` ∈ {0, 1}, hai phương án mỗi đơn.

**Mục tiêu:** tối đa hóa tổng lợi nhuận kỳ vọng của danh mục được duyệt, tính theo PD (không dùng `TARGET`).

**Bảy ràng buộc (ngưỡng cố định, không được nới):**

| # | Ràng buộc | Giá trị |
|---|---|---|
| C1 | Tỷ lệ rủi ro = khối lượng vỡ nợ kỳ vọng / dư nợ giải ngân | ≤ 22% |
| C2 | Tỷ lệ duyệt = số đơn duyệt / tổng số đơn | ≥ 45% |
| C3 | Tổng giải ngân | ≤ 350 tỷ VND |
| C4 | Tỷ trọng dư nợ có kỳ hạn ≥ 36 tháng | ≤ 45% |
| C5 | Tỷ trọng **số khoản** duyệt thuộc `CLX_MICRO` | ≥ 12% |
| C6 | Tỷ trọng **số khoản** duyệt có lãi suất > 40% | ≤ 20% |
| C7 | Tỷ lệ duyệt không tăng khi rủi ro tăng, trong từng nhóm sản phẩm | — |

**Luật chơi của đề:** tối ưu theo `PD`, không bao giờ theo `TARGET` (chỉ dùng ở tập kiểm thử cuối); không được nới ngưỡng để đạt khả thi; phải nêu rõ rằng các tham số hợp đồng và lợi nhuận là số tổng hợp của đề.

## 3. Đầu vào và các cột dẫn xuất

- 17.976 đơn tháng 5/2015 ghép với `pd_predictions.csv` (cho đúng 1 dòng PD mỗi đơn).
- Các cột hợp đồng dùng ở đây (bị cấm ở Challenge 1): `AMOUNT_VND` (5 triệu đến 250 triệu), `TENOR_MONTHS` (3 đến 60), `INTEREST_RATE` (0,16 đến 0,50), `MONTHLY_INSTALLMENT`, `PRODUCT_GROUP`.
- Kiểm tra dữ liệu: không có lãi suất bằng 0; `MONTHLY_INSTALLMENT` khớp đúng công thức annuity với lãi suất/12 (sai số 0), nhưng được làm tròn tới đồng.
- Nếu duyệt hết, tổng dư nợ là 574,4 tỷ (so với ngân sách 350 tỷ).
- Cơ cấu sản phẩm: `CLX_SMALL` 9.777 đơn, `CLX_MID` 4.257, `CLX_MICRO` 3.619, `CLX_LARGE` 323.

**Các cột thô được tạo thêm** (chỉ những cột này mới được dùng trong điều kiện `WHERE` của ràng buộc, vì mẫu số của các phép tổng hợp phải cố định trước khi chạy solver):

| Cột | Định nghĩa |
|---|---|
| `RISK_BAND` | 10 dải PD theo ngưỡng cố định: [0; 0,05), [0,05; 0,10), [0,10; 0,15), [0,15; 0,20), [0,20; 0,25), [0,25; 0,32), [0,32; 0,40), [0,40; 0,50), [0,50; 0,65), [0,65; 1,01) |
| `TENOR_BUCKET` | `T_SHORT` (≤ 11 tháng), `T_MID` (12 đến 35), `T_LONG` (≥ 36) |
| `HIGH_IR_FLAG` | 1 nếu lãi suất > 40% |
| `AMOUNT_M` | `AMOUNT_VND` / 10^6 (đơn vị triệu) |
| `APPROVE_ACTION_BAU` | Cột toàn 1: chính sách "duyệt hết" cho flow `simulation` |

**Vì sao ngưỡng cố định thay vì phân vị:** phân vị dịch chuyển theo cơ cấu đơn của tháng sau; một chính sách có định nghĩa dải trôi theo tháng thì không giám sát được. Dải hẹp (5 điểm) ở vùng PD thấp, nơi lợi nhuận được quyết định; rộng hơn từ 32%, nơi gần như mọi khoản vay đều lỗ. Ngưỡng 0,25 và 0,32 bao quanh PD hòa vốn trung vị (khoảng 24%).
Kích cỡ dải khỏe (nhỏ nhất 1.090 đơn, 6,1%). 7 trong 40 ô (sản phẩm × dải) có dưới 30 đơn, tất cả thuộc `CLX_LARGE` (dải 2 đến 8).

## 4. Kinh tế của một khoản vay

Hàm tính nằm ở `notebooks/functions/flatten.py` (đọc trường thô và giả định vào các thuộc tính có tiền tố) và `notebooks/functions/economics.py` (bốn bước: `amortisation`, `profit_good`, `profit_bad`, `metrics`). Quy ước tiền tố: `p_` trường thô, `a_` hành động, `f_` giả định (fact), `v_` giá trị tính được.

### 4.1 Giả định (số tổng hợp của đề)

| Tham số | Giá trị |
|---|---|
| Chi phí vốn | 9,0%/năm trên dư nợ bình quân |
| Chi phí thu hút khách | 300.000 VND mỗi khoản duyệt |
| Chi phí phục vụ | 25.000 VND mỗi khoản mỗi tháng |
| Thời điểm vỡ nợ | sau `0,35 × kỳ hạn` kỳ trả |
| Thu hồi | 20% dư nợ gốc tại thời điểm vỡ nợ |

### 4.2 Công thức (khoản vay annuity, r = lãi suất/12, I = số tiền trả hàng tháng, A = số tiền vay, n = kỳ hạn)

- Dư nợ sau m kỳ: `B(m) = A(1+r)^m − I((1+r)^m − 1)/r`, về 0 khi hết kỳ (nếu r = 0 thì trả đều). Dư nợ bình quân là trung bình của B(0)…B(n); vì trả nợ dồn về cuối kỳ nên lớn hơn nhiều so với A/2.
- **Nếu khoản vay trả đủ:**
  `profit_good = (I·n − A) − 9% · dư nợ bình quân · n/12 − 25.000·n − 300.000`
- **Nếu vỡ nợ** sau `m = max(round(0,35·n), 1)` kỳ:
  - gốc đã trả = `A − B(m)`; lãi đã thu = `I·m − gốc đã trả` (không phải `I·m`, vì một phần mỗi kỳ là gốc);
  - chi phí vốn và phục vụ chỉ chạy trong `m` tháng; chi phí vốn tính trên dư nợ bình quân của đoạn đời ngắn;
  - lỗ tín dụng = `B(m) · (1 − 20%)`;
  - `profit_bad = lãi đã thu − chi phí vốn − chi phí phục vụ − lỗ tín dụng − 300.000` (chi phí thu hút vẫn phải trả).
- **Lợi nhuận kỳ vọng nếu duyệt** = `(1 − PD)·profit_good + PD·profit_bad`. Nếu từ chối: mọi giá trị bằng 0.
- Các biến cho ràng buộc: `v_approved` (0/1), `v_disbursed_amount_m` (dư nợ, đơn vị triệu), `v_pd_disbursed` (PD, dùng làm đầu vào của trung bình có trọng số), `v_exp_bad_volume_m`, `v_exp_profit`. Thêm hai biến chỉ dùng cho thí nghiệm điều kiện số học: `v_disbursed_amount` (VND thô) và `v_exp_profit_m` (triệu).

### 4.3 Kiểm tra độc lập

Năm hợp đồng (nhỏ/lớn, ngắn/dài, lãi cao) được tính lại bằng lịch trả nợ **từng tháng** (`B_m = B_{m−1}(1+r) − I`, không dùng công thức đóng) và so với pipeline: mọi thành phần khớp (lãi, dư nợ bình quân, chi phí vốn, lỗ khi vỡ nợ, `profit_good`, `profit_bad`). Chênh lệch lớn nhất 34,7 đồng trên khoản 250 triệu, do `MONTHLY_INSTALLMENT` được làm tròn nên lịch trả còn dư vài đồng ở kỳ cuối. Ngưỡng của phép kiểm tra: 1 đồng mỗi tháng kỳ hạn.

### 4.4 Kinh tế cho thấy gì

Điểm hòa vốn `PD* = profit_good / (profit_good − profit_bad)`. Chỉ **44,4%** đơn có PD < PD* (lợi nhuận kỳ vọng dương), PD* trung vị 24,1%.

| Sản phẩm | Số đơn | PD* trung vị | PD trung vị | Tỷ lệ đơn có lãi |
|---|---|---|---|---|
| `CLX_LARGE` | 323 | 0,457 | 0,153 | 81% |
| `CLX_MID` | 4.257 | 0,387 | 0,210 | 68% |
| `CLX_SMALL` | 9.777 | 0,236 | 0,294 | 44% |
| `CLX_MICRO` | 3.619 | 0,075 | 0,395 | **16%** |

Lý do micro, kỳ hạn ngắn không sinh lời: với micro kỳ hạn ≤ 6 tháng (2.092 khoản), ticket trung vị 10,4 triệu, lãi trung vị chỉ 0,75 triệu, trong khi chi phí cố định (thu hút + phục vụ) là 0,38 triệu, tức **50% tiền lãi**. Trần lãi bị chặn bởi đời vay ngắn và ticket nhỏ, còn khi vỡ nợ (sau 35% kỳ hạn) vẫn còn một phần lớn gốc chưa trả (khoảng 70% với khoản micro ngắn kiểm tra ở trên): lỗ lớn gấp nhiều lần lãi của một khoản trả đủ. Với micro dưới 12 tháng, PD* là 4,8% nên chỉ 8% khoản đáng cho vay. Đây là gốc của mâu thuẫn với C5: đề bắt buộc cấp tín dụng đúng cho nhóm có kinh tế tệ nhất.

## 5. Mô hình hóa: biến, ràng buộc, ba flow

### 5.1 Hệ thống ràng buộc (viết bằng ngôn ngữ luật của `pydo`)

| # | Luật (rút gọn) |
|---|---|
| C1 | `WEIGHTED_AVG(v_pd_disbursed, v_disbursed_amount_m) <= 0.22` |
| C2 | `AVG(v_approved) >= 0.45` |
| C3 | `AVG(v_disbursed_amount_m) <= 350000 / 17976` (viết theo đầu đơn, xem 5.2) |
| C4 | `SUM(v_disbursed_amount_m) WHERE (TENOR_BUCKET = 'T_LONG') <= SUM(v_disbursed_amount_m) * 0.45` |
| C5 | `SUM(v_approved) WHERE (PRODUCT_GROUP = 'CLX_MICRO') >= SUM(v_approved) * 0.12` |
| C6 | `SUM(v_approved) WHERE (HIGH_IR_FLAG = 1) <= SUM(v_approved) * 0.20` |
| C7 | Với mỗi sản phẩm và mỗi cặp dải liền kề (lo, hi): `AVG(v_approved) WHERE (sản phẩm, dải lo) >= AVG(v_approved) WHERE (sản phẩm, dải hi)` |

Tổng cộng 42 luật (C1 đến C6 cộng 36 luật C7: 4 sản phẩm × 9 cặp dải). C7 chia theo sản phẩm, không chia thêm theo kỳ hạn (36 luật đã nhiều; chia thêm sẽ để lại nhiều ô chỉ vài đơn). Cặp bị bỏ nếu một trong hai ô rỗng; ô thưa nhưng không rỗng vẫn giữ.

### 5.2 Vì sao một số luật an toàn khi chạy mẫu

C1 (tỷ số hai tổng), C2 (trung bình), C4, C5, C6 (tỷ trọng viết dạng `SUM(...) <= SUM(...) * k`) và C7 (tỷ lệ duyệt theo ô) đều là **tỷ lệ**, nên có nghĩa như nhau trên mẫu 25% và trên toàn bộ. Chỉ C3 là số tuyệt đối, nên được viết theo đầu đơn (`AVG(...) <= 350.000/N_full`) để không bị lỏng gấp 4 lần khi chạy mẫu.

### 5.3 Ba flow và baseline

| Flow | Ý nghĩa |
|---|---|
| `simulation` | Đánh giá một chính sách cố định (ở đây: duyệt hết, đọc từ cột `APPROVE_ACTION_BAU`) |
| `unconstrained` | Phương án tốt nhất của từng đơn không có luật: duyệt đúng các khoản có lợi nhuận kỳ vọng dương (trần) |
| `constrained` | Tối ưu dưới 42 luật: bài toán nguyên nhị phân giải bằng HiGHS, một biến nhị phân mỗi đơn |

Kỳ vọng `simulation ≤ constrained ≤ unconstrained`; thứ tự này được assert trong notebook.

## 6. Kết quả của ba flow

| | BAU | Trần | Tối ưu có ràng buộc |
|---|---|---|---|
| Lợi nhuận kỳ vọng | 11,98 tỷ | 61,18 tỷ | 60,89 tỷ (60,92 tỷ khi giải tới gap 0,01%) |
| Tỷ lệ rủi ro (C1 ≤ 22%) | 32,9% | 14,6% | 14,4% |
| Tỷ lệ duyệt (C2 ≥ 45%) | 100% | 44,4% | 48,5% |
| Dư nợ (C3 ≤ 350 tỷ) | 574,4 tỷ | 324,5 tỷ | 335,3 tỷ |
| Kỳ hạn dài (C4 ≤ 45%) | 33,3% | 47,9% | 45,0% |
| Micro (C5 ≥ 12%) | 20,1% | 7,2% | 12,0% |
| Lãi cao (C6 ≤ 20%) | 18,0% | 7,7% | 8,5% |

- **BAU** vi phạm C1 và C3.
- **Trần** vi phạm 4 luật: C2 (44,4% < 45%), C4 (47,9% > 45%), C5 (7,2% < 12%) và một luật C7 của `CLX_MICRO` (dải 4 sang 5). Danh sách này là lý do cần mô hình có ràng buộc.
- **Tối ưu có ràng buộc** giữ **99,5% của trần**, nhỏ hơn trần 0,29 tỷ (0,47%). Solver: tối ưu chứng minh được, gap 0,05% (yêu cầu 1%), khoảng 32 giây, một nút duy nhất (nới lỏng LP đã nguyên).
- **Mẫu 25% so với toàn bộ:** lợi nhuận quy đổi lệch dưới 2% so với chạy toàn bộ (giữa +0,9% và +1,8% ở các lần chạy), tỷ lệ duyệt tương đương, mọi luật đều thỏa, kết luận không đổi; số cuối cùng luôn lấy từ chạy toàn bộ.

## 7. Chẩn đoán

Các phép quét nhiều lần dùng `Scenario.with_constraints` (đổi luật mà không tính lại kinh tế). Số tiêu đề lấy từ chạy toàn bộ; các phép quét nhiều lần (đường biên, chi phí từng luật riêng, bisection) chạy trên mẫu 25% rồi quy đổi, và cao hơn chạy toàn bộ khoảng 1,8%, nên chỉ đọc hình dạng.

### 7.1 Điều kiện số học

Cùng một mô hình ở ba cách chọn đơn vị: (A) mọi đại lượng bằng VND thô, (B) dư nợ theo triệu, mục tiêu theo VND (bản dùng cho kết quả chính), (C) cả hai theo triệu. Dải hệ số: 1,1e12 (A), 4,8e11 (B), **1,1e6 (C)**; chỉ (C) dưới ngưỡng cảnh báo 1e9. Quyết định **giống nhau** ở cả ba (khác dưới 0,05% số đơn, đúng bằng dung sai solver), lợi nhuận như nhau, thời gian giải xấp xỉ nhau (khoảng 2 giây trên mẫu). Vấn đề có thật nhưng vô hại ở quy mô này, vì dung sai của solver là tuyệt đối; khuyến nghị dùng (C) (`v_exp_profit_m`).

### 7.2 Ràng buộc nào chi phối, ràng buộc nào còn dư

- **Binding:** C4 (150.748 triệu so với giới hạn 150.871 triệu) và C5 (1.046 khoản so với tối thiểu 1.045,9).
- **Còn dư địa, mỗi luật vì một lý do khác nhau:**
  - C1: rủi ro 14,4% so với 22% (7,6 điểm dư). Mục tiêu lợi nhuận vốn đã chọn khoản PD thấp (vì một vụ vỡ nợ lỗ gấp nhiều lần lãi của khoản trả đủ); chính trần không ràng buộc đã có rủi ro 14,6%.
  - C3: 335 so với 350 tỷ. Trần không ràng buộc chỉ giải ngân 324 tỷ.
  - C6: khoản lãi cao chỉ 8,5% số khoản duyệt so với 20%.
  - C2: 48,5% so với 45%. Chỉ theo lợi nhuận thì chỉ duyệt 44,4%; phần tăng thêm 4 điểm đến từ C5 ép duyệt micro.
- **C7:** 36 luật, 4 luật được gắn cờ binding nhưng cả 4 là tầm thường (duyệt 100% hoặc 0% ở cả hai ô); không có luật nào thật sự chặt. Không nên đếm số luật `binding` thô.

### 7.3 Profit bridge

BAU **11,98 tỷ** → trần **61,18 tỷ** (+49,20 tỷ, +411%, ×5,1) → tối ưu có ràng buộc **60,89 tỷ** (−0,29 tỷ). Gần như toàn bộ giá trị là chính sách tín dụng: không cho vay các khoản làm mất tiền. Bảy luật kinh doanh tốn 0,47% của trần.

### 7.4 Giá bóng (shadow price; giải lại toàn bộ, một ngưỡng đổi một bước)

| Luật | Nới một bước | Siết một bước |
|---|---|---|
| **C4** (±1 điểm phần trăm) | **+0,100 tỷ** | **−0,164 tỷ** |
| **C5** (±1 điểm) | **+0,030 tỷ** | **−0,047 tỷ** |
| C1 (1 điểm), C2 (1 điểm), C3 (10 tỷ), C6 (1 điểm) | 0 | 0 |

Con số cho người ra quyết định: **một điểm phần trăm khẩu vị rủi ro (C1) hiện đáng giá 0 đồng**; luật tốn tiền là trần kỳ hạn dài, khoảng 0,10 tỷ mỗi điểm. Mức chênh giữa siết và nới là hình dạng thường gặp khi những hoán đổi tốt nhất được làm trước.

### 7.5 Đường biên hiệu quả (mẫu 25%, quy đổi)

- **Giới hạn rủi ro (C1):** vô nghiệm từ 8% trở xuống, 56,7 tỷ ở 10%, 60,9 tỷ ở 12%, 61,9 tỷ ở 14%, phẳng khoảng 62,0 tỷ từ 16%. Mức 22% nằm sâu trong vùng phẳng.
- **Ngân sách (C3):** phẳng từ khoảng 325 tỷ, 60,7 tỷ ở 300 tỷ, 57,7 tỷ ở 275 tỷ, 53,2 tỷ ở 250 tỷ.
- **Ngưỡng duyệt tối thiểu (C2):** phẳng đến 50%, rồi 61,1 tỷ ở 55%, 58,2 tỷ ở 60%, 53,4 tỷ ở 65%, 45,9 tỷ ở 70%.

### 7.6 Giá của C5 và C6, và của từng luật chạy riêng

- C6 tốn 0 (không binding).
- **C5 tốn 0,055 tỷ VND** (0,09% lợi nhuận): ép thêm **286 khoản micro** (761 lên 1.047), tức khoảng **0,19 triệu VND lợi nhuận kỳ vọng mất mỗi khoản thêm**.
- Mỗi luật chạy riêng (mẫu, quy đổi): C4 0,40 tỷ, C5 0,09 tỷ, C7 0,03 tỷ, C2 0,004 tỷ, C1/C3/C6 bằng 0. Tổng các luật riêng lẻ 0,52 tỷ so với 0,44 tỷ khi chạy cùng nhau: các luật chồng lấn, nên chi phí không cộng được.
- **Lập luận:** ủng hộ trả giá C5 (rẻ, giữ cấp tín dụng cho nhóm chiếm 20% đơn, khách micro có thể thành khách lớn sau này, một giá trị mà mô hình không nắm); phản đối (đây đúng là nhóm có kinh tế tệ nhất, và mô hình PD kém chính xác nhất ở đó). Kết luận: giữ C5 vì hóa đơn nhỏ và cố định; luật nên đàm phán lại là C4, đắt gấp khoảng bảy lần mỗi điểm.

### 7.7 Căng thẳng giữa C1 và C2

Đề dự đoán C1 (rủi ro ≤ 22%) và C2 (duyệt ≥ 45%) mâu thuẫn. Ở đây không có mâu thuẫn: giữ duyệt ≥ 45% và các luật còn lại, rủi ro thấp nhất đạt được là **8,7%** (so với giới hạn 22%); giữ rủi ro ≤ 22%, tỷ lệ duyệt cao nhất đạt được là **76,5%** (so với sàn 45%). (Bisection trên mẫu, sai số khoảng 0,1 điểm.)

### 7.8 Giá trị của mô hình PD

Làm nhiễu logit của PD (độ lệch chuẩn 1,39) cho Gini giảm từ 0,701 xuống 0,551 (`TARGET` chỉ dùng để **đo** Gini, không dùng để quyết định). Tối ưu lại với PD nhiễu rồi đánh giá chính sách đó bằng PD tốt:

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
- **Tập luật binding không đổi.** Đề kỳ vọng mô hình tốt hơn đẩy luật rủi ro ra khỏi biên, nhưng C1 vốn đã dư (14,4%) và vẫn dư với mô hình yếu (17,8% tính bằng PD tốt, dưới 22%). Giá trị của mô hình thể hiện ở lợi nhuận, không ở luật nào binding. Giả định: PD của Challenge 1 gần sự thật hơn PD nhiễu; kiểm tra bằng kết quả thực là mục 9.

### 7.9 Độ nhạy của giả định lợi nhuận

| Giả định | Lợi nhuận (theo chính nó) | Tỷ lệ duyệt | Quyết định đổi | Regret theo giả định gốc |
|---|---|---|---|---|
| Gốc: thu hồi 20%, vỡ nợ tại 35% kỳ hạn | 60,92 tỷ | 48,5% | — | — |
| Thu hồi 10% | 57,51 tỷ | 46,3% | 2,3% | 0,21 tỷ |
| Thu hồi 30% | 64,83 tỷ | 50,6% | 2,2% | 0,21 tỷ |
| Vỡ nợ tại 25% kỳ hạn | 56,14 tỷ | 45,3% | 3,3% | 0,44 tỷ |
| Vỡ nợ tại 50% kỳ hạn | 70,83 tỷ | 50,2% | 4,4% | 0,80 tỷ |

**Chính sách ổn định, mức lợi nhuận thì không.** Đổi thu hồi 10 điểm đổi khoảng 2% quyết định và regret 0,21 tỷ (0,3% lợi nhuận). Thời điểm vỡ nợ nhạy hơn (tới 4,4% quyết định, regret 0,80 tỷ, 1,3%) và là giả định nên làm chắc trước. Tập luật binding luôn có C4 và C5; C3 (ngân sách) thêm vào khi giả định làm khoản vay sinh lời hơn.

## 8. Chưng cất thành chính sách dạng luật

18.000 quyết định riêng lẻ không phải là chính sách: không ai triển khai, kiểm toán hay giải thích được, và không khái quát cho đơn tháng sau.

### 8.1 Phương pháp (`pydo.DecisionTree`)

1. Trồng CART trên **các cột thô** (`RISK_BAND`, `PRODUCT_GROUP`, `TENOR_BUCKET`, `AMOUNT_M`, `INTEREST_RATE`) để dự đoán quyết định của nghiệm tối ưu cấp tài khoản. Mỗi đơn được gán trọng số theo mức đáng giá của quyết định (|lợi nhuận nếu duyệt − nếu từ chối|), để cây tiêu các lần chia vào nơi có tiền.
2. Chỉ giữ **phân hoạch** của cây (mỗi lá là một nhóm đơn).
3. **Tối ưu lại một hành động chung cho mỗi lá** với **đúng bộ ràng buộc cũ**: một bài MILP nhỏ mà biến là lá, không phải đơn. Kết quả là một tập luật nhỏ vẫn khả thi (hoặc lỗi rõ ràng "phân hoạch quá thô: cần cây sâu hơn", không bao giờ vi phạm âm thầm).

### 8.2 Chọn cỡ cây

| Cây | Số lá | Lợi nhuận kỳ vọng | Giữ lại so với tối ưu cấp tài khoản |
|---|---|---|---|
| độ sâu 4, ngưỡng chia mặc định của template | 11 | 53,40 tỷ | 87,7% |
| độ sâu 6, ngưỡng chia mặc định | 17 | 55,73 tỷ | 91,5% |
| độ sâu 5, ngưỡng chia 0 | 27 | 55,22 tỷ | 90,6% |
| **độ sâu 6, ngưỡng chia 0 (được chọn)** | **32** | **59,31 tỷ** | **97,35%** |

Ngưỡng chia mặc định của template (`min_gini_gain = 0,0001`) ngăn CART chia theo `PRODUCT_GROUP`, đúng chiều mà C5 và C7 được viết, nên các lá không thể định hình để thỏa luật với giá rẻ; chi phí giữ ở mức 9 đến 12%. Đặt ngưỡng bằng 0 thì chia theo sản phẩm xuất hiện, chi phí xuống 2,65%. Quy tắc chọn cỡ, đặt trước: **cây khả thi nhỏ nhất có chi phí không quá 3%**. Đây là một **bảng** 32 luật, không phải "vài luật"; cây 17 lá dễ đọc hơn nhưng tốn 5,2 tỷ (8,5%). Dùng `PD` liên tục thay cho dải cho 97,42% so với 97,35%: không tốt hơn, và luật dải cố định dễ triển khai và giám sát hơn.

### 8.3 Chi phí của việc đơn giản hóa

**Đơn giản hóa 17.976 quyết định thành 32 luật tốn 1,61 tỷ VND, tức 2,65% nghiệm tối ưu cấp tài khoản** (59,31 tỷ so với 60,92 tỷ).

- Chỉ lấy hành động đa số của mỗi lá thì giữ 59,66 tỷ (97,9%) nhưng **vi phạm 3 trong 42 luật**; tối ưu lại hành động theo lá tốn thêm 0,35 tỷ và **không vi phạm luật nào**.
- Mất mát **phân tán**: lá tệ nhất chiếm 15% tổng mất mát, năm lá tệ nhất khoảng 54%; không có một nhánh nào mà cây sâu hơn sẽ bù lại phần lớn.
- **Không học vẹt:** đồng thuận với quyết định cấp tài khoản 92,9% (tập train của cây) và 92,1% (tập test), lợi nhuận giữ lại 97,6% và 96,9%. Chỉ tháng 5 có PD nên không có mẫu ngoài thời gian; đây là chia ngẫu nhiên đơn.

### 8.4 Cây dùng và hướng của các biến

Cây dùng cả 5 biến. SHAP trung bình tuyệt đối: `RISK_BAND` 0,37 (áp đảo), kỳ hạn (`T_SHORT` 0,13), `INTEREST_RATE` 0,03, `AMOUNT_M` 0,003, `PRODUCT_GROUP` 0,0005. Hướng đúng kinh tế của mục 4:

- Rủi ro càng cao càng bị từ chối (tương quan duyệt/dải −0,83).
- Kỳ hạn ngắn bị từ chối nhiều (tỷ lệ duyệt: 29% `T_SHORT`, 57% `T_MID`, 81% `T_LONG`): hiệu ứng chi phí cố định.
- Khoản lớn được duyệt nhiều hơn (+0,23): chi phí cố định được pha loãng.
- Lãi suất: tương quan thô với duyệt là −0,35 trông ngược nhưng bị nhiễu (lãi cao đi kèm PD cao). **Trong cùng dải rủi ro và kỳ hạn**, các lá duyệt khi lãi trên ngưỡng và từ chối khi dưới: lãi bù đắp rủi ro đúng như mong đợi.
- `PRODUCT_GROUP` gần như không có trọng số SHAP nhưng vẫn cần trong phân hoạch (mục 8.2); sản phẩm chủ yếu được mang qua kỳ hạn, ticket và lãi.

Không có biến nào đẩy sai hướng, nên không có dấu hiệu lỗi dữ liệu hay lỗi mô hình lợi nhuận.

### 8.5 Chính sách cuối cùng chấm bằng cùng một bộ đánh giá

Cây thỏa **cả 42 luật**: duyệt 52,2%, rủi ro 14,6%, dư nợ **347,6 tỷ** (giới hạn 350), kỳ hạn dài **44,6%** (giới hạn 45%), micro 12,9%, lãi cao 8,5%. Cây duyệt nhiều hơn nghiệm cấp tài khoản (52,2% so với 48,5%) và dùng gần hết ngân sách vì phải quyết định cả lá; nó khả thi nhưng ít dư địa hơn. So với **ngưỡng PD đơn tốt nhất** (PD ≤ 0,362, không biết sản phẩm hay kỳ hạn): 54,94 tỷ (90% tối ưu) và vi phạm luật ngân sách (354 tỷ), nên biết sản phẩm, kỳ hạn và lãi đáng khoảng 4,4 tỷ so với một ngưỡng điểm đơn thuần.

## 9. Tập kiểm thử bằng kết quả thực tế

Chỉ ở đây `TARGET` được dùng: với mỗi khoản được duyệt, lợi nhuận thực tế là `profit_good` nếu `TARGET = 0` và `profit_bad` nếu `TARGET = 1`.

| | Kỳ vọng | Thực hiện | Chênh |
|---|---|---|---|
| BAU (duyệt hết) | 11,98 tỷ | 10,57 tỷ | −1,41 tỷ (−11,8%) |
| Trần không ràng buộc | 61,18 tỷ | 60,33 tỷ | −0,85 tỷ (−1,4%) |
| Tối ưu cấp tài khoản | 60,92 tỷ | 60,30 tỷ | −0,63 tỷ (−1,0%) |
| **Cây 32 luật** | **59,31 tỷ** | **58,28 tỷ** | **−1,03 tỷ (−1,7%)** |

Chính sách cây thu thực tế **58,28 tỷ, gấp 5,5 lần BAU**. Tỷ lệ rủi ro thực tế của nó là **15,0%** (kỳ vọng 14,6%, giới hạn 22%): không vi phạm luật rủi ro ngoài đời thực.

**Quy kết chênh lệch** (hai nguồn được tách riêng):

- **Mức PD:** định giá lại sổ đã duyệt bằng cách nhân mọi PD với **một hệ số chung** để PD trung bình bằng bad rate thực tế của sổ được duyệt (giả định gốc, giữ thứ hạng): lợi nhuận kỳ vọng đổi **−0,58 tỷ** (nghiệm cấp tài khoản; PD trung bình 13,6% so với 13,9% thực tế) và **−1,40 tỷ** (cây; 14,2% so với 14,7%). Mô hình Challenge 1 hơi lạc quan 0,2 đến 0,5 điểm phần trăm.
- **Thứ hạng, cơ cấu, ngẫu nhiên:** phần còn lại là **−0,04 tỷ** và **+0,37 tỷ**: không có gì hệ thống.
- **Có phải ngẫu nhiên?** Mô phỏng Monte Carlo (vẽ `TARGET` từ chính PD của ta) cho khoảng dao động 1 độ lệch chuẩn của lợi nhuận thực khoảng **1,1 tỷ**; lợi nhuận thực nằm ở phân vị 31% (cấp tài khoản) và 21% (cây). Các chênh lệch (0,5 và 0,9 sigma) phù hợp với PD đúng và may rủi bình thường; một tháng không phân biệt được một PD thấp 0,3 điểm với nhiễu.
- **Giả định lợi nhuận** (mục 7.9) làm mức lợi nhuận của cùng chính sách đổi nhiều (kỳ vọng 53,9 đến 68,3 tỷ, thực hiện 52,7 đến 67,5 tỷ) nhưng chênh lệch giữa kỳ vọng và thực hiện giữ ở −0,8 đến −1,2 tỷ. Bất định về giả định đổi **mức** lợi nhuận; sai số PD đổi **chênh lệch**; cái đầu lớn hơn.

## 10. Sản phẩm bàn giao

### `optimal_policy.csv`

Một file, 81 dòng, 17 cột. Cột `record_type` phân biệt ba khối: `leaf_rule` (32 dòng), `portfolio_kpi` và `constraint` (một dòng cho mỗi luật trong 42 luật).

| Cột | Ý nghĩa |
|---|---|
| `record_type`, `name` | Loại dòng và tên (`leaf 0`, tên KPI, tên luật) |
| `rule_description` | Luật dạng chữ (với `leaf_rule`) hoặc luật ràng buộc (với `constraint`) |
| `RISK_BAND`, `PRODUCT_GROUP`, `TENOR_BUCKET` | Điều kiện của lá trên ba chiều này (`any` nếu không giới hạn) |
| `other_conditions` | Điều kiện trên `AMOUNT_M` và `INTEREST_RATE` |
| `action` | `approve` hoặc `reject` |
| `n_apps`, `approved_n` | Số đơn trong lá và số đơn được duyệt |
| `approved_volume_m`, `exp_bad_volume_m` | Dư nợ giải ngân và khối lượng vỡ nợ kỳ vọng (triệu VND) |
| `exp_profit` | Lợi nhuận kỳ vọng (VND) |
| `value`, `limit`, `slack`, `satisfied` | Với `portfolio_kpi` và `constraint`: giá trị, giới hạn, độ dư và trạng thái |

### `constrained_output.csv`

Export cấp tài khoản của solver (17.976 dòng): quyết định và các biến đầu ra cho từng đơn.

### `models/clx_approval_tree.json` và `.sql`

Chính sách cuối cùng dạng luật, đọc được không cần scikit-learn (JSON), và biểu thức SQL `CASE` cho bộ máy luật nói SQL. Định dạng trao đổi cây quyết định chuyên dụng (`.fjdt`) không được `pydo` tạo ra; SQL thay thế.

### Kiểm tra ràng buộc trên chính các file đã lưu

Một bộ kiểm tra **không dùng chung mã với solver** (chỉ biết bảy luật của đề): đọc lại `constrained_output.csv`; nạp lại cây từ JSON rồi áp cho toàn bộ 17.976 đơn; và đối chiếu `optimal_policy.csv` (tổng số đơn khớp, số duyệt khớp, mọi dòng ràng buộc thỏa). Cả ba thỏa đủ 7 luật.
Tùy chọn `project.save(...)` (lưu cả dự án thành zip) không được `pydo` hỗ trợ.

## 11. Khuyến nghị

1. **Chính sách:** duyệt đơn từ các dải rủi ro thấp, chọn lọc hơn ở kỳ hạn ngắn (chi phí cố định ăn hết biên lợi nhuận) và ở khoản rủi ro cao trừ khi lãi suất bù đắp; giữ ít nhất 12% số khoản duyệt là micro và nằm trong ngân sách 350 tỷ cùng trần 45% kỳ hạn dài. Áp dụng bằng **32 luật** dễ đọc, duyệt khoảng **52%** đơn, giải ngân **348 tỷ VND** ở rủi ro kỳ vọng **14,6%**, thỏa cả bảy luật.
2. **Giá trị so với duyệt hết:** **+47,3 tỷ VND lợi nhuận kỳ vọng** (59,31 so với 11,98 tỷ, ×5,0); **+47,7 tỷ thực hiện** (58,28 so với 10,57 tỷ, ×5,5). Gần như toàn bộ đến từ việc không cho vay các khoản làm mất tiền; nghiệm tối ưu cấp tài khoản sẽ thêm 1,6 tỷ nữa.
3. **Luật nên cân nhắc nới: C4 (trần kỳ hạn dài).** Mỗi điểm phần trăm giới hạn đáng khoảng 0,10 tỷ (siết một điểm mất 0,16 tỷ); chính sách không ràng buộc nằm ở 47,9% so với trần 45%, nên nới tới khoảng 48% thu hồi gần hết 0,29 tỷ mà bảy luật tốn. **C5 là bảo hiểm rẻ**: 0,055 tỷ tổng cộng, khoảng 0,19 triệu mỗi khoản micro thêm. **C1, C2, C3, C6 hiện không tốn gì**: có thể hạ giới hạn rủi ro từ 22% xuống khoảng 15% mà không mất lợi nhuận, nên luật rủi ro không phải là thứ đang bảo vệ danh mục; mục tiêu lợi nhuận mới là thứ đó.
4. **Chi phí triển khai (chưng cất):** 1,61 tỷ VND, 2,65% của tối ưu cấp tài khoản (kỳ vọng 59,31 so với 60,92 tỷ; thực hiện 58,28 so với 60,30 tỷ, trong khi một độ lệch chuẩn ngẫu nhiên là 1,1 tỷ). Cây 17 luật tốn 5,2 tỷ (8,5%): giá của một trang ngắn hơn.
5. **Điểm yếu lớn nhất:** (a) con số lợi nhuận dựa trên giả định kinh tế tổng hợp; chính sách ổn định (đổi 2 đến 4% quyết định) nhưng mức lợi nhuận thì không (khoảng 54 đến 68 tỷ cho cùng chính sách), thời điểm vỡ nợ cần làm chắc trước; (b) một tháng và một mô hình PD: PD hơi lạc quan 0,2 đến 0,5 điểm không tách được khỏi may rủi, và cây chưa được kiểm tra ngoài thời gian; (c) ít dư địa: cây dùng 347,6 trên 350 tỷ và 44,6% trên 45%, nên một thay đổi cơ cấu đơn có thể đẩy nó vượt luật, phải tối ưu lại và kiểm tra lại hằng tháng chứ không đóng băng; (d) các ô thưa: luật C7 của `CLX_LARGE` dựa trên ô 17 đến 29 đơn, và decile PD an toàn nhất bị dự báo 2,0% so với 0,7% thực, nên khoản an toàn nhất bị định giá thấp chút; (e) công cụ là `pydo` tự xây: được kiểm tra bằng vét cạn và bộ kiểm tra độc lập nhưng chưa được đối chiếu với một công cụ tối ưu khác; nên tái tạo nghiệm tối ưu cấp tài khoản trên một công cụ độc lập trước khi đưa vào vận hành.

## 12. Nhật ký quyết định và giả định

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Công cụ | `pydo` tự xây | Kiểm soát toàn bộ logic tối ưu và kiểm thử độc lập được |
| Ngưỡng ràng buộc | Cố định theo đề | Đề cấm nới |
| C3 | Viết theo đầu đơn | Bất biến khi chạy mẫu |
| C7 | Theo sản phẩm, bỏ cặp có ô rỗng, giữ ô thưa | Tránh nhân quá nhiều luật; không để dải nào mất chốt đơn điệu |
| Dải PD | 10 dải ngưỡng cố định | Giám sát được, không trôi |
| Đơn vị | Dư nợ theo triệu | Điều kiện số học (mục 7.1) |
| Gap solver | 1% cho kết quả chính, 0,01% cho các phép quét | Kết quả chính theo đề; phép quét cần sai số nhỏ hơn hiệu ứng |
| Giá trị mô hình PD | Đánh giá bằng kỳ vọng theo PD tốt | `TARGET` chỉ dùng ở tập kiểm thử |
| Cỡ cây | Nhỏ nhất có chi phí ≤ 3% | Quy tắc đặt trước |
| Ngưỡng chia của cây | 0 (đưa vào lưới) | Mặc định của template chặn chia theo sản phẩm |
| Định dạng triển khai | JSON và SQL | Không có exporter `.fjdt` |

## 13. Hạn chế và việc tiếp theo

**Hạn chế**

- Kinh tế là số tổng hợp; một tháng dữ liệu; PD của một mô hình.
- Chưa có mẫu ngoài thời gian cho cây (chỉ tháng 5 có PD).
- Cây có ít dư địa trên C3 và C4.
- Chưa so sánh với một công cụ tối ưu khác.
- Các phép quét nhiều lần chạy trên mẫu 25% và cao hơn bản đầy đủ khoảng 1,8%.
- Thử nghiệm thu hồi/thời điểm vỡ nợ làm đổi giả định cho toàn bộ sổ; không có phân bố theo sản phẩm.

**Với thêm thời gian**

1. Làm chắc thời điểm vỡ nợ và thu hồi từ dữ liệu thực.
2. Tối ưu lại và kiểm tra ràng buộc hằng tháng; thêm biên an toàn cho C3 và C4.
3. Đối chiếu nghiệm với một công cụ tối ưu khác.
4. Kiểm tra cây trên tháng kế tiếp khi có PD.
5. Thử luật C7 chia theo cả kỳ hạn, với mẫu đủ lớn.
6. Chia lợi nhuận vỡ nợ theo sản phẩm (thu hồi khác nhau).

## 14. Tái lập kết quả

1. Chạy notebook 01 để có `results/pd_predictions.csv`.
2. Mở `notebooks/02_portfolio_optimization.ipynb` với kernel `pydo-project`, thư mục làm việc là `notebooks/`; `pydo` đã được cài ở chế độ chỉnh sửa được bởi `pip install -r requirements.txt` (notebook `import pydo`).
3. Chạy toàn bộ từ trên xuống (khoảng 11 phút trên toàn bộ danh mục). Notebook tạo `notebooks/functions/`, `results/models/`, `results/constrained_output.csv`, `results/optimal_policy.csv`.
4. Kiểm tra thư viện: `python -m unittest discover -s tests`.

Cấu trúc notebook: 0 môi trường; 1 danh mục; 2 kinh tế (2b kiểm tra độc lập); 3 dựng dự án; 4 BAU; 5 trần; 6 mô hình có ràng buộc; 7 chẩn đoán cơ bản; 7b trả lời năm câu hỏi (điều kiện số học, binding, profit bridge, giá bóng, đường biên, chi phí C5 và C6, căng thẳng C1 và C2, giá trị mô hình PD, độ nhạy); 8 chưng cất; 9 tập kiểm thử; 10 sản phẩm bàn giao; 11 khuyến nghị.
