# `pydo`: thư viện tối ưu quyết định

**PyDO** (Pythonic Decision Optimizer) là một thư viện tối ưu quyết định nhỏ, tự chứa, được xây trong dự án này; import bằng `import pydo as do`. Các khái niệm của nó (Portfolio, Variables, Facts, Functions, Project, Scenario, Constraints) là các khái niệm phổ biến của tối ưu quyết định; thư viện không phụ thuộc vào công cụ thương mại hay nội bộ nào.

Phiên bản 0.0.1. Mã ở `src/pydo/` (khoảng 2.000 dòng), kiểm thử ở `tests/` (43 test).

## Mục lục

1. [Làm được gì](#1-làm-được-gì)
2. [Yêu cầu và cấu trúc](#2-yêu-cầu-và-cấu-trúc)
3. [Khởi động nhanh](#3-khởi-động-nhanh)
4. [Khái niệm](#4-khái-niệm)
5. [Ngôn ngữ luật ràng buộc](#5-ngôn-ngữ-luật-ràng-buộc)
6. [Mô hình toán học](#6-mô-hình-toán-học)
7. [Ba flow và vòng đời của một Scenario](#7-ba-flow-và-vòng-đời-của-một-scenario)
8. [Báo cáo chẩn đoán](#8-báo-cáo-chẩn-đoán)
9. [Chưng cất thành luật: `DecisionTree`](#9-chưng-cất-thành-luật-decisiontree)
10. [So sánh và đăng ký kết quả](#10-so-sánh-và-đăng-ký-kết-quả)
11. [Lỗi](#11-lỗi)
12. [Tham chiếu API](#12-tham-chiếu-api)
13. [Kiểm thử](#13-kiểm-thử)
14. [Giới hạn đã biết](#14-giới-hạn-đã-biết)

---

## 1. Làm được gì

Bạn mô tả **kinh tế** (hàm Python thuần cho một đơn và một phương án), **không gian quyết định** (biến) và **luật kinh doanh** (ràng buộc). `pydo`:

1. đánh giá một chính sách cố định (flow `simulation`);
2. tìm phương án tốt nhất cho từng đơn khi không có luật (flow `unconstrained`);
3. giải bài toán có ràng buộc dưới dạng quy hoạch nguyên hỗn hợp bằng HiGHS (flow `constrained`);
4. chẩn đoán lời giải: độ dư của từng luật, sức khỏe số học của ma trận, giá bóng bằng cách giải lại;
5. chưng cất lời giải cấp tài khoản thành một tập luật nhỏ, đọc được, vẫn thỏa ràng buộc (`DecisionTree`).

Bài toán mẫu của dự án là duyệt/từ chối 17.976 đơn vay với 7 ràng buộc (xem [challenge_2_portfolio_optimization.md](challenge_2_portfolio_optimization.md)).

## 2. Yêu cầu và cấu trúc

**Phụ thuộc:** Python 3.11, numpy, pandas, scipy (HiGHS qua `scipy.optimize.milp`). Chỉ `DecisionTree` cần thêm scikit-learn; `explain()` cần thêm shap và matplotlib (nhập khi cần).

**Dùng:** cài từ thư mục gốc của repo bằng `pip install -e .` (hoặc `pip install -r requirements.txt`, đã gồm dòng này), rồi `import pydo as do`.

| Module | Nội dung |
|---|---|
| `portfolio.py` | `Portfolio`: danh mục tài khoản, lấy mẫu phân tầng |
| `facts.py`, `variables.py` | `Facts` (giả định có tên), `Variables` (biến quyết định và miền) |
| `functions.py` | `Functions`: nạp và chạy các hàm tính (kinh tế) |
| `project.py` | `Project`: gắn mọi thứ lại, tạo `Scenario`, đăng ký và báo cáo kết quả |
| `rules.py` | Bộ phân tích cú pháp luật, lớp `Constraints` |
| `model.py` | Biên dịch luật thành ma trận MILP, gọi solver |
| `scenario.py` | `Scenario`: ba flow, báo cáo, `with_constraints`, `apply_policy` |
| `tree.py` | `DecisionTree` |
| `reports.py` | `Report`: so sánh các kịch bản |
| `errors.py` | Hệ thống lỗi |

## 3. Khởi động nhanh

Một bài toán nhỏ (500 đơn): duyệt hay từ chối để tối đa lợi nhuận kỳ vọng, với rủi ro ≤ 15%, duyệt ≥ 40% và nhóm `small` ≥ 30% số khoản duyệt.

File `econ.py` (hàm kinh tế, mỗi hàm có chữ ký `def f(obj)`):

```python
def flatten(obj):
    obj.p_amount = obj.account["AMOUNT"]          # p_: trường thô của danh mục
    obj.p_pd = obj.account["PD"]
    obj.a_approve = obj.decision["APPROVE"]       # a_: hành động đang được đánh giá
    obj.f_margin = obj.facts.get("margin")        # f_: giả định
    obj.f_lgd = obj.facts.get("lgd")

def economics(obj):
    good = obj.p_amount * obj.f_margin
    bad = -obj.p_amount * obj.f_lgd
    obj.v_exp_profit = obj.a_approve * ((1 - obj.p_pd) * good + obj.p_pd * bad)   # v_: giá trị tính được
    obj.v_approved = obj.a_approve
    obj.v_amount = obj.a_approve * obj.p_amount
    obj.v_pd = obj.p_pd
```

Script:

```python
import numpy as np, pandas as pd, pydo as do

rng = np.random.default_rng(0)
n = 500
apps = pd.DataFrame({"ID": [f"A{i:04d}" for i in range(n)],
                     "SEGMENT": rng.choice(["small", "large"], n),
                     "AMOUNT": rng.integers(100, 1000, n),
                     "PD": rng.uniform(0.02, 0.6, n).round(3)})

project = do.Project("Quick start")
project.facts = do.Facts().set("margin", 0.20, "profit rate of a good loan").set("lgd", 0.50, "loss rate of a bad loan")
project.variables = do.Variables().set("APPROVE", [0, 1])
project.portfolio = do.Portfolio(apps, key_column="ID")
project.functions = do.Functions(["econ.py"])
project.objective, project.direction = "v_exp_profit", "maximize"
project.output_variables = ["v_exp_profit", "v_approved", "v_amount", "v_pd"]

ceiling = project.scenario(name="Ceiling", flow="unconstrained").build().optimize()
rules = (do.Constraints("business rules")
         .set(name="Risk", rule="WEIGHTED_AVG(v_pd, v_amount) <= 0.15")
         .set(name="Approval", rule="AVG(v_approved) >= 0.40")
         .set(name="Small share", rule="SUM(v_approved) WHERE (SEGMENT = 'small') >= SUM(v_approved) * 0.30"))
best = project.scenario(name="Constrained", flow="constrained", constraints=rules).build().optimize(gap=0.0)

print(f"ceiling {ceiling.total:,.0f} | constrained {best.total:,.0f}")
print(best.constraint_report()[["constraint", "lhs", "op", "rhs", "slack", "satisfied"]])
```

Kết quả của ví dụ: trần 11.481, có ràng buộc 11.420; luật rủi ro binding (0,150), duyệt 42,4%, nhóm `small` 91 khoản so với tối thiểu 63,6. `best.solver_report()` cho trạng thái `optimal` và 500 biến.

## 4. Khái niệm

### 4.1 `Portfolio`

Danh mục tài khoản, một dòng mỗi tài khoản.

```python
do.Portfolio(frame, key_column, random_sampling=1.0, stratify_columns=None, random_state=42)
```

- `key_column` phải duy nhất và không rỗng.
- `random_sampling` ∈ (0, 1]: tỷ lệ mẫu. **Phát triển trên mẫu, chạy toàn bộ ở bước cuối.**
- `stratify_columns`: lấy mẫu trong từng tổ hợp các cột này, mọi tầng không rỗng giữ ít nhất một đơn. Liệt kê **mọi cột xuất hiện trong mệnh đề `WHERE`** của luật, nếu không một luật theo ô có thể còn rất ít đơn hoặc không còn đơn nào.
- Thuộc tính: `full` (toàn bộ), `data` (phần đang dùng), `n_full`, `n_active`, `sampling_fraction`, `scale` (= `n_full/n_active`, nhân vào tổng của mẫu để ước lượng tổng toàn bộ), `columns`.

### 4.2 `Facts`

Các giả định có tên (chi phí vốn, tỷ lệ thu hồi…).

```python
facts = do.Facts().set("recovery_rate", 0.20, "thu hồi trên dư nợ")
facts.get("recovery_rate")     # 0.2; thiếu thì UnknownFactError kèm danh sách fact có
facts.copy()                   # bản sao để chạy độ nhạy
```

Lý do tồn tại: đổi **một dòng** rồi chạy lại cả nghiên cứu, thay vì sửa file hàm.

### 4.3 `Variables`

Biến quyết định và miền giá trị. Mỗi **phương án** (treatment) là một giá trị của mọi biến.

```python
do.Variables().set("APPROVE_ACTION", [0, 1])
```

Nhiều biến cho tích Descartes các miền theo thứ tự khai báo. **Phương án đầu tiên là phương án tham chiếu** (ở bài mẫu: từ chối).

### 4.4 `Functions` và đối tượng `obj`

```python
do.Functions(["functions/flatten.py", "functions/economics.py"])   # đường dẫn, hoặc các hàm Python
```

- Nạp **mọi hàm công khai có đúng một tham số** trong mỗi file, theo thứ tự file và thứ tự định nghĩa; chạy **một lần cho mỗi cặp (tài khoản, phương án)**. Mỗi bước đọc thứ bước trước ghi lên `obj`.
- `obj.account`: dict dòng thô của danh mục; `obj.decision`: dict `{biến: giá trị}` của phương án đang xét; `obj.facts`: đối tượng `Facts`.
- Quy ước tiền tố thuộc tính: `p_` trường thô, `a_` hành động, `f_` giả định, `v_` giá trị tính được.
- **Biến tính được** là thuộc tính `v_*` mang giá trị số. Hàm lỗi cho thông báo kèm tên hàm và mã tài khoản.
- Phương án từ chối nên cho **mọi biến `v_*` bằng 0** (nhân với `a_approve`); nếu không, đơn bị từ chối sẽ vô tình đóng góp vào tổng của danh mục.

### 4.5 `Project`

Gắn mọi thứ lại. **Thứ tự gán có ý nghĩa**:

```python
project = do.Project("tên", "mô tả")
project.facts = ...; project.variables = ...; project.portfolio = ...      # 1. ba thứ này trước
project.functions = do.Functions([...])                                    # 2. chạy thử trên một tài khoản để tìm các biến v_
project.objective = "v_exp_profit"; project.direction = "maximize"         # 3. phải là biến v_ số có thật
project.output_variables = [...]                                           # 4. các biến muốn xuất ra
```

Gán `functions` trước khi có `portfolio`/`variables`/`facts`, hoặc `objective` trước `functions`, hoặc tên biến không tồn tại cho `ProjectError` có hướng dẫn. `project.calculated_variables` liệt kê các biến `v_` tìm được. Các biến được ràng buộc dùng sẽ được xuất tự động dù không có trong `output_variables`.

## 5. Ngôn ngữ luật ràng buộc

Luật là chuỗi giống SQL trên các cặp tài khoản–phương án được chọn.

```python
rules = do.Constraints("mô tả")
rules.set(name="C1 Risk", rule="WEIGHTED_AVG(v_pd, v_amount) <= 0.22", description="...")
```

Cú pháp được kiểm tra ngay khi `set`; lỗi nêu tên luật.

```
rule      := side OP side                    -- toàn cục (một vế là số) hoặc tương đối (hai vế là phép tổng hợp)
           | cột OP giá_trị                  -- luật cấp tài khoản
side      := factor { ('*' | '/') factor }   -- tối đa một phép tổng hợp, còn lại là số
factor    := ['-'] SỐ | phép_tổng_hợp
phép_tổng_hợp := SUM(biến) | AVG(biến) | WEIGHTED_AVG(biến, trọng_số) | QUICK_WEIGHTED_AVG(biến, trọng_số)
                 [ WHERE điều_kiện ]
điều_kiện := cmp | điều_kiện AND điều_kiện | điều_kiện OR điều_kiện | NOT điều_kiện | '(' điều_kiện ')'
cmp       := cột OP giá_trị | cột IN (giá_trị, ...)
OP        := <=  |  >=  |  =
```

Từ khóa không phân biệt hoa thường; tên cột và biến thì có. Số được viết `350_000` hoặc `1e9` đều được.

**Các luật của trò chơi mà bộ phân tích thực thi:**

| Quy tắc | Lý do |
|---|---|
| `<` và `>` bị từ chối (dùng `<=`, `>=`) | Bộ dựng tuyến tính cần cận không chặt |
| `WHERE` chỉ được dùng **cột thô của danh mục**, không dùng biến `v_` | Mẫu số của phép tổng hợp phải cố định trước khi chạy solver; biến `v_` phụ thuộc vào phương án được chọn |
| Phép tổng hợp có `WHERE` khớp **0 tài khoản** là lỗi | Một luật rỗng không được im lặng bỏ qua |
| `WEIGHTED_AVG` chỉ so với **một số** (không so với phép tổng hợp khác, không nhân hệ số) | Tỷ số của hai tổng phụ thuộc quyết định là phi tuyến |
| `QUICK_WEIGHTED_AVG` yêu cầu mọi trọng số > 0 | Bỏ hàng kiểm tra mẫu số; tài khoản bị từ chối có trọng số 0 nên chỉ an toàn khi trọng số luôn dương |

Ba loại luật:

- **Toàn cục:** `AVG(v_approved) >= 0.45`.
- **Tương đối:** `SUM(v_amount) WHERE (TENOR = 'L') <= SUM(v_amount) * 0.45`.
- **Cấp tài khoản:** một so sánh trần trên biến (ví dụ `v_ok = 1`): các cặp vi phạm bị **loại** khỏi lựa chọn (bị tỉa).

Biến trong phép tổng hợp có thể là biến `v_` hoặc **cột số thô** của danh mục.

## 6. Mô hình toán học

**Biến.** Mỗi tài khoản chọn đúng một phương án. Phương án đầu tiên (tham chiếu) được **khử**: `x_i0 = 1 − Σ_t y_it`. Với duyệt/từ chối, mô hình chỉ có **một biến nhị phân `y_i` mỗi tài khoản** và không có hàng ràng buộc gán. (Với nhiều hơn hai phương án có thêm hàng `Σ_t y_it ≤ 1`.)

**Tuyến tính hóa.** Mọi giá trị là hàm tuyến tính của `y`. Với giá trị `v_it` của phương án t và mặt nạ `c_i` của `WHERE`:

| Luật | Dạng tuyến tính |
|---|---|
| `SUM(v) WHERE c` | `Σ_i c_i ( v_i0 + Σ_t (v_it − v_i0) y_it )` |
| `AVG(v) WHERE c` | như trên chia cho số tài khoản khớp `c` (mẫu số cố định) |
| `WEIGHTED_AVG(v, w) <= K` | `Σ_i c_i w_it (v_it − K) x_it ≤ 0` (tỷ số hai tổng được đưa về tuyến tính) |
| `A <= B * k` | `A − k·B ≤ 0` |

`WEIGHTED_AVG` thường (không phải `QUICK_`) thêm một **hàng kiểm tra mẫu số** `Σ w x ≥ 1e-6` để tỷ số có nghĩa. Một luật không phụ thuộc quyết định nào mà vi phạm sẽ báo `InfeasibleModelError` ngay lúc biên dịch.

**Luật cấp tài khoản** tỉa phương án: nếu phương án tham chiếu bị cấm thì biến được ép bằng 1; nếu mọi phương án của một tài khoản bị cấm thì `InfeasibleModelError`.

**Giải.** `scipy.optimize.milp` (HiGHS) với tùy chọn `time_limit` và `mip_rel_gap`; nghiệm được làm tròn rồi đối chiếu: tổng mục tiêu từ lựa chọn xuất ra phải khớp với tổng do solver báo, nếu không báo lỗi nội bộ.

**Quy mô thực tế:** 17.976 biến, 43 hàng, khoảng 155.000 phần tử khác 0, thời gian giải khoảng 20 đến 35 giây trên máy phát triển.

## 7. Ba flow và vòng đời của một Scenario

```python
scn = project.scenario(name, description="", flow="constrained", constraints=None,
                       variable_to_decision=None, decisions=None, solver=None)
scn.build(n_jobs=1)          # tính kinh tế cho mọi cặp (tài khoản, phương án)
scn.preflight()              # (constrained) quy mô bài toán, số cặp bị tỉa, cảnh báo ô thưa
scn.optimize(solver="highs", time_limit=600, gap=0.01, n_jobs=None, verbose=False)
scn.score, scn.total, scn.total_portfolio
scn.export("file.csv")       # lựa chọn từng tài khoản + biến đầu ra
```

| Flow | Hành vi |
|---|---|
| `simulation` | Một chính sách **cố định**: `variable_to_decision={"APPROVE_ACTION": "APPROVE_ACTION_BAU"}` (đọc từ cột của danh mục, cột này phải có **trước** khi tạo `Portfolio`) **hoặc** `decisions={"APPROVE_ACTION": 1}` (hằng); không bao giờ truyền cả hai |
| `unconstrained` | Tốt nhất của từng tài khoản không luật; khi hòa lấy phương án đầu của miền (từ chối) |
| `constrained` | MILP như mục 6; cần `constraints` không rỗng |

- `constraints` cũng nhận được ở `simulation` và `unconstrained`: không áp luật nhưng `constraint_report()` cho biết chính sách đó vi phạm luật nào (đây là "slide đầu tiên" của đề).
- `score` là **trung bình mục tiêu mỗi tài khoản**; `total` là tổng trên các tài khoản đang dùng; `total_portfolio = total × portfolio.scale` (tổng quy đổi cho toàn bộ, quan trọng khi dùng mẫu).
- **Tính lại có chọn lọc:** `scn.with_constraints(luật_mới, name=None)` trả về một kịch bản `constrained` **dùng lại các cặp đã tính** (không tính lại kinh tế). Dùng cho giá bóng, đường biên, "nếu ngưỡng này đổi". Luật mới chỉ được đọc các biến đã được tính. Mặc định kịch bản mới không chiếm tên đã đăng ký của kịch bản gốc.
- **Chấm điểm một chính sách có sẵn:** `scn.apply_policy(decisions, name="policy")` (Series theo mã tài khoản hoặc mảng theo thứ tự danh mục) trả về kịch bản kiểu `simulation` đã có `export()`, `constraint_report()`, tổng. Đây là cách một cây hay một ngưỡng được chấm bằng đúng bộ đánh giá của solver. Chỉ hỗ trợ một biến quyết định.

## 8. Báo cáo chẩn đoán

**`solver_report()`** (Series): `flow`, `status` (`optimal` hoặc hết thời gian), `objective_total`, `best_bound`, `gap_requested`, `gap_achieved`, `time_seconds`, `time_limit`, `nodes`, `variables`, `rows`, `nonzeros`, `pruned_pairs`, `message`.

**`constraint_report(binding_tol=1e-3)`**: mỗi luật so với lựa chọn hiện tại.

| Cột | Ý nghĩa |
|---|---|
| `constraint`, `kind`, `rule` | Tên, loại (`global`, `relative`, `account`), nội dung |
| `lhs`, `op`, `rhs` | Giá trị hai vế (luật cấp tài khoản: số tài khoản vi phạm so với 0) |
| `slack` | Độ dư có dấu (dương là còn dư, âm là vi phạm) |
| `violation`, `satisfied` | Mức vi phạm; thỏa (dung sai tương đối 1e-6) |
| `binding` | Độ dư trong khoảng `binding_tol` × độ lớn của cận (1e-3 của 0,45 tương đương khoảng 8 tài khoản) |
| `accounts_in_scope` | Số tài khoản mà luật nói tới |

Lưu ý khi đọc `binding`: luật theo ô kiểu "100% duyệt ở cả hai ô" cho `lhs = rhs = 1` và được gắn binding dù không giới hạn gì; luật đếm có thể cho độ dư dưới một tài khoản nhưng vượt `binding_tol`.

**`coefficient_report(warn_range=1e9)`**: với từng hàng và với toàn mô hình (`WHOLE MODEL`, `OBJECTIVE`): số phần tử khác 0, hệ số nhỏ nhất, lớn nhất và dải. Cảnh báo bằng `warnings.warn` khi dải ≥ 1e9. **`pydo` không tự đổi thang đơn vị**; chọn đơn vị phù hợp (ví dụ triệu thay vì đồng).

**`preflight()`** (in và trả về bảng các hàng luật): số tài khoản, số phương án, số biến sau khi khử phương án tham chiếu, số hàng và phần tử khác 0, số cặp bị tỉa, và cảnh báo các luật có dưới 30 tài khoản trong phạm vi.

## 9. Chưng cất thành luật: `DecisionTree`

Lời giải cấp tài khoản của solver là một quyết định riêng cho từng đơn: không triển khai, kiểm toán hay giải thích được. `DecisionTree` biến nó thành một tập luật nhỏ trên **các cột thô**.

```python
tree = do.DecisionTree(max_depth=4, max_splits=None, min_weight=100, min_gini_gain=0.0001,
                       sample_weight="stake", random_state=42)
tree.fit(scenario=solved_constrained, variables=["RISK_BAND", "PRODUCT_GROUP", "TENOR_BUCKET", "AMOUNT_M", "INTEREST_RATE"],
         solver="highs", time_limit=300, gap=0.05, constrained_growth="posthoc")
```

**Thuật toán (`posthoc`):**

1. Trồng CART (scikit-learn) để dự đoán quyết định của lời giải cấp tài khoản. Biến phân loại được mã hóa one-hot; trọng số mỗi đơn là mức đáng giá của quyết định (`|mục tiêu nếu duyệt − nếu từ chối|`, `sample_weight="stake"`) hoặc đều nhau (`None`). 70% tài khoản dùng để trồng cây, 30% để kiểm tra.
2. Chỉ giữ **phân hoạch** (lá của từng tài khoản).
3. Giải một **MILP nhỏ** trên các lá: mỗi lá một biến nhị phân (duyệt cả lá hoặc từ chối cả lá), với **đúng ma trận ràng buộc cũ** (nhân với ma trận thuộc lá). Kết quả là hành động chung của mỗi lá, tối đa hóa mục tiêu dưới mọi luật.
4. Nếu không có cách gán lá nào thỏa mọi luật: `InfeasibleModelError` kèm hướng dẫn "phân hoạch quá thô, cần cây sâu hơn hoặc thêm biến".

**Tham số:** `max_depth`; `max_splits` là số lần chia tối đa của cả cây (tối đa `max_splits + 1` lá; `0` cho cây một lá); `min_weight` là số tài khoản tối thiểu mỗi lá; `min_gini_gain` ánh xạ sang `min_impurity_decrease` của scikit-learn. **Ngưỡng `min_gini_gain` ảnh hưởng mạnh**: giá trị mặc định 0,0001 trong dự án này ngăn cây chia theo `PRODUCT_GROUP`, và giữ cây ở khoảng 88 đến 93% nghiệm tối ưu; giá trị 0 cho 97,35% với 32 lá. Hãy đưa nó vào lưới chọn cỡ cây.

**Thuộc tính và phương thức:**

| Thành phần | Nội dung |
|---|---|
| `attempted_features`, `selected_features` | Biến được đề nghị và biến thực sự được dùng |
| `rules_` | Danh sách `LeafRule(leaf, description, conditions, action, n_apps)` |
| `leaves_` | Bảng mỗi lá: luật, hành động, hành động đa số của CART, số đơn, tỷ lệ duyệt của nghiệm gốc, lợi nhuận theo nghiệm gốc và theo lá, mất mát |
| `predict(frame)` | Quyết định cho mọi dòng, **chỉ từ luật** (không cần scikit-learn) |
| `performance()` | Train/test: đồng thuận với nghiệm cấp tài khoản, tỷ lệ duyệt, lợi nhuận giữ lại. Không có mẫu ngoài thời gian |
| `objective_report()`, `print_objectives()` | **Chi phí triển khai**: nghiệm cấp tài khoản, cây với hành động đa số (trước khi tối ưu lại), cây cuối; mỗi dòng kèm số luật vi phạm |
| `visualize()` | Hình cây nằm ngang, mỗi lá một dòng, ghi hành động cuối và số đơn |
| `explain(plot_type="beeswarm", max_background=100, max_samples=250)` | SHAP trên các biến được chọn |
| `save(path, overwrite=False)`, `DecisionTree.load(path)` | JSON (đọc được, không cần scikit-learn) |
| `to_sql(table)` | Biểu thức SQL `CASE` |
| `policy_` | Kịch bản chính sách của cây (đã có `constraint_report()`) |

**Quy ước luật:** điều kiện số nguyên đọc tự nhiên (`RISK_BAND = 6`, `7 <= RISK_BAND <= 8`, không phải `<= 6.5`); điều kiện loại trừ trên mọi level trừ một sẽ thành `=`. Các lá chia đều là một **phân hoạch**: mỗi tài khoản rơi vào đúng một lá; level chưa từng thấy gây lỗi rõ ràng thay vì chấm sai.

**Định dạng JSON** (`format: "pydo-decision-tree"`, `version: 1`): `source` (tên kịch bản nguồn), `variable`, `action_values`, `features` (đặc tả từng biến: số hay phân loại, danh sách level), `params`, và `leaves` (mỗi lá: `leaf`, `rule`, `action`, `accounts`, `conditions` theo từng biến: `{lo, hi}` với cận dưới loại trừ và cận trên bao gồm, hoặc `{in, not_in}`).

**Giới hạn:** chỉ **một biến quyết định có hai phương án**; chỉ `constrained_growth="posthoc"`.

## 10. So sánh và đăng ký kết quả

Mỗi `Scenario` tự đăng ký trong `Project` theo tên (tên trùng thì cái sau thay cái trước, **trừ** kịch bản tạo từ `with_constraints` không có tên, để biến thể "nếu đổi" không chiếm tên của bản gốc).

```python
project.register(scenario, name=None)             # đặt (hoặc đổi) tên mà report() sẽ tìm thấy
project.attach_tree(name, tree, description="", source_scenario=None)   # đăng ký chính sách của cây
rep = project.report(scenarios=[...], reference="BAU")
rep.scenario_summary()
```

`scenario_summary()` trả về mỗi kịch bản một dòng: `flow`, số tài khoản, lợi nhuận kỳ vọng toàn danh mục (tỷ), trung bình mỗi tài khoản, tỷ lệ duyệt, và so với tham chiếu (tuyệt đối và phần trăm). Tên bắt đầu bằng `_` là kịch bản nội bộ và không xuất hiện trong báo cáo mặc định.

## 11. Lỗi

| Lỗi | Khi nào |
|---|---|
| `PyDOError` | Lớp cơ sở của mọi lỗi cố ý |
| `ProjectError` | Gán sai thứ tự, mục tiêu không tồn tại, tham số kịch bản sai, thiếu build, chính sách ngoài miền |
| `UnknownFactError` | Hàm đọc fact chưa khai báo (cũng là `KeyError`) |
| `ConstraintError` | Luật sai cú pháp hoặc vi phạm quy tắc (strict, `WHERE` trên biến `v_`, ô rỗng, biến không tồn tại, `QUICK_WEIGHTED_AVG` có trọng số ≤ 0) |
| `SolverError` | Solver dừng không có nghiệm, solver không hỗ trợ |
| `InfeasibleModelError` | Không có phương án thỏa mọi luật. Thông báo chỉ dẫn các bước: kiểm tra hiệu chuẩn PD trước, thêm luật từng cái một, tìm ô rỗng, so C-duyệt với C-rủi ro; **không** gợi ý nới ngưỡng ngẫu nhiên |

## 12. Tham chiếu API

| Lớp | Phương thức và thuộc tính chính |
|---|---|
| `Portfolio(frame, key_column, random_sampling, stratify_columns, random_state)` | `full`, `data`, `columns`, `n_full`, `n_active`, `sampling_fraction`, `scale` |
| `Facts()` | `set(name, value, description)`, `get(name, default)`, `names()`, `as_dict()`, `copy()` |
| `Variables()` | `set(name, domain)`, `names`, `domain(name)`, `treatments()` |
| `Functions(sources)` | `run(obj, key)`, `labels()` (đường dẫn tệp hoặc các hàm) |
| `Constraints(description)` | `set(name, rule, description)`, `get`, `names()`, `copy()`, lặp qua các luật |
| `Project(name, description)` | `facts`, `variables`, `portfolio`, `functions`, `objective`, `direction`, `output_variables`, `calculated_variables`, `scenario(...)`, `register`, `attach_tree`, `report` |
| `Scenario` | `build`, `preflight`, `optimize`, `with_constraints`, `apply_policy`, `solver_report`, `constraint_report`, `coefficient_report`, `export`, `summary`; thuộc tính `pairs`, `selected`, `score`, `total`, `total_portfolio`, `info` |
| `DecisionTree` | xem mục 9 |
| `Report` | `scenario_summary()` |

## 13. Kiểm thử

43 test trong ba file, chạy bằng `unittest` (không cần pytest): `python -m unittest discover -s tests -v`.

| File | Số test | Nội dung chính |
|---|---|---|
| `tests/test_phase1.py` | 16 | Khối cơ bản và hai flow đầu: `Portfolio` (xác thực khóa, lấy mẫu phân tầng, tầng nhỏ vẫn giữ, tính tất định, `scale`), `Facts`, `Variables`, `Functions` (thứ tự nạp, lỗi nêu tên hàm và tài khoản), thứ tự gán của `Project`, `simulation` khớp tính tay, `unconstrained` chọn đúng các khoản có lợi nhuận dương, quy tắc hòa, quy đổi mẫu về toàn bộ |
| `tests/test_phase2.py` | 18 | Bộ phân tích luật (hợp lệ và bị từ chối), **so lời giải với vét cạn độc lập** (8 danh mục ngẫu nhiên 12 đơn và một bài ba phương án 7 đơn, gồm cả trường hợp vô nghiệm), luật cấp tài khoản tỉa đúng, `WHERE` trên biến `v_`, ô rỗng, `QUICK_WEIGHTED_AVG`, báo cáo vi phạm của chính sách cố định, cờ binding, cảnh báo ma trận mong manh, ngưỡng viết theo đầu đơn đúng khi chạy mẫu, `with_constraints` dùng lại các cặp và khớp một kịch bản tạo mới |
| `tests/test_phase5.py` | 9 | `DecisionTree`: **tối ưu theo lá khớp vét cạn mọi cách gán lá**, chi phí và mất mát theo lá cộng đúng, phân hoạch và vòng JSON, luật số nguyên đọc tự nhiên, lỗi khi phân hoạch quá thô, kiểm tra đầu vào; `apply_policy` dùng đúng bộ đánh giá; `attach_tree` và `report`; ba phương án không hỗ trợ chưng cất |

Vét cạn liệt kê mọi tổ hợp quyết định (hoặc mọi tổ hợp hành động của lá) và tính lại các ràng buộc bằng công thức numpy riêng, không dùng mã của gói. Ngoài các test, `pydo` được chạy trên bài thật (17.976 đơn, 42 luật) với bộ kiểm tra ràng buộc độc lập trong notebook Challenge 2.

## 14. Giới hạn đã biết

- `n_jobs` được nhận nhưng **bị bỏ qua** (một tiến trình); không ảnh hưởng kết quả, chỉ ảnh hưởng tốc độ. `scipy.optimize.milp` không cho đặt số luồng.
- Chỉ có solver `highs`.
- `DecisionTree` chỉ hỗ trợ một biến quyết định có hai phương án; `apply_policy` chỉ một biến.
- Phép tổng hợp chỉ nhận một tên biến (không biểu thức), `WEIGHTED_AVG` chỉ so với số.
- Không có `project.save` (lưu cả dự án thành zip) và không có exporter `.fjdt` của FICO; có JSON và SQL thay thế.
- Tìm giá bóng bằng cách giải lại (`with_constraints`), không lấy từ đối ngẫu LP.
- Không tự đổi thang đơn vị; người dùng chọn đơn vị để ma trận đủ lành (xem `coefficient_report`).
- Hàm kinh tế chạy từng cặp bằng Python thuần: 36.000 cặp mất vài giây; mô hình lớn hơn nhiều sẽ cần vector hóa.
- Chưa được đối chiếu với một công cụ tối ưu quyết định khác.
