# Train Bilbo trên Kaggle (GPU T4 miễn phí)

Toàn bộ code + data đã nằm trong repo này, nên trên Kaggle chỉ cần
clone về và chạy. Không cần upload Kaggle Dataset riêng.

## Bước 1 — Tạo Notebook

1. Vào https://www.kaggle.com/code → **New Notebook**
2. Bên phải, mục **Settings**:
   - **Accelerator** → chọn **GPU T4 x2** (hoặc P100)
   - **Internet** → bật **On** (để `git clone` được)

## Bước 2 — Clone repo

Tạo 1 cell và chạy:

```python
!git clone https://github.com/NguyenTuan-UET/dga-hybrid-detection.git
%cd dga-hybrid-detection
```

> Nếu repo là private, dùng token:
> `!git clone https://<TOKEN>@github.com/NguyenTuan-UET/dga-hybrid-detection.git`

## Bước 3 — Kiểm tra GPU

```python
import tensorflow as tf
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
```

Phải thấy ít nhất 1 GPU. Nếu list rỗng → quay lại Settings bật Accelerator.

(Kaggle đã cài sẵn tensorflow/sklearn/pandas/numpy nên **không cần**
`pip install`. Nếu muốn chắc chắn đúng version: `!pip install -r requirements.txt`)

## Bước 4 — Train

**Experiment 1 — Classification (4 family chung):**

```python
!python src/train.py
```

**Experiment 2 — Generalizability (leave-one-out 4 family):**

```python
!python src/train.py --exp2
```

Mỗi experiment train Bilbo 3 lần rồi in trung bình 6 metric
(Accuracy, Precision, Recall, F1, AUC, FPR).

## Bước 5 — Lấy model đã train về (tùy chọn)

Model lưu ở `saved_models/*.keras`. Để tải về máy:

```python
from IPython.display import FileLink
import shutil
shutil.make_archive('/kaggle/working/saved_models', 'zip', 'saved_models')
FileLink('/kaggle/working/saved_models.zip')
```

Hoặc nhấn **Output** trong notebook để tải file.

---

## Dùng model để dự đoán domain mới

Sau khi train (model nằm ở `saved_models/`), chạy:

```python
# Dự đoán vài domain
!python src/predict.py google.com mortiscontrastatim.com

# Đọc từ file (mỗi dòng 1 domain)
!python src/predict.py --file domains.txt
```

Tải model về máy thì chạy `predict.py` y hệt ở local. Script tự
ensemble 3 model (trung bình `bilbo_run*.keras`) cho kết quả ổn định.

## Lưu ý

- **Giới hạn free:** ~30h GPU/tuần, mỗi phiên tối đa 12h. Đủ thừa cho
  job này (vài phút đến vài chục phút trên T4).
- **Dữ liệu:** gozi/matsnu/suppobox = 13.500 domain mỗi family,
  bigviktor = 8.000, cộng Alexa cân bằng.
- Sửa cấu hình train ở đầu `src/train.py`: `BATCH_SIZE`, `MAX_EPOCHS`,
  `N_RUNS`.
