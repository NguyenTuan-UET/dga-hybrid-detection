# Bilbo DGA Detector

Phát hiện domain **Dictionary DGA** (domain do malware sinh tự động bằng cách ghép từ
trong từ điển) bằng mô hình deep learning lai **CNN + LSTM** ("Bilbo"). Repo gồm:

- **Web demo** (Gradio) — nhập domain, xem điểm DGA trực quan.
- **CLI dự đoán** — chấm điểm domain từ tham số hoặc file.
- **Script train lại** model trên tập dữ liệu kèm sẵn.

> Phần học thuật (tóm tắt paper, nguồn dữ liệu, các kiến trúc so sánh) nằm ở
> [cuối README](#về-mô-hình-bilbo-paper-gốc).

---

## 1. Cấu trúc thư mục

```
bilbo-bagging-hybrid/
├── app.py                   # Web demo (Gradio)
├── requirements.txt         # Thư viện cần cài
├── src/
│   ├── predict.py           # Dự đoán domain (CLI)
│   ├── train.py             # Train lại model
│   └── preprocess.py        # Encode domain giống lúc train
├── models/                  # Kiến trúc các mô hình (bilbo, cnn, lstm, ann, mit)
├── saved_models/            # Model .keras đã train  ⚠️ KHÔNG kèm trong git
├── DGA_domains_dataset/     # Dữ liệu DGA (đã có sẵn trong repo)
├── data/alexa-top-1m.csv    # Domain lành tính (đã có sẵn trong repo)
└── domains_test.txt         # File domain mẫu để thử predict --file
```

## 2. Yêu cầu

- **Python 3.10 – 3.12** (khuyến nghị 3.12). *Lưu ý: TensorFlow chưa hỗ trợ tốt 3.13.*
- **git** để clone repo.
- GPU **không bắt buộc** — demo và dự đoán chạy tốt trên CPU. Train tập đầy đủ trên
  CPU khá chậm → nên dùng GPU (xem [phần Train](#6-train-lại-model)).

---

## 3. Cài đặt

Các lệnh giống nhau giữa hai hệ điều hành, chỉ khác cách tạo/kích hoạt môi trường ảo.

### 🐧 Ubuntu / Linux

```bash
# (nếu chưa có) cài python venv
sudo apt update && sudo apt install -y python3-venv git

git clone https://github.com/NguyenTuan-UET/dga-hybrid-detection.git
cd dga-hybrid-detection

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 🪟 Windows (PowerShell)

```powershell
git clone https://github.com/NguyenTuan-UET/dga-hybrid-detection.git
cd dga-hybrid-detection

py -m venv venv
venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> **Nếu PowerShell chặn `Activate.ps1`** ("running scripts is disabled"), chạy 1 lần:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Hoặc dùng Command Prompt (cmd) thay cho PowerShell: `venv\Scripts\activate.bat`

Khi đã kích hoạt venv, dấu nhắc sẽ có tiền tố `(venv)`. Mỗi lần mở terminal mới để
chạy dự án, nhớ kích hoạt lại venv trước.

---

## 4. ⚠️ Lấy model trước khi chạy demo / dự đoán

Model đã train (`saved_models/*.keras`) **không được commit lên git** (xem `.gitignore`).
Bản clone mới sẽ chưa có model, nên `app.py` và `predict.py` sẽ báo lỗi *"Không tìm thấy
model (.keras)"* cho tới khi bạn có model. Chọn **một** trong hai cách:

1. **Đã có sẵn model** (bạn vừa train, hoặc được chia sẻ): đặt các file `bilbo_run*.keras`
   vào thư mục `saved_models/` là xong.
2. **Tự train** rồi lấy model: làm theo [phần 6](#6-train-lại-model). Nhanh nhất là train
   trên Kaggle (GPU T4 miễn phí) — xem hướng dẫn chi tiết ở [KAGGLE.md](KAGGLE.md), sau đó
   tải `saved_models.zip` về và giải nén vào thư mục `saved_models/`.

> Script dự đoán dùng **ensemble** (trung bình) các file `bilbo_run*.keras` để kết quả ổn định.

---

## 5. Chạy

> Tất cả lệnh dưới đây chạy từ thư mục gốc dự án, **sau khi đã kích hoạt venv**.
> Ubuntu dùng `python3`, Windows dùng `python` (hoặc `py`) — ngoài ra giống hệt nhau.

### 5.1. Web demo (giao diện)

```bash
python app.py          # Ubuntu: python3 app.py
```

Lần đầu sẽ nạp model (vài giây đến vài chục giây tùy máy), khi thấy dòng `Sẵn sàng!` thì
mở trình duyệt vào **http://127.0.0.1:7860**. Nhập domain (mỗi dòng một domain), bấm
**Kiểm tra** để xem điểm DGA và phân loại.

### 5.2. Dự đoán bằng CLI

```bash
# Vài domain trực tiếp
python src/predict.py google.com mortiscontrastatim.com

# Đọc từ file (mỗi dòng 1 domain) — dùng file mẫu kèm sẵn
python src/predict.py --file domains_test.txt

# Hạ ngưỡng để bắt nhiều DGA lạ hơn (mặc định 0.5)
python src/predict.py --threshold 0.4 believekeylab.in

# Chế độ tương tác: gõ từng domain, Enter để chấm, 'quit' để thoát
python src/predict.py
```

---

## 6. Train lại model

Dữ liệu đã nằm sẵn trong repo (`DGA_domains_dataset/` + `data/alexa-top-1m.csv`), không cần
tải thêm.

```bash
# Train nhanh trên tập mẫu nhỏ (thử nhanh, hợp với CPU)
python src/train.py --sample

# Train đầy đủ (Experiment 1 — Classification). Chậm trên CPU → nên dùng GPU
python src/train.py

# Experiment 2 — Generalizability (leave-one-out trên 4 family DGA)
python src/train.py --exp2
```

Mỗi lần train chạy 3 lượt rồi in trung bình 6 metric (Accuracy, Precision, Recall, F1,
AUC, FPR). Model lưu vào `saved_models/`. Cấu hình (`BATCH_SIZE`, `MAX_EPOCHS`, `N_RUNS`)
nằm ở đầu file `src/train.py`.

**Khuyến nghị dùng Kaggle (GPU T4 miễn phí)** cho train đầy đủ — chi tiết ở [KAGGLE.md](KAGGLE.md).

---

## 7. Khắc phục sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Không tìm thấy model (.keras)` | Chưa có model trong `saved_models/`. Xem [phần 4](#4--lấy-model-trước-khi-chạy-demo--dự-đoán). |
| Cổng 7860 đang bận | Tắt tiến trình `app.py` cũ, hoặc để Gradio tự nhảy sang cổng khác (7861...). |
| Giao diện demo bị nền tối / xám | App đã ép chế độ sáng; thử **refresh** (Ctrl/Cmd+R) tab cũ, hoặc cài `gradio>=6.0` đúng theo `requirements.txt`. |
| Nhiều log/cảnh báo TensorFlow, "Could not find cuda drivers" | Bình thường khi chạy CPU — không phải lỗi. Log đã được giảm bớt sẵn. |
| Windows muốn dùng GPU | TensorFlow bản native Windows không hỗ trợ GPU; dùng **WSL2 (Ubuntu)** nếu cần GPU. Chạy CPU vẫn đủ cho demo/predict. |
| `Activate.ps1` bị chặn trên PowerShell | Chạy `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, hoặc dùng `activate.bat` trong cmd. |

---
---

# Về mô hình Bilbo (paper gốc)

Code to go with the paper ["Real-Time Detection of Dictionary DGA Network Traffic using Deep Learning"](https://arxiv.org/abs/2003.12805) and our presentations at [ShmooCon 2018](https://www.youtube.com/watch?v=99hniQYB6VM), [Deep Learning World 2018](https://www.predictiveanalyticsworld.com/machinelearningtimes/wise-practitioner-deep-learning-world-interview-series-domenic-puzio-capital-one/9315/), and the Australian Cyber Security Centre Conference (2018).

## Paper Abstract

Botnets and malware continue to avoid detection by static rules engines when using domain generation algorithms (DGAs) for callouts to unique, dynamically generated web addresses. Common DGA detection techniques fail to reliably detect DGA variants that combine random dictionary words to create domain names that closely mirror legitimate domains. To combat this, we created a novel hybrid neural network, **Bilbo the "bagging" model**, that analyses domains and scores the likelihood they are generated by such algorithms and therefore are potentially malicious. Bilbo is the first parallel usage of a convolutional neural network (CNN) and a long short-term memory (LSTM) network for DGA detection. Our unique architecture is found to be the most consistent in performance in terms of AUC, F1 score, and accuracy when generalising across different dictionary DGA classification tasks compared to current state-of-the-art deep learning architectures. We validate using reverse-engineered dictionary DGA domains and detail our real-time implementation strategy for scoring real-world network logs within a large financial enterprise. In four hours of actual network traffic, the model discovered at least five potential command-and-control networks that commercial vendor tools did not flag.

## Data

The data used for the experiments documented are from DGArchive, Alexa Top 1 Million, and from live enterprise logs. While we cannot publish the enterprise logs, we can publish/direct you to the other data sets.

### [DGArchive](https://dgarchive.caad.fkie.fraunhofer.de/welcome/)

DGArchive is lead by Daniel Plohmann when we needed the data for our experiments. If you would like access to the same data, you must request it through him. See the details on how on [their website](https://dgarchive.caad.fkie.fraunhofer.de/welcome/).

### Alexa Top 1 Million

This dataset was collected in an Amazon S3 Bucket, but appears to have stopped being released since we retrieved it in 2017. We have included it here in this repository for your convenience.

## Models

Based on other experiments done on DGA detection, we compared Bilbo to four other deep learning model architectures:

* Artificial Neural Network (ANN) - Single layer
* Long Short-Term Memory (LSTM) Network
* Convolutional Neural Network (CNN)
* MIT's CNN-LSTM Hybrid Model - adapted in Vosoughi, et al. (2016 - Tweet2vec) and Yu, et al. (2018)

More details on each model, including the Keras/Tensorflow architecture we used with Python 3.7, is available within the `models/` directory.

## Acknowledgements

Thank you to Capital One for the incredible opportunity to deploy a machine learning model developed for research into a live environment for evaluation. To [Jason Trost](https://medium.com/@jason_trost), your mentorship and intellectual curiosity inspires everyone around you. We appreciate your and Capital One's support to publish our work as an academic paper after our talks in industry.

To the reviewers at our last attempted venues, thank you for the incredible feedback that greatly improved our analysis.
