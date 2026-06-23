"""
Dự đoán 1 domain là DGA (độc hại) hay lành tính bằng model Bilbo đã train.

Cách dùng:
    # Dự đoán vài domain trực tiếp
    python src/predict.py google.com mortiscontrastatim.com

    # Đọc danh sách domain từ file (mỗi dòng 1 domain)
    python src/predict.py --file domains.txt

    # Chế độ tương tác: gõ từng domain, Enter để dự đoán, 'quit' để thoát
    python src/predict.py

    # Chỉ định model cụ thể (mặc định: thư mục saved_models/)
    python src/predict.py --model saved_models/bilbo_run1.keras google.com
"""
import os
import sys
import glob
import argparse

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # bớt log TensorFlow

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

from preprocess import encode_domain  # encode giống hệt lúc train

DEFAULT_MODEL_DIR = os.path.join(BASE, 'saved_models')
THRESHOLD = 0.5


def clean_domain(raw: str) -> str:
    """Chuẩn hóa input về dạng giống data train: bỏ scheme, path, port, www."""
    d = raw.strip().lower()
    if '://' in d:           # bỏ http:// https://
        d = d.split('://', 1)[1]
    d = d.split('/', 1)[0]   # bỏ path
    d = d.split('?', 1)[0]   # bỏ query
    d = d.split(':', 1)[0]   # bỏ port
    if d.startswith('www.'):
        d = d[4:]
    return d


def load_models(model_path: str):
    """
    Nạp model. Nếu là file .keras -> 1 model.
    Nếu là thư mục -> nạp tất cả bilbo_run*.keras để ensemble (trung bình).
    """
    from tensorflow.keras.models import load_model

    if os.path.isfile(model_path):
        paths = [model_path]
    else:
        paths = sorted(glob.glob(os.path.join(model_path, 'bilbo_run*.keras')))
        if not paths:  # fallback: bất kỳ .keras nào trong thư mục
            paths = sorted(glob.glob(os.path.join(model_path, '*.keras')))

    if not paths:
        sys.exit(f"[Lỗi] Không tìm thấy model (.keras) tại: {model_path}\n"
                 f"      Hãy train trước (python src/train.py) hoặc tải model về.")

    print(f"[Model] Nạp {len(paths)} model: {[os.path.basename(p) for p in paths]}")
    return [load_model(p) for p in paths]


def predict(models, domains):
    """Trả về (domain_đã_chuẩn_hóa, xác_suất_DGA) cho mỗi domain."""
    cleaned = [clean_domain(d) for d in domains]
    X = np.array([encode_domain(d) for d in cleaned], dtype='int32')
    # Ensemble: trung bình xác suất của các model
    probs = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
    return cleaned, probs


def print_results(domains, probs, threshold=THRESHOLD):
    print(f"\n  {'Domain':<40} {'Score':>8}   Kết luận")
    print("  " + "-" * 70)
    for d, p in zip(domains, probs):
        if p >= threshold:
            verdict = f"⚠  DGA (độc hại)  [tin cậy {p*100:.1f}%]"
        else:
            verdict = f"✓  Lành tính       [tin cậy {(1-p)*100:.1f}%]"
        print(f"  {d:<40} {p:>8.4f}   {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dự đoán domain DGA bằng Bilbo")
    parser.add_argument('domains', nargs='*', help="Các domain cần kiểm tra")
    parser.add_argument('--file', help="File chứa danh sách domain (mỗi dòng 1 domain)")
    parser.add_argument('--model', default=DEFAULT_MODEL_DIR,
                        help="Đường dẫn file .keras hoặc thư mục chứa model")
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help="Ngưỡng phân loại (mặc định 0.5)")
    args = parser.parse_args()

    models = load_models(args.model)

    # Gom danh sách domain từ tham số và/hoặc file
    domains = list(args.domains)
    if args.file:
        with open(args.file) as f:
            domains += [line.strip() for line in f if line.strip()]

    if domains:
        cleaned, probs = predict(models, domains)
        print_results(cleaned, probs, args.threshold)
    else:
        # Chế độ tương tác
        print("\n=== Chế độ tương tác — gõ domain rồi Enter ('quit' để thoát) ===")
        while True:
            try:
                raw = input("domain> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.lower() in ('quit', 'exit', 'q'):
                break
            cleaned, probs = predict(models, [raw])
            print_results(cleaned, probs, args.threshold)


if __name__ == '__main__':
    main()
