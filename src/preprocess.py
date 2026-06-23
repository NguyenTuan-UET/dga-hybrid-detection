import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Hằng số (đúng theo paper) ──────────────────────────────────────────────
MAX_STRING_LENGTH = 63   # độ dài tối đa 1 domain
MAX_INDEX         = 42   # số lượng ký tự hợp lệ + padding + buffer
RANDOM_SEED       = 42

# 40 ký tự hợp lệ trong domain: a-z, 0-9, hyphen, dot, underscore, tilde
# Index 0 dành cho padding, index 1-40 cho ký tự, index 41 cho ký tự lạ
VALID_CHARS = {c: i + 1 for i, c in enumerate(
    'abcdefghijklmnopqrstuvwxyz0123456789-._~@!#$%^&*()'[:40]
)}

# 4 họ dictionary DGA mà paper tập trung
DICTIONARY_FAMILIES = ['gozi', 'matsnu', 'suppobox', 'bigviktor']


# ── Bước 1: Encode domain thành mảng số ────────────────────────────────────

def encode_domain(domain: str) -> list:
    """
    Chuyển chuỗi domain thành mảng số nguyên độ dài MAX_STRING_LENGTH.
    Ký tự không hợp lệ → index 41. Padding bằng 0 ở đầu.

    Ví dụ: "google.com" → [0, 0, ..., 7, 15, 15, 7, 12, 5, 38, 3, 15, 13]
    """
    domain = domain.lower().strip()
    encoded = [VALID_CHARS.get(c, MAX_INDEX - 1) for c in domain]

    # Cắt nếu dài hơn MAX_STRING_LENGTH
    encoded = encoded[:MAX_STRING_LENGTH]

    # Padding 0 vào đầu cho đủ MAX_STRING_LENGTH
    padding = [0] * (MAX_STRING_LENGTH - len(encoded))
    return padding + encoded


# ── Bước 2: Load dữ liệu ───────────────────────────────────────────────────

def load_dga_domains(dga_path: str, families: list = DICTIONARY_FAMILIES) -> pd.DataFrame:
    """
    Đọc dga_domains_full.csv, lọc ra các họ dictionary DGA cần thiết.
    Format file: label(dga/legit), family, domain
    """
    df = pd.read_csv(dga_path, header=None, names=['label', 'family', 'domain'])
    df = df[df['family'].isin(families)].copy()
    df['target'] = 1  # malicious
    print(f"[DGA] Loaded {len(df)} domains từ {families}")
    print(df['family'].value_counts().to_string())
    return df[['domain', 'target', 'family']]


def load_alexa_domains(alexa_path: str, n_samples: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Đọc alexa-top-1m.csv, sample ngẫu nhiên n_samples domain.
    Format file: rank, domain
    """
    df = pd.read_csv(alexa_path, header=None, names=['rank', 'domain'])
    df = df.sample(n=n_samples, random_state=seed).copy()
    df['target'] = 0  # benign
    df['family'] = 'alexa'
    print(f"[Alexa] Sampled {len(df)} benign domains")
    return df[['domain', 'target', 'family']]


# ── Bước 3: Ghép và xử lý toàn bộ dataset ─────────────────────────────────

def build_dataset(dga_path: str, alexa_path: str, seed: int = RANDOM_SEED):
    """
    Tạo dataset cân bằng 50/50 từ DGA và Alexa.
    Trả về: X (mảng số), y (nhãn 0/1), families (tên họ DGA)
    """
    # Load malicious
    dga_df = load_dga_domains(dga_path)
    n_malicious = len(dga_df)

    # Load benign với số lượng bằng malicious
    alexa_df = load_alexa_domains(alexa_path, n_samples=n_malicious, seed=seed)

    # Gộp lại
    df = pd.concat([dga_df, alexa_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    print(f"\n[Dataset] Tổng: {len(df)} domains | Malicious: {n_malicious} | Benign: {n_malicious}")

    # Encode từng domain thành mảng số
    print("[Encode] Đang encode domains...")
    X = np.array([encode_domain(d) for d in df['domain']], dtype='int32')
    y = np.array(df['target'], dtype='int32')
    families = np.array(df['family'])

    return X, y, families


# ── Bước 3b: Dataset cho Generalizability Test ─────────────────────────────

def build_generalizability_dataset(dga_path: str, alexa_path: str,
                                   train_families: list, test_family: str,
                                   seed: int = RANDOM_SEED):
    """
    Xây dựng dataset cho Experiment 2 (Leave-one-out Generalizability Test).
    - Train set: DGA từ train_families + Alexa cân bằng (split 90/10 train/val)
    - Test set:  DGA từ test_family (unseen) + Alexa cân bằng
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # ── Train set ─────────────────────────────────────────────────────────
    train_dga   = load_dga_domains(dga_path, families=train_families)
    n_train     = len(train_dga)
    train_alexa = load_alexa_domains(alexa_path, n_samples=n_train, seed=seed)

    df_train = pd.concat([train_dga, train_alexa], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    X_all = np.array([encode_domain(d) for d in df_train['domain']], dtype='int32')
    y_all = np.array(df_train['target'], dtype='int32')

    # 90% train, 10% val (dùng cho EarlyStopping)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=seed, stratify=y_all
    )

    print(f"\n[Gen-Train] Families: {train_families}")
    print(f"            Train={len(X_train)} | Val={len(X_val)}")

    # ── Test set (unseen family) ───────────────────────────────────────────
    test_dga   = load_dga_domains(dga_path, families=[test_family])
    n_test     = len(test_dga)
    test_alexa = load_alexa_domains(alexa_path, n_samples=n_test, seed=seed + 1000)

    df_test = pd.concat([test_dga, test_alexa], ignore_index=True)
    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    X_test = np.array([encode_domain(d) for d in df_test['domain']], dtype='int32')
    y_test = np.array(df_test['target'], dtype='int32')

    print(f"[Gen-Test]  Family: {test_family} → {len(X_test)} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Bước 4: Split train / test / holdout ───────────────────────────────────

def split_data(X, y, test_size=0.1, holdout_size=0.1, seed: int = RANDOM_SEED):
    """
    Chia dữ liệu thành 3 tập đúng theo paper:
      - train:   80%
      - test:    10%  (dùng trong quá trình train / early stopping)
      - holdout: 10%  (chỉ dùng sau khi train xong để báo cáo kết quả)
    """
    # Tách holdout trước
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y, test_size=holdout_size, random_state=seed, stratify=y
    )

    # Tách test từ phần còn lại
    test_ratio = test_size / (1 - holdout_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=seed, stratify=y_temp
    )

    print(f"\n[Split]")
    print(f"  Train:   {len(X_train):>6} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Test:    {len(X_test):>6} ({len(X_test)/len(X)*100:.0f}%)")
    print(f"  Holdout: {len(X_holdout):>6} ({len(X_holdout)/len(X)*100:.0f}%)")

    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


# ── Entry point: chạy thử trực tiếp file này ───────────────────────────────

if __name__ == '__main__':
    import os

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DGA_PATH   = os.path.join(BASE, 'DGA_domains_dataset', 'dga_domains_full.csv')
    ALEXA_PATH = os.path.join(BASE, 'data', 'alexa-top-1m.csv')

    # Dùng sample để test nhanh
    SAMPLE_PATH = os.path.join(BASE, 'DGA_domains_dataset', 'dga_domains_sample.csv')
    USE_SAMPLE  = True  # ← đổi thành False khi train thật

    if USE_SAMPLE:
        print("=== Chạy với SAMPLE (test nhanh) ===\n")
        dga_path = SAMPLE_PATH
        # Sample chỉ có malicious, lấy benign từ alexa
        dga_df   = load_dga_domains(dga_path)
        n        = len(dga_df)
        alexa_df = load_alexa_domains(ALEXA_PATH, n_samples=n)
        df       = pd.concat([dga_df, alexa_df], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
        X        = np.array([encode_domain(d) for d in df['domain']], dtype='int32')
        y        = np.array(df['target'], dtype='int32')
    else:
        print("=== Chạy với FULL dataset ===\n")
        X, y, _ = build_dataset(DGA_PATH, ALEXA_PATH)

    X_train, X_test, X_holdout, y_train, y_test, y_holdout = split_data(X, y)

    print(f"\n[OK] X_train shape: {X_train.shape}")
    print(f"[OK] Mẫu encode: {X_train[0][:10]}... (10 giá trị đầu)")
    print(f"[OK] Label mẫu: {y_train[:5]}")
