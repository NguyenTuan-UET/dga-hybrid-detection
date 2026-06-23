import os
import sys
import importlib.util
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

from preprocess import (build_dataset, split_data, load_dga_domains, load_alexa_domains,
                        encode_domain, build_generalizability_dataset,
                        RANDOM_SEED, DICTIONARY_FAMILIES)

# ── Cấu hình (đúng theo paper) ─────────────────────────────────────────────
BATCH_SIZE = 512
MAX_EPOCHS = 10
N_RUNS     = 3      # train 3 lần, lấy trung bình
THRESHOLD  = 0.5

# ── Đường dẫn ──────────────────────────────────────────────────────────────
DGA_PATH    = os.path.join(BASE, 'DGA_domains_dataset', 'dga_domains_full.csv')
ALEXA_PATH  = os.path.join(BASE, 'data', 'alexa-top-1m.csv')
SAMPLE_PATH = os.path.join(BASE, 'DGA_domains_dataset', 'dga_domains_sample.csv')
BILBO_PATH  = os.path.join(BASE, 'models', 'bilbo-hybrid', 'bilbo-hybrid.py')
SAVE_DIR    = os.path.join(BASE, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)


def load_bilbo():
    spec = importlib.util.spec_from_file_location('bilbo', BILBO_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_bilbo


def evaluate(model, X_holdout, y_holdout):
    y_prob = model.predict(X_holdout, verbose=0).flatten()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    acc = accuracy_score(y_holdout, y_pred)
    f1  = f1_score(y_holdout, y_pred)
    auc = roc_auc_score(y_holdout, y_prob)
    fpr = np.sum((y_pred == 1) & (y_holdout == 0)) / np.sum(y_holdout == 0)

    return {'accuracy': acc, 'f1': f1, 'auc': auc, 'fpr': fpr}


def train_bilbo(X_train, y_train, X_test, y_test, X_holdout, y_holdout):
    build_fn   = load_bilbo()
    all_results = []

    for run in range(1, N_RUNS + 1):
        print(f"\n{'='*50}")
        print(f"  Bilbo — Run {run}/{N_RUNS}")
        print(f"{'='*50}")

        model     = build_fn()
        save_path = os.path.join(SAVE_DIR, f'bilbo_run{run}.keras')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        model.fit(
            X_train, y_train,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        results = evaluate(model, X_holdout, y_holdout)
        all_results.append(results)

        print(f"\n  Run {run} holdout → "
              f"Acc: {results['accuracy']:.4f} | "
              f"F1: {results['f1']:.4f} | "
              f"AUC: {results['auc']:.4f} | "
              f"FPR: {results['fpr']:.4f}")

    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}

    print(f"\n{'='*50}")
    print(f"  BILBO — Trung bình {N_RUNS} lần chạy")
    print(f"{'='*50}")
    print(f"  Accuracy : {avg['accuracy']:.4f}")
    print(f"  F1 Score : {avg['f1']:.4f}")
    print(f"  AUC      : {avg['auc']:.4f}")
    print(f"  FPR      : {avg['fpr']:.4f}")
    print(f"{'='*50}")

    return avg


def run_generalizability_trial(train_families, test_family, dga_path, trial_num):
    """1 trial của leave-one-out: train trên N-1 families, test trên 1 unseen family."""
    print(f"\n{'='*60}")
    print(f"  Trial {trial_num}: train={train_families}")
    print(f"             test ={test_family} (unseen)")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test = build_generalizability_dataset(
        dga_path, ALEXA_PATH, train_families, test_family
    )

    build_fn    = load_bilbo()
    all_results = []

    for run in range(1, N_RUNS + 1):
        print(f"\n  -- Run {run}/{N_RUNS} --")
        model     = build_fn()
        save_path = os.path.join(SAVE_DIR, f'bilbo_gen_t{trial_num}_run{run}.keras')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        model.fit(
            X_train, y_train,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        results = evaluate(model, X_test, y_test)
        all_results.append(results)
        print(f"  Run {run} → Acc:{results['accuracy']:.4f} F1:{results['f1']:.4f} "
              f"AUC:{results['auc']:.4f} FPR:{results['fpr']:.4f}")

    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}
    return avg


def generalizability_test(dga_path):
    """Experiment 2: Generalizability Test — leave-one-out trên 4 families."""
    print("\n" + "="*60)
    print("  EXPERIMENT 2: GENERALIZABILITY TEST")
    print("  Leave-one-out: train trên 3 families, test trên 1 unseen")
    print("="*60)

    trial_results = []
    for i, test_family in enumerate(DICTIONARY_FAMILIES, 1):
        train_families = [f for f in DICTIONARY_FAMILIES if f != test_family]
        avg = run_generalizability_trial(train_families, test_family, dga_path, trial_num=i)
        trial_results.append({'test_family': test_family, **avg})

    # Bảng tổng kết
    print("\n" + "="*60)
    print("  EXPERIMENT 2 — SUMMARY")
    print("="*60)
    header = f"  {'Unseen Family':<14} {'Accuracy':>9} {'F1':>9} {'AUC':>9} {'FPR':>9}"
    print(header)
    print("  " + "-"*52)
    for r in trial_results:
        print(f"  {r['test_family']:<14} {r['accuracy']:>9.4f} {r['f1']:>9.4f} "
              f"{r['auc']:>9.4f} {r['fpr']:>9.4f}")
    print("  " + "-"*52)
    means = {k: np.mean([r[k] for r in trial_results]) for k in ['accuracy', 'f1', 'auc', 'fpr']}
    print(f"  {'MEAN':<14} {means['accuracy']:>9.4f} {means['f1']:>9.4f} "
          f"{means['auc']:>9.4f} {means['fpr']:>9.4f}")
    print("="*60)

    return trial_results


if __name__ == '__main__':
    USE_SAMPLE = '--sample' in sys.argv   # python train.py --sample
    RUN_EXP2   = '--exp2'   in sys.argv   # python train.py --exp2 [--sample]

    dga_path = SAMPLE_PATH if USE_SAMPLE else DGA_PATH
    mode_str = "SAMPLE" if USE_SAMPLE else "FULL"

    if RUN_EXP2:
        print(f"=== EXPERIMENT 2: GENERALIZABILITY TEST [{mode_str}] ===\n")
        generalizability_test(dga_path)
    else:
        print(f"=== EXPERIMENT 1: CLASSIFICATION [{mode_str}] ===\n")

        if USE_SAMPLE:
            dga_df   = load_dga_domains(SAMPLE_PATH)
            alexa_df = load_alexa_domains(ALEXA_PATH, n_samples=len(dga_df))
            df       = pd.concat([dga_df, alexa_df]).sample(frac=1, random_state=RANDOM_SEED)
            X = np.array([encode_domain(d) for d in df['domain']], dtype='int32')
            y = np.array(df['target'], dtype='int32')
        else:
            X, y, _ = build_dataset(DGA_PATH, ALEXA_PATH)

        X_train, X_test, X_holdout, y_train, y_test, y_holdout = split_data(X, y)
        train_bilbo(X_train, y_train, X_test, y_test, X_holdout, y_holdout)
