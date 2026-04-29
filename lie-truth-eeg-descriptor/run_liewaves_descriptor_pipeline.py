"""
Structured Multi-Domain EEG Descriptor Pipeline for Lie and Truth Classification

This script implements the preprocessing, window segmentation, feature extraction,
classification, and evaluation pipeline used in the MethodsX manuscript.

The implementation includes:
1. deterministic EEG preprocessing,
2. overlapping sliding-window segmentation,
3. structured multi-domain descriptor extraction,
4. fold-wise normalization,
5. window-level and grouped validation,
6. meta-correlation ablation analysis.

The LieWaves dataset is not redistributed in this repository.
Users should download the dataset from the official Mendeley Data repository.
"""

import os, re, time, json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

from scipy.signal import welch, butter, filtfilt, iirnotch, hilbert
from scipy.stats import skew, kurtosis, ttest_rel

from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ========
# CONFIG
# ========
DATA_DIR = "./data/LieWaves"
RAW_DIR = os.path.join(DATA_DIR, "Raw")
SUBJECT_STIMULI_XLSX = os.path.join(DATA_DIR, "Subject_Stimuli.xlsx")

CHAN_ORDER = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]
SF = 128
WINDOW_SIZE = 384
STEP_SIZE = 16                 
N_SPLITS = 5
RANDOM_STATE = 42

# Validation settings
RUN_WINDOW_LEVEL_SKF = True     
RUN_SUBJECT_GROUPK   = False     
RUN_SESSION_GROUPK   = False    
RUN_LOSO             = False    
RUN_NON_OVERLAP      = False    
RUN_META_ABLATION    = True     

# SVM is included in the main window-level evaluation.
# It is disabled by default for optional grouped validation to reduce runtime.
INCLUDE_SVM_IN_STRICT_VALIDATION = False
INCLUDE_SVM_IN_WINDOW_LEVEL = True

# Output
OUT_DIR = "./results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# LDA plot options
LDA_JITTER = 0.02
LDA_POINT_SIZE = 10
LDA_ALPHA = 0.75
LDA_MAX_POINTS = 20000

# ============================================================
# Filtering functions
# ============================================================
def bandpass(x):
    b, a = butter(4, [0.5, 45], btype='bandpass', fs=SF)
    return filtfilt(b, a, x)

def notch(x):
    b, a = iirnotch(50, 30, fs=SF)
    return filtfilt(b, a, x)

def preprocess(arr):
    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(arr.shape[1]):
        out[:, ch] = bandpass(notch(arr[:, ch]))
    return out

# ============================================================
# Feature extraction functions
# ============================================================
def hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0, v1, v2 = np.var(x), np.var(dx), np.var(ddx)
    act = v0
    mob = np.sqrt(v1/v0) if v0 > 0 else 0
    comp = np.sqrt(v2/v1)/mob if (v1 > 0 and mob > 0) else 0
    return act, mob, comp

def wpli(x, y):
    ax, ay = hilbert(x), hilbert(y)
    im = np.imag(ax * np.conj(ay))
    return float(np.abs(np.mean(im)) / (np.mean(np.abs(im)) + 1e-6))

def fast_corr(a, b):
    am = a - a.mean()
    bm = b - b.mean()
    return float(np.dot(am, bm) / (np.std(a) * np.std(b) * len(a) + 1e-6))

def extract_descriptor(window):
    C = window.shape[0]
    temporal, fractal, spectral = [], [], []

    for ch in range(C):
        x = (window[ch] - np.mean(window[ch])) / (np.std(window[ch]) + 1e-6)

        feats = [np.mean(x), np.std(x), skew(x), kurtosis(x)]
        feats += list(hjorth(x))
        temporal += feats

        N = len(x); L = np.sum(np.abs(np.diff(x)))
        fractal.append(float(np.log(N) / (np.log(N) + np.log(L/N) + 1e-6)))

        f, psd = welch(x, SF, nperseg=256)
        bands = [(1,4),(4,8),(8,12),(12,30)]
        abs_bp = np.array([np.sum(psd[(f>=b0) & (f<=b1)]) for b0,b1 in bands], dtype=float)
        rel_bp = abs_bp / (np.sum(abs_bp) + 1e-6)
        sent = -np.sum((psd/np.sum(psd)) * np.log(psd/np.sum(psd) + 1e-12))

        spectral += abs_bp.tolist() + rel_bp.tolist() + [float(sent)]

    spatial = []
    for i, j in combinations(range(C), 2):
        spatial.append(float(np.mean(window[i]) - np.mean(window[j])))
        spatial.append(wpli(window[i], window[j]))

    fTS = np.array(temporal + spatial, dtype=np.float32)
    fCorr = np.array([fast_corr(fTS, np.log1p(np.abs(fTS)))], dtype=np.float32)
    fF = np.array(fractal, dtype=np.float32)
    fSP = np.array(spectral, dtype=np.float32)

    # Feature order:
    # temporal + spatial + meta-correlation + fractal + spectral
    # 35 + 20 + 1 + 5 + 45 = 106 features.
    return np.concatenate([fTS, fCorr, fF, fSP]).astype(np.float32)

# ============================================================
# Feature group definitions
# ============================================================
def group_slices(n_channels=5):
    n_temporal = n_channels * 7
    n_pairs = n_channels * (n_channels - 1) // 2
    n_spatial = n_pairs * 2
    n_fTS = n_temporal + n_spatial
    n_fCorr = 1
    n_fF = n_channels
    n_fSP = n_channels * 9

    s_fTS = slice(0, n_fTS)
    s_fCorr = slice(s_fTS.stop, s_fTS.stop + n_fCorr)
    s_fF = slice(s_fCorr.stop, s_fCorr.stop + n_fF)
    s_fSP = slice(s_fF.stop, s_fF.stop + n_fSP)
    s_full = slice(0, s_fSP.stop)

    return {"fTS": s_fTS, "fCorr": s_fCorr, "fF": s_fF, "fSP": s_fSP, "full": s_full}

SLICES = group_slices(n_channels=len(CHAN_ORDER))

# ============================================================
# FEATURE NAMES — ORIGINAL LOGIC KEPT
# ============================================================
def build_feature_names():
    names = []
    tnames = ["mean","std","skew","kurt","hjorth_act","hjorth_mob","hjorth_comp"]
    for ch in CHAN_ORDER:
        for tn in tnames:
            names.append(f"temporal_{ch}_{tn}")

    pairs = list(combinations(range(len(CHAN_ORDER)), 2))
    for (i,j) in pairs:
        names.append(f"spatial_asym_{CHAN_ORDER[i]}-{CHAN_ORDER[j]}")
        names.append(f"spatial_wpli_{CHAN_ORDER[i]}-{CHAN_ORDER[j]}")

    names.append("fCorr_corr(fTS,log1p(|fTS|))")

    for ch in CHAN_ORDER:
        names.append(f"fractal_{ch}")

    bnames = ["delta","theta","alpha","beta"]
    for ch in CHAN_ORDER:
        for bn in bnames:
            names.append(f"spectral_absbp_{ch}_{bn}")
        for bn in bnames:
            names.append(f"spectral_relbp_{ch}_{bn}")
        names.append(f"spectral_entropy_{ch}")

    return names

FEAT_NAMES = build_feature_names()

# ============================================================
# Data loading and metadata construction
# ============================================================
def load_label_map():
    df = pd.read_excel(SUBJECT_STIMULI_XLSX)
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df[['SUBJECT','SESSION','LIE/TRUTH']]
    df['SUBJECT'] = df['SUBJECT'].astype(str).str.replace('S','', regex=False).astype(int)
    df['SESSION'] = df['SESSION'].astype(str).str.replace('S','', regex=False).astype(int)
    df['KEY'] = list(zip(df['SUBJECT'], df['SESSION']))
    label_map = dict(zip(df['KEY'], df['LIE/TRUTH']))
    print("Total labels loaded:", len(label_map))
    return label_map

def parse_filename(f):
    m = re.match(r"S(\d+)S(\d+)\.csv", f)
    return (int(m.group(1)), int(m.group(2))) if m else None

def sliding_window(data, step_size=STEP_SIZE):
    return [data[:, i:i+WINDOW_SIZE] for i in range(0, data.shape[1]-WINDOW_SIZE+1, step_size)]

def load_data(step_size=STEP_SIZE):
    label_map = load_label_map()
    X, y = [], []
    subjects, sessions, file_groups = [], [], []
    filenames, window_indices = [], []

    for f in sorted(os.listdir(RAW_DIR)):
        key = parse_filename(f)
        if key is None or key not in label_map:
            continue

        subject_id, session_id = key
        df = pd.read_csv(os.path.join(RAW_DIR, f))[CHAN_ORDER]
        arr = preprocess(df.values).T  # (C,T)

        windows = sliding_window(arr, step_size=step_size)
        for wi, w in enumerate(windows):
            X.append(extract_descriptor(w))
            y.append(int(label_map[key]))
            subjects.append(subject_id)
            sessions.append(session_id)
            file_groups.append(f"S{subject_id:02d}_Sess{session_id:02d}")
            filenames.append(f)
            window_indices.append(wi)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    meta = pd.DataFrame({
        "subject": subjects,
        "session": sessions,
        "file_group": file_groups,
        "filename": filenames,
        "window_index": window_indices,
        "label": y
    })
    return X, y, meta

# ============================================================
# Plotting and result saving utilities
# ============================================================
def save_confusion_matrix(y_true, y_pred, title, outpath, labels=("Lie (0)", "Truth (1)")):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=list(labels))
    disp.plot(cmap="Blues", colorbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def paired_ttest_matrix(fold_metrics, metric_key, out_csv, out_png, title):
    model_names = list(fold_metrics.keys())
    n = len(model_names)
    pmat = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                pmat[i, j] = np.nan
            elif i < j:
                a = np.array(fold_metrics[model_names[i]][metric_key], dtype=float)
                b = np.array(fold_metrics[model_names[j]][metric_key], dtype=float)
                if len(a) == len(b) and len(a) > 1:
                    p = ttest_rel(a, b).pvalue
                else:
                    p = np.nan
                pmat[i, j] = p
                pmat[j, i] = p

    dfp = pd.DataFrame(pmat, index=model_names, columns=model_names)
    dfp.to_csv(out_csv, index=True)

    plt.figure()
    im = plt.imshow(pmat, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(n), model_names)
    plt.yticks(range(n), model_names)
    plt.title(title)

    for i in range(n):
        for j in range(n):
            txt = "-" if np.isnan(pmat[i, j]) else f"{pmat[i, j]:.3g}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def lda_plot_like_example(X, y, out_png,
                         max_points=LDA_MAX_POINTS,
                         jitter=LDA_JITTER,
                         seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    if (max_points is not None) and (X.shape[0] > max_points):
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp = X[idx]
        yp = y[idx]
    else:
        Xp = X
        yp = y

    Xs = StandardScaler().fit_transform(Xp)

    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(Xs, yp).ravel()

    y0 = (rng.random(np.sum(yp == 0)) - 0.5) * jitter
    y1 = (rng.random(np.sum(yp == 1)) - 0.5) * jitter

    plt.figure(figsize=(8, 4))
    plt.scatter(z[yp == 0], y0, s=LDA_POINT_SIZE, alpha=LDA_ALPHA, label="Lie (0)")
    plt.scatter(z[yp == 1], y1, s=LDA_POINT_SIZE, alpha=LDA_ALPHA, label="Truth (1)")
    plt.yticks([])
    plt.xlabel("LDA projection (LD1)")
    plt.title("LDA Separability View — Lie vs Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def save_barh(values, names, title, out_png, topk=10):
    idx = np.argsort(values)[::-1][:topk]
    vals = values[idx][::-1]
    nms = [names[i] for i in idx][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), nms)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def rf_importance_by_group(mean_imp, slices_dict, out_csv, out_png):
    rows = []
    for g in ["fTS", "fCorr", "fF", "fSP", "full"]:
        sl = slices_dict[g]
        rows.append((g, float(np.sum(mean_imp[sl]))))

    df = pd.DataFrame(rows, columns=["Group", "Total_importance"])
    df.to_csv(out_csv, index=False)

    plt.figure()
    plt.bar(df["Group"], df["Total_importance"])
    plt.title("RF Importance by Feature Group (Sum of Importances)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ============================================================
# Classifier configuration
# ============================================================
def make_models(include_svm=True):
    models = {
        "RF": RandomForestClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.015,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )

    if include_svm:
        models["SVM"] = SVC(
            C=50,
            gamma="auto",
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )
    return models

# ============================================================
# FEATURE SETS FOR ABLATION
# ============================================================
def get_feature_sets(X):
    idx_full = np.arange(X.shape[1])
    idx_no_meta = np.setdiff1d(idx_full, np.arange(SLICES["fCorr"].start, SLICES["fCorr"].stop))
    feature_sets = {
        "full_106": idx_full,
        "without_meta_corr": idx_no_meta,
    }
    return feature_sets

# ============================================================
# OVERLAP / LEAKAGE DIAGNOSTICS
# ============================================================
def save_overlap_report(step_size, meta, out_json, out_csv):
    overlap_samples = WINDOW_SIZE - step_size
    overlap_percent = overlap_samples / WINDOW_SIZE * 100 if step_size < WINDOW_SIZE else 0.0
    report = {
        "window_size_samples": WINDOW_SIZE,
        "step_size_samples": step_size,
        "sampling_rate_hz": SF,
        "window_duration_seconds": WINDOW_SIZE / SF,
        "step_duration_seconds": step_size / SF,
        "overlap_samples": max(overlap_samples, 0),
        "overlap_percent_adjacent_windows": overlap_percent,
        "n_windows": int(len(meta)),
        "n_subjects": int(meta["subject"].nunique()),
        "n_file_groups": int(meta["file_group"].nunique()),
        "class_counts": {str(k): int(v) for k, v in meta["label"].value_counts().sort_index().items()}
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    meta.groupby(["subject", "label"]).size().reset_index(name="n_windows").to_csv(out_csv, index=False)
    print("\n================ Overlap / Leakage Diagnostic ================")
    print(json.dumps(report, indent=2))
    print("==============================================================\n")
    return report

# ============================================================
# CV EVALUATOR — ADDED FOR REUSABILITY
# ============================================================
def evaluate_cv(X, y, cv, cv_name, out_prefix, groups=None, include_svm=True, feature_indices=None):
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])
    X_eval = X[:, feature_indices]

    models = make_models(include_svm=include_svm)
    fold_metrics = {m: defaultdict(list) for m in models}
    all_true = {m: [] for m in models}
    all_pred = {m: [] for m in models}
    rf_importances = []
    fold_rows = []

    if groups is None:
        split_iter = cv.split(X_eval, y)
    else:
        split_iter = cv.split(X_eval, y, groups=groups)

    for fold, (tr, te) in enumerate(split_iter, start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_eval[tr])
        Xte = scaler.transform(X_eval[te])
        ytr, yte = y[tr], y[te]

        fold_group_info = ""
        if groups is not None:
            test_groups = sorted(set(np.array(groups)[te].tolist()))
            fold_group_info = ",".join(map(str, test_groups[:10]))
            if len(test_groups) > 10:
                fold_group_info += f",...(+{len(test_groups)-10})"

        for name, clf in models.items():
            t0 = time.perf_counter()
            clf.fit(Xtr, ytr)
            t1 = time.perf_counter()

            pred = clf.predict(Xte)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(Xte)[:, 1]
            else:
                proba = pred.astype(float)
            t2 = time.perf_counter()

            acc = accuracy_score(yte, pred)
            rec = recall_score(yte, pred, average="binary", zero_division=0)
            f1  = f1_score(yte, pred, zero_division=0)
            try:
                auc = roc_auc_score(yte, proba)
            except Exception:
                auc = np.nan

            fold_metrics[name]["acc"].append(acc)
            fold_metrics[name]["recall"].append(rec)
            fold_metrics[name]["f1"].append(f1)
            fold_metrics[name]["auc"].append(auc)
            fold_metrics[name]["train_time"].append(t1 - t0)
            fold_metrics[name]["infer_time"].append(t2 - t1)

            all_true[name].extend(yte.tolist())
            all_pred[name].extend(pred.tolist())

            fold_rows.append({
                "cv_name": cv_name,
                "fold": fold,
                "model": name,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "test_groups": fold_group_info,
                "acc": acc,
                "recall": rec,
                "f1": f1,
                "auc": auc,
                "train_time": t1 - t0,
                "infer_time": t2 - t1
            })

            if name == "RF" and hasattr(clf, "feature_importances_"):
                # Importance corresponds to selected feature indices.
                imp_full = np.zeros(X.shape[1], dtype=float)
                imp_full[feature_indices] = clf.feature_importances_
                rf_importances.append(imp_full.copy())

        print(f"[{cv_name}] Fold {fold} done. Test size={len(te)}")

    # Summary
    summary_rows = []
    for name in models:
        acc = np.array(fold_metrics[name]["acc"], dtype=float)
        rec = np.array(fold_metrics[name]["recall"], dtype=float)
        f1  = np.array(fold_metrics[name]["f1"], dtype=float)
        auc = np.array(fold_metrics[name]["auc"], dtype=float)
        trt = np.array(fold_metrics[name]["train_time"], dtype=float)
        inf = np.array(fold_metrics[name]["infer_time"], dtype=float)

        summary_rows.append({
            "CV": cv_name,
            "Feature_Set": out_prefix,
            "Model": name,
            "ACC_mean(%)": float(np.nanmean(acc)*100),
            "ACC_std(%)":  float(np.nanstd(acc)*100),
            "Recall_mean(%)": float(np.nanmean(rec)*100),
            "Recall_std(%)":  float(np.nanstd(rec)*100),
            "F1_mean(%)":  float(np.nanmean(f1)*100),
            "F1_std(%)":   float(np.nanstd(f1)*100),
            "AUC_mean(%)": float(np.nanmean(auc)*100),
            "AUC_std(%)":  float(np.nanstd(auc)*100),
            "Train_mean(s)": float(np.nanmean(trt)),
            "Train_std(s)":  float(np.nanstd(trt)),
            "Infer_mean(s)": float(np.nanmean(inf)),
            "Infer_std(s)":  float(np.nanstd(inf)),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_folds = pd.DataFrame(fold_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, f"summary_{out_prefix}_{cv_name}.csv"), index=False)
    df_folds.to_csv(os.path.join(TAB_DIR, f"fold_metrics_{out_prefix}_{cv_name}.csv"), index=False)

    print(f"\n================ Summary: {cv_name} | {out_prefix} ================")
    for r in summary_rows:
        print(f"\nModel: {r['Model']}")
        print(f"ACC   : {r['ACC_mean(%)']:.2f}% ± {r['ACC_std(%)']:.2f}%")
        print(f"Recall: {r['Recall_mean(%)']:.2f}% ± {r['Recall_std(%)']:.2f}%")
        print(f"F1    : {r['F1_mean(%)']:.2f}% ± {r['F1_std(%)']:.2f}%")
        print(f"AUC   : {r['AUC_mean(%)']:.2f}% ± {r['AUC_std(%)']:.2f}%")
        print(f"Train : {r['Train_mean(s)']:.4f}s ± {r['Train_std(s)']:.4f}s")
        print(f"Infer : {r['Infer_mean(s)']:.4f}s ± {r['Infer_std(s)']:.4f}s")
    print("==============================================================\n")

    # Confusion matrices
    for name in models:
        save_confusion_matrix(
            np.array(all_true[name]), np.array(all_pred[name]),
            title=f"LieWaves — {cv_name} Confusion ({name})",
            outpath=os.path.join(FIG_DIR, f"confusion_{out_prefix}_{cv_name}_{name}.png"),
            labels=("Lie (0)", "Truth (1)")
        )

    # Paired t-test only when at least 2 folds and comparable models
    try:
        paired_ttest_matrix(
            fold_metrics,
            metric_key="acc",
            out_csv=os.path.join(TAB_DIR, f"pvalues_acc_{out_prefix}_{cv_name}.csv"),
            out_png=os.path.join(FIG_DIR, f"pvalues_acc_{out_prefix}_{cv_name}.png"),
            title=f"Paired t-test p-values ({cv_name} ACC)"
        )
    except Exception as e:
        print(f"[WARN] Paired t-test skipped for {cv_name}: {e}")

    # RF feature importance
    if len(rf_importances) > 0:
        mean_imp = np.mean(np.vstack(rf_importances), axis=0)
        feat_names = FEAT_NAMES if len(FEAT_NAMES) == X.shape[1] else [f"f{i}" for i in range(X.shape[1])]
        df_fi = pd.DataFrame({"feature": feat_names, "importance": mean_imp})
        df_fi.sort_values("importance", ascending=False).to_csv(
            os.path.join(TAB_DIR, f"rf_feature_importance_all_{out_prefix}_{cv_name}.csv"),
            index=False
        )
        save_barh(
            mean_imp, feat_names,
            title=f"RF Feature Importance Top 10 — {cv_name}",
            out_png=os.path.join(FIG_DIR, f"rf_feature_importance_top10_{out_prefix}_{cv_name}.png"),
            topk=10
        )
        rf_importance_by_group(
            mean_imp, SLICES,
            out_csv=os.path.join(TAB_DIR, f"rf_importance_by_group_{out_prefix}_{cv_name}.csv"),
            out_png=os.path.join(FIG_DIR, f"rf_importance_by_group_{out_prefix}_{cv_name}.png")
        )

    return df_summary, df_folds

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Save config for reproducibility / MethodsX code availability
    config = {
        "DATA_DIR": DATA_DIR,
        "RAW_DIR": RAW_DIR,
        "SUBJECT_STIMULI_XLSX": SUBJECT_STIMULI_XLSX,
        "CHAN_ORDER": CHAN_ORDER,
        "SF": SF,
        "WINDOW_SIZE": WINDOW_SIZE,
        "STEP_SIZE": STEP_SIZE,
        "N_SPLITS": N_SPLITS,
        "RANDOM_STATE": RANDOM_STATE,
        "RUN_WINDOW_LEVEL_SKF": RUN_WINDOW_LEVEL_SKF,
        "RUN_SUBJECT_GROUPK": RUN_SUBJECT_GROUPK,
        "RUN_SESSION_GROUPK": RUN_SESSION_GROUPK,
        "RUN_LOSO": RUN_LOSO,
        "RUN_NON_OVERLAP": RUN_NON_OVERLAP,
        "RUN_META_ABLATION": RUN_META_ABLATION,
        "INCLUDE_SVM_IN_STRICT_VALIDATION": INCLUDE_SVM_IN_STRICT_VALIDATION,
        "INCLUDE_SVM_IN_WINDOW_LEVEL": INCLUDE_SVM_IN_WINDOW_LEVEL,
        "HAS_XGB": HAS_XGB
    }
    with open(os.path.join(TAB_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 1) Load original overlapping-window dataset
    X, y, meta = load_data(step_size=STEP_SIZE)
    print("Total samples:", len(X))
    print("Dataset shape:", X.shape)
    print("Metadata shape:", meta.shape)
    meta.to_csv(os.path.join(TAB_DIR, "window_metadata_original_step16.csv"), index=False)

    if len(FEAT_NAMES) != X.shape[1]:
        print(f"[WARN] Feature name count {len(FEAT_NAMES)} != X features {X.shape[1]}. Using generic names.")
        FEAT_NAMES = [f"f{i}" for i in range(X.shape[1])]

    save_overlap_report(
        step_size=STEP_SIZE,
        meta=meta,
        out_json=os.path.join(TAB_DIR, "overlap_leakage_report_step16.json"),
        out_csv=os.path.join(TAB_DIR, "windows_per_subject_label_step16.csv")
    )

    # LDA plot for original full descriptor
    lda_plot_like_example(X, y, os.path.join(FIG_DIR, "lda_separability_original_step16.png"))

    # Save group definitions
    group_def = pd.DataFrame([
        {"Group": "fTS",   "start": SLICES["fTS"].start,   "stop": SLICES["fTS"].stop,   "dim": SLICES["fTS"].stop - SLICES["fTS"].start},
        {"Group": "fCorr", "start": SLICES["fCorr"].start, "stop": SLICES["fCorr"].stop, "dim": SLICES["fCorr"].stop - SLICES["fCorr"].start},
        {"Group": "fF",    "start": SLICES["fF"].start,    "stop": SLICES["fF"].stop,    "dim": SLICES["fF"].stop - SLICES["fF"].start},
        {"Group": "fSP",   "start": SLICES["fSP"].start,   "stop": SLICES["fSP"].stop,   "dim": SLICES["fSP"].stop - SLICES["fSP"].start},
        {"Group": "full",  "start": SLICES["full"].start,  "stop": SLICES["full"].stop,  "dim": SLICES["full"].stop - SLICES["full"].start},
    ])
    group_def.to_csv(os.path.join(TAB_DIR, "feature_group_slices.csv"), index=False)

    # 2) Define feature sets for full and meta-correlation ablation
    feature_sets = get_feature_sets(X) if RUN_META_ABLATION else {"full_106": np.arange(X.shape[1])}

    all_summaries = []

    # 3) Window-level StratifiedKFold evaluation
    if RUN_WINDOW_LEVEL_SKF:
        for fs_name, fs_idx in feature_sets.items():
            # Evaluate the full descriptor and the descriptor without meta-correlation.
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            summary, _ = evaluate_cv(
                X, y,
                cv=cv,
                cv_name="window_stratified5fold",
                out_prefix=fs_name,
                groups=None,
                include_svm=INCLUDE_SVM_IN_WINDOW_LEVEL,
                feature_indices=fs_idx
            )
            all_summaries.append(summary)

    # 4) Optional subject-wise GroupKFold validation
    if RUN_SUBJECT_GROUPK:
        for fs_name, fs_idx in feature_sets.items():
            cv = GroupKFold(n_splits=N_SPLITS)
            summary, _ = evaluate_cv(
                X, y,
                cv=cv,
                cv_name="subject_group5fold",
                out_prefix=fs_name,
                groups=meta["subject"].values,
                include_svm=INCLUDE_SVM_IN_STRICT_VALIDATION,
                feature_indices=fs_idx
            )
            all_summaries.append(summary)

    # 5) Optional session/file-wise GroupKFold
    if RUN_SESSION_GROUPK:
        for fs_name, fs_idx in feature_sets.items():
            cv = GroupKFold(n_splits=N_SPLITS)
            summary, _ = evaluate_cv(
                X, y,
                cv=cv,
                cv_name="session_group5fold",
                out_prefix=fs_name,
                groups=meta["file_group"].values,
                include_svm=INCLUDE_SVM_IN_STRICT_VALIDATION,
                feature_indices=fs_idx
            )
            all_summaries.append(summary)

    # 6) Optional Leave-One-Subject-Out. This can be slow.
    if RUN_LOSO:
        for fs_name, fs_idx in feature_sets.items():
            cv = LeaveOneGroupOut()
            summary, _ = evaluate_cv(
                X, y,
                cv=cv,
                cv_name="LOSO_subject",
                out_prefix=fs_name,
                groups=meta["subject"].values,
                include_svm=INCLUDE_SVM_IN_STRICT_VALIDATION,
                feature_indices=fs_idx
            )
            all_summaries.append(summary)

    # 7) Optional non-overlapping windows sensitivity analysis
    if RUN_NON_OVERLAP:
        X_no, y_no, meta_no = load_data(step_size=WINDOW_SIZE)
        meta_no.to_csv(os.path.join(TAB_DIR, "window_metadata_nonoverlap_step384.csv"), index=False)
        save_overlap_report(
            step_size=WINDOW_SIZE,
            meta=meta_no,
            out_json=os.path.join(TAB_DIR, "overlap_leakage_report_step384.json"),
            out_csv=os.path.join(TAB_DIR, "windows_per_subject_label_step384.csv")
        )
        lda_plot_like_example(X_no, y_no, os.path.join(FIG_DIR, "lda_separability_nonoverlap_step384.png"))

        feature_sets_no = get_feature_sets(X_no) if RUN_META_ABLATION else {"full_106": np.arange(X_no.shape[1])}
        for fs_name, fs_idx in feature_sets_no.items():
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            summary, _ = evaluate_cv(
                X_no, y_no,
                cv=cv,
                cv_name="nonoverlap_window_stratified5fold",
                out_prefix=fs_name,
                groups=None,
                include_svm=INCLUDE_SVM_IN_WINDOW_LEVEL,
                feature_indices=fs_idx
            )
            all_summaries.append(summary)

            cvg = GroupKFold(n_splits=N_SPLITS)
            summary, _ = evaluate_cv(
                X_no, y_no,
                cv=cvg,
                cv_name="nonoverlap_subject_group5fold",
                out_prefix=fs_name,
                groups=meta_no["subject"].values,
                include_svm=INCLUDE_SVM_IN_STRICT_VALIDATION,
                feature_indices=fs_idx
            )
            all_summaries.append(summary)

    # 8) Save combined summary
    if len(all_summaries) > 0:
        df_all = pd.concat(all_summaries, ignore_index=True)
        df_all.to_csv(os.path.join(TAB_DIR, "ALL_SUMMARY_COMBINED.csv"), index=False)
        print("\n================ ALL SUMMARY COMBINED ================")
        print(df_all[["CV", "Feature_Set", "Model", "ACC_mean(%)", "ACC_std(%)", "Recall_mean(%)", "F1_mean(%)", "AUC_mean(%)"]])
        print("======================================================\n")

    print(f"✅ All figures saved to: {FIG_DIR}")
    print(f"✅ All tables saved to:  {TAB_DIR}")
