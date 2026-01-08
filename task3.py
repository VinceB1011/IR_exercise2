import os
import re
import time
import glob
import csv
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import scipy.ndimage as ndimage

# PATHS
DB_PKL_PATH = "music_database_full.pkl"
QUERY_ROOT = "queries"
OUT_DIR = "task3_results"
os.makedirs(OUT_DIR, exist_ok=True)

# AUDIO / STFT PARAMS (must match Task 2!)
SR = 22050
N_FFT = 2048
HOP = 1024
BIN_MAX = 128
DB_DURATION = 30.0  # Task 2: first 30 seconds
Q_DURATION = 10.0  # Task 1: ~10 seconds

PEAK_THRESH = 0.01
TOL_FREQ = 1
TOL_TIME = 1

# Task 1 constraint: query start within first 20 seconds.
MAX_QUERY_START_SEC = 20.0
MAX_SHIFT_FRAMES = int(np.round(MAX_QUERY_START_SEC * SR / HOP))

# Choose one: "tp", "f1","precision", "recall"
SCORE_METRIC = "tp"

QUERY_SUBFOLDERS = {
    "Original": "Original",
    "Noise": "Noise",
    "Coding": "Coding",
    "Mobile": "Mobile",
}

def extract_track_id(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    return m.group(1) if m else None


def gt_db_filename_from_id(track_id: str) -> str:
    return f"{track_id}.mp3"


def collect_queries(query_root: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for dtype, sub in QUERY_SUBFOLDERS.items():
        folder = os.path.join(query_root, sub)
        if dtype in ("Original", "Noise"):
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*.wav")))
        elif dtype == "Coding":
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*.mp3")))
        elif dtype == "Mobile":
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*.wav")))
        else:
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*")))
    return out


def stft_mag(path: str, duration: Optional[float]) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True, duration=duration)
    X = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann")
    Y = np.abs(X[:BIN_MAX, :])
    return Y


def compute_constellation_map_bool(Y: np.ndarray, dist_freq: int, dist_time: int,
                                   thresh: float = PEAK_THRESH) -> np.ndarray:
    mx = ndimage.maximum_filter(Y, size=(2 * dist_freq + 1, 2 * dist_time + 1), mode="constant")
    C = (Y == mx) & (Y > thresh)
    return C.astype(np.bool_)


def score_from_counts(tp: int, fn: int, fp: int, metric: str) -> float:
    if metric == "tp":
        return float(tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if metric == "precision":
        return prec
    if metric == "recall":
        return rec
    if metric == "f1":
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    raise ValueError("metric must be one of: tp, precision, recall, f1")


def query_dilation(C_Q: np.ndarray, tol_freq: int, tol_time: int) -> np.ndarray:
    return ndimage.maximum_filter(C_Q, size=(2 * tol_freq + 1, 2 * tol_time + 1), mode="constant").astype(np.bool_)


def match_binary_matrices_tol(
        C_ref: np.ndarray,
        C_est_max: np.ndarray,
        sum_C_est: int,
) -> Tuple[int, int, int]:
    tp = int(np.sum(C_est_max & C_ref))
    fn = int(np.sum(C_ref)) - tp
    fp = sum_C_est - tp
    return tp, fn, fp


def sliding_best_score(
        C_D: np.ndarray,
        C_Q: np.ndarray,
        tol_freq: int,
        tol_time: int,
        metric: str,
        max_shift_frames: int,
) -> Tuple[float, int]:
    K, L = C_D.shape
    K2, N = C_Q.shape
    assert K == K2, "Frequency bins must match"
    assert L >= N, "Query must be shorter than document"

    M = L - N
    M = min(M, max_shift_frames)

    C_Q_max = query_dilation(C_Q, tol_freq=tol_freq, tol_time=tol_time)
    sum_C_Q = int(np.sum(C_Q))

    best_score = -1e9
    best_shift = 0

    for m in range(M + 1):
        C_D_crop = C_D[:, m:m + N]
        tp, fn, fp = match_binary_matrices_tol(C_D_crop, C_Q_max, sum_C_Q)
        sc = score_from_counts(tp, fn, fp, metric)

        if sc > best_score:
            best_score = sc
            best_shift = m

    return float(best_score), int(best_shift)


def identify_query_against_db(
        query_path: str,
        db_boolmaps: Dict[str, np.ndarray],
        cfg: dict,
        tol_freq: int,
        tol_time: int,
        metric: str,
) -> Tuple[str, float]:
    Yq = stft_mag(query_path, duration=Q_DURATION)
    C_Q = compute_constellation_map_bool(Yq, dist_freq=cfg["k"], dist_time=cfg["t"])

    best_track = ""
    best_score = -1e9

    for track_name, C_D in db_boolmaps.items():
        if C_D.shape[1] < C_Q.shape[1]:
            continue

        sc, _shift = sliding_best_score(
            C_D=C_D,
            C_Q=C_Q,
            tol_freq=tol_freq,
            tol_time=tol_time,
            metric=metric,
            max_shift_frames=MAX_SHIFT_FRAMES,
        )
        if sc > best_score:
            best_score = sc
            best_track = track_name

    return best_track, float(best_score)


def peaks_to_boolmap(peaks: np.ndarray, K: int, N: int) -> np.ndarray:
    C = np.zeros((K, N), dtype=np.bool_)
    if peaks is None or len(peaks) == 0:
        return C

    peaks = np.asarray(peaks)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError("Expected peaks array of shape (num_peaks, 2)")

    f = np.clip(peaks[:, 0].astype(int), 0, K - 1)
    t = np.clip(peaks[:, 1].astype(int), 0, N - 1)
    C[f, t] = True
    return C


def infer_db_frames_from_duration(duration_s: float) -> int:
    return int(np.ceil(duration_s * SR / HOP)) + 2


def load_db_from_pickle(db_pkl_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    with open(db_pkl_path, "rb") as f:
        db_all = pickle.load(f)
    if not isinstance(db_all, dict):
        raise ValueError("Pickle does not contain a dict at top level.")
    return db_all


@dataclass
class Row:
    distortion: str
    config: str
    k: int
    t: int
    metric: str
    tol_freq: int
    tol_time: int
    hits: int
    total: int
    accuracy: float
    avg_query_time_s: float


def write_csv(path: str, rows: List[Row]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    # load queries
    queries = collect_queries(QUERY_ROOT)
    for dtype, qfiles in queries.items():
        print(f"{dtype}: {len(qfiles)} queries found", flush=True)

    # load Task-2 DB
    db_all = load_db_from_pickle(DB_PKL_PATH)

    cfg_names = list(db_all.keys())
    print("\nConfigs found in pickle:", cfg_names, flush=True)

    CFG_PARAMS = {
        "small": (5, 5),
        "k5_t15": (5, 15),
        "k5_t40": (5, 40),
        "k15_t5": (15, 5),
        "medium": (15, 15),
        "k15_t40": (15, 40),
        "k40_t5": (40, 5),
        "k40_t15": (40, 15),
        "large": (40, 40),
        "best_v1": (12, 15),
        "best_v2": (18, 15),
        "best_v3": (15, 20),
    }

    CONFIGS = []
    missing = []
    for name in cfg_names:
        if name not in CFG_PARAMS:
            missing.append(name)
        else:
            k, t = CFG_PARAMS[name]
            CONFIGS.append({"name": name, "k": k, "t": t})

    if missing:
        print("\nMissing k/t mapping for configs:", missing, flush=True)

    N_DB = infer_db_frames_from_duration(DB_DURATION)
    K_DB = BIN_MAX

    all_rows: List[Row] = []
    per_distortion: Dict[str, List[Row]] = {k: [] for k in queries.keys()}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        print(f"\n=== CONFIG {cfg_name} (k={cfg['k']}, t={cfg['t']}) ===", flush=True)

        t0 = time.time()
        db_peaks_cfg: Dict[str, np.ndarray] = db_all[cfg_name]
        db_boolmaps: Dict[str, np.ndarray] = {}
        for track_name, peaks in db_peaks_cfg.items():
            db_boolmaps[track_name] = peaks_to_boolmap(peaks, K=K_DB, N=N_DB)
        print(f"DB bool maps built for {len(db_boolmaps)} tracks in {time.time() - t0:.1f}s", flush=True)

        for distortion, qfiles in queries.items():
            if not qfiles:
                print(f"  - No queries for {distortion}", flush=True)
                continue

            hits = 0
            times = []
            total = len(qfiles)

            for qi, qpath in enumerate(qfiles, start=1):
                tid = extract_track_id(qpath)
                if tid is None:
                    continue
                gt = gt_db_filename_from_id(tid)

                tq0 = time.time()
                pred, score = identify_query_against_db(
                    query_path=qpath,
                    db_boolmaps=db_boolmaps,
                    cfg=cfg,
                    tol_freq=TOL_FREQ,
                    tol_time=TOL_TIME,
                    metric=SCORE_METRIC,
                )
                dt = time.time() - tq0
                times.append(dt)

                if pred == gt:
                    hits += 1

            avg_t = float(np.mean(times)) if times else 0.0
            row = Row(
                distortion=distortion,
                config=cfg_name,
                k=cfg["k"],
                t=cfg["t"],
                metric=SCORE_METRIC,
                tol_freq=TOL_FREQ,
                tol_time=TOL_TIME,
                hits=hits,
                total=total,
                accuracy=hits / total if total else 0.0,
                avg_query_time_s=avg_t,
            )
            all_rows.append(row)
            per_distortion[distortion].append(row)
            print(f"  => {distortion}: hits {hits}/{total}, acc={row.accuracy:.2f}, avg_query_time={avg_t:.2f}s",
                  flush=True)

    write_csv(os.path.join(OUT_DIR, "all_results.csv"), all_rows)
    for distortion, rows in per_distortion.items():
        write_csv(os.path.join(OUT_DIR, f"{distortion.lower()}_table.csv"), rows)

    print("\nDone. CSVs written to:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
