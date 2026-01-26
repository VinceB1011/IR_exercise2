import os
import re
import time
import glob
import csv
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import librosa
import scipy.ndimage as ndimage

from utils import get_peaks, generate_hashes_from_peaks

# ============================================================
# PATHS (edit if needed)
# ============================================================
QUERY_ROOT = "queries"
OUT_DIR = "ms22_task2_results"
os.makedirs(OUT_DIR, exist_ok=True)

DB_PKLS = {
    "std": "db_std.pkl",
    "wide_t": "db_wide_t.pkl",
    "wide_f": "db_wide_f.pkl",
    "dense": "db_dense.pkl",
    "super_dense": "db_super_dense.pkl"
}

# ============================================================
# AUDIO / STFT PARAMS (must match indexing)
# ============================================================
SR = 22050
N_FFT = 2048
HOP = 1024
BIN_MAX = 128  # optional: keep peaks in lower bins like in Milestone 1

# Query length from Milestone 1
Q_DURATION = 10.0

# Peak detection parameters (best config from Milestone 2.1)
PEAK_K = 18
PEAK_T = 15
PEAK_THRESH = 0.01

# Queries (same as Milestone 1)
QUERY_SUBFOLDERS = {
    "Original": "Original",
    "Noise": "Noise",
    "Coding": "Coding",
    "Mobile": "Mobile",
}


# ============================================================
# Helpers
# ============================================================
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
        if dtype in ("Original", "Noise", "Mobile"):
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*.wav")))
        elif dtype == "Coding":
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*.mp3")))
        else:
            out[dtype] = sorted(glob.glob(os.path.join(folder, "*")))
    return out

def stft_mag(path: str, duration: Optional[float]) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True, duration=duration)
    X = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann")
    Y = np.abs(X)
    return Y

def load_hash_db(path: str) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, str]]:
    """
    Expects pickle structure:
      {'index': {hash: [(song_id, t_db), ...]}, 'map': {song_id: filename}}
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["index"], obj["map"]


# ============================================================
# Hash-based matching (Wang-style offset voting)
# ============================================================
def identify_query_hash(
    query_path: str,
    index: Dict[int, List[Tuple[int, int]]],
    song_map: Dict[int, str],
    dt_min: int,
    dt_max: int,
    df_max: int,
    fan_out: int,
) -> Tuple[str, int]:
    """
    Returns (best_filename, best_vote).
    best_vote is the highest count of consistent offsets for the winning song.
    """
    Yq = stft_mag(query_path, duration=Q_DURATION)
    peaks_q = get_peaks(Yq, PEAK_K, PEAK_T, PEAK_THRESH)
    pairs_q = generate_hashes_from_peaks(peaks_q, dt_min, dt_max, df_max, fan_out)

    if not pairs_q:
        return "", 0

    # votes[(song_id, delta)] += 1, where delta = t_db - t_q
    votes = Counter()

    for h, t_q in pairs_q:
        postings = index.get(h)
        if not postings:
            continue
        for song_id, t_db in postings:
            delta = int(t_db) - int(t_q)
            votes[(song_id, delta)] += 1

    if not votes:
        return "", 0

    # score each song by its best delta bin
    best_song = None
    best_vote = -1

    # aggregate max over delta for each song
    max_per_song: Dict[int, int] = defaultdict(int)
    for (song_id, delta), c in votes.items():
        if c > max_per_song[song_id]:
            max_per_song[song_id] = c

    for song_id, v in max_per_song.items():
        if v > best_vote:
            best_vote = v
            best_song = song_id

    if best_song is None:
        return "", 0

    return song_map[best_song], int(best_vote)


# ============================================================
# Results table
# ============================================================
@dataclass
class Row:
    distortion: str
    config: str
    dt_min: int
    dt_max: int
    df_max: int
    fan_out: int
    hits: int
    total: int
    accuracy: float
    avg_query_time_s: float
    avg_best_vote: float

def write_csv(path: str, rows: List[Row]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# ============================================================
# MAIN
# ============================================================
def main():
    queries = collect_queries(QUERY_ROOT)
    for dtype, qfiles in queries.items():
        print(f"{dtype}: {len(qfiles)} queries found", flush=True)

    # Same 4 configs as your colleague used for indexing:
    # (name, dt_min, dt_max, df_max, fan_out)
    CONFIGS = [
        ("std",    10,  50,  50, 3),
        ("wide_t", 10, 100,  50, 3),
        ("wide_f", 10,  50, 100, 3),
        ("dense",  10,  50,  50, 6),
        ("super_dense", 10, 60, 100, 15)
    ]

    all_rows: List[Row] = []

    # also create a combined pool of queries
    combined_queries: List[Tuple[str, str]] = []
    for dist, files in queries.items():
        for p in files:
            combined_queries.append((dist, p))

    for cfg_name, dt_min, dt_max, df_max, fan_out in CONFIGS:
        db_path = DB_PKLS[cfg_name]
        print(f"\n=== HASH CONFIG {cfg_name} ({db_path}) ===", flush=True)

        index, song_map = load_hash_db(db_path)
        print(f"Loaded index: {len(index)} unique hashes, {len(song_map)} songs", flush=True)
        # per distortion
        for distortion, qfiles in queries.items():
            if not qfiles:
                continue

            hits = 0
            times = []
            votes = []
            total = len(qfiles)

            for qpath in qfiles:
                tid = extract_track_id(qpath)
                if tid is None:
                    continue
                gt = gt_db_filename_from_id(tid)

                t0 = time.time()
                pred, best_vote = identify_query_hash(
                    query_path=qpath,
                    index=index,
                    song_map=song_map,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    df_max=df_max,
                    fan_out=fan_out,
                )
                dt = time.time() - t0
                times.append(dt)
                votes.append(best_vote)

                if pred == gt:
                    hits += 1

            avg_t = float(np.mean(times)) if times else 0.0
            avg_vote = float(np.mean(votes)) if votes else 0.0

            row = Row(
                distortion=distortion,
                config=cfg_name,
                dt_min=dt_min,
                dt_max=dt_max,
                df_max=df_max,
                fan_out=fan_out,
                hits=hits,
                total=total,
                accuracy=hits / total if total else 0.0,
                avg_query_time_s=avg_t,
                avg_best_vote=avg_vote,
            )
            all_rows.append(row)
            print(f"  => {distortion}: hits {hits}/{total}, acc={row.accuracy:.2f}, avg_time={avg_t:.2f}s, avg_vote={avg_vote:.1f}", flush=True)

        # combined
        hits = 0
        times = []
        votes = []
        total = len(combined_queries)

        for distortion, qpath in combined_queries:
            tid = extract_track_id(qpath)
            if tid is None:
                continue
            gt = gt_db_filename_from_id(tid)

            t0 = time.time()
            pred, best_vote = identify_query_hash(
                query_path=qpath,
                index=index,
                song_map=song_map,
                dt_min=dt_min,
                dt_max=dt_max,
                df_max=df_max,
                fan_out=fan_out,
            )
            dt = time.time() - t0
            times.append(dt)
            votes.append(best_vote)
            if pred == gt:
                hits += 1

        avg_t = float(np.mean(times)) if times else 0.0
        avg_vote = float(np.mean(votes)) if votes else 0.0

        all_rows.append(Row(
            distortion="Combined",
            config=cfg_name,
            dt_min=dt_min,
            dt_max=dt_max,
            df_max=df_max,
            fan_out=fan_out,
            hits=hits,
            total=total,
            accuracy=hits / total if total else 0.0,
            avg_query_time_s=avg_t,
            avg_best_vote=avg_vote,
        ))
        print(f"  => Combined: hits {hits}/{total}, acc={hits/total:.2f}, avg_time={avg_t:.2f}s, avg_vote={avg_vote:.1f}", flush=True)

    write_csv(os.path.join(OUT_DIR, "all_results.csv"), all_rows)

    per_distortion = defaultdict(list)
    for r in all_rows:
        per_distortion[r.distortion].append(r)

    for distortion, rows in per_distortion.items():
        fname = f"{distortion.lower()}_table.csv"
        write_csv(os.path.join(OUT_DIR, fname), rows)

    print("\nDone. CSVs written to:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
