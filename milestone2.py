import os
import librosa
import numpy as np
import scipy.ndimage as ndimage
import sys
import pickle
from utils import get_peaks, generate_hashes_from_peaks


configs = [
    ("std", 10, 50, 50, 3),      # Deine aktuelle
    ("wide_t", 10, 100, 50, 3),   # Größeres Zeitfenster
    ("wide_f", 10, 50, 100, 3),   # Größeres Frequenzfenster
    ("dense", 10, 50, 50, 6),      # Mehr Paare (Fan-out)
    ("super_dense", 10, 60, 100, 15) # noch mehr Paare und größeres Zeit Fenster
]

def run_indexing(config_name, dt_min, dt_max, df_max, fan_out):
  
    audio_folder = 'data/40'
    song_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
    inverted_index = {}
    song_map = {}

    for song_id, song in enumerate(song_files):
        song_map[song_id] = song
        path = os.path.join(audio_folder, song)
        y, sr = librosa.load(path, sr=22050)
        Y = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))

        duration = librosa.get_duration(y=y, sr=sr)
        print(f"\n[{song_id+1}/{len(song_files)}] Processing: {song}")
        print(f"  > Duration: {duration:.2f}s")

        peaks = get_peaks(Y, 18, 15)
        print(f"  > Peaks found: {len(peaks)} (Density: {len(peaks)/duration:.2f} peaks/s)")
        pairs = generate_hashes_from_peaks(peaks, dt_min, dt_max, df_max, fan_out)

        for h, t1 in pairs:
            if h not in inverted_index:
                inverted_index[h] = []
            inverted_index[h].append((song_id, t1))

    print("\n" + "="*30)
    print("DATABASE SUMMARY")
    print(f"Total Unique Hashes: {len(inverted_index)}")

    # Calculate total number of pointers stored
    total_entries = sum(len(v) for v in inverted_index.values())
    print(f"Total Occurrences Stored: {total_entries}")

    # Memory usage
    size_mb = sys.getsizeof(inverted_index) / (1024**2)
    print(f"Approx. RAM usage of Index: {size_mb:.2f} MB")
    print("="*30)   

    print(f"\nSummary for {config_name}: {len(inverted_index)} unique hashes.")

    filename = f"db_{config_name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump({'index': inverted_index, 'map': song_map}, f)
    print(f"Configuration {config_name} saved.")

def main():
    for name, dt_min, dt_max, df_max, fan_out in configs:
        run_indexing(name, dt_min, dt_max, df_max, fan_out)

if __name__ == "__main__":
    main()