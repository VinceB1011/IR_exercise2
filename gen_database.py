import os
import librosa
import numpy as np
import scipy.ndimage as ndimage
import pickle
import matplotlib.pyplot as plt

CONFIGS = [
    # Coarse Configs (mit den alten Namen für 5/15/40)
    {'name': 'small',   'k': 5,  't': 5},
    {'name': 'k5_t15',  'k': 5,  't': 15},
    {'name': 'k5_t40',  'k': 5,  't': 40},
    {'name': 'k15_t5',  'k': 15, 't': 5},
    {'name': 'medium',  'k': 15, 't': 15},
    {'name': 'k15_t40', 'k': 15, 't': 40},
    {'name': 'k40_t5',  'k': 40, 't': 5},
    {'name': 'k40_t15', 'k': 40, 't': 15},
    {'name': 'large',   'k': 40, 't': 40},
    
    # Refined Configs
    {'name': 'best_v1', 'k': 12, 't': 15},
    {'name': 'best_v2', 'k': 18, 't': 15},
    {'name': 'best_v3', 'k': 15, 't': 20}
]

def get_peaks(Y, k, t):
    # Effiziente Berechnung Constellation Map
    size = (2*k + 1, 2*t + 1)
    result = ndimage.maximum_filter(Y, size=size, mode='constant')
    cmap = (Y == result) & (Y > 0.01)
    return np.argwhere(cmap)

database = {cfg['name']: {} for cfg in CONFIGS}

audio_folder = 'data/40'
song_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]

for song in song_files:
    path = os.path.join(audio_folder, song)
    y, sr = librosa.load(path, duration=30, sr=22050)
    Y = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))

    for cfg in CONFIGS: 
        peaks = get_peaks(Y, cfg['k'], cfg['t'])
        database[cfg['name']][song] = peaks

with open('music_database_full.pkl', 'wb') as f:
    pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Caputred in database!")