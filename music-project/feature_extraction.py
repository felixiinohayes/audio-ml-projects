import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import IPython.display as ipd

# Import samples
kick_samples = [lr.load(p)[0] for p in Path().glob('Kicks/Kick*.wav')]
sr = [lr.load(p)[1] for p in Path().glob('Kicks/Kick*.wav')]

def extract_features(signal,sr):
    return [
        np.mean(lr.feature.zero_crossing_rate(signal)[0]),
        lr.feature.spectral_centroid(y=signal)[0,0],
        lr.feature.spectral_bandwidth(y=signal)[0,0],
        lr.feature.spectral_contrast(y=signal)[0,0],
        lr.feature.spectral_flatness(y=signal)[0,0],
        find_attack_time(signal,sr)[0]
    ]

def find_attack_time(signal,sr):
    frame_length = 512
    hop_length = 256

    S = lr.stft(signal)
    envelope = lr.feature.rms(y=S, frame_length=frame_length, hop_length=hop_length, center=True)

    # Find the onset (the point where the signal exceeds a small threshold)
    onset_index = np.argmax(envelope > 0.01)  # Choose a threshold suitable for your sample

    # Find the peak (the highest point in the envelope)
    peak_index = np.argmax(envelope)

    # Calculate attack time
    attack_time = (peak_index - onset_index) / sr
    return attack_time

plt.figure(figsize=(5,14))

kick_features = np.array([extract_features(x,sr) for x in kick_samples])

scaler = MinMaxScaler((0,1))
scaled_features = scaler.fit_transform(kick_features)

print(scaled_features)