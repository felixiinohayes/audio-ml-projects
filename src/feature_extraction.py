import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa as lr
from scipy.ndimage import uniform_filter1d
import IPython.display as ipd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

kick_samples = [lr.load(p)[0] for p in Path().glob('data/Kicks/*.wav')]
sr = [lr.load(p)[1] for p in Path().glob('data/Kicks/*.wav')]

def extract_features(signal,sr):
    return [
        # np.mean(lr.feature.zero_crossing_rate(signal)[0]),
        lr.feature.spectral_centroid(y=signal)[0,0],
        # lr.feature.spectral_bandwidth(y=signal)[0,0]
        # lr.feature.spectral_contrast(y=signal)[0,0],
        # lr.feature.spectral_flatness(y=signal)[0,0]
        attack_time(signal),
        decay_time(signal)
    ]

def attack_time(signal):
    rms = lr.feature.rms(y=signal, frame_length=856, hop_length=40)
    times = lr.frames_to_time(np.arange(len(rms[0])), hop_length=40)
    rms = uniform_filter1d(rms, size=5)
    peak_index = np.argmax(rms[0][:1000])
    return times[peak_index]
    

def decay_time(signal):
    rms = lr.feature.rms(y=signal, frame_length=856, hop_length=40)
    times = lr.frames_to_time(np.arange(len(rms[0])), hop_length=40)
    rms = uniform_filter1d(rms, size=5)
    peak_index = np.argmax(rms)
    decay_threshold = rms[0][peak_index] * 0.1
    decay_index = peak_index + np.argmax(rms[0][peak_index:] < decay_threshold) # Get index at which amplitude has fallen to 30% of peak
    decay_time = times[decay_index] - times[peak_index]
    return decay_time

plt.figure(figsize=(5,14))

kick_features = np.array([extract_features(x,sr) for x in kick_samples])

scaler = MinMaxScaler((0,1))
scaled_features = scaler.fit_transform(kick_features[:,:])

weights = np.array([1,0.2,0.8])
weighted_features = scaled_features * weights


# Apply K-means clustering
n_clusters = 2  # Change based on how many different types of kicks expected
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(scaled_features)

selected_sample_index = 1

# Calculate similarity within the cluster
similarity_scores = pairwise_distances(weighted_features)
sorted_indices = np.argsort(similarity_scores[selected_sample_index])  # Sort by highest similarity

recommended = sorted_indices[5:]
print(sorted_indices)