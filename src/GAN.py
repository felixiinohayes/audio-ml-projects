import librosa as lr
import numpy as np
from pathlib import Path

def preprocess_audio(file_path, target_sample_rate=16000, target_length=16384):
    # Load audio file
    audio, sr = lr.load(file_path, sr=target_sample_rate)
    
    # Truncate or pad the audio to the target length
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    
    return audio

kick_samples = [preprocess_audio(p) for p in Path().glob('data/Kicks/*.wav')]
