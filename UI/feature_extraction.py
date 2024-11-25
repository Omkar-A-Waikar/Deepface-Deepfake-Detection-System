import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Loop through your audio files and create a dataset
data = []
labels = []

# Adjust these paths to where your audio files are stored
for label, folder in [('real', 'data/real'), ('fake', 'data/fake')]:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        features = extract_features(file_path)
        data.append(features)
        labels.append(label)

# Save features to a CSV for future use
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('audio_features.csv', index=False)
