import joblib
import numpy as np
import librosa

# Load the model
model = joblib.load('audio_fake_detector.pkl')

def predict_audio_fake(file_path):
    # Extract features from the uploaded audio file
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)  # Reshape for single sample
    
    # Predict using the loaded model
    prediction = model.predict(mfccs_mean)
    return "Fake" if prediction[0] == 1 else "Real"