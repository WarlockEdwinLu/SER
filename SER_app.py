import streamlit as st
import numpy as np
import librosa
import joblib
import os

# Load trained model (Scikit-learn joblib format)
model_filename = "speech_emotion_recognition_model.joblib"
model = joblib.load(model_filename)

# Emotion labels (same order as training)
emotions = ['calm', 'happy', 'fearful', 'disgust']

# Function to extract audio features using `librosa.load()`
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_name, sr=None)  # Load audio file
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None  # Return None if file cannot be processed

    result = np.array([])

    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        result = np.hstack((result, np.mean(mfccs.T, axis=0)))

    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        result = np.hstack((result, np.mean(chroma.T, axis=0)))

    if mel:
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        result = np.hstack((result, np.mean(mel.T, axis=0)))

    return result

# Streamlit UI
st.title("🎤 Speech Emotion Recognition App")
st.write("Upload an audio file and the model will predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save file temporarily
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Ensure correct file writing

    # Extract features
    features = extract_feature(temp_filename)
    
    if features is not None:  # Ensure valid features before prediction
        features = features.reshape(1, -1)  # Reshape for prediction

        # Predict emotion
        prediction = model.predict(features)[0]
        st.subheader(f"Predicted Emotion: **{prediction.upper()}**")

    # Clean up temporary file
    os.remove(temp_filename)
