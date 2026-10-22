
import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "/content/speech_emotion_lstm.keras"
ENCODER_PATH = "/content/label_encoder.pkl"

model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

audio_path = input("Enter audio file path: ").strip()

if os.path.isfile(audio_path):
    audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
    features = pad_sequences(
        [mfcc], maxlen=130, dtype="float32",
        padding="post", truncating="post"
    )

    prediction = model.predict(features, verbose=0)
    emotion = encoder.inverse_transform([np.argmax(prediction)])[0]
    print("Predicted emotion:", emotion)
else:
    print("Audio file not found. Check the file path.")
