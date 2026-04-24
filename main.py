from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import tempfile

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    y, sr = librosa.load(temp_path)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_index = np.argmax(np.mean(chroma, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_index]

    return {
        "bpm": round(tempo),
        "key": key,
        "duration": round(duration, 2),
        "rms_energy": round(float(rms), 4),
        "spectral_centroid": round(float(centroid), 2),
        "spectral_rolloff": round(float(rolloff), 2)
    }
