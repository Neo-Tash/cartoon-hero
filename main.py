from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# ✅ CORS (Frontend connection fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route (optional health check)
@app.get("/")
async def root():
    return {"message": "SlickCoherence Audio Analysis API is running"}

# ✅ ANALYZE ENDPOINT (THIS ENABLES FILE UPLOAD IN SWAGGER)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Load audio
        y, sr = librosa.load(tmp_path, sr=None)

        # 🔹 Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # 🔹 Duration
        duration = librosa.get_duration(y=y, sr=sr)

        # 🔹 Energy
        rms = np.mean(librosa.feature.rms(y=y))

        # 🔹 Spectral Features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # 🔹 Key Detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))

        keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']

        key = keys[key_index]

        # Clean up temp file
        os.remove(tmp_path)

        return {
            "success": True,
            "bpm": round(float(tempo)),
            "key": key,
            "duration": round(float(duration), 2),
            "rms_energy": round(float(rms), 4),
            "spectral_centroid": round(float(centroid), 2),
            "spectral_rolloff": round(float(rolloff), 2)
        }

    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )
