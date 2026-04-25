from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import librosa
import numpy as np
import tempfile

app = FastAPI()

# ✅ CORS CONFIG (FIXES "Failed to fetch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ PRE-FLIGHT HANDLER (CRITICAL FOR BROWSER REQUESTS)
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return Response(status_code=200)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)

        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Duration
        duration = librosa.get_duration(y=y, sr=sr)

        # RMS Energy
        rms = np.mean(librosa.feature.rms(y=y))

        # Spectral features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # Key detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]

        return JSONResponse(content={
            "success": True,
            "bpm": round(float(tempo)),
            "key": key,
            "duration": round(duration, 2),
            "rms_energy": round(float(rms), 4),
            "spectral_centroid": round(float(centroid), 2),
            "spectral_rolloff": round(float(rolloff), 2)
        })

    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=200
        )
