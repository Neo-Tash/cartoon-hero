from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import librosa
import numpy as np
import tempfile

# --- AI IMPORTS ---
import tensorflow as tf
import tensorflow_hub as hub

app = FastAPI()

# ✅ CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ PRE-FLIGHT HANDLER
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return Response(status_code=200)


# -----------------------------
# 🔥 SAFE AI MODEL LOADER (STEP 2)
# -----------------------------
yamnet_model = None

def get_yamnet_model():
    global yamnet_model
    if yamnet_model is None:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model


def run_yamnet(y, sr):
    model = get_yamnet_model()

    # Convert to mono + resample to 16kHz
    waveform = librosa.resample(y, orig_sr=sr, target_sr=16000)
    waveform = waveform.astype(np.float32)

    scores, embeddings, spectrogram = model(waveform)

    mean_scores = np.mean(scores.numpy(), axis=0)
    return mean_scores


# -----------------------------
# 🎼 EXISTING ANALYZER (UNCHANGED)
# -----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)

        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Duration
        duration = librosa.get_duration(y=y, sr=sr)

        # RMS Energy
        rms = np.mean(librosa.feature.rms(y=y))

        # Spectral Features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # Key Detection (basic)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']
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
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=200)
