from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SlickCoherence Audio Analysis API is running"}


# 🎧 CAM ELOT MAP
CAMELOT_MAP = {
    "C": ("8B", "Major"),
    "G": ("9B", "Major"),
    "D": ("10B", "Major"),
    "A": ("11B", "Major"),
    "E": ("12B", "Major"),
    "B": ("1B", "Major"),
    "F#": ("2B", "Major"),
    "C#": ("3B", "Major"),
    "G#": ("4B", "Major"),
    "D#": ("5B", "Major"),
    "A#": ("6B", "Major"),
    "F": ("7B", "Major"),

    "A Minor": ("8A", "Minor"),
    "E Minor": ("9A", "Minor"),
    "B Minor": ("10A", "Minor"),
    "F# Minor": ("11A", "Minor"),
    "C# Minor": ("12A", "Minor"),
    "G# Minor": ("1A", "Minor"),
    "D# Minor": ("2A", "Minor"),
    "A# Minor": ("3A", "Minor"),
    "F Minor": ("4A", "Minor"),
    "C Minor": ("5A", "Minor"),
    "G Minor": ("6A", "Minor"),
    "D Minor": ("7A", "Minor"),
}


def get_camelot(key):
    if key in CAMELOT_MAP:
        return CAMELOT_MAP[key]
    return ("Unknown", "Unknown")


def harmonic_matches(camelot):
    if camelot == "Unknown":
        return []

    num = int(camelot[:-1])
    letter = camelot[-1]

    prev_num = 12 if num == 1 else num - 1
    next_num = 1 if num == 12 else num + 1

    return [
        f"{camelot} (same key)",
        f"{prev_num}{letter} (energy drop)",
        f"{next_num}{letter} (energy boost)",
        f"{num}{'A' if letter == 'B' else 'B'} (relative key)"
    ]


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))

        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # 🎼 KEY DETECTION
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))

        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        base_key = keys[key_index]

        # 🎧 SIMPLE MAJOR/MINOR GUESS
        if rms < 0.05:
            key = f"{base_key} Minor"
        else:
            key = base_key

        camelot, scale = get_camelot(key)
        matches = harmonic_matches(camelot)

        # 🧠 AI LOGIC
        if tempo < 90:
            genre = "Ambient / Chill"
            mood = "Calm"
        elif tempo < 115:
            genre = "House / Afro"
            mood = "Groovy"
        else:
            genre = "Dance / Electronic"
            mood = "Energetic"

        vocal_type = "Instrumental" if rms < 0.05 else "Vocal"

        if tempo > 120:
            drum_style = "Trap"
        elif 100 <= tempo <= 120:
            drum_style = "Amapiano / House"
        else:
            drum_style = "Slow Groove"

        mix_suggestion = f"Best mix keys: {', '.join(matches)}"

        os.remove(tmp_path)

        return JSONResponse(content={
            "success": True,
            "bpm": round(float(tempo)),
            "key": key,
            "camelot": camelot,
            "scale": scale,
            "harmonic_matches": matches,

            "duration": round(duration, 2),
            "rms_energy": round(float(rms), 4),
            "spectral_centroid": round(float(centroid), 2),
            "spectral_rolloff": round(float(rolloff), 2),

            "ai_genre": genre,
            "mood": mood,
            "vocal_type": vocal_type,
            "drum_style": drum_style,
            "mix_suggestion": mix_suggestion
        })

    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )
