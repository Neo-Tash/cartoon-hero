from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preflight handler
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return Response(status_code=200)

# 🎼 Camelot Mapping
camelot_map = {
    "C Major": "8B", "G Major": "9B", "D Major": "10B", "A Major": "11B",
    "E Major": "12B", "B Major": "1B", "F# Major": "2B", "C# Major": "3B",
    "F Major": "7B", "Bb Major": "6B", "Eb Major": "5B", "Ab Major": "4B",
    "A Minor": "8A", "E Minor": "9A", "B Minor": "10A", "F# Minor": "11A",
    "C# Minor": "12A", "G# Minor": "1A", "D# Minor": "2A", "A# Minor": "3A",
    "D Minor": "7A", "G Minor": "6A", "C Minor": "5A", "F Minor": "4A"
}

# 🎧 Harmonic Mixing Suggestions
def get_harmonic_matches(camelot):
    if not camelot:
        return []
    num = int(camelot[:-1])
    letter = camelot[-1]

    same = f"{num}{letter}"
    up = f"{(num % 12) + 1}{letter}"
    down = f"{12 if num == 1 else num - 1}{letter}"
    swap = f"{num}{'B' if letter == 'A' else 'A'}"

    return list(set([same, up, down, swap]))

# 🎹 Mode Detection
def detect_mode(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    chroma_mean = np.mean(chroma, axis=1)
    major_corr = np.correlate(chroma_mean, major_profile)
    minor_corr = np.correlate(chroma_mean, minor_profile)

    return "Major" if major_corr > minor_corr else "Minor"

# 🎸 Instrument Detection (simple heuristic)
def detect_instruments(centroid, rms):
    instruments = []

    if centroid < 1500:
        instruments.append("Bass / Kick")

    if 1500 <= centroid <= 3500:
        instruments.append("Piano / Synth / Vocals")

    if centroid > 3500:
        instruments.append("Hi-Hats / Cymbals / Bright Synths")

    if rms > 0.08:
        instruments.append("High Energy Elements")

    if rms < 0.02:
        instruments.append("Soft / Ambient Layers")

    return instruments

# 🎼 Genre Detection
def detect_genre(tempo, rms, centroid):
    genre = "Unknown"

    if tempo < 90:
        if rms < 0.03:
            genre = "Ambient / Chill"
        else:
            genre = "Hip-Hop / Soul"

    elif 90 <= tempo <= 115:
        if centroid < 2500:
            genre = "R&B / Pop"
        else:
            genre = "Afro / Amapiano"

    elif 115 < tempo <= 135:
        genre = "House / Dance"

    elif tempo > 135:
        if centroid > 4000:
            genre = "EDM / Trap"
        else:
            genre = "Techno"

    if rms > 0.08:
        genre += " (High Energy)"
    elif rms < 0.02:
        genre += " (Low Energy)"

    return genre

# 🚀 MAIN ANALYSIS ENDPOINT
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)

        # 🎧 Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # ⏱ Duration
        duration = librosa.get_duration(y=y, sr=sr)

        # 🔊 RMS Energy
        rms = np.mean(librosa.feature.rms(y=y))

        # 🎚 Spectral Features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # 🎼 Key Detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]

        # 🎹 Mode
        mode = detect_mode(y, sr)
        full_key = f"{key} {mode}"

        # 🎧 Camelot
        camelot = camelot_map.get(full_key, None)

        # 🎼 Harmonic Suggestions
        harmonic_matches = get_harmonic_matches(camelot)

        # 🎸 Instruments
        instruments = detect_instruments(centroid, rms)

        # 🎼 Genre
        genre = detect_genre(tempo, rms, centroid)

        return JSONResponse(content={
            "success": True,
            "bpm": round(float(tempo)),
            "key": key,
            "mode": mode,
            "full_key": full_key,
            "camelot": camelot,
            "harmonic_matches": harmonic_matches,
            "genre": genre,
            "duration": round(duration, 2),
            "rms_energy": round(float(rms), 4),
            "spectral_centroid": round(float(centroid), 2),
            "spectral_rolloff": round(float(rolloff), 2),
            "instruments_detected": instruments
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=200)
