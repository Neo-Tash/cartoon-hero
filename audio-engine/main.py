import os
import math
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SlickCoherence Audio Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_float(value, fallback=None):
    try:
        if value is None:
            return fallback
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return fallback
        return value
    except Exception:
        return fallback


def normalize_array(values: np.ndarray, max_points: int = 1600) -> List[float]:
    """Downsample and normalize array values to 0..1 for frontend display/storage."""
    if values is None or len(values) == 0:
        return []

    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.abs(values)

    if len(values) > max_points:
        bucket_size = int(math.ceil(len(values) / max_points))
        padded_length = bucket_size * max_points
        padded = np.pad(values, (0, padded_length - len(values)), mode="constant")
        values = padded.reshape(max_points, bucket_size).max(axis=1)

    max_val = float(np.max(values)) if len(values) else 0.0
    if max_val <= 0:
        return [0.0 for _ in values]

    normalized = values / max_val
    return [round(float(v), 4) for v in normalized]


def build_waveform_peaks(y: np.ndarray, max_points: int = 1600) -> List[float]:
    if y is None or len(y) == 0:
        return []

    y_abs = np.abs(y)
    return normalize_array(y_abs, max_points=max_points)


def build_energy_curve(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512, max_points: int = 96) -> List[float]:
    if y is None or len(y) == 0:
        return []

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return normalize_array(rms, max_points=max_points)


def estimate_loudness_db(y: np.ndarray) -> float:
    if y is None or len(y) == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms <= 0:
        return -60.0
    db = 20 * math.log10(rms)
    return round(max(-60.0, min(0.0, db)), 2)


def detect_key(y: np.ndarray, sr: int) -> Tuple[str, float]:
    """Basic Krumhansl-Schmuckler key detection using chroma features."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        total = np.sum(chroma_mean)
        if total > 0:
            chroma_mean = chroma_mean / total

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        scores = []

        for i in range(12):
            major_score = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
            minor_score = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])
            if not np.isnan(major_score):
                scores.append((major_score, f"{notes[i]} Major"))
            if not np.isnan(minor_score):
                scores.append((minor_score, f"{notes[i]} Minor"))

        if not scores:
            return "Unknown", 0.0

        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_key = scores[0]
        confidence = max(0.0, min(1.0, (best_score + 1) / 2))
        return best_key, round(confidence, 2)
    except Exception:
        return "Unknown", 0.0


def detect_drop_time(energy_curve: List[float], duration: float) -> float | None:
    """Simple v1 drop estimate: strongest energy-rise point after intro section."""
    if not energy_curve or not duration or duration <= 0:
        return None

    arr = np.asarray(energy_curve, dtype=np.float32)
    if len(arr) < 8:
        return None

    # Ignore first 15% to avoid selecting the very beginning.
    start_idx = max(1, int(len(arr) * 0.15))
    diffs = np.diff(arr)
    search = diffs[start_idx:]
    if len(search) == 0:
        return None

    best_idx = int(np.argmax(search)) + start_idx
    if search[best_idx - start_idx] < 0.03:
        # If no strong rise exists, use strongest energy point after intro.
        best_idx = int(np.argmax(arr[start_idx:])) + start_idx

    seconds_per_point = duration / len(arr)
    return round(float(best_idx * seconds_per_point), 2)


@app.get("/health")
def health():
    return {"status": "ok", "service": "slickcoherence-audio-engine-v1"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)) -> Dict:
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        # mono=True is best for BPM/key/energy v1. sr=None preserves original sample rate.
        y, sr = librosa.load(tmp_path, sr=None, mono=True)

        if y is None or len(y) == 0:
            raise HTTPException(status_code=422, detail="Could not decode audio")

        duration = safe_float(librosa.get_duration(y=y, sr=sr), 0.0)

        # BPM and beat grid
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, trim=False)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if tempo.size else 0.0
        bpm = round(safe_float(tempo, 0.0), 2)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist() if len(beat_frames) else []
        beat_grid = [round(float(t), 3) for t in beat_times[:500]]
        first_beat_offset = beat_grid[0] if beat_grid else 0

        # Confidence v1 estimate based on detected beats and tempo validity.
        beat_count = len(beat_grid)
        expected_beats = max(1, int((duration / 60) * bpm)) if bpm else 1
        beat_density_score = min(1.0, beat_count / expected_beats) if expected_beats > 0 else 0.0
        bpm_confidence = round(float(max(0.15, min(0.95, beat_density_score))), 2) if bpm else 0.0

        waveform_peaks = build_waveform_peaks(y, max_points=1600)
        energy_curve = build_energy_curve(y, max_points=96)
        energy = round(float(np.mean(energy_curve)), 4) if energy_curve else 0.0
        loudness = estimate_loudness_db(y)
        key, key_confidence = detect_key(y, sr)
        drop_time = detect_drop_time(energy_curve, duration)

        # Peak/drop candidates from onset envelope.
        peak_frames = librosa.util.peak_pick(onset_env, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.2, wait=12)
        peak_times = librosa.frames_to_time(peak_frames, sr=sr).tolist() if len(peak_frames) else []
        peaks = [round(float(t), 2) for t in peak_times[:40]]

        return {
            "success": True,
            "analysis": {
                "bpm": bpm or 120,
                "key": key,
                "energy": energy,
                "loudness": loudness,
                "duration": round(float(duration), 2),
                "sample_rate": int(sr),
                "channels": 1,
                "waveform_peaks": waveform_peaks,
                "energy_curve": energy_curve,
                "beat_grid": beat_grid,
                "first_beat_offset": first_beat_offset,
                "peaks": peaks,
                "drop_time": drop_time,
                "confidence": {
                    "bpm": bpm_confidence,
                    "key": key_confidence,
                    "waveform": 0.85 if waveform_peaks else 0.0,
                },
                "analysis_method": "python_librosa_v1"
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(exc)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
