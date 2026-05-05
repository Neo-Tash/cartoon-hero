import os
import math
import tempfile
import subprocess
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SlickCoherence Audio Engine", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # 60MB safety guard
ANALYSIS_SECONDS = 90                # keep Railway response fast and stable
TARGET_SR = 22050                    # downsample for speed


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


def run_cmd(cmd: List[str], timeout: int = 25) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def get_duration_ffprobe(path: str) -> Optional[float]:
    try:
        result = run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ], timeout=12)
        if result.returncode != 0:
            return None
        duration = safe_float(result.stdout.decode("utf-8", errors="ignore").strip(), None)
        return round(duration, 2) if duration and duration > 0 else None
    except Exception:
        return None


def convert_preview_to_wav(src_path: str) -> str:
    """Decode only a short preview to PCM WAV so librosa does not hang/crash on MP3."""
    wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = wav.name
    wav.close()

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-t", str(ANALYSIS_SECONDS),
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-vn",
        wav_path,
    ]
    result = run_cmd(cmd, timeout=30)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")[:800]
        raise HTTPException(status_code=422, detail=f"FFmpeg decode failed: {err}")
    return wav_path


def normalize_array(values: np.ndarray, max_points: int = 1600) -> List[float]:
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
    return normalize_array(np.abs(y), max_points=max_points)


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
    return round(max(-60.0, min(0.0, 20 * math.log10(rms))), 2)


def normalize_bpm(bpm: float) -> float:
    """Keep detected tempo in a DJ-friendly range."""
    bpm = safe_float(bpm, 0.0) or 0.0
    while bpm and bpm < 70:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return round(bpm, 2) if bpm > 0 else 120.0


def detect_bpm_and_beats(y: np.ndarray, sr: int, duration: float) -> Tuple[float, List[float], float]:
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            onset_envelope=onset_env,
            trim=False,
            units="frames",
        )
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if tempo.size else 0.0
        bpm = normalize_bpm(float(tempo))
        beat_times_preview = librosa.frames_to_time(beat_frames, sr=sr).tolist() if len(beat_frames) else []
        beat_grid = [round(float(t), 3) for t in beat_times_preview[:500]]

        beat_count = len(beat_grid)
        expected = max(1, int((min(duration or ANALYSIS_SECONDS, ANALYSIS_SECONDS) / 60) * bpm))
        confidence = round(float(max(0.15, min(0.95, beat_count / expected))), 2) if bpm else 0.0
        return bpm, beat_grid, confidence
    except Exception:
        return 120.0, [], 0.15


def detect_key(y: np.ndarray, sr: int) -> Tuple[str, float]:
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)
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
            maj = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
            minr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])
            if not np.isnan(maj):
                scores.append((maj, f"{notes[i]} Major"))
            if not np.isnan(minr):
                scores.append((minr, f"{notes[i]} Minor"))
        if not scores:
            return "Unknown", 0.0
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_key = scores[0]
        return best_key, round(max(0.0, min(1.0, (best_score + 1) / 2)), 2)
    except Exception:
        return "Unknown", 0.0


def detect_drop_time(energy_curve: List[float], duration: float) -> Optional[float]:
    if not energy_curve or not duration or duration <= 0:
        return None
    arr = np.asarray(energy_curve, dtype=np.float32)
    if len(arr) < 8:
        return None
    start_idx = max(1, int(len(arr) * 0.15))
    diffs = np.diff(arr)
    search = diffs[start_idx:]
    if len(search) == 0:
        return None
    best_idx = int(np.argmax(search)) + start_idx
    if search[best_idx - start_idx] < 0.03:
        best_idx = int(np.argmax(arr[start_idx:])) + start_idx
    seconds_per_point = min(duration, ANALYSIS_SECONDS) / len(arr)
    return round(float(best_idx * seconds_per_point), 2)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "slickcoherence-audio-engine-v1",
        "version": "1.2.0",
        "mode": "railway_safe_preview_analysis",
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)) -> Dict:
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename or "upload.mp3")[1] or ".mp3"
    source_path = None
    wav_path = None

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio file too large for v1 analysis. Please keep files under 60MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            source_path = tmp.name
            tmp.write(raw)

        duration = get_duration_ffprobe(source_path)
        wav_path = convert_preview_to_wav(source_path)
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        if y is None or len(y) == 0:
            raise HTTPException(status_code=422, detail="Could not decode audio preview")

        preview_duration = safe_float(librosa.get_duration(y=y, sr=sr), 0.0) or 0.0
        full_duration = duration or round(float(preview_duration), 2)

        bpm, beat_grid, bpm_confidence = detect_bpm_and_beats(y, sr, full_duration)
        waveform_peaks = build_waveform_peaks(y, max_points=1600)
        energy_curve = build_energy_curve(y, max_points=96)
        energy = round(float(np.mean(energy_curve)), 4) if energy_curve else 0.0
        loudness = estimate_loudness_db(y)
        key, key_confidence = detect_key(y, sr)
        drop_time = detect_drop_time(energy_curve, full_duration)

        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            peak_frames = librosa.util.peak_pick(onset_env, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.2, wait=12)
            peak_times = librosa.frames_to_time(peak_frames, sr=sr).tolist() if len(peak_frames) else []
            peaks = [round(float(t), 2) for t in peak_times[:60]]
        except Exception:
            peaks = []

        return {
            "success": True,
            "analysis": {
                "bpm": bpm,
                "key": key,
                "energy": energy,
                "loudness": loudness,
                "duration": round(float(full_duration), 2),
                "sample_rate": int(sr),
                "channels": 1,
                "waveform_peaks": waveform_peaks,
                "energy_curve": energy_curve,
                "beat_grid": beat_grid,
                "first_beat_offset": beat_grid[0] if beat_grid else 0,
                "peaks": peaks,
                "drop_time": drop_time,
                "confidence": {
                    "bpm": bpm_confidence,
                    "key": key_confidence,
                    "waveform": 0.85 if waveform_peaks else 0.0,
                },
                "analysis_method": "python_librosa_ffmpeg_preview_v1_2",
                "analysis_engine_status": "python_connected",
                "analysis_window_seconds": ANALYSIS_SECONDS,
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(exc)}")
    finally:
        for path in [source_path, wav_path]:
            if path:
                try:
                    os.remove(path)
                except Exception:
                    pass
