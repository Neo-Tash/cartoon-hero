# SlickCoherence Audio Engine v1

Python FastAPI microservice for real audio analysis.

## Endpoints

- `GET /health`
- `POST /analyze` with multipart form-data field named `file`

## Returns

- Real BPM estimate
- Basic key estimate
- Duration
- Loudness
- Waveform peaks
- Energy curve
- Beat grid foundation
- First beat offset
- Drop time estimate
- Confidence scores

## Railway setup

Deploy this folder as a separate Railway service.
Make sure Railway uses the provided `nixpacks.toml` so FFmpeg and libsndfile are available for MP3/WAV decoding.
