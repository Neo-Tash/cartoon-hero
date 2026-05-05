# SlickCoherence Audio Engine v1.3

Railway-safe FastAPI audio analysis engine.

Fix in v1.3:
- Removes dependency on system `ffmpeg` binary being installed.
- Uses `imageio-ffmpeg` bundled FFmpeg binary for decode.
- Uses `mutagen` for duration metadata where possible.

Health check:
`GET /health`

Analyze:
`POST /analyze` with multipart field `file`.
