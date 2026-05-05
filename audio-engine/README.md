# SlickCoherence Audio Engine v1.2 Railway Safe

This version decodes only a 90-second preview with FFmpeg at 22050 Hz mono before running librosa. This prevents Railway 502 crashes on MP3 analysis while still returning real BPM, key estimate, waveform peaks, energy curve, beat grid foundation, duration, sample rate, and loudness.
