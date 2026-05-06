import express from "express";
import cors from "cors";
import multer from "multer";
import { createClient } from "@supabase/supabase-js";
import {
  createGenerationJob,
  getGenerationStatus,
  saveGeneratedTrack,
  getGeneratedTracks
} from "./generationProviders/providerRouter.js";

const app = express();

// Use memory storage so Railway does not rely on temporary disk storage.
const upload = multer({ storage: multer.memoryStorage() });

/* =========================
   SUPABASE CONNECTION
========================= */
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

/* =========================
   AUDIO ENGINE CONNECTION
   Set this in Railway on your MAIN API service:
   AUDIO_ENGINE_URL=https://your-audio-engine.up.railway.app
========================= */
const AUDIO_ENGINE_URL = (process.env.AUDIO_ENGINE_URL || "").replace(/\/$/, "");

/* =========================
   LIGHTWEIGHT FALLBACK AUDIO ANALYSIS HELPERS
   Used only if AUDIO_ENGINE_URL is missing or the Python engine fails.
========================= */
const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));

const safeNumber = (value, fallback = 0) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const sanitizeFileName = (name = "track") => {
  return String(name)
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_")
    .slice(0, 140);
};

const normalizePeaks = (peaks = []) => {
  const safePeaks = Array.isArray(peaks) ? peaks.map((v) => Math.abs(Number(v) || 0)) : [];
  const max = Math.max(...safePeaks, 0.0001);
  return safePeaks.map((value) => Number(clamp(value / max).toFixed(4)));
};

const readPcmSample = (buffer, offset, bitsPerSample, audioFormat) => {
  if (offset < 0 || offset >= buffer.length) return 0;

  if (audioFormat === 3 && bitsPerSample === 32 && offset + 4 <= buffer.length) {
    return clamp(Math.abs(buffer.readFloatLE(offset)), 0, 1);
  }

  if (bitsPerSample === 8 && offset + 1 <= buffer.length) {
    return Math.abs((buffer.readUInt8(offset) - 128) / 128);
  }

  if (bitsPerSample === 16 && offset + 2 <= buffer.length) {
    return Math.abs(buffer.readInt16LE(offset) / 32768);
  }

  if (bitsPerSample === 24 && offset + 3 <= buffer.length) {
    const value = buffer.readIntLE(offset, 3);
    return Math.abs(value / 8388608);
  }

  if (bitsPerSample === 32 && offset + 4 <= buffer.length) {
    return Math.abs(buffer.readInt32LE(offset) / 2147483648);
  }

  return 0;
};

const extractWavPeaks = (buffer, targetPeakCount = 1600) => {
  try {
    if (buffer.length < 44) return null;
    if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
      return null;
    }

    let offset = 12;
    let fmt = null;
    let dataStart = null;
    let dataSize = null;

    while (offset + 8 <= buffer.length) {
      const chunkId = buffer.toString("ascii", offset, offset + 4);
      const chunkSize = buffer.readUInt32LE(offset + 4);
      const chunkDataStart = offset + 8;

      if (chunkId === "fmt ") {
        fmt = {
          audioFormat: buffer.readUInt16LE(chunkDataStart),
          channels: buffer.readUInt16LE(chunkDataStart + 2),
          sampleRate: buffer.readUInt32LE(chunkDataStart + 4),
          byteRate: buffer.readUInt32LE(chunkDataStart + 8),
          blockAlign: buffer.readUInt16LE(chunkDataStart + 12),
          bitsPerSample: buffer.readUInt16LE(chunkDataStart + 14)
        };
      }

      if (chunkId === "data") {
        dataStart = chunkDataStart;
        dataSize = chunkSize;
        break;
      }

      offset = chunkDataStart + chunkSize + (chunkSize % 2);
    }

    if (!fmt || dataStart === null || !dataSize || !fmt.sampleRate || !fmt.channels || !fmt.blockAlign) {
      return null;
    }

    const totalFrames = Math.floor(dataSize / fmt.blockAlign);
    const duration = totalFrames / fmt.sampleRate;
    const peakCount = Math.max(400, Math.min(targetPeakCount, Math.floor(totalFrames / 200) || targetPeakCount));
    const framesPerPeak = Math.max(1, Math.floor(totalFrames / peakCount));
    const bytesPerSample = fmt.bitsPerSample / 8;
    const peaks = [];

    for (let i = 0; i < peakCount; i++) {
      const frameStart = i * framesPerPeak;
      const frameEnd = Math.min(totalFrames, frameStart + framesPerPeak);
      let peak = 0;

      for (let frame = frameStart; frame < frameEnd; frame += Math.max(1, Math.floor(framesPerPeak / 48))) {
        let channelPeak = 0;
        for (let ch = 0; ch < fmt.channels; ch++) {
          const sampleOffset = dataStart + frame * fmt.blockAlign + ch * bytesPerSample;
          channelPeak += readPcmSample(buffer, sampleOffset, fmt.bitsPerSample, fmt.audioFormat);
        }
        peak = Math.max(peak, channelPeak / fmt.channels);
      }

      peaks.push(peak);
    }

    return {
      waveform_peaks: normalizePeaks(peaks),
      duration: Number(duration.toFixed(2)),
      sample_rate: fmt.sampleRate,
      channels: fmt.channels,
      analysis_method: "wav_pcm_fallback"
    };
  } catch (err) {
    console.error("WAV peak extraction failed:", err);
    return null;
  }
};

const extractBufferPeaks = (buffer, targetPeakCount = 1600) => {
  const safeBuffer = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer || []);
  const length = safeBuffer.length;
  const peakCount = Math.max(400, Math.min(targetPeakCount, Math.floor(length / 900) || targetPeakCount));
  const bytesPerPeak = Math.max(1, Math.floor(length / peakCount));
  const peaks = [];

  for (let i = 0; i < peakCount; i++) {
    const start = i * bytesPerPeak;
    const end = Math.min(length, start + bytesPerPeak);
    let peak = 0;
    let sum = 0;
    let count = 0;

    for (let j = start; j < end; j += 12) {
      const value = Math.abs((safeBuffer[j] - 128) / 128);
      peak = Math.max(peak, value);
      sum += value;
      count++;
    }

    const avg = count ? sum / count : 0;
    peaks.push((peak * 0.72) + (avg * 0.28));
  }

  return {
    waveform_peaks: normalizePeaks(peaks),
    duration: null,
    sample_rate: null,
    channels: null,
    analysis_method: "compressed_file_byte_peaks_fallback"
  };
};

const buildEnergyCurve = (waveformPeaks = [], points = 64) => {
  if (!waveformPeaks.length) return [];
  const bucketSize = Math.max(1, Math.floor(waveformPeaks.length / points));
  const curve = [];

  for (let i = 0; i < points; i++) {
    const start = i * bucketSize;
    const end = Math.min(waveformPeaks.length, start + bucketSize);
    const bucket = waveformPeaks.slice(start, end);
    const avg = bucket.length ? bucket.reduce((sum, value) => sum + value, 0) / bucket.length : 0;
    curve.push(Number(avg.toFixed(4)));
  }

  return curve;
};

const detectPeakTimes = (waveformPeaks = [], duration = null, maxPeaks = 5) => {
  if (!waveformPeaks.length || !duration) return [];

  const candidates = waveformPeaks
    .map((value, index) => ({ value, index }))
    .filter((item) => item.index > waveformPeaks.length * 0.12 && item.index < waveformPeaks.length * 0.92)
    .sort((a, b) => b.value - a.value);

  const selected = [];
  const minSpacingSeconds = 8;

  for (const item of candidates) {
    const time = (item.index / waveformPeaks.length) * duration;
    const tooClose = selected.some((existing) => Math.abs(existing - time) < minSpacingSeconds);
    if (!tooClose) selected.push(Number(time.toFixed(2)));
    if (selected.length >= maxPeaks) break;
  }

  return selected.sort((a, b) => a - b);
};

const fallbackAnalyzeAudioBuffer = (file) => {
  const wavResult = extractWavPeaks(file.buffer);
  const peakResult = wavResult || extractBufferPeaks(file.buffer);
  const waveformPeaks = peakResult.waveform_peaks || [];
  const energyCurve = buildEnergyCurve(waveformPeaks);
  const energy = energyCurve.length
    ? energyCurve.reduce((sum, value) => sum + value, 0) / energyCurve.length
    : 0.5;

  const detectedPeakTimes = detectPeakTimes(waveformPeaks, peakResult.duration, 5);
  const dropTime = detectedPeakTimes.length
    ? detectedPeakTimes[Math.min(2, detectedPeakTimes.length - 1)]
    : null;

  return {
    bpm: 120,
    key: "A Minor",
    energy: Number(clamp(energy).toFixed(2)),
    loudness: -6.5,
    duration: peakResult.duration,
    waveform_peaks: waveformPeaks,
    energy_curve: energyCurve,
    peaks: detectedPeakTimes,
    drop_time: dropTime,
    beat_grid: [],
    first_beat_offset: 0,
    confidence: {
      bpm: 0.2,
      key: 0.2,
      waveform: peakResult.analysis_method.includes("wav") ? 0.8 : 0.45
    },
    analysis_method: peakResult.analysis_method,
    analysis_engine_status: "fallback_zero_dependency",
    sample_rate: peakResult.sample_rate,
    channels: peakResult.channels
  };
};

const normalizeEngineAnalysis = (rawAnalysis = {}) => {
  const waveformPeaks = Array.isArray(rawAnalysis.waveform_peaks)
    ? rawAnalysis.waveform_peaks.map((v) => Number(clamp(Number(v) || 0).toFixed(4)))
    : [];

  const energyCurve = Array.isArray(rawAnalysis.energy_curve)
    ? rawAnalysis.energy_curve.map((v) => Number(clamp(Number(v) || 0).toFixed(4)))
    : buildEnergyCurve(waveformPeaks, 96);

  const beatGrid = Array.isArray(rawAnalysis.beat_grid)
    ? rawAnalysis.beat_grid.map((v) => Number(safeNumber(v, 0).toFixed(3))).filter((v) => v >= 0)
    : [];

  const peaks = Array.isArray(rawAnalysis.peaks)
    ? rawAnalysis.peaks.map((v) => Number(safeNumber(v, 0).toFixed(2))).filter((v) => v >= 0)
    : [];

  const confidence = rawAnalysis.confidence && typeof rawAnalysis.confidence === "object"
    ? rawAnalysis.confidence
    : {};

  return {
    bpm: safeNumber(rawAnalysis.bpm, 120),
    key: rawAnalysis.key || "Unknown",
    energy: Number(clamp(safeNumber(rawAnalysis.energy, 0.5)).toFixed(4)),
    loudness: safeNumber(rawAnalysis.loudness, -6.5),
    duration: rawAnalysis.duration === null || rawAnalysis.duration === undefined ? null : safeNumber(rawAnalysis.duration, null),
    waveform_peaks: waveformPeaks,
    energy_curve: energyCurve,
    peaks,
    drop_time: rawAnalysis.drop_time === null || rawAnalysis.drop_time === undefined ? null : safeNumber(rawAnalysis.drop_time, null),
    beat_grid: beatGrid,
    first_beat_offset: safeNumber(rawAnalysis.first_beat_offset, beatGrid[0] || 0),
    confidence: {
      bpm: Number(clamp(safeNumber(confidence.bpm, rawAnalysis.bpm_confidence || 0.5)).toFixed(2)),
      key: Number(clamp(safeNumber(confidence.key, rawAnalysis.key_confidence || 0.35)).toFixed(2)),
      waveform: Number(clamp(safeNumber(confidence.waveform, waveformPeaks.length ? 0.85 : 0)).toFixed(2))
    },
    analysis_method: rawAnalysis.analysis_method || "python_audio_engine",
    analysis_engine_status: "python_connected",
    sample_rate: rawAnalysis.sample_rate ?? null,
    channels: rawAnalysis.channels ?? null
  };
};

const analyzeWithPythonAudioEngine = async (file) => {
  if (!AUDIO_ENGINE_URL) {
    console.warn("AUDIO_ENGINE_URL is not set. Using fallback analysis.");
    return null;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);

  try {
    const form = new FormData();
    const blob = new Blob([file.buffer], { type: file.mimetype || "application/octet-stream" });
    form.append("file", blob, file.originalname || "track.mp3");

    const response = await fetch(`${AUDIO_ENGINE_URL}/analyze`, {
      method: "POST",
      body: form,
      signal: controller.signal
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(`Audio engine error ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    const rawAnalysis = result?.analysis || result;

    if (!rawAnalysis || typeof rawAnalysis !== "object") {
      throw new Error("Audio engine returned no analysis object");
    }

    return normalizeEngineAnalysis(rawAnalysis);
  } catch (err) {
    console.error("Python audio engine failed. Falling back to local analysis:", err.message || err);
    return null;
  } finally {
    clearTimeout(timeout);
  }
};

/* =========================
   MIDDLEWARE
========================= */
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(express.json({ limit: "10mb" }));

/* =========================
   HEALTH CHECK
========================= */
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    audio_engine_configured: Boolean(AUDIO_ENGINE_URL),
    audio_engine_url: AUDIO_ENGINE_URL || null
  });
});



/* =========================
   SLICKCOHERENCE MUSIC GENERATION PROVIDER ADAPTER SYSTEM

   Phase 2 provider architecture:
   - mock: active safe testing provider today.
   - external_placeholder: reserved for future third-party AI music APIs.
   - slickcoherence_model_placeholder: reserved for the future SlickCoherence-owned AI model.

   Important architecture rule:
   The frontend should keep using /api/generate-music no matter which provider
   powers the generation behind the scenes. This keeps SlickCoherence ready for
   external APIs now and the owned Option B model later.
========================= */

app.post("/api/generate-music", async (req, res) => {
  try {
    const result = await createGenerationJob(req.body || {});

    if (!result?.success) {
      return res.status(400).json(result || {
        success: false,
        message: "Selected music generation provider is not available yet."
      });
    }

    res.json(result);
  } catch (err) {
    console.error("Generate music provider router error:", err);
    res.status(500).json({ success: false, error: "Music generation job failed" });
  }
});

app.get("/api/generation-status/:jobId", async (req, res) => {
  try {
    const { jobId } = req.params;
    const provider = req.query.provider || "mock";
    const result = await getGenerationStatus(jobId, provider);

    if (!result?.success) {
      return res.status(result?.statusCode || 404).json({
        success: false,
        message: result?.message || "Job not found"
      });
    }

    res.json(result);
  } catch (err) {
    console.error("Generation status provider router error:", err);
    res.status(500).json({ success: false, error: "Generation status check failed" });
  }
});

app.post("/api/save-generated-track", async (req, res) => {
  try {
    const result = await saveGeneratedTrack(req.body || {});
    res.json(result);
  } catch (err) {
    console.error("Save generated track provider router error:", err);
    res.status(500).json({ success: false, error: "Generated track save failed" });
  }
});

app.get("/api/my-generated-tracks/:userId", async (req, res) => {
  try {
    const provider = req.query.provider || "mock";
    const result = await getGeneratedTracks(req.params.userId, provider);
    res.json(result);
  } catch (err) {
    console.error("Generated tracks provider router error:", err);
    res.status(500).json({ success: false, error: "Generated tracks lookup failed" });
  }
});

/* =========================
   LOGIN (TEMP)
========================= */
app.post("/api/login", (req, res) => {
  const { email, password } = req.body;

  if (email === "admin@slickcoherence.com" && password === "password123") {
    return res.json({
      token: "demo-token-123",
      user: { email, username: "Admin" }
    });
  }

  return res.status(401).json({ error: "Invalid credentials" });
});

/* =========================
   ANALYZE + SAVE
========================= */
app.post("/api/analyze", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const userId = req.headers.authorization;
    if (!userId) return res.status(401).json({ error: "Missing user identity" });

    const file = req.file;
    const safeOriginalName = sanitizeFileName(file.originalname || "track");
    const fileName = `${Date.now()}-${safeOriginalName}`;

    // Upload to Supabase Storage first so playback stays stable.
    const { error: uploadError } = await supabase.storage
      .from("tracks")
      .upload(fileName, file.buffer, {
        contentType: file.mimetype,
        upsert: false
      });

    if (uploadError) {
      console.error("Supabase upload error:", uploadError);
      return res.status(500).json({ error: "Upload failed" });
    }

    // Get public playback URL.
    const { data: publicUrlData } = supabase
      .storage
      .from("tracks")
      .getPublicUrl(fileName);

    const fileUrl = publicUrlData.publicUrl;

    // Real analysis from Python service. If it fails, use fallback so uploads do not break.
    const pythonAnalysis = await analyzeWithPythonAudioEngine(file);
    const analysis = pythonAnalysis || fallbackAnalyzeAudioBuffer(file);

    // Save to DB.
    const { data: insertedRows, error } = await supabase.from("analyses").insert([{
      user_id: userId,
      filename: file.originalname,
      bpm: analysis.bpm,
      key: analysis.key,
      energy: analysis.energy,
      analysis_data: analysis,
      file_url: fileUrl
    }]).select("*");

    if (error) throw error;

    await supabase.from("activities").insert([{
      user_id: userId,
      action: "analyze",
      metadata: {
        filename: file.originalname,
        analysis_method: analysis.analysis_method,
        analysis_engine_status: analysis.analysis_engine_status,
        waveform_peaks_count: analysis.waveform_peaks?.length || 0,
        beat_grid_count: analysis.beat_grid?.length || 0,
        bpm: analysis.bpm,
        key: analysis.key
      }
    }]);

    res.json({
      success: true,
      analysis,
      file_url: fileUrl,
      data: insertedRows?.[0] || null
    });

  } catch (err) {
    console.error("Analyze error:", err);
    res.status(500).json({ error: "Analysis failed" });
  }
});

/* =========================
   GET ANALYSES
========================= */
app.get("/api/my-analyses/:userId", async (req, res) => {
  const { data, error } = await supabase
    .from("analyses")
    .select("*")
    .eq("user_id", req.params.userId)
    .order("created_at", { ascending: false });

  if (error) return res.status(500).json({ error: error.message });

  res.json({ success: true, data });
});

/* =========================
   GET ACTIVITY
========================= */
app.get("/api/activity/:userId", async (req, res) => {
  const { data, error } = await supabase
    .from("activities")
    .select("*")
    .eq("user_id", req.params.userId)
    .limit(10);

  if (error) return res.status(500).json({ error: error.message });

  res.json({ success: true, data });
});

/* =========================
   START SERVER
========================= */
const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`🎧 Audio Engine: ${AUDIO_ENGINE_URL || "not configured - fallback mode"}`);
});
