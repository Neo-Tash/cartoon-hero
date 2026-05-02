import express from "express";
import cors from "cors";
import multer from "multer";
import { createClient } from "@supabase/supabase-js";

const app = express();
const upload = multer({ dest: "uploads/" });

/* =========================
   SUPABASE CONNECTION
========================= */
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

/* =========================
   MIDDLEWARE
========================= */
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(express.json());
app.use("/uploads", express.static("uploads"));

/* =========================
   HEALTH CHECK
========================= */
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
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

    const analysis = {
      bpm: 120,
      key: "A Minor",
      energy: 0.82,
      loudness: -6.5
    };

    const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    const { error } = await supabase.from("analyses").insert([{
      user_id: userId,
      filename: req.file.originalname,
      bpm: analysis.bpm,
      key: analysis.key,
      energy: analysis.energy,
      analysis_data: analysis,
      file_url: fileUrl
    }]);

    if (error) throw error;

    await supabase.from("activities").insert([{
      user_id: userId,
      action: "analyze",
      metadata: { filename: req.file.originalname }
    }]);

    res.json({ success: true, analysis });

  } catch (err) {
    console.error(err);
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
   AI DJ CLOUD ENGINE
========================= */

/* 🎧 1. LOG MIX */
app.post("/api/log-mix", async (req, res) => {
  try {
    const { userId, fromTrack, toTrack } = req.body;

    if (!userId || !fromTrack || !toTrack) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const bpmDiff = Math.abs((fromTrack.bpm || 120) - (toTrack.bpm || 120));
    const energyDiff = (toTrack.energy || 0.5) - (fromTrack.energy || 0.5);
    const keyMatch = (fromTrack.key || "") === (toTrack.key || "");

    const { error } = await supabase.from("mix_history").insert([{
      user_id: userId,
      from_track: fromTrack.id || fromTrack,
      to_track: toTrack.id || toTrack,
      bpm_diff: bpmDiff,
      energy_diff: energyDiff,
      key_match: keyMatch,
      timing: "on_phrase"
    }]);

    if (error) throw error;

    res.json({ success: true });

  } catch (err) {
    console.error("log-mix error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* 🧠 2. USER PREFERENCES ENGINE */
const getUserPreferences = async (userId) => {
  const { data } = await supabase
    .from("mix_history")
    .select("*")
    .eq("user_id", userId);

  if (!data || data.length < 5) {
    return { bpmTolerance: 10, energyBias: 0, keyStrictness: 0.7 };
  }

  let totalBpm = 0, totalEnergy = 0, keyMatches = 0;

  data.forEach(d => {
    totalBpm += d.bpm_diff || 0;
    totalEnergy += d.energy_diff || 0;
    if (d.key_match) keyMatches++;
  });

  return {
    bpmTolerance: (totalBpm / data.length) + 2,
    energyBias: totalEnergy / data.length,
    keyStrictness: keyMatches / data.length
  };
};

/* 🤖 3. AI TRACK SUGGESTION */
app.post("/api/ai-suggest", async (req, res) => {
  try {
    const { userId, currentTrack, library } = req.body;

    if (!userId || !currentTrack || !library) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const prefs = await getUserPreferences(userId);

    let best = null;
    let bestScore = -1;

    library.forEach(track => {
      if (!track || track.id === currentTrack.id) return;

      const bpmDiff = Math.abs((currentTrack.bpm || 120) - (track.bpm || 120));
      const energyDiff = (track.energy || 0.5) - (currentTrack.energy || 0.5);
      const keyMatch = currentTrack.key === track.key;

      let score = 0;

      score += bpmDiff < prefs.bpmTolerance ? 1 : 0.5;
      score += (prefs.energyBias > 0 ? (energyDiff >= 0 ? 1 : 0.5) : 0.8);
      score += keyMatch ? prefs.keyStrictness : 0.3;

      if (score > bestScore) {
        bestScore = score;
        best = track;
      }
    });

    res.json({ success: true, suggestion: best, confidence: bestScore });

  } catch (err) {
    console.error("ai-suggest error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* 📊 4. UPDATE AI PREFERENCES */
app.post("/api/update-preferences", async (req, res) => {
  try {
    const { userId } = req.body;

    const prefs = await getUserPreferences(userId);

    const { data: existing } = await supabase
      .from("ai_preferences")
      .select("*")
      .eq("user_id", userId)
      .single();

    if (existing) {
      await supabase.from("ai_preferences").update({
        bpm_tolerance: prefs.bpmTolerance,
        energy_bias: prefs.energyBias,
        key_strictness: prefs.keyStrictness
      }).eq("user_id", userId);
    } else {
      await supabase.from("ai_preferences").insert([{
        user_id: userId,
        bpm_tolerance: prefs.bpmTolerance,
        energy_bias: prefs.energyBias,
        key_strictness: prefs.keyStrictness
      }]);
    }

    res.json({ success: true, preferences: prefs });

  } catch (err) {
    console.error("update-preferences error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* 📈 5. USER STATS */
app.get("/api/user-stats/:userId", async (req, res) => {
  try {
    const { data } = await supabase
      .from("mix_history")
      .select("*")
      .eq("user_id", req.params.userId);

    if (!data || !data.length) {
      return res.json({ totalMixes: 0 });
    }

    let totalBpm = 0, totalEnergy = 0, keyMatches = 0;

    data.forEach(m => {
      totalBpm += m.bpm_diff || 0;
      totalEnergy += m.energy_diff || 0;
      if (m.key_match) keyMatches++;
    });

    res.json({
      totalMixes: data.length,
      avgBpmDiff: totalBpm / data.length,
      avgEnergyDiff: totalEnergy / data.length,
      keyMatchRate: keyMatches / data.length
    });

  } catch (err) {
    console.error("user-stats error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* =========================
   START SERVER
========================= */
const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
