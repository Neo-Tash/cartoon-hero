import express from "express";
import cors from "cors";
import multer from "multer";
import { createClient } from "@supabase/supabase-js";

const app = express();

// 🔥 IMPORTANT: use memory storage instead of disk
const upload = multer({ storage: multer.memoryStorage() });

/* =========================
   SUPABASE CONNECTION
========================= */
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY // 🔥 use service role for uploads
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
   ANALYZE + SAVE (FIXED)
========================= */
app.post("/api/analyze", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const userId = req.headers.authorization;
    if (!userId) return res.status(401).json({ error: "Missing user identity" });

    const file = req.file;

    const fileName = `${Date.now()}-${file.originalname}`;

    // 🔥 Upload to Supabase Storage
    const { error: uploadError } = await supabase.storage
      .from("tracks") // ⚠️ bucket must exist in Supabase
      .upload(fileName, file.buffer, {
        contentType: file.mimetype,
      });

    if (uploadError) {
      console.error(uploadError);
      return res.status(500).json({ error: "Upload failed" });
    }

    // 🔥 Get PUBLIC URL
    const { data: publicUrlData } = supabase
      .storage
      .from("tracks")
      .getPublicUrl(fileName);

    const fileUrl = publicUrlData.publicUrl;

    // 🔥 Analysis placeholder
    const analysis = {
      bpm: 120,
      key: "A Minor",
      energy: 0.82,
      loudness: -6.5
    };

    // 🔥 Save to DB
    const { error } = await supabase.from("analyses").insert([{
      user_id: userId,
      filename: file.originalname,
      bpm: analysis.bpm,
      key: analysis.key,
      energy: analysis.energy,
      analysis_data: analysis,
      file_url: fileUrl // ✅ FIXED
    }]);

    if (error) throw error;

    await supabase.from("activities").insert([{
      user_id: userId,
      action: "analyze",
      metadata: { filename: file.originalname }
    }]);

    // 🔥 IMPORTANT: return file_url to frontend
    res.json({
      success: true,
      analysis,
      file_url: fileUrl
    });

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
   START SERVER
========================= */
const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
