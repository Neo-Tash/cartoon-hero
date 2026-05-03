import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import path from "path";
import { createClient } from "@supabase/supabase-js";

const app = express();

/* =========================
   ENSURE UPLOAD FOLDER EXISTS
========================= */
const uploadDir = path.join(process.cwd(), "uploads");

if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

/* =========================
   MULTER STORAGE (FIXED)
========================= */
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const safeName = file.originalname.replace(/\s+/g, "_");
    const uniqueName = Date.now() + "-" + safeName;
    cb(null, uniqueName);
  }
});

const upload = multer({ storage });

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
app.use("/uploads", express.static(uploadDir));

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
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const userId = req.headers.authorization;
    if (!userId) {
      return res.status(401).json({ error: "Missing user identity" });
    }

    // 🔥 FORCE PRODUCTION URL (DO NOT USE req.get("host") HERE)
    const BASE_URL = "https://cartoon-hero-production.up.railway.app";

    const fileUrl = `${BASE_URL}/uploads/${req.file.filename}`;

    const analysis = {
      bpm: 120,
      key: "A Minor",
      energy: 0.82,
      loudness: -6.5
    };

    const { error } = await supabase.from("analyses").insert([{
      user_id: userId,
      filename: req.file.originalname,
      bpm: analysis.bpm,
      key: analysis.key,
      energy: analysis.energy,
      analysis_data: analysis,
      file_url: fileUrl // ✅ GUARANTEED VALUE NOW
    }]);

    if (error) throw error;

    await supabase.from("activities").insert([{
      user_id: userId,
      action: "analyze",
      metadata: { filename: req.file.originalname }
    }]);

    res.json({
      success: true,
      analysis,
      file_url: fileUrl
    });

  } catch (err) {
    console.error("ANALYZE ERROR:", err);
    res.status(500).json({ error: err.message });
  }
});

/* =========================
   GET ANALYSES
========================= */
app.get("/api/my-analyses/:userId", async (req, res) => {
  const { data, error
