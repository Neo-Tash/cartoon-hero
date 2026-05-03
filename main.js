import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import path from "path";
import PocketBase from "pocketbase";

const app = express();
const PORT = process.env.PORT || 3000;

// =============================
// CONFIG
// =============================
app.use(cors());
app.use(express.json());

// PocketBase
const pb = new PocketBase("http://127.0.0.1:8090");

// =============================
// UPLOAD SETUP
// =============================
const uploadDir = "uploads";

// Ensure uploads folder exists
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Multer config
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueName = Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
    cb(null, uniqueName);
  },
});

const upload = multer({ storage });

// =============================
// STATIC FILE SERVING
// =============================
app.use("/uploads", express.static(path.resolve(uploadDir)));

// =============================
// HEALTH CHECK
// =============================
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

// =============================
// ANALYZE TRACK (FIXED)
// =============================
app.post("/api/analyze", upload.single("file"), async (req, res) => {
  try {
    const { userId } = req.body;

    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // 🔥 IMPORTANT: Generate file URL
    const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // Dummy analysis (replace later with real engine)
    const bpm = 120;
    const key = "A Minor";
    const energy = 0.82;

    const record = await pb.collection("analyses").create({
      user_id: userId || "guest",
      filename: req.file.originalname,
      file_url: fileUrl, // ✅ FIXED HERE
      bpm,
      key,
      energy,
    });

    res.json({
      success: true,
      data: record,
    });
  } catch (err) {
    console.error("Analyze error:", err);
    res.status(500).json({ error: err.message });
  }
});

// =============================
// GET USER ANALYSES
// =============================
app.get("/api/my-analyses/:userId", async (req, res) => {
  try {
    const { userId } = req.params;

    const records = await pb.collection("analyses").getFullList({
      filter: `user_id="${userId}"`,
      sort: "-created",
    });

    res.json({
      success: true,
      data: records,
    });
  } catch (err) {
    console.error("Fetch analyses error:", err);
    res.status(500).json({ error: err.message });
  }
});

// =============================
// LOG MIX
// =============================
app.post("/api/log-mix", async (req, res) => {
  try {
    const { userId, fromTrack, toTrack } = req.body;

    if (!userId || !fromTrack || !toTrack) {
      return res.status(400).json({
        error: "Missing required fields: userId, fromTrack, toTrack",
      });
    }

    const bpmDiff = Math.abs((fromTrack.bpm || 120) - (toTrack.bpm || 120));
    const energyDiff = (toTrack.energy || 0.5) - (fromTrack.energy || 0.5);
    const keyMatch = (fromTrack.key || "C") === (toTrack.key || "C");

    await pb.collection("mix_history").create({
      user_id: userId,
      from_track: fromTrack.id || fromTrack,
      to_track: toTrack.id || toTrack,
      bpm_diff: bpmDiff,
      energy_diff: energyDiff,
      key_match: keyMatch,
      timing: "on_phrase",
    });

    res.json({ success: true });
  } catch (err) {
    console.error("Log mix error:", err);
    res.status(500).json({ error: err.message });
  }
});

// =============================
// USER STATS
// =============================
app.get("/api/user-stats/:userId", async (req, res) => {
  try {
    const { userId } = req.params;

    const mixes = await pb.collection("mix_history").getFullList({
      filter: `user_id="${userId}"`,
    });

    res.json({
      totalMixes: mixes.length,
    });
  } catch (err) {
    console.error("User stats error:", err);
    res.status(500).json({ error: err.message });
  }
});

// =============================
// START SERVER
// =============================
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
