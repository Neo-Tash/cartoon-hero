import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import path from "path";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// =============================
// SAFE UPLOAD CONFIG
// =============================
const uploadDir = "uploads";

if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// 🔥 LIMIT FILE SIZE (VERY IMPORTANT)
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => {
      const name = Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
      cb(null, name);
    },
  }),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB max
  },
});

// =============================
// STATIC FILE SERVING
// =============================
app.use("/uploads", express.static(path.resolve(uploadDir)));

// =============================
// HEALTH
// =============================
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

// =============================
// ANALYZE (SAFE VERSION)
// =============================
app.post("/api/analyze", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // TEMP fake analysis (no heavy CPU)
    const result = {
      bpm: 120,
      key: "A Minor",
      energy: 0.82,
    };

    res.json({
      success: true,
      data: {
        id: Date.now(),
        filename: req.file.originalname,
        file_url: fileUrl,
        ...result,
      },
    });
  } catch (err) {
    console.error("Analyze crash:", err);
    res.status(500).json({ error: err.message });
  }
});

// =============================
// LOG MIX (NO DB YET → SAFE)
// =============================
app.post("/api/log-mix", async (req, res) => {
  try {
    const { userId, fromTrack, toTrack } = req.body;

    if (!userId || !fromTrack || !toTrack) {
      return res.status(400).json({ error: "Missing fields" });
    }

    console.log("Mix logged:", { userId, fromTrack, toTrack });

    res.json({ success: true });
  } catch (err) {
    console.error("Log mix crash:", err);
    res.status(500).json({ error: err.message });
  }
});

// =============================
// USER STATS (SAFE MOCK)
// =============================
app.get("/api/user-stats/:userId", async (req, res) => {
  try {
    res.json({ totalMixes: 0 });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// =============================
// GLOBAL ERROR HANDLER
// =============================
app.use((err, req, res, next) => {
  console.error("GLOBAL ERROR:", err);
  res.status(500).json({ error: "Server error" });
});

// =============================
// START
// =============================
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
