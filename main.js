import express from "express";
import cors from "cors";
import multer from "multer";

const app = express();
const upload = multer({ dest: "uploads/" });

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
   TEMP IN-MEMORY STORAGE
   (Will reset on restart)
========================= */
let analyses = [];

/* =========================
   HEALTH CHECK
========================= */
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

/* =========================
   LOGIN
========================= */
app.post("/api/login", (req, res) => {
  const { email, password } = req.body;

  // TEMP LOGIN (replace later with real auth)
  if (email === "admin@slickcoherence.com" && password === "password123") {
    return res.json({
      token: "demo-token-123",
      user: {
        id: email, // used as userId
        email,
        username: "Admin"
      }
    });
  }

  return res.status(401).json({
    error: "Invalid credentials"
  });
});

/* =========================
   ANALYZE
========================= */
app.post("/api/analyze", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  // Fake analysis (replace later with real engine)
  res.json({
    success: true,
    filename: req.file.originalname,
    size: req.file.size,
    analysis: {
      bpm: 120,
      key: "A Minor",
      energy: 0.82,
      loudness: -6.5
    }
  });
});

/* =========================
   SAVE ANALYSIS
========================= */
app.post("/api/save-analysis", (req, res) => {
  try {
    const { userId, filename, analysis } = req.body;

    if (!userId || !analysis) {
      return res.status(400).json({ error: "Missing data" });
    }

    const newAnalysis = {
      id: Date.now().toString(),
      userId,
      filename,
      bpm: analysis.bpm,
      key: analysis.key,
      energy: analysis.energy,
      full: analysis,
      createdAt: new Date().toISOString()
    };

    analyses.push(newAnalysis);

    res.json({
      success: true,
      data: newAnalysis
    });

  } catch (err) {
    res.status(500).json({
      error: "Failed to save analysis"
    });
  }
});

/* =========================
   GET USER ANALYSES
========================= */
app.get("/api/my-analyses/:userId", (req, res) => {
  try {
    const { userId } = req.params;

    const userAnalyses = analyses.filter(a => a.userId === userId);

    res.json({
      success: true,
      data: userAnalyses
    });

  } catch (err) {
    res.status(500).json({
      error: "Failed to fetch analyses"
    });
  }
});

/* =========================
   START SERVER
========================= */
const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
