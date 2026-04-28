import express from "express";
import cors from "cors";
import multer from "multer";

const app = express();
const upload = multer({ dest: "uploads/" });

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
   LOGIN (FIXED)
========================= */
app.post("/api/login", (req, res) => {
  const { email, password } = req.body;

  // 🔥 TEMP LOGIN (for testing)
  if (email === "admin@slickcoherence.com" && password === "password123") {
    return res.json({
      token: "demo-token-123",
      user: {
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

  // 🔥 FAKE ANALYSIS (for now)
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
   START SERVER
========================= */
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
