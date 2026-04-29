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
   ANALYZE + SAVE TO DB (FIXED)
========================= */
app.post("/api/analyze", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // 🔥 REQUIRE USER ID (NO MORE GUEST)
    const userId = req.headers.authorization;

    if (!userId) {
      return res.status(401).json({ error: "Missing user identity" });
    }

    console.log("Saving analysis for user:", userId);

    // 🔥 FAKE ANALYSIS (for now)
    const analysis = {
      bpm: 120,
      key: "A Minor",
      energy: 0.82,
      loudness: -6.5
    };

    /* =========================
       SAVE ANALYSIS
    ========================= */
    const { error: saveError } = await supabase
      .from("analyses")
      .insert([
        {
          user_id: userId,
          filename: req.file.originalname,
          bpm: analysis.bpm,
          key: analysis.key,
          energy: analysis.energy,
          analysis_data: analysis
        }
      ]);

    if (saveError) {
      console.error("SUPABASE SAVE ERROR:", saveError);
      return res.status(500).json({ error: saveError.message });
    }

    /* =========================
       LOG ACTIVITY
    ========================= */
    const { error: activityError } = await supabase
      .from("activities")
      .insert([
        {
          user_id: userId,
          action: "analyze",
          metadata: {
            filename: req.file.originalname
          }
        }
      ]);

    if (activityError) {
      console.error("ACTIVITY SAVE ERROR:", activityError);
    }

    /* =========================
       RESPONSE
    ========================= */
    res.json({
      success: true,
      filename: req.file.originalname,
      analysis
    });

  } catch (err) {
    console.error("SERVER ERROR:", err);
    res.status(500).json({ error: "Analysis failed" });
  }
});

/* =========================
   GET USER ANALYSES
========================= */
app.get("/api/my-analyses/:userId", async (req, res) => {
  try {
    const { userId } = req.params;

    console.log("Fetching analyses for:", userId);

    const { data, error } = await supabase
      .from("analyses")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("FETCH ERROR:", error);
      return res.status(500).json({ error: error.message });
    }

    res.json({ success: true, data });

  } catch (err) {
    console.error("SERVER ERROR:", err);
    res.status(500).json({ error: err.message });
  }
});

/* =========================
   GET ACTIVITY
========================= */
app.get("/api/activity/:userId", async (req, res) => {
  try {
    const { userId } = req.params;

    console.log("Fetching activity for:", userId);

    const { data, error } = await supabase
      .from("activities")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(10);

    if (error) {
      console.error("ACTIVITY FETCH ERROR:", error);
      return res.status(500).json({ error: error.message });
    }

    res.json({ success: true, data });

  } catch (err) {
    console.error("SERVER ERROR:", err);
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
