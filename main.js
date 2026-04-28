import express from 'express';
import cors from 'cors';
import multer from 'multer';

const app = express();

// =========================
// ✅ CORS CONFIG (VERY IMPORTANT)
// =========================
app.use(cors({
  origin: [
    'https://slickcoherence.com',
    'http://localhost:3000'
  ],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// =========================
// ✅ MIDDLEWARE
// =========================
app.use(express.json());

// =========================
// ✅ FILE UPLOAD SETUP
// =========================
const storage = multer.memoryStorage();
const upload = multer({ storage });

// =========================
// ✅ HEALTH CHECK ROUTE
// =========================
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// =========================
// ✅ LOGIN ROUTE (TEMP BASIC)
// =========================
app.post('/api/login', (req, res) => {
  const { email, password } = req.body;

  // TEMP LOGIN (you can replace later with real auth)
  if (email === 'admin@slickcoherence.com' && password === 'password123') {
    return res.json({
      success: true,
      token: 'demo-token-123'
    });
  }

  res.status(401).json({
    success: false,
    error: 'Invalid credentials'
  });
});

// =========================
// ✅ ANALYZE ROUTE
// =========================
app.post('/api/analyze', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded'
      });
    }

    console.log('🎧 File received:', req.file.originalname);

    // 🔥 MOCK ANALYSIS (replace later with real AI)
    const analysis = {
      bpm: 128,
      key: 'A Minor',
      energy: 0.82,
      duration: 180,
      loudness: -6.5,
      genre: 'Amapiano / House',
      mood: 'Energetic',
    };

    res.json({
      success: true,
      filename: req.file.originalname,
      size: req.file.size,
      analysis
    });

  } catch (error) {
    console.error('❌ Analyze error:', error);

    res.status(500).json({
      success: false,
      error: 'Analysis failed'
    });
  }
});

// =========================
// ✅ DEFAULT ROUTE (IMPORTANT)
// =========================
app.get('/', (req, res) => {
  res.send('SlickCoherence API is running 🚀');
});

// =========================
// ❌ REMOVE ANY OLD /hcgi ROUTES (DO NOT ADD THEM)
// =========================


// =========================
// ✅ START SERVER
// =========================
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
