import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';

const app = express();
const PORT = process.env.PORT || 3000;

/* =========================
   ✅ CORS FIX (CRITICAL)
========================= */
app.use(cors({
  origin: 'https://slickcoherence.com',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  credentials: true
}));

app.options('*', cors({
  origin: 'https://slickcoherence.com',
  credentials: true
}));

/* =========================
   ✅ MIDDLEWARE
========================= */
app.use(express.json());
app.use(morgan('dev'));

/* =========================
   ✅ FILE UPLOAD SETUP
========================= */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB
});

/* =========================
   ✅ HEALTH CHECK
========================= */
app.get('/hcgi/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

/* =========================
   ✅ AUTH TEST
========================= */
app.get('/hcgi/api/auth/test', (req, res) => {
  res.json({ status: 'auth working' });
});

/* =========================
   ✅ LOGIN ROUTE
========================= */
app.post('/hcgi/api/auth/login', (req, res) => {
  const { email, password } = req.body;

  if (email === 'admin@slickcoherence.com' && password === 'AdminSlick2024!') {
    return res.json({
      success: true,
      token: 'demo-token-123',
      user: {
        email,
        role: 'admin'
      }
    });
  }

  return res.status(401).json({ error: 'Invalid credentials' });
});

/* =========================
   ✅ ANALYZE ROUTE (FIXES YOUR ERROR)
========================= */
app.post('/hcgi/api/analyze', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // 🔥 TEMP MOCK ANALYSIS (we upgrade later)
    const result = {
      success: true,
      fileName: req.file.originalname,
      size: req.file.size,
      analysis: {
        bpm: 124,
        key: 'A Minor',
        energy: 'Medium',
        duration: '3:45',
        message: 'Analysis complete (mock data)'
      }
    };

    return res.json(result);

  } catch (error) {
    console.error('Analyze error:', error);
    return res.status(500).json({ error: 'Analysis failed' });
  }
});

/* =========================
   ✅ START SERVER
========================= */
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
