import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// ==========================
// FILE UPLOAD SETUP
// ==========================
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB
});

// ==========================
// HEALTH CHECK
// ==========================
app.get('/hcgi/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ==========================
// AUTH TEST
// ==========================
app.get('/hcgi/api/auth/test', (req, res) => {
  res.json({ status: 'auth working' });
});

// ==========================
// LOGIN (TEMP)
// ==========================
app.post('/hcgi/api/auth/login', (req, res) => {
  const { email, password } = req.body;

  if (email === 'admin@slickcoherence.com' && password === 'AdminSlick2024!') {
    return res.json({
      success: true,
      token: 'demo-token-123',
      user: { email, role: 'admin' }
    });
  }

  return res.status(401).json({ error: 'Invalid credentials' });
});

// ==========================
// 🔥 ANALYZE ROUTE (FIX)
// ==========================
app.post('/hcgi/api/analyze', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Fake analysis (for now)
    const result = {
      fileName: req.file.originalname,
      size: req.file.size,
      bpm: 124,
      key: 'A Minor',
      energy: 0.78,
      message: 'Analysis complete (mock data)'
    };

    res.json(result);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Analysis failed' });
  }
});

// ==========================
// START SERVER
// ==========================
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});
