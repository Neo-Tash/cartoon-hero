import express from 'express';
import cors from 'cors';
import morgan from 'morgan';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// Health check
app.get('/hcgi/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Test route
app.get('/hcgi/api/auth/test', (req, res) => {
  res.json({ status: 'auth working' });
});

// Basic login route (temporary)
app.post('/hcgi/api/auth/login', (req, res) => {
  const { email, password } = req.body;

  if (email === 'admin@slickcoherence.com' && password === 'AdminSlick2024!') {
    return res.json({
      success: true,
      user: { email, role: 'admin' }
    });
  }

  return res.status(401).json({ error: 'Invalid credentials' });
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});
