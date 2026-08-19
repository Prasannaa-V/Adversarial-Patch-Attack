/**
 * dashboard/backend/routes/auth.js
 * Operator authentication & JWT token generation service.
 */

const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const dbService = require('../services/db');

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || 'infraedge_secure_jwt_secret_key_2026';

// POST /api/auth/login — Authenticate operator and issue JWT token
router.post('/login', express.json(), async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password required' });
  }

  try {
    const operator = await dbService.get('SELECT * FROM operators WHERE username = ?', [username]);
    if (!operator) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const validPassword = bcrypt.compareSync(password, operator.password_hash);
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign(
      { id: operator.id, username: operator.username, role: operator.role },
      JWT_SECRET,
      { expiresIn: '8h' }
    );

    res.json({
      status: 'ok',
      token,
      user: { id: operator.id, username: operator.username, role: operator.role }
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
