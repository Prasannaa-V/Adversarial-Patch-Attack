/**
 * dashboard/backend/routes/gate.js
 * Manual gate override route protected by strict JWT authentication middleware.
 */

const express = require('express');
const jwt = require('jsonwebtoken');
const router = express.Router();

const JWT_SECRET = process.env.JWT_SECRET || 'infraedge_secure_jwt_secret_key_2026';

// Strict JWT Authentication Middleware
const authenticateJWT = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized: Missing or malformed Authorization Bearer header' });
  }

  const token = authHeader.split(' ')[1];
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Forbidden: Invalid or expired JWT token' });
    }
    req.user = user;
    next();
  });
};

// POST /api/gate - Trigger gate manual override (Strictly Protected by JWT)
router.post('/', express.json(), authenticateJWT, (req, res) => {
  const { action } = req.body; // "lock" or "unlock"
  if (action !== 'lock' && action !== 'unlock') {
    return res.status(400).json({ error: 'Action must be "lock" or "unlock"' });
  }

  const mqttBridge = req.app.get('mqttBridge');
  const io = req.app.get('io');

  if (!mqttBridge) {
    return res.status(500).json({ error: 'MQTT Bridge unavailable' });
  }

  try {
    mqttBridge.publishOverride(action);
    
    // Broadcast status change immediately
    if (io) {
      io.emit('gate_status_change', { status: action === 'lock' ? 'LOCKED' : 'UNLOCKED' });
    }

    res.json({ status: 'ok', gate: action === 'lock' ? 'LOCKED' : 'UNLOCKED', user: req.user.username });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
