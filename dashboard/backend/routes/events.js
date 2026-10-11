/**
 * dashboard/backend/routes/events.js
 * Routes for retrieving, posting, and acknowledging attack events (async version).
 */

const express = require('express');
const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const router = express.Router();
const dbService = require('../services/db');

const HMAC_SHARED_SECRET = process.env.HMAC_SHARED_SECRET || 'smartgate_zero_trust_hmac_secret_2026';
const JWT_SECRET = process.env.JWT_SECRET || 'infraedge_secure_jwt_secret_key_2026';

function canonicalJson(payloadObj) {
  const keys = Object.keys(payloadObj).filter(k => k !== 'hmac_sig').sort();
  const parts = keys.map(k => {
    const v = payloadObj[k];
    let s;
    if (typeof v === 'string') s = JSON.stringify(v);
    else if (typeof v === 'boolean') s = v ? 'true' : 'false';
    else if (typeof v === 'number') {
      if (Number.isInteger(v)) s = String(v);
      else {
        s = v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
        if (s === '-0' || s === '') s = '0';
      }
    } else if (v === null || v === undefined) s = 'null';
    else s = JSON.stringify(v);
    return `"${k}":${s}`;
  });
  return '{' + parts.join(',') + '}';
}

function verifyHmac(payloadObj) {
  if (!payloadObj || !payloadObj.hmac_sig) return false;
  const received = String(payloadObj.hmac_sig);
  const canonical = canonicalJson(payloadObj);
  const expected = crypto.createHmac('sha256', HMAC_SHARED_SECRET).update(canonical).digest('hex');
  const a = Buffer.from(received);
  const b = Buffer.from(expected);
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

function authenticateJWT(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized: Missing Authorization Bearer header' });
  }
  const token = authHeader.split(' ')[1];
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: 'Forbidden: Invalid or expired token' });
    req.user = user;
    next();
  });
}

// GET /api/events - Retrieve paginated attack alerts
router.get('/', async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;

  try {
    const data = await dbService.getEvents(page, limit);
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/events - Intake new attack alert from python pipeline (HMAC-required)
router.post('/', async (req, res) => {
  const { confidence, frame_id, frame_path, timestamp, hmac_sig } = req.body;

  if (confidence === undefined || !frame_id) {
    return res.status(400).json({ error: 'Missing confidence or frame_id' });
  }
  if (typeof confidence !== 'number' || confidence < 0 || confidence > 1) {
    return res.status(400).json({ error: 'confidence must be a number in [0, 1]' });
  }
  if (!hmac_sig || !verifyHmac(req.body)) {
    return res.status(401).json({ error: 'Invalid or missing HMAC signature' });
  }

  try {
    const id = await dbService.insertEvent({
      timestamp: timestamp || new Date().toISOString(),
      frame_id,
      frame_path: frame_path || '',
      confidence,
      gate_status: 'LOCKED',
      operator_ack: 0,
      notes: ''
    });

    // Notify all connected frontend web sockets immediately
    const io = req.app.get('io');
    if (io) {
      io.emit('attack_alert', {
        id,
        timestamp: timestamp || new Date().toISOString(),
        frame_id,
        frame_path,
        confidence,
        gate_status: 'LOCKED',
        operator_ack: 0
      });
    }

    res.status(201).json({ status: 'ok', id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/events/:id/ack - Acknowledge alert (operator JWT required)
router.post('/:id/ack', express.json(), authenticateJWT, async (req, res) => {
  const { id } = req.params;
  const { notes } = req.body;

  try {
    const ok = await dbService.ackEvent(id, notes || '');
    if (!ok) {
      return res.status(404).json({ error: 'Event not found' });
    }
    
    // Broadcast event ack
    const io = req.app.get('io');
    if (io) {
      io.emit('event_acked', { id, notes });
    }

    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
