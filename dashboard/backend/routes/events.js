/**
 * dashboard/backend/routes/events.js
 * Routes for retrieving, posting, and acknowledging attack events (async version).
 */

const express = require('express');
const router = express.Router();
const dbService = require('../services/db');

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

// POST /api/events - Intake new attack alert from python pipeline
router.post('/', async (req, res) => {
  const { confidence, frame_id, frame_path, timestamp } = req.body;

  if (confidence === undefined || !frame_id) {
    return res.status(400).json({ error: 'Missing confidence or frame_id' });
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

// POST /api/events/:id/ack - Acknowledge alert
router.post('/:id/ack', express.json(), async (req, res) => {
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
