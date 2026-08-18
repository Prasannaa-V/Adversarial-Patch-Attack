/**
 * dashboard/backend/routes/gate.js
 * Manual gate override route.
 */

const express = require('express');
const router = express.Router();

// POST /api/gate - Trigger gate manual override
router.post('/', (express.json()), (req, res) => {
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
    
    // Broadcast status change immediately (optimistic UI update)
    if (io) {
      io.emit('gate_status_change', { status: action === 'lock' ? 'LOCKED' : 'UNLOCKED' });
    }

    res.json({ status: 'ok', gate: action === 'lock' ? 'LOCKED' : 'UNLOCKED' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
