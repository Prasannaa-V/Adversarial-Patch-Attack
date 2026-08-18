/**
 * dashboard/backend/routes/stats.js
 * Analytical statistics routes (async version).
 */

const express = require('express');
const router = express.Router();
const dbService = require('../services/db');

// GET /api/stats - Retrieve aggregated statistics
router.get('/', async (req, res) => {
  try {
    const stats = await dbService.getStats();
    res.json(stats);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
