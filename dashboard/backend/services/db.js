/**
 * dashboard/backend/services/db.js
 * SQLite Database initialization and service methods using the standard sqlite3 package.
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');
const bcrypt = require('bcryptjs');

const dbPath = path.join(__dirname, '../db/infraedge.db');
const schemaPath = path.join(__dirname, '../db/schema.sql');

// Ensure db directory exists
fs.mkdirSync(path.dirname(dbPath), { recursive: true });

// Open SQLite database
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('[DB] Error opening database:', err.message);
  } else {
    console.log('[DB] Connected to SQLite database');
  }
});

// Helper wrapper to run query with Promise
const run = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function (err) {
      if (err) reject(err);
      else resolve(this);
    });
  });
};

// Helper wrapper to get single row
const get = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
};

// Helper wrapper to get all rows
const all = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
};

// Helper to run raw SQL script (multiple statements)
const exec = (sql) => {
  return new Promise((resolve, reject) => {
    db.exec(sql, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
};

// Initialize schema and seed operator account
const initDb = async () => {
  try {
    const schema = fs.readFileSync(schemaPath, 'utf8');
    await exec(schema);
    
    // Check if operator already exists
    const userRow = await get('SELECT COUNT(*) as count FROM operators WHERE username = ?', ['admin']);
    if (userRow.count === 0) {
      const hash = bcrypt.hashSync('password123', 10);
      await run('INSERT INTO operators (username, password_hash, role) VALUES (?, ?, ?)', ['admin', hash, 'admin']);
      console.log('[DB] Seeded default administrator account: admin / password123');
    }
  } catch (err) {
    console.error('[DB] Schema initialization error:', err.message);
  }
};

const initDbPromise = initDb();

module.exports = {
  db,
  initDbPromise,
  run,
  get,
  all,
  
  // Event Operations
  async insertEvent(event) {
    const sql = `
      INSERT OR IGNORE INTO attack_events (timestamp, frame_id, frame_path, confidence, gate_status, operator_ack, notes)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;
    const result = await run(sql, [
      event.timestamp,
      event.frame_id,
      event.frame_path || '',
      event.confidence,
      event.gate_status || 'LOCKED',
      event.operator_ack || 0,
      event.notes || ''
    ]);
    return result.lastID;
  },

  async getEvents(page = 1, limit = 20) {
    const offset = (page - 1) * limit;
    const rows = await all('SELECT * FROM attack_events ORDER BY timestamp DESC LIMIT ? OFFSET ?', [limit, offset]);
    const countRow = await get('SELECT COUNT(*) as count FROM attack_events');
    const total = countRow.count;
    return {
      rows,
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit)
    };
  },

  async ackEvent(id, notes = '') {
    const result = await run('UPDATE attack_events SET operator_ack = 1, notes = ? WHERE id = ?', [notes, id]);
    return result.changes > 0;
  },

  // Health / Stats Operations
  async insertHealth(health) {
    const sql = `
      INSERT INTO system_health (timestamp, fps, sdnl_latency_ms, esp32_online)
      VALUES (?, ?, ?, ?)
    `;
    const result = await run(sql, [
      health.timestamp,
      health.fps,
      health.sdnl_latency_ms,
      health.esp32_online
    ]);
    return result.lastID;
  },

  async getStats() {
    const totalRow = await get('SELECT COUNT(*) as count FROM attack_events');
    const avgRow = await get('SELECT AVG(confidence) as avg FROM attack_events');
    const pendingRow = await get('SELECT COUNT(*) as count FROM attack_events WHERE operator_ack = 0');
    
    // Attack counts grouped by hour for today (UTC)
    const hourlyDistribution = await all(`
      SELECT strftime('%H', timestamp) as hour, COUNT(*) as count 
      FROM attack_events 
      WHERE timestamp >= date('now', 'start of day')
      GROUP BY hour 
      ORDER BY hour ASC
    `);

    return {
      totalAttacks: totalRow.count,
      avgConfidence: parseFloat((avgRow.avg || 0).toFixed(4)),
      pendingAck: pendingRow.count,
      hourlyDistribution
    };
  }
};
