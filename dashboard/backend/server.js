/**
 * server.js
 * Express server + Socket.IO entry point. Integrates database, JWT auth, and MQTT bridge.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const { Server } = require('socket.io');
try { require('dotenv').config(); } catch (_) { /* dotenv optional */ }

const authRouter = require('./routes/auth');
const eventRouter = require('./routes/events');
const gateRouter = require('./routes/gate');
const statsRouter = require('./routes/stats');
const MQTTBridge = require('./services/mqtt_bridge');
const dbService = require('./services/db');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.ALLOWED_ORIGINS || '*',
    methods: ['GET', 'POST']
  }
});

// Port configuration
const PORT = process.env.PORT || 3000;
const MQTT_BROKER_URL = process.env.MQTT_URL || 'mqtt://localhost:1883';

// Middleware
app.use(cors());
app.use(express.json());

// Serve saved frame image folder statically
const projectRoot = path.join(__dirname, '../..');
const framesDir = path.join(projectRoot, 'data/frames');
app.use('/frames', express.static(framesDir));

// Initialize and link components
const mqttBridge = new MQTTBridge(MQTT_BROKER_URL, io);
mqttBridge.connect();

app.set('io', io);
app.set('mqttBridge', mqttBridge);

// REST Routes
app.use('/api/auth', authRouter);
app.use('/api/events', eventRouter);
app.use('/api/gate', gateRouter);
app.use('/api/stats', statsRouter);

// Live annotated frame snapshot (written by Python pipeline at ~5 FPS)
const liveFramePath = path.join(projectRoot, 'data/frames/live_frame.jpg');
app.get('/api/live-frame', (req, res) => {
  const fs = require('fs');
  if (!fs.existsSync(liveFramePath)) {
    return res.status(404).json({ error: 'No live frame available — pipeline not running' });
  }
  try {
    const stats = fs.statSync(liveFramePath);
    const ageMs = Date.now() - stats.mtimeMs;
    if (ageMs > 10000) {
      return res.status(404).json({ error: 'No live frame available — pipeline stalled' });
    }
  } catch {}
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
  res.setHeader('Content-Type', 'image/jpeg');
  res.sendFile(liveFramePath);
});

// HTTP Heartbeat from Python pipeline (keeps dashboard active without MQTT broker)
app.post('/api/heartbeat', (req, res) => {
  const { fps, sdnl_latency_ms } = req.body || {};
  io.emit('pipeline_heartbeat', {
    fps: typeof fps === 'number' ? fps : 0,
    sdnl_latency_ms: typeof sdnl_latency_ms === 'number' ? sdnl_latency_ms : 0,
    timestamp: new Date().toISOString()
  });
  res.json({ ok: true });
});

// Fallback index endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    name: 'Smart Infrastructure Security Gateway API',
    uptime: process.uptime()
  });
});

// Health endpoint for demo/smoke checks (no auth; liveness only, no secrets)
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    uptime_s: Math.round(process.uptime()),
    esp32_online: !!mqttBridge.esp32Online,
    time: new Date().toISOString(),
  });
});

// WebSocket Connection Handlers
io.on('connection', (socket) => {
  console.log(`[Socket.IO] Client connected: ${socket.id}`);
  socket.emit('esp32_status', { online: mqttBridge.esp32Online });
  socket.on('disconnect', () => {
    console.log(`[Socket.IO] Client disconnected: ${socket.id}`);
  });
});

// Await DB Schema Init before listening to prevent cold-start race (H8 fix)
async function startServer() {
  await dbService.initDbPromise;
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`=== Smart Infrastructure Security Backend Operational ===`);
    console.log(`  REST API Port: ${PORT}`);
    console.log(`  MQTT Broker:   ${MQTT_BROKER_URL}`);
    console.log(`  Frame Storage: ${framesDir}`);
  });
}

startServer();
