/**
 * server.js
 * Express server + Socket.IO entry point. Integrates database and MQTT bridge.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const { Server } = require('socket.io');

const eventRouter = require('./routes/events');
const gateRouter = require('./routes/gate');
const statsRouter = require('./routes/stats');
const MQTTBridge = require('./services/mqtt_bridge');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: '*',
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
// So the dashboard can render frame_path like: http://localhost:3000/frames/2026-08-17/attack_abc.jpg
const projectRoot = path.join(__dirname, '../..');
const framesDir = path.join(projectRoot, 'data/frames');
app.use('/frames', express.static(framesDir));

// Initialize and link components
const mqttBridge = new MQTTBridge(MQTT_BROKER_URL, io);
mqttBridge.connect();

app.set('io', io);
app.set('mqttBridge', mqttBridge);

// REST Routes
app.use('/api/events', eventRouter);
app.use('/api/gate', gateRouter);
app.use('/api/stats', statsRouter);

// Fallback index endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    name: 'InfraEdge-Shield Security Gateway API',
    uptime: process.uptime()
  });
});

// WebSocket Connection Handlers
io.on('connection', (socket) => {
  console.log(`[Socket.IO] New client connected: ${socket.id}`);
  
  // Send current ESP32 online state on connection
  socket.emit('esp32_status', { online: mqttBridge.esp32Online });

  socket.on('disconnect', () => {
    console.log(`[Socket.IO] Client disconnected: ${socket.id}`);
  });
});

// Start listening
server.listen(PORT, '0.0.0.0', () => {
  console.log(`=== InfraEdge-Shield Backend Operational ===`);
  console.log(`  REST API Port: ${PORT}`);
  console.log(`  MQTT Broker:   ${MQTT_BROKER_URL}`);
  console.log(`  Frame Storage: ${framesDir}`);
});
