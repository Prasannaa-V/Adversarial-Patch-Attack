/**
 * dashboard/backend/services/mqtt_bridge.js
 * Bridges MQTT topics (broker -> Node.js) directly to React clients via Socket.IO.
 * Also handles listening for ESP32 alerts & status changes (async database version).
 */

const mqtt = require('mqtt');
const dbService = require('./db');

class MQTTBridge {
  constructor(brokerUrl, io) {
    this.brokerUrl = brokerUrl;
    this.io = io;
    this.client = null;
    this.esp32Online = false;
    this.lastHeartbeatTime = 0;

    // Check ESP32 online status based on heartbeat/status activity
    setInterval(() => {
      const now = Date.now();
      const offline = now - this.lastHeartbeatTime > 15000;
      if (offline && this.esp32Online) {
        this.esp32Online = false;
        this.io.emit('esp32_status', { online: false });
        console.log('[MQTT Bridge] ESP32 considered offline (no messages for 15s)');
      }
    }, 5000);
  }

  connect() {
    console.log(`[MQTT Bridge] Connecting to broker at ${this.brokerUrl}`);
    this.client = mqtt.connect(this.brokerUrl);

    this.client.on('connect', () => {
      console.log('[MQTT Bridge] Connected to MQTT broker');
      
      // Subscribe to all infraedge topics
      this.client.subscribe([
        'infraedge/attack_alert',
        'infraedge/attack_ack',
        'infraedge/override',
        'infraedge/heartbeat',
        'infraedge/status'
      ], (err) => {
        if (err) {
          console.error('[MQTT Bridge] Subscription error:', err);
        } else {
          console.log('[MQTT Bridge] Subscribed to infraedge topics');
        }
      });
    });

    this.client.on('message', async (topic, message) => {
      await this.handleMessage(topic, message.toString());
    });

    this.client.on('error', (err) => {
      console.error('[MQTT Bridge] Connection error:', err);
    });
  }

  async handleMessage(topic, payloadStr) {
    let payload = {};
    try {
      payload = JSON.parse(payloadStr);
    } catch (e) {
      payload = { raw: payloadStr };
    }

    // Update watchdog timer on any message from ESP32 or heartbeat from pipeline
    if (topic === 'infraedge/status' || topic === 'infraedge/attack_ack') {
      this.esp32Online = true;
      this.lastHeartbeatTime = Date.now();
      this.io.emit('esp32_status', { online: true });
    }

    if (topic === 'infraedge/heartbeat') {
      this.io.emit('pipeline_heartbeat', payload);
      // Log pipeline health to DB periodically (every ~30s or when values change)
      if (Math.random() < 0.1) { // lightweight sample logging
        try {
          await dbService.insertHealth({
            timestamp: new Date().toISOString(),
            fps: payload.fps || 0,
            sdnl_latency_ms: payload.sdnl_latency_ms || 0,
            esp32_online: this.esp32Online ? 1 : 0
          });
        } catch (err) {
          console.error('[MQTT Bridge] Failed to save health status to DB:', err.message);
        }
      }
    }

    // Broadcast message to WebSockets
    this.io.emit('mqtt_message', { topic, payload });

    // Handle specific topic actions
    if (topic === 'infraedge/attack_alert') {
      console.log(`[MQTT Bridge] ALERT: Attack detected (conf=${payload.confidence})`);
      // Event gets stored via pipeline HTTP POST endpoint as well, 
      // but if the HTTP POST fails, we capture and save it here as backup:
      try {
        if (payload.frame_id) {
          const checkRow = await dbService.get('SELECT COUNT(*) as count FROM attack_events WHERE frame_id = ?', [payload.frame_id]);
          if (checkRow.count === 0) {
            await dbService.insertEvent({
              timestamp: payload.timestamp || new Date().toISOString(),
              frame_id: payload.frame_id,
              frame_path: payload.frame_path || '',
              confidence: payload.confidence,
              gate_status: 'LOCKED',
              operator_ack: 0,
              notes: 'Stored via MQTT bridge backup'
            });
          }
        }
      } catch (err) {
        console.error('[MQTT Bridge] Failed to save backup alert to DB:', err.message);
      }
    }

    if (topic === 'infraedge/attack_ack') {
      console.log(`[MQTT Bridge] ESP32 Acknowledged gate: LOCKED`);
      this.io.emit('gate_status_change', { status: 'LOCKED' });
    }
  }

  publishOverride(action) {
    if (!this.client) return;
    const payload = JSON.stringify({
      action: action, // "lock" or "unlock"
      timestamp: new Date().toISOString()
    });
    this.client.publish('infraedge/override', payload, { qos: 1 });
    console.log(`[MQTT Bridge] Published override: ${action}`);
  }
}

module.exports = MQTTBridge;
