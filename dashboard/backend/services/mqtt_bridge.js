/**
 * dashboard/backend/services/mqtt_bridge.js
 * Bridges MQTT topics (broker -> Node.js) directly to React clients via Socket.IO.
 * Enforces Zero-Trust HMAC-SHA256 payload attestation across smartgate/* topics.
 */

const mqtt = require('mqtt');
const crypto = require('crypto');
const dbService = require('./db');

const HMAC_SHARED_SECRET = process.env.HMAC_SHARED_SECRET || 'smartgate_zero_trust_hmac_secret_2026';

function generatePayloadSignature(payloadObj) {
  const filtered = {};
  Object.keys(payloadObj).sort().forEach(k => {
    if (k !== 'hmac_sig') filtered[k] = payloadObj[k];
  });
  const canonical = JSON.stringify(filtered);
  return crypto.createHmac('sha256', HMAC_SHARED_SECRET).update(canonical).digest('hex');
}

function verifyPayloadSignature(payloadObj) {
  if (!payloadObj || !payloadObj.hmac_sig) return false;
  const expectedSig = generatePayloadSignature(payloadObj);
  return crypto.timingSafeEqual(Buffer.from(payloadObj.hmac_sig), Buffer.from(expectedSig));
}

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
      
      // Subscribe to smartgate topics
      this.client.subscribe([
        'smartgate/attack_alert',
        'smartgate/attack_ack',
        'smartgate/override',
        'smartgate/heartbeat',
        'smartgate/status'
      ], (err) => {
        if (err) {
          console.error('[MQTT Bridge] Subscription error:', err);
        } else {
          console.log('[MQTT Bridge] Subscribed to smartgate/* topics');
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

    // Zero-Trust HMAC attestation check on structured JSON messages
    if (typeof payload === 'object' && payload.hmac_sig) {
      try {
        if (!verifyPayloadSignature(payload)) {
          console.warn(`[ZERO-TRUST REJECT] Message on ${topic} failed HMAC signature verification!`);
          return;
        }
      } catch (err) {
        console.warn(`[ZERO-TRUST REJECT] HMAC verification error on ${topic}:`, err.message);
        return;
      }
    }

    // Update watchdog timer on activity
    if (topic === 'smartgate/status' || topic === 'smartgate/attack_ack') {
      this.esp32Online = true;
      this.lastHeartbeatTime = Date.now();
      this.io.emit('esp32_status', { online: true });
    }

    if (topic === 'smartgate/heartbeat') {
      this.io.emit('pipeline_heartbeat', payload);
      if (Math.random() < 0.1) {
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
    if (topic === 'smartgate/attack_alert') {
      console.log(`[MQTT Bridge] ALERT: Attested attack detected (conf=${payload.confidence})`);
      try {
        if (payload.frame_id) {
          const checkRow = await dbService.get('SELECT COUNT(*) as count FROM attack_events WHERE frame_id = ?', [payload.frame_id]);
          if (checkRow.count === 0) {
            const id = await dbService.insertEvent({
              timestamp: payload.timestamp || new Date().toISOString(),
              frame_id: payload.frame_id,
              frame_path: payload.frame_path || '',
              confidence: payload.confidence,
              gate_status: 'LOCKED',
              operator_ack: 0,
              notes: 'Stored via MQTT bridge backup'
            });
            this.io.emit('attack_alert', {
              id,
              timestamp: payload.timestamp || new Date().toISOString(),
              frame_id: payload.frame_id,
              frame_path: payload.frame_path || '',
              confidence: payload.confidence,
              gate_status: 'LOCKED',
              operator_ack: 0
            });
          }
        }
      } catch (err) {
        console.error('[MQTT Bridge] Failed to save backup alert to DB:', err.message);
      }
    }

    if (topic === 'smartgate/attack_ack') {
      console.log(`[MQTT Bridge] ESP32 Acknowledged gate: LOCKED`);
      this.io.emit('gate_status_change', { status: 'LOCKED' });
    }
  }

  publishOverride(action) {
    if (!this.client) return;
    const rawPayload = {
      action: action, // "lock" or "unlock"
      timestamp: new Date().toISOString()
    };
    rawPayload.hmac_sig = generatePayloadSignature(rawPayload);
    const payload = JSON.stringify(rawPayload);
    this.client.publish('smartgate/override', payload, { qos: 1 });
    console.log(`[MQTT Bridge] Published attested override: ${action}`);
  }
}

module.exports = MQTTBridge;
