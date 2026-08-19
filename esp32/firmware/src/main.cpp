/*
 * InfraEdge-Shield — ESP32 Failsafe Firmware
 * File: esp32/firmware/src/main.cpp
 *
 * State Machine: IDLE <-> LOCKED <-> OVERRIDE
 * Watchdog: auto-lock gate if no heartbeat from PC in 15 seconds
 * MQTT QoS 1 for safety-critical messages
 *
 * PlatformIO: see platformio.ini
 * Flash and upload via: pio run --target upload
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Ticker.h>
#include <mbedtls/md.h>
#include <vector>
#include <algorithm>

// ── Shared Secret for Zero-Trust HMAC-SHA256 Attestation ──────────────────────
const char* HMAC_SECRET_KEY = "smartgate_zero_trust_hmac_secret_2026";

bool verifyPayloadHmac(const JsonObject& doc) {
  if (!doc.containsKey("hmac_sig")) return false;
  String receivedSig = doc["hmac_sig"].as<String>();

  std::vector<String> keys;
  for (JsonPair kv : doc) {
    String k = kv.key().c_str();
    if (k != "hmac_sig") {
      keys.push_back(k);
    }
  }
  std::sort(keys.begin(), keys.end());

  String canonical = "{";
  for (size_t i = 0; i < keys.size(); i++) {
    if (i > 0) canonical += ",";
    String k = keys[i];
    canonical += "\"" + k + "\":";
    if (doc[k].is<JsonString>()) {
      canonical += "\"" + doc[k].as<String>() + "\"";
    } else if (doc[k].is<float>() || doc[k].is<double>()) {
      char fbuf[32];
      snprintf(fbuf, sizeof(fbuf), "%.4f", doc[k].as<float>());
      canonical += String(fbuf);
    } else {
      canonical += doc[k].as<String>();
    }
  }
  canonical += "}";

  byte hmacResult[32];
  mbedtls_md_context_t ctx;
  mbedtls_md_type_t md_type = MBEDTLS_MD_SHA256;
  mbedtls_md_init(&ctx);
  mbedtls_md_setup(&ctx, mbedtls_md_info_from_type(md_type), 1);
  mbedtls_md_hmac_starts(&ctx, (const unsigned char*)HMAC_SECRET_KEY, strlen(HMAC_SECRET_KEY));
  mbedtls_md_hmac_update(&ctx, (const unsigned char*)canonical.c_str(), canonical.length());
  mbedtls_md_hmac_finish(&ctx, hmacResult);
  mbedtls_md_free(&ctx);

  char calculatedSig[65];
  for (int i = 0; i < 32; i++) {
    sprintf(&calculatedSig[i * 2], "%02x", hmacResult[i]);
  }
  calculatedSig[64] = '\0';

  return receivedSig.equalsIgnoreCase(calculatedSig);
}

// ── WiFi Credentials ─────────────────────────────────────────────────────────
// EDIT THESE before flashing
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// ── MQTT Broker ───────────────────────────────────────────────────────────────
// Set to the IP address of the PC running Mosquitto
#define MQTT_BROKER_IP   "192.168.1.100"   // EDIT: PC's local IP
#define MQTT_BROKER_PORT 1883
#define MQTT_CLIENT_ID   "infraedge_esp32"

// ── GPIO Pins ─────────────────────────────────────────────────────────────────
#define RELAY_PIN     26   // Gate lock relay (LOW = de-energized = LOCKED; HIGH = energized = UNLOCKED for fail-secure posture)
#define LED_RED       27   // Attack alert indicator (5 Hz flash)
#define BUZZER_PIN    25   // Alarm tone (880 Hz PWM)
#define LED_GREEN     32   // System healthy indicator
#define OVERRIDE_BTN  33   // Manual override pushbutton (GPIO33 supports internal INPUT_PULLUP)

// ── MQTT Topics ───────────────────────────────────────────────────────────────
const char* TOPIC_ALERT     = "smartgate/attack_alert";
const char* TOPIC_ACK       = "smartgate/attack_ack";
const char* TOPIC_OVERRIDE  = "smartgate/override";
const char* TOPIC_HEARTBEAT = "smartgate/heartbeat";
const char* TOPIC_STATUS    = "smartgate/status";

// ── State Machine ─────────────────────────────────────────────────────────────
enum SystemState { IDLE, LOCKED, OVERRIDE_UNLOCKED };
SystemState currentState = IDLE;

// ── Timing ────────────────────────────────────────────────────────────────────
unsigned long lastHeartbeatMs    = 0;
unsigned long lockStartMs        = 0;
const unsigned long WATCHDOG_MS  = 15000UL;  // 15 seconds
const unsigned long LOCKOUT_MS   = 30000UL;  // 30-second auto-release

// ── Buzzer (non-blocking) ───────────────────────────────────────────────────
// The alarm tone is started when the gate locks and stopped from loop() after
// BUZZER_MS, so mqttCallback()/activateLock() never block mqttClient.loop().
unsigned long buzzerStartMs      = 0;
bool          buzzerActive       = false;
const unsigned long BUZZER_MS    = 2000UL;   // 2-second alarm tone

// ── LED Flash ─────────────────────────────────────────────────────────────────
Ticker ledTicker;
bool   ledState = false;

// ── MQTT + WiFi clients ───────────────────────────────────────────────────────
WiFiClient     wifiClient;
PubSubClient   mqttClient(wifiClient);

// ── Function Declarations ─────────────────────────────────────────────────────
void connectWiFi();
void connectMQTT();
void publishACK();
void publishStatus();
void activateLock();
void deactivateLock();
void startLedFlash();
void stopLedFlash();
void startBuzzer();
void stopBuzzer();
void checkBuzzer();
void mqttCallback(char* topic, byte* payload, unsigned int length);
void checkWatchdog();
void checkLockoutTimer();
void handleOverrideButton();


// =============================================================================
// Setup
// =============================================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== InfraEdge-Shield ESP32 Booting ===");

  // GPIO setup
  pinMode(RELAY_PIN,    OUTPUT);
  pinMode(LED_RED,      OUTPUT);
  pinMode(LED_GREEN,    OUTPUT);
  pinMode(BUZZER_PIN,   OUTPUT);
  pinMode(OVERRIDE_BTN, INPUT_PULLUP);

  // Safe default: gate locked on boot until heartbeat received
  activateLock();
  Serial.println("Boot default: GATE LOCKED (waiting for pipeline heartbeat)");

  connectWiFi();

  mqttClient.setServer(MQTT_BROKER_IP, MQTT_BROKER_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setKeepAlive(30);
  mqttClient.setBufferSize(512);

  connectMQTT();

  lastHeartbeatMs = millis();  // Start watchdog clock after boot
}


// =============================================================================
// Main Loop
// =============================================================================
void loop() {
  // Maintain MQTT connection
  if (!mqttClient.connected()) {
    connectMQTT();
  }
  mqttClient.loop();

  // Safety checks
  checkWatchdog();
  checkLockoutTimer();
  handleOverrideButton();
  checkBuzzer();      // Non-blocking: stops the alarm tone after BUZZER_MS

  // Green LED = system healthy (IDLE state, connected)
  bool healthy = (currentState == IDLE) && mqttClient.connected();
  digitalWrite(LED_GREEN, healthy ? HIGH : LOW);

  delay(50);  // 20 Hz loop
}


// =============================================================================
// MQTT Callback — handles all incoming messages
// =============================================================================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String topicStr(topic);
  String payloadStr;
  for (unsigned int i = 0; i < length; i++) {
    payloadStr += (char)payload[i];
  }

  Serial.print("[MQTT] Topic: "); Serial.println(topicStr);
  Serial.print("[MQTT] Payload: "); Serial.println(payloadStr);

  // ── Attack Alert ──────────────────────────────────────────────────────────
  if (topicStr == TOPIC_ALERT) {
    StaticJsonDocument<256> doc;
    DeserializationError err = deserializeJson(doc, payloadStr);
    if (err) {
      Serial.println("[ERROR] JSON parse failed for attack_alert");
      return;
    }

    JsonObject obj = doc.as<JsonObject>();
    if (!verifyPayloadHmac(obj)) {
      Serial.println("[ZERO-TRUST REJECT] attack_alert failed HMAC-SHA256 signature attestation!");
      return;
    }

    float confidence = doc["confidence"] | 0.0f;
    Serial.printf("[ATTACK] Attested Alert Confirmed! confidence=%.3f\n", confidence);
    activateLock();
    lockStartMs = millis();
    publishACK();
  }

  // ── Manual Override ───────────────────────────────────────────────────────
  else if (topicStr == TOPIC_OVERRIDE) {
    StaticJsonDocument<128> doc;
    DeserializationError err = deserializeJson(doc, payloadStr);
    if (err) return;

    JsonObject obj = doc.as<JsonObject>();
    if (!verifyPayloadHmac(obj)) {
      Serial.println("[ZERO-TRUST REJECT] override command failed HMAC-SHA256 signature attestation!");
      return;
    }

    String action = doc["action"] | "lock";
    if (action == "unlock") {
      Serial.println("[OVERRIDE] Unlocking gate");
      deactivateLock();
      currentState = OVERRIDE_UNLOCKED;
    } else if (action == "lock") {
      Serial.println("[OVERRIDE] Locking gate");
      activateLock();
    }
    publishStatus();
  }

  // ── Heartbeat ─────────────────────────────────────────────────────────────
  else if (topicStr == TOPIC_HEARTBEAT) {
    lastHeartbeatMs = millis();
    // If gate was auto-locked by watchdog and heartbeat resumes, unlock
    if (currentState == LOCKED && (millis() - lockStartMs > LOCKOUT_MS)) {
      deactivateLock();
    }
  }
}


// =============================================================================
// Gate Control (Fail-Secure Posture: De-Energized LOW = LOCKED)
// =============================================================================
void activateLock() {
  digitalWrite(RELAY_PIN, LOW);    // De-energized coil (LOW) → Gate physically LOCKED (Fail-Secure)
  currentState = LOCKED;
  lockStartMs = millis();
  startLedFlash();
  startBuzzer();                   // Non-blocking 880 Hz tone; stopped by checkBuzzer()
  Serial.println("[GATE] LOCKED — Relay de-energized LOW (Fail-Secure)");
}

void deactivateLock() {
  digitalWrite(RELAY_PIN, HIGH);   // Energized coil (HIGH) → Gate UNLOCKED (Energize-to-Release)
  currentState = IDLE;
  stopLedFlash();
  stopBuzzer();                    // Silence alarm when the gate is released
  digitalWrite(LED_RED, LOW);
  Serial.println("[GATE] UNLOCKED — Relay energized HIGH");
}


// =============================================================================
// Buzzer — non-blocking alarm tone (880 Hz PWM on channel 0)
// Started by activateLock(); stopped from loop() via checkBuzzer() so the MQTT
// callback returns immediately and ACK latency stays well under 200 ms.
// =============================================================================
void startBuzzer() {
  ledcSetup(0, 880, 8);            // Channel 0, 880 Hz, 8-bit resolution
  ledcAttachPin(BUZZER_PIN, 0);
  ledcWrite(0, 128);               // 50% duty cycle
  buzzerActive  = true;
  buzzerStartMs = millis();
}

void stopBuzzer() {
  if (!buzzerActive) return;
  ledcWrite(0, 0);                 // Stop buzzer
  ledcDetachPin(BUZZER_PIN);
  buzzerActive = false;
}

void checkBuzzer() {
  if (buzzerActive && (millis() - buzzerStartMs > BUZZER_MS)) {
    stopBuzzer();
  }
}


// =============================================================================
// LED Flash at 5 Hz
// =============================================================================
void toggleLed() {
  ledState = !ledState;
  digitalWrite(LED_RED, ledState ? HIGH : LOW);
}

void startLedFlash() {
  ledTicker.attach(0.1, toggleLed);  // 10 toggles/s = 5 Hz
}

void stopLedFlash() {
  ledTicker.detach();
  digitalWrite(LED_RED, LOW);
  ledState = false;
}


// =============================================================================
// MQTT Publishing
// =============================================================================
void publishACK() {
  StaticJsonDocument<128> doc;
  doc["gate_status"] = "LOCKED";
  doc["timestamp"]   = millis();
  char buf[128];
  serializeJson(doc, buf);
  mqttClient.publish(TOPIC_ACK, buf, false);
  Serial.println("[MQTT] ACK published: LOCKED");
}

void publishStatus() {
  StaticJsonDocument<128> doc;
  doc["gate"]     = (currentState == LOCKED) ? "LOCKED" : "UNLOCKED";
  doc["uptime_s"] = millis() / 1000;
  doc["state"]    = currentState;
  char buf[128];
  serializeJson(doc, buf);
  mqttClient.publish(TOPIC_STATUS, buf, false);
}


// =============================================================================
// Watchdog — fail-secure auto-lock
// =============================================================================
void checkWatchdog() {
  if (millis() - lastHeartbeatMs > WATCHDOG_MS) {
    if (currentState != LOCKED) {
      Serial.println("[WATCHDOG] Heartbeat timeout! Auto-locking gate.");
      activateLock();
    }
  }
}

void checkLockoutTimer() {
  // Auto-release lock after 30 seconds if no further alerts
  if (currentState == LOCKED && (millis() - lockStartMs > LOCKOUT_MS)) {
    // Only auto-release if heartbeat is live (PC is online)
    if (millis() - lastHeartbeatMs < WATCHDOG_MS) {
      Serial.println("[LOCKOUT] 30s expired + PC online — auto-unlocking");
      deactivateLock();
      publishStatus();
    }
  }
}


// =============================================================================
// Manual Override Button (GPIO 33 INPUT_PULLUP)
// Debounced: short press (<1s) = toggle gate
// =============================================================================
bool lastBtnState = HIGH;
unsigned long btnPressMs = 0;

void handleOverrideButton() {
  bool btn = digitalRead(OVERRIDE_BTN);
  if (btn == LOW && lastBtnState == HIGH) {
    btnPressMs = millis();
  }
  if (btn == HIGH && lastBtnState == LOW) {
    unsigned long held = millis() - btnPressMs;
    if (held > 50 && held < 3000) {  // Debounce 50ms, max 3s
      if (currentState == LOCKED) {
        Serial.println("[BTN] Manual unlock");
        deactivateLock();
      } else {
        Serial.println("[BTN] Manual lock");
        activateLock();
      }
      publishStatus();
    }
  }
  lastBtnState = btn;
}


// =============================================================================
// WiFi + MQTT Connection
// =============================================================================
void connectWiFi() {
  Serial.printf("Connecting to WiFi: %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi failed! Check SSID/password. Continuing offline...");
  }
}

void connectMQTT() {
  int attempts = 0;
  while (!mqttClient.connected() && attempts < 5) {
    Serial.printf("MQTT connecting to %s:%d ... ", MQTT_BROKER_IP, MQTT_BROKER_PORT);
    if (mqttClient.connect(MQTT_CLIENT_ID)) {
      Serial.println("connected");
      // Subscribe to relevant topics
      mqttClient.subscribe(TOPIC_ALERT,    1);  // QoS 1
      mqttClient.subscribe(TOPIC_OVERRIDE, 1);  // QoS 1
      mqttClient.subscribe(TOPIC_HEARTBEAT, 0); // QoS 0
      Serial.println("Subscribed to: attack_alert, override, heartbeat");
      lastHeartbeatMs = millis();  // Reset watchdog on reconnect
    } else {
      Serial.printf("failed (rc=%d), retry in 3s\n", mqttClient.state());
      delay(3000);
      attempts++;
    }
  }
}
