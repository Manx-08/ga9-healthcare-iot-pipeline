// ============================================================
// GA9 Healthcare IoT Monitoring System
// Stage 2: On-Device TinyML Classification
// Hardware: Seeed XIAO nRF52840 Sense
// Model:    Edge Impulse exported Arduino library
//
// This sketch reads vital signs from three sensors and runs
// the exported TinyML model to classify each 5-minute window
// as Normal, Watch, or Alert — entirely on-device, with no
// cloud connectivity required.
//
// HOW TO USE:
// 1. Install your Edge Impulse library via Sketch -> Include
//    Library -> Add .ZIP Library (the file you downloaded)
// 2. Replace "ei-healthcare-iot-monitor-arduino_inferencing.h"
//    below with the exact header filename from your export
//    (check the src/ folder inside the zip for the .h file name)
// 3. Upload to XIAO nRF52840 via USB-C
// ============================================================

// ── Replace this with your exact exported library header name ──
#include <ei-healthcare-iot-monitor-arduino_inferencing.h>

// ── Sensor libraries ──
#include <Wire.h>
#include <SparkFun_MAX3010x_Sensor_Algorithm.h>  // MAX30102 HR + SpO2
#include <SparkFun_MAX3010x_Pulse_Oximetry.h>
#include <Adafruit_MLX90614.h>                   // MLX90614ESF temperature

// ── Pin definitions ──
const int ECG_PIN       = A0;   // AD8232 OUTPUT
const int LO_PLUS_PIN   = D2;   // AD8232 LO+ leads-off detect
const int LO_MINUS_PIN  = D3;   // AD8232 LO- leads-off detect
const int LED_GREEN     = D4;   // Green LED  -> Normal state
const int LED_RED       = D5;   // Red LED    -> Alert state
const int LED_YELLOW    = D6;   // Yellow LED -> Watch state
const int VIBRATION_PIN = D7;   // Vibration motor via transistor

// ── Timing ──
// One reading every 30 seconds, 10 readings per inference window
// This matches the window size used during Edge Impulse training
const unsigned long SAMPLE_INTERVAL_MS = 30000UL;  // 30 seconds
const int           WINDOW_SIZE        = 10;        // readings per window

// ── Sensor objects ──
MAX30105          particleSensor;
Adafruit_MLX90614 mlx;

// ── Circular buffer for inference window ──
// Stores the last WINDOW_SIZE readings before running inference
float hrBuffer[WINDOW_SIZE]   = {0};
float spo2Buffer[WINDOW_SIZE] = {0};
float tempBuffer[WINDOW_SIZE] = {0};
int   bufferIndex             = 0;
bool  bufferFull              = false;

// ── Classification result tracking ──
String lastClassification = "normal";
unsigned long lastSampleTime = 0;

// ── Simulated sensor fallback ─────────────────────────────────
// If sensors are not yet wired, set SIMULATE = true to test
// the inference pipeline with realistic fake readings.
// Set to false once hardware is assembled.
const bool SIMULATE = true;

float simulateHR() {
  // Returns a realistic resting HR with occasional spike for testing
  static int callCount = 0;
  callCount++;
  if (callCount > 70 && callCount < 80) return 108.0 + random(-5, 5);
  return 72.0 + sin(callCount * 0.3) * 4 + random(-2, 2);
}

float simulateSpO2() {
  static int callCount = 0;
  callCount++;
  if (callCount > 100 && callCount < 115) return 91.0 + random(-1, 1);
  return 97.5 + random(-1, 1) * 0.5;
}

float simulateTemp() {
  return 36.6 + random(-1, 1) * 0.1;
}

// ─────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  // Non-blocking Serial wait (max 3 seconds)
  unsigned long startWait = millis();
  while (!Serial && (millis() - startWait < 3000)) { delay(10); }

  Serial.println("GA9 Healthcare IoT Monitor — Stage 2 TinyML");
  Serial.println("Model: " EI_CLASSIFIER_PROJECT_NAME);
  Serial.print("Window size: ");
  Serial.print(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  Serial.println(" features");

  // ── Output pins ──
  pinMode(LED_GREEN,     OUTPUT);
  pinMode(LED_RED,       OUTPUT);
  pinMode(LED_YELLOW,    OUTPUT);
  pinMode(VIBRATION_PIN, OUTPUT);
  pinMode(LO_PLUS_PIN,   INPUT);
  pinMode(LO_MINUS_PIN,  INPUT);

  // Startup signal: all LEDs on briefly
  digitalWrite(LED_GREEN,  HIGH);
  digitalWrite(LED_YELLOW, HIGH);
  digitalWrite(LED_RED,    HIGH);
  delay(500);
  digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED,    LOW);

  if (!SIMULATE) {
    // ── Initialise MAX30102 ──
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
      Serial.println("ERROR: MAX30102 not found. Check wiring.");
      while (true) { delay(1000); }
    }
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);

    // ── Initialise MLX90614ESF ──
    if (!mlx.begin()) {
      Serial.println("ERROR: MLX90614ESF not found. Check wiring.");
      while (true) { delay(1000); }
    }
    Serial.println("Sensors initialised.");
  } else {
    Serial.println("SIMULATION MODE — no hardware required.");
  }

  Serial.println("Collecting readings every 30 seconds.");
  Serial.println("Inference runs after every 10 readings (5-minute window).");
  Serial.println("─────────────────────────────────────────────────────");
}

// ── Read one set of vital sign values ────────────────────────

struct VitalReading {
  float hr;
  float spo2;
  float temp;
  bool  valid;
};

VitalReading readSensors() {
  if (SIMULATE) {
    return { simulateHR(), simulateSpO2(), simulateTemp(), true };
  }

  // MAX30102: collect 100 samples for SpO2 + HR calculation
  const byte RATE_SIZE = 4;
  byte  rates[RATE_SIZE];
  byte  rateSpot = 0;
  long  lastBeat = 0;
  float beatsPerMinute;
  int   beatAvg = 0;

  long  irBuffer[100], redBuffer[100];
  int32_t spo2Value;
  int8_t  validSPO2;
  int32_t heartRate;
  int8_t  validHeartRate;

  for (byte i = 0; i < 100; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }

  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, 100, redBuffer,
    &spo2Value, &validSPO2,
    &heartRate, &validHeartRate
  );

  float tempC = mlx.readObjectTempC();

  if (!validHeartRate || !validSPO2 || heartRate <= 0 || spo2Value <= 0) {
    return { 0, 0, 0, false };
  }

  return { (float)heartRate, (float)spo2Value, tempC, true };
}

// ── Run Edge Impulse inference on the current window ─────────

String runInference() {
  // Build the feature array that Edge Impulse expects
  // Format matches what we trained on: 10 windows x 3 features = 30 values
  // Order: hr_0, spo2_0, temp_0, hr_1, spo2_1, temp_1, ...
  float features[WINDOW_SIZE * 3];

  for (int i = 0; i < WINDOW_SIZE; i++) {
    int idx = (bufferIndex + i) % WINDOW_SIZE;
    features[i * 3 + 0] = hrBuffer[idx];
    features[i * 3 + 1] = spo2Buffer[idx];
    features[i * 3 + 2] = tempBuffer[idx];
  }

  // Wrap features in Edge Impulse signal structure
  signal_t signal;
  int err = numpy::signal_from_buffer(features, WINDOW_SIZE * 3, &signal);
  if (err != 0) {
    Serial.print("ERROR: signal_from_buffer failed: ");
    Serial.println(err);
    return "error";
  }

  // Run classifier
  ei_impulse_result_t result = { 0 };
  err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.print("ERROR: run_classifier failed: ");
    Serial.println(err);
    return "error";
  }

  // Print all class probabilities for debugging
  Serial.println("\n── Inference result ──");
  float maxScore    = 0;
  String bestLabel  = "normal";

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    Serial.print("  ");
    Serial.print(result.classification[ix].label);
    Serial.print(": ");
    Serial.println(result.classification[ix].value, 4);

    if (result.classification[ix].value > maxScore) {
      maxScore = result.classification[ix].value;
      bestLabel = String(result.classification[ix].label);
    }
  }

  // Map Edge Impulse label to three-tier classification
  // "anomaly" with high confidence -> "alert"
  // "anomaly" with lower confidence -> "watch"
  if (bestLabel == "anomaly") {
    return (maxScore >= 0.75) ? "alert" : "watch";
  }
  return "normal";
}

// ── Actuate LEDs and vibration based on classification ───────

void actuate(String classification) {
  digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED,    LOW);
  digitalWrite(VIBRATION_PIN, LOW);

  if (classification == "normal") {
    digitalWrite(LED_GREEN, HIGH);
  } else if (classification == "watch") {
    digitalWrite(LED_YELLOW, HIGH);
  } else if (classification == "alert") {
    digitalWrite(LED_RED, HIGH);
    // Vibration pulse: 3 short bursts
    for (int i = 0; i < 3; i++) {
      digitalWrite(VIBRATION_PIN, HIGH);
      delay(200);
      digitalWrite(VIBRATION_PIN, LOW);
      delay(150);
    }
  }
}

// ── Main loop ────────────────────────────────────────────────

void loop() {
  unsigned long now = millis();

  // Wait for next sample interval (non-blocking)
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // Read sensors
  VitalReading reading = readSensors();

  if (!reading.valid) {
    Serial.println("WARNING: Invalid sensor reading — skipping this sample.");
    return;
  }

  // Print reading to Serial for monitoring
  Serial.print("[Reading ");
  Serial.print(bufferIndex + 1);
  Serial.print("/");
  Serial.print(WINDOW_SIZE);
  Serial.print("]  HR: ");
  Serial.print(reading.hr, 1);
  Serial.print(" bpm  SpO2: ");
  Serial.print(reading.spo2, 1);
  Serial.print("%  Temp: ");
  Serial.print(reading.temp, 2);
  Serial.println("°C");

  // Store in circular buffer
  hrBuffer[bufferIndex]   = reading.hr;
  spo2Buffer[bufferIndex] = reading.spo2;
  tempBuffer[bufferIndex] = reading.temp;
  bufferIndex = (bufferIndex + 1) % WINDOW_SIZE;

  if (bufferIndex == 0) bufferFull = true;

  // Run inference once we have a full window
  if (!bufferFull) {
    Serial.print("Collecting baseline window... (");
    Serial.print(WINDOW_SIZE - bufferIndex);
    Serial.println(" readings remaining)");
    return;
  }

  // Run TinyML inference
  String classification = runInference();
  lastClassification = classification;

  Serial.print("── Classification: ");
  Serial.print(classification);
  Serial.println(" ──");

  // Actuate physical outputs
  actuate(classification);
}
