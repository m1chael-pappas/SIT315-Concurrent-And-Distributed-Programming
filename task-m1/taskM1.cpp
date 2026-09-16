/*
 * SIT315 M1QP - Interrupt-driven sense-think-act system (QP4)
 *
 * Room security monitor. Three interrupt sources feed a non-blocking
 * superloop via volatile flags:
 *
 *   INT0  (D2)         PIR motion, RISING edge, attachInterrupt()
 *   PCINT0 vector      arm button (D8/PB0) and door sensor (D9/PB1)
 *   TIMER1_COMPA       1 Hz periodic tick, CTC mode
 *
 * Concurrency model: ISRs capture state and set flags only. All decision
 * logic, debouncing and I/O runs in loop() context. Shared ISR/loop data
 * is volatile and copied under cli/sei to guarantee atomic reads.
 *
 * Alarm condition: armed && (motion || doorOpen). Disarm clears the alarm.
 *
 * NOTE: remove the Arduino.h include below when pasting into TinkerCad;
 * its build wrapper generates conflicting C-linkage prototypes with it.
 */

#include <Arduino.h>

/* Pin map. D8/D9 chosen deliberately: both on PORTB so a single PCI
 * vector services them. */
const uint8_t PIN_MOTION    = 2;   // INT0
const uint8_t PIN_ARM_BTN   = 8;   // PB0 / PCINT0
const uint8_t PIN_DOOR      = 9;   // PB1 / PCINT1
const uint8_t PIN_HEARTBEAT = 12;
const uint8_t PIN_ALARM     = 13;

/* ISR -> loop handoff. volatile: accessed from both interrupt and
 * mainline context. */
volatile bool motionFlag = false;
volatile bool pciFlag = false;
volatile uint8_t portBSnapshot = 0;  // PINB latched at interrupt time
volatile bool timerTick = false;

/* Mainline-only state. Not volatile by design: never touched by ISRs. */
bool armed = false;
bool alarmActive = false;
bool doorOpen = false;
uint8_t lastPortB = 0;               // reference for edge/change detection
unsigned long lastArmChange = 0;
unsigned long lastDoorChange = 0;
const unsigned long DEBOUNCE_MS = 50;
unsigned long uptimeSeconds = 0;
const uint8_t STATUS_PERIOD_S = 5;   // status output decimated from 1 Hz tick
uint8_t statusCounter = 0;

/* --------------------------------------------------------------------
 * ISRs. Contract: O(1), no delay(), no Serial, no shared-state mutation
 * beyond the designated flags. Deferred processing happens in loop().
 * ------------------------------------------------------------------ */

void motionISR() {
  motionFlag = true;
}

/* PCI is per-port, not per-pin: the vector fires on any unmasked PORTB
 * change without identifying the source. Latch PINB here; loop() derives
 * the changed pin(s) by XOR against the previous known state. */
ISR(PCINT0_vect) {
  portBSnapshot = PINB;
  pciFlag = true;
}

ISR(TIMER1_COMPA_vect) {
  timerTick = true;
}

/* -------------------------------------------------------------------- */

void setupPinChangeInterrupts() {
  PCICR |= (1 << PCIE0);                       // enable PORTB PCI group
  PCMSK0 |= (1 << PCINT0) | (1 << PCINT1);     // unmask D8, D9 only
}

/* Timer1, CTC, /1024 prescaler: 16 MHz / 1024 = 15625 Hz timebase.
 * TOP = 15624 -> compare match at exactly 1 Hz. Registers are configured
 * with interrupts masked to avoid a spurious match mid-setup. */
void setupTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A = 15624;
  TCCR1B |= (1 << WGM12);                // CTC, TOP = OCR1A
  TCCR1B |= (1 << CS12) | (1 << CS10);   // clk/1024
  TIMSK1 |= (1 << OCIE1A);
  interrupts();
}

void setup() {
  Serial.begin(9600);

  pinMode(PIN_MOTION, INPUT);            // PIR drives the line, active-high
  pinMode(PIN_ARM_BTN, INPUT_PULLUP);    // active-low, switched to GND
  pinMode(PIN_DOOR, INPUT_PULLUP);
  pinMode(PIN_HEARTBEAT, OUTPUT);
  pinMode(PIN_ALARM, OUTPUT);

  lastPortB = PINB;                      // baseline before PCI enable

  attachInterrupt(digitalPinToInterrupt(PIN_MOTION), motionISR, RISING);
  setupPinChangeInterrupts();
  setupTimer1();

  Serial.println(F("=== Room Security Monitor ==="));
  Serial.println(F("D8: arm/disarm | D9: door | D2: motion"));
  Serial.println(F("System starts DISARMED"));
}

/* -------------------------------------------------------------------- */

void evaluateAlarm(bool motionEvent) {
  if (armed && (motionEvent || doorOpen)) {
    if (!alarmActive) {
      alarmActive = true;
      Serial.println(F(">>> ALARM TRIGGERED <<<"));
    }
  }
}

/* Deferred PCI handler. Resolves which PORTB pin(s) changed, debounces,
 * and updates system state. Runs in mainline context so it is free to
 * use millis() and Serial. */
void processPinChange() {
  uint8_t snapshot;

  /* Atomic copy: PCINT0_vect may rewrite portBSnapshot between the two
   * statements otherwise, tearing the read/clear pair. */
  noInterrupts();
  snapshot = portBSnapshot;
  pciFlag = false;
  interrupts();

  uint8_t changed = snapshot ^ lastPortB;
  unsigned long now = millis();

  /* Arm/disarm: falling edge only (active-low). Debounce window rejects
   * contact bounce retriggering the vector. */
  if (changed & (1 << PB0)) {
    bool pressed = !(snapshot & (1 << PB0));
    if (pressed && (now - lastArmChange > DEBOUNCE_MS)) {
      lastArmChange = now;
      armed = !armed;
      if (!armed) {
        alarmActive = false;
      }
      Serial.print(F("[EVENT] System "));
      Serial.println(armed ? F("ARMED") : F("DISARMED"));
    }
  }

  /* Door sensor: press toggles latched open/closed state rather than
   * tracking the momentary line level. */
  if (changed & (1 << PB1)) {
    bool pressed = !(snapshot & (1 << PB1));
    if (pressed && (now - lastDoorChange > DEBOUNCE_MS)) {
      lastDoorChange = now;
      doorOpen = !doorOpen;
      Serial.print(F("[EVENT] Door "));
      Serial.println(doorOpen ? F("OPENED") : F("CLOSED"));
      evaluateAlarm(false);
    }
  }

  lastPortB = snapshot;
}

/* 1 Hz periodic task. Heartbeat toggles every tick; status output is
 * decimated to every STATUS_PERIOD_S ticks. Reads system state only,
 * never mutates it - keeps the time domain decoupled from events. */
void processTimerTick() {
  timerTick = false;
  uptimeSeconds++;

  digitalWrite(PIN_HEARTBEAT, !digitalRead(PIN_HEARTBEAT));

  statusCounter++;
  if (statusCounter < STATUS_PERIOD_S) {
    return;
  }
  statusCounter = 0;

  Serial.print(F("[STATUS t="));
  Serial.print(uptimeSeconds);
  Serial.print(F("s] armed="));
  Serial.print(armed ? F("YES") : F("no"));
  Serial.print(F(" door="));
  Serial.print(doorOpen ? F("OPEN") : F("closed"));
  Serial.print(F(" alarm="));
  Serial.println(alarmActive ? F("ACTIVE") : F("off"));
}

/* Non-blocking superloop: drain pending flags, then reconcile outputs
 * with current state. Worst-case iteration is bounded; no busy-waits. */
void loop() {
  if (pciFlag) {
    processPinChange();
  }

  if (motionFlag) {
    motionFlag = false;
    Serial.println(F("[EVENT] Motion detected"));
    evaluateAlarm(true);
  }

  if (timerTick) {
    processTimerTick();
  }

  digitalWrite(PIN_ALARM, alarmActive ? HIGH : LOW);
}