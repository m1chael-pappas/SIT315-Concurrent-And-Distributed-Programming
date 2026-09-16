# SIT315 M1QP - Room Security Monitor (4/4 QP)

An interrupt-driven sense-think-act system built for Module 1 of SIT315. The system is a simple room security monitor that combines external interrupts, pin change interrupts and a timer interrupt on an Arduino Uno.

## How it works

- **D8 button** arms or disarms the system (pin change interrupt, PCINT0)
- **D9 button** acts as a door sensor, each press toggles open/closed (pin change interrupt, PCINT1)
- **D2 PIR sensor** detects motion (external interrupt via attachInterrupt)
- **D13 LED** is the alarm output
- **D12 LED** is a heartbeat that toggles every second from a Timer1 interrupt; a status line prints to Serial every 5 seconds

The alarm only triggers when the system is armed AND either the door opens or motion is detected. Disarming clears the alarm.

## Interrupt design

| Source | Type | Vector | What the ISR does |
|---|---|---|---|
| D2 motion | External (INT0) | `motionISR` | sets `motionFlag` |
| D8 + D9 | Pin change (PORTB) | `PCINT0_vect` | snapshots `PINB`, sets `pciFlag` |
| Timer1 CTC @ 1 Hz | Timer | `TIMER1_COMPA_vect` | sets `timerTick` |

All ISRs only set volatile flags. The main loop handles debouncing (50 ms via millis), figures out which PORTB pin changed using an XOR against the last known state, runs the decision logic and drives the LEDs. Shared data is copied with interrupts briefly disabled so reads are atomic. A tick counter in the loop derives the 5-second status rate from the 1 Hz timer.

Timer1 runs in CTC mode with a 1024 prescaler and `OCR1A = 15624`, giving exactly one interrupt per second (16 MHz / 1024 / 15625).

## Wiring

- D2 -> PIR signal pin (or pushbutton to 5V with a pull-down resistor in TinkerCad)
- D8 -> pushbutton to GND (internal pull-up enabled, pressed = LOW)
- D9 -> pushbutton to GND (internal pull-up enabled, pressed = LOW)
- D12 -> LED + 220R resistor -> GND
- D13 -> LED + 220R resistor -> GND

See `circuit-diagram.png` for the full schematic.

## Running it

1. Open `taskM1.cpp` in the Arduino IDE (or paste into a TinkerCad Arduino sketch)
2. Upload to an Uno (or start the TinkerCad simulation)
3. Open the Serial Monitor at 9600 baud
4. Press D8 to arm, then press D9 or trigger the PIR and watch the alarm fire while the heartbeat keeps ticking

## Files

- `taskM1.cpp` - full source
- `circuit-diagram.png` - wiring schematic