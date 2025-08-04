Need to implement:
- a safety mechanism that detects if we use fallback values too much (defaults for temp, vibration, etc) and flags an error in all my test files

Information flow:
Physics Models (bearing degradation, raw physics) → Subsystem Classes (eg. reaction wheel subsystem) → System Orchestrator (telemetry generator)

User Interface Layer
    ↓ (mission parameters)
Telemetry Generator (System Orchestrator)
    ↓ (operational commands & time steps)
Reaction Wheel Subsystem Class
    ↓ (operating conditions)
Bearing Degradation Model (Physics Layer)
    ↑ (bearing state & predictions)
    ↑ (performance impacts)
    ↑ (telemetry data streams)
User Interface Layer (Data Portal)


High level purpose of different classes:
The bearing degradation model knows physics, but doesn't know about operational modes
The reaction wheel subsystem class knows both physics AND operational context
The telemetry generator coordinates but doesn't know wheel specifics


Example Information Flow:
Bearing Model ↔ Reaction Wheel:
- Bearing provides: "Friction coefficient increased to 0.035"
- RW interprets: "This means 15% higher motor current at 1000 RPM"

Reaction Wheel ↔ Sensor Processor:
- RW provides: "True vibration level is 0.005 g RMS"
- Sensor converts: "Accelerometer reading: 0.0048 g ± 0.0003 g noise"

Sensor Processor ↔ Telemetry Generator:
- Sensor provides: "Temperature reading: 23.7°C with 0.1°C bias"
- Telemetry packages: "RW_TEMP: 23.7°C at timestamp 14:35:22"

NOTE ABOUT TEMP:
Is the Temperature in reaction wheel class the Same as in the Bearing Degradation Model?
No—they represent two different (but related) physical things:
In Bearing Degradation Model:
- Temperature is the internal bearing/grease temperature—a hidden/internal state used by physics to predict wear, lubrication breakdown, etc.
- This is often idealized or modeled "under the hood" in the bearing model.

In Reaction Wheel class (_physics_to_temperature):
- Temperature is the external wheel housing temperature—what would actually be measured by a thermistor or telemetered to ground.
- This is the simulated sensor readout, usually (but not always) correlated with the internal bearing temperature.
- In real spacecraft, housing temp is often lower than true bearing temp (and may lag physically).





Core DS&A Elements to Include:
- Time-series windowing using sliding window technique (fundamental for telemetry) ------ USEFUL
Use: "I used a sliding window to detect sensor trends for predictive maintenance"

- Priority queues/heaps for alert ranking and maintenance scheduling -- USEFUL
Use: "In spacecrafttelemetrygenerator, use a minheap to proritize maintenance alerts by urgency. 

- Hash maps for efficient sensor data lookup and feature caching ---- USEFUL
Use: "I cached expensive degradation calculations to avoid redundant computation in bearingdegradation model. USes memoization."

- Simple graph structures to model component dependencies
- Basic dynamic programming for remaining useful life calculations



Understanding the Problem Domain
A spacecraft has many components, each with continous telemetry streams:
- Reaction wheels (for orientation control)
- Battery systems (power management)
- Thermal control systems (temperature regulation)
- Communication systems (data transmission)


Phase 1: Data Generation and Understanding (Day 1-2)
Instead of just finding a dataset, we'll create synthetic spacecraft telemetry:

Smart Synthetic Data Generation Strategy:
- Create data generator class
- The DS&A element here: Use hash maps to efficiently store and lookup sensor configurations, and sliding windows to generate correlated time-series data.

The data should include:
- Temperature sensors: Gradual drift indicates thermal system issues
- Vibration sensors: Spikes suggest mechanical problems
- Power draw: Changes indicate electrical system health
- Communication signal strength: Degradation shows antenna/transmitter issues

Key Learning Questions to Answer:
- How do healthy systems behave over time?
- What patterns indicate gradual degradation vs. sudden failure?
- How do different failure modes manifest in the data?