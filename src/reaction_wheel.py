"""
ReactionWheelSubsystem
----------------------
Simulates a spacecraft reaction wheel unit, orchestrating bearing degradation physics,
converting internal states into subsystem-level effects (current, vibration, temperature),
and producing signals ready for telemetry or diagnostics.

Integrates user commands, mission profiles, and bearing physics for E2E RW modeling.
"""

from src.bearing_degradation import BearingDegradationModel, BearingState


class ReactionWheelSubsystem:

    # Vibration conversion constants (literature-based)
    BASE_VIBRATION = 0.01  # g, minimum healthy wheel baseline
    VIBRATION_WEAR_GAIN = 0.04  # g per unit wear
    VIBRATION_ROUGHNESS_GAIN = 0.07  # g per μm roughness above baseline  
    VIBRATION_FRICTION_GAIN = 0.1  # g per unit friction above 0.02
    ROUGHNESS_BASELINE = 0.32  # μm, new bearing baseline
    FRICTION_BASELINE = 0.02  # baseline friction for vibration calculation
    
    # Current draw constants (literature-based)
    IDLE_CURRENT = 0.08  # A, low-load (0.05–0.12A typical for cubesat class)
    CURRENT_GAIN = 0.3  # A per unit friction * speed/1000
    
    # Temperature constants (literature-based)
    AMBIENT_TEMP = 20.0  # deg C, typical spacecraft housing
    TEMP_FRICTION_GAIN = 3.0  # deg C per friction*load_factor
    
    # Performance metrics constants (literature-based)
    MAX_TORQUE_NM = 0.05  # Newton meter, de-rates as wear grows
    POINTING_JITTER_BASE_ARCSEC = 0.1  # arcsec, best-case pointing stability
    POINTING_JITTER_VIBRATION_GAIN = 20  # arcsec per 0.01g increase in vibration


    def __init__(self, 
                wheel_id=0, # Since most real spacecraft have 3–4 reaction wheels (for full 3-axis + redundancy control)
                operational_mode='IDLE', # "IDLE" is a common and safe default state in spacecraft systems. UI allows users to select or change operational modes (such as 'IDLE', 'NOMINAL', 'SAFE', 'HIGH_TORQUE', etc.)
                load_factor=1.0): # Considered an initialization parameter since load factor can be thought of as a mission property
        self.bearing_model = BearingDegradationModel()
        self.bearing_state = BearingState()
        self.operational_mode = operational_mode
        self.load_factor = load_factor
        self.wheel_id = wheel_id

        # Telemetry storage
        self.latest_telemetry = {}

    def update(self, mission_time_hours, commands):
        """
        Main orchestration function.
        - Decodes commands to operating conditions
        - Updates bearing state via physics model
        - Calculates observable effects (current, vibration, temperature)
        - Packages telemetry dictionary

        Inputs:
            mission_time_hours: current mission time (float)
            commands: dict, e.g. {'target_speed_rpm': 3600, 'load_factor': 1.2, 'mode': 'NOMINAL'}
        """
        speed_rpm = commands.get('target_speed_rpm', 0.0)
        load_factor = commands.get('load_factor', self.load_factor) # Incase the user wants to override the mission load factor, they have the oppurtunity to. 
        bearing_temp = commands.get('bearing_temperature', 20.0)
        dt = commands.get('timestep_hours', 1.0)

        op_conditions = {
            'speed_rpm': speed_rpm,
            'bearing_temperature': bearing_temp,
            'load_factor': load_factor,
        }
        
        # Update bearing state
        self.bearing_state = self.bearing_model.update_bearing_state(self.bearing_state, op_conditions, dt)
        physics = self.bearing_model.get_physical_properties(self.bearing_state)

        # Translate bearing state to subsystem signals
        vibration = self._physics_to_vibration(physics)
        current = self._physics_to_current(physics, speed_rpm)
        housing_temp = self._physics_to_temperature(physics, load_factor)

    
        # Store for export/streaming
        self.latest_telemetry = {
            'mission_time_hours': mission_time_hours,
            'mode': commands.get('mode', self.operational_mode),
            'wheel_id': self.wheel_id,
            'speed_rpm': speed_rpm,
            'load_factor': load_factor,
            'bearing_wear_level': physics['wear_level'],
            'bearing_friction_coeff': physics['friction_coefficient'],
            'bearing_surface_roughness': physics['surface_roughness'],
            'bearing_lubrication_quality': physics['lubrication_quality'],
            'measured_vibration': vibration,
            'motor_current': current,
            'housing_temperature': housing_temp,
        }


    def _physics_to_vibration(self, bearing_physics):
        """
        Converts physics state to vibration RMS (arbitrary units for demo).
        Vibration increases as wear, roughness, and friction rise.

        Literature: Vibration from reaction wheels is commonly characterized by RMS amplitude in g or m/s², typically on the order of 0.01–0.1g for healthy small wheels. 
        Vibration increases nonlinearly as bearing wear or roughness grows, with literature describing increases of 2–10 times in failing bearings.
        """
        vib = (self.BASE_VIBRATION
               + self.VIBRATION_WEAR_GAIN * bearing_physics['wear_level']
               + self.VIBRATION_ROUGHNESS_GAIN * max(0, bearing_physics['surface_roughness'] - self.ROUGHNESS_BASELINE)
               + self.VIBRATION_FRICTION_GAIN * (bearing_physics['friction_coefficient'] - self.FRICTION_BASELINE))
        return vib


    def _physics_to_current(self, bearing_physics, wheel_speed):
        """
        Estimates motor current draw (A).
        Increases with friction and wheel speed.

        Literature: Idle (zero friction) reaction wheel current for compact wheels is typically around 0.05–0.12A. 
        Current rises by ~0.2–0.5A per 0.1 increase in friction coefficient and climbs rapidly at high RPM. 
        Scaling by friction and RPM is supported by hardware models and NASA jitter reports.
        """
        load = self.CURRENT_GAIN * bearing_physics['friction_coefficient'] * abs(wheel_speed) / 1000.0
        return self.IDLE_CURRENT + load


    def _physics_to_temperature(self, bearing_physics, operational_load):
        """
        Approximates wheel housing temperature (deg C).
        Increases with operating load and friction.

        Literature: Housing temperature for wheels rarely exceeds ambient by more than a few °C under normal operations, 
        but with increased friction and load, rises of +2–5°C or more are plausible, especially in vacuum or poor conduction cases.

        """
        delta = self.TEMP_FRICTION_GAIN * bearing_physics['friction_coefficient'] * operational_load
        return self.AMBIENT_TEMP + delta

    def get_performance_metrics(self):
        """
        Returns a snapshot of metrics indicating RW health/performance.

        Torque: Maximum torque in Nm for a small wheel is typically around 0.03–0.08Nm, and drops proportionally as wear increases and friction rises.
        Point Jitter: Pointing stability can typically be maintained <1 arcsec for high-quality systems, but can degrade by up to 10–50 arcsec in the presence of high vibration or degraded wheels.
        """
        metrics = {
            'max_torque_Nm': max(0.0, self.MAX_TORQUE_NM * (1 - self.bearing_state.wear_level)),  # Declines with wear
            'pointing_jitter_arcsec': self.POINTING_JITTER_BASE_ARCSEC + self.POINTING_JITTER_VIBRATION_GAIN * self.latest_telemetry.get('measured_vibration', 0.01)  # Vibration causes pointing error
        }
        return metrics
     
    def predict_maintenance_timeline(self, current_state, mission_profile):
        """
        Projects maintenance/failure horizon (hours) using internal physics.
        mission_profile: dict with typical profile for coming operation (temp, load_factor)

        Returns: estimated_hours_to_failure
        """
        pred = self.bearing_model.predict_wear_progression(self.bearing_state, mission_profile, time_horizon_hours=5000.0)
        return {'hours_to_maintenance': pred['time_to_failure_hours']}

    def get_telemetry(self):
        """Return the latest telemetry dictionary"""
        return self.latest_telemetry