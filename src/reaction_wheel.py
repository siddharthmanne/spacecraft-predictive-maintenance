"""
ReactionWheelSubsystem
----------------------
Simulates a spacecraft reaction wheel unit, orchestrating bearing degradation physics,
converting internal states into subsystem-level effects (current, vibration, temperature),
and producing signals ready for telemetry or diagnostics.

Integrates user commands, mission profiles, and bearing physics for E2E RW modeling.
"""

import random
import numpy as np
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
    CURRENT_SPIKE_PROBABILITY = 0.05  # 5% chance of current spike during rapid friction changes
    CURRENT_SPIKE_MAGNITUDE = 2.5  # Spike multiplier for worst-case 200-500% draws
    
    # Temperature constants (literature-based)
    HOUSING_TEMPERATURE_BASE = 21.0  # deg C, typical spacecraft housing (updated from 20.0)
    TEMP_FRICTION_GAIN = 4.0  # deg C per friction*load_factor (increased for realistic heat generation)
    TEMP_HEAT_GENERATION_FACTOR = 3.5  # Additional temperature rise from friction heat
    
    # Performance metrics constants (literature-based)
    MAX_TORQUE_NM = 0.05  # Newton meter, de-rates as wear grows
    POINTING_JITTER_BASE_ARCSEC = 0.1  # arcsec, best-case pointing stability
    POINTING_JITTER_VIBRATION_GAIN = 20  # arcsec per 0.01g increase in vibration


    def __init__(self, 
                wheel_id=0, # Since most real spacecraft have 3–4 reaction wheels (for full 3-axis + redundancy control)
                operational_mode='IDLE', # "IDLE" is a common and safe default state in spacecraft systems. UI allows users to select or change operational modes (such as 'IDLE', 'NOMINAL', 'SAFE', 'HIGH_TORQUE', etc.)
                spacecraft_payload=0.0,
                maneuvering=False): 
        self.bearing_model = BearingDegradationModel()
        self.bearing_state = BearingState()
        self.operational_mode = operational_mode
        self.spacecraft_payload = spacecraft_payload
        self.wheel_id = wheel_id

        # Telemetry storage
        self.latest_telemetry = {}

    def update_one_hour(self, commands):
        """
        Main orchestration function.
        - Decodes commands to operating conditions
        - Updates bearing state via physics model
        - Calculates observable effects (current, vibration, temperature)
        - Packages telemetry dictionary

        Inputs:
            commands: dict, e.g. {'target_speed_rpm': 3600, 'spacecraft_payload': 1200, 'mode': 'NOMINAL'}
        """
        speed_rpm = commands.get('target_speed_rpm', 0.0)
        spacecraft_payload = commands.get('spacecraft_payload', self.spacecraft_payload) # Incase the user wants to override the mission payload, they have the oppurtunity to. 
        maneuvering = commands.get('maneuvering', False)

        op_conditions = {
            'speed_rpm': speed_rpm,
            'spacecraft_payload': spacecraft_payload,
            'maneuvering': maneuvering
        }
        
        # Update bearing state
        self.bearing_state = self.bearing_model.update_bearing_state_one_hour(self.bearing_state, op_conditions)

        # Translate bearing state to subsystem signals
        vibration = self._physics_to_vibration()
        current = self._physics_to_current(speed_rpm)
        housing_temperature = self._physics_to_housing_temperature(op_conditions, current)

    
        # Store for export/streaming
        self.latest_telemetry = {
            # Intrinsic physical values
            'mode': commands.get('mode', self.operational_mode),
            'maneuvering': maneuvering,
            'bearing_load_N': self.bearing_model.bearing_load_N,  # Newtons, calculated from RPM and payload
            'wheel_id': self.wheel_id,
            'speed_rpm': speed_rpm,
            'spacecraft_payload': spacecraft_payload,
            'bearing_wear_level': self.bearing_state.wear_level,
            'bearing_friction_coeff': self.bearing_state.friction_coefficient,
            'bearing_surface_roughness': self.bearing_state.surface_roughness,
            'bearing_lubrication_quality': self.bearing_state.lubrication_quality,
            'bearing_temperature': self.bearing_state.bearing_temperature,

            # Sensor values
            'vibration': vibration,
            'motor_current': current,
            'housing_temperature': housing_temperature
        }


    def _physics_to_vibration(self):
        """
        Converts physics state to vibration RMS (arbitrary units for demo).
        Vibration increases as wear, roughness, and friction rise.

        Literature: Vibration from reaction wheels is commonly characterized by RMS amplitude in g or m/s², typically on the order of 0.01–0.1g for healthy small wheels. 
        Vibration increases nonlinearly as bearing wear or roughness grows, with literature describing increases of 2–10 times in failing bearings.
        """
        vib = (self.BASE_VIBRATION
               + self.VIBRATION_WEAR_GAIN * self.bearing_state.wear_level
               + self.VIBRATION_ROUGHNESS_GAIN * max(0, self.bearing_state.surface_roughness - self.ROUGHNESS_BASELINE)
               + self.VIBRATION_FRICTION_GAIN * (self.bearing_state.friction_coefficient - self.FRICTION_BASELINE))
        return vib


    def _physics_to_current(self, wheel_speed):
        """
        Estimates motor current draw (A).
        Increases with friction and wheel speed.

        Literature: Idle (zero friction) reaction wheel current for compact wheels is typically around 0.05–0.12A. 
        Current rises by ~0.2–0.5A per 0.1 increase in friction coefficient and climbs rapidly at high RPM. 
        Research shows current can increase by 200-500% in severely degraded bearings.
        """
        
        # Base current calculation
        load = self.CURRENT_GAIN * self.bearing_state.friction_coefficient * abs(wheel_speed) / 1000.0
        base_current = self.IDLE_CURRENT + load
        
        # Stochastic current spikes during rapid friction surges
        # Simulates worst-case 200-500% draws seen in orbit
        if (self.bearing_state.friction_coefficient > 0.1 and  # High friction condition
            random.random() < self.CURRENT_SPIKE_PROBABILITY):
            spike_multiplier = 1.0 + (self.CURRENT_SPIKE_MAGNITUDE - 1.0) * random.random()
            return base_current * spike_multiplier
        
        return base_current


    def _physics_to_housing_temperature(self, operating_conditions, motor_current=None):
        """
        Approximates wheel housing temperature (deg C).
        Increases with operating load and friction.

        Literature: Housing temperature for wheels rarely exceeds ambient by more than a few °C under normal operations, 
        but with increased friction and load, rises of +2–5°C or more are plausible, especially in vacuum or poor conduction cases.
        Heat generation outweighs dissipation as friction grows, leading to net temperature rise.

        """
        # Heat source 1: Bearing friction
        surface_velocity = operating_conditions['speed_rpm'] * 2 * np.pi / 60 * self.bearing_model.physics_constants['bearing_radius']  # m/s
        bearing_friction_heat = self.bearing_state.friction_coefficient * self.bearing_state.bearing_load * surface_velocity
        
        #Heat source 2: Motor losses
        speed_factor = abs(operating_conditions['speed_rpm']) / 1000.0  # Scale speed for current draw
        mass_factor = operating_conditions.get('spacecraft_payload', 1.0) / 1000.0  # kg
        base_motor_heat = 0.5  # Watts

        maneuver_heat = base_motor_heat * speed_factor * mass_factor * 0.1  # Scale with speed and payload
        total_motor_heat = base_motor_heat + maneuver_heat

        total_heat_watts = bearing_friction_heat + total_motor_heat

        return (self.HOUSING_TEMPERATURE_BASE + total_heat_watts * 0.1)  # Convert watts to temperature rise, assuming 10% efficiency in heat dissipation

        # Base temperature rise from friction
        friction_delta = self.TEMP_FRICTION_GAIN * self.bearing_state.friction_coefficient * operational_load

        wear_heat = 2.0 * self.bearing_state.wear_level * operational_load  # Add wear-driven heat

        lubrication_factor = 1.5 if self.bearing_state.lubrication_quality < 0.3 else 1.0
        
        total_heat = (friction_delta + wear_heat) * lubrication_factor
        return self.HOUSING_TEMPERATURE_BASE + total_heat

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