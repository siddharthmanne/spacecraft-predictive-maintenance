import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import logging
from enum import Enum
from datetime import datetime, timedelta
import random


class ReactionWheelSubsystem:
    """    
    This module implements a physics-based telemetry generator for spacecraft reaction wheel assemblies.

    The goal is to create realistic sensor data that includes normal operations, environmental effects,
    and various failure modes that occur in real spacecraft systems.

    Key Design Principles:
    1. Physics-based modeling: All relationships based on real spacecraft engineering
    2. Configurable failure modes: Controllable degradation patterns for ML training
    3. Efficient algorithms: Sliding windows and hash maps for real-time performance
    4. Aerospace-realistic: Parameters match actual spacecraft component specifications

    Should simulate:
    1. Reaction Wheel Assembly
    - needs lubrication
    - generates microvibrations
    - saturation (may be at max speed, and speed then needs to be reduced)
    - 
    2. Battery System
    3. Thermal Control System

    Key insight: Real spacecraft data has:
    - Gaussian noise: Random variations around true values
    - Quantization noise: Digital conversion artifacts
    - Drift: Gradual calibration shifts over time
    - Outliers: Occasional spurious readings from electromagnetic interference

    - Normal operational cycles
    - Gradual degradation patterns  
    - Sudden anomalies
    - Environmental influences

    Key DS&A elements:
    - Hash maps for sensor configuration lookup
    - Sliding windows for time-series generation
    - Priority queues for alert ranking (future)

    Failures: 
    Bearing wear account for majority of RWA failures:
    1, Week 1-20: Vibration has 10-50% increase in amplitude at specific bearing frequencies. Temp has minimal change (less than 2C), slight increase in current ripple but mean current unchangned
    2, Weeks 20-40: Vibration has 50-200% increase, Temp 3-8C inc, 5-15% inc in average current draw with ripple increase
    3, Weeks 40-50: Vibration 200%+ increase, Temp 10-20C+ inc with potential thermal runaway, Current: 20%+ inc with erratic patterns
    4, Temp has rapid uncontrolled rise, becoming dominant indicator. Vibration may dec due to bearing seuzre. Current extreme spikes or motor stall. 
    - temp rise rate is great predictor of remaining useful life

    Lubration issues, often root cause of bearing wear
    Electrical failure less common but can be catastrophic

  


    """
    def __init__(self, 
                wheel_id="RWA_1", 
                mission_hours=8760):

        self.wheel_id = wheel_id

        # Hash map for sensor configurations - O(1) lookup
        self.sensors = {
            'vibration_g': {
                'range': (0.001, 0.1),
                'noise_std': 0.0005,
                'sampling_rate': 100,  # Reduced from 1000
                'units': 'g_rms'
            },
            'temperature_c': {
                'range': (-20, 85),
                'noise_std': 0.2,
                'sampling_rate': 1, # Since temperature doesn't change faster than this in space applications
                'units': 'degrees_celsius'
            },
            'motor_current_ma': {
                'range': (50, 1200),
                'noise_std': 5.0,
                'sampling_rate': 10,
                'units': 'milliamps'
            }
        }

        self.sensor_windows = {
            'temperature_c': SlidingWindow(window_size=10),
            'vibration_g': SlidingWindow(window_size=20),  # Need more samples for vibration
            'motor_current_ma': SlidingWindow(window_size=15)
        }

        # Initialize degradation model
        self.degradation_model = BearingDegradationModel()

        # The last sample times for different sampling rates
        self.last_sample_times = {sensor: 0.0 for sensor in self.sensors}


        # Health status sensors
        self.HEALTH_STATUS = {
            'bearing_condition_index': {
                'calculation': 'vibration_rms * temperature_rise_factor',
                'range': (0, 100),
                'units': 'health_percentage'
            },
            'lubrication_health': {
                'calculation': 'f(temperature_gradient, friction_power)',
                'range': (0, 100),
                'units': 'health_percentage'
            },
            'thermal_stress_index': {
                'calculation': 'temperature_rate_of_change * current_increase',
                'range': (0, 50),
                'units': 'stress_units'
            }
        }

        # Correlation matrix for sensor dependencies
        SENSOR_CORRELATIONS = {
            ('motor_current_ma', 'wheel_speed_rpm'): 0.85,  # Higher speed = higher current
            ('temperature_c', 'motor_current_ma'): 0.72,   # Current generates heat
            ('vibration_rms_g', 'wheel_speed_rpm'): 0.45,  # Speed affects vibration
            ('supply_voltage_v', 'motor_current_ma'): -0.35  # Voltage droop under load
        }

        # State machine for RWA operation
        OPERATIONAL_MODES = {
            'standby': {'power_fraction': 0.1, 'speed_target': 0},
            'momentum_bias': {'power_fraction': 0.4, 'speed_target': 3000},
            'active_control': {'power_fraction': 1.0, 'speed_target': 'variable'},
            'safe_mode': {'power_fraction': 0.05, 'speed_target': 0}
        }

        # Failure modes
        BEARING_DEGRADATION = {
            'name': 'bearing_wear',
            'onset_time_hours': (8760, 35040),  # 1-4 years random onset
            'progression_rate': 'exponential',
            'affected_sensors': {
                'vibration_rms_g': {
                    'degradation_factor': 1.5,  # Multiplier increases over time
                    'noise_increase': 2.0  # Additional noise component
                },
                'motor_current_ma': {
                    'degradation_factor': 1.2,  # Higher current for same speed
                    'offset_drift': 20  # Gradual baseline increase
                },
                'speed_stability': {
                    'degradation_factor': 3.0  # Speed becomes less stable
                }
            },
            'failure_threshold': {
                'vibration_rms_g': 0.05  # Critical vibration level
            }
        }

        MOTOR_DEGRADATION = {
            'name': 'motor_winding_wear',
            'onset_time_hours': (17520, 52560),  # 2-6 years
            'progression_rate': 'linear',
            'affected_sensors': {
                'motor_current_ma': {
                    'degradation_factor': 1.3,
                    'temperature_sensitivity': 1.1  # Worse at high temps
                },
                'temperature_c': {
                    'offset_drift': 8,  # Motor runs hotter
                    'thermal_resistance_increase': 0.15
                },
                'motor_efficiency': {
                    'degradation_factor': 0.95  # Decreasing efficiency
                }
            }
        }

        ELECTRONICS_FAILURE = {
            'name': 'controller_degradation',
            'onset_time_hours': (26280, 87600),  # 3-10 years
            'progression_rate': 'step_function',  # Can be sudden
            'affected_sensors': {
                'wheel_speed_rpm': {
                    'control_lag_increase': 2.0,  # Slower response
                    'overshoot_tendency': 1.4
                },
                'supply_voltage_v': {
                    'voltage_droop': 0.5  # Higher impedance
                }
            },
            'intermittent_faults': True  # Can have temporary failures
        }

        
        MISSION_PHASES = {
            'launch': {
                'duration_hours': 2,
                'vibration_multiplier': 5.0,  # High launch vibration
                'temperature_range': (-20, 60),
                'duty_cycle': 0.8  # High activity
            },
            
            'cruise': {
                'duration_hours': 8760,  # 1 year example
                'vibration_multiplier': 1.0,  # Nominal
                'temperature_range': (-30, 50),
                'duty_cycle': 0.3  # Low activity
            },
            
            'operational': {
                'duration_hours': 26280,  # 3 years
                'vibration_multiplier': 1.2,  # Some maneuvers
                'temperature_range': (-35, 70),
                'duty_cycle': 0.6  # Moderate activity
            }
        }

        ORBITAL_PARAMETERS = {
            'orbital_period_minutes': 90,  # Low Earth orbit
            'eclipse_fraction': 0.35,  # Time in Earth shadow
            'temperature_swing': 40,  # °C difference sun/shadow
            'radiation_environment': 'LEO',  # Affects electronics degradation
            'micrometeorite_rate': 1e-8  # Impacts per hour per m²
        }

        # Physics models for initialization
        THERMAL_MODEL = {
            'thermal_time_constant': 300,  # Seconds to respond to changes
            'power_to_temp_coefficient': 2.5,  # °C per Watt
            'ambient_coupling': 0.3,  # How much external temp affects internal
            'thermal_noise_bandwidth': 0.01  # Hz - slow thermal variations
        }

        MECHANICAL_MODEL = {
            'inertia_kg_m2': 0.012,  # Wheel moment of inertia
            'friction_coefficient': 0.02,  # Bearing friction
            'bearing_stiffness': 1e6,  # N/m - affects vibration transmission
            'resonant_frequencies': [45, 120, 380]  # Hz - structural resonances
        }




        # Hash map for failure mode definitions
        self.failure_modes = {
            'thermal_drift': {
                'affected_sensors': ['temp_sensor_1'],
                'onset_time': None,
                'drift_rate': 0.1,  # degrees per hour
                'type': 'gradual'
            },
            'mechanical_wear': {
                'affected_sensors': ['vibration_sensor_1'],
                'onset_time': None,
                'spike_frequency': 0.05,
                'type': 'intermittent'
            },
            'power_degradation': {
                'affected_sensors': ['power_sensor_1'],
                'onset_time': None,
                'degradation_rate': 0.5,  # watts per hour
                'type': 'gradual'
            },
            'antenna_degradation': {
                'affected_sensors': ['comm_signal_1'],
                'onset_time': None,
                'degradation_rate': 0.2,  # dB per hour
                'type': 'gradual'
            }
        }

        # Sliding window configuration
        self.window_size = 24  # hours
        self.sampling_rate = 1  # samples per hour