
"""
Uses physics to simulate bearing degradation. 
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import hashlib

@dataclass
class BearingState:
    """
    Physical state parameters for bearing degradation
    The values below represent the initial state of a new bearing.
    """
    wear_level: float = 0.0          # New bearing starts at 0% wear
    friction_coefficient: float = 0.02  # New bearing has nominal  friction
    surface_roughness: float = 0.32      # micrometers, new bearing baseline roughness
    lubrication_quality: float = 1.0    # New bearing has near perfect lubrication
    temperature_history: float = 15.0   # New bearing runs at typical spacecraft lower bound (since usual operating range is 15-20C)

@dataclass
class BearingDegradationModel:
    def __init__(self):
        """
        Hash map for caching expensive calculations.

        Why?
        My UI shows a timelapse over months/years of mission time. Without caching, the same physics calculations repeat thousands of times:
        """
        self._physics_cache = {}
        self._state_cache = {}
        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0

        # Physics constants derived from bearing engineering literature
        self.physics_constants = {
            'wear_rate_base': 1.2e-10,           # Base wear per operating hour
            'friction_wear_factor': 1.5,      # How friction accelerates wear
            'activation_energy_ratio': 0.045,
            'load_stress_exponent': 1.33,      # Load stress relationship
            # 'vibration_threshold': 0.01,      # g-force threshold for damage
            'lubrication_depletion_rate': 2.1e-7, # Per operating hour
            'critical_temperature_threshold': 36.0,  # NASA maximum test temperature
            'lubrication_temp_factor' : 0.08, #accelerated depltion at high temp
            'surface_roughness_factor': 0.1,       # μm increase per wear unit
            'friction_increase_factor': 0.3,        # Friction coefficient increase
            'reference_temperature' : 20.0
        }


    # CORE SIMULATION FUNCTIONS
    def update_bearing_state(self, current_state: BearingState, operating_conditions: Dict[str, float], time_delta_hours: float) -> BearingState:
        """
        CORE FUNCTION: Updates bearing wear based on operating conditions
        Called every simulation timestep (1 hour) by Reaction Wheel class
        
        Input: temperature, speed, load from higher-level classes
        Output: New BearingState with updated physical properties (wear level, surface roughness, etc)
        """
        # Check cache first
        cache_key = self.generate_cache_key(current_state, operating_conditions, time_delta_hours)
        if cache_key in self._physics_cache:
            self.cache_hits += 1
            return self._physics_cache[cache_key].copy()
        
        self.cache_misses += 1

        # Extract contextual operating conditions
        speed_rpm = operating_conditions.get('speed_rpm', 0.0) # If missing, assume stopped
        temperature = operating_conditions.get('temperature', 20.0) # If missing, assume typical operating temp between 15-20C, average 17.5C
        load_factor = operating_conditions.get('load_factor', 1.0) # If missing, assume nominal load

        # Calculate incremenetal wear when rotating
        if speed_rpm > 0:
            # Physics-based wear progression using Archard's wear law
            temp_factor = np.exp(self.physics_constants['activation_energy_ratio'] * 
                                (temperature - self.physics_constants['reference_temperature']) / 
                                (temperature + 273.15) )  # Reference temperature for physics equation. Convert to kelvin for proper Arrhenius

            # Hertzian contact stress relationship
            load_factor_adj = load_factor ** self.physics_constants['load_stress_exponent']

            # Base wear increment - Archard's law
            base_wear = (
                self.physics_constants['wear_rate_base'] * 
                temp_factor *
                load_factor_adj *
                time_delta_hours
            )

            # State dependent acceleration in wear
            wear_acceleration = self._calculate_wear_acceleration(current_state, operating_conditions)

            total_wear_increment = base_wear * wear_acceleration

        else:
            total_wear_increment = 0.0

        # Update lubrication quality (accelerates with wear)
        lubrication_loss = self._calculate_lubrication_loss(current_state, temperature, time_delta_hours)

        # Surface roughness accelerates as a function of existing roughness and wear
        new_surface_roughness = current_state.surface_roughness + (
            total_wear_increment * self.physics_constants['surface_roughness_factor'] *
            (1.0 + current_state.wear_level)
        )
        
        # Update friction coefficient based on wear and surface condition
        new_friction = self._calculate_friction_coefficient(
            current_state, total_wear_increment
        )

        # Temperature history with exponential moving average
        new_temp_history = (
            0.9 * current_state.temperature_history + 
            0.1 * temperature
        )

        # Create new state
        new_state = BearingState(
            wear_level=min(1.0, current_state.wear_level + total_wear_increment),
            friction_coefficient=new_friction,
            surface_roughness=new_surface_roughness,
            lubrication_quality=max(0.0, current_state.lubrication_quality - lubrication_loss),
            temperature_history=new_temp_history
        )
        
        # Cache result for UI performance
        self._physics_cache[cache_key] = new_state
        return new_state



    def _calculate_wear_acceleration(self, bearing_state: BearingState, conditions: Dict[str, float]) -> float:
        """
        STATE-DEPENDENT wear acceleration 
        
        Physics Basis:
        - Early life: Linear wear progression (Archard's law)
        - Mid-life: Gradual acceleration due to surface roughening
        - End-of-life: Exponential acceleration due to debris generation
        
        Parameters:
        - bearing_state: Current physical condition
        
        Returns: Acceleration factor (1.0 = nominal rate, 2.0 = 2x faster)
        """

        # Literature shows quadratic relationship for bearing life
        wear_factor = 1.0 + (bearing_state.wear_level ** 2.0) * 2.0 

        # Lubrication degradation acceleration - as lubricaton fails, metal on metal contact increases wear exponentially
        lube_degradation = 1.0 - bearing_state.lubrication_quality
        lube_factor = 1.0 + (lube_degradation ** 1.5) * 2.5  # Max 3.5x when no lubrication
        

        # Surface roughness acceleration
        # Rougher surfaces increase contact stress and wear
        baseline_roughness = 0.32  # Your new bearing baseline
        roughness_ratio = bearing_state.surface_roughness / baseline_roughness
        roughness_factor = 1.0 + 0.5 * max(0, roughness_ratio - 1.0)  # Only when rougher than new

        #Combined wear acceleration
        total_acceleration = wear_factor * lube_factor * roughness_factor

        # Cap acceleration to prevent numerical instability (5x max)
        return min(5.0, total_acceleration)

    def _calculate_lubrication_loss(self, bearing_state: BearingState, temperature: float, time_delta_hours: float) -> float:
        """
        Calculate lubrication degradation using grease chemistry kinetics.
        
        Architecture Role: Internal physics for lubricant depletion
        
        Physics Basis:
        - Base depletion rate from grease evaporation/oxidation
        - Temperature acceleration (Arrhenius relationship)
        - Wear particle contamination effects
        
        Parameters:
        - bearing_state: Current condition (affects contamination)
        - temperature: Operating temperature (°C)
        - time_delta_hours: Time step for integration
        
        Returns: Lubrication quality loss (0.0 to 1.0 scale)
        """

        # Base depletion rate (evaporation + oxidation)
        base_loss = self.physics_constants['lubrication_depletion_rate'] * time_delta_hours
        # Temperature acceleration 
        if temperature > self.physics_constants['critical_temperature']:
            temp_acceleration = 1.0 + (
                self.physics_constants['lubrication_temp_factor'] * 
                (temperature - self.physics_constants['critical_temperature'])
            )
        else:
            temp_acceleration = 1.0
        
        # Wear particle contamination effect
        # Metal particles catalyze grease oxidation
        contamination_factor = 1.0 + 3.0 * bearing_state.wear_level
        
        # Total lubrication loss
        total_loss = base_loss * temp_acceleration * contamination_factor
        
        return total_loss
    
    def _calculate_friction_coefficient(self, bearing_state: BearingState, wear_increment: float) -> float:
        """
        Calculate friction coefficient based on surface condition and lubrication.
        
        Architecture Role: Material property calculation for physics engine
        
        Physics Basis:
        - Base friction from material pairing (steel-on-steel)
        - Surface roughness effects (Coulomb friction)
        - Lubrication film thickness effects (Stribeck curve)
        
        Parameters:
        - bearing_state: Current physical condition
        - wear_increment: Recent wear to update friction
        
        Returns: Updated friction coefficient
        """
        # Base friction coefficient for your new bearing (0.02 is good)
        base_friction = 0.02
        
        # Surface roughness contribution
        roughness_increase = (
            bearing_state.surface_roughness / 0.32 - 1.0  # Normalized to new bearing
        ) * 0.01  # 0.01 friction increase per unit roughness ratio
        
        # Lubrication film effectiveness (Stribeck relationship)
        # Full film: low friction, boundary lubrication: high friction
        lube_multiplier = 1.0 + 2.0 * (1.0 - bearing_state.lubrication_quality) ** 0.5
        
        # Wear contribution (fresh metal exposure)
        wear_contribution = (
            self.physics_constants['friction_increase_factor'] * 
            bearing_state.wear_level
        )
        
        # Combined friction calculation
        new_friction = (base_friction + roughness_increase + wear_contribution) * lube_multiplier
        
        # Physical limits for steel bearings
        return min(new_friction, 0.15)  # Maximum reasonable friction coefficient



    def get_physical_properties(self, bearing_state: BearingState) -> Dict[str, float]:
        """
        Returns: Pure physics properties
        These are the instrinsic material properties of the bearings. 
        """
        return {
            'wear_level': bearing_state.wear_level,           # Raw wear progression
            'surface_roughness': bearing_state.surface_roughness,     # Physical surface condition  
            'friction_coefficient': bearing_state.friction_coefficient, # Material friction property
            'lubrication_quality': bearing_state.lubrication_quality    # Lubricant condition
        }
    
    def predict_wear_progression(self, current_state: BearingState, conditions: Dict[str, float], time_horizon_hours: float) -> Dict[str, float]:
        """
        Uses physics models to project how wear will advance under expected operating conditions (that are fed by higher level classes)
        """
        # Use cached physics calculations to project forward
        temperature = conditions.get('temperature', 20.0)
        load_factor = conditions.get('load_factor', 1.0)
        
        # Calculate wear rate under expected conditions
        temp_factor = np.exp(self.physics_constants['temperature_acceleration'] * (temperature - 20.0))
        load_factor_adj = load_factor ** self.physics_constants['load_stress_exponent']
        
        hourly_wear_rate = (
            self.physics_constants['wear_rate_base'] * 
            temp_factor * 
            load_factor_adj
        )
        
        # Project wear timeline
        projected_wear = current_state.wear_level + (hourly_wear_rate * time_horizon_hours)
        time_to_failure = (0.95 - current_state.wear_level) / max(hourly_wear_rate, 1e-10)
        
        return {
            'projected_wear_level': min(1.0, projected_wear),
            'time_to_failure_hours': time_to_failure,
            'wear_rate_per_hour': hourly_wear_rate
        }
        
        

    
    # UTILITY FUNCTIONS

    def _generate_cache_key(self, bearing_state: BearingState, conditions: Dict[str, float], time_delta: float) -> str:
        """
        Generate consistent cache keys for physics calculations.
        
        Architecture Role: Performance optimization for UI timelapse
        
        Why This Approach:
        - Integer discretization prevents floating-point cache misses
        - Includes all parameters that affect physics
        - Fast string-based hashing
        
        Parameters:
        - bearing_state: Current bearing condition
        - conditions: Operating conditions dictionary  
        - time_delta: Simulation timestep
        
        Returns: Consistent hash key string
        """
        
        # Discretize floating point values to prevent cache misses
        discretized = {
            'wear': int(bearing_state.wear_level * 10000),      # 0.01% precision
            'friction': int(bearing_state.friction_coefficient * 1000),  # 0.001 precision
            'roughness': int(bearing_state.surface_roughness * 100),     # 0.01 μm precision
            'lube': int(bearing_state.lubrication_quality * 1000),       # 0.001 precision
            'temp': int(conditions.get('temperature', 17.5) * 10),       # 0.1°C precision
            'load': int(conditions.get('load_factor', 1.0) * 100),       # 0.01 precision
            'speed': int(conditions.get('speed_rpm', 0.0) / 10),         # 10 RPM precision
            'dt': int(time_delta * 100)                                  # 0.01 hour precision
        }
        
        # Create sorted key string for consistency
        key_parts = [f"{k}:{v}" for k, v in sorted(discretized.items())]
        key_string = "_".join(key_parts)
        
        # Use MD5 hash for fixed-length keys
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def get_cache_statistics(self) -> Dict[str, int]:
        """Return caching performance metrics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(1, total_requests)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 1),
            'cached_items': len(self._physics_cache)
        }