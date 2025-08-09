
"""
Module: bearing_degradation.py

Description:
This module implements a physics-based bearing degradation simulation model for spacecraft components (e.g., reaction wheels).
It contains the BearingState dataclass that encapsulates physical state parameters of the bearing,
and the BearingDegradationModel class that simulates time-dependent wear, lubrication degradation,
multi-factor friction development, and physical state updates based on operating conditions.
The model uses established physics laws and empirical constants calibrated from aerospace literature,
and supports caching of expensive calculations for performance.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import copy

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
    bearing_temperature: float = 15.0   # New bearing runs at typical spacecraft lower bound (since usual operating range is 15-20C)

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
        # Adjusted to meet specifications: max wear by year 8-9, surface roughness max 8-9μm by year 6+,
        # friction max 0.15 by year 6-8, lubrication min 0.15 by year 6-7
        self.physics_constants = {

            'wear_rate_base': 9.0e-6,           # Reduced significantly for slower wear progression (max by year 8-9)

            'activation_energy_ratio': 0.035,   # Slightly reduced temperature sensitivity

            'load_stress_exponent': 1.25,       # Reduced load sensitivity

            'lubrication_depletion_rate': 9.0e-6, # Reduced to maintain min 0.15 quality by year 6-7
            'lubrication_minimum': 0.15,        # Minimum lubrication quality (degradation floor)
            'critical_temperature_threshold': 60.0,  # Increased threshold for better lubrication retention
            'lubrication_temp_factor' : 0.06,   # Reduced temperature acceleration
            'surface_roughness_factor': 0.20,   # Adjusted for max 8-9μm by year 6+
            'surface_roughness_max': 9.0,       # Maximum surface roughness in micrometers
            'friction_increase_factor': 0.08,   # Reduced for max 0.15 friction by year 6-8
            'friction_maximum': 0.15,           # Maximum friction coefficient
            'reference_temperature' : 293.15, # Starting bearing temperature in Kelvin (20°C)
        }


    # CORE SIMULATION FUNCTIONS
    def update_bearing_state_one_hour(self, current_state: BearingState, operating_conditions: Dict[str, float]) -> BearingState:
        """
        CORE FUNCTION: Updates bearing wear based on operating conditions
        Called every simulation timestep (1 hour) by Reaction Wheel class

        Reason for small timestep: When updating wear and lubrication (that are time-dependent degradation), we assume all other relevant factors remain constant. This estimation is ok for small timesteps like one hour, but during big time steps, you 'average out' all those feedbacks and lose realism.

        Input: temperature, speed, load from higher-level classes
        Output: New BearingState with updated physical properties (wear level, surface roughness, etc)
        """
        # Check cache first
        cache_key = self._generate_cache_key(current_state, operating_conditions)
        if cache_key in self._physics_cache:
            self.cache_hits += 1
            return copy.deepcopy(self._physics_cache[cache_key])
        
        self.cache_misses += 1

        # Extract contextual operating conditions
        speed_rpm = operating_conditions.get('speed_rpm', 0.0) # If missing, assume stopped
        load_factor = operating_conditions.get('load_factor', 1.0) # If missing, assume nominal load

        # Calculate incremenetal wear when rotating
        if speed_rpm > 0:
            # Physics-based wear progression using Archard's wear law
            temp_factor = np.exp(self.physics_constants['activation_energy_ratio'] * 
                                (current_state.bearing_temperature - self.physics_constants['reference_temperature']) / 
                                (current_state.bearing_temperature + 273.15) )  # Reference temperature for physics equation. Convert to kelvin for proper Arrhenius

            # Hertzian contact stress relationship
            load_factor_adj = load_factor ** self.physics_constants['load_stress_exponent']

            # Base wear increment - Archard's law
            base_wear = (
                self.physics_constants['wear_rate_base'] * 
                temp_factor *
                load_factor_adj 
            )

            # State dependent acceleration in wear
            wear_acceleration = self._calculate_wear_acceleration(current_state, operating_conditions)

            total_wear_increment = base_wear * wear_acceleration

        else:
            total_wear_increment = 0.0

        # Update lubrication quality (accelerates with wear) with minimum floor
        lubrication_loss = self._calculate_lubrication_loss(current_state, current_state.bearing_temperature)
        min_lubrication = self.physics_constants['lubrication_minimum']
        
        # Apply progressive scaling that slows down as it approaches minimum
        current_lube = current_state.lubrication_quality
        if current_lube > min_lubrication:
            # Scale loss to approach minimum asymptotically
            lube_factor = max(0.1, (current_lube - min_lubrication) / (1.0 - min_lubrication))
            adjusted_loss = lubrication_loss * lube_factor
        else:
            adjusted_loss = 0.0  # No further loss below minimum

        # Surface roughness accelerates as a function of existing roughness and wear
        # Apply progressive scaling that slows down as it approaches maximum
        current_roughness = current_state.surface_roughness
        max_roughness = self.physics_constants['surface_roughness_max']
        roughness_factor = max(0.1, 1.0 - (current_roughness / max_roughness) ** 1.5)  # Slows as approaching max
        
        new_surface_roughness = min(
            max_roughness,
            current_roughness + (
                total_wear_increment * self.physics_constants['surface_roughness_factor'] *
                (1.0 + current_state.wear_level) * roughness_factor
            )
        )
        
        # Update friction coefficient based on wear and surface condition
        new_friction = self._calculate_friction_coefficient(
            current_state, total_wear_increment
        )

        # Calculate bearing temperature based on operating conditions
        # Higher speeds and loads generate more heat
        # Temperature rise from friction and speed
        """ 
        temp evolution: T_new = T_old + (Q_gen - Q_loss) * dt / thermal capacity--> Qgen/loss = f(speed, friction, load)
        visc update: visc_new = visc_old * exp(-beta * T_new) * degradactor_factor(time)
        film thickness: h_new = f(visc_new, speed, load, s_r)

        wear_volume = K * (load/hardness) * sliding distance * dt
        s_r_new = s_r_old + f(wear_volume, contact_stress)

        # fric
        lambda_ratio = h_new / composite_roughness
        friction_coefficient = f(lambda_ratio, regime_transition_logic)
        """
        speed_temp_rise = (speed_rpm / 1000.0) * 2.0  # ~2°C per 1000 RPM
        load_temp_rise = (load_factor - 1.0) * 5.0    # ~5°C per unit load above nominal
        friction_temp_rise = current_state.friction_coefficient * 50.0  # Higher friction = more heat
        
        # Calculate target temperature
        target_temperature = self.physics_constants['reference_temperature'] + speed_temp_rise + max(0, load_temp_rise) + friction_temp_rise
        
        # Use exponential moving average to simulate thermal time constant
        new_bearing_temperature = (
            0.8 * current_state.bearing_temperature + 
            0.2 * target_temperature
        )

        # Create new state
        new_state = BearingState(
            wear_level=min(1.0, max(0, current_state.wear_level + total_wear_increment)), # Ensures wear level is relative value between 0 and 1
            friction_coefficient=new_friction,
            surface_roughness=new_surface_roughness,
            lubrication_quality=max(min_lubrication, current_state.lubrication_quality - adjusted_loss),
            bearing_temperature=new_bearing_temperature
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

        # Cap acceleration to prevent numerical instability (q5x max)
        return min(15.0, total_acceleration)

    def _calculate_lubrication_loss(self, bearing_state: BearingState, temperature: float) -> float:
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
        base_loss = self.physics_constants['lubrication_depletion_rate'] 
        # Temperature acceleration 
        if temperature > self.physics_constants['critical_temperature_threshold']:
            temp_acceleration = 1.0 + (
                self.physics_constants['lubrication_temp_factor'] * 
                (temperature - self.physics_constants['critical_temperature_threshold'])
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
        """ WHAT IS THE ROLE OF SECOND PARAMETER"""
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
        
        # Combined friction calculation with progressive scaling near maximum
        base_friction_calc = (base_friction + roughness_increase + wear_contribution) * lube_multiplier
        max_friction = self.physics_constants['friction_maximum']
        
        # Apply progressive scaling that slows down as it approaches maximum
        if base_friction_calc < max_friction:
            new_friction = base_friction_calc
        else:
            new_friction = max_friction
        
        # Physical limits for steel bearings
        return min(new_friction, max_friction)



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
        temperature = current_state.bearing_temperature
        load_factor = conditions.get('load_factor', 1.0)
        
        # Calculate wear rate under expected conditions
        temp_factor = np.exp(self.physics_constants['activation_energy_ratio'] * (temperature - self.physics_constants['reference_temperature']) / (temperature + 273.15))
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

    def _generate_cache_key(self, bearing_state: BearingState, conditions: Dict[str, float]) -> str:
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
        
        Returns: Consistent hash key string
        """
        
        # Discretize floating point values to prevent cache misses
        discretized = {
            'wear': int(bearing_state.wear_level * 100000),     # 0.001% precision (increased)
            'friction': int(bearing_state.friction_coefficient * 10000),  # 0.0001 precision (increased)
            'roughness': int(bearing_state.surface_roughness * 1000),     # 0.001 μm precision (increased)
            'lube': int(bearing_state.lubrication_quality * 10000),       # 0.0001 precision (increased)
            'temp': int(bearing_state.bearing_temperature * 100),         # 0.01°C precision (increased)
            'load': int(conditions.get('load_factor', 1.0) * 1000),       # 0.001 precision (increased)
            'speed': int(conditions.get('speed_rpm', 0.0) / 1),          # 1 RPM precision (increased)
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