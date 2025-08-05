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
    bearing_temperature: float = 20.0   # New bearing runs at typical spacecraft lower bound (since usual operating range is 15-20C)

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
            # Base depletion rates
            'wear_rate_base': 1.2e-6,           # Reduced from 7.5e-6 to reach ~1.0 over 10 years
            'lubrication_depletion_rate': 2.5e-6, # Per operating hour
            'wear_activation_energy': 9.0e20, # Activation energy for wear acceleration (Joules)
            'load_stress_exponent': 1.0,      # Load stress relationship

            # Dependencies
            'lubrication_temp_factor': 0.5,  #increased temperature acceleration factor
            'surface_roughness_factor': 0.8,       # μm increase per wear unit
            'friction_increase_factor': 0.7,        # Friction coefficient increase
            'heat_generation_factor': 3.5,         # Temperature rise per unit friction increase

            # Critical thresholds
            'critical_temperature_threshold': 50.0,  # Temp at which lubrication quality begins decreasing significantly
            'debris_threshold_wear': 0.3,          # Wear level at which debris generation accelerates roughness
            'debris_acceleration_factor': 4.0,      # Exponential roughness increase after debris threshold

            #Max/min values
            'extreme_temperature_threshold': 80.0,        # Temperature threshold for discrete lubrication failure
            'max_surface_roughness': 8.0,        # Maximum roughness in micrometers
            'min_lubrication_quality': 0.05,       # Minimum residual lubrication
              
            'reference_temperature': 20.0,
            'baseline_surface_roughness': 0.32,  # μm, new bearing baseline
            'thermal_capacity': 0.8,               # Thermal inertia factor for temperature history
            'boltzmann_constant': 1.380649e-23,  # J/K, for physics calculations
            'thermal_conductance': 0.5,   # W/°C, example value—calibrate experimentally
            'mass_cp': 100.0,             # J/°C, combined bearing mass×specific heat
            'gas_constant': 8.314,         # J/(mol·K), universal gas constant
            'lubrication_activation_energy': 75e3,  # J/mol, typical activation energy for grease oxidation

            # Surface roughness and debris physics
            'running_in_duration': 0.05,  # Wear level where running-in completes (5%)
            'running_in_smoothing': 0.8,  # Roughness reduction factor during running-in
            'debris_onset_wear': 0.15,    # Earlier debris generation (15% instead of 50%)
            'debris_size_factor': 2.5,    # Particle size growth with wear
            'debris_contamination_threshold': 0.3,  # When debris contamination dominates
            'roughness_wear_exponent': 1.8,  # Non-linear roughness-wear relationship
            'max_debris_acceleration': 50.0,  # Higher realistic maximum

            # --- Friction model constants ---
            'viscosity_ref':          0.09,   # Pa·s at reference_temperature (example space-qualified grease)
            'viscosity_activation':   6.0e3,  # J/mol, viscosity’s Arrhenius slope
            'stribeck_exponent':      0.75,   # Stribeck curve slope (µ ~ (ηV/P)^-n) in mixed regime
            'asperity_plough_factor': 0.8,    # Additional µ per µm roughness when debris dominates
            'debris_friction_limit':  0.18,   # Upper µ when severe debris present
            'temperature_friction_slope': 1.5e-3,  # ∆µ per °C above reference (thin-film thinning)
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
        # cache_key = self._generate_cache_key(current_state, operating_conditions, time_delta_hours)
        # if cache_key in self._physics_cache:
            # self.cache_hits += 1
            # return copy.deepcopy(self._physics_cache[cache_key])
        
        # self.cache_misses += 1

        # Extract contextual operating conditions
        speed_rpm = operating_conditions.get('speed_rpm', 0.0) # If missing, assume stopped
        load_factor = operating_conditions.get('load_factor', 1.0) # If missing, assume nominal load

        # 1) Calculate incremental wear when rotating
        if speed_rpm > 0:
            # Physics-based wear progression using Archard's wear law
            R = self.physics_constants['gas_constant']  # J/(mol·K)
            Ea_wear = self.physics_constants['wear_activation_energy']  # J/mol
            T_K = current_state.bearing_temperature + 273.15
            T_ref_K = self.physics_constants['reference_temperature'] + 273.15

            temp_factor = np.exp(-Ea_wear/R * (1/T_K - 1/T_ref_K))
                                
            # Hertzian contact stress relationship
            # load_factor_adj = load_factor ** self.physics_constants['load_stress_exponent']

            # Base wear increment - Archard's law
            base_wear = (
                self.physics_constants['wear_rate_base'] * 
                temp_factor *
                load_factor *
                time_delta_hours
            )

            # State dependent acceleration in wear
            wear_acceleration = self._calculate_wear_acceleration(current_state)

            total_wear_increment = base_wear * wear_acceleration

        else:
            total_wear_increment = 0.0

        # 2) Update lubrication quality (accelerates with wear)
        lubrication_loss = self._calculate_lubrication_loss(current_state, current_state.bearing_temperature, time_delta_hours)

        # 3) Surface roughness with debris-driven acceleration
        new_surface_roughness = self._calculate_surface_roughness(
                                current_state, total_wear_increment
                                )

        # 4) Update friction coefficient based on wear and surface condition
        new_friction = self._calculate_friction_coefficient(
        current_state,
        total_wear_increment,
        speed_rpm=speed_rpm,
        load_N=operating_conditions.get('radial_load_N', 1.0)
)

        # 5) Compute net heat flow: Q_gen from friction minus Q_loss via conduction
        # Q_gen is joules per second generated by friction
        Q_gen = self.physics_constants['heat_generation_factor'] * new_friction * load_factor * max(1.0, speed_rpm/1000)
        #hA is conductive heat loss to housing
        hA = self.physics_constants['thermal_conductance']  # W/°C, add this constant
        mass_cp = self.physics_constants['mass_cp']         # J/°C, add this constant

        # dT/dt = (Q_gen - hA*(T - Tambient)) / (m·c_p)
        dT = (Q_gen - hA*(current_state.bearing_temperature - self.physics_constants['reference_temperature'])) \
            * (time_delta_hours * 3600) / mass_cp

        new_bearing_temperature = current_state.bearing_temperature + dT

        # Create new state
        new_state = BearingState(
            wear_level=min(1.0, max(0, current_state.wear_level + total_wear_increment)),
            friction_coefficient=new_friction,
            surface_roughness=new_surface_roughness,
            lubrication_quality=max(self.physics_constants['min_lubrication_quality'], current_state.lubrication_quality - lubrication_loss),
            bearing_temperature=new_bearing_temperature
        )
        # Cache result for UI performance
        # self._physics_cache[cache_key] = new_state
        return new_state



    def _calculate_wear_acceleration(self, bearing_state: BearingState) -> float:
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

        # As bearing surfaces roughen, real contact area grows nonlinearly, accelerating wear. 
        # A quadratic (exponent = 2) is a reasonable first approximation.
        wear_factor = 1.0 + (bearing_state.wear_level ** 2.0) * 1.0  

        # Lubrication degradation acceleration - as lubrication fails, metal on metal contact increases wear exponentially
        lubrication_degradation = 1.0 - bearing_state.lubrication_quality
        lube_factor = np.exp(2*lubrication_degradation)  # Exponential increase in wear with lubrication loss
        

        # Surface roughness acceleration
        # Rougher surfaces increase contact stress and wear
        roughness_ratio = bearing_state.surface_roughness / self.physics_constants['baseline_surface_roughness']
        roughness_factor = 1.0 + (roughness_ratio - 1.0)**1.3  # Nonlinear increase in wear with roughness  

        #Combined wear acceleration
        total_acceleration = wear_factor * lube_factor * roughness_factor

        # Cap acceleration to prevent numerical instability (q5x max)
        return min(self.physics_constants['max_debris_acceleration'], total_acceleration)

    def _calculate_lubrication_loss(self, bearing_state: BearingState, bearing_temperature: float, time_delta_hours: float) -> float:
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
        base_loss = self.physics_constants['lubrication_depletion_rate'] * time_delta_hours

        #Arrhenius temperature acceleration
        Ea_lube = self.physics_constants['lubrication_activation_energy']  
        R = self.physics_constants['gas_constant']  # J/(mol·K), universal gas constant

        T_K = bearing_temperature + 273.15
        T_ref_K = self.physics_constants['reference_temperature'] + 273.15

        # Arrhenius acceleration (1 at T_ref, >1 above)
        temp_acceleration = np.exp((-Ea_lube/R) * (1/T_K - 1/T_ref_K))

        
        # Wear particle contamination effect
        # Metal particles catalyze grease oxidation
        contamination_factor = 1.0 + 5.0 * bearing_state.wear_level
        
        # Total lubrication loss
        total_loss = base_loss * temp_acceleration * contamination_factor
        
        # Discrete lubrication failure for extreme temperature excursions
        if bearing_temperature > self.physics_constants['extreme_temperature_threshold']:
            total_loss = min(total_loss + 0.15, 1.0)  # Sudden loss under extreme thermal stress
        
        return total_loss
    
    def _calculate_surface_roughness(self, bearing_state: BearingState, wear_increment: float) -> float:
        """
        Calculate surface roughness evolution with proper running-in and debris physics
        
        Physics Basis:
        - Running-in phase: Initial smoothing (0-5% wear)  
        - Normal wear phase: Gradual roughening (5-15% wear)
        - Debris generation phase: Accelerated roughening (>15% wear)
        - Contamination phase: Exponential roughening (>30% wear)
        """
        current_roughness = bearing_state.surface_roughness
        wear_level = bearing_state.wear_level
        baseline = self.physics_constants['baseline_surface_roughness']

        # Phase 1: Running-in (initial smoothing)
        if wear_level < self.physics_constants['running_in_duration']:
            # New bearings actually get smoother initially
            smoothing_factor = self.physics_constants['running_in_smoothing']
            target_roughness = baseline * smoothing_factor
            
            # Exponential approach to smoother surface
            smoothing_rate = 5.0  # How quickly running-in occurs
            progress = wear_level / self.physics_constants['running_in_duration']
            running_in_roughness = baseline + (target_roughness - baseline) * (1 - np.exp(-smoothing_rate * progress))
            
            # Base roughness change from wear
            base_change = wear_increment * self.physics_constants['surface_roughness_factor']
            return max(running_in_roughness + base_change, target_roughness)
        
        # Phase 2: Normal wear progression  
        elif wear_level < self.physics_constants['debris_onset_wear']:
            # Standard roughness increase with non-linear wear relationship
            base_change = wear_increment * self.physics_constants['surface_roughness_factor']
            wear_factor = (wear_level ** self.physics_constants['roughness_wear_exponent'])
            return current_roughness + base_change * (1.0 + wear_factor)
        
        # Phase 3: Debris generation begins
        elif wear_level < self.physics_constants['debris_contamination_threshold']:
            base_change = wear_increment * self.physics_constants['surface_roughness_factor']
            
            # Debris particle generation acceleration
            debris_progress = (wear_level - self.physics_constants['debris_onset_wear']) / \
                            (self.physics_constants['debris_contamination_threshold'] - self.physics_constants['debris_onset_wear'])
            
            # Debris size grows with wear level
            particle_size_factor = 1.0 + self.physics_constants['debris_size_factor'] * debris_progress
            
            # Moderate exponential acceleration
            debris_acceleration = 1.0 + (np.exp(2.0 * debris_progress) - 1.0) * particle_size_factor
            
            return current_roughness + base_change * debris_acceleration
        
        # Phase 4: Debris contamination dominates
        else:
            base_change = wear_increment * self.physics_constants['surface_roughness_factor']
            
            # Severe contamination effects
            contamination_level = wear_level - self.physics_constants['debris_contamination_threshold']
            contamination_factor = min(self.physics_constants['max_debris_acceleration'], 
                                    np.exp(4.0 * contamination_level))
            
            # Lubrication quality affects debris suspension
            lube_effect = 2.0 / (bearing_state.lubrication_quality + 0.1)  # Worse lube = more debris settling
            
            total_acceleration = contamination_factor * lube_effect
            new_roughness = current_roughness + base_change * total_acceleration
            
            # Cap at maximum physical roughness
            return min(new_roughness, self.physics_constants['max_surface_roughness'])

    def _calculate_friction_coefficient(
            self,
            bearing_state: BearingState,
            wear_increment: float,
            speed_rpm: float = 0.0,
            load_N: float = 1.0
    ) -> float:
        """
        Mixed-regime friction model with Stribeck behaviour, roughness & debris effects.
        """
        # ---------- 1. Hydrodynamic / mixed-film component ----------
        # Dynamic viscosity Arrhenius drop with temperature
        R = self.physics_constants['gas_constant']
        eta_ref = self.physics_constants['viscosity_ref']
        Ea_eta = self.physics_constants['viscosity_activation']
        
        T_K = bearing_state.bearing_temperature + 273.15
        T_ref_K = self.physics_constants['reference_temperature'] + 273.15
        eta = eta_ref * np.exp(-Ea_eta/R * (1/T_K - 1/T_ref_K))
        
        # Sommerfeld number proxy: (η·V) / P   — V≈π·d·speed, use speed_rpm scaled
        V = max(speed_rpm, 1.0) * 2*np.pi/60  # rad/s proxy
        P = max(load_N, 1.0)                  # normalise if you don’t have real load
        mixed_mu = 0.008 * (eta*V/P) ** (-self.physics_constants['stribeck_exponent'])
        
        # ---------- 2. Boundary / asperity component ----------
        # Base steel-on-steel boundary µ
        base_boundary_mu = 0.02
        
        # Roughness contribution (non-linear)
        rough_ratio = bearing_state.surface_roughness / self.physics_constants['baseline_surface_roughness']
        asperity_mu = base_boundary_mu + (rough_ratio - 1.0) * self.physics_constants['asperity_plough_factor'] * 0.01
        
        # Lubrication film penalty (goes up as lube_quality ↓)
        lube_penalty = 1.0 + 3.0 * (1.0 - bearing_state.lubrication_quality) ** 0.7
        
        # Debris penalty – significant when roughness already high or wear > 30 %
        if bearing_state.wear_level > 0.3:
            debris_penalty = 1.0 + 4.0 * (bearing_state.wear_level - 0.3)
        else:
            debris_penalty = 1.0
        
        boundary_mu = asperity_mu * lube_penalty * debris_penalty
        
        # ---------- 3. Temperature-induced thin-film rise ----------
        temp_penalty = 1.0 + self.physics_constants['temperature_friction_slope'] * \
                    max(0.0, bearing_state.bearing_temperature - self.physics_constants['reference_temperature'])
        
        # ---------- 4. Combine using harmonic mean (common in mixed lubrication) ----------
        combined_mu = 1.0 / (1.0/mixed_mu + 1.0/boundary_mu)
        combined_mu *= temp_penalty
        
        # Physical upper bound
        combined_mu = min(combined_mu, self.physics_constants['debris_friction_limit'])
        return combined_mu

    # def _calculate_friction_coefficient(self, bearing_state: BearingState, wear_increment: float) -> float: 
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
        
        # Combined friction calculation
        new_friction = (base_friction + roughness_increase + wear_contribution) * lube_multiplier
        
        # Physical limits for steel bearings
        return min(new_friction, 0.20)  # Maximum reasonable friction coefficient

    
    def predict_wear_progression(self, current_state: BearingState, conditions: Dict[str, float], time_horizon_hours: float) -> Dict[str, float]:
        """
        Uses physics models to project how wear will advance under expected operating conditions (that are fed by higher level classes)
        """
        # Use cached physics calculations to project forward
        bearing_temperature = current_state.bearing_temperature
        load_factor = conditions.get('load_factor', 1.0)

        # Calculate wear rate under expected conditions using proper Arrhenius relationship
        R = self.physics_constants['gas_constant']  # J/(mol·K)
        Ea_wear = self.physics_constants['wear_activation_energy']  # J/mol
        T_K = bearing_temperature + 273.15
        T_ref_K = self.physics_constants['reference_temperature'] + 273.15

        temp_factor = np.exp(-Ea_wear/R * (1/T_K - 1/T_ref_K))
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

    def get_physical_properties(self, bearing_state: BearingState) -> Dict[str, float]:
        """
        Return the current physical properties of the bearing as a dictionary.
        
        Parameters:
        - bearing_state: Current bearing state
        
        Returns: Dictionary with physical properties
        """
        return {
            'wear_level': bearing_state.wear_level,
            'surface_roughness': bearing_state.surface_roughness,
            'friction_coefficient': bearing_state.friction_coefficient,
            'lubrication_quality': bearing_state.lubrication_quality,
            'bearing_temperature': bearing_state.bearing_temperature
        }
        

    
    # UTILITY FUNCTIONS

    # def _generate_cache_key(self, bearing_state: BearingState, conditions: Dict[str, float], time_delta: float) -> str:
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
        def safe_int(val, scale, default=0):
            v = val if not np.isnan(val) and not np.isinf(val) else default
            return int(round(v * scale))

        discretized = {
            'wear': safe_int(bearing_state.wear_level, 10000, 0),      # 0.01% precision
            'friction': safe_int(bearing_state.friction_coefficient, 1000, 0),  # 0.001 precision
            'roughness': safe_int(bearing_state.surface_roughness, 100, 0),     # 0.01 μm precision
            'lube': safe_int(bearing_state.lubrication_quality, 1000, 0),       # 0.001 precision
            'temp': safe_int(bearing_state.bearing_temperature, 10, 200),       # 0.1°C precision
            'load': safe_int(conditions.get('load_factor', 1.0), 100, 100),       # 0.01 precision
            'speed': safe_int(conditions.get('speed_rpm', 0.0), 0.1, 0),         # 10 RPM precision
            'dt': safe_int(time_delta, 100, 0)                                  # 0.01 hour precision
        }
        
        # Create sorted key string for consistency
        key_parts = [f"{k}:{v}" for k, v in sorted(discretized.items())]
        key_string = "_".join(key_parts)
        
        # Use MD5 hash for fixed-length keys
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    #def get_cache_statistics(self) -> Dict[str, int]:
        """Return caching performance metrics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(1, total_requests)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 1),
            'cached_items': len(self._physics_cache)
        }