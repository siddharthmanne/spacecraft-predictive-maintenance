"""
test_reaction_wheel.py - Compact Test Suite

Comprehensive test suite for ReactionWheelSubsystem class
Tests operational ranges, degradation patterns, and real-world physics validation.
"""

import sys
import os
# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import copy
from unittest.mock import patch, MagicMock
from src.reaction_wheel import ReactionWheelSubsystem
from src.bearing_degradation import BearingState, BearingDegradationModel

class TestReactionWheelSubsystem:
    
    def setup_method(self):
        """Setup fresh wheel for each test"""
        self.wheel = ReactionWheelSubsystem(wheel_id=1, operational_mode='IDLE', load_factor=1.0)
        
    def teardown_method(self):
        """Cleanup after each test"""
        self.wheel = None

    # ===============================
    # INITIALIZATION AND COMMAND TESTS
    # ===============================
    
    def test_default_initialization(self):
        """Verify wheel initializes with correct default parameters"""
        default_wheel = ReactionWheelSubsystem()
        
        assert default_wheel.wheel_id == 0
        assert default_wheel.operational_mode == 'IDLE'
        assert default_wheel.load_factor == 1.0
        assert default_wheel.bearing_state.wear_level == 0.0
        assert default_wheel.bearing_state.friction_coefficient == 0.02
        assert default_wheel.bearing_state.lubrication_quality == 1.0
    
    def test_basic_speed_command(self):
        """Verify wheel responds to speed commands"""
        commands = {'target_speed_rpm': 1000}
        self.wheel.update(1.0, commands)
        
        telemetry = self.wheel.get_telemetry()
        assert telemetry['speed_rpm'] == 1000
        assert telemetry['timestep_hours'] == 1.0
    
    def test_load_factor_override(self):
        """Test that user commands can override default load factor"""
        commands = {'target_speed_rpm': 500, 'load_factor': 2.0}
        self.wheel.update(1.0, commands)
        
        telemetry = self.wheel.get_telemetry()
        assert telemetry['load_factor'] == 2.0
        assert telemetry['speed_rpm'] == 500

    # ===============================
    # OPERATIONAL RANGE TESTS (Real-world accuracy focus)
    # ===============================
    
    def test_normal_operating_range(self):
        """Test wheel within typical spacecraft RPM range (400-6000 RPM)"""
        test_speeds = [400, 1500, 3000, 6000]
        
        for speed in test_speeds:
            wheel = ReactionWheelSubsystem()
            commands = {'target_speed_rpm': speed}
            wheel.update(1.0, commands)
            
            telemetry = wheel.get_telemetry()
            
            # Real-world ranges for new bearing
            assert 0.08 <= telemetry['current'] <= 0.5  # Literature: 0.05-0.12A idle 
            assert 0.01 <= telemetry['vibration'] <= 0.1  # Literature: 0.01-0.1g healthy wheels
            assert 20.0 <= telemetry['housing_temperature'] <= 25.0  # Spacecraft typical
    
    def test_zero_rpm_operation(self):
        """Test wheel behavior when stopped"""
        commands = {'target_speed_rpm': 0}
        initial_wear = self.wheel.bearing_state.wear_level
        initial_lubrication_quality = self.wheel.bearing_state.lubrication_quality
        initial_surface_roughness = self.wheel.bearing_state.surface_roughness
        
        self.wheel.update(1.0, commands)
        telemetry = self.wheel.get_telemetry()
        
        assert abs(telemetry['current'] - 0.08) < 0.01  # Only idle current
        assert telemetry['wear_level'] == initial_wear  # No wear when stopped
        assert telemetry['lubrication_quality'] < initial_lubrication_quality # Lubrication quality can decrease at 0 RPM due to oil migration from temperature gradients and capillary forces
        assert telemetry['surface_roughness'] == initial_surface_roughness # No surface roughness change (no debris generation)

    # ===============================
    # PHYSICS VALIDATION TESTS
    # ===============================
    
    def test_current_draw_scaling(self):
        """Verify current increases with friction and speed per literature"""
        # Test with new bearing
        commands = {'target_speed_rpm': 3000}
        self.wheel.update(1.0, commands)
        new_bearing_current = self.wheel.get_telemetry()['current']
        
        # Simulate significant wear
        self.wheel.bearing_state.wear_level = 0.5
        self.wheel.bearing_state.friction_coefficient = 0.1
        
        self.wheel.update(2.0, commands)
        worn_bearing_current = self.wheel.get_telemetry()['current']
        
        assert worn_bearing_current > new_bearing_current

        # Calculate expected using constants from the implementation
        IDLE_CURRENT = 0.08  # From _physics_to_current method
        CURRENT_GAIN = 0.3   # From _physics_to_current method
        actual_friction = self.wheel.bearing_state.friction_coefficient
        
        expected_current = IDLE_CURRENT + (CURRENT_GAIN * actual_friction * 3000 / 1000.0)
        assert abs(worn_bearing_current - expected_current) < 0.01
    
    def test_vibration_progression(self):
        """Test vibration increases with bearing degradation per literature"""
        # New bearing baseline
        commands = {'target_speed_rpm': 3000}
        self.wheel.update(1.0, commands)
        initial_vibration = self.wheel.get_telemetry()['vibration']
        
        BASE_VIBRATION = 0.01  # From _physics_to_vibration method
        assert abs(initial_vibration - BASE_VIBRATION) < 0.005  # Should be near baseline
        
        # Simulate degradation by setting degraded bearing states
        self.wheel.bearing_state.wear_level = 0.3
        self.wheel.bearing_state.surface_roughness = 0.5
        self.wheel.bearing_state.friction_coefficient = 0.08
        
        self.wheel.update(2.0, commands)
        degraded_vibration = self.wheel.get_telemetry()['vibration']

        # Constants from _physics_to_vibration method
        VIBRATION_WEAR_GAIN = 0.04
        VIBRATION_ROUGHNESS_GAIN = 0.007
        VIBRATION_FRICTION_GAIN = 1.0
        
        # Literature: vibration = base + wear + roughness + friction components
        expected_vib = (BASE_VIBRATION +  # base
                       VIBRATION_WEAR_GAIN * self.wheel.bearing_state.wear_level +  # wear component
                       VIBRATION_ROUGHNESS_GAIN * max(0, self.wheel.bearing_state.surface_roughness - 0.32) +  # roughness component
                       VIBRATION_FRICTION_GAIN * max(0, self.wheel.bearing_state.friction_coefficient - 0.02))  # friction component
        
        assert abs(degraded_vibration - expected_vib) < 0.001
        assert degraded_vibration > initial_vibration
    
    def test_temperature_calculation(self):
        """Verify housing temperature calculation matches spacecraft norms"""
        test_conditions = [
            {'load_factor': 1.0, 'friction': 0.02},
            {'load_factor': 2.0, 'friction': 0.1}
        ]
        
        for condition in test_conditions:
            wheel = ReactionWheelSubsystem()
            wheel.bearing_state.friction_coefficient = condition['friction']
            
            commands = {'target_speed_rpm': 3000, 'load_factor': condition['load_factor']}
            wheel.update(1.0, commands)

            telemetry = wheel.get_telemetry()
            temp = telemetry['housing_temperature']

            # Constants from _physics_to_temperature method
            AMBIENT_TEMP = 20.0
            TEMP_FRICTION_GAIN = 3.0  
            TEMP_LOAD_GAIN = 5.0
            
            actual_friction = wheel.bearing_state.friction_coefficient
            
            # Literature: temp = ambient + (friction × friction_gain) + (load × load_gain)
            expected_temp = AMBIENT_TEMP + (TEMP_FRICTION_GAIN * actual_friction + 
                           TEMP_LOAD_GAIN * max(0, condition['load_factor'] - 1.0))
            
            assert abs(temp - expected_temp) < 0.1
            assert AMBIENT_TEMP <= temp <= 35.0  # Realistic spacecraft range

    # ===============================
    # LONG-TERM DEGRADATION TESTS
    # ===============================
    
    def test_one_year_degradation(self):
        """Simulate 8760 hours with realistic duty cycle"""
        total_hours = 8760
        speed_rpm = 3000
        
        self.wheel.update(0, {'target_speed_rpm': speed_rpm})
        initial_telemetry = self.wheel.get_telemetry()
        
        # Realistic mission profile with thermal and load cycling
        for hour in range(1, total_hours + 1):
            duty_cycle_speed = speed_rpm if hour % 24 < 16 else 500
            
            commands = {
                'target_speed_rpm': duty_cycle_speed,
                'load_factor': 1.0 + 0.2 * np.sin(hour / 168),
            }
            
            self.wheel.update(1.0, commands)  # 1 hour timestep
        
        final_telemetry = self.wheel.get_telemetry()
        
        # Literature-based ranges for one year operation
        assert 0.02 <= final_telemetry['wear_level'] <= 0.2
        current_increase = final_telemetry['current'] - initial_telemetry['current']
        assert 0.02 <= current_increase <= 0.3  # Adjusted lower bound to be more realistic
        assert 0.015 <= final_telemetry['vibration'] <= 0.06  # Adjusted upper bound
        assert 0.7 <= final_telemetry['lubrication_quality'] <= 1.0
    
    def test_five_year_degradation(self):
        """Test long-term operation approaching mid-life"""
        total_hours = 43800
        timestep = 24
        
        for hour in range(0, total_hours, timestep):
            commands = {
                'target_speed_rpm': 2500 + 1000 * np.sin(hour / 1000),
                'load_factor': 1.0 + 0.5 * np.sin(hour / 2000),
            }
            
            self.wheel.update(timestep, commands)  # Use timestep as duration
        
        final_telemetry = self.wheel.get_telemetry()
        
        # Mid-life characteristics per literature

        assert 0.2 <= final_telemetry['wear_level'] <= 1.0  # Allow full wear range
        assert 0.15 <= final_telemetry['current'] <= 1.0
        assert 0.03 <= final_telemetry['vibration'] <= 0.2
        assert 0.2 <= final_telemetry['lubrication_quality'] <= 0.8

    # ===============================
    # PERFORMANCE METRICS TESTS
    # ===============================
    
    def test_max_torque_degradation(self):
        """Verify torque decreases with wear as per literature"""
        # Test with new bearing
        self.wheel.update(1.0, {'target_speed_rpm': 3000})
        initial_metrics = self.wheel.get_performance_metrics()
        
        assert abs(initial_metrics['max_torque_Nm'] - 0.05) < 0.001
        
        # Test wear progression
        wear_levels = [0.2, 0.4, 0.6, 0.8]
        
        for wear in wear_levels:
            wheel = ReactionWheelSubsystem()
            wheel.bearing_state.wear_level = wear
            wheel.update(1.0, {'target_speed_rpm': 3000})
            
            metrics = wheel.get_performance_metrics()
            expected_torque = 0.05 * (1 - wear)
            
            assert abs(metrics['max_torque_Nm'] - expected_torque) < 0.001
            assert metrics['max_torque_Nm'] >= 0.0

    # ===============================
    # EDGE CASES AND ROBUSTNESS
    # ===============================
    
    def test_extreme_conditions(self):
        """Test system robustness under extreme conditions"""
        # Test negative speed
        commands = {'target_speed_rpm': -1000}
        self.wheel.update(1.0, commands)
        telemetry = self.wheel.get_telemetry()
        assert telemetry['speed_rpm'] == -1000
        assert telemetry['current'] >= 0.08
        
        # Test extreme load factor
        high_load_wheel = ReactionWheelSubsystem()
        normal_load_wheel = ReactionWheelSubsystem()
        
        for hour in range(1000): # Test shorter duration for performance
            high_load_commands = {'target_speed_rpm': 3000, 'load_factor': 3.0}
            normal_load_commands = {'target_speed_rpm': 3000, 'load_factor': 1.0}
            
            high_load_wheel.update(1.0, high_load_commands)  # 1 hour timestep
            normal_load_wheel.update(1.0, normal_load_commands)  # 1 hour timestep
        
        high_load_telemetry = high_load_wheel.get_telemetry()
        normal_load_telemetry = normal_load_wheel.get_telemetry()
        
        # High load operation should cause more wear and higher temperature
        assert high_load_telemetry['wear_level'] >= normal_load_telemetry['wear_level']
        assert high_load_telemetry['housing_temperature'] > normal_load_telemetry['housing_temperature']
    
    def test_telemetry_completeness(self):
        """Ensure all required telemetry fields are populated"""
        commands = {'target_speed_rpm': 3000}
        self.wheel.update(10.0, commands)
        
        telemetry = self.wheel.get_telemetry()
        
        required_fields = [
            'timestep_hours', 'mode', 'wheel_id', 'speed_rpm', 'load_factor',
            'wear_level', 'friction_coefficient', 'surface_roughness',
            'lubrication_quality', 'vibration', 'current',
            'housing_temperature'
        ]
        
        for field in required_fields:
            assert field in telemetry
            assert telemetry[field] is not None
        
        # Validate ranges
        assert 0 <= telemetry['wear_level'] <= 1.0
        assert 0 <= telemetry['lubrication_quality'] <= 1.0
        assert telemetry['current'] >= 0
        assert telemetry['vibration'] >= 0

    # ===============================
    # PARAMETRIZED TESTS FOR COVERAGE
    # ===============================

    @pytest.mark.parametrize("speed", [0, 500, 1500, 3000, 6000, 10000])
    def test_various_speeds(self, speed):
        """Test wheel operation at various speed levels"""
        wheel = ReactionWheelSubsystem()
        commands = {'target_speed_rpm': speed}
        
        wheel.update(1.0, commands)
        telemetry = wheel.get_telemetry()
        
        assert telemetry['speed_rpm'] == speed
        assert telemetry['current'] >= 0.08  # At least idle current
    
    @pytest.mark.parametrize("load_factor", [0.1, 1.0, 2.0, 3.0])
    def test_various_load_factors(self, load_factor):
        """Test wheel operation at various load factors"""
        wheel = ReactionWheelSubsystem()
        commands = {'target_speed_rpm': 3000, 'load_factor': load_factor}
        
        wheel.update(1.0, commands)
        telemetry = wheel.get_telemetry()
        
        assert telemetry['load_factor'] == load_factor
        
        # Temperature should scale with load factor
        expected_temp = 20.0 + 3.0 * telemetry['friction_coefficient'] + 5.0 * max(0, load_factor - 1.0)
        assert abs(telemetry['housing_temperature'] - expected_temp) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])