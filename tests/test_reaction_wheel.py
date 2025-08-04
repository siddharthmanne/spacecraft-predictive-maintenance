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
        assert telemetry['mission_time_hours'] == 1.0
    
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
            assert 0.08 <= telemetry['motor_current'] <= 0.5  # Literature: 0.05-0.12A idle 
            assert 0.01 <= telemetry['measured_vibration'] <= 0.1  # Literature: 0.01-0.1g healthy wheels
            assert 20.0 <= telemetry['housing_temperature'] <= 25.0  # Spacecraft typical
    
    def test_zero_rpm_operation(self):
        """Test wheel behavior when stopped"""
        commands = {'target_speed_rpm': 0}
        initial_wear = self.wheel.bearing_state.wear_level
        initial_lubrication_quality = self.wheel.bearing_state.lubrication_quality
        initial_surface_roughness = self.wheel.bearing_state.surface_roughness
        
        self.wheel.update(1.0, commands)
        telemetry = self.wheel.get_telemetry()
        
        assert abs(telemetry['motor_current'] - 0.08) < 0.01  # Only idle current
        assert telemetry['bearing_wear_level'] == initial_wear  # No wear when stopped
        assert telemetry['bearing_lubrication_quality'] < initial_lubrication_quality # Lubrication quality can decrease at 0 RPM due to oil migration from temperature gradients and capillary forces
        assert telemetry['bearing_surface_roughness'] == initial_surface_roughness # No surface roughness change (no debris generation)

    # ===============================
    # PHYSICS VALIDATION TESTS
    # ===============================
    
    def test_current_draw_scaling(self):
        """Verify current increases with friction and speed per literature"""
        # Test with new bearing
        commands = {'target_speed_rpm': 3000}
        self.wheel.update(1.0, commands)
        new_bearing_current = self.wheel.get_telemetry()['motor_current']
        
        # Simulate significant wear
        self.wheel.bearing_state.wear_level = 0.5
        self.wheel.bearing_state.friction_coefficient = 0.1
        
        self.wheel.update(2.0, commands)
        worn_bearing_current = self.wheel.get_telemetry()['motor_current']
        
        assert worn_bearing_current > new_bearing_current

        # Get the actual physics state the model computed
        physics = self.wheel.bearing_model.get_physical_properties(self.wheel.bearing_state)
        actual_friction = physics['friction_coefficient']
        
        # Calculate expected using class constants and updated physics values
        expected_current = self.wheel.IDLE_CURRENT + (self.wheel.CURRENT_GAIN * actual_friction * 3000 / 1000.0)
        assert abs(worn_bearing_current - expected_current) < 0.01
    
    def test_vibration_progression(self):
        """Test vibration increases with bearing degradation per literature"""
        # New bearing baseline
        commands = {'target_speed_rpm': 3000}
        self.wheel.update(1.0, commands)
        initial_vibration = self.wheel.get_telemetry()['measured_vibration']
        
        assert abs(initial_vibration - self.wheel.BASE_VIBRATION) < 0.005  # Should be near baseline
        
        # Simulate degradation by setting degraded bearing states
        self.wheel.bearing_state.wear_level = 0.3
        self.wheel.bearing_state.surface_roughness = 0.5
        self.wheel.bearing_state.friction_coefficient = 0.08
        
        self.wheel.update(2.0, commands)
        degraded_vibration = self.wheel.get_telemetry()['measured_vibration']

        # Get the actual physics state the model computed
        physics = self.wheel.bearing_model.get_physical_properties(self.wheel.bearing_state)
        
        # Literature: vibration = base + wear + roughness + friction components
        expected_vib = (self.wheel.BASE_VIBRATION +  # base
                       self.wheel.VIBRATION_WEAR_GAIN * physics['wear_level'] +  # wear component
                       self.wheel.VIBRATION_ROUGHNESS_GAIN * max(0, physics['surface_roughness'] - self.wheel.ROUGHNESS_BASELINE) +  # roughness component
                       self.wheel.VIBRATION_FRICTION_GAIN * (physics['friction_coefficient'] - self.wheel.FRICTION_BASELINE))  # friction component
        
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

            # Get the actual physics state the model computed after update
            physics = wheel.bearing_model.get_physical_properties(wheel.bearing_state)
            actual_friction = physics['friction_coefficient']
            
            # Literature: temp = ambient + (friction × load × gain)
            expected_temp = self.wheel.AMBIENT_TEMP + (self.wheel.TEMP_FRICTION_GAIN * actual_friction * condition['load_factor'])
            
            assert abs(temp - expected_temp) < 0.1
            assert self.wheel.AMBIENT_TEMP <= temp <= 35.0  # Realistic spacecraft range

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
                'bearing_temperature': 18.0 + 4 * np.sin(hour / 24),
                'timestep_hours': 1.0
            }
            
            self.wheel.update(hour, commands)
        
        final_telemetry = self.wheel.get_telemetry()
        
        # Literature-based ranges for one year operation
        assert 0.02 <= final_telemetry['bearing_wear_level'] <= 0.2
        current_increase = final_telemetry['motor_current'] - initial_telemetry['motor_current']
        assert 0.05 <= current_increase <= 0.3
        assert 0.015 <= final_telemetry['measured_vibration'] <= 0.05
        assert 0.7 <= final_telemetry['bearing_lubrication_quality'] <= 1.0
    
    def test_five_year_degradation(self):
        """Test long-term operation approaching mid-life"""
        total_hours = 43800
        timestep = 24
        
        for hour in range(0, total_hours, timestep):
            commands = {
                'target_speed_rpm': 2500 + 1000 * np.sin(hour / 1000),
                'load_factor': 1.0 + 0.5 * np.sin(hour / 2000),
                'bearing_temperature': 20.0 + 10 * np.sin(hour / 8760),
                'timestep_hours': timestep
            }
            
            self.wheel.update(hour, commands)
        
        final_telemetry = self.wheel.get_telemetry()
        
        # Mid-life characteristics per literature

        assert 0.2 <= final_telemetry['bearing_wear_level'] <= 0.7
        assert 0.15 <= final_telemetry['motor_current'] <= 1.0
        assert 0.03 <= final_telemetry['measured_vibration'] <= 0.2
        assert 0.2 <= final_telemetry['bearing_lubrication_quality'] <= 0.8

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
        assert telemetry['motor_current'] >= 0.08
        
        # Test extreme temperature
        hot_wheel = ReactionWheelSubsystem()
        normal_wheel = ReactionWheelSubsystem()
        
        for hour in range(9766): # Test 1 year of operation
            hot_commands = {'target_speed_rpm': 3000, 'bearing_temperature': 80.0}
            normal_commands = {'target_speed_rpm': 3000, 'bearing_temperature': 20.0}
            
            hot_wheel.update(hour, hot_commands)
            normal_wheel.update(hour, normal_commands)
        
        hot_telemetry = hot_wheel.get_telemetry()
        normal_telemetry = normal_wheel.get_telemetry()
        
        # Hot operation should degrade lubrication faster
        assert hot_telemetry['bearing_lubrication_quality'] < normal_telemetry['bearing_lubrication_quality']
    
    def test_telemetry_completeness(self):
        """Ensure all required telemetry fields are populated"""
        commands = {'target_speed_rpm': 3000}
        self.wheel.update(10.0, commands)
        
        telemetry = self.wheel.get_telemetry()
        
        required_fields = [
            'mission_time_hours', 'mode', 'wheel_id', 'speed_rpm', 'load_factor',
            'bearing_wear_level', 'bearing_friction_coeff', 'bearing_surface_roughness',
            'bearing_lubrication_quality', 'measured_vibration', 'motor_current',
            'housing_temperature'
        ]
        
        for field in required_fields:
            assert field in telemetry
            assert telemetry[field] is not None
        
        # Validate ranges
        assert 0 <= telemetry['bearing_wear_level'] <= 1.0
        assert 0 <= telemetry['bearing_lubrication_quality'] <= 1.0
        assert telemetry['motor_current'] >= 0
        assert telemetry['measured_vibration'] >= 0

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
        assert telemetry['motor_current'] >= 0.08  # At least idle current
    
    @pytest.mark.parametrize("load_factor", [0.1, 1.0, 2.0, 3.0])
    def test_various_load_factors(self, load_factor):
        """Test wheel operation at various load factors"""
        wheel = ReactionWheelSubsystem()
        commands = {'target_speed_rpm': 3000, 'load_factor': load_factor}
        
        wheel.update(1.0, commands)
        telemetry = wheel.get_telemetry()
        
        assert telemetry['load_factor'] == load_factor
        
        # Temperature should scale with load factor
        expected_temp = 20.0 + 3.0 * telemetry['bearing_friction_coeff'] * load_factor
        assert abs(telemetry['housing_temperature'] - expected_temp) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
