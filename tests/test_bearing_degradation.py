"""
Module: test_bearing_degradation.py

Description:
This module provides a comprehensive pytest suite for validating the functionality and accuracy
of the bearing_degradation.py module. It includes unit tests for state initialization, wear progression,
lubrication loss, friction coefficient calculations, caching mechanisms, edge cases, and integration scenarios.
The tests ensure correctness, physical plausibility, performance, and robustness of the bearing degradation simulator.
"""

import sys
import os
# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import patch
import time
from src.bearing_degradation import BearingDegradationModel, BearingState


@pytest.fixture
def model():
    """Create a fresh model instance for each test"""
    return BearingDegradationModel()

@pytest.fixture
def new_bearing():
    """Create a new bearing state for testing"""
    return BearingState()

@pytest.fixture
def worn_bearing():
    """Create a worn bearing state for testing"""
    return BearingState(
        wear_level=0.6,
        friction_coefficient=0.08,
        surface_roughness=0.8,
        lubrication_quality=0.4,
        bearing_temperature=27.0
    )


class TestBearingState:
    """Test the BearingState dataclass"""
    
    def test_bearing_state_initialization(self):
        """Test default initialization values"""
        state = BearingState()
        assert state.wear_level == 0.0
        assert state.friction_coefficient == 0.02
        assert state.surface_roughness == 0.32
        assert state.lubrication_quality == 1.0
        assert state.bearing_temperature == 20.0
    
    def test_bearing_state_custom_values(self):
        """Test custom initialization values"""
        state = BearingState(
            wear_level=0.5,
            friction_coefficient=0.08,
            surface_roughness=1.2,
            lubrication_quality=0.6,
            bearing_temperature=25.0
        )
        assert state.wear_level == 0.5
        assert state.friction_coefficient == 0.08
        assert state.surface_roughness == 1.2
        assert state.lubrication_quality == 0.6
        assert state.bearing_temperature == 25.0



class TestUpdateBearingState:
    """Test the core update_bearing_state function"""
    
    def test_no_wear_when_stopped(self, model, new_bearing):
        """Bearing shouldn't wear when not rotating"""
        conditions = {
            'speed_rpm': 0.0,
            'load_factor': 1.0
        }
        
        new_state = model.update_bearing_state(new_bearing, conditions, 10.0)
        
        # Wear level should remain the same when stopped
        assert new_state.wear_level == new_bearing.wear_level
        # But lubrication and temperature history may still change
        assert new_state.lubrication_quality <= new_bearing.lubrication_quality
    
    def test_wear_progression_with_rotation(self, model, new_bearing):
        """Bearing should wear when rotating"""
        conditions = {
            'speed_rpm': 1000.0,
            'load_factor': 1.0
        }
        
        new_state = model.update_bearing_state(new_bearing, conditions, 1.0)
        
        # New wear levels should increase
        assert new_state.wear_level > new_bearing.wear_level
        # Surface roughness may decrease during running-in (early wear), then increase later
        if new_bearing.wear_level < model.physics_constants['running_in_duration']:
            assert new_state.surface_roughness <= new_bearing.surface_roughness
        else:
            assert new_state.surface_roughness >= new_bearing.surface_roughness
        # Friction should not decrease below base, but may increase with wear
        assert new_state.friction_coefficient >= new_bearing.friction_coefficient
    
    def test_temperature_effects(self, model, new_bearing):
        """Higher temperature should increase wear rate"""
        # Set up two bearings with different temperatures
        base_bearing = BearingState()
        hot_bearing = BearingState(bearing_temperature=60.0)
        conditions = {
            'speed_rpm': 1000.0,
            'load_factor': 1.0
        }
        base_state = model.update_bearing_state(base_bearing, conditions, 1000)
        hot_state = model.update_bearing_state(hot_bearing, conditions, 1000)
        assert hot_state.wear_level > base_state.wear_level
        assert hot_state.lubrication_quality < base_state.lubrication_quality
    
    def test_load_factor_effects(self, model, new_bearing):
        """Higher load should increase wear rate"""
        light_conditions = {
            'speed_rpm': 1000.0,
            'load_factor': 0.5
        }
        
        heavy_conditions = {
            'speed_rpm': 1000.0,
            'load_factor': 2.0
        }
        
        light_state = model.update_bearing_state(new_bearing, light_conditions, 1.0)
        heavy_state = model.update_bearing_state(new_bearing, heavy_conditions, 1.0)
        
        assert heavy_state.wear_level > light_state.wear_level
    
    def test_wear_level_bounds(self, model, worn_bearing):
        """Wear level should be capped at 1.0"""
        # Create a nearly failed bearing
        nearly_failed = BearingState(wear_level=0.99, bearing_temperature=60.0)
        extreme_conditions = {
            'speed_rpm': 10000.0,
            'load_factor': 5.0
        }
        new_state = model.update_bearing_state(nearly_failed, extreme_conditions, 1000.0)
        assert new_state.wear_level <= 1.0
    
    def test_lubrication_quality_bounds(self, model, worn_bearing):
        """Lubrication quality should be bounded between 0.0 and 1.0"""
        # Create a bearing with poor lubrication
        poor_lube = BearingState(lubrication_quality=0.01, bearing_temperature=60.0)
        extreme_conditions = {
            'speed_rpm': 1000.0,
            'load_factor': 1.0
        }
        new_state = model.update_bearing_state(poor_lube, extreme_conditions, 1000.0)
        assert new_state.lubrication_quality >= 0.0
    
    def test_missing_conditions_defaults(self, model, new_bearing):
        """Should handle missing operating conditions gracefully"""
        # Empty conditions dictionary
        empty_conditions = {}
        new_state = model.update_bearing_state(new_bearing, empty_conditions, 1.0)
        # Should not crash and should use defaults
        assert new_state.wear_level == new_bearing.wear_level  # No speed = no wear
        assert new_state is not None

class TestWearAcceleration:
    """Test the _calculate_wear_acceleration function"""
    
    def test_new_bearing_acceleration(self, model, new_bearing):
        """New bearing should have minimal acceleration"""
        wear_acceleration = model._calculate_wear_acceleration(new_bearing)
        assert 0.9 <= wear_acceleration <= 1.5

    def test_worn_bearing_acceleration(self, model, worn_bearing):
        """Worn bearing should have higher acceleration"""
        wear_acceleration = model._calculate_wear_acceleration(worn_bearing)
        assert wear_acceleration > 2.0

    def test_acceleration_bounds(self, model):
        """Acceleration should be capped at 50.0 (max_debris_acceleration)"""
        failed_bearing = BearingState(
            wear_level=0.99,
            lubrication_quality=0.01,
            surface_roughness=5.0
        )
        acceleration = model._calculate_wear_acceleration(failed_bearing)
        assert acceleration <= 50.0

class TestLubricationLoss:
    """Test the _calculate_lubrication_loss function"""
    
    def test_base_lubrication_loss(self, model, new_bearing):
        """Test basic lubrication loss calculation"""
        loss = model._calculate_lubrication_loss(new_bearing, new_bearing.bearing_temperature, 1.0)
        assert loss > 0.0
        assert loss < 0.1  # Should be small for new bearing

    def test_temperature_acceleration(self, model, new_bearing):
        """High temperature should accelerate lubrication loss"""
        hot_bearing = BearingState(bearing_temperature=60.0)
        normal_loss = model._calculate_lubrication_loss(new_bearing, new_bearing.bearing_temperature, 1.0)
        hot_loss = model._calculate_lubrication_loss(hot_bearing, hot_bearing.bearing_temperature, 1.0)
        assert hot_loss > normal_loss

    def test_wear_contamination_effect(self, model, worn_bearing):
        """Worn bearing should have accelerated lubrication loss"""
        new_loss = model._calculate_lubrication_loss(BearingState(), 20.0, 1.0)
        worn_loss = model._calculate_lubrication_loss(worn_bearing, worn_bearing.bearing_temperature, 1.0)
        assert worn_loss > new_loss

class TestFrictionCalculation:
    """Test the _calculate_friction_coefficient function"""
    
    def test_new_bearing_friction(self, model, new_bearing):
        """New bearing should have low friction"""
        friction = model._calculate_friction_coefficient(new_bearing, 100.0)
        
        assert 0.015 <= friction <= 0.05  # Should be at, just below, or above base friction after just 100 hours of operation
    
    def test_worn_bearing_friction(self, model, worn_bearing):
        """Worn bearing should have higher friction"""
        new_friction = model._calculate_friction_coefficient(BearingState(), 0.0)
        worn_friction = model._calculate_friction_coefficient(worn_bearing, 0.0)
        
        assert worn_friction > new_friction
    
    def test_friction_bounds(self, model):
        """Friction should be capped at 0.20"""
        # Create extremely degraded bearing
        failed_bearing = BearingState(
            wear_level=0.99,
            lubrication_quality=0.01,
            surface_roughness=10.0,
            friction_coefficient=0.2
        )
        
        friction = model._calculate_friction_coefficient(failed_bearing, 0.1)
        assert friction <= 0.20

class TestPhysicalProperties:
    """Test the get_physical_properties function"""
    
    def test_properties_format(self, model, new_bearing):
        """Properties should return correct dictionary format"""
        props = model.get_physical_properties(new_bearing)
        
        required_keys = ['wear_level', 'surface_roughness', 'friction_coefficient', 'lubrication_quality']
        
        assert all(key in props for key in required_keys)
        assert all(isinstance(props[key], float) for key in required_keys)
    
    def test_properties_values(self, model, worn_bearing):
        """Properties should match bearing state values"""
        props = model.get_physical_properties(worn_bearing)
        
        assert props['wear_level'] == worn_bearing.wear_level
        assert props['surface_roughness'] == worn_bearing.surface_roughness
        assert props['friction_coefficient'] == worn_bearing.friction_coefficient
        assert props['lubrication_quality'] == worn_bearing.lubrication_quality

class TestWearPrediction:
    """Test the predict_wear_progression function"""
    
    def test_prediction_format(self, model, new_bearing):
        """Prediction should return correct dictionary format"""
        conditions = {'load_factor': 1.0}
        prediction = model.predict_wear_progression(new_bearing, conditions, 100.0)
        
        required_keys = ['projected_wear_level', 'time_to_failure_hours', 'wear_rate_per_hour']
        
        assert all(key in prediction for key in required_keys)
        assert all(isinstance(prediction[key], float) for key in required_keys)
    
    def test_prediction_bounds(self, model, new_bearing):
        """Projected wear should be bounded properly"""
        hotBearing = BearingState(bearing_temperature=60.0)
        conditions = {'load_factor': 5.0}
        prediction = model.predict_wear_progression(hotBearing, conditions, 10000.0)
        
        assert prediction['projected_wear_level'] <= 1.0
        assert prediction['time_to_failure_hours'] > 0.0
        assert prediction['wear_rate_per_hour'] > 0.0

class TestCaching:
    """Test the caching functionality"""
    
    def test_cache_key_generation(self, model, new_bearing):
        """Cache keys should be consistent"""
        conditions = {'speed_rpm': 1000.0, 'load_factor': 1.0}
        
        key1 = model._generate_cache_key(new_bearing, conditions, 1.0)
        key2 = model._generate_cache_key(new_bearing, conditions, 1.0)
        
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 16  # MD5 hash truncated to 16 chars
    
    def test_cache_functionality(self, model, new_bearing):
        """Caching should improve performance"""
        conditions = {'speed_rpm': 1000.0, 'load_factor': 1.0}
        
        # First call should miss cache
        start_time = time.time()
        state1 = model.update_bearing_state(new_bearing, conditions, 1.0)
        first_call_time = time.time() - start_time
        
        # Second call should hit cache
        start_time = time.time()
        state2 = model.update_bearing_state(new_bearing, conditions, 1.0)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert state1.wear_level == state2.wear_level
        assert state1.friction_coefficient == state2.friction_coefficient
        
        # Cache should be working
        assert model.cache_hits > 0
        assert model.cache_misses > 0
    
    def test_cache_statistics(self, model, new_bearing):
        """Cache statistics should be tracked correctly"""
        initial_stats = model.get_cache_statistics()
        
        conditions = {'speed_rpm': 1000.0, 'load_factor': 1.0}
        
        # Make some calls
        model.update_bearing_state(new_bearing, conditions, 1.0)  # Cache miss
        model.update_bearing_state(new_bearing, conditions, 1.0)  # Cache hit
        
        final_stats = model.get_cache_statistics()
        
        assert final_stats['cache_misses'] > initial_stats['cache_misses']
        assert final_stats['cache_hits'] > initial_stats['cache_hits']
        assert 'hit_rate_percent' in final_stats
        assert 'cached_items' in final_stats

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_negative_time_delta(self, model, new_bearing):
        """Should handle negative time delta gracefully"""
        conditions = {'speed_rpm': 1000.0, 'load_factor': 1.0}
        try:
            state = model.update_bearing_state(new_bearing, conditions, -1.0)
            assert state.wear_level >= new_bearing.wear_level
        except ValueError:
            pass
    
    def test_extreme_conditions(self, model, new_bearing):
        """Should handle extreme operating conditions"""
        extreme_bearing = BearingState(bearing_temperature=200.0)
        extreme_conditions = {
            'speed_rpm': 100000.0,
            'load_factor': 100.0
        }
        state = model.update_bearing_state(extreme_bearing, extreme_conditions, 1.0)
        assert 0.0 <= state.wear_level <= 1.0
        assert 0.0 <= state.lubrication_quality <= 1.0
        assert state.friction_coefficient <= 0.20
        assert state.surface_roughness >= 0.32
    
    def test_zero_time_delta(self, model, new_bearing):
        """Should handle zero time delta"""
        conditions = {'speed_rpm': 1000.0, 'load_factor': 1.0}
        state = model.update_bearing_state(new_bearing, conditions, 0.0)
        assert state.wear_level == new_bearing.wear_level

class TestPhysicsConstants:
    """Test physics constants are reasonable"""
    
    def test_constants_exist(self, model):
        """All required physics constants should exist"""
        required_constants = [
            'wear_rate_base',
            'load_stress_exponent',
            'lubrication_depletion_rate',
            'critical_temperature_threshold',
            'lubrication_temp_factor',
            'surface_roughness_factor',
            'friction_increase_factor',
            'reference_temperature'
        ]
        for constant in required_constants:
            assert constant in model.physics_constants
            assert isinstance(model.physics_constants[constant], (int, float))
    
    def test_constants_reasonable_values(self, model):
        """Physics constants should have reasonable values"""
        constants = model.physics_constants
        
        assert constants['wear_rate_base'] > 0
        assert constants['critical_temperature_threshold'] > 0
        assert constants['reference_temperature'] > 0
        assert 0 < constants['load_stress_exponent'] < 5
        assert constants['lubrication_depletion_rate'] > 0

# Integration tests
class TestIntegration:
    """Integration tests for full system behavior"""
    
    def test_bearing_lifecycle(self, model):
        """Test complete bearing lifecycle from new to failure"""
        bearing = BearingState()
        conditions = {
            'speed_rpm': 3000.0,
            'load_factor': 1.2
        }
        
        wear_progression = []
        friction_progression = []
        
        # Simulate ~6 years of operation
        for hour in range(52596):
            bearing = model.update_bearing_state(bearing, conditions, 1.0)
            
            if hour % 2880 == 0:  # Record every ~ 4 months
                wear_progression.append(bearing.wear_level)
                friction_progression.append(bearing.friction_coefficient)
        
        # Verify progressive degradation
        assert all(wear_progression[i] <= wear_progression[i+1] 
                  for i in range(len(wear_progression)-1))
        
        # Final state should be significantly degraded after 6 years, since some spacecraft bearings have operational capacity of 3-5 years without maintenance
        assert 0.7 < bearing.wear_level <= 1.0
        assert 0.1 < bearing.friction_coefficient <= 0.2
        assert 0.05 <= bearing.lubrication_quality <= 0.7
    
    def test_different_operating_profiles(self, model):
        """Test different operating profiles produce different results"""
        profiles = ['light', 'normal', 'heavy']
        profile_details = { 'light': {'bearing_temperature': 20.0, 'conditions': {'speed_rpm': 1000.0, 'load_factor': 0.8}},
                           'normal': {'bearing_temperature': 25.0, 'conditions': {'speed_rpm': 3000.0, 'load_factor': 1.2}},
                           'heavy': {'bearing_temperature': 30.0, 'conditions': {'speed_rpm': 5000.0, 'load_factor': 2.0}}}
        
        results = []
        
        for profile in profiles:
            bearing = BearingState(bearing_temperature=profile_details[profile]['bearing_temperature'])
            # Run for 100 hours
            for _ in range(100):
                bearing = model.update_bearing_state(bearing, profile_details[profile]['conditions'], 1.0)
            results.append(bearing.wear_level)
        
        # Heavy duty should cause more wear than normal, normal more than light
        assert results[2] > results[1] > results[0]

# Performance tests
class TestPerformance:
    """Performance and scalability tests"""
    
    def test_cache_performance(self, model):
        """Caching should significantly improve repeated calculations"""
        bearing = BearingState()
        conditions = {'speed_rpm': 1000.0, 'load_factor': 1.0}
        
        # Warm up cache
        model.update_bearing_state(bearing, conditions, 1.0)
        
        # Time repeated calls
        start_time = time.time()
        for _ in range(100):
            model.update_bearing_state(bearing, conditions, 1.0)
        elapsed_time = time.time() - start_time
        
        # Should complete quickly with caching
        assert elapsed_time < 0.1  # Less than 100ms for 100 calls
        
        # Cache hit rate should be high
        stats = model.get_cache_statistics()
        assert stats['hit_rate_percent'] > 90
