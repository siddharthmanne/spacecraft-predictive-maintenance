"""
Test BearingDegradationModel
---------------------------
Unit and progression tests for the bearing degradation physics model.
"""
import unittest
from src.bearing_degradation import BearingDegradationModel, BearingState

class TestBearingDegradationModel(unittest.TestCase):
    def setUp(self):
        self.model = BearingDegradationModel()
        self.initial_state = BearingState()
        self.conditions = {
            'speed_rpm': 3500,
            'load_factor': 1.0,
            'spacecraft_payload': 1200,
            'maneuvering': True
        }

    def test_initial_state(self):
        # Check initial values
        self.assertAlmostEqual(self.initial_state.wear_level, 0.0)
        self.assertAlmostEqual(self.initial_state.lubrication_quality, 1.0)
        self.assertAlmostEqual(self.initial_state.friction_coefficient, 0.02)
        self.assertAlmostEqual(self.initial_state.surface_roughness, 0.32)
        self.assertAlmostEqual(self.initial_state.bearing_temperature, 15.0)

    def test_one_hour_update(self):
        # After one hour, wear should increase, lube should decrease
        state = self.model.update_bearing_state_one_hour(self.initial_state, self.conditions)
        self.assertGreater(state.wear_level, 0.0)
        self.assertLess(state.lubrication_quality, 1.0)
        self.assertGreaterEqual(state.friction_coefficient, 0.02)
        self.assertGreaterEqual(state.surface_roughness, 0.32)

    def test_long_term_progression(self):
        # Simulate 10 years
        state = self.initial_state
        for _ in range(10 * 365 * 24):
            state = self.model.update_bearing_state_one_hour(state, self.conditions)
        self.assertGreaterEqual(state.wear_level, 0.9)
        self.assertLessEqual(state.lubrication_quality, 0.6)  # More realistic expectation for spacecraft bearings
        self.assertGreaterEqual(state.surface_roughness, 3.0)
        self.assertGreaterEqual(state.friction_coefficient, 0.1)

    def test_checkpoint_values(self):
        # Check specific time checkpoints
        checkpoints = [0, 30*24, 90*24, 180*24, 365*24, 3*365*24, 5*365*24, 10*365*24]
        state = self.initial_state
        results = []
        for hour in range(1, checkpoints[-1]+1):
            state = self.model.update_bearing_state_one_hour(state, self.conditions)
            if hour in checkpoints:
                results.append((hour, state))
        # Print results for manual inspection
        for hour, s in results:
            print(f"Hour {hour}: Wear={s.wear_level:.3f}, Lube={s.lubrication_quality:.3f}, Friction={s.friction_coefficient:.3f}, Roughness={s.surface_roughness:.2f}")

if __name__ == "__main__":
    unittest.main()
