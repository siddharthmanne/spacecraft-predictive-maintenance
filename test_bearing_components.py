"""
Focused Bearing Degradation Component Test

This test isolates individual calculation components to identify which functions
are causing unexpected behavior in the bearing degradation model.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.bearing_degradation import BearingDegradationModel, BearingState
from typing import Dict, List
import sys
import os

class BearingComponentTester:
    def __init__(self):
        self.model = BearingDegradationModel()
    
    def test_individual_components(self):
        """Test each calculation component independently"""
        print("=== Individual Component Testing ===\n")
        
        # Test 1: Wear acceleration function
        print("1. Testing Wear Acceleration Function:")
        self._test_wear_acceleration()
        
        # Test 2: Lubrication loss function  
        print("\n2. Testing Lubrication Loss Function:")
        self._test_lubrication_loss()
        
        # Test 3: Surface roughness function
        print("\n3. Testing Surface Roughness Function:")
        self._test_surface_roughness()
        
        # Test 4: Friction coefficient function
        print("\n4. Testing Friction Coefficient Function:")
        self._test_friction_coefficient()
        
        # Test 5: Bearing load function
        print("\n5. Testing Bearing Load Function:")
        self._test_bearing_load()
        
        # Test 6: Integration test
        print("\n6. Integration Test - Step by Step:")
        self._test_step_by_step_integration()
    
    def _test_wear_acceleration(self):
        """Test how wear acceleration changes with bearing state"""
        print("  Testing wear acceleration across different bearing states...")
        
        # Test across wear levels
        wear_levels = np.linspace(0, 0.95, 20)
        lube_qualities = [1.0, 0.5, 0.1]  # Good, medium, poor lubrication
        roughness_values = [0.32, 2.0, 6.0]  # Low, medium, high roughness
        
        results = []
        
        for wear in wear_levels:
            for lube in lube_qualities:
                for rough in roughness_values:
                    state = BearingState(
                        wear_level=wear,
                        lubrication_quality=lube,
                        surface_roughness=rough
                    )
                    accel = self.model._calculate_wear_acceleration(state)
                    results.append({
                        'wear': wear,
                        'lube': lube, 
                        'roughness': rough,
                        'acceleration': accel
                    })
        
        # Analyze results
        max_accel = max(r['acceleration'] for r in results)
        min_accel = min(r['acceleration'] for r in results)
        
        print(f"    Acceleration range: {min_accel:.2f} to {max_accel:.2f}")
        
        # Check if acceleration increases with wear
        good_lube_results = [r for r in results if r['lube'] == 1.0 and r['roughness'] == 0.32]
        wear_progression = [r['wear'] for r in good_lube_results]
        accel_progression = [r['acceleration'] for r in good_lube_results]
        
        is_increasing = all(accel_progression[i] >= accel_progression[i-1] for i in range(1, len(accel_progression)))
        print(f"    Acceleration increases with wear: {is_increasing}")
        
        if max_accel > 100:
            print(f"    ⚠️  WARNING: Very high acceleration factor ({max_accel:.1f})")
        
        # Plot acceleration vs wear for different conditions
        plt.figure(figsize=(10, 6))
        for lube in lube_qualities:
            wear_vals = [r['wear'] for r in results if r['lube'] == lube and r['roughness'] == 0.32]
            accel_vals = [r['acceleration'] for r in results if r['lube'] == lube and r['roughness'] == 0.32]
            plt.plot(wear_vals, accel_vals, 'o-', label=f'Lube Quality: {lube}')
        
        plt.xlabel('Wear Level')
        plt.ylabel('Wear Acceleration Factor')
        plt.title('Wear Acceleration vs Wear Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('wear_acceleration_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _test_lubrication_loss(self):
        """Test lubrication loss calculation"""
        print("  Testing lubrication loss rates...")
        
        temperatures = np.linspace(20, 80, 20)  # 20°C to 80°C
        wear_levels = [0.0, 0.3, 0.7, 0.95]
        
        results = []
        
        for temp in temperatures:
            for wear in wear_levels:
                state = BearingState(wear_level=wear, bearing_temperature=temp)
                loss_rate = self.model._calculate_lubrication_loss(state, temp)
                results.append({
                    'temperature': temp,
                    'wear': wear,
                    'loss_rate': loss_rate
                })
        
        # Calculate time to complete lubrication loss
        base_loss = min(r['loss_rate'] for r in results if r['wear'] == 0.0 and r['temperature'] == 20)
        max_loss = max(r['loss_rate'] for r in results)
        
        time_to_failure_base = 1.0 / (base_loss * 8760)  # Years at base rate
        time_to_failure_max = 1.0 / (max_loss * 8760)   # Years at max rate
        
        print(f"    Base loss rate: {base_loss:.2e} per hour")
        print(f"    Max loss rate: {max_loss:.2e} per hour")
        print(f"    Time to complete loss (base conditions): {time_to_failure_base:.1f} years")
        print(f"    Time to complete loss (worst conditions): {time_to_failure_max:.1f} years")
        
        if time_to_failure_base > 50:
            print(f"    ⚠️  WARNING: Base lubrication loss too slow ({time_to_failure_base:.1f} years)")
        elif time_to_failure_base < 2:
            print(f"    ⚠️  WARNING: Base lubrication loss too fast ({time_to_failure_base:.1f} years)")
        
        # Plot temperature effect
        plt.figure(figsize=(10, 6))
        for wear in wear_levels:
            temp_vals = [r['temperature'] for r in results if r['wear'] == wear]
            loss_vals = [r['loss_rate'] for r in results if r['wear'] == wear]
            plt.semilogy(temp_vals, loss_vals, 'o-', label=f'Wear: {wear}')
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Lubrication Loss Rate (per hour)')
        plt.title('Lubrication Loss Rate vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lubrication_loss_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _test_surface_roughness(self):
        """Test surface roughness calculation"""
        print("  Testing surface roughness progression...")
        
        # Test different phases of roughness evolution
        wear_levels = np.linspace(0, 0.95, 100)
        wear_increment = 1e-5  # Small increment
        
        results = []
        base_state = BearingState()
        
        for wear in wear_levels:
            state = BearingState(
                wear_level=wear,
                surface_roughness=0.32 + wear * 2.0,  # Assume some progression
                lubrication_quality=max(0.05, 1.0 - wear)
            )
            
            new_roughness = self.model._calculate_surface_roughness(state, wear_increment)
            roughness_change = new_roughness - state.surface_roughness
            
            results.append({
                'wear': wear,
                'current_roughness': state.surface_roughness,
                'new_roughness': new_roughness,
                'roughness_change': roughness_change
            })
        
        # Analyze phases
        running_in_end = self.model.physics_constants['running_in_duration']
        debris_start = self.model.physics_constants['debris_onset_wear']
        contamination_start = self.model.physics_constants['debris_contamination_threshold']
        
        print(f"    Running-in phase: 0 to {running_in_end:.3f} wear")
        print(f"    Normal wear phase: {running_in_end:.3f} to {debris_start:.3f} wear")
        print(f"    Debris phase: {debris_start:.3f} to {contamination_start:.3f} wear")
        print(f"    Contamination phase: {contamination_start:.3f} to 1.0 wear")
        
        # Check for smoothing in running-in phase
        running_in_results = [r for r in results if r['wear'] < running_in_end]
        if running_in_results:
            avg_change_running_in = np.mean([r['roughness_change'] for r in running_in_results])
            print(f"    Average roughness change in running-in: {avg_change_running_in:.2e}")
            if avg_change_running_in > 0:
                print(f"    ⚠️  WARNING: Roughness increasing during running-in (should decrease)")
        
        # Check final roughness
        final_roughness = results[-1]['new_roughness']
        print(f"    Final roughness at 95% wear: {final_roughness:.2f} μm")
        
        if final_roughness < 2.0:
            print(f"    ⚠️  WARNING: Final roughness too low ({final_roughness:.2f} μm)")
        elif final_roughness > 10.0:
            print(f"    ⚠️  WARNING: Final roughness too high ({final_roughness:.2f} μm)")
        
        # Plot roughness progression
        plt.figure(figsize=(10, 6))
        wear_vals = [r['wear'] for r in results]
        rough_vals = [r['new_roughness'] for r in results]
        change_vals = [r['roughness_change'] for r in results]
        
        plt.subplot(2, 1, 1)
        plt.plot(wear_vals, rough_vals, 'g-', linewidth=2)
        plt.axvline(running_in_end, color='blue', linestyle='--', alpha=0.7, label='Running-in end')
        plt.axvline(debris_start, color='orange', linestyle='--', alpha=0.7, label='Debris start')
        plt.axvline(contamination_start, color='red', linestyle='--', alpha=0.7, label='Contamination start')
        plt.ylabel('Surface Roughness (μm)')
        plt.title('Surface Roughness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(wear_vals, change_vals, 'purple', linewidth=2)
        plt.axvline(running_in_end, color='blue', linestyle='--', alpha=0.7)
        plt.axvline(debris_start, color='orange', linestyle='--', alpha=0.7)
        plt.axvline(contamination_start, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Wear Level')
        plt.ylabel('Roughness Change Rate')
        plt.title('Roughness Change Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('surface_roughness_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _test_friction_coefficient(self):
        """Test friction coefficient calculation"""
        print("  Testing friction coefficient calculation...")
        
        # Test across different conditions
        speeds = [0, 100, 1000, 3000, 6000]  # RPM
        loads = [10, 30, 50, 100]  # Newtons
        wear_levels = [0.0, 0.3, 0.7, 0.95]
        
        results = []
        
        for speed in speeds:
            for load in loads:
                for wear in wear_levels:
                    state = BearingState(
                        wear_level=wear,
                        surface_roughness=0.32 + wear * 5.0,
                        lubrication_quality=max(0.05, 1.0 - wear),
                        bearing_temperature=20 + wear * 30
                    )
                    
                    friction = self.model._calculate_friction_coefficient(
                        state, 1e-6, speed_rpm=speed, load_N=load
                    )
                    
                    results.append({
                        'speed': speed,
                        'load': load,
                        'wear': wear,
                        'friction': friction
                    })
        
        # Analyze results
        min_friction = min(r['friction'] for r in results)
        max_friction = max(r['friction'] for r in results)
        
        print(f"    Friction range: {min_friction:.4f} to {max_friction:.4f}")
        
        # Check new bearing friction (should be around 0.02)
        new_bearing_friction = [r['friction'] for r in results if r['wear'] == 0.0 and r['speed'] == 3000 and r['load'] == 30]
        if new_bearing_friction:
            avg_new_friction = np.mean(new_bearing_friction)
            print(f"    New bearing friction (3000 RPM, 30N): {avg_new_friction:.4f}")
            if avg_new_friction > 0.05:
                print(f"    ⚠️  WARNING: New bearing friction too high ({avg_new_friction:.4f})")
        
        # Check worn bearing friction
        worn_bearing_friction = [r['friction'] for r in results if r['wear'] == 0.95]
        if worn_bearing_friction:
            avg_worn_friction = np.mean(worn_bearing_friction)
            print(f"    Worn bearing friction (95% wear): {avg_worn_friction:.4f}")
            if avg_worn_friction < 0.08:
                print(f"    ⚠️  WARNING: Worn bearing friction too low ({avg_worn_friction:.4f})")
        
        # Plot friction vs wear for different speeds
        plt.figure(figsize=(10, 6))
        for speed in [0, 1000, 3000]:
            wear_vals = [r['wear'] for r in results if r['speed'] == speed and r['load'] == 30]
            friction_vals = [r['friction'] for r in results if r['speed'] == speed and r['load'] == 30]
            plt.plot(wear_vals, friction_vals, 'o-', label=f'{speed} RPM')
        
        plt.xlabel('Wear Level')
        plt.ylabel('Friction Coefficient')
        plt.title('Friction Coefficient vs Wear Level (30N load)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('friction_coefficient_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _test_bearing_load(self):
        """Test bearing load calculation"""
        print("  Testing bearing load calculation...")
        
        speeds = np.linspace(0, 6000, 20)  # 0 to 6000 RPM
        payloads = [500, 1000, 2000]  # kg
        
        results = []
        
        for speed in speeds:
            for payload in payloads:
                conditions = {
                    'speed_rpm': speed,
                    'spacecraft_payload': payload,
                    'maneuvering': True
                }
                load = self.model._calculate_bearing_load(speed, conditions)
                results.append({
                    'speed': speed,
                    'payload': payload,
                    'load': load
                })
        
        # Analyze results
        min_load = min(r['load'] for r in results)
        max_load = max(r['load'] for r in results)
        
        print(f"    Load range: {min_load:.1f}N to {max_load:.1f}N")
        
        # Check static load (0 RPM)
        static_loads = [r['load'] for r in results if r['speed'] == 0]
        if static_loads:
            avg_static = np.mean(static_loads)
            preload = self.model.physics_constants['bearing_preload_N']
            print(f"    Static load (0 RPM): {avg_static:.1f}N")
            print(f"    Expected preload: {preload:.1f}N")
            if abs(avg_static - preload) > 5:
                print(f"    ⚠️  WARNING: Static load doesn't match preload")
        
        # Check high speed loads
        high_speed_loads = [r['load'] for r in results if r['speed'] == 6000]
        if high_speed_loads:
            avg_high_speed = np.mean(high_speed_loads)
            print(f"    High speed load (6000 RPM): {avg_high_speed:.1f}N")
        
        # Plot load vs speed
        plt.figure(figsize=(10, 6))
        for payload in payloads:
            speed_vals = [r['speed'] for r in results if r['payload'] == payload]
            load_vals = [r['load'] for r in results if r['payload'] == payload]
            plt.plot(speed_vals, load_vals, 'o-', label=f'{payload} kg payload')
        
        plt.xlabel('Speed (RPM)')
        plt.ylabel('Bearing Load (N)')
        plt.title('Bearing Load vs Speed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('bearing_load_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _test_step_by_step_integration(self):
        """Test step-by-step integration to see which calculations dominate"""
        print("  Testing step-by-step integration...")
        
        # Start with new bearing
        state = BearingState()
        conditions = {
            'speed_rpm': 3000.0,
            'spacecraft_payload': 1000.0,
            'maneuvering': True
        }
        
        print(f"    Initial state:")
        print(f"      Wear: {state.wear_level:.6f}")
        print(f"      Lubrication: {state.lubrication_quality:.6f}")
        print(f"      Roughness: {state.surface_roughness:.6f} μm")
        print(f"      Friction: {state.friction_coefficient:.6f}")
        
        # Run for several time steps and track changes
        n_steps = 1000
        step_size = 24  # 24 hours per step
        
        changes = {
            'wear': [],
            'lubrication': [],
            'roughness': [],
            'friction': [],
            'temperature': []
        }
        
        for i in range(n_steps):
            old_state = state
            state = self.model.update_bearing_state_one_hour(state, conditions)
            
            # Track changes
            changes['wear'].append(state.wear_level - old_state.wear_level)
            changes['lubrication'].append(state.lubrication_quality - old_state.lubrication_quality)
            changes['roughness'].append(state.surface_roughness - old_state.surface_roughness)
            changes['friction'].append(state.friction_coefficient - old_state.friction_coefficient)
            changes['temperature'].append(state.bearing_temperature - old_state.bearing_temperature)
            
            if i < 10 or i % 100 == 0:
                print(f"    Step {i:4d}: Wear={state.wear_level:.6f}, "
                      f"Lube={state.lubrication_quality:.6f}, "
                      f"Rough={state.surface_roughness:.6f}, "
                      f"Friction={state.friction_coefficient:.6f}")
        
        # Analyze which parameters change most rapidly
        print(f"\n    Average changes per step:")
        for param, change_list in changes.items():
            avg_change = np.mean(np.abs(change_list))
            total_change = np.sum(change_list) if param != 'lubrication' else -np.sum(change_list)
            print(f"      {param.capitalize()}: {avg_change:.2e} per step, total: {total_change:.6f}")
        
        # Plot changes over time
        plt.figure(figsize=(12, 8))
        
        for i, (param, change_list) in enumerate(changes.items()):
            plt.subplot(2, 3, i+1)
            plt.plot(change_list)
            plt.title(f'{param.capitalize()} Changes')
            plt.xlabel('Step')
            plt.ylabel('Change per step')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('step_by_step_changes.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_component_tests():
    """Run the component testing"""
    try:
        print("Starting focused component testing...")
        print("This will test each calculation function independently.\n")
        
        tester = BearingComponentTester()
        tester.test_individual_components()
        
        print("\n=== Component Testing Complete ===")
        print("Review the plots and output to identify which calculations need adjustment.")
        
    except Exception as e:
        print(f"Error during component testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the component tests
    run_component_tests()
