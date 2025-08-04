#!/usr/bin/env python3
"""
Degradation Visualization Script
-------------------------------
Tracks all physical properties from bearing degradation and reaction wheel subsystems
over different time periods to visualize the physics feedback loops and degradation progression.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.reaction_wheel import ReactionWheelSubsystem
from src.bearing_degradation import BearingState
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class DegradationVisualizer:
    def __init__(self):
        """Initialize the visualizer with a fresh reaction wheel"""
        self.wheel = ReactionWheelSubsystem()
        self.time_periods = {
            'Day 1': 24,
            'Month 1': 30 * 24,
            'Month 3': 3 * 30 * 24,
            'Month 6': 6 * 30 * 24,
            'Year 1': 365 * 24,
            'Year 2': 2 * 365 * 24,
            'Year 3': 3 * 365 * 24,
            'Year 4': 4 * 365 * 24,
            'Year 5': 5 * 365 * 24,
            'Year 6': 6 * 365 * 24,
            'Year 7': 7 * 365 * 24,
            'Year 8': 8 * 365 * 24,
            'Year 9': 9 * 365 * 24,
            'Year 10': 10 * 365 * 24,
        }
        
    def run_degradation_simulation(self, mission_profile='realistic'):
        """
        Run degradation simulation with specified mission profile
        
        Args:
            mission_profile: 'realistic', 'continuous', 'extreme', or 'conservative'
        """
        print(f"Running {mission_profile} mission profile...")
        print("=" * 80)
        
        # Initialize data storage
        data_points = []
        
        # Mission profiles
        profiles = {
            'realistic': {
                'speed_base': 3000,
                'speed_variation': 500,
                'load_base': 1.0,
                'load_variation': 0.2,
                'temp_base': 18.0,
                'temp_variation': 4.0,
                'duty_cycle': True
            },
            'continuous': {
                'speed_base': 3000,
                'speed_variation': 0,
                'load_base': 1.0,
                'load_variation': 0,
                'temp_base': 20.0,
                'temp_variation': 0,
                'duty_cycle': False
            },
            'extreme': {
                'speed_base': 4000,
                'speed_variation': 1000,
                'load_base': 1.5,
                'load_variation': 0.5,
                'temp_base': 25.0,
                'temp_variation': 10.0,
                'duty_cycle': False
            },
            'conservative': {
                'speed_base': 2000,
                'speed_variation': 200,
                'load_base': 0.8,
                'load_variation': 0.1,
                'temp_base': 15.0,
                'temp_variation': 2.0,
                'duty_cycle': True
            }
        }
        
        profile = profiles[mission_profile]
        
        # Track checkpoints
        checkpoints = list(self.time_periods.values())
        current_checkpoint_idx = 0
        
        # Run simulation
        max_hours = max(self.time_periods.values())
        
        for hour in range(1, max_hours + 1):
            # Calculate operating conditions based on profile
            if profile['duty_cycle']:
                # Realistic duty cycle: 16 hours at full speed, 8 hours at reduced speed
                duty_cycle_speed = profile['speed_base'] if hour % 24 < 16 else profile['speed_base'] * 0.3
            else:
                duty_cycle_speed = profile['speed_base']
            
            # Add variations
            speed = duty_cycle_speed + profile['speed_variation'] * np.sin(hour / 168)  # Weekly variation
            load_factor = profile['load_base'] + profile['load_variation'] * np.sin(hour / 168)
            temperature = profile['temp_base'] + profile['temp_variation'] * np.sin(hour / 24)
            
            # Update wheel
            commands = {
                'target_speed_rpm': speed,
                'load_factor': load_factor,
                'bearing_temperature': temperature,
                'timestep_hours': 1.0
            }
            
            self.wheel.update(hour, commands)
            
            # Check if we've reached a checkpoint
            if current_checkpoint_idx < len(checkpoints) and hour >= checkpoints[current_checkpoint_idx]:
                checkpoint_name = list(self.time_periods.keys())[current_checkpoint_idx]
                data_point = self._collect_data_point(hour, checkpoint_name, commands)
                data_points.append(data_point)
                current_checkpoint_idx += 1
        
        return pd.DataFrame(data_points)
    
    def _collect_data_point(self, hour, checkpoint_name, commands):
        """Collect all physical properties at a checkpoint"""
        telemetry = self.wheel.get_telemetry()
        bearing_state = self.wheel.bearing_state
        
        return {
            'checkpoint': checkpoint_name,
            'hours': hour,
            'days': hour / 24,
            'years': hour / (365 * 24),
            
            # Operating conditions
            'speed_rpm': telemetry['speed_rpm'],
            'load_factor': telemetry['load_factor'],
            'bearing_temperature': commands['bearing_temperature'],
            
            # Bearing degradation properties
            'wear_level': bearing_state.wear_level,
            'friction_coefficient': bearing_state.friction_coefficient,
            'surface_roughness': bearing_state.surface_roughness,
            'lubrication_quality': bearing_state.lubrication_quality,
            'temperature_history': bearing_state.temperature_history,
            
            # Reaction wheel outputs
            'measured_vibration': telemetry['measured_vibration'],
            'housing_temperature': telemetry['housing_temperature'],
            'motor_current': telemetry['motor_current'],
            
            # Performance metrics
            'max_torque_Nm': self.wheel.get_performance_metrics()['max_torque_Nm'],
            'pointing_jitter_arcsec': self.wheel.get_performance_metrics()['pointing_jitter_arcsec'],
        }
    
    def print_degradation_summary(self, df):
        """Print a comprehensive degradation summary"""
        print("\n" + "=" * 80)
        print("DEGRADATION SUMMARY")
        print("=" * 80)
        
        # Format the dataframe for display
        display_df = df.copy()
        
        # Round numeric columns for readability
        numeric_columns = ['wear_level', 'friction_coefficient', 'surface_roughness', 
                          'lubrication_quality', 'measured_vibration', 'housing_temperature', 
                          'motor_current', 'max_torque_Nm', 'pointing_jitter_arcsec']
        
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        # Display key metrics
        print("\nKey Degradation Metrics:")
        print("-" * 50)
        
        for _, row in display_df.iterrows():
            print(f"\n{row['checkpoint']} ({row['years']:.1f} years):")
            print(f"  Wear Level: {row['wear_level']:.1%}")
            print(f"  Lubrication Quality: {row['lubrication_quality']:.1%}")
            print(f"  Friction Coefficient: {row['friction_coefficient']:.4f}")
            print(f"  Surface Roughness: {row['surface_roughness']:.3f} μm")
            print(f"  Motor Current: {row['motor_current']:.3f} A")
            print(f"  Vibration: {row['measured_vibration']:.4f} g")
            print(f"  Housing Temp: {row['housing_temperature']:.1f}°C")
            print(f"  Max Torque: {row['max_torque_Nm']:.4f} Nm")
            print(f"  Pointing Jitter: {row['pointing_jitter_arcsec']:.1f} arcsec")
        
        return display_df
    
    def analyze_physics_feedback(self, df):
        """Analyze the physics feedback loops"""
        print("\n" + "=" * 80)
        print("PHYSICS FEEDBACK LOOP ANALYSIS")
        print("=" * 80)
        
        # Temperature effects on lubrication
        print("\n1. Temperature Effects on Lubrication:")
        print("-" * 40)
        for _, row in df.iterrows():
            temp_effect = "High" if row['bearing_temperature'] > 50 else "Normal"
            print(f"{row['checkpoint']}: Temp={row['bearing_temperature']:.1f}°C ({temp_effect}) → "
                  f"Lubrication={row['lubrication_quality']:.1%}")
        
        # Lubrication effects on wear
        print("\n2. Lubrication Effects on Wear:")
        print("-" * 40)
        for i, row in df.iterrows():
            if i > 0:
                prev_row = df.iloc[i-1]
                lube_degradation = prev_row['lubrication_quality'] - row['lubrication_quality']
                wear_acceleration = (row['wear_level'] - prev_row['wear_level']) / (row['hours'] - prev_row['hours'])
                print(f"{row['checkpoint']}: Lube loss={lube_degradation:.4f} → "
                      f"Wear acceleration={wear_acceleration:.6f}/hour")
        
        # Wear effects on friction and current
        print("\n3. Wear Effects on Friction and Current:")
        print("-" * 40)
        for _, row in df.iterrows():
            print(f"{row['checkpoint']}: Wear={row['wear_level']:.1%} → "
                  f"Friction={row['friction_coefficient']:.4f} → "
                  f"Current={row['motor_current']:.3f}A")
        
        # Friction effects on temperature
        print("\n4. Friction Effects on Temperature:")
        print("-" * 40)
        for _, row in df.iterrows():
            temp_rise = row['housing_temperature'] - 20.0  # Ambient temperature
            print(f"{row['checkpoint']}: Friction={row['friction_coefficient']:.4f} → "
                  f"Temp rise={temp_rise:.1f}°C")
    
    def run_all_profiles(self):
        """Run simulation with all mission profiles"""
        profiles = ['realistic', 'continuous', 'extreme', 'conservative']
        
        for profile in profiles:
            print(f"\n{'='*20} {profile.upper()} MISSION PROFILE {'='*20}")
            
            # Reset wheel for each profile
            self.wheel = ReactionWheelSubsystem()
            
            # Run simulation
            df = self.run_degradation_simulation(profile)
            
            # Print summary
            self.print_degradation_summary(df)
            
            # Analyze physics feedback
            self.analyze_physics_feedback(df)
            
            # Save to CSV
            filename = f"degradation_data_{profile}.csv"
            df.to_csv(filename, index=False)
            print(f"\nData saved to: {filename}")

def main():
    """Main function to run the visualization"""
    print("Spacecraft Bearing Degradation Visualization")
    print("=" * 80)
    print("This script simulates bearing degradation over different time periods")
    print("and analyzes the physics feedback loops between temperature, lubrication,")
    print("wear, friction, and performance metrics.")
    print()
    
    visualizer = DegradationVisualizer()
    
    # Run with realistic profile by default
    print("Running realistic mission profile...")
    df = visualizer.run_degradation_simulation('realistic')
    visualizer.print_degradation_summary(df)
    visualizer.analyze_physics_feedback(df)
    
    # Save data
    df.to_csv('degradation_data_realistic.csv', index=False)
    print(f"\nData saved to: degradation_data_realistic.csv")
    
    # Ask if user wants to run all profiles
    print("\n" + "="*80)
    print("Would you like to run all mission profiles (realistic, continuous, extreme, conservative)?")
    print("This will generate 4 CSV files with comprehensive degradation data.")
    
    # For now, just run the realistic profile
    print("Running realistic profile only. To run all profiles, call visualizer.run_all_profiles()")

if __name__ == "__main__":
    main() 