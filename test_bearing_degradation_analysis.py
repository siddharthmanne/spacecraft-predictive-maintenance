"""
Comprehensive Bearing Degradation Analysis Test

This test file analyzes the interdependencies between different bearing degradation calculations
to identify which parameters are affecting others and whether the progression rates are realistic.

Expected behavior over 5-10 years:
- Wear: 0 → 1.0 (100%)
- Lubrication: 1.0 → 0.0 (complete degradation)
- Surface roughness: 0.32 → 8.0 μm
- Friction: 0.02 → 0.15 (near failure)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.bearing_degradation import BearingDegradationModel, BearingState
from typing import Dict, List
import sys
import os

class BearingDegradationAnalyzer:
    def __init__(self):
        self.model = BearingDegradationModel()
        self.simulation_data = []
        
    def run_comprehensive_analysis(self, years: int = 10, save_plots: bool = True):
        """
        Run a comprehensive analysis over specified years to track parameter interdependencies
        """
        print(f"=== Bearing Degradation Analysis - {years} Year Simulation ===\n")
        
        # Standard operating conditions
        operating_conditions = {
            'speed_rpm': 3000.0,  # Typical reaction wheel speed
            'spacecraft_payload': 1000.0,  # kg
            'maneuvering': True
        }
        
        # Run simulation
        simulation_data = self._run_simulation(years, operating_conditions)
        
        # Analyze interdependencies
        self._analyze_parameter_relationships(simulation_data)
        
        # Check progression rates
        self._check_progression_rates(simulation_data, years)
        
        # Identify problematic calculations
        self._identify_calculation_issues(simulation_data)
        
        # Generate plots
        if save_plots:
            self._generate_analysis_plots(simulation_data, years)
        
        return simulation_data
    
    def _run_simulation(self, years: int, operating_conditions: Dict) -> List[Dict]:
        """Run the simulation and collect detailed data"""
        hours_total = years * 365 * 24
        data = []
        
        # Initial state
        current_state = BearingState()
        
        print(f"Initial State:")
        print(f"  Wear: {current_state.wear_level:.4f}")
        print(f"  Lubrication: {current_state.lubrication_quality:.4f}")
        print(f"  Surface Roughness: {current_state.surface_roughness:.4f} μm")
        print(f"  Friction: {current_state.friction_coefficient:.4f}")
        print(f"  Temperature: {current_state.bearing_temperature:.2f}°C")
        print(f"  Bearing Load: {current_state.bearing_load:.2f}N\n")
        
        # Sample every 24 hours (daily) to reduce data volume but maintain resolution
        sample_interval = 24
        
        for hour in range(0, hours_total, sample_interval):
            # Calculate individual components to track their contributions
            wear_acceleration = self.model._calculate_wear_acceleration(current_state)
            lubrication_loss = self.model._calculate_lubrication_loss(current_state, current_state.bearing_temperature)
            bearing_load = self.model._calculate_bearing_load(operating_conditions['speed_rpm'], operating_conditions)
            
            # Calculate what surface roughness would be with a small wear increment
            small_wear_increment = 1e-6  # Tiny increment for testing
            surface_roughness = self.model._calculate_surface_roughness(current_state, small_wear_increment)
            
            # Calculate friction with current conditions
            friction = self.model._calculate_friction_coefficient(
                current_state, 
                small_wear_increment,
                speed_rpm=operating_conditions['speed_rpm'],
                load_N=bearing_load
            )
            
            # Store detailed data
            data_point = {
                'hour': hour,
                'day': hour / 24,
                'year': hour / (365 * 24),
                'wear_level': current_state.wear_level,
                'lubrication_quality': current_state.lubrication_quality,
                'surface_roughness': current_state.surface_roughness,
                'friction_coefficient': current_state.friction_coefficient,
                'bearing_temperature': current_state.bearing_temperature,
                'bearing_load': bearing_load,
                
                # Component analysis
                'wear_acceleration': wear_acceleration,
                'lubrication_loss_rate': lubrication_loss,
                'calculated_surface_roughness': surface_roughness,
                'calculated_friction': friction,
                
                # Rate analysis (derivatives)
                'wear_rate': 0.0,  # Will be calculated
                'lube_rate': 0.0,  # Will be calculated
                'roughness_rate': 0.0,  # Will be calculated
                'friction_rate': 0.0,  # Will be calculated
            }
            
            # Calculate rates from previous data point
            if len(data) > 0:
                dt = sample_interval  # hours
                prev = data[-1]
                data_point['wear_rate'] = (current_state.wear_level - prev['wear_level']) / dt
                data_point['lube_rate'] = (current_state.lubrication_quality - prev['lubrication_quality']) / dt
                data_point['roughness_rate'] = (current_state.surface_roughness - prev['surface_roughness']) / dt
                data_point['friction_rate'] = (current_state.friction_coefficient - prev['friction_coefficient']) / dt
            
            data.append(data_point)
            
            # Update state for next iteration
            current_state = self.model.update_bearing_state_one_hour(current_state, operating_conditions)
            
            # Progress indicator
            if hour % (hours_total // 20) == 0:
                progress = (hour / hours_total) * 100
                print(f"Progress: {progress:5.1f}% - Year {hour/(365*24):4.1f} - "
                      f"Wear: {current_state.wear_level:6.4f}, "
                      f"Lube: {current_state.lubrication_quality:6.4f}, "
                      f"Rough: {current_state.surface_roughness:6.3f}μm, "
                      f"Friction: {current_state.friction_coefficient:6.4f}")
        
        print(f"\nFinal State after {years} years:")
        print(f"  Wear: {current_state.wear_level:.4f}")
        print(f"  Lubrication: {current_state.lubrication_quality:.4f}")
        print(f"  Surface Roughness: {current_state.surface_roughness:.4f} μm")
        print(f"  Friction: {current_state.friction_coefficient:.4f}")
        print(f"  Temperature: {current_state.bearing_temperature:.2f}°C\n")
        
        return data
    
    def _analyze_parameter_relationships(self, data: List[Dict]):
        """Analyze how parameters affect each other"""
        print("=== Parameter Relationship Analysis ===")
        
        df = pd.DataFrame(data)
        
        # Calculate correlations
        correlation_params = ['wear_level', 'lubrication_quality', 'surface_roughness', 
                            'friction_coefficient', 'bearing_temperature', 'wear_acceleration']
        corr_matrix = df[correlation_params].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Identify strongest relationships
        print("\nStrongest Parameter Relationships:")
        for i, param1 in enumerate(correlation_params):
            for j, param2 in enumerate(correlation_params):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[param1, param2]
                    if abs(corr_val) > 0.8:  # Strong correlation
                        print(f"  {param1} ↔ {param2}: {corr_val:.3f}")
        
        # Analyze rate relationships
        print("\nRate Analysis (how fast each parameter changes):")
        rate_params = ['wear_rate', 'lube_rate', 'roughness_rate', 'friction_rate']
        avg_rates = df[rate_params].mean()
        for param in rate_params:
            param_name = param.replace('_rate', '').replace('_', ' ').title()
            print(f"  Average {param_name} Rate: {avg_rates[param]:.6f} per hour")
    
    def _check_progression_rates(self, data: List[Dict], years: int):
        """Check if progression rates meet expectations"""
        print("\n=== Progression Rate Validation ===")
        
        final_data = data[-1]
        initial_data = data[0]
        
        # Expected vs actual progression
        expectations = {
            'wear_level': (0.0, 1.0, "0 → 1.0"),
            'lubrication_quality': (1.0, 0.0, "1.0 → 0.0"),
            'surface_roughness': (0.32, 8.0, "0.32 → 8.0 μm"),
            'friction_coefficient': (0.02, 0.15, "0.02 → 0.15")
        }
        
        print(f"\nProgression over {years} years:")
        for param, (expected_start, expected_end, description) in expectations.items():
            actual_start = initial_data[param]
            actual_end = final_data[param]
            actual_change = actual_end - actual_start
            expected_change = expected_end - expected_start
            
            progress_percent = (actual_change / expected_change) * 100 if expected_change != 0 else 0
            
            print(f"  {param.replace('_', ' ').title()}:")
            print(f"    Expected: {description}")
            print(f"    Actual: {actual_start:.4f} → {actual_end:.4f}")
            print(f"    Progress: {progress_percent:.1f}% of expected change")
            
            if progress_percent < 20:
                print(f"    ⚠️  WARNING: Very slow progression!")
            elif progress_percent > 200:
                print(f"    ⚠️  WARNING: Too fast progression!")
            elif 80 <= progress_percent <= 120:
                print(f"    ✅ Good progression rate")
            print()
    
    def _identify_calculation_issues(self, data: List[Dict]):
        """Identify which calculations might be problematic"""
        print("=== Calculation Issue Analysis ===")
        
        df = pd.DataFrame(data)
        
        # Check for unrealistic jumps or plateaus
        issues = []
        
        # Check wear progression
        wear_diff = df['wear_level'].diff().abs()
        if wear_diff.max() > 0.01:  # More than 1% jump in one day
            issues.append("Wear level has unrealistic jumps")
        
        wear_rate_changes = df['wear_rate'].diff().abs()
        if wear_rate_changes.max() > df['wear_rate'].std() * 5:
            issues.append("Wear rate has excessive variability")
        
        # Check lubrication degradation
        lube_rate = abs(df['lube_rate'].mean())
        expected_lube_rate = 1.0 / (10 * 365 * 24)  # Should degrade to 0 in 10 years
        if lube_rate < expected_lube_rate * 0.1:
            issues.append("Lubrication degradation too slow")
        elif lube_rate > expected_lube_rate * 10:
            issues.append("Lubrication degradation too fast")
        
        # Check surface roughness
        roughness_final = df['surface_roughness'].iloc[-1]
        if roughness_final < 1.0:
            issues.append("Surface roughness not increasing enough")
        
        # Check friction coefficient
        friction_final = df['friction_coefficient'].iloc[-1]
        if friction_final < 0.05:
            issues.append("Friction coefficient not increasing enough")
        
        # Check for parameter coupling issues
        wear_vs_lube_corr = df['wear_level'].corr(df['lubrication_quality'])
        if wear_vs_lube_corr > -0.8:  # Should be strongly negative
            issues.append("Weak coupling between wear and lubrication degradation")
        
        # Check acceleration factors
        accel_mean = df['wear_acceleration'].mean()
        accel_max = df['wear_acceleration'].max()
        if accel_max < 2.0:
            issues.append("Wear acceleration factor too low - no end-of-life acceleration")
        
        if accel_mean > 10.0:
            issues.append("Wear acceleration factor too high on average")
        
        # Report issues
        if issues:
            print("Identified Issues:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("No major calculation issues identified.")
        
        # Detailed parameter analysis
        print(f"\nDetailed Parameter Statistics:")
        stats_params = ['wear_level', 'lubrication_quality', 'surface_roughness', 
                       'friction_coefficient', 'wear_acceleration']
        
        for param in stats_params:
            series = df[param]
            print(f"  {param.replace('_', ' ').title()}:")
            print(f"    Range: {series.min():.4f} → {series.max():.4f}")
            print(f"    Mean: {series.mean():.4f}, Std: {series.std():.4f}")
            
            # Check for plateaus (consecutive similar values)
            diff_series = series.diff().abs()
            plateau_points = (diff_series < series.std() * 0.01).sum()
            plateau_percent = (plateau_points / len(series)) * 100
            if plateau_percent > 50:
                print(f"    ⚠️  {plateau_percent:.1f}% plateau points detected")
            print()
    
    def _generate_analysis_plots(self, data: List[Dict], years: int):
        """Generate comprehensive analysis plots"""
        print("Generating analysis plots...")
        
        df = pd.DataFrame(data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Bearing Degradation Analysis - {years} Year Simulation', fontsize=16)
        
        # Plot 1: Main parameters over time
        ax = axes[0, 0]
        ax.plot(df['year'], df['wear_level'], 'r-', label='Wear Level', linewidth=2)
        ax.plot(df['year'], df['lubrication_quality'], 'b-', label='Lubrication Quality', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Normalized Value (0-1)')
        ax.set_title('Wear and Lubrication Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Surface roughness and friction
        ax = axes[0, 1]
        ax2 = ax.twinx()
        line1 = ax.plot(df['year'], df['surface_roughness'], 'g-', label='Surface Roughness', linewidth=2)
        line2 = ax2.plot(df['year'], df['friction_coefficient'], 'orange', label='Friction Coefficient', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Surface Roughness (μm)', color='g')
        ax2.set_ylabel('Friction Coefficient', color='orange')
        ax.set_title('Surface Roughness and Friction')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rates of change
        ax = axes[1, 0]
        ax.plot(df['year'], df['wear_rate'] * 8760, 'r-', label='Wear Rate (per year)', alpha=0.7)
        ax.plot(df['year'], abs(df['lube_rate']) * 8760, 'b-', label='|Lube Rate| (per year)', alpha=0.7)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Rate per Year')
        ax.set_title('Parameter Change Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 4: Wear acceleration factor
        ax = axes[1, 1]
        ax.plot(df['year'], df['wear_acceleration'], 'purple', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Wear Acceleration Factor')
        ax.set_title('Wear Acceleration Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Temperature and load
        ax = axes[2, 0]
        ax2 = ax.twinx()
        line1 = ax.plot(df['year'], df['bearing_temperature'], 'red', label='Temperature', linewidth=2)
        line2 = ax2.plot(df['year'], df['bearing_load'], 'brown', label='Bearing Load', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Temperature (°C)', color='red')
        ax2.set_ylabel('Bearing Load (N)', color='brown')
        ax.set_title('Temperature and Load Evolution')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Phase diagram (wear vs other parameters)
        ax = axes[2, 1]
        scatter = ax.scatter(df['wear_level'], df['friction_coefficient'], 
                           c=df['year'], cmap='viridis', alpha=0.6, s=10)
        ax.set_xlabel('Wear Level')
        ax.set_ylabel('Friction Coefficient')
        ax.set_title('Friction vs Wear (colored by time)')
        plt.colorbar(scatter, ax=ax, label='Time (years)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'bearing_analysis_{years}year.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        
        plt.show()
        
        # Generate correlation heatmap
        self._plot_correlation_heatmap(df)
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Generate correlation heatmap"""
        plt.figure(figsize=(12, 10))
        
        correlation_params = ['wear_level', 'lubrication_quality', 'surface_roughness', 
                            'friction_coefficient', 'bearing_temperature', 'wear_acceleration',
                            'bearing_load']
        
        corr_matrix = df[correlation_params].corr()
        
        import seaborn as sns
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        
        plt.savefig('bearing_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved as: bearing_correlation_heatmap.png")
        plt.show()

def run_bearing_analysis():
    """Main function to run the bearing degradation analysis"""
    try:
        analyzer = BearingDegradationAnalyzer()
        
        print("Starting comprehensive bearing degradation analysis...")
        print("This will simulate 10 years of operation and analyze parameter interdependencies.\n")
        
        # Run analysis
        simulation_data = analyzer.run_comprehensive_analysis(years=10, save_plots=True)
        
        # Save data to CSV for further analysis
        df = pd.DataFrame(simulation_data)
        csv_filename = 'bearing_simulation_data.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nSimulation data saved to: {csv_filename}")
        
        print("\n=== Analysis Complete ===")
        print("Review the plots and output above to identify calculation issues.")
        print("Key things to look for:")
        print("1. Are progression rates realistic?")
        print("2. Are parameters properly coupled?")
        print("3. Do acceleration factors work correctly?")
        print("4. Are there any unrealistic jumps or plateaus?")
        
        return simulation_data
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Configure Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the analysis
    data = run_bearing_analysis()
