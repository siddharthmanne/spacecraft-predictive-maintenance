"""
Visualize Bearing Degradation Progression
-----------------------------------------
Simulates and visualizes the evolution of bearing physical properties over time using the BearingDegradationModel.
"""
import matplotlib.pyplot as plt
import numpy as np
from src.bearing_degradation import BearingDegradationModel, BearingState

# Time checkpoints in hours
TIME_CHECKPOINTS = {
    'Day 0': 0,
    'Month 1': 30 * 24,
    'Month 3': 90 * 24,
    'Month 6': 180 * 24,
    'Year 1': 365 * 24,
    'Year 3': 3 * 365 * 24,
    'Year 5': 5 * 365 * 24,
    'Year 6': 6 * 365 * 24,
    'Year 7': 7 * 365 * 24,
    'Year 8': 8 * 365 * 24,
    'Year 9': 9 * 365 * 24,
    'Year 10': 10 * 365 * 24
}

# Simulation parameters
OPERATING_CONDITIONS = {
    'speed_rpm': 3500,
    'load_factor': 1.0,
    'spacecraft_payload': 1200,
    'maneuvering': True
}


def simulate_degradation(model, initial_state, total_hours):
    """
    Simulate bearing degradation for total_hours, returning a list of BearingState at each hour.
    """
    states = [initial_state]
    state = initial_state
    for hour in range(1, total_hours + 1):
        state = model.update_bearing_state_one_hour(state, OPERATING_CONDITIONS)
        states.append(state)
    return states


def main():
    model = BearingDegradationModel()
    initial_state = BearingState()
    max_hours = TIME_CHECKPOINTS['Year 10']
    states = simulate_degradation(model, initial_state, max_hours)

    # Extract checkpoint indices
    checkpoint_indices = {name: hour for name, hour in TIME_CHECKPOINTS.items()}

    # Prepare data for plotting
    hours = np.arange(0, max_hours + 1)
    wear = [s.wear_level for s in states]
    lube = [s.lubrication_quality for s in states]
    friction = [s.friction_coefficient for s in states]
    roughness = [s.surface_roughness for s in states]
    temp = [s.bearing_temperature for s in states]

    # Plot progression
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(hours, wear, label='Wear Level')
    for name, idx in checkpoint_indices.items():
        plt.scatter(idx, wear[idx], label=name)
    plt.xlabel('Hours'); plt.ylabel('Wear Level (0-1)'); plt.title('Wear Progression')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(hours, lube, label='Lubrication Quality')
    for name, idx in checkpoint_indices.items():
        plt.scatter(idx, lube[idx], label=name)
    plt.xlabel('Hours'); plt.ylabel('Lubrication Quality (0-1)'); plt.title('Lubrication Degradation')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(hours, friction, label='Friction Coefficient')
    for name, idx in checkpoint_indices.items():
        plt.scatter(idx, friction[idx], label=name)
    plt.xlabel('Hours'); plt.ylabel('Friction Coefficient'); plt.title('Friction Progression')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(hours, roughness, label='Surface Roughness (μm)')
    for name, idx in checkpoint_indices.items():
        plt.scatter(idx, roughness[idx], label=name)
    plt.xlabel('Hours'); plt.ylabel('Surface Roughness (μm)'); plt.title('Surface Roughness Evolution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('bearing_degradation_progression.png')
    plt.show()

    # Print checkpoint summary
    print("\nCheckpoint Summary:")
    for name, idx in checkpoint_indices.items():
        print(f"{name} ({idx} hr): Wear={wear[idx]:.3f}, Lube={lube[idx]:.3f}, Friction={friction[idx]:.3f}, Roughness={roughness[idx]:.2f}, Temp={temp[idx]:.2f}")

if __name__ == "__main__":
    main()
