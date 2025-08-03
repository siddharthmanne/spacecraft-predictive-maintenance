"""
Simulates a specific reaction wheel's behaviour.
"""

from src.bearing_degradation import BearingDegradationModel


class ReactionWheelSubsystem:
    def __init__(self):
        self.bearing_model = BearingDegradationModel()
        self.bearing_state = BearingState()
        self.operational_mode = 

    def update(self, mission_time_hours, commands):
        """
        Main orchestration function that coordinates bearing updates with operational context
        """
        # Translates mission commands to bearing operating conditions
        # Calls bearing_model.update_bearing_state()
        # Converts bearing physics to observable effect (eg. higher friction -> increased current draw, bearing wear -> increased vibration)
        # Generates telemetry measurements
    
    def _physics_to_vibration(bearing_physics):
        """
        Converts raw bearing wear/roughness into reaction wheel-specific vibration signatures.
        Only RW subsystem knows how bearing defects manifest as wheel vibrations.
        """

    def _physics_to_current(bearing_physics, wheel_speed):
        """
        Translates bearing friction into motor current draw based on wheel speed and load

        """
    def _physics_to_temperature(bearing_physics, operational_load):
        """
        Calculates how bearing condition affects wheel housing temperature
        """

    def get_performance_metrics(self):
        """
        Calculates how bearing degradation affects RW performance (torque capability, pointing accuracy)
        """
     
    def predict_maintenance_timeline(current_state, mission_profile):
        """
        Converts physics-based wear predictions into mission-relevant maintenance schedules
        """
        # Uses bearing_model.predict_time_to_failure()
        # Converts mission relevant timeframes
        # Returns "days to maintenance" for UI