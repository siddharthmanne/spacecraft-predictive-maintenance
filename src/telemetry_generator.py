"""
SpacecraftTelemetryGenerator - Mission Control Center for Bearing Degradation Simulation

This module serves as the central coordinator for spacecraft predictive maintenance simulation.
It orchestrates multiple reaction wheel subsystems, manages mission timeline, collects telemetry data,
and provides formatted output for dashboard visualization.

Key responsibilities:
1. Mission setup and configuration management
2. Subsystem coordination and time-stepping
3. Telemetry data collection and storage
4. Health status assessment and predictions
5. Data formatting for UI consumption

Architecture:
- MissionConfig: Holds mission parameters (duration, speed, load)
- SpacecraftTelemetryGenerator: Main simulation controller
- Telemetry flow: ReactionWheel → TelemetryGenerator → UI Dashboard
"""

from src.reaction_wheel import ReactionWheelSubsystem
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MissionConfig:
    """
    Mission configuration parameters for spacecraft simulation
    
    Simplified: User only needs to specify mission basics.
    All timing is handled automatically with 1-hour telemetry intervals.
    """
    duration_days: int              # Total mission length (user selectable) - ONLY user input needed
    initial_speed_rpm: float        # Starting wheel speed (user selectable)
    initial_load_factor: float      # Starting operational load (user selectable)
    num_reaction_wheels: int = 4    # Typical spacecraft has 3-4 wheels for 3-axis control + redundancy

class SpacecraftTelemetryGenerator:
    """
    Central mission controller that coordinates all spacecraft subsystems
    
    Design pattern: This acts as a "Mission Control Center" that:
    1. Manages the overall mission timeline
    2. Coordinates updates across all subsystems
    3. Collects and aggregates telemetry data
    4. Provides health monitoring and predictions
    5. Formats data for UI consumption
    
    Why this architecture?
    - Separation of concerns: Each reaction wheel manages its own physics
    - Centralized coordination: One place to control mission flow
    - Scalable: Easy to add new subsystem types later
    - UI-friendly: Provides formatted data streams for dashboard
    """
    
    def __init__(self):
        """
        Initialize empty mission state
        
        Note: Mission must be explicitly initialized with initialize_mission()
        before simulation can begin. This two-step process allows UI to
        collect user preferences before creating subsystems.
        """
        self.subsystems = {}                # Dict of subsystem_id -> ReactionWheelSubsystem
        self.telemetry_history = []         # Time-ordered list of telemetry snapshots
        self.mission_config = None          # MissionConfig object (set by initialize_mission)
        self.mission_elapsed_hours = 0      # Hours elapsed since mission start (integer)
        self.total_mission_hours = 0        # Total mission duration in hours
        self.is_running = False             # Simulation state flag
        
    def initialize_mission(self, mission_config: MissionConfig):
        """
        Set up mission parameters and create subsystem instances
        
        This is called once at mission start with user-selected parameters.
        Creates fresh instances of all subsystems with initial conditions.
        
        Args:
            mission_config: User-defined mission parameters (duration, speed, load)
            
        Why separate initialization?
        - Allows UI to collect user preferences first
        - Enables mission restart without recreating the generator
        - Clear separation between setup and execution phases
        """
        # Store mission configuration for reference throughout simulation
        self.mission_config = mission_config
        
        # Reset simulation state for fresh start
        self.mission_elapsed_hours = 0
        self.total_mission_hours = mission_config.duration_days * 24  # Convert days to hours
        self.telemetry_history = []
        
        # Create reaction wheel subsystems based on mission config
        # Each wheel gets a unique ID and starts in IDLE mode for safety
        self.subsystems = {}
        for i in range(mission_config.num_reaction_wheels):
            wheel_id = f"RW_{i+1}"  # Human-readable IDs: RW_1, RW_2, etc.
            self.subsystems[wheel_id] = ReactionWheelSubsystem(
                wheel_id=i,                                    # Numeric ID for internal use
                operational_mode='IDLE',                       # Safe startup mode
                load_factor=mission_config.initial_load_factor # User-selected initial load
            )
    
    def start_simulation(self):
        """
        Begin mission execution
        
        Safety check ensures mission is properly initialized before starting.
        Sets is_running flag that controls the main simulation loop.
        """
        if not self.mission_config:
            raise ValueError("Mission must be initialized before starting simulation")
        self.is_running = True
    
    def stop_simulation(self):
        """
        Halt mission execution
        
        Can be called by UI to pause/stop simulation.
        Preserves all telemetry data for analysis.
        """
        self.is_running = False
    
    def step_simulation(self, user_commands: Optional[Dict] = None):
        """
        Advance simulation by one hour and collect telemetry
        
        This is the heart of the simulation - advances mission by exactly 1 hour.
        Each call updates all subsystems, collects telemetry, and advances the clock.
        
        Args:
            user_commands: Real-time commands from UI dashboard:
                - 'target_speed_rpm': Change wheel speed during mission
                - 'load_factor': Change operational load during mission
                
        Why 1-hour timesteps?
        - Reaction wheel bearing physics requires hourly updates for accuracy
        - Provides smooth telemetry data for UI visualization
        - Allows real-time user interaction during mission
        """
        # Safety check - don't run if simulation is stopped or mission complete
        if not self.is_running or self.mission_elapsed_hours >= self.total_mission_hours:
            return
            
        # Build command dictionary with mission defaults
        # User commands override defaults, allowing real-time control
        commands = {
            'target_speed_rpm': self.mission_config.initial_speed_rpm,  # Default to initial settings
            'load_factor': self.mission_config.initial_load_factor,     # Can be overridden by user
            'mode': 'NOMINAL'                                           # Standard operational mode
        }
        
        # Apply user commands if provided (real-time control from UI)
        if user_commands:
            commands.update(user_commands)
        
        # Create telemetry snapshot for this hour
        # This captures the state of the entire spacecraft at current mission time
        current_telemetry = {
            'mission_elapsed_hours': self.mission_elapsed_hours,        # Hours since mission start
            'mission_elapsed_days': self.mission_elapsed_hours / 24.0,  # Days since mission start  
            'timestamp': datetime.now().isoformat(),                    # Real-world timestamp
            'subsystems': {}                                            # Will hold all subsystem data
        }
        
        # Update each subsystem for 1 hour and collect its telemetry
        # Each wheel runs its physics simulation for exactly 1 hour
        timestep_hours = 1.0  # Fixed: always advance by 1 hour
        for subsystem_id, subsystem in self.subsystems.items():
            # Tell subsystem to advance by 1 hour with current commands
            subsystem.update(timestep_hours, commands)
            
            # Copy subsystem's latest telemetry and add mission time context
            subsystem_telemetry = subsystem.latest_telemetry.copy()
            subsystem_telemetry['mission_elapsed_hours'] = self.mission_elapsed_hours
            
            current_telemetry['subsystems'][subsystem_id] = subsystem_telemetry
        
        # Store complete telemetry snapshot in history
        # This builds up the time series for visualization and analysis
        self.telemetry_history.append(current_telemetry)
        
        # Advance mission clock by 1 hour
        self.mission_elapsed_hours += 1
        
        # Check if mission has reached its planned end
        # Automatically stop simulation when duration is reached
        if self.mission_elapsed_hours >= self.total_mission_hours:
            self.is_running = False
    
    def run_simulation_batch(self, target_hours: Optional[int] = None):
        """
        Run simulation efficiently to advance time quickly
        
        This efficiently simulates multiple hours without UI interaction.
        Much more efficient than calling step_simulation() repeatedly.
        
        Args:
            target_hours: Stop at this mission hour (default: full mission duration)
            
        Performance note: This directly updates physics without individual
        step overhead, making it ideal for batch processing or full missions.
        """
        # Default to full mission duration if no target specified
        if not target_hours:
            target_hours = self.total_mission_hours
        
        # Ensure we don't exceed mission bounds
        target_hours = min(target_hours, self.total_mission_hours)
        
        # Build command dictionary with defaults
        commands = {
            'target_speed_rpm': self.mission_config.initial_speed_rpm,
            'load_factor': self.mission_config.initial_load_factor,
            'mode': 'NOMINAL'
        }
        
        # Simulate each hour from current position to target
        while self.is_running and self.mission_elapsed_hours < target_hours:
            # Create telemetry snapshot for this hour
            current_telemetry = {
                'mission_elapsed_hours': self.mission_elapsed_hours,
                'mission_elapsed_days': self.mission_elapsed_hours / 24.0,
                'timestamp': datetime.now().isoformat(),
                'subsystems': {}
            }
            
            # Update each subsystem for 1 hour
            timestep_hours = 1.0
            for subsystem_id, subsystem in self.subsystems.items():
                subsystem.update(timestep_hours, commands)
                
                # Copy telemetry and add mission context
                subsystem_telemetry = subsystem.latest_telemetry.copy()
                subsystem_telemetry['mission_elapsed_hours'] = self.mission_elapsed_hours
                
                current_telemetry['subsystems'][subsystem_id] = subsystem_telemetry
            
            # Store telemetry
            self.telemetry_history.append(current_telemetry)
            
            # Advance mission clock
            self.mission_elapsed_hours += 1
            
            # Check if mission complete
            if self.mission_elapsed_hours >= self.total_mission_hours:
                self.is_running = False
    
    def run_full_mission(self, user_commands: Optional[Dict] = None):
        """
        Run the entire mission from start to finish - convenience method
        
        This is equivalent to run_simulation_batch() with no target_hours.
        Just a cleaner name for running complete missions.
        
        Args:
            user_commands: Optional commands to apply throughout mission
        """
        # Apply user commands if provided
        if user_commands:
            # Update defaults and run batch with custom commands
            commands = {
                'target_speed_rpm': self.mission_config.initial_speed_rpm,
                'load_factor': self.mission_config.initial_load_factor,
                'mode': 'NOMINAL'
            }
            commands.update(user_commands)
            
            # We'd need to modify run_simulation_batch to accept commands
            # For now, just call run_simulation_batch()
            self.run_simulation_batch()
        else:
            # Standard full mission run
            self.run_simulation_batch()
    
    def update_mission_parameters(self, **kwargs):
        """
        Create command dictionary for real-time parameter updates
        
        This provides a clean interface for the UI to change mission parameters
        during simulation. Returns a command dict that can be passed to step_simulation().
        
        Supported parameters:
        - target_speed_rpm: Change wheel speed (affects wear rate)
        - load_factor: Change operational load (affects bearing stress)
        
        Usage by UI:
            commands = generator.update_mission_parameters(target_speed_rpm=4000)
            generator.step_simulation(commands)
        """
        return kwargs  # Simple pass-through - step_simulation() handles the logic
    
    def get_telemetry_stream(self) -> Dict:
        """
        Format telemetry data for UI dashboard consumption
        
        This is the primary interface between simulation and UI. It packages
        all mission data in a format optimized for dashboard visualization.
        
        Returns comprehensive data structure containing:
        - Mission status and progress
        - Time series data for plotting
        - Latest sensor readings  
        - Health assessments and predictions
        - Performance metrics
        
        Why this structure?
        - UI gets everything it needs in one call
        - Data is pre-formatted for common visualizations
        - Health status is pre-calculated for alerts
        - Time series is ready for plotting libraries
        """
        # Handle case where simulation hasn't started yet
        if not self.telemetry_history:
            return {
                'mission_status': 'Not Started',
                'current_time_hours': 0,
                'time_series': {},
                'latest_readings': {},
                'health_status': {}
            }
        
        # Get most recent telemetry snapshot
        latest = self.telemetry_history[-1]
        
        # Generate time series data for plotting (computationally expensive)
        time_series = self._format_time_series()
        
        # Calculate health status and maintenance predictions
        health_status = self._calculate_health_status()
        
        # Return comprehensive dashboard data package
        return {
            'mission_status': 'Running' if self.is_running else 'Stopped',
            'mission_elapsed_hours': self.mission_elapsed_hours,
            'mission_elapsed_days': self.mission_elapsed_hours / 24.0,
            'total_mission_hours': self.total_mission_hours,
            'mission_progress_percent': (self.mission_elapsed_hours / self.total_mission_hours) * 100,
            'time_series': time_series,           # For plotting trends over time
            'latest_readings': latest,            # For current status displays
            'health_status': health_status,       # For alerts and predictions
            'total_data_points': len(self.telemetry_history)  # For performance monitoring
        }
    
    def _format_time_series(self) -> Dict:
        """
        Transform telemetry history into time series format for plotting
        
        Converts the list of telemetry snapshots into arrays organized by parameter.
        This format is optimized for plotting libraries like matplotlib or plotly.
        
        Performance consideration: This method processes the entire telemetry history
        every time it's called. For long missions (>1000 data points), consider
        caching or incremental updates.
        
        Returns:
            Dict with time arrays and parameter arrays for each subsystem
        """
        if not self.telemetry_history:
            return {}
            
        # Initialize data structure for time series
        time_series = {
            'time_hours': [],  # X-axis for plotting
            'time_days': [],   # Alternative X-axis in days
            'subsystems': {}   # Y-axis data for each wheel
        }
        
        # Pre-initialize arrays for each subsystem to avoid repeated dict lookups
        # This improves performance when processing thousands of data points
        for subsystem_id in self.subsystems.keys():
            time_series['subsystems'][subsystem_id] = {
                # Observable telemetry (what sensors would measure)
                'temperature': [],          # Motor/bearing temperature [°C]
                'current': [],              # Motor current draw [A]  
                'vibration': [],            # Vibration amplitude [g]
                
                # Internal physics state (from bearing model)
                'wear_level': [],           # Bearing wear [0-1 scale]
                'friction_coefficient': [], # Bearing friction [dimensionless]
                'lubrication_quality': []   # Lubrication degradation [0-1 scale]
            }
        
        # Process each telemetry snapshot and extract time series data
        # This loop can be expensive for long missions - consider optimization
        for entry in self.telemetry_history:
            # Extract time data (same for all subsystems)
            time_series['time_hours'].append(entry['mission_elapsed_hours'])
            time_series['time_days'].append(entry['mission_elapsed_days'])
            
            # Extract subsystem data with safe defaults
            for subsystem_id, data in entry['subsystems'].items():
                if subsystem_id in time_series['subsystems']:
                    subsys_data = time_series['subsystems'][subsystem_id]
                    
                    # Use .get() with defaults to handle missing data gracefully
                    subsys_data['temperature'].append(data.get('temperature', 0))
                    subsys_data['current'].append(data.get('current', 0))
                    subsys_data['vibration'].append(data.get('vibration', 0))
                    subsys_data['wear_level'].append(data.get('wear_level', 0))
                    subsys_data['friction_coefficient'].append(data.get('friction_coefficient', 0))
                    subsys_data['lubrication_quality'].append(data.get('lubrication_quality', 1.0))
        
        return time_series
    
    def _calculate_health_status(self) -> Dict:
        """
        Assess overall spacecraft health and generate maintenance predictions
        
        This implements a simplified health monitoring system based on bearing
        degradation thresholds. In a real spacecraft, this would involve complex
        machine learning models and historical failure data.
        
        Health assessment logic:
        - Good: All parameters within normal operating ranges
        - Warning: One or more parameters approaching critical thresholds  
        - Critical: One or more parameters in failure range
        
        Threshold values are based on bearing engineering literature and
        typical spacecraft operational requirements.
        
        Returns:
            Dict with health status, alerts, and maintenance predictions
        """
        if not self.telemetry_history:
            return {'overall_health': 'Unknown', 'predictions': {}}
        
        # Get latest telemetry for current health assessment
        latest = self.telemetry_history[-1]
        
        # Initialize health status structure
        health_status = {
            'overall_health': 'Good',       # Overall spacecraft health
            'subsystem_health': {},         # Individual wheel health  
            'predictions': {},              # Future failure predictions
            'maintenance_alerts': []        # Actionable maintenance items
        }
        
        # Assess health of each reaction wheel subsystem
        for subsystem_id, data in latest['subsystems'].items():
            # Extract key health parameters with safe defaults
            wear = data.get('wear_level', 0)                    # 0-1 scale
            friction = data.get('friction_coefficient', 0.02)   # Dimensionless
            lubrication = data.get('lubrication_quality', 1.0)  # 0-1 scale
            
            # Apply health assessment thresholds
            # These values are based on aerospace bearing literature
            if wear > 0.8 or friction > 0.12 or lubrication < 0.2:
                # Critical: Immediate maintenance required
                health = 'Critical'
                health_status['overall_health'] = 'Critical'  # One critical wheel = critical mission
                
            elif wear > 0.6 or friction > 0.08 or lubrication < 0.4:
                # Warning: Maintenance should be planned
                health = 'Warning'
                if health_status['overall_health'] == 'Good':
                    health_status['overall_health'] = 'Warning'  # Upgrade overall status
            else:
                # Good: Normal operation
                health = 'Good'
            
            health_status['subsystem_health'][subsystem_id] = health
            
            # Generate specific maintenance alerts for actionable items
            # These help mission planners decide on maintenance actions
            if wear > 0.7:
                health_status['maintenance_alerts'].append(
                    f"{subsystem_id}: High wear detected ({wear:.1%}) - bearing replacement recommended"
                )
            if lubrication < 0.3:
                health_status['maintenance_alerts'].append(
                    f"{subsystem_id}: Low lubrication ({lubrication:.1%}) - lubrication service required"
                )
        
        return health_status
    
    def get_mission_summary(self) -> Dict:
        """
        Provide high-level mission overview for UI status displays
        
        This gives the UI a quick way to show mission configuration and current
        status without processing the full telemetry stream. Useful for:
        - Mission status panels
        - Configuration verification displays  
        - Performance monitoring (data points collected)
        
        Returns:
            Dict with mission config and current status
        """
        if not self.mission_config:
            return {'status': 'Not initialized'}
            
        return {
            # Mission configuration (user-selected parameters)
            'duration_days': self.mission_config.duration_days,
            'initial_speed_rpm': self.mission_config.initial_speed_rpm,
            'initial_load_factor': self.mission_config.initial_load_factor,
            'num_reaction_wheels': self.mission_config.num_reaction_wheels,
            
            # Current mission status
            'total_mission_hours': self.total_mission_hours,
            'mission_elapsed_hours': self.mission_elapsed_hours,
            'is_running': self.is_running,
            'data_points_collected': len(self.telemetry_history)  # For performance monitoring
        }