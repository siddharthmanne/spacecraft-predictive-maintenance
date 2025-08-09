import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import threading
import queue
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.telemetry_generator import SpacecraftTelemetryGenerator, MissionConfig

# Configure page
st.set_page_config(
    page_title="Spacecraft Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit's running indicator to reduce visual noise
hide_streamlit_style = """
<style>
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Ensure a consistent dark background across the whole app (main and sidebar)
st.markdown("""
<style>
/* Typography & color system */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
:root {
    --bg: #0b1020;
    --surface: #11162a;
    --surface-2: #0e1326;
    --border: rgba(148,163,184,0.25);
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --accent: #38bdf8; /* cyan-400 */
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}
:root { color-scheme: dark; }

html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji" !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg) !important;
    border-right: 1px solid var(--border);
}
.block-container { padding-top: 1rem; }

/* Headings & general text */
h1, h2, h3, h4, h5, h6 { color: var(--text); font-weight: 600; }
p, span, label, small, div { color: var(--text); }
[data-testid="stMarkdownContainer"] * { color: var(--text) !important; }
[data-testid="stWidgetLabel"] { color: var(--text) !important; font-weight: 600; }

/* Cards */
.main-header {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    text-align: center;
    letter-spacing: 0.02em;
    margin-bottom: 0.75rem;
}
.mission-card {
    background: linear-gradient(180deg, var(--surface), var(--surface-2));
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 24px rgba(0,0,0,0.25);
}
.status-normal { color: var(--success); font-weight: 600; }
.status-warning { color: var(--warning); font-weight: 600; }
.status-critical { color: var(--danger); font-weight: 600; }
.metric-card {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

/* Inputs & widgets */
.stTextInput>div>div>input,
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* Select caret and text (corrected) */
[data-baseweb="select"] span { color: var(--text) !important; }
div[data-baseweb="select"] input { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--text) !important; }

/* Slider */
[data-baseweb="slider"] * { color: var(--text) !important; }
[data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(56,189,248,0.15) !important;
}
[data-baseweb="slider"] > div > div {
    background: rgba(56,189,248,0.35) !important; /* active track */
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: var(--accent) !important;
    color: #0b1020 !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.05);
}

/* Secondary buttons */
.stButton [data-testid="baseButton-secondary"] { background: #1f2937 !important; }

/* Alerts */
.stAlert { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }

/* Tables */
[data-testid="stTable"] { background: var(--surface) !important; }
[data-testid="stTable"] table { color: var(--text) !important; }

/* Remove overly bright default white backgrounds */
.css-1dp5vir, .css-1n76uvr, .ef3psqc2 { background: transparent !important; }

/* Pulse (kept subtle) */
.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.65;} 100%{opacity:1;} }

/* --- Hardened overrides for selects, menus, download button, focus & spinner --- */
/* Force dark theme for all dropdowns (landing + mission) */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* CLOSED control: ensure container always dark */
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stSelectbox"] [aria-haspopup="listbox"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    box-shadow: none !important;
}

/* OPEN menu portal surfaces */
div[data-baseweb="popover"] { background-color: transparent !important; }
body > div[data-baseweb="layer"],
body > div[data-baseweb="popover"],
body > div[role="dialog"] { background: transparent !important; }
body > div[data-baseweb="layer"] [data-baseweb="menu"],
body > div[data-baseweb="popover"] [data-baseweb="menu"],
body > div[role="dialog"] [data-baseweb="menu"],
[data-baseweb="menu"],
[role="listbox"],
ul[role="listbox"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* Menu content colors (corrected) */
div[data-baseweb="menu"] li, div[data-baseweb="menu"] *, [role="option"], [role="listbox"] * {
    color: var(--text) !important;
    opacity: 1 !important;
}

/* Active/hover options */
[role="option"][aria-selected="true"],
div[data-baseweb="menu"] li[aria-selected="true"],
div[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] li:focus {
    background-color: rgba(56,189,248,0.12) !important;
    color: var(--text) !important;
}

/* Remove blue focus/selection highlights; use subtle accent */
*:focus { outline: none !important; }
.stTextInput>div>div>input:focus,
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within,
div[data-baseweb="textarea"]:focus-within,
div[role="slider"]:focus,
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
[data-testid="stSelectbox"] [role="combobox"]:focus,
[data-testid="stSelectbox"] [role="combobox"]:focus-visible {
    box-shadow: 0 0 0 2px rgba(56,189,248,0.20) !important;
    border-color: var(--accent) !important;
    outline: none !important;
}
::selection { background: rgba(148,163,184,0.25) !important; color: var(--text) !important; }
* { -webkit-tap-highlight-color: rgba(0,0,0,0) !important; }
/* Links shouldn't be bright blue */
a, a:visited { color: var(--text) !important; text-decoration: underline dotted rgba(148,163,184,0.4); }
a:hover { color: var(--accent) !important; }

/* Hide Streamlit spinner/overlay to prevent page darkening */
[data-testid="stSpinner"], .stSpinner, [data-testid="stStatusWidget"], [data-testid="stModal"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS for theme adjustments (kept for structure, now aligned with above)
st.markdown("""
<style>
/* Extra specificity to ensure dropdowns and buttons never show white */
.stSelectbox > div, .stMultiSelect > div { box-shadow: none !important; }
.stDownloadButton > button { box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)

# Final robust select/menu overrides (single source of truth)
st.markdown("""
<style>
:root { color-scheme: dark; }

/* Closed control surface */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSelectbox"] [aria-haspopup="listbox"],
button[role="combobox"],
button[aria-haspopup="listbox"] {
  background-color: var(--surface) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stSelectbox"] [data_baseweb="select"] *,
[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
  color: var(--text) !important;
  fill: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}

/* Remove blue focus ring */
[data-testid="stSelectbox"] *:focus,
[data-testid="stSelectbox"] *:focus-visible { outline: none !important; box-shadow: none !important; }

/* Menu portal surfaces */
body > div[data-baseweb="layer"],
body > div[data-baseweb="popover"],
body > div[role="dialog"] { background: transparent !important; }

body > div[data-baseweb="layer"] [data_baseweb="menu"],
body > div[data-baseweb="popover"] [data_baseweb="menu"],
body > div[data-baseweb="layer"] [role="listbox"],
body > div[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"],
[role="listbox"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
}

/* Options */
[role="option"], [data-baseweb="menu"] li {
  background-color: var(--surface) !important;
  color: var(--text) !important;
}
[role="option"][aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"],
[data-baseweb="menu"] li:hover,
[role="option"]:hover {
  background-color: rgba(56,189,248,0.12) !important;
  color: var(--text) !important;
}

/* Disabled options */
[role="option"][aria-disabled="true"] { color: var(--text-muted) !important; opacity: 0.7 !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_active' not in st.session_state:
    st.session_state.simulation_active = False
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'telemetry_data' not in st.session_state:
    st.session_state.telemetry_data = []
if 'telemetry_queue' not in st.session_state:
    st.session_state.telemetry_queue = queue.Queue(maxsize=100)
if 'sim_speed' not in st.session_state:
    st.session_state.sim_speed = 1
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'health_status' not in st.session_state:
    st.session_state.health_status = "All Systems Normal"

class SimulationEngine:
    def __init__(self):
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue(maxsize=100)
        self.batch_mode = False

    def start(self, generator, speed=1):
        if self.running:
            return
        self.running = True
        self.batch_mode = speed == "Skip to End" or (isinstance(speed, (int, float)) and speed >= 50)
        # The thread now calls the corrected _run_loop
        self.thread = threading.Thread(target=self._run_loop, args=(generator, speed))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_loop(self, generator, speed):
        """
        Improved run loop with batching for high speeds.
        - Uses generator.advance_simulation() to support fractional-hour stepping with internal accumulation.
        - Implements batching for high speeds to reduce UI update frequency.
        - The 'speed' parameter controls how many simulation DAYS are processed per second of real time.
        """
        if speed == "Skip to End":
            # Special case: jump to end immediately
            self._skip_to_end(generator)
            return
        
        # Convert days per second to hours per second (24 hours = 1 day)
        hours_per_second = speed * 24
        
        # Determine batching strategy based on speed
        if self.batch_mode:
            # For high speeds (>=50 days/sec), batch simulation steps and update UI less frequently
            batch_size = min(int(hours_per_second / 10), 240)  # Max 10 days per batch
            ui_update_interval = max(0.1, 1.0 / 10)  # Update UI at most 10 times per second
        else:
            # For normal speeds, update more frequently
            batch_size = 1
            ui_update_interval = 1.0 / hours_per_second if hours_per_second > 0 else 0.1

        steps_since_ui_update = 0
        last_ui_update = time.time()
        
        while self.running:
            loop_start_time = time.time()
            
            try:
                # Process a batch of simulation steps
                for _ in range(batch_size):
                    if not self.running or not generator.is_running:
                        break
                    # Advance by one hour per inner step; UI speed is controlled by outer pacing
                    generator.advance_simulation(1.0)
                    steps_since_ui_update += 1
                
                # Update UI only at specified intervals
                current_time = time.time()
                if (current_time - last_ui_update) >= ui_update_interval:
                    telemetry = generator.get_telemetry_stream()
                    if telemetry:
                        health_status = self._update_health_status(telemetry)
                        data_item = {
                            'timestamp': datetime.now(),
                            'mission_hours': generator.mission_elapsed_hours,
                            'mission_days': generator.mission_elapsed_hours / 24,
                            'data': telemetry,
                            'health_status': health_status
                        }
                        try:
                            self.data_queue.put_nowait(data_item)
                        except queue.Full:
                            try:
                                self.data_queue.get_nowait() # Make space by removing the oldest item
                                self.data_queue.put_nowait(data_item)
                            except queue.Empty:
                                pass
                    
                    last_ui_update = current_time
                    steps_since_ui_update = 0
                
                # If the simulation is over, stop the loop.
                if not generator.is_running:
                    self.stop()
                    break

                # Wait intelligently to maintain the desired simulation speed
                if not self.batch_mode:
                    elapsed_time = time.time() - loop_start_time
                    sleep_duration = max(0, ui_update_interval - elapsed_time)
                    time.sleep(sleep_duration)
                else:
                    # For batch mode, minimal sleep to prevent CPU spinning
                    time.sleep(0.01)

            except Exception as e:
                # Stop the simulation on error to prevent broken loops.
                self.stop()
                break

    def _skip_to_end(self, generator):
        """Efficiently skip to the end of the mission with progress indication"""
        try:
            total_hours = generator.mission_config.duration_days * 24
            current_hours = generator.mission_elapsed_hours
            remaining_hours = total_hours - current_hours
            
            # Add progress tracking to session state
            if 'skip_progress' not in st.session_state:
                st.session_state.skip_progress = 0.0
            
            # Process in chunks with progress updates
            chunk_size = max(1, remaining_hours // 100)  # 100 progress updates
            processed = 0
            
            while generator.is_running and self.running:
                # Process a chunk
                hours_this_chunk = min(chunk_size, remaining_hours - processed)
                if hours_this_chunk <= 0:
                    break
                generator.advance_simulation(float(hours_this_chunk))
                processed += hours_this_chunk
                
                # Update progress
                progress = processed / remaining_hours if remaining_hours > 0 else 1.0
                st.session_state.skip_progress = min(progress, 1.0)
                
                if processed >= remaining_hours:
                    break
            
            # Final telemetry update
            telemetry = generator.get_telemetry_stream()
            if telemetry:
                health_status = self._update_health_status(telemetry)
                data_item = {
                    'timestamp': datetime.now(),
                    'mission_hours': generator.mission_elapsed_hours,
                    'mission_days': generator.mission_elapsed_hours / 24,
                    'data': telemetry,
                    'health_status': health_status
                }
                try:
                    self.data_queue.put_nowait(data_item)
                except queue.Full:
                    pass
            
            # Clear progress indicator
            st.session_state.skip_progress = 1.0
            self.stop()
            
        except Exception as e:
            self.stop()

    def _update_health_status(self, telemetry):
        """Update health status based on telemetry data"""
        if not telemetry or 'health_status' not in telemetry:
            return "All Systems Normal"

        health_info = telemetry['health_status']
        overall_health = health_info.get('overall_health', 'Good')
        alerts = health_info.get('maintenance_alerts', [])

        if overall_health == 'Critical':
            return f"CRITICAL: {len(alerts)} alert(s). {alerts[0] if alerts else ''}"
        elif overall_health == 'Warning':
            return f"WARNING: {len(alerts)} alert(s). {alerts[0] if alerts else ''}"
        else:
            return "All Systems Normal"

def landing_page():
    st.markdown('<div class="main-header">Spacecraft Mission Control</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mission-card">
    <h3>Mission Configuration</h3>
    Initialize your spacecraft reaction wheel subsystem simulation
    </div>
    """, unsafe_allow_html=True)
    
    # Clear any previous setup values to ensure clean state
    setup_keys = ['setup_duration_days', 'setup_initial_rpm', 'setup_load_factor', 'setup_num_wheels', 'setup_display_mode']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Subsystem", ["Reaction Wheel"], key="setup_subsystem")
        
        duration_days = st.slider("Mission Duration (Days)", 1, 3650, 365, 
                                help="1-10 years mission length", key="setup_duration_days")
        if duration_days > 365:
            st.info(f"Duration: {duration_days/365:.1f} years")
            
        load_factor = st.slider("Initial Load Factor", 0.7, 1.5, 1.0, 0.01, key="setup_load_factor")
        
        # Add display mode selection for initial setup
        st.selectbox("Display Mode", ["Simple", "Advanced"], key="setup_display_mode")
        
    with col2:
        initial_rpm = st.slider("Initial Speed (RPM)", 0, 7000, 3000, key="setup_initial_rpm")
        num_wheels = st.slider("Number of Wheels", 1, 8, 4, key="setup_num_wheels")
        
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("Start Simulation", type="primary", use_container_width=True):
            with st.spinner("Initializing mission systems..."):
                try:
                    # Create mission config
                    config = MissionConfig(
                        duration_days=duration_days,
                        initial_speed_rpm=initial_rpm,
                        initial_load_factor=load_factor,
                        num_reaction_wheels=num_wheels
                    )
                    
                    # Initialize generator
                    generator = SpacecraftTelemetryGenerator()
                    generator.initialize_mission(config)
                    generator.start_simulation()
                    
                    # Store in session state
                    st.session_state.generator = generator
                    st.session_state.mission_config = config
                    st.session_state.simulation_active = True
                    st.session_state.telemetry_data = [] # Clear old data
                    
                    # Clear setup UI state to prevent confusion
                    setup_keys = ['setup_duration_days', 'setup_initial_rpm', 'setup_load_factor', 
                                'setup_num_wheels', 'setup_display_mode', 'setup_subsystem']
                    for key in setup_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Initialize runtime controls with setup values
                    st.session_state.current_load_factor = config.initial_load_factor
                    st.session_state.current_rpm = config.initial_speed_rpm
                    st.session_state.current_display_mode = st.session_state.get('setup_display_mode', 'Simple')
                    
                    # Start simulation engine
                    st.session_state.sim_engine.start(generator, st.session_state.sim_speed)
                    
                    st.success("Mission initialized. Telemetry uplink established.")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Mission initialization failed: {e}")

def mission_dashboard():
    # Header with mission progress
    if st.session_state.telemetry_data:
        latest = st.session_state.telemetry_data[-1]
        mission_days = latest['mission_days']
        total_days = st.session_state.mission_config.duration_days
        progress = min(mission_days / total_days, 1.0)
        
        health_status_text = st.session_state.health_status
        health_class = 'normal'
        if 'CRITICAL' in health_status_text:
            health_class = 'critical'
        elif 'WARNING' in health_status_text:
            health_class = 'warning'

        st.markdown(f"""
        <div class="mission-card">
        <h3>Mission Progress: Day {mission_days:.1f} / {total_days} ({progress:.1%})</h2>
        <div class="status-{health_class}">
        {health_status_text}
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(progress)
    else:
        st.info("Waiting for initial telemetry data...")

    # Control panel with clear runtime labeling
    st.markdown("### Mission Control Panel")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Runtime Load Factor control (different from initial setup)
        if 'current_load_factor' not in st.session_state:
            st.session_state.current_load_factor = st.session_state.mission_config.initial_load_factor
        
        load_factor = st.slider("Runtime Load Factor", 0.7, 1.5, st.session_state.current_load_factor, 0.01, 
                               help="Adjust current operational load", key="runtime_load_factor")
        if load_factor != st.session_state.current_load_factor:
            st.session_state.current_load_factor = load_factor
    
    with col2:
        # Runtime Speed control (different from initial setup)
        if 'current_rpm' not in st.session_state:
            st.session_state.current_rpm = st.session_state.mission_config.initial_speed_rpm
        
        rpm = st.slider("Runtime Speed (RPM)", 0, 7000, st.session_state.current_rpm, 
                       help="Adjust current wheel speed", key="runtime_rpm")
        if rpm != st.session_state.current_rpm:
            st.session_state.current_rpm = rpm
    
    with col3:
        # Runtime Display mode (separate from setup)
        if 'current_display_mode' not in st.session_state:
            st.session_state.current_display_mode = 'Simple'
        
        display_mode = st.selectbox("Display Mode", ["Simple", "Advanced"], 
                                  index=0 if st.session_state.current_display_mode == "Simple" else 1,
                                  key="runtime_display_mode")
        st.session_state.current_display_mode = display_mode
    
    with col4:
        # Simulation speed options including skip to end
        speed_options = [1.0, 10.0, 100.0, "Skip to End"]
        current_speed = st.session_state.sim_speed if st.session_state.sim_speed != "Skip to End" else "Skip to End"
        if current_speed not in speed_options: 
            current_speed = 10.0
        
        new_speed = st.selectbox("Simulation Speed (days/sec)", speed_options, 
                               index=speed_options.index(current_speed),
                               help="Higher speeds use intelligent batching for performance")
        
        # Show progress bar for "Skip to End" mode
        if st.session_state.sim_speed == "Skip to End" and 'skip_progress' in st.session_state:
            progress = st.session_state.skip_progress
            st.progress(progress, text=f"Skipping to end... {progress:.1%}")
        
        if new_speed != st.session_state.sim_speed:
            if new_speed == "Skip to End":
                # Skip to the end of the mission with progress indication
                st.session_state.sim_engine.stop()
                time.sleep(0.1)
                st.session_state.sim_speed = new_speed
                st.session_state.sim_engine.start(st.session_state.generator, new_speed)
            else:
                st.session_state.sim_speed = new_speed
                # Restart the engine with new speed
                st.session_state.sim_engine.stop()
                time.sleep(0.1)
                st.session_state.sim_engine.start(st.session_state.generator, new_speed)
    
    with col5:
        wheel_options = [f"RW_{i+1}" for i in range(st.session_state.mission_config.num_reaction_wheels)]
        selected_wheel = st.selectbox("Focus Wheel", wheel_options)

    # NEW: Export daily averages (selected wheel or all wheels)
    with st.container():
        export_scope = st.radio("Export CSV scope", ["Selected Wheel", "All Wheels"], horizontal=True)
        export_df = build_daily_average_export(selected_wheel if export_scope == "Selected Wheel" else None,
                                               all_wheels=(export_scope == "All Wheels"))
        disabled = export_df.empty
        file_label = selected_wheel if export_scope == "Selected Wheel" else "all_wheels"
        file_name = f"mission_daily_averages_{file_label}.csv"
        st.download_button(
            label="Export CSV (Daily averages)",
            data=export_df.to_csv(index=False) if not disabled else "",
            file_name=file_name,
            mime="text/csv",
            disabled=disabled,
            help="Exports per-day averages weighted by hours, independent of UI update rate"
        )
    
    # Abort button on a separate row for better visibility
    col_abort1, col_abort2, col_abort3 = st.columns([2, 1, 2])
    with col_abort2:
        if st.button("Abort Mission", type="secondary", use_container_width=True):
            # Immediate reset of all simulation data
            st.session_state.sim_engine.stop()
            st.session_state.simulation_active = False
            st.session_state.generator = None
            st.session_state.telemetry_data = []
            st.session_state.health_status = "All Systems Normal"
            st.session_state.sim_speed = 10.0
            
            # Clear all session state variables for clean restart
            keys_to_clear = [
                'mission_config', 'current_load_factor', 'current_rpm', 'current_display_mode',
                'skip_progress', 'runtime_load_factor', 'runtime_rpm', 'runtime_display_mode'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Dashboard - show appropriate mode based on selection
    if display_mode == "Advanced":
        advanced_dashboard(selected_wheel)
    else:
        simple_dashboard(selected_wheel)

def simple_dashboard(selected_wheel):
    if not st.session_state.telemetry_data:
        st.info("Waiting for telemetry data...")
        return
    
    df = create_telemetry_dataframe(selected_wheel)
    if df is None or df.empty:
        st.info(f"No data yet for {selected_wheel}. Please wait.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Primary time series plot
        metric_display = st.selectbox("Primary Metric", ["Current", "Vibration", "Housing Temperature"])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['mission_days'],
            y=df[metric_display],
            mode='lines+markers',
            name=metric_display,
            line=dict(color='#38bdf8', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"{metric_display} vs Mission Time",
            xaxis_title="Mission Days",
            yaxis_title=metric_display,
            xaxis=dict(range=[0, None]),  # Always start x-axis at 0
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0', family='Inter, ui-sans-serif, system-ui')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary cards
        latest_data = df.iloc[-1]
            
        st.markdown(f"""
        <div class="metric-card">
        <h4>Current Draw</h4>
        <h2>{latest_data['Current']:.2f} A</h2>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="metric-card">
        <h4>Vibration</h4>
        <h2>{latest_data['Vibration']:.3f} g</h2>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="metric-card">
        <h4>Temperature</h4>
        <h2>{latest_data['Housing Temperature']:.1f}°C</h2>
        </div>
        """, unsafe_allow_html=True)

def advanced_dashboard(selected_wheel):
    if not st.session_state.telemetry_data:
        st.info("Waiting for telemetry data...")
        return
    
    df = create_telemetry_dataframe(selected_wheel)
    if df is None or df.empty:
        st.info(f"No data yet for {selected_wheel}. Please wait.")
        return
    
    # Observable metrics row
    st.subheader("Observable Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_metric_plot(df, 'Vibration', 'Vibration (g)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_metric_plot(df, 'Current', 'Current (A)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_metric_plot(df, 'Housing Temperature', 'Temperature (°C)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance indicators row
    st.subheader("Performance Indicators")
    col1, col2 = st.columns(2)
    
    # These metrics were missing from create_telemetry_dataframe, let's add them
    if 'max_torque_Nm' in df.columns:
        with col1:
            fig = create_metric_plot(df, 'max_torque_Nm', 'Torque Capability (Nm)')
            st.plotly_chart(fig, use_container_width=True)
    
        with col2:
            fig = create_metric_plot(df, 'pointing_jitter_arcsec', 'Pointing Jitter (arcsec)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Internal physics row
    st.subheader("Internal Bearing Physics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_dual_metric_plot(df, ['wear_level', 'lubrication_quality'], 
                                    'Wear & Lubrication')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_metric_plot(df, 'friction_coefficient', 'Friction Coefficient')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_metric_plot(df, 'surface_roughness', 'Surface Roughness (μm)')
        st.plotly_chart(fig, use_container_width=True)

def create_telemetry_dataframe(selected_wheel):
    if not st.session_state.telemetry_data:
        return pd.DataFrame() # Return empty DataFrame
    
    data_points = []
    for entry in st.session_state.telemetry_data:
        # The structure is data -> latest_readings -> subsystems
        if 'data' in entry and 'latest_readings' in entry['data']:
            subsystems = entry['data']['latest_readings'].get('subsystems', {})
            wheel_data = subsystems.get(selected_wheel)

            # Health status also contains performance metrics
            health_status = entry['data'].get('health_status', {})
            # This part seems complex, let's simplify based on the generator output
            # The telemetry generator does not add performance metrics to the main stream.
            # Let's assume they are part of the subsystem telemetry for now.
            
            if wheel_data:
                # Add performance metrics. They are calculated in ReactionWheelSubsystem but not added to telemetry.
                # For this example, we'll extract them if available, otherwise default to 0.
                perf_metrics = health_status.get('performance_metrics', {}).get(selected_wheel, {})

                data_points.append({
                    'mission_days': entry.get('mission_days', 0),
                    'Current': wheel_data.get('current', 0),
                    'Vibration': wheel_data.get('vibration', 0),
                    'Housing Temperature': wheel_data.get('housing_temperature', 0),
                    'wear_level': wheel_data.get('wear_level', 0),
                    'friction_coefficient': wheel_data.get('friction_coefficient', 0),
                    'lubrication_quality': wheel_data.get('lubrication_quality', 1.0),
                    'surface_roughness': wheel_data.get('surface_roughness', 0.32),
                    'max_torque_Nm': perf_metrics.get('max_torque_Nm', 0),
                    'pointing_jitter_arcsec': perf_metrics.get('pointing_jitter_arcsec', 0)
                })
    
    return pd.DataFrame(data_points) if data_points else pd.DataFrame()

def create_metric_plot(df, metric, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['mission_days'],
        y=df[metric],
        mode='lines', # Use lines for smoother plots
        name=title,
        line=dict(color='#38bdf8', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Mission Days",
        yaxis_title=title.split('(')[-1].strip(')') # Use unit for axis title
        if '(' in title else title,
        xaxis=dict(range=[0, None]),  # Always start x-axis at 0
        plot_bgcolor='rgba(26,26,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', size=10, family='Inter, ui-sans-serif, system-ui'),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_dual_metric_plot(df, metrics, title):
    fig = go.Figure()
    colors = ['#38bdf8', '#f97316']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=df['mission_days'],
            y=df[metric],
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Mission Days",
        xaxis=dict(range=[0, None]),  # Always start x-axis at 0
        plot_bgcolor='rgba(26,26,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', size=10, family='Inter, ui-sans-serif, system-ui'),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# --- NEW: Daily average export helpers ---

def _collect_daily_averages_from_history(history, wheel_filter=None):
    """Compute per-day averages from generator.telemetry_history.
    Splits multi-hour steps across day boundaries and weights by hours.
    If wheel_filter is provided, only that wheel ID is processed. Returns a DataFrame.
    """
    if not history:
        return pd.DataFrame()

    # daily_sums[(wheel_id, day)] -> {'metric': sum_over_hours, 'hours': total_hours}
    daily_sums = {}

    # Metrics to average if present in telemetry
    metric_keys = [
        'current', 'vibration', 'housing_temperature',
        'wear_level', 'friction_coefficient', 'lubrication_quality',
        'surface_roughness', 'bearing_temperature', 'speed_rpm', 'load_factor'
    ]

    for entry in history:
        start_hour_block = entry.get('mission_elapsed_hours', 0)
        subsystems = entry.get('subsystems', {})
        for wheel_id, tele in subsystems.items():
            if wheel_filter and wheel_id != wheel_filter:
                continue
            duration = int(round(tele.get('timestep_hours', 1)))
            if duration <= 0:
                duration = 1

            # Snapshot values assumed constant across this block for averaging
            values = {k: tele.get(k) for k in metric_keys if k in tele}

            start = float(start_hour_block)
            remaining = float(duration)
            while remaining > 0:
                day_index = int(start // 24)
                day_end_hour = (day_index + 1) * 24
                span = min(remaining, day_end_hour - start)

                key = (wheel_id, day_index)
                if key not in daily_sums:
                    daily_sums[key] = {'hours': 0.0}
                    for mk in values:
                        daily_sums[key][mk] = 0.0

                for mk, val in values.items():
                    if val is not None:
                        daily_sums[key][mk] += float(val) * span
                daily_sums[key]['hours'] += span

                start += span
                remaining -= span

    # Build rows
    rows = []
    for (wheel_id, day_index), agg in sorted(daily_sums.items(), key=lambda x: (x[0][0], x[0][1])):
        hours = max(agg.get('hours', 0.0), 1e-9)
        row = {
            'wheel_id': wheel_id,
            'mission_day': day_index,
            'hours_contributed': agg.get('hours', 0.0)
        }
        for mk in metric_keys:
            if mk in agg:
                row[mk] = agg[mk] / hours
        rows.append(row)

    return pd.DataFrame(rows)


def build_daily_average_export(selected_wheel: str | None, all_wheels: bool = False) -> pd.DataFrame:
    """Public helper for Streamlit to build a daily-average DataFrame from the generator.
    If all_wheels is True, exports all wheels; otherwise only the selected_wheel.
    """
    generator = st.session_state.get('generator')
    if generator is None or not getattr(generator, 'telemetry_history', None):
        return pd.DataFrame()

    wheel_filter = None if all_wheels else selected_wheel
    df = _collect_daily_averages_from_history(generator.telemetry_history, wheel_filter)

    # Add readable column order
    preferred_order = [
        'mission_day', 'wheel_id', 'hours_contributed',
        'current', 'vibration', 'housing_temperature',
        'wear_level', 'friction_coefficient', 'lubrication_quality',
        'surface_roughness', 'bearing_temperature', 'speed_rpm', 'load_factor'
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    return df[cols] if not df.empty else df

# Main app logic
def main():
    # Initialize session state variables once
    if 'simulation_active' not in st.session_state:
        st.session_state.simulation_active = False
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'telemetry_data' not in st.session_state:
        st.session_state.telemetry_data = []
    if 'sim_speed' not in st.session_state:
        st.session_state.sim_speed = 1.0 # A reasonable default speed (1 day per second)
    if 'health_status' not in st.session_state:
        st.session_state.health_status = "All Systems Normal"
    if 'sim_engine' not in st.session_state:
        st.session_state.sim_engine = SimulationEngine()
    
    # Main application logic
    if not st.session_state.simulation_active:
        landing_page()
    else:
        # If simulation is active, process new data from the queue
        try:
            while not st.session_state.sim_engine.data_queue.empty():
                item = st.session_state.sim_engine.data_queue.get_nowait()
                st.session_state.telemetry_data.append(item)
                
                if 'health_status' in item:
                    st.session_state.health_status = item['health_status']
                
                # Limit the stored data to prevent memory issues on long runs
                if len(st.session_state.telemetry_data) > 2000:
                    st.session_state.telemetry_data = st.session_state.telemetry_data[-2000:]
        except queue.Empty:
            pass
        
        # Now, draw the mission dashboard
        mission_dashboard()
        
        # Use a placeholder to update smoothly without rerunning the entire app
        if st.session_state.sim_engine.running or st.session_state.sim_speed == "Skip to End":
            # Faster refresh; keep UI snappy without dark overlay
            time.sleep(0.05)
            st.rerun()

if __name__ == "__main__":
    main()
