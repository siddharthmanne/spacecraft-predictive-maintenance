#!/bin/bash

set -e

echo "Starting Spacecraft Mission Control Dashboard..."
echo "Initializing telemetry systems..."

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Launch Streamlit app
echo "🌟 Launching Mission Control at http://localhost:8501"
streamlit run app.py
