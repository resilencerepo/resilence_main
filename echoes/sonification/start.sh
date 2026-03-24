#!/bin/bash
set -e

# Check Python is installed
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "  macOS:  brew install python  (or download from https://www.python.org)"
    echo "  Linux:  sudo apt install python3 python3-venv  (or equivalent)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment for the first time..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install / update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Launch the app
echo ""
echo "Starting Sonification app... (a browser tab will open automatically)"
echo "Press Ctrl+C to stop the app."
echo ""
python -m streamlit run app_streamlit_local.py --server.headless false
