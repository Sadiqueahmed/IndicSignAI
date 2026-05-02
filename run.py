#!/usr/bin/env python3
"""
IndicSignAI Launcher
Runs the Flask application from the src directory
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Change to src directory for proper template/static resolution
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

# Run the app
from app import app, socketio

if __name__ == '__main__':
    print("\n" + "="*80)
    print("INDICSIGNAI - Starting from organized structure")
    print("="*80)
    print("Project reorganized into clean architecture:")
    print("  - src/          : Main application")
    print("  - models/       : Trained models")
    print("  - templates/    : HTML templates")
    print("  - static/       : CSS/JS files")
    print("  - training/     : Training scripts")
    print("  - archive/      : Legacy files")
    print("="*80 + "\n")
    
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
