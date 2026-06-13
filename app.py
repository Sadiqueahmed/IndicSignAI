#!/usr/bin/env python3
"""
IndicSignAI - Root Launcher
This file redirects to the main application in src/
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Change to src directory so relative paths work
src_dir = os.path.join(os.path.dirname(__file__), 'src')
os.chdir(src_dir)

# Now import and run the app
from app import app, socketio

if __name__ == '__main__':
    import os
    os.makedirs('../templates', exist_ok=True)
    os.makedirs('../static', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../training/data/training_data', exist_ok=True)
    
    # NOTE: load_models() is already called at import time by src/app.py (line 34)
    
    print("\n" + "="*80)
    print("INDICSIGNAI - READY")
    print("="*80)
    print(f"✓ Running from: {os.getcwd()}")
    print(f"✓ Templates: ../templates")
    print(f"✓ Static: ../static")
    print("\n🌐 Open: http://localhost:5000")
    print("="*80)
    
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
