import sys
import app
from app import load_models

print('Running Model Load Diagnostic...')
try:
    load_models()
    print('DIAGNOSTIC SUCCESS: Models Loaded Successfully.')
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
