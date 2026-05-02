# Optimized detection parameters for better performance
# Copy these values to src/app.py

# Frame processing - process every frame for maximum detection rate
PROCESS_EVERY_N_FRAMES = 1

# Lowered thresholds for more sensitive detection
CONFIDENCE_THRESHOLD = 0.05  # Was 0.08
MIN_CONFIDENCE_FOR_DISPLAY = 0.03  # Was 0.05
STABLE_FRAMES_REQUIRED = 1
COOLDOWN_FRAMES = 1  # Faster switching between signs

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Additional optimizations:
# 1. Use tf.function for model inference (faster)
# 2. Batch predictions when possible
# 3. Use GPU if available
# 4. Reduce model input preprocessing overhead
