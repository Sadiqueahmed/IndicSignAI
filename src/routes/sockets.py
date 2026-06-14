"""
sockets.py — WebSocket Handlers for Real-Time ISL_IMAGE Inference

Architecture:
  - Frontend sends camera frame (base64 JPEG) + MediaPipe landmarks via 'frame' event
  - Backend decodes frame → BGR→RGB → passes to ISL_IMAGE model (MobileNetV2 + Transformer)
  - The model crops the hand region using landmarks and predicts from 160×160 RGB image
  - Confidence threshold > 0.65 enforced before emitting result
  - Emits 'prediction' event with { sign, confidence }
"""

import base64
import numpy as np
import cv2
import traceback
import threading
from flask_socketio import emit
from flask import request

# Import the ISL_IMAGE model handler
from src.models.isl_image_model import predict_from_hand, get_isl_image_model, load_isl_image_model

# Import core for logging and model state checks
import src.core as core
from src.core import logger

# ══════════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD — only accept predictions > 0.65
# (matches ISLImageModel.confidence_threshold)
# ══════════════════════════════════════════════════════════════════════
PREDICTION_CONFIDENCE_THRESHOLD = 0.65

# Per-client session state
client_sessions = {}

# Store the socketio instance for background task usage
_socketio = None

# Flag to track if ISL_IMAGE model is loaded
_isl_image_model_loaded = False


def _ensure_isl_image_model():
    """Ensure the ISL_IMAGE model is loaded (lazy load on first use)."""
    global _isl_image_model_loaded
    if not _isl_image_model_loaded:
        model = get_isl_image_model()
        if model.model is None:
            success = load_isl_image_model()
            if success:
                print("[ISL_IMAGE] ✓ Model loaded successfully on first socket connection")
                _isl_image_model_loaded = True
            else:
                print("[ISL_IMAGE] ✗ Failed to load ISL_IMAGE model")
        else:
            _isl_image_model_loaded = True


class _LandmarkPoint:
    """Lightweight object mimicking MediaPipe landmark with .x, .y, .z attributes."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _HandLandmarks:
    """Lightweight object mimicking MediaPipe hand_landmarks with .landmark list."""
    __slots__ = ('landmark',)

    def __init__(self, landmark_list):
        self.landmark = landmark_list


def _reconstruct_landmarks(landmark_data):
    """Convert raw landmark array from frontend into a MediaPipe-compatible object.

    Args:
        landmark_data: list of 21 dicts with {x, y, z} (normalised 0-1 coords)
                       OR list of 63 floats [x0, y0, z0, x1, y1, z1, ...]

    Returns:
        _HandLandmarks object compatible with predict_from_hand_crop(),
        or None if the data is invalid.
    """
    if not landmark_data:
        return None

    points = []

    # Format A: list of 21 dicts with {x, y, z}
    if isinstance(landmark_data[0], dict):
        for lm in landmark_data:
            points.append(_LandmarkPoint(
                float(lm.get('x', 0)),
                float(lm.get('y', 0)),
                float(lm.get('z', 0))
            ))
    # Format B: flat list of 63 floats
    elif isinstance(landmark_data[0], (int, float)) and len(landmark_data) >= 63:
        for i in range(0, 63, 3):
            points.append(_LandmarkPoint(
                float(landmark_data[i]),
                float(landmark_data[i + 1]),
                float(landmark_data[i + 2])
            ))
    else:
        return None

    if len(points) != 21:
        return None

    return _HandLandmarks(points)


def register_socket_handlers(socketio):
    global _socketio
    _socketio = socketio

    @socketio.on('connect')
    def handle_ws_connect():
        sid = request.sid
        print(f"\n{'='*60}")
        print(f"[WS CONNECT] Client connected: {sid}")
        print(f"[WS CONNECT] Pipeline: ISL_IMAGE (MobileNetV2 + Transformer, 160×160)")
        print(f"[WS CONNECT] Confidence threshold: {PREDICTION_CONFIDENCE_THRESHOLD}")
        print(f"{'='*60}\n")

        # Ensure model is loaded
        _ensure_isl_image_model()
        model = get_isl_image_model()

        client_sessions[sid] = {
            'prediction_count': 0,
            'inference_lock': threading.Lock(),
        }

        emit('status', {
            'status': 'connected',
            'model': 'isl_image' if model.model is not None else 'none',
            'confidence_threshold': PREDICTION_CONFIDENCE_THRESHOLD,
        })

    @socketio.on('disconnect')
    def handle_ws_disconnect():
        sid = request.sid
        if sid in client_sessions:
            del client_sessions[sid]
        print(f"[WS DISCONNECT] Client disconnected: {sid}")

    @socketio.on('frame')
    def handle_ws_frame(data):
        """Receive camera frame + landmarks from client and run ISL_IMAGE inference.

        Expected data format:
          {
            'image': '<base64 JPEG string>',
            'landmarks': [ {x, y, z}, ... ] (21 points) or [x0, y0, z0, ...] (63 floats)
          }

        Pipeline:
          1. Decode base64 → OpenCV BGR image
          2. Convert BGR → RGB (CRITICAL for correct color channels)
          3. Reconstruct MediaPipe-compatible landmarks from raw data
          4. Call predict_from_hand(rgb_frame, landmarks) — crops hand & runs MobileNet
          5. If confidence > 0.65, emit 'prediction' with { sign, confidence }
        """
        try:
            sid = request.sid
            if sid not in client_sessions:
                client_sessions[sid] = {
                    'prediction_count': 0,
                    'inference_lock': threading.Lock(),
                }
            session = client_sessions[sid]

            if 'image' not in data:
                return

            # Check if landmarks are present
            landmarks_raw = data.get('landmarks')
            if not landmarks_raw:
                return  # Can't crop hand without landmarks

            # ══════════════════════════════════════════════════════════
            # STEP 1: DECODE BASE64 JPEG → OpenCV BGR frame
            # ══════════════════════════════════════════════════════════
            base64_data = data['image']
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]

            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                print(f"[FRAME ERROR] Failed to decode base64 frame from {sid[:8]}...")
                return

            # ══════════════════════════════════════════════════════════
            # STEP 2: BGR → RGB color correction (CRITICAL)
            # ══════════════════════════════════════════════════════════
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ══════════════════════════════════════════════════════════
            # STEP 3: RECONSTRUCT MEDIAPIPE-COMPATIBLE LANDMARKS
            # ══════════════════════════════════════════════════════════
            hand_landmarks = _reconstruct_landmarks(landmarks_raw)
            if hand_landmarks is None:
                print(f"[LANDMARKS ERROR] Invalid landmark data from {sid[:8]}...")
                return

            # ══════════════════════════════════════════════════════════
            # STEP 4: RUN ISL_IMAGE INFERENCE (non-blocking)
            # ══════════════════════════════════════════════════════════
            if session['inference_lock'].acquire(blocking=False):
                session['prediction_count'] += 1
                pred_num = session['prediction_count']

                # Run in background to avoid blocking the WebSocket thread
                socketio.start_background_task(
                    _run_isl_image_inference,
                    sid, frame_rgb, hand_landmarks, pred_num
                )
            # If lock is held, skip this frame (inference already running)

        except Exception as e:
            print(f"[FRAME HANDLER ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()


def _run_isl_image_inference(sid, frame_rgb, hand_landmarks, pred_num):
    """Background task: runs ISL_IMAGE model inference.

    Steps:
      1. Call predict_from_hand(frame_rgb, hand_landmarks)
         - This crops the hand region using landmark bounding box
         - Resizes to 160×160, applies MobileNetV2 preprocessing
         - Runs through the Transformer classification head
      2. If valid and confidence > 0.65, emit 'prediction'
    """
    try:
        model = get_isl_image_model()
        if model.model is None:
            print(f"[INFERENCE ERROR] ISL_IMAGE model is NOT loaded — cannot predict!")
            _socketio.emit('prediction', {
                'sign': None,
                'confidence': 0.0,
                'status': 'Model not loaded',
            }, to=sid)
            return

        # Run prediction — predict_from_hand handles cropping + preprocessing
        result = predict_from_hand(frame_rgb, hand_landmarks, is_rgb=True)

        if result is None:
            # Could not crop hand or prediction failed
            _socketio.emit('prediction', {
                'sign': None,
                'confidence': 0.0,
                'status': 'Hand crop failed',
            }, to=sid)
            return

        sign = result['sign']
        confidence = result['confidence']
        is_valid = result.get('is_valid', False)

        print(f"[INFERENCE #{pred_num}] sign='{sign}' conf={confidence:.4f} valid={is_valid}")

        # ══════════════════════════════════════════════════════════
        # STEP 5: CONFIDENCE GATE — only emit if > 0.65 AND valid
        # ══════════════════════════════════════════════════════════
        if is_valid and confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            print(f"[INFERENCE #{pred_num}] ✓ CONFIDENT: '{sign}' @ {confidence:.4f}")

            _socketio.emit('prediction', {
                'sign': sign,
                'confidence': float(confidence),
            }, to=sid)
        else:
            # Below threshold — emit diagnostic info
            _socketio.emit('prediction', {
                'sign': None,
                'confidence': float(confidence),
                'raw_sign': sign if sign != 'Detecting...' else None,
                'status': f'Low confidence: {sign} ({confidence:.2%})',
            }, to=sid)

    except Exception as e:
        print(f"\n[INFERENCE ERROR] Prediction #{pred_num} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        _socketio.emit('prediction', {
            'sign': None,
            'confidence': 0.0,
            'status': f'Inference error: {str(e)}',
        }, to=sid)

    finally:
        # Release the inference lock
        if sid in client_sessions:
            session = client_sessions[sid]
            if session['inference_lock'].locked():
                session['inference_lock'].release()
