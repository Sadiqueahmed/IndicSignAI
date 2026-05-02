import base64
import numpy as np
import cv2
import traceback
from flask_socketio import emit
from flask import request

# Import the core MODULE so we access mutable globals (hand_detector, etc.)
# via core.hand_detector — NOT by-value imports which snapshot None.
import src.core as core
from src.core import (
    fast_predict, get_stable_prediction, extract_landmarks_for_model,
    create_client_smoother, logger
)

# Manage WebSocket state per client session
client_sessions = {}

def register_socket_handlers(socketio):
    
    @socketio.on('connect')
    def handle_ws_connect():
        client_sessions[request.sid] = {
            'smoother': create_client_smoother(),
        }
        logger.info(f"Client connected: {request.sid}")
        emit('status', {
            'status': 'connected', 
            'sklearn': core.sklearn_model is not None,
            'tflite': core.tflite_interpreter is not None,
        })

    @socketio.on('disconnect')
    def handle_ws_disconnect():
        if request.sid in client_sessions:
            del client_sessions[request.sid]
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('frame')
    def handle_ws_frame(data):
        """Receive full camera frame or landmarks from client and run model."""
        try:
            sid = request.sid
            if sid not in client_sessions:
                client_sessions[sid] = {
                    'smoother': create_client_smoother(),
                }
            session = client_sessions[sid]
            
            # ── 1. FAST PATH: Landmarks directly from frontend ──
            if 'landmarks' in data:
                raw_lms = data['landmarks']
                if not raw_lms or len(raw_lms) != 21:
                    return

                # Mock an object that _bbox_normalize_landmarks can use
                class MockLandmark:
                    def __init__(self, x, y, z):
                        self.x, self.y, self.z = x, y, z
                class MockHandLandmark:
                    def __init__(self, lms):
                        self.landmark = [MockLandmark(lm['x'], lm['y'], lm.get('z', 0.0)) for lm in lms]

                mock_lm = MockHandLandmark(raw_lms)
                
                # Normalise using the exact same logic as training
                normed = core._bbox_normalize_landmarks(mock_lm)
                
                if len(normed) == 63:
                    landmarks_arr = np.array(normed, dtype=np.float32)
                    pred_class, conf, model_type = fast_predict(landmarks_arr)
                    
                    if pred_class is not None:
                        # Temporal smoothing
                        stable_pred = get_stable_prediction(
                            pred_class, conf,
                            smoother=session['smoother']
                        )
                        
                        # ALWAYS emit to keep frontend diagnostic panel alive
                        emit('prediction', {
                            'sign': stable_pred,  # Will be None if not stable
                            'confidence': float(conf),
                            'raw_confidence': float(conf),
                            'model': model_type,
                            'raw_sign': pred_class
                        })
                return

            # ── 2. SLOW PATH: Legacy Image decoding (Disabled to prevent lag) ──
            if 'image' in data:
                # We skip image decoding entirely now because it's too slow
                # and the frontend handles MediaPipe extraction.
                return
                    
        except Exception as e:
            logger.error(f"WebSocket frame error: {traceback.format_exc()}")
