"""
sockets.py — WebSocket Handlers for Real-Time CNN+BiLSTM Inference

Architecture:
  - Frontend sends raw camera frames as base64 JPEG via WebSocket 'frame' event
  - Backend decodes → preprocesses (BGR→RGB, 224×224) → appends to per-client deque(maxlen=20)
  - When deque reaches 20 frames → run model inference
  - After confident prediction, pop 5 oldest frames (sliding window)
  - Confidence threshold > 0.75 enforced before emitting result
  - Emits 'prediction_result' event with { word, confidence }
"""

import base64
import numpy as np
import cv2
import traceback
import threading
from collections import deque
from flask_socketio import emit
from flask import request

# Import the core MODULE so we access mutable globals via core.xxx
import src.core as core
from src.core import (
    logger,
    CNN_BILSTM_SEQ_LEN, CNN_BILSTM_SLIDE_POP,
    CNN_BILSTM_FRAME_SIZE, CNN_BILSTM_CLASS_MAPPING,
)

# ══════════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD — Module 2 spec: only accept predictions > 0.75
# ══════════════════════════════════════════════════════════════════════
PREDICTION_CONFIDENCE_THRESHOLD = 0.75

# Per-client session state
client_sessions = {}

# Store the socketio instance for background task usage
_socketio = None


def register_socket_handlers(socketio):
    global _socketio
    _socketio = socketio

    @socketio.on('connect')
    def handle_ws_connect():
        sid = request.sid
        print(f"\n{'='*60}")
        print(f"[WS CONNECT] Client connected: {sid}")
        print(f"[WS CONNECT] CNN+BiLSTM model loaded: {core.cnn_bilstm_model is not None}")
        print(f"[WS CONNECT] Class mapping keys: {list(core.class_mapping.keys())}")
        print(f"[WS CONNECT] Seq length required: {CNN_BILSTM_SEQ_LEN}")
        print(f"[WS CONNECT] Confidence threshold: {PREDICTION_CONFIDENCE_THRESHOLD}")
        print(f"{'='*60}\n")

        client_sessions[sid] = {
            # ── Module 1: Per-client frame buffer (deque maxlen=20) ──
            'frame_buffer': deque(maxlen=CNN_BILSTM_SEQ_LEN),
            'prediction_count': 0,
            'inference_lock': threading.Lock(),
        }

        emit('status', {
            'status': 'connected',
            'model': 'cnn_bilstm' if core.cnn_bilstm_model is not None else 'none',
            'seq_len': CNN_BILSTM_SEQ_LEN,
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
        """Receive camera frame from client and run CNN+BiLSTM inference.

        Expected data format:
          { 'image': '<base64 JPEG string>' }

        MODULE 1 — Frame Processing:
          1. Decode base64 → OpenCV BGR image
          2. Convert BGR → RGB (CRITICAL for correct color channels)
          3. Resize to 224×224
          4. Append to per-client deque(maxlen=20)

        MODULE 2 — Inference Execution:
          5. Only run when len(deque) == 20
          6. np.expand_dims to create (1, 20, 224, 224, 3) batch
          7. model.predict() → argmax → label_map lookup
          8. Confidence gate > 0.75
          9. Sliding window: popleft() × 5 after confident prediction

        MODULE 3 — Translation Handoff:
          10. Emit 'prediction_result' with { word, confidence }
        """
        try:
            sid = request.sid
            if sid not in client_sessions:
                client_sessions[sid] = {
                    'frame_buffer': deque(maxlen=CNN_BILSTM_SEQ_LEN),
                    'prediction_count': 0,
                    'inference_lock': threading.Lock(),
                }
            session = client_sessions[sid]

            if 'image' not in data:
                return

            # ══════════════════════════════════════════════════════════
            # MODULE 1: DECODE & PREPROCESS (CRITICAL BGR→RGB step)
            # ══════════════════════════════════════════════════════════

            # Step 1: Decode base64 JPEG → OpenCV BGR frame
            base64_data = data['image']
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]

            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                print(f"[FRAME ERROR] Failed to decode base64 frame from {sid}")
                return

            # Step 2: BGR → RGB color correction (CRITICAL)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Step 3: Resize strictly to 224×224
            frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Keep as uint8 — the model has a built-in Rescaling(1./255) layer
            frame_final = frame_resized.astype(np.uint8)

            print(f"[FRAME PREPROCESS] sid={sid[:8]}... | "
                  f"decoded_shape={frame_bgr.shape} → rgb_resized={frame_final.shape} | "
                  f"dtype={frame_final.dtype} | "
                  f"pixel_range=[{frame_final.min()}, {frame_final.max()}]")

            # Step 4: Append to per-client deque(maxlen=20)
            session['frame_buffer'].append(frame_final)
            buffer_len = len(session['frame_buffer'])

            print(f"[BUFFER STATUS] sid={sid[:8]}... | "
                  f"buffer_fill={buffer_len}/{CNN_BILSTM_SEQ_LEN} | "
                  f"ready={'YES' if buffer_len >= CNN_BILSTM_SEQ_LEN else 'NO'}")

            # Emit buffer progress to frontend
            emit('buffer_status', {
                'count': buffer_len,
                'required': CNN_BILSTM_SEQ_LEN,
                'ready': buffer_len >= CNN_BILSTM_SEQ_LEN
            })

            # ══════════════════════════════════════════════════════════
            # MODULE 2: SAFE INFERENCE EXECUTION
            # ══════════════════════════════════════════════════════════

            # Step 5: Only run when len(deque) == 20
            if buffer_len >= CNN_BILSTM_SEQ_LEN:
                if session['inference_lock'].acquire(blocking=False):
                    # Snapshot the buffer (copy to avoid mutation during inference)
                    frame_snapshot = list(session['frame_buffer'])
                    session['prediction_count'] += 1
                    pred_num = session['prediction_count']

                    print(f"\n{'─'*50}")
                    print(f"[INFERENCE TRIGGER] Prediction #{pred_num} for {sid[:8]}...")
                    print(f"[INFERENCE TRIGGER] Buffer snapshot: {len(frame_snapshot)} frames")
                    print(f"{'─'*50}")

                    # Run inference in a background task (non-blocking)
                    socketio.start_background_task(
                        _run_inference_background,
                        sid, frame_snapshot, pred_num
                    )
                else:
                    print(f"[INFERENCE SKIP] Lock held — inference already running for {sid[:8]}...")

        except Exception as e:
            print(f"[FRAME HANDLER ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()


def _run_inference_background(sid, frame_snapshot, pred_num):
    """Background task: runs CNN+BiLSTM inference without blocking the WebSocket thread.

    MODULE 2 — Steps 6-9:
      6. Convert list → numpy array → expand_dims → (1, 20, 224, 224, 3)
      7. model.predict() → argmax → class name from label map
      8. Confidence gate > 0.75
      9. Sliding window: popleft() × 5 after confident prediction

    MODULE 3 — Step 10:
      Emit 'prediction_result' with { word, confidence }
    """
    try:
        if core.cnn_bilstm_model is None:
            print(f"[INFERENCE ERROR] CNN+BiLSTM model is NOT loaded — cannot predict!")
            _socketio.emit('prediction_result', {
                'word': None,
                'confidence': 0.0,
                'status': 'Model not loaded',
                'prediction_num': pred_num,
            }, to=sid)
            return

        # ── Step 6: DIMENSION EXPANSION ──────────────────────────────
        # Convert list of 20 frames to numpy array: (20, 224, 224, 3)
        frames_array = np.array(frame_snapshot, dtype=np.float32)
        print(f"[INFERENCE #{pred_num}] frames_array.shape = {frames_array.shape}")
        print(f"[INFERENCE #{pred_num}] frames_array.dtype = {frames_array.dtype}")
        print(f"[INFERENCE #{pred_num}] pixel min={frames_array.min():.1f}, max={frames_array.max():.1f}")

        # MUST use np.expand_dims to create (1, 20, 224, 224, 3)
        batch = np.expand_dims(frames_array, axis=0)
        print(f"[INFERENCE #{pred_num}] batch.shape (after expand_dims) = {batch.shape}")
        print(f"[INFERENCE #{pred_num}] EXPECTED shape = (1, {CNN_BILSTM_SEQ_LEN}, 224, 224, 3)")

        # Verify shape is correct
        expected_shape = (1, CNN_BILSTM_SEQ_LEN, 224, 224, 3)
        if batch.shape != expected_shape:
            print(f"[INFERENCE #{pred_num}] ⚠ SHAPE MISMATCH! Got {batch.shape}, expected {expected_shape}")
            return

        # ── Step 7: MODEL PREDICT + LABEL DECODE ─────────────────────
        import tensorflow as tf
        batch_tensor = tf.constant(batch)
        predictions = core.cnn_bilstm_model(batch_tensor, training=False)[0].numpy()

        print(f"[INFERENCE #{pred_num}] predictions.shape = {predictions.shape}")
        print(f"[INFERENCE #{pred_num}] predictions = [{', '.join(f'{p:.6f}' for p in predictions)}]")

        # argmax to find highest probability class
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])

        print(f"[INFERENCE #{pred_num}] argmax class_idx = {class_idx}")
        print(f"[INFERENCE #{pred_num}] confidence = {confidence:.6f}")

        # Label decode: use class_mapping.json (str keys) → fallback to built-in mapping
        predicted_word = core.class_mapping.get(
            str(class_idx),
            CNN_BILSTM_CLASS_MAPPING.get(class_idx, f"Class_{class_idx}")
        )
        print(f"[INFERENCE #{pred_num}] predicted_word = '{predicted_word}'")

        # ── Step 8: CONFIDENCE THRESHOLD > 0.75 ─────────────────────
        if confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            print(f"\n[INFERENCE #{pred_num}] ✓ CONFIDENT PREDICTION: '{predicted_word}' @ {confidence:.4f}")
            print(f"[INFERENCE #{pred_num}] ✓ Threshold {PREDICTION_CONFIDENCE_THRESHOLD} PASSED")

            # ── Step 9: SLIDING WINDOW — popleft() × 5 ──────────────
            if sid in client_sessions:
                session = client_sessions[sid]
                pops = min(CNN_BILSTM_SLIDE_POP, len(session['frame_buffer']))
                for _ in range(pops):
                    session['frame_buffer'].popleft()
                buffer_after = len(session['frame_buffer'])
                print(f"[SLIDING WINDOW] Popped {pops} frames | buffer now: {buffer_after}/{CNN_BILSTM_SEQ_LEN}")

            # ── Step 10 (MODULE 3): EMIT prediction_result ───────────
            _socketio.emit('prediction_result', {
                'word': predicted_word,
                'confidence': float(confidence),
            }, to=sid)
            print(f"[EMIT] prediction_result → {{ word: '{predicted_word}', confidence: {confidence:.4f} }}")

        else:
            print(f"\n[INFERENCE #{pred_num}] ✗ BELOW THRESHOLD: '{predicted_word}' @ {confidence:.4f}")
            print(f"[INFERENCE #{pred_num}] ✗ Need > {PREDICTION_CONFIDENCE_THRESHOLD}, got {confidence:.4f}")

            # Emit diagnostic (no 'word') so frontend can show status
            _socketio.emit('prediction_result', {
                'word': None,
                'confidence': float(confidence),
                'raw_sign': predicted_word,
                'status': f'Low confidence: {predicted_word} ({confidence:.2%})',
                'prediction_num': pred_num,
            }, to=sid)

    except Exception as e:
        print(f"\n[INFERENCE ERROR] Prediction #{pred_num} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        _socketio.emit('prediction_result', {
            'word': None,
            'confidence': 0.0,
            'status': f'Inference error: {str(e)}',
            'prediction_num': pred_num,
        }, to=sid)

    finally:
        # Release the inference lock
        if sid in client_sessions:
            session = client_sessions[sid]
            if session['inference_lock'].locked():
                session['inference_lock'].release()
                print(f"[LOCK] Released inference lock for {sid[:8]}...")
