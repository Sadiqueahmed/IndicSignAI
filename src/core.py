"""
core.py — Global Application State & Utilities
Contains all model loading, initialization, global variables,
and reusable inference/translation functions to be shared across Blueprints.
"""

import os
import sys
import cv2
import json
import uuid
import numpy as np
import logging
from collections import deque, Counter

# MediaPipe
import mediapipe as mp

# Internal modules
from .models.bengali_to_meitei import ensure_meitei_mayek

logger = logging.getLogger(__name__)

# ==========================================
# GLOBAL STATE & CONFIG
# ==========================================

# Models & MediaPipe Instances
camera = None
hand_detector = None
tflite_interpreter = None
tflite_input_details = None
tflite_output_details = None
tf_model = None
sklearn_model = None          # RandomForestClassifier / sklearn model

# NLP & Translation
translation_engine = None
_deep_translator_available = False

class_mapping = {}

# Paths — ABSOLUTE resolution based on this file's directory to prevent
# WinError 2 when Flask's CWD differs from the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_THIS_DIR)  # project root
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ISL_VIDEO_DIR = os.path.join(BASE_DIR, 'INDIAN SIGN LANGUAGE ANIMATED VIDEOS',
                             'INDIAN SIGN LANGUAGE ANIMATED VIDEOS')

# Inference parameters
CONFIDENCE_GATE = 0.75        # require high confidence before accepting
CONSECUTIVE_REQUIRED = 3
SMOOTH_BUFFER_SIZE = 15       # temporal smoothing window (frames)
SMOOTH_AGREEMENT_PCT = 0.70   # 70% of buffer must agree

# Video processing
MP_CONFIDENCE_GATE = 0.70
MODEL_CONFIDENCE_MIN = 0.30

# ==========================================
# LANGUAGE DEFINITIONS
# ==========================================
LANGUAGE_NAMES = {
    'assamese': 'Assamese',
    'hindi': 'Hindi',
    'manipuri': 'Manipuri (Meitei)',
    'dzongkha': 'Dzongkha',
    'nepali': 'Nepali',
    'english': 'English'
}

DEEP_TRANSLATOR_CODES = {
    'assamese': 'as',
    'hindi': 'hi',
    'manipuri': 'mni-Mtei',
    'dzongkha': 'dz',
    'nepali': 'ne',
    'english': 'en',
}

STT_LANG_CODES = {
    'assamese': 'as-IN',
    'hindi': 'hi-IN',
    'manipuri': 'mni-IN',
    'dzongkha': 'dz-BT',
    'nepali': 'ne-NP',
    'english': 'en-US',
}

# ==========================================
# INITIALIZATION
# ==========================================

def _resolve_model_path(relative_path, description="Model"):
    """Resolve a model path to an absolute path, with diagnostic logging.
    
    Tries multiple candidate locations:
      1. Exactly as given (already absolute or relative to CWD)
      2. Relative to MODELS_DIR
      3. Relative to BASE_DIR (project root)
    """
    candidates = [
        os.path.abspath(relative_path),
        os.path.join(MODELS_DIR, os.path.basename(relative_path)),
        os.path.join(BASE_DIR, os.path.basename(relative_path)),
        os.path.join(BASE_DIR, relative_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  [RESOLVED] {description}: {p}")
            return p
    # None found — print every path we tried
    print(f"  [ERROR] FILE NOT FOUND for {description}. Tried:")
    for p in candidates:
        print(f"    FILE NOT FOUND AT PATH: {p}")
    return None


def load_models():
    global hand_detector, tf_model, sklearn_model, class_mapping
    global tflite_interpreter, tflite_input_details, tflite_output_details
    global translation_engine, _deep_translator_available

    print("\n" + "="*50)
    print("INITIALIZING INDICSIGNAI MODELS")
    print(f"  BASE_DIR  = {BASE_DIR}")
    print(f"  MODELS_DIR = {MODELS_DIR}")
    print(f"  CWD        = {os.getcwd()}")
    print("="*50)

    # 1. MediaPipe
    mp_hands = mp.solutions.hands
    hand_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

    # 2. Class Mapping — absolute path resolution
    try:
        map_path = _resolve_model_path('class_mapping.json', 'Class Mapping')
        if map_path:
            with open(map_path, 'r') as f:
                class_mapping = json.load(f)
            print(f"[OK] Loaded {len(class_mapping)} classes from {map_path}")
        else:
            print("[WARNING] class_mapping.json not found at any candidate path")
    except Exception as e:
        print(f"[X] Failed to load class_mapping.json: {type(e).__name__}: {e}")

    # 3a. Sklearn / Joblib Model — PRIMARY (trained RandomForest on 63-dim landmarks)
    pkl_path = _resolve_model_path('sign_language_model.pkl', 'Sklearn Model')
    if pkl_path:
        try:
            import joblib
            sklearn_model = joblib.load(pkl_path)
            
            # ── MONKEY PATCH FOR SCIKIT-LEARN 1.8.0 COMPATIBILITY ──
            # The model was trained in 1.3.0. In 1.4+, DecisionTrees require the
            # 'monotonic_cst' attribute. We manually inject it to prevent predict() from crashing.
            if hasattr(sklearn_model, 'estimators_'):
                for estimator in sklearn_model.estimators_:
                    if not hasattr(estimator, 'monotonic_cst'):
                        estimator.monotonic_cst = None
                        
            n_feat = getattr(sklearn_model, 'n_features_in_', '?')
            n_cls = getattr(sklearn_model, 'n_classes_', '?')
            print(f"[OK] Sklearn model loaded: {type(sklearn_model).__name__} "
                  f"(features={n_feat}, classes={n_cls}) from {pkl_path}")
        except Exception as e:
            print(f"[X] Sklearn model failed: {type(e).__name__}: {e}")
            sklearn_model = None
    else:
        print("[WARNING] No sklearn .pkl model found")

    # 3b. TFLite Model (fallback)
    tflite_path = _resolve_model_path('sign_language_model.tflite', 'TFLite Model')
    if tflite_path is None:
        tflite_path = _resolve_model_path('ISL_IMAGE.tflite', 'TFLite Model (ISL_IMAGE fallback)')

    if tflite_path:
        try:
            import tensorflow as tf
            tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
            tflite_interpreter.allocate_tensors()
            tflite_input_details = tflite_interpreter.get_input_details()
            tflite_output_details = tflite_interpreter.get_output_details()
            print(f"[OK] TFLite model loaded successfully from {tflite_path}")
        except Exception as e:
            print(f"[X] TFLite model failed: {type(e).__name__}: {e}")
            tflite_interpreter = None

    # 3c. Keras fallback
    model_path = _resolve_model_path('sign_language_model.keras', 'Keras Model')
    if model_path:
        try:
            from tensorflow.keras.models import load_model
            tf_model = load_model(model_path, compile=False)
            print(f"[OK] Keras model loaded from {model_path}")
        except Exception as e:
            print(f"[X] Could not load Keras model: {type(e).__name__}: {e}")

    # 4. Deep Translator
    try:
        from deep_translator import GoogleTranslator
        _deep_translator_available = True
        print("[OK] deep-translator loaded")
    except ImportError:
        _deep_translator_available = False
        print("[X] deep-translator not available")

    # 5. Translation Engine
    try:
        from .models.translation import TranslationModel
        translation_engine = TranslationModel()
        print("[OK] Translation engine (TranslationModel) loaded")
    except Exception as e:
        import traceback
        print(f"[X] Failed to load translation engine: {type(e).__name__}: {e}")
        traceback.print_exc()
        translation_engine = None

    print("="*50 + "\n")

# ==========================================
# COMPUTER VISION & INFERENCE UTILS
# ==========================================

def get_camera():
    """Open camera with Windows DirectShow backend to prevent WinError 2."""
    global camera
    if camera is None or not camera.isOpened():
        if sys.platform == 'win32':
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("[get_camera] Failed to open camera device 0")
    return camera


def _bbox_normalize_landmarks(hand_landmarks_obj):
    """Bounding-box normalise a single hand's 21 landmarks to [0,1].
    
    Produces scale- and position-invariant features so a sign
    at 30 cm and 150 cm from the camera yields identical vectors.
    
    Returns flat list of 63 floats (21 landmarks × 3 coords).
    """
    xs = [lm.x for lm in hand_landmarks_obj.landmark]
    ys = [lm.y for lm in hand_landmarks_obj.landmark]
    zs = [lm.z for lm in hand_landmarks_obj.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    x_range = (x_max - x_min) or 1e-6
    y_range = (y_max - y_min) or 1e-6
    z_range = (z_max - z_min) or 1e-6

    normed = []
    for lm in hand_landmarks_obj.landmark:
        normed.append((lm.x - x_min) / x_range)
        normed.append((lm.y - y_min) / y_range)
        normed.append((lm.z - z_min) / z_range)
    return normed


def _wrist_normalize_landmarks(hand_landmarks_obj, image_width, image_height):
    """Wrist-relative normalisation matching the training data pipeline.
    
    Algorithm (matches utils/hand_landmark_utils.py pre_process_landmark):
      1. Convert MediaPipe normalised coords to pixel coordinates.
      2. Subtract the wrist (landmark 0) position from all landmarks.
      3. Flatten to 1-D list and divide by max absolute value.
    
    Returns flat list of 42 floats (21 landmarks × 2 coords: x, y).
    """
    import copy
    import itertools

    # Step 1: Convert to pixel coords
    landmark_point = []
    for lm in hand_landmarks_obj.landmark:
        lx = min(int(lm.x * image_width), image_width - 1)
        ly = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])

    # Step 2: Subtract wrist position
    temp = copy.deepcopy(landmark_point)
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    # Step 3: Flatten and normalise by max absolute value
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat)) if flat else 1
    if max_val > 0:
        flat = [n / max_val for n in flat]
    return flat


def extract_landmarks_for_model(frame, results):
    """Extract and normalise landmarks for the PRIMARY hand.
    
    Uses bounding-box normalisation producing a 63-dim vector
    (21 landmarks × 3 coords: x, y, z) — matching the universal
    landmark_utils.py pipeline used in training.
    
    Returns numpy array of shape (63,) or None.
    """
    if not results.multi_hand_landmarks:
        return None

    # Use the first (primary) detected hand
    hand_lm = results.multi_hand_landmarks[0]
    normed = _bbox_normalize_landmarks(hand_lm)

    # ── DIAGNOSTIC: Log raw landmark count and normalised vector length ──
    print(f"[DIAG extract_landmarks] hands_detected={len(results.multi_hand_landmarks)}, "
          f"raw_landmarks={len(hand_lm.landmark)}, normed_len={len(normed)}")

    if len(normed) == 63:
        arr = np.array(normed, dtype=np.float32)
        print(f"[DIAG extract_landmarks] output shape={arr.shape}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
        return arr
    else:
        print(f"[DIAG extract_landmarks] SHAPE MISMATCH: expected 63, got {len(normed)}")
    return None


_tflite_shape_warned = False

def fast_predict(landmark_vector):
    """Predict from a single-frame (63,) landmark vector.
    
    Priority order:
      1. Sklearn model (RandomForest .pkl) — PRIMARY
      2. TFLite interpreter — if input shape matches
      3. Keras model — fallback
    
    Returns (class_name, confidence, model_type) or (None, 0.0, None).
    """
    global _tflite_shape_warned
    try:
        if landmark_vector is None:
            return None, 0.0, None

        input_1d = landmark_vector.flatten().astype(np.float32)

        # ── DIAGNOSTIC: Log input shape before any model call ──
        print(f"[DIAG fast_predict] input_1d.shape={input_1d.shape}, "
              f"dtype={input_1d.dtype}")

        # ── 1. Sklearn Model (PRIMARY) ───────────────────────────────
        if sklearn_model is not None:
            try:
                X = input_1d.reshape(1, -1)
                n_feat = getattr(sklearn_model, 'n_features_in_', '?')
                print(f"[DIAG fast_predict] sklearn expects {n_feat} features, "
                      f"input has {X.shape[1]}")
                pred_class_idx = int(sklearn_model.predict(X)[0])
                # Get probability for confidence score
                if hasattr(sklearn_model, 'predict_proba'):
                    proba = sklearn_model.predict_proba(X)[0]
                    confidence = float(proba.max())
                    # ── DIAGNOSTIC: Log full probability distribution ──
                    proba_str = ', '.join(f"{p:.4f}" for p in proba)
                    print(f"[DIAG fast_predict] sklearn proba=[{proba_str}], "
                          f"pred_idx={pred_class_idx}, confidence={confidence:.4f}")
                else:
                    confidence = 1.0
                class_name = class_mapping.get(str(pred_class_idx), f"Class_{pred_class_idx}")
                print(f"[DIAG fast_predict] sklearn result: '{class_name}' @ {confidence:.4f}")
                return class_name, confidence, "sklearn"
            except Exception as e:
                logger.warning(f"[fast_predict] sklearn failed: {e}")
                import traceback
                traceback.print_exc()

        # ── 2. TFLite ────────────────────────────────────────────────
        if tflite_interpreter:
            input_data = input_1d.reshape(1, -1)
            expected_shape = tuple(tflite_input_details[0]['shape'])
            print(f"[DIAG fast_predict] TFLite expects shape {expected_shape}, "
                  f"input has {input_data.shape}")
            if input_data.shape == expected_shape:
                tflite_interpreter.set_tensor(tflite_input_details[0]['index'], input_data)
                tflite_interpreter.invoke()
                predictions = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])[0]
                class_idx = np.argmax(predictions)
                confidence = float(predictions[class_idx])
                class_name = class_mapping.get(str(class_idx), f"Class_{class_idx}")
                print(f"[DIAG fast_predict] TFLite result: '{class_name}' @ {confidence:.4f}")
                return class_name, confidence, "tflite"
            elif not _tflite_shape_warned:
                logger.warning(
                    f"[fast_predict] TFLite shape mismatch: "
                    f"expects {expected_shape}, got {input_data.shape}. Skipping."
                )
                _tflite_shape_warned = True

        # ── 3. Keras ─────────────────────────────────────────────────
        if tf_model:
            input_data = input_1d.reshape(1, -1)
            print(f"[DIAG fast_predict] Keras input shape={input_data.shape}")
            predictions = tf_model.predict(input_data, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            class_name = class_mapping.get(str(class_idx), f"Class_{class_idx}")
            print(f"[DIAG fast_predict] Keras result: '{class_name}' @ {confidence:.4f}")
            return class_name, confidence, "keras"

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
    return None, 0.0, None


def create_client_smoother():
    """Create a new per-client temporal smoothing state.
    
    Returns a dict with 'history' and 'confidence' deques.
    Each WebSocket client gets its own smoother so predictions
    don't corrupt across sessions.
    """
    return {
        'history': deque(maxlen=SMOOTH_BUFFER_SIZE),
        'confidence': deque(maxlen=SMOOTH_BUFFER_SIZE),
    }


def get_stable_prediction(new_pred, new_conf, smoother=None):
    """Temporal smoothing using a rolling Counter buffer.
    
    Only outputs a prediction when:
      - The prediction confidence is >= CONFIDENCE_GATE (0.75)
      - The same sign appears in >= 70% of the last 15 frames
    
    Args:
        new_pred:  predicted class name string
        new_conf:  confidence score from model
        smoother:  per-client smoother dict (from create_client_smoother).
                   If None, uses a fallback single-client buffer.
    
    Returns the stable prediction string, or None if not yet stable.
    """
    # Only append to buffer if confidence meets the gate
    if new_conf < CONFIDENCE_GATE:
        return None

    if smoother is None:
        # Fallback for legacy generate_frames() path
        smoother = _fallback_smoother

    smoother['history'].append(new_pred)
    smoother['confidence'].append(new_conf)

    if len(smoother['history']) < 3:
        return None

    counts = Counter(smoother['history'])
    most_common_sign, most_common_count = counts.most_common(1)[0]
    agreement_ratio = most_common_count / len(smoother['history'])

    if agreement_ratio >= SMOOTH_AGREEMENT_PCT:
        # Compute average confidence for the dominant sign
        sign_confs = [
            c for pred, c in zip(smoother['history'], smoother['confidence'])
            if pred == most_common_sign
        ]
        avg_conf = sum(sign_confs) / len(sign_confs) if sign_confs else 0.0
        if avg_conf >= CONFIDENCE_GATE:
            return most_common_sign

    return None

# Fallback smoother for the MJPEG generate_frames() path
_fallback_smoother = create_client_smoother()


def generate_frames():
    """MJPEG frame generator for legacy server-side camera feed.
    Uses CAP_DSHOW on Windows and bbox-normalized landmarks.
    """
    cap = get_camera()
    if cap is None or not cap.isOpened():
        logger.error("[generate_frames] Camera failed to open. Aborting.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(frame_rgb)

        display_text = "No Sign Detected"
        color = (0, 0, 255)

        if results.multi_hand_landmarks:
            landmarks = extract_landmarks_for_model(frame, results)

            if landmarks is not None:
                pred_class, conf, _ = fast_predict(landmarks)

                if pred_class is not None:
                    stable_pred = get_stable_prediction(pred_class, conf)
                    if stable_pred:
                        display_text = f"{stable_pred} ({conf:.0%})"
                        color = (0, 255, 0)

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# TRANSLATION & NLP UTILS
# ==========================================

from .models.nlp_grammar import correct_sentence

def _translate_to_english(text, source_lang):
    global translation_engine
    if source_lang == 'english':
        return text

    src_code = DEEP_TRANSLATOR_CODES.get(source_lang)

    # --- Primary: deep-translator (GoogleTranslator) ---
    if src_code is not None:
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source=src_code, target='en').translate(text)
            if translated and translated.strip():
                logger.info(f'[deep-translator] {source_lang} -> en: "{text}" -> "{translated}"')
                return translated
        except ImportError:
            logger.warning('[deep-translator] Package not installed. Run: pip install deep-translator')
        except Exception as e:
            import traceback
            logger.error(f'[deep-translator] Translation failed for source="{source_lang}" input="{text}": {type(e).__name__}: {e}')
            traceback.print_exc()

    # --- Fallback: NLLB TranslationModel ---
    if translation_engine:
        try:
            translated = translation_engine.translate_regional_to_english(text, source_lang)
            if translated and not translated.startswith('['):
                return translated
        except Exception as e:
            import traceback
            logger.error(f'[TranslationModel] Reverse translation failed for source="{source_lang}" input="{text}": {type(e).__name__}: {e}')
            traceback.print_exc()
    else:
        logger.error('[_translate_to_english] translation_engine is None — TranslationModel failed to load at startup.')

    return text

def _map_words_to_videos(corrected_text):
    """Map each word in corrected_text to an ISL video URL.
    
    Uses a case-insensitive filename lookup so that "hello" → "Hello.mp4"
    regardless of how the filename is stored on disk.
    Returns (sequence, words) where sequence is a list of dicts.
    """
    words = corrected_text.split()
    sequence = []

    # Build case-insensitive lookup map once per call
    lower_map = {}
    if os.path.isdir(ISL_VIDEO_DIR):
        for fname in os.listdir(ISL_VIDEO_DIR):
            lower_map[fname.lower()] = fname
        logger.info(f"[_map_words_to_videos] ISL_VIDEO_DIR has {len(lower_map)} files: {ISL_VIDEO_DIR}")
    else:
        logger.error(f"[_map_words_to_videos] ISL_VIDEO_DIR does not exist: {ISL_VIDEO_DIR}")

    for word in words:
        # Generate candidate filenames in priority order
        candidates = [
            word.capitalize() + '.mp4',
            word.upper() + '.mp4',
            word.lower() + '.mp4',
            word + '.mp4',
        ]
        found = False
        for candidate in candidates:
            resolved = lower_map.get(candidate.lower())
            if resolved:
                sequence.append({
                    'word': word,
                    'video_url': f'/api/video/{resolved}',
                    'has_video': True
                })
                logger.info(f"[_map_words_to_videos] '{word}' → {resolved}")
                found = True
                break

        if not found:
            logger.warning(f"[_map_words_to_videos] No video for word: '{word}' (tried: {candidates})")
            sequence.append({
                'word': word,
                'video_url': None,
                'has_video': False
            })

    return sequence, words
