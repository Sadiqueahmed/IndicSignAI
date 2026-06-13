"""
core.py — Global Application State & Utilities
Contains all model loading, initialization, global variables,
and reusable inference/translation functions to be shared across Blueprints.

=== MODEL ARCHITECTURE (CNN+BiLSTM) ===
The PRIMARY model is CNN+BiLstm.keras which uses:
  - EfficientNetB0 backbone (TimeDistributed)
  - BiLSTM temporal sequence layer
  - Input shape: (batch, 20, 224, 224, 3) — 20 RGB frames at 224x224
  - Built-in Rescaling layer (1./255), so pass 0-255 uint8 pixel values
  - Output: softmax over 6 classes

The OLD sklearn/TFLite/Keras landmark-based pipeline is preserved but
commented out. Uncomment the relevant sections to revert.
"""

import os
import sys
import cv2
import json
import uuid
import base64
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

# ── NEW: CNN+BiLSTM Model ──
cnn_bilstm_model = None
CNN_BILSTM_FRAME_SIZE = (224, 224)   # Model expects 224x224 RGB frames
CNN_BILSTM_SEQ_LEN = 20             # Model expects 20-frame sequences
CNN_BILSTM_SLIDE_POP = 5            # Pop 5 frames after prediction (sliding window)

# CNN+BiLSTM class mapping — matches the training labels
# (same order as models/class_mapping.json)
CNN_BILSTM_CLASS_MAPPING = {
    0: "INDIA",
    1: "NO",
    2: "GOOD",
    3: "HELLO",
    4: "HELP",
    5: "THANK"
}

# ── OLD Models (preserved for fallback) ──
camera = None
hand_detector = None
# tflite_interpreter = None          # COMMENTED OUT — old landmark pipeline
# tflite_input_details = None
# tflite_output_details = None
# tf_model = None                    # COMMENTED OUT — old .keras/.h5 fallback
# sklearn_model = None               # COMMENTED OUT — old RandomForest on landmarks

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

# ── CNN+BiLSTM model path ──
CNN_BILSTM_MODEL_DIR = os.path.join(BASE_DIR, 'CNN+BiLstm.keras')

# Inference parameters (used by temporal smoothing)
CONFIDENCE_GATE = 0.85              # Strict gate — only accept high-confidence predictions
CONSECUTIVE_REQUIRED = 2
SMOOTH_BUFFER_SIZE = 3              # Short buffer — fast stabilisation with strict gate
SMOOTH_AGREEMENT_PCT = 0.80         # 80% agreement over the buffer

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
    global hand_detector, class_mapping
    global cnn_bilstm_model
    global translation_engine, _deep_translator_available

    # ── OLD globals preserved but not used ──
    # global tf_model, sklearn_model
    # global tflite_interpreter, tflite_input_details, tflite_output_details

    print("\n" + "="*60)
    print("INITIALIZING INDICSIGNAI MODELS (CNN+BiLSTM Pipeline)")
    print(f"  BASE_DIR  = {BASE_DIR}")
    print(f"  MODELS_DIR = {MODELS_DIR}")
    print(f"  CNN_BILSTM = {CNN_BILSTM_MODEL_DIR}")
    print(f"  CWD        = {os.getcwd()}")
    print("="*60)

    # 1. MediaPipe Hands (OPTIONAL for CNN+BiLSTM pipeline)
    # The CNN+BiLSTM model processes raw video frames and does NOT need
    # server-side MediaPipe. MediaPipe only runs on the frontend (JS) for
    # visual skeleton drawing. We still try to init it here for the legacy
    # /api/process-video fallback path, but failures are non-fatal.
    try:
        mp_hands = mp.solutions.hands
        hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        print("[OK] MediaPipe Hands initialized (legacy fallback)")
    except Exception as e:
        print(f"[WARNING] MediaPipe Hands init failed (non-fatal for CNN+BiLSTM): {e}")
        hand_detector = None

    # 2. Class Mapping — absolute path resolution
    try:
        map_path = _resolve_model_path('class_mapping.json', 'Class Mapping')
        if map_path:
            with open(map_path, 'r') as f:
                class_mapping = json.load(f)
            print(f"[OK] Loaded {len(class_mapping)} classes from {map_path}")
        else:
            print("[WARNING] class_mapping.json not found — using built-in CNN_BILSTM_CLASS_MAPPING")
            class_mapping = {str(k): v for k, v in CNN_BILSTM_CLASS_MAPPING.items()}
    except Exception as e:
        print(f"[X] Failed to load class_mapping.json: {type(e).__name__}: {e}")
        class_mapping = {str(k): v for k, v in CNN_BILSTM_CLASS_MAPPING.items()}

    # ══════════════════════════════════════════════════════════════
    # 3. PRIMARY MODEL: CNN+BiLSTM
    #
    # Architecture (from config.json analysis):
    #   Input(20, 224, 224, 3)
    #   → TimeDistributed(EfficientNetB0, pooling='avg')  → (batch, 20, 1280)
    #   → TimeDistributed(Dense(512))                     → (batch, 20, 512)
    #   → TimeDistributed(LayerNormalization)              → (batch, 20, 512)
    #   → Bidirectional(LSTM(256))                         → (batch, 512)
    #   → Dense(512, relu) → BatchNorm → Dropout(0.4)
    #   → Dense(256, relu) → Dropout(0.3)
    #   → Dense(50, softmax)
    #
    # Loading strategy:
    #   1. Try loading pre-converted CNN+BiLstm_v3.keras (clean Keras 3 file)
    #   2. Fall back to programmatic rebuild + manual H5 weight transfer
    # ══════════════════════════════════════════════════════════════

    # Paths to try
    converted_model_path = os.path.join(BASE_DIR, 'CNN+BiLstm_v3.keras')
    legacy_model_dir = CNN_BILSTM_MODEL_DIR  # CNN+BiLstm.keras directory
    legacy_weights = os.path.join(legacy_model_dir, 'model.weights.h5')

    # ── Strategy 1: Try loading pre-converted clean .keras file ──
    if os.path.isfile(converted_model_path):
        try:
            import tensorflow as tf
            from tensorflow import keras
            print(f"[...] Loading CNN+BiLSTM from converted file: {converted_model_path}")
            cnn_bilstm_model = keras.models.load_model(converted_model_path, compile=False)
            print(f"[OK] CNN+BiLSTM loaded from converted .keras file")
        except Exception as e:
            import traceback
            print(f"[X] Failed to load converted model: {type(e).__name__}: {e}")
            print(f"    Deleting stale converted file and falling back to rebuild...")
            traceback.print_exc()
            cnn_bilstm_model = None
            # Delete the broken converted file so we don't try it again
            try:
                os.remove(converted_model_path)
                print(f"    [OK] Deleted stale {converted_model_path}")
            except OSError:
                pass

    # ── Strategy 2: Programmatic rebuild + manual H5 weight transfer ──
    # Runs if Strategy 1 didn't produce a model (file missing OR load failed)
    if cnn_bilstm_model is None and os.path.isdir(legacy_model_dir) and os.path.isfile(legacy_weights):
        try:
            import tensorflow as tf
            from tensorflow import keras
            from keras import layers, models
            from collections import defaultdict
            import h5py

            print(f"[...] Rebuilding CNN+BiLSTM from architecture + H5 weights")
            print(f"      weights: {legacy_weights}")

            # Step A: Read ALL H5 weight datasets into a flat dict
            h5_datasets = {}
            with h5py.File(legacy_weights, 'r') as f:
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and \
                       not name.startswith(('optimizer/', 'metrics/')):
                        # Normalise path separators
                        h5_datasets[name.replace('\\', '/')] = np.array(obj)
                f.visititems(_collect)

            print(f"  [INFO] Found {len(h5_datasets)} weight arrays in H5 file")

            # Helper: get weight arrays from a specific H5 path prefix
            def _get_vars_at(prefix):
                """Collect vars/N arrays under a given path prefix, sorted by N."""
                results = []
                for path, arr in h5_datasets.items():
                    if path.startswith(prefix) and '/vars/' in path:
                        idx_str = path.rsplit('/vars/', 1)[-1]
                        try:
                            idx = int(idx_str)
                            results.append((idx, arr))
                        except ValueError:
                            pass
                results.sort(key=lambda x: x[0])
                return [v for _, v in results]

            # Shorthand for top-level layer groups
            _LCD = '_layer_checkpoint_dependencies/'

            def _get_layer_vars(layer_name):
                return _get_vars_at(f"{_LCD}{layer_name}/vars/")

            # Step B: Build exact architecture
            #
            # Keras 3 BREAKS TimeDistributed(Functional), so we use a custom
            # subclassed layer that manually merges batch+time dimensions,
            # applies the EfficientNet backbone, and reshapes back.
            import tensorflow as tf

            efficientnet = keras.applications.EfficientNetB0(
                include_top=False, weights='imagenet',
                input_shape=(CNN_BILSTM_FRAME_SIZE[0], CNN_BILSTM_FRAME_SIZE[1], 3),
                pooling='avg'
            )
            efficientnet.trainable = False

            class TimeDistributedBackbone(layers.Layer):
                """Apply a backbone model across the time dimension.
                
                Manually reshapes (batch, time, H, W, C) → (batch*time, H, W, C),
                applies the backbone, then reshapes back to (batch, time, features).
                This avoids the Keras 3 bug where TimeDistributed(Functional) fails.
                """
                def __init__(self, backbone, seq_len, **kwargs):
                    super().__init__(**kwargs)
                    self.backbone = backbone
                    self.seq_len = seq_len

                def call(self, inputs):
                    batch_size = tf.shape(inputs)[0]
                    # Merge batch + time → (batch*time, H, W, C)
                    x = tf.reshape(inputs, [-1] + list(inputs.shape[2:]))
                    # Apply backbone → (batch*time, features)
                    x = self.backbone(x, training=False)
                    # Reshape back → (batch, time, features)
                    feat_dim = x.shape[-1]
                    x = tf.reshape(x, [batch_size, self.seq_len, feat_dim])
                    return x

                def compute_output_shape(self, input_shape):
                    # input_shape: (batch, time, H, W, C)
                    return (input_shape[0], self.seq_len, 1280)

                def get_config(self):
                    config = super().get_config()
                    config.update({
                        'seq_len': self.seq_len,
                    })
                    return config

            inp = layers.Input(
                shape=(CNN_BILSTM_SEQ_LEN,) + CNN_BILSTM_FRAME_SIZE + (3,),
                name='input_2'
            )
            # EfficientNet across time frames
            x = TimeDistributedBackbone(
                efficientnet, CNN_BILSTM_SEQ_LEN,
                name='time_distributed'
            )(inp)
            # Dense + LayerNorm work fine with TimeDistributed in Keras 3
            x = layers.TimeDistributed(layers.Dense(512), name='time_distributed_2')(x)
            x = layers.TimeDistributed(layers.LayerNormalization(), name='time_distributed_3')(x)
            x = layers.Bidirectional(layers.LSTM(256), name='bidirectional')(x)
            x = layers.Dense(512, activation='relu', name='dense_1')(x)
            x = layers.BatchNormalization(name='batch_normalization_top')(x)
            x = layers.Dropout(0.4, name='dropout')(x)
            x = layers.Dense(256, activation='relu', name='dense_2')(x)
            x = layers.Dropout(0.3, name='dropout_1')(x)
            out = layers.Dense(50, activation='softmax', name='dense_3')(x)
            cnn_bilstm_model = models.Model(inp, out, name='model')
            print(f"  [OK] Architecture built successfully")

            # Step C: Assign non-EfficientNet weights from H5
            weight_log = []

            # TimeDistributed Dense(1280→512) — H5: time_distributed_3/layer/vars/
            arrs = _get_vars_at(f"{_LCD}time_distributed_3/layer/vars/")
            if arrs:
                cnn_bilstm_model.get_layer('time_distributed_2').layer.set_weights(arrs)
                weight_log.append(f'TD-Dense(512) [{len(arrs)} arrays]')
            else:
                print("  [WARN] No weights found for TD-Dense(512)")

            # TimeDistributed LayerNorm — H5: time_distributed_5/layer/vars/
            arrs = _get_vars_at(f"{_LCD}time_distributed_5/layer/vars/")
            if arrs:
                cnn_bilstm_model.get_layer('time_distributed_3').layer.set_weights(arrs)
                weight_log.append(f'TD-LayerNorm [{len(arrs)} arrays]')
            else:
                print("  [WARN] No weights found for TD-LayerNorm")

            # Bidirectional LSTM — read forward and backward cells SEPARATELY
            # H5 structure:
            #   bidirectional/forward_layer/cell/vars/  → [kernel, recurrent_kernel, bias]
            #   bidirectional/backward_layer/cell/vars/ → [kernel, recurrent_kernel, bias]
            # Keras 3 Bidirectional expects:
            #   [fwd_kernel, fwd_recurrent_kernel, fwd_bias,
            #    bwd_kernel, bwd_recurrent_kernel, bwd_bias]
            fwd_arrs = _get_vars_at(f"{_LCD}bidirectional/forward_layer/cell/vars/")
            bwd_arrs = _get_vars_at(f"{_LCD}bidirectional/backward_layer/cell/vars/")
            if len(fwd_arrs) == 3 and len(bwd_arrs) == 3:
                lstm_weights = fwd_arrs + bwd_arrs  # [fwd_k, fwd_r, fwd_b, bwd_k, bwd_r, bwd_b]
                cnn_bilstm_model.get_layer('bidirectional').set_weights(lstm_weights)
                weight_log.append(f'BiLSTM(256) [fwd:{[a.shape for a in fwd_arrs]}, bwd:{[a.shape for a in bwd_arrs]}]')
            else:
                print(f"  [WARN] BiLSTM weights incomplete: fwd={len(fwd_arrs)}, bwd={len(bwd_arrs)}")

            # Dense(512, relu) — H5: dense/vars/
            arrs = _get_layer_vars('dense')
            if arrs:
                cnn_bilstm_model.get_layer('dense_1').set_weights(arrs)
                weight_log.append(f'Dense(512) [{len(arrs)} arrays]')
            else:
                print("  [WARN] No weights found for Dense(512)")

            # BatchNormalization — H5: batch_normalization/vars/
            arrs = _get_layer_vars('batch_normalization')
            if arrs and len(arrs) == 4 and arrs[0].shape == (512,):
                cnn_bilstm_model.get_layer('batch_normalization_top').set_weights(arrs)
                weight_log.append(f'BatchNorm [{len(arrs)} arrays]')
            else:
                print(f"  [WARN] BatchNorm weights issue: {len(arrs) if arrs else 0} arrays")

            # Dense(256, relu) — H5: dense_2/vars/
            arrs = _get_layer_vars('dense_2')
            if arrs:
                cnn_bilstm_model.get_layer('dense_2').set_weights(arrs)
                weight_log.append(f'Dense(256) [{len(arrs)} arrays]')
            else:
                print("  [WARN] No weights found for Dense(256)")

            # Dense(50, softmax) — H5: dense_4/vars/
            arrs = _get_layer_vars('dense_4')
            if arrs:
                cnn_bilstm_model.get_layer('dense_3').set_weights(arrs)
                weight_log.append(f'Dense(50) [{len(arrs)} arrays]')
            else:
                print("  [WARN] No weights found for Dense(50)")

            print(f"  [OK] Weights assigned: {', '.join(weight_log)}")
            print(f"  [INFO] EfficientNet backbone: ImageNet weights (frozen)")

            # Save converted model for faster loading next time
            try:
                cnn_bilstm_model.save(converted_model_path)
                print(f"  [OK] Saved converted model to {converted_model_path}")
            except Exception as save_err:
                print(f"  [WARN] Could not save converted model: {save_err}")

            print(f"[OK] CNN+BiLSTM model rebuilt and weights loaded!")

        except Exception as e:
            import traceback
            print(f"[X] CNN+BiLSTM rebuild FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            cnn_bilstm_model = None

    elif cnn_bilstm_model is None:
        if os.path.exists(CNN_BILSTM_MODEL_DIR):
            print(f"[X] CNN+BiLSTM directory exists but missing model.weights.h5")
        else:
            print(f"[X] CNN+BiLSTM model not found at: {CNN_BILSTM_MODEL_DIR}")
            print(f"    Also checked: {converted_model_path}")
        cnn_bilstm_model = None

    # Diagnostic: print model info
    if cnn_bilstm_model is not None:
        try:
            in_shape = cnn_bilstm_model.input_shape
        except Exception:
            in_shape = "unknown"
        try:
            out_shape = cnn_bilstm_model.output_shape
        except Exception:
            out_shape = "unknown"
        try:
            params = f"{cnn_bilstm_model.count_params():,}"
        except Exception:
            params = "unknown"
        print(f"     Input shape:  {in_shape}")
        print(f"     Output shape: {out_shape}")
        print(f"     Parameters:   {params}")

    # ══════════════════════════════════════════════════════════════
    # OLD MODEL LOADING (COMMENTED OUT — preserved for fallback)
    # To revert to the old pipeline, uncomment this block and set
    # cnn_bilstm_model = None above.
    # ══════════════════════════════════════════════════════════════

    # # 3a. Sklearn / Joblib Model — PRIMARY (trained RandomForest on 63-dim landmarks)
    # pkl_path = _resolve_model_path('sign_language_model.pkl', 'Sklearn Model')
    # if pkl_path:
    #     try:
    #         import joblib
    #         sklearn_model = joblib.load(pkl_path)
    #
    #         # ── MONKEY PATCH FOR SCIKIT-LEARN 1.8.0 COMPATIBILITY ──
    #         if hasattr(sklearn_model, 'estimators_'):
    #             for estimator in sklearn_model.estimators_:
    #                 if not hasattr(estimator, 'monotonic_cst'):
    #                     estimator.monotonic_cst = None
    #
    #         n_feat = getattr(sklearn_model, 'n_features_in_', '?')
    #         n_cls = getattr(sklearn_model, 'n_classes_', '?')
    #         print(f"[OK] Sklearn model loaded: {type(sklearn_model).__name__} "
    #               f"(features={n_feat}, classes={n_cls}) from {pkl_path}")
    #     except Exception as e:
    #         print(f"[X] Sklearn model failed: {type(e).__name__}: {e}")
    #         sklearn_model = None
    # else:
    #     print("[WARNING] No sklearn .pkl model found")

    # # 3b. TFLite Model (fallback)
    # tflite_path = _resolve_model_path('sign_language_model.tflite', 'TFLite Model')
    # if tflite_path is None:
    #     tflite_path = _resolve_model_path('ISL_IMAGE.tflite', 'TFLite Model (ISL_IMAGE fallback)')
    #
    # if tflite_path:
    #     try:
    #         import tensorflow as tf
    #         tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
    #         tflite_interpreter.allocate_tensors()
    #         tflite_input_details = tflite_interpreter.get_input_details()
    #         tflite_output_details = tflite_interpreter.get_output_details()
    #         print(f"[OK] TFLite model loaded successfully from {tflite_path}")
    #     except Exception as e:
    #         print(f"[X] TFLite model failed: {type(e).__name__}: {e}")
    #         tflite_interpreter = None

    # # 3c. Keras fallback (.h5 / .keras single-frame model)
    # model_path = _resolve_model_path('sign_language_model.keras', 'Keras Model')
    # if model_path:
    #     try:
    #         from tensorflow.keras.models import load_model
    #         tf_model = load_model(model_path, compile=False)
    #         print(f"[OK] Keras model loaded from {model_path}")
    #     except Exception as e:
    #         print(f"[X] Could not load Keras model: {type(e).__name__}: {e}")

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

    print("="*60)
    if cnn_bilstm_model is not None:
        print("[OK] PRIMARY ENGINE: CNN+BiLSTM (20-frame video sequence)")
    else:
        print("[!!] WARNING: CNN+BiLSTM not loaded -- inference will be unavailable")
    print("="*60 + "\n")

# ==========================================
# CNN+BiLSTM INFERENCE PIPELINE
# ==========================================

def preprocess_frame_for_cnn(frame_bgr):
    """Preprocess a single BGR OpenCV frame for the CNN+BiLSTM model.
    
    Steps:
      1. Convert BGR → RGB
      2. Resize to 224×224
      3. Keep as uint8 (0-255) — the model has a built-in Rescaling(1./255) layer
    
    Returns numpy array of shape (224, 224, 3), dtype=uint8.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, CNN_BILSTM_FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    return frame_resized.astype(np.uint8)


def cnn_bilstm_predict(frame_buffer):
    """Run CNN+BiLSTM inference on a full 20-frame buffer.
    
    Args:
        frame_buffer: list/deque of 20 preprocessed frames,
                      each of shape (224, 224, 3), dtype uint8
    
    Returns:
        (class_name, confidence, "cnn_bilstm") or (None, 0.0, None)
    """
    if cnn_bilstm_model is None:
        logger.warning("[cnn_bilstm_predict] Model not loaded")
        return None, 0.0, None
    
    if len(frame_buffer) < CNN_BILSTM_SEQ_LEN:
        return None, 0.0, None
    
    try:
        # Stack frames: (20, 224, 224, 3) → add batch dim → (1, 20, 224, 224, 3)
        frames_array = np.array(list(frame_buffer), dtype=np.float32)
        batch = np.expand_dims(frames_array, axis=0)
        
        print(f"[DIAG cnn_bilstm_predict] input shape={batch.shape}, "
              f"dtype={batch.dtype}, min={batch.min():.1f}, max={batch.max():.1f}")
        
        # Run prediction — use direct call instead of model.predict() to avoid
        # tf.function retracing issues with TimeDistributed+LayerNormalization
        import tensorflow as tf
        batch_tensor = tf.constant(batch)
        predictions = cnn_bilstm_model(batch_tensor, training=False)[0].numpy()
        
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])
        
        # Map class index to name
        class_name = class_mapping.get(str(class_idx),
                                        CNN_BILSTM_CLASS_MAPPING.get(class_idx, f"Class_{class_idx}"))
        
        proba_str = ', '.join(f"{p:.4f}" for p in predictions)
        print(f"[DIAG cnn_bilstm_predict] probabilities=[{proba_str}]")
        print(f"[DIAG cnn_bilstm_predict] result: '{class_name}' @ {confidence:.4f}")
        
        return class_name, confidence, "cnn_bilstm"
        
    except Exception as e:
        logger.error(f"[cnn_bilstm_predict] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, None


def decode_base64_frame(base64_data):
    """Decode a base64-encoded JPEG image to an OpenCV BGR frame.
    
    Handles both raw base64 and data-URI format (data:image/jpeg;base64,...).
    Returns numpy array (H, W, 3) in BGR format, or None on failure.
    """
    try:
        # Strip data URI prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        
        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"[decode_base64_frame] Failed: {e}")
        return None


# ==========================================
# COMPUTER VISION & INFERENCE UTILS (OLD — preserved)
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
    
    NOTE: This function is part of the OLD landmark pipeline.
    It is preserved for fallback compatibility.
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
    
    NOTE: This function is part of the OLD landmark pipeline.
    It is preserved for fallback compatibility.
    """
    import copy
    import itertools

    landmark_point = []
    for lm in hand_landmarks_obj.landmark:
        lx = min(int(lm.x * image_width), image_width - 1)
        ly = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])

    temp = copy.deepcopy(landmark_point)
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat)) if flat else 1
    if max_val > 0:
        flat = [n / max_val for n in flat]
    return flat


def extract_landmarks_for_model(frame, results):
    """Extract and normalise landmarks for the PRIMARY hand.
    
    NOTE: This function is part of the OLD landmark pipeline.
    It is preserved for fallback compatibility.
    """
    if not results.multi_hand_landmarks:
        return None

    hand_lm = results.multi_hand_landmarks[0]
    normed = _bbox_normalize_landmarks(hand_lm)

    if len(normed) == 63:
        arr = np.array(normed, dtype=np.float32)
        return arr
    return None


def fast_predict(landmark_vector):
    """Predict from a single-frame (63,) landmark vector.
    
    NOTE: This function is part of the OLD landmark pipeline.
    It is preserved for fallback compatibility. The CNN+BiLSTM pipeline
    uses cnn_bilstm_predict() instead.
    
    Returns (class_name, confidence, model_type) or (None, 0.0, None).
    """
    # Currently disabled — CNN+BiLSTM is the primary pipeline.
    # Uncomment and restore global references to re-enable.
    logger.warning("[fast_predict] Old landmark pipeline is disabled. "
                   "Use cnn_bilstm_predict() instead.")
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
      - The prediction confidence is >= CONFIDENCE_GATE
      - The same sign appears in >= SMOOTH_AGREEMENT_PCT of the buffer
    
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
        smoother = _fallback_smoother

    smoother['history'].append(new_pred)
    smoother['confidence'].append(new_conf)

    if len(smoother['history']) < 2:
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
    
    NOTE: This uses the OLD landmark pipeline. For the new CNN+BiLSTM
    pipeline, use the WebSocket frame handler in sockets.py.
    """
    cap = get_camera()
    if cap is None or not cap.isOpened():
        logger.error("[generate_frames] Camera failed to open. Aborting.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        display_text = "CNN+BiLSTM mode — use WebSocket"
        color = (0, 165, 255)  # Orange to indicate mode change

        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
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
