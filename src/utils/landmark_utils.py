"""
landmark_utils.py — Universal MediaPipe landmark extraction & normalisation
============================================================================
THE SINGLE SOURCE OF TRUTH for how landmarks are processed in IndicSignAI.

Import and use `extract_and_normalize_landmarks` in BOTH:
  - training/scripts/train_model.py  (when building keypoint.csv / feature arrays)
  - src/app.py                        (when processing live camera / video blob)

This guarantees that the vectors fed to model.predict() at inference time are
mathematically identical to the vectors the model was trained on.

Algorithm
---------
1.  Run MediaPipe Hands (static_image_mode=True) on the BGR frame.
2.  Reject frames where MediaPipe confidence < mp_confidence_gate.
3.  For the primary hand, collect all 21 (x, y, z) landmarks.
4.  Compute the 3-D bounding box:
        x_range = x_max - x_min  (same for y, z)
5.  Normalise every coordinate:
        x_norm = (x - x_min) / x_range   →  [0.0, 1.0]
6.  Return a flat numpy array of shape (63,).

Why bounding-box normalisation?
    A "thumbs-up" at 30 cm from the camera and at 150 cm produces
    *identical* normalised vectors — the model is fully scale-invariant.

Returns
-------
  landmarks : np.ndarray shape (63,) or None
      None is returned when no hand is detected or confidence is too low.
  debug_info : dict
      Contains 'hand_landmarks' (MediaPipe object) and 'results' so callers
      can draw the skeleton for visualisation / debugging.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

# ── MediaPipe singleton (re-used across calls for speed) ──────────────────────
_mp_hands   = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_mp_styles  = mp.solutions.drawing_styles

# Number of landmarks × coordinates (21 × 3 = 63)
LANDMARK_FEATURE_DIM = 63


def extract_and_normalize_landmarks(
    frame_bgr: np.ndarray,
    mp_confidence_gate: float = 0.80,
) -> tuple:
    """Extract MediaPipe hand landmarks and return a bbox-normalised feature
    vector that is IDENTICAL whether used in training or inference.

    Parameters
    ----------
    frame_bgr : np.ndarray
        A BGR image (as returned by cv2.VideoCapture or cv2.imread).
    mp_confidence_gate : float
        Minimum MediaPipe hand-detection confidence to accept the frame.
        Frames below this threshold return (None, {}).

    Returns
    -------
    landmarks : np.ndarray of shape (63,) or None
    debug_info : dict with keys:
        'hand_landmarks'  — raw MediaPipe NormalizedLandmarkList (or None)
        'results'         — full MediaPipe results object (or None)
        'bbox_px'         — (x_min, y_min, x_max, y_max) in pixels (or None)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, {'hand_landmarks': None, 'results': None, 'bbox_px': None}

    # Always work on RGB for MediaPipe
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with _mp_hands.Hands(
        static_image_mode=True,          # treats every frame independently
        max_num_hands=1,
        min_detection_confidence=mp_confidence_gate,
        min_tracking_confidence=0.5,
    ) as detector:
        results = detector.process(rgb)

    if not results or not results.multi_hand_landmarks:
        return None, {'hand_landmarks': None, 'results': results, 'bbox_px': None}

    hand_lm = results.multi_hand_landmarks[0]  # primary hand

    # ── Step 1: Collect raw normalised coordinates (MediaPipe frame space) ──
    xs = [lm.x for lm in hand_lm.landmark]   # 21 values ∈ [0, 1] of image
    ys = [lm.y for lm in hand_lm.landmark]
    zs = [lm.z for lm in hand_lm.landmark]

    # ── Step 2: Compute per-axis bounding box ────────────────────────────────
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    x_range = (x_max - x_min) or 1e-6   # guard zero-division (flat hand edge-on)
    y_range = (y_max - y_min) or 1e-6
    z_range = (z_max - z_min) or 1e-6

    # ── Step 3: Normalise every landmark to [0, 1] within its bounding box ──
    normed: list[float] = []
    for lm in hand_lm.landmark:
        normed.append((lm.x - x_min) / x_range)
        normed.append((lm.y - y_min) / y_range)
        normed.append((lm.z - z_min) / z_range)

    landmarks = np.array(normed, dtype=np.float32)  # shape (63,)

    # ── Debug info: pixel-space bounding box ─────────────────────────────────
    h, w = frame_bgr.shape[:2]
    bbox_px = (
        int(x_min * w),
        int(y_min * h),
        int(x_max * w),
        int(y_max * h),
    )

    debug_info = {
        'hand_landmarks': hand_lm,
        'results':        results,
        'bbox_px':        bbox_px,
    }
    return landmarks, debug_info


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    debug_info: dict,
    landmark_vector: np.ndarray | None,
    label: str = "",
    confidence: float = 0.0,
) -> np.ndarray:
    """Draw the MediaPipe skeleton and overlay the normalised vector stats
    onto a BGR frame.  Returns the annotated frame (copy).

    Parameters
    ----------
    frame_bgr       : original BGR frame
    debug_info      : dict returned by extract_and_normalize_landmarks
    landmark_vector : the (63,) array passed to the model (may be None)
    label           : predicted class label to print
    confidence      : model confidence score
    """
    canvas = frame_bgr.copy()
    hand_lm = debug_info.get('hand_landmarks')

    if hand_lm:
        # Draw skeleton
        _mp_drawing.draw_landmarks(
            canvas,
            hand_lm,
            _mp_hands.HAND_CONNECTIONS,
            _mp_styles.get_default_hand_landmarks_style(),
            _mp_styles.get_default_hand_connections_style(),
        )

        # Draw bounding box
        bbox = debug_info.get('bbox_px')
        if bbox:
            pad = 10
            h, w = canvas.shape[:2]
            bx1 = max(0, bbox[0] - pad)
            by1 = max(0, bbox[1] - pad)
            bx2 = min(w, bbox[2] + pad)
            by2 = min(h, bbox[3] + pad)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

    # Overlay landmark vector stats
    if landmark_vector is not None:
        stats = (
            f"vec min={landmark_vector.min():.3f} "
            f"max={landmark_vector.max():.3f} "
            f"mean={landmark_vector.mean():.3f}"
        )
        # Print first 9 values on screen for quick sanity-check
        vec_preview = "  ".join(f"{v:.2f}" for v in landmark_vector[:9])
        lines = [
            f"LANDMARK VECTOR (first 9 of {len(landmark_vector)})",
            vec_preview,
            stats,
        ]
        if label:
            lines.insert(0, f"PRED: {label}  ({confidence:.1%})")

        y_cursor = 30
        for line in lines:
            # Black shadow
            cv2.putText(canvas, line, (11, y_cursor + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # White text
            cv2.putText(canvas, line, (10, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_cursor += 22
    else:
        cv2.putText(canvas, "NO HAND / BELOW CONFIDENCE GATE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return canvas
