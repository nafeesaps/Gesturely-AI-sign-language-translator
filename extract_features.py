"""
extract_features.py
───────────────────
Shared feature extractor for the LSTM Word Sign Recognition pipeline.

Feature vector per frame = 88 floats:
  [0:42]   Right hand — normalized shape  (zeros if absent)
  [42:84]  Left hand  — normalized shape  (zeros if absent)
  [84:88]  Pose-relative hand position (zeros if pose not detected)

Pose-relative block (4 values):
  idx 84 — right_wrist Y relative to shoulders  (0=chest/shoulder height, negative=face)
  idx 85 — right_wrist X relative to center 
  idx 86 — left_wrist  Y 
  idx 87 — left_wrist  X 

Why POSE instead of FACE?
  When you point to your chest for 'my', you naturally look down slightly.
  This causes the Face Mesh module to instantly drop tracking, giving 0.0.
  The Pose module (shoulders/nose) is extremely robust and will never drop
  just because your head tilts.
"""

import numpy as np
import mediapipe as mp

# ── MediaPipe Holistic ────────────────────────────────────────────────────────
_mp_holistic  = mp.solutions.holistic
_mp_drawing   = mp.solutions.drawing_utils
_holistic     = _mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Pose landmark indices
_NOSE_IDX      = 0 
_L_SHOULDER_IDX = 11
_R_SHOULDER_IDX = 12

FEATURE_SIZE = 88   # exported constant used by all scripts

# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_hand(landmarks_proto):
    """
    Normalize 21 MediaPipe hand landmarks to be
    position-, scale-, and rotation-invariant.
    Returns a flat list of 42 floats.
    """
    row = []
    for pt in landmarks_proto.landmark:
        row.extend([pt.x, pt.y])
    lm = np.array(row, dtype=np.float32).reshape(21, 2)

    # 1. Translate wrist to origin
    lm -= lm[0]

    # 2. Rotate so wrist→middle-MCP axis points straight up
    ref = lm[9]
    angle = np.arctan2(ref[0], -ref[1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    lm = lm @ rot.T

    # 3. Scale by wrist-to-middle-MCP distance
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale

    return lm.flatten().tolist()


def _pose_relative_pos(wrist_x, wrist_y, pose_lms):
    """
    Return (rel_y, rel_x) of a wrist position relative to the POSE (shoulders),
    normalised by shoulder width.
    
    This is vastly more robust than Face Mesh because Pose tracking doesn't 
    stop when the user looks down or touches their chest.
    """
    if pose_lms is None:
        return 0.0, 0.0

    nose = pose_lms.landmark[_NOSE_IDX]
    l_sh = pose_lms.landmark[_L_SHOULDER_IDX]
    r_sh = pose_lms.landmark[_R_SHOULDER_IDX]

    # Use shoulder width as our scale factor (very stable)
    dx = l_sh.x - r_sh.x
    dy = l_sh.y - r_sh.y
    shoulder_width = (dx**2 + dy**2)**0.5

    if shoulder_width < 1e-4:
        return 0.0, 0.0

    # Center of shoulders
    center_y = (l_sh.y + r_sh.y) / 2.0

    # rel_y = 0.0 at chest/shoulders. Negative is up towards face. Positive is down towards stomach.
    rel_y = (wrist_y - center_y) / shoulder_width
    rel_x = (wrist_x - nose.x)   / shoulder_width

    return float(rel_y), float(rel_x)


# ── Public API ────────────────────────────────────────────────────────────────

def process_frame(bgr_frame):
    """
    Run MediaPipe Holistic on one BGR frame.
    Returns the raw holistic results object (pass to extract_keypoints).
    """
    rgb = bgr_frame[:, :, ::-1].copy()  # BGR → RGB
    results = _holistic.process(rgb)
    return results


def extract_keypoints(results, bgr_frame=None):
    """
    Build the 88-float feature vector from a holistic results object.
    """
    right_hand = np.zeros(42, dtype=np.float32)
    left_hand  = np.zeros(42, dtype=np.float32)
    pos        = np.zeros(4,  dtype=np.float32)  # [R_y, R_x, L_y, L_x]

    pose_lms = results.pose_landmarks

    # ── Right hand ────────────────────────────────────────────────────────
    if results.right_hand_landmarks:
        rh = results.right_hand_landmarks
        right_hand = np.array(_normalize_hand(rh), dtype=np.float32)

        wx = rh.landmark[0].x   # wrist
        wy = rh.landmark[0].y
        pos[0], pos[1] = _pose_relative_pos(wx, wy, pose_lms)

        if bgr_frame is not None:
            _mp_drawing.draw_landmarks(
                bgr_frame, rh, _mp_holistic.HAND_CONNECTIONS)

    # ── Left hand ─────────────────────────────────────────────────────────
    if results.left_hand_landmarks:
        lh = results.left_hand_landmarks
        left_hand = np.array(_normalize_hand(lh), dtype=np.float32)

        wx = lh.landmark[0].x
        wy = lh.landmark[0].y
        pos[2], pos[3] = _pose_relative_pos(wx, wy, pose_lms)

        if bgr_frame is not None:
            _mp_drawing.draw_landmarks(
                bgr_frame, lh, _mp_holistic.HAND_CONNECTIONS)

    # Draw pose mapping for visual feedback
    if bgr_frame is not None and pose_lms is not None:
        _mp_drawing.draw_landmarks(
            bgr_frame, pose_lms, _mp_holistic.POSE_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            _mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )

    features = np.concatenate([right_hand, left_hand, pos])  # shape (88,)
    return features, bgr_frame


def close():
    """Release MediaPipe resources."""
    _holistic.close()
