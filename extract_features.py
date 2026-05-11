import numpy as np
import mediapipe as mp

# ===============================
# MEDIA PIPE HANDS (ONE HAND ONLY)
# ===============================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,   # ✅ ONE HAND ONLY (IMPORTANT)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# FEATURE SIZE
# ===============================
FEATURE_SIZE = 42   # 21 landmarks × (x, y)

# ===============================
# PROCESS FRAME
# ===============================
def process_frame(frame):
    rgb = frame[:, :, ::-1]
    return hands.process(rgb)

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_keypoints(results, frame):

    features = np.zeros(FEATURE_SIZE)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]  # only one hand

        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y])

        features = np.array(data)

        # safety check
        if len(features) > FEATURE_SIZE:
            features = features[:FEATURE_SIZE]
        elif len(features) < FEATURE_SIZE:
            features = np.pad(features, (0, FEATURE_SIZE - len(features)))

    return features.astype(np.float32), frame