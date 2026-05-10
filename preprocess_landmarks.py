import numpy as np

def normalize_landmarks(landmarks):

    lm = np.array(landmarks, dtype=np.float32).reshape(21, 2)

    # Translate wrist to origin
    lm = lm - lm[0]

    # Rotate
    ref = lm[9]
    angle = np.arctan2(ref[0], -ref[1])

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rot = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    lm = lm @ rot.T

    # Scale
    scale = np.linalg.norm(lm[9])

    if scale > 1e-6:
        lm = lm / scale

    return lm.flatten().tolist()