import os
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf
import mediapipe as mp
import webbrowser

from flask import Flask, render_template, Response, jsonify
from threading import Timer
from collections import deque

# ===============================
# FEATURE EXTRACTOR (88 FEATURES)
# ===============================
from extract_features import process_frame, extract_keypoints, FEATURE_SIZE

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)

print("🟢 STEP 1: Flask initialized")

# ===============================
# SAFE CUSTOM OBJECTS (FIX ORTHOGONAL ISSUE)
# ===============================
safe_custom_objects = {
    "Orthogonal": tf.keras.initializers.Orthogonal,
    "DTypePolicy": tf.keras.mixed_precision.Policy
}

# ===============================
# LETTER MODEL
# ===============================
print("➡ Loading letter model...")

try:
    from tensorflow.keras.models import load_model

    letter_model = load_model(
        "letter_model.h5",
        compile=False,
        custom_objects=safe_custom_objects
    )

    print("✔ Letter model loaded")

except Exception as e:
    print("❌ Letter model failed:", e)
    letter_model = None

# ===============================
# WORD MODEL (88 FEATURES)
# ===============================
# ===============================
# WORD MODEL FIX (ONLY)
# ===============================
print("➡ Loading word model...")

word_model = None

try:
    import tensorflow as tf

    # Fix BOTH issues:
    # 1) Orthogonal initializer
    # 2) Dense quantization_config

    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense

    # --- PATCH Dense to ignore legacy arg ---
    _old_dense_init = Dense.__init__

    def _patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        _old_dense_init(self, *args, **kwargs)

    Dense.__init__ = _patched_dense_init

    # --- SAFE LOAD ---
    word_model = load_model(
        "word_lstm_model.h5",
        compile=False,
        custom_objects={
            "Orthogonal": tf.keras.initializers.Orthogonal
        }
    )

    print("✔ Word model loaded successfully")

except Exception as e:
    print("❌ Word model failed:", e)
    word_model = None
# ===============================
# LETTER ENCODER
# ===============================
try:
    with open("letter_encoder.pickle", "rb") as f:
        encoder = pickle.load(f)
    print("✔ Letter encoder loaded")
except Exception as e:
    print("❌ Letter encoder failed:", e)
    encoder = None

# ===============================
# ACTIONS (MUST MATCH TRAINING)
# ===============================
ACTIONS = [
    'hello', 'my', 'thanks', 'iloveyou',
    'yes', 'no', 'nothing', 'help',
    'me', 'fine', 'name', 'more',
    'learn', 'forget', 'right',
    'need', 'same'
]

# ===============================
# MEDIAPIPE HANDS (LETTER MODE ONLY)
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("✔ MediaPipe ready")

# ===============================
# GLOBAL STATE
# ===============================
mode = "letter"

sentence = ""
current_word = ""

sequence = deque(maxlen=30)
predictions = deque(maxlen=15)

last_letter = ""
last_word_pred = ""
last_time = 0
word_lock_time = 0

COOLDOWN = 2.0

# ===============================
# CAMERA STREAM
# ===============================
def generate_frames():

    global sentence, current_word, mode
    global sequence, predictions
    global last_letter, last_word_pred
    global last_time, word_lock_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ CAMERA NOT OPENED")
        return

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # ===============================
        # WORD MODE (88 FEATURES)
        # ===============================
        if mode == "word" and word_model:

            try:
                results = process_frame(frame)
                features, frame = extract_keypoints(results, frame)

                sequence.append(features)

                if len(sequence) == 30:

                    inp = np.array(sequence, dtype=np.float32).reshape(1, 30, FEATURE_SIZE)

                    pred = word_model.predict(inp, verbose=0)[0]

                    idx = np.argmax(pred)
                    conf = float(np.max(pred))

                    predictions.append(idx)

                    most_common = max(set(predictions), key=predictions.count)
                    count = predictions.count(most_common)

                    now = time.time()

                    if (
                        conf > 0.70 and
                        count >= 10 and
                        (now - word_lock_time) > COOLDOWN
                    ):

                        word = ACTIONS[most_common]

                        if word != "nothing" and word != last_word_pred:

                            sentence += word + " "
                            last_word_pred = word

                            print(f"✅ WORD: {word} ({conf:.2f})")

                        word_lock_time = now
                        sequence.clear()
                        predictions.clear()

            except Exception as e:
                print("WORD ERROR:", e)

        # ===============================
        # LETTER MODE
        # ===============================
        elif mode == "letter" and letter_model:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    letter_row = []
                    for lm in hand_landmarks.landmark:
                        letter_row += [lm.x, lm.y]

                    if len(letter_row) == 42:

                        inp = np.array(letter_row, dtype=np.float32).reshape(1, 1, 42)

                        try:
                            pred = letter_model.predict(inp, verbose=0)
                            idx = np.argmax(pred)
                            conf = float(pred[0][idx])

                            if conf > 0.85 and encoder:

                                try:
                                    letter = encoder.inverse_transform([idx])[0]
                                except:
                                    letter = str(idx)

                                now = time.time()

                                if letter != last_letter or (now - last_time) > 0.7:
                                    current_word += letter
                                    last_letter = letter
                                    last_time = now

                        except Exception as e:
                            print("LETTER ERROR:", e)

        # ===============================
        # UI
        # ===============================
        cv2.rectangle(frame, (0, 0), (1200, 150), (0, 0, 0), -1)

        cv2.putText(frame, f"Mode: {mode}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, "Text: " + current_word,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "Sentence: " + sentence,
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# ===============================
# ROUTES (UNCHANGED)
# ===============================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    return jsonify({"sentence": sentence, "mode": mode})

@app.route('/set_mode/<m>')
def set_mode(m):
    global mode, sequence, predictions, current_word, last_letter, last_word_pred

    if m in ["letter", "word"]:
        mode = m
        sequence.clear()
        predictions.clear()
        current_word = ""
        last_letter = ""
        last_word_pred = ""

    return jsonify({"mode": mode})

@app.route('/clear')
def clear():
    global sentence, current_word, sequence, predictions, last_letter, last_word_pred

    sentence = ""
    current_word = ""
    sequence.clear()
    predictions.clear()
    last_letter = ""
    last_word_pred = ""

    return jsonify({"status": "cleared"})

@app.route('/sign_to_text')
def sign_to_text():
    return render_template('sign_to_text.html')

@app.route('/text_to_sign')
def text_to_sign():
    return render_template('text_to_sign.html')

# ===============================
# OPEN BROWSER
# ===============================
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":

    Timer(2, open_browser).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )