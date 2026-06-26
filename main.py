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

from extract_features import process_frame, extract_keypoints, FEATURE_SIZE


app = Flask(__name__)

print(" Flask initialized")


safe_custom_objects = {
    "Orthogonal": tf.keras.initializers.Orthogonal,
    "DTypePolicy": tf.keras.mixed_precision.Policy
}


print(" Loading letter model...")

try:
    from tensorflow.keras.models import load_model

    letter_model = load_model(
        "letter_model.h5",
        compile=False,
        custom_objects=safe_custom_objects
    )

    print(" Letter model loaded")

except Exception as e:
    print(" Letter model failed:", e)
    letter_model = None


print(" Loading word model...")

word_model = None

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense

    _old_dense_init = Dense.__init__

    def _patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        _old_dense_init(self, *args, **kwargs)

    Dense.__init__ = _patched_dense_init

    word_model = load_model(
        "wor_model.h5",
        compile=False,
        custom_objects={
            "Orthogonal": tf.keras.initializers.Orthogonal
        }
    )

    print(" Word model loaded successfully")
    print("WORD MODEL STATUS:", word_model)

except Exception as e:
    print("Word model failed:", e)
    word_model = None


with open("wor_encoder.pickle", "rb") as f:
    word_encoder = pickle.load(f)

ACTIONS = word_encoder.classes_

with open("letter_encoder.pickle", "rb") as f:
    letter_encoder = pickle.load(f)

print("Encoders loaded")


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("✔ MediaPipe ready")

mode = "letter"

sentence = ""
current_word = ""

sequence = deque(maxlen=30)
predictions = deque(maxlen=10)

last_word = ""
last_time = 0

cooldown = 1.5

# LETTER TIMER
last_detection_time = time.time()

# slower sentence formation
LETTER_TIMEOUT = 4.0

# repeat control
last_letter = ""
last_letter_time = 0

# same letter allowed only after delay
REPEAT_DELAY = 2.0


def generate_frames():

    global sentence
    global current_word
    global mode

    global sequence
    global predictions

    global last_word
    global last_time
    global last_detection_time

    global last_letter
    global last_letter_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" CAMERA NOT OPENED")
        return

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        # WORD MODE
        if mode == "word" and word_model:

            try:
                results = process_frame(frame)

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                features, frame = extract_keypoints(results, frame)

                sequence.append(features)

                if len(sequence) == 30:

                    inp = np.array(sequence, dtype=np.float32)
                    inp = inp.reshape(1, 30, FEATURE_SIZE)

                    pred = word_model.predict(inp, verbose=0)[0]

                    idx = np.argmax(pred)
                    conf = float(np.max(pred))

                    predictions.append(idx)

                    most_common = max(
                        set(predictions),
                        key=predictions.count
                    )

                    word = ACTIONS[most_common]

                    now = time.time()

                    if conf > 0.55 and (now - last_time) > cooldown:

                        if word != "nothing" and word != last_word:

                            current_word = word
                            sentence += word + " "

                            last_word = word
                            last_time = now

                            print("WORD:", word, conf)

                        sequence.clear()
                        predictions.clear()

            except Exception as e:
                print("WORD ERROR:", e)

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

                        inp = np.array(
                            letter_row,
                            dtype=np.float32
                        ).reshape(1, 1, 42)

                        pred = letter_model.predict(
                            inp,
                            verbose=0
                        )

                        idx = np.argmax(pred)

                        conf = float(pred[0][idx])

                        if conf > 0.85:

                            try:
                                letter = letter_encoder.inverse_transform([idx])[0]

                            except:
                                letter = str(idx)

                            now = time.time()

                            allow = False

                            # new letter
                            if letter != last_letter:
                                allow = True

                            # same letter after delay
                            elif (now - last_letter_time) > REPEAT_DELAY:
                                allow = True

                            if allow:

                                current_word += letter

                                print("LETTER:", letter)

                                last_letter = letter
                                last_letter_time = now

                            last_detection_time = now

        # LETTER WORD CREATION
        if mode == "letter":

            now = time.time()

            if (
                len(current_word) > 0 and
                (now - last_detection_time) > LETTER_TIMEOUT
            ):

                sentence += current_word + " "

                print("WORD FORMED:", current_word)

                current_word = ""

                last_letter = ""

        cv2.rectangle(
            frame,
            (0, 0),
            (1200, 150),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Mode: {mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Word: " + current_word,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Sentence: " + sentence,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        _, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

    cap.release()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_sentence')
def get_sentence():

    return jsonify({
        "sentence": sentence,
        "mode": mode
    })


@app.route('/set_mode/<m>')
def set_mode(m):

    global mode
    global sequence
    global predictions
    global current_word
    global last_word
    global last_letter

    if m in ["letter", "word"]:

        mode = m

        sequence.clear()
        predictions.clear()

        current_word = ""
        last_word = ""
        last_letter = ""

    return jsonify({
        "mode": mode
    })


@app.route('/clear')
def clear():

    global sentence
    global current_word
    global sequence
    global predictions
    global last_word
    global last_letter

    sentence = ""
    current_word = ""

    sequence.clear()
    predictions.clear()

    last_word = ""
    last_letter = ""

    return jsonify({
        "status": "cleared"
    })

@app.route('/erase_last')
def erase_last():

    global sentence

    words = sentence.strip().split()

    if len(words) > 0:
        words.pop()

    sentence = " ".join(words)

    if len(sentence) > 0:
        sentence += " "

    return jsonify({
        "sentence": sentence
    })
    
@app.route('/sign_to_text')
def sign_to_text():
    return render_template('sign_to_text.html')


@app.route('/text_to_sign')
def text_to_sign():
    return render_template('text_to_sign.html')


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":

    Timer(2, open_browser).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )