import time
import random
import cv2
import mediapipe as mp
import pyautogui

# -------------------------
# Config
# -------------------------
COOLDOWN = 1.0      # seconds between gesture triggers
MIN_DETECTION_CONF = 0.6

# Key mappings (change to suit your player)
KEY_PLAY_PAUSE = "space"            # play / pause toggle
KEY_STOP = "s"                      # stop (or another key your player uses)
KEY_NEXT = ["ctrl", "right"]        # next track (ctrl+right)
KEY_PREV = ["ctrl", "left"]         # previous track (ctrl+left)

# -------------------------
# MediaPipe setup
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=MIN_DETECTION_CONF)

# landmark tip ids
TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# -------------------------
# Utility: fingers up
# -------------------------
def fingers_up(hand_landmarks, handedness_label):
    """
    Returns list of 5 ints: 1 if finger is up, else 0.
    Order: [thumb, index, middle, ring, pinky]
    """
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: compare tip.x with previous landmark x depending on handedness
    # Note: frame will be mirrored (we flip), so Right/Left mapping matches user's right/left visually.
    if handedness_label == "Right":
        fingers.append(1 if lm[TIP_IDS[0]].x < lm[TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if lm[TIP_IDS[0]].x > lm[TIP_IDS[0] - 1].x else 0)

    # Other fingers: tip.y < pip.y => finger up
    for id in range(1, 5):
        fingers.append(1 if lm[TIP_IDS[id]].y < lm[TIP_IDS[id] - 2].y else 0)

    return fingers

# -------------------------
# Gesture classification
# -------------------------
def classify_gesture(hand_landmarks, handedness_label):
    """
    Returns one of:
    'open_palm', 'fist', 'thumbs_up', 'thumbs_down', 'point_right', 'point_left', or None
    """
    lm = hand_landmarks.landmark
    f = fingers_up(hand_landmarks, handedness_label)
    total_up = sum(f)

    # Open palm
    if total_up == 5:
        return "open_palm"

    # Fist
    if total_up == 0:
        return "fist"

    # Thumbs up/down: thumb up (1) and all other fingers down (0)
    if f[0] == 1 and f[1] == 0 and f[2] == 0 and f[3] == 0 and f[4] == 0:
        # compare thumb tip y with wrist (0) to see up or down
        # wrist landmark index is 0
        thumb_tip_y = lm[TIP_IDS[0]].y
        wrist_y = lm[0].y
        if thumb_tip_y < wrist_y:
            return "thumbs_up"
        else:
            return "thumbs_down"

    # Point (index finger extended, others folded)
    if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0:
        # check direction: index tip x vs pip x to see left or right (frame is mirrored)
        idx_tip_x = lm[TIP_IDS[1]].x
        idx_mcp_x = lm[TIP_IDS[1] - 2].x
        if idx_tip_x > idx_mcp_x:   # pointing right (in mirrored frame)
            return "point_right"
        else:
            return "point_left"

    return None

# -------------------------
# Action trigger helpers
# -------------------------
last_action_time = 0.0

def trigger_action(action):
    global last_action_time
    now = time.time()
    if now - last_action_time < COOLDOWN:
        return False
    last_action_time = now

    if action == "open_palm":
        # toggle play/pause
        pyautogui.press(KEY_PLAY_PAUSE)
        print("[ACTION] Play/Pause")
        return True
    if action == "thumbs_up":
        pyautogui.press(KEY_PLAY_PAUSE)
        print("[ACTION] Play")
        return True
    if action == "thumbs_down":
        pyautogui.press(KEY_STOP)
        print("[ACTION] Stop")
        return True
    if action == "point_right":
        # next track - compound hotkey
        if isinstance(KEY_NEXT, list):
            pyautogui.hotkey(*KEY_NEXT)
        else:
            pyautogui.press(KEY_NEXT)
        print("[ACTION] Next track")
        return True
    if action == "point_left":
        if isinstance(KEY_PREV, list):
            pyautogui.hotkey(*KEY_PREV)
        else:
            pyautogui.press(KEY_PREV)
        print("[ACTION] Previous track")
        return True

    return False

# -------------------------
# Main loop
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        gesture = None
        handedness_label = "Right"

        if res.multi_hand_landmarks and res.multi_handedness:
            hand_landmarks = res.multi_hand_landmarks[0]
            handedness_label = res.multi_handedness[0].classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = classify_gesture(hand_landmarks, handedness_label)
            if gesture:
                triggered = trigger_action(gesture)
                # Show the recognized gesture on the frame
                if triggered:
                    cv2.putText(frame, f"Detected: {gesture}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Detected (cooldown): {gesture}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 180), 2)

        # UI hints
        cv2.putText(frame, "Gestures: ✋ Pause/Toggle | 👍 Play | 👎 Stop | 👉 Next | 👈 Prev",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("Gesture Media Controller", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
