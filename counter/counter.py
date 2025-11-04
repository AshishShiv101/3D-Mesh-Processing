import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# Finger tip landmark indices (Thumb, Index, Middle, Ring, Pinky)
TIP_IDS = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks, handedness_label):
    """Returns list [thumb, index, middle, ring, pinky] (1 = up, 0 = down)"""
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb
    if handedness_label == "Right":
        fingers.append(1 if lm[TIP_IDS[0]].x < lm[TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if lm[TIP_IDS[0]].x > lm[TIP_IDS[0] - 1].x else 0)

    # Other 4 fingers
    for id in range(1, 5):
        fingers.append(1 if lm[TIP_IDS[id]].y < lm[TIP_IDS[id] - 2].y else 0)

    return fingers

# --- Main Program ---
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural control
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_count, right_count = 0, 0  # per-hand counters

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks, label)
            count = sum(fingers)

            # Place label on screen near wrist
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f"{label}: {count}", (cx - 50, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Update totals
            if label == "Left":
                left_count = count
            elif label == "Right":
                right_count = count

    total = left_count + right_count

    # FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time) if prev_time else 0
    prev_time = cur_time

    # Draw black header box
    cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Left: {left_count}  Right: {right_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Total: {total}  FPS: {int(fps)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Dual-Hand Finger Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
