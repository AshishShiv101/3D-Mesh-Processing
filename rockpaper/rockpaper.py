import cv2
import mediapipe as mp
import random
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

# Finger tip indices in MediaPipe landmark scheme
TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

def fingers_up(hand_landmarks, handedness_label):
    """Returns list of 5 ints: 1 if finger is up, else 0."""
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

def gesture_from_fingers(fingers):
    s = sum(fingers)
    if s == 5:
        return "Paper"
    if s == 0:
        return "Rock"
    if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "Scissors"
    return None

def decide_winner(player, computer):
    if player == computer:
        return "Tie"
    wins = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}
    return "You Win!" if (player, computer) in wins else "Computer Wins!"

# Game setup
options = ["Rock", "Paper", "Scissors"]

cap = cv2.VideoCapture(0)
prev_time = 0
cooldown = 1.0  # seconds between gesture checks
restart_delay = 5.0  # seconds before game restarts

# Game state variables
last_result_text = ""
last_player_choice = ""
last_computer_choice = ""
round_end_time = None  # track when round ended

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = None
    handedness_label = "Right"

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_landmarks = result.multi_hand_landmarks[0]
        handedness_label = result.multi_handedness[0].classification[0].label
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(hand_landmarks, handedness_label)
        gesture = gesture_from_fingers(fingers)

        now = time.time()
        if gesture and now - prev_time > cooldown and not last_result_text:
            # Play round
            prev_time = now
            player_choice = gesture
            computer_choice = random.choice(options)
            result_text = decide_winner(player_choice, computer_choice)

            last_player_choice = player_choice
            last_computer_choice = computer_choice
            last_result_text = result_text
            round_end_time = now  # mark end of round

    # Restart logic: reset game 5 seconds after round ended
    if round_end_time and time.time() - round_end_time >= restart_delay:
        last_result_text = ""
        last_player_choice = ""
        last_computer_choice = ""
        round_end_time = None

    # Draw UI
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Your: {last_player_choice}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer: {last_computer_choice}", (250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, last_result_text, (520, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Countdown display before restart
    if round_end_time:
        remaining = int(restart_delay - (time.time() - round_end_time))
        cv2.putText(frame, f"Next round in {remaining}s...",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    else:
        cv2.putText(frame, "Make Rock (fist), Paper (open palm) or Scissors (index+middle). Press 'q' to quit.",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Rock-Paper-Scissors Gesture Game", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
