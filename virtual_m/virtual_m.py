# virtual_mouse.py
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

# Config
CAM_W, CAM_H = 640, 480        # webcam frame size
FRAME_REDUCTION = 100          # reduce active area from edges (makes control easier)
SMOOTHING = 7                  # higher = smoother (but more lag)
CLICK_DIST_THRESHOLD = 40      # pixel distance in webcam coords to consider "pinch"/click
CLICK_DEBOUNCE = 0.35         # seconds between registered clicks

# Setup
pyautogui.FAILSAFE = False     # optional: disable top-left abort; be careful!
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# smoothing buffer
prev_positions = deque(maxlen=SMOOTHING)
last_click_time = 0

def smooth_point(pt):
    prev_positions.append(pt)
    arr = np.array(prev_positions)
    return np.mean(arr, axis=0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # draw a rectangle showing the active area
        cv2.rectangle(frame,
                      (FRAME_REDUCTION, FRAME_REDUCTION),
                      (CAM_W - FRAME_REDUCTION, CAM_H - FRAME_REDUCTION),
                      (255, 0, 255), 2)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark

            # landmarks: index fingertip is 8, thumb tip is 4
            ix, iy = int(lm[8].x * CAM_W), int(lm[8].y * CAM_H)
            tx, ty = int(lm[4].x * CAM_W), int(lm[4].y * CAM_H)

            # show landmarks
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # calculate pinch distance
            dist = np.hypot(tx - ix, ty - iy)

            # constrain index finger inside the active rectangle
            ix = np.clip(ix, FRAME_REDUCTION, CAM_W - FRAME_REDUCTION)
            iy = np.clip(iy, FRAME_REDUCTION, CAM_H - FRAME_REDUCTION)

            # map webcam coords to screen coords
            # interpolate from active rectangle -> full screen
            mapped_x = np.interp(ix, (FRAME_REDUCTION, CAM_W - FRAME_REDUCTION), (0, SCREEN_W))
            mapped_y = np.interp(iy, (FRAME_REDUCTION, CAM_H - FRAME_REDUCTION), (0, SCREEN_H))

            # smoothing
            smooth_x, smooth_y = smooth_point((mapped_x, mapped_y))

            # move the mouse
            pyautogui.moveTo(SCREEN_W - smooth_x, smooth_y)  # invert X if frame is mirrored; adjust as needed

            # visual cursor on the frame
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"Dist: {int(dist)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # click when pinch is close enough (and debounce)
            now = time.time()
            if dist < CLICK_DIST_THRESHOLD and (now - last_click_time) > CLICK_DEBOUNCE:
                last_click_time = now
                pyautogui.click()
                cv2.putText(frame, "CLICK", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

        cv2.imshow("Virtual Mouse", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
