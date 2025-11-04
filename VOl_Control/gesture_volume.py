import time
import math
import cv2
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities  # pycaw provides EndpointVolume via device.EndpointVolume

# =========================
# Config
# =========================
MAX_HANDS = 1
DET_CONF = 0.7
TRK_CONF = 0.7

MIN_DIST = 50      # distance -> min volume
MAX_DIST = 250     # distance -> max volume

SMOOTHING_ALPHA = 0.35  # exponential smoothing for volume
MUTE_DEBOUNCE = 0.8     # seconds between allowed mute toggles

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DET_CONF,
    min_tracking_confidence=TRK_CONF
)

# =========================
# PyCAW - get EndpointVolume
# =========================
speakers = AudioUtilities.GetSpeakers()
volume = speakers.EndpointVolume  # modern pycaw: EndpointVolume is IAudioEndpointVolume

# Get dB range
min_vol_db, max_vol_db, _ = volume.GetVolumeRange()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_db = None
last_pct = None
last_mute_toggle = 0

def is_muted():
    # GetMute returns (1 or 0) in many implementations; catch exceptions
    try:
        return bool(volume.GetMute())
    except Exception:
        # fallback: assume unmuted
        return False

def set_mute(state: bool):
    # state True => mute, False => unmute
    try:
        volume.SetMute(int(state), None)
    except Exception:
        pass

try:
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            break

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w = img.shape[:2]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark

                # Landmarks: thumb tip = 4, index tip = 8, middle tip = 12
                x_thumb, y_thumb = int(lm[4].x * w), int(lm[4].y * h)
                x_index, y_index = int(lm[8].x * w), int(lm[8].y * h)
                x_middle, y_middle = int(lm[12].x * w), int(lm[12].y * h)

                # Draw markers
                cv2.circle(img, (x_thumb, y_thumb), 8, (255, 0, 255), -1)
                cv2.circle(img, (x_index, y_index), 8, (255, 0, 255), -1)
                cv2.circle(img, (x_middle, y_middle), 8, (255, 0, 255), -1)
                cv2.line(img, (x_thumb, y_thumb), (x_index, y_index), (255, 0, 255), 2)

                # --- Volume control using thumb-index distance ---
                length = math.hypot(x_index - x_thumb, y_index - y_thumb)
                length_clamped = clamp(length, MIN_DIST, MAX_DIST)
                target_db = np.interp(length_clamped, [MIN_DIST, MAX_DIST], [min_vol_db, max_vol_db])

                # smoothing
                if last_db is None:
                    smooth_db = target_db
                else:
                    smooth_db = (SMOOTHING_ALPHA * target_db) + ((1 - SMOOTHING_ALPHA) * last_db)
                last_db = smooth_db

                # apply volume
                try:
                    volume.SetMasterVolumeLevel(float(smooth_db), None)
                except Exception:
                    pass

                # percent for UI
                pct = np.interp(length_clamped, [MIN_DIST, MAX_DIST], [0, 100])
                if last_pct is None:
                    smooth_pct = pct
                else:
                    smooth_pct = (SMOOTHING_ALPHA * pct) + ((1 - SMOOTHING_ALPHA) * last_pct)
                last_pct = smooth_pct

                # Draw volume bar
                bar_y = int(np.interp(length_clamped, [MIN_DIST, MAX_DIST], [400, 150]))
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, bar_y), (85, 400), (0, 255, 0), -1)
                cv2.putText(img, f'{int(smooth_pct)}%', (38, 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # --- Mute toggle: thumb touches middle finger ---
                thumb_mid_dist = math.hypot(x_middle - x_thumb, y_middle - y_thumb)
                MUTE_TRIGGER_DIST = 40  # tuneable threshold in pixels
                now = time.time()
                if thumb_mid_dist < MUTE_TRIGGER_DIST and (now - last_mute_toggle) > MUTE_DEBOUNCE:
                    # toggle mute
                    current = is_muted()
                    set_mute(not current)
                    last_mute_toggle = now

                # feedback: show muted status
                muted = is_muted()
                if muted:
                    cv2.putText(img, 'MUTED', (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    cv2.putText(img, 'UNMUTED', (w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.putText(img, 'Thumb-Index: volume | Thumb-Middle: toggle mute', (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('Hand Volume & Mute Control', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
