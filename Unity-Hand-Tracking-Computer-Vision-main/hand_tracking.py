# -*- coding: utf-8 -*-
"""
Unity Car Hand Tracking - MediaPipe 0.10+ Tasks API
====================================================
Detects hand gestures via webcam using MediaPipe Tasks API,
sends car control commands to Unity via peaceful-pie.

Gestures:
  1 finger (index)  -> Forward
  V-sign (2 fingers)-> Forward + Left
  Open palm         -> Handbrake
  Fist              -> Reverse
  No hand           -> Idle

Press [Q] to quit.
"""

import sys
import io
import os
import urllib.request
import threading
import collections

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Peaceful-Pie ─────────────────────────────────────────────────────────────
UNITY_CONNECTED = False
unity_comms = None
try:
    from peaceful_pie.unity_comms import UnityComms
    PORT = 5000
    unity_comms = UnityComms(PORT)
    print(f"[OK] Connected to Unity on port {PORT}")
    UNITY_CONNECTED = True
except Exception as e:
    print(f"[WARNING] Unity not connected: {e}")
    print("[INFO] Standalone mode - gesture display only")

# ── Model Download ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"[INFO] Downloading hand_landmarker.task (~6 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[OK] Model saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)
else:
    print(f"[OK] Model found: {MODEL_PATH}")

# ── Landmark IDs ──────────────────────────────────────────────────────────────
WRIST                = 0
THUMB_IP, THUMB_TIP  = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP     = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_TIP   = 13, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# ── Gesture Logic ─────────────────────────────────────────────────────────────
def pts(lms, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in lms]

def is_open_palm(p):
    return (p[THUMB_TIP][1]   < p[THUMB_IP][1]   and
            p[INDEX_TIP][1]   < p[INDEX_PIP][1]   and
            p[MIDDLE_TIP][1]  < p[MIDDLE_PIP][1]  and
            p[RING_TIP][1]    < p[RING_MCP][1]    and
            p[PINKY_TIP][1]   < p[PINKY_PIP][1])

def is_fist(p):
    return (p[INDEX_TIP][1]  > p[INDEX_MCP][1]  and
            p[MIDDLE_TIP][1] > p[MIDDLE_MCP][1] and
            p[RING_TIP][1]   > p[RING_MCP][1]   and
            p[PINKY_TIP][1]  > p[PINKY_MCP][1])

def is_index_up(p):
    if is_open_palm(p):
        return False
    return (p[INDEX_TIP][1] < p[INDEX_DIP][1] < p[INDEX_PIP][1] < p[INDEX_MCP][1] < p[WRIST][1])

def is_v_sign(p):
    if not is_index_up(p) or is_open_palm(p):
        return False
    return (p[MIDDLE_TIP][1] < p[MIDDLE_DIP][1] < p[MIDDLE_PIP][1] < p[MIDDLE_MCP][1] < p[WRIST][1])

def classify(p):
    if is_open_palm(p): return "handbrake"
    if is_v_sign(p):    return "forward_left"
    if is_index_up(p):  return "forward"
    if is_fist(p):      return "reverse"
    return "idle"

# ── Unity Commands ────────────────────────────────────────────────────────────
def _send_action_blocking(gesture):
    """Internal: blocking Unity call (runs in background thread)."""
    if not UNITY_CONNECTED or unity_comms is None:
        return
    try:
        if gesture == "forward":
            unity_comms.GoForward_5000()
        elif gesture == "reverse":
            unity_comms.GoReverse_5000()
        elif gesture == "forward_left":
            unity_comms.GoForward_5000()
            unity_comms.TurnLeft_5000()
        elif gesture == "handbrake":
            unity_comms.Handbrake_5000()
    except Exception:
        pass  # Ignore Unity errors

def send_action(gesture):
    """Send action to Unity in a background thread to prevent UI freeze."""
    threading.Thread(target=_send_action_blocking, args=(gesture,), daemon=True).start()

# ── Camera ────────────────────────────────────────────────────────────────────
def find_camera():
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] Camera found at index {idx}")
                return cap
            cap.release()
    return None

# ── Anomaly Detector ──────────────────────────────────────────────────────────
class AnomalyDetector:
    """Detects sudden large movements of the hand between frames."""
    def __init__(self, window=20, z_thresh=3.0):
        self.velocities = collections.deque(maxlen=window)
        self.prev_pts  = None
        self.z_thresh  = z_thresh
        self.is_anomaly = False
        self.score      = 0.0

    def update(self, pixel_pts):
        """pixel_pts: list of (x,y) from pts() helper."""
        if self.prev_pts is None:
            self.prev_pts = pixel_pts
            return
        # mean displacement of all landmarks this frame
        disp = sum(
            ((x - px)**2 + (y - py)**2)**0.5
            for (x, y), (px, py) in zip(pixel_pts, self.prev_pts)
        ) / max(len(pixel_pts), 1)
        self.prev_pts = pixel_pts
        self.velocities.append(disp)
        if len(self.velocities) < 5:
            return
        import statistics
        mean = statistics.mean(self.velocities)
        stdev = statistics.stdev(self.velocities) or 1e-6
        self.score = (disp - mean) / stdev
        self.is_anomaly = self.score > self.z_thresh

# ── Action Predictor ──────────────────────────────────────────────────────────
class ActionPredictor:
    """Predicts likely next gesture based on recent gesture history."""
    TRANSITIONS = {
        "forward":      "forward",
        "forward_left": "forward",
        "reverse":      "idle",
        "handbrake":    "idle",
        "idle":         "forward",
    }
    def __init__(self, history_len=8):
        self.history = collections.deque(maxlen=history_len)

    def update(self, gesture):
        if not self.history or self.history[-1] != gesture:
            self.history.append(gesture)

    def predict_next(self):
        if not self.history:
            return "unknown"
        return self.TRANSITIONS.get(self.history[-1], "unknown")

# ── HUD ───────────────────────────────────────────────────────────────────────
COLORS = {
    "forward":      (80, 255, 80),
    "forward_left": (0, 220, 255),
    "handbrake":    (50, 80, 255),
    "reverse":      (100, 100, 255),
    "idle":         (180, 180, 180),
}
LABELS = {
    "forward":      ">>> FORWARD",
    "forward_left": ">>> FWD + LEFT (V-sign)",
    "handbrake":    "!!! HANDBRAKE !!!",
    "reverse":      "<<< REVERSE",
    "idle":         "-- IDLE --",
}

def draw_hud(frame, gesture, connected, next_gesture="unknown", anomaly=False):
    h, w = frame.shape[:2]
    color = COLORS.get(gesture, (180, 180, 180))
    label = LABELS.get(gesture, "IDLE")

    # ── Top panel ──
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 105), (15, 15, 28), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "UNITY CAR  -  HAND TRACKING", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    cv2.putText(frame, label, (14, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2)

    # Predicted next gesture
    next_label = LABELS.get(next_gesture, next_gesture.upper())
    cv2.putText(frame, f"NEXT: {next_label}", (14, 97),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)

    badge = "UNITY: CONNECTED" if connected else "UNITY: STANDALONE"
    badge_c = (50, 255, 110) if connected else (50, 150, 255)
    cv2.putText(frame, badge, (w - 360, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, badge_c, 2)

    # ── Anomaly alert ──
    if anomaly:
        ov3 = frame.copy()
        cv2.rectangle(ov3, (0, 110), (w, 148), (0, 0, 180), -1)
        cv2.addWeighted(ov3, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "!!! ANOMALY: SUDDEN MOVEMENT DETECTED !!!",
                    (14, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)

    # ── Bottom guide ──
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 110), (w, h), (10, 10, 22), -1)
    cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)

    guides = [
        "1 finger -> Forward  |  V-sign -> Fwd+Left",
        "Open palm -> Handbrake  |  Fist -> Reverse  |  [Q] Quit",
    ]
    for i, g in enumerate(guides):
        cv2.putText(frame, g, (12, h - 72 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)

# ── Connection drawing with Tasks API results ─────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, lms, w, h):
    p = pts(lms, w, h)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, p[a], p[b], (255, 255, 255), 2)
    for x, y in p:
        cv2.circle(frame, (x, y), 5, (0, 255, 180), -1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=RunningMode.VIDEO,
    )
    detector = HandLandmarker.create_from_options(opts)

    cap = find_camera()
    if cap is None:
        print("[ERROR] No camera found! Connect a webcam and try again.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== UNITY CAR HAND TRACKING ===")
    print("  [1 finger]   -> Forward")
    print("  [V-sign]     -> Forward + Left")
    print("  [Open palm]  -> Handbrake")
    print("  [Fist]       -> Reverse")
    print("  [Q]          -> Quit\n")

    ts_ms = 0
    current_gesture = "idle"

    anomaly_detector = AnomalyDetector(window=20, z_thresh=3.0)
    action_predictor  = ActionPredictor(history_len=8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Cannot read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms += 33
        current_gesture = "idle"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, ts_ms)

        if result.hand_landmarks:
            for lms in result.hand_landmarks:
                draw_landmarks(frame, lms, w, h)
                p = pts(lms, w, h)
                current_gesture = classify(p)  # unchanged gesture logic
                # ── AI observers (read-only, do not affect gesture) ──
                anomaly_detector.update(p)
                action_predictor.update(current_gesture)
                break
        else:
            current_gesture = "idle"

        next_gesture = action_predictor.predict_next()
        is_anomaly   = anomaly_detector.is_anomaly

        send_action(current_gesture)
        draw_hud(frame, current_gesture, UNITY_CONNECTED,
                 next_gesture=next_gesture, anomaly=is_anomaly)
        cv2.imshow("Unity Car - Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
