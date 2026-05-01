# -*- coding: utf-8 -*-
"""
Topic 8.3: Action Recognition in VR using AI
============================================
Pipeline:
Webcam -> MediaPipe Pose (33 points) -> 30-frame Buffer -> LSTM Model -> Action
Also features:
- Anomaly Detection (velocity/spike based)
- Next-Action Prediction
- VR Bridge to Unity via peaceful-pie
"""

import sys
import io
import os
import cv2
import numpy as np
import collections
import threading
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import urllib.request

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_LEN = 30
FEATURES = 132 # 33 * 4
PORT = 5000
UNITY_CONNECTED = False
unity_comms = None

# Load peaceful-pie
try:
    from peaceful_pie.unity_comms import UnityComms
    unity_comms = UnityComms(PORT)
    print(f"[OK] Connected to Unity on port {PORT}")
    UNITY_CONNECTED = True
except Exception as e:
    print(f"[WARNING] Unity not connected: {e}")
    print("[INFO] Standalone mode - tracking display only")

# ── Load Model & Labels ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "action_lstm.keras")

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("Please run `python create_pretrained_lstm.py` first.")
    sys.exit(1)

lstm_model = tf.keras.models.load_model(MODEL_PATH)

# The actions we trained on
ACTIONS = [
    "idle",            # 0
    "walk_forward",    # 1
    "walk_backward",   # 2
    "turn_left",       # 3
    "turn_right",      # 4
    "stop",            # 5
    "jump"             # 6
]

# ── MediaPipe Tasks API Setup ─────────────────────────────────────────────────
MODEL_URL_POSE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_landmarker_lite.task")

if not os.path.exists(POSE_MODEL_PATH):
    print(f"[INFO] Downloading pose_landmarker_lite.task...")
    try:
        urllib.request.urlretrieve(MODEL_URL_POSE, POSE_MODEL_PATH)
        print(f"[OK] Downloaded {POSE_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to download pose model: {e}")
        sys.exit(1)

base_opts = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
pose_opts = PoseLandmarkerOptions(
    base_options=base_opts,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=RunningMode.VIDEO
)
pose_estimator = PoseLandmarker.create_from_options(pose_opts)

# Connection lines for drawing
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

# ── Core Classes ──────────────────────────────────────────────────────────────
class AnomalyDetector:
    def __init__(self, threshold=0.15):
        self.threshold = threshold
        self.prev_points = None
    
    def check(self, current_points):
        # current_points shape: (33, 4)
        if self.prev_points is None:
            self.prev_points = current_points
            return False, 0.0
        
        # Calculate mean displacement
        # Just use x and y for simplicity
        disp = np.linalg.norm(current_points[:, :2] - self.prev_points[:, :2], axis=1)
        mean_disp = np.mean(disp)
        
        self.prev_points = current_points
        
        is_anomaly = mean_disp > self.threshold
        return is_anomaly, mean_disp

class ActionPredictor:
    def __init__(self):
        self.history = collections.deque(maxlen=10)
        # Simple transition logic for demonstration
        self.transitions = {
            "idle": "walk_forward",
            "walk_forward": "stop",
            "walk_backward": "idle",
            "turn_left": "walk_forward",
            "turn_right": "walk_forward",
            "stop": "idle",
            "jump": "idle"
        }
    
    def add_action(self, action):
        if action != "idle" or len(self.history) == 0 or self.history[-1] != "idle":
            self.history.append(action)
            
    def predict_next(self):
        if len(self.history) == 0:
            return "unknown"
        curr = self.history[-1]
        return self.transitions.get(curr, "unknown")

# ── Unity Commands ────────────────────────────────────────────────────────────
def _send_blocking(action):
    """Internal: blocking Unity call (runs in background thread)."""
    if not UNITY_CONNECTED or unity_comms is None:
        return
    try:
        if action == "walk_forward":
            unity_comms.GoForward_5000()
        elif action == "walk_backward":
            unity_comms.GoReverse_5000()
        elif action == "turn_left":
            unity_comms.TurnLeft_5000()
        elif action == "turn_right":
            unity_comms.TurnRight_5000()
        elif action == "stop":
            unity_comms.Handbrake_5000()
    except Exception:
        pass  # Ignore Unity errors

def send_action_to_unity(action):
    """Send action to Unity in a background thread to prevent UI freeze."""
    threading.Thread(target=_send_blocking, args=(action,), daemon=True).start()

# ── HUD Drawing ───────────────────────────────────────────────────────────────
COLORS = {
    "idle":          (180, 180, 180),
    "walk_forward":  (80, 255, 80),
    "walk_backward": (100, 100, 255),
    "turn_left":     (0, 220, 255),
    "turn_right":    (0, 220, 255),
    "stop":          (50, 80, 255),
    "jump":          (255, 100, 200)
}

def draw_hud(frame, action, conf, anomaly, next_act, connected):
    h, w = frame.shape[:2]
    
    # Top Panel
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 110), (15, 15, 28), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "TOPIC 8.3: AI ACTION RECOGNITION VR", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    
    # Action
    color = COLORS.get(action, (180, 180, 180))
    cv2.putText(frame, f"ACTION: {action.upper()} ({conf*100:.1f}%)", (14, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Prediction
    cv2.putText(frame, f"PREDICT NEXT: {next_act.upper()}", (14, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Anomaly
    if anomaly:
        cv2.putText(frame, "!!! ANOMALY DETECTED !!!", (w//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Unity Status
    badge = "UNITY: CONNECTED" if connected else "UNITY: STANDALONE"
    badge_c = (50, 255, 110) if connected else (50, 150, 255)
    cv2.putText(frame, badge, (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_c, 2)
                
    # Instructions Bottom
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 50), (w, h), (10, 10, 22), -1)
    cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Stand in front of camera to control. Press [Q] to quit.", (14, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No camera found.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State variables
    sequence = []
    current_action = "idle"
    current_conf = 0.0
    
    anomaly_detector = AnomalyDetector()
    action_predictor = ActionPredictor()
    
    print("\n=== TOPIC 8.3 ACTION RECOGNITION STARTING ===")
    
    # Optional fallback gesture logic if synthetic LSTM is confused by real human
    # Since LSTM is trained on pure synthetic points, it will be inaccurate on a real person
    # So we'll combine it with a heuristic override for demonstration
    use_heuristic_override = True

    ts_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms += 33
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = pose_estimator.detect_for_video(mp_img, ts_ms)
        
        is_anomaly = False
        
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            
            # Draw manually
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (255,255,255), 2)
            for x, y in points:
                cv2.circle(frame, (x,y), 4, (0,255,180), -1)
            
            # Extract features
            frame_features = np.zeros((33, 4))
            for i, lm in enumerate(landmarks):
                frame_features[i] = [lm.x, lm.y, lm.z, lm.visibility]
            
            # Anomaly Detection
            is_anomaly, disp = anomaly_detector.check(frame_features)
            
            # Add to sequence buffer
            sequence.append(frame_features.flatten())
            if len(sequence) > SEQ_LEN:
                sequence.pop(0)
                
            # Predict with LSTM
            if len(sequence) == SEQ_LEN:
                input_data = np.expand_dims(np.array(sequence), axis=0)
                preds = lstm_model.predict(input_data, verbose=0)[0]
                idx = np.argmax(preds)
                lstm_action = ACTIONS[idx]
                current_conf = float(preds[idx])
                
                if use_heuristic_override:
                    # Heuristic rules to override synthetic LSTM for real demonstration
                    left_wrist = frame_features[15]
                    right_wrist = frame_features[16]
                    nose = frame_features[0]
                    
                    if left_wrist[1] < nose[1] and right_wrist[1] < nose[1]:
                        current_action = "stop"
                    elif left_wrist[0] > 0.8: # flipped, so right of screen
                        current_action = "turn_right"
                    elif right_wrist[0] < 0.2:
                        current_action = "turn_left"
                    else:
                        current_action = lstm_action # fallback to LSTM
                else:
                    current_action = lstm_action
                
                action_predictor.add_action(current_action)
                
        else:
            current_action = "idle"
            current_conf = 0.0

        next_action = action_predictor.predict_next()
        
        # Dispatch
        send_action_to_unity(current_action)
        
        # Render
        draw_hud(frame, current_action, current_conf, is_anomaly, next_action, UNITY_CONNECTED)
        cv2.imshow("VR Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
