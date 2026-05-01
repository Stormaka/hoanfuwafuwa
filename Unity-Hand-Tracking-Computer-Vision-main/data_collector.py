# -*- coding: utf-8 -*-
"""
Data Collector for Hand Gesture LSTM Training
==============================================
Thu thập dữ liệu cử chỉ tay thủ công để train LSTM.

Cách dùng:
  1. Chạy: python data_collector.py
  2. Đặt tay vào camera
  3. Bấm phím để bắt đầu ghi sequence cho từng cử chỉ:
       [0] -> idle         (không có cử chỉ)
       [1] -> forward      (1 ngón trỏ)
       [2] -> forward_left (chữ V)
       [3] -> handbrake    (xoè bàn tay)
       [4] -> reverse      (nắm tay)
  4. Giữ nguyên cử chỉ trong 30 frames (khoảng 1 giây)
  5. Lặp lại tối thiểu 30 lần mỗi class
  6. Bấm [Q] để thoát khi xong
"""

import sys, io, os, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN   = 30      # frames per sequence
FEATURES  = 63      # 21 landmarks × 3 (x, y, z)
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sequences")

CLASSES = {
    0: "idle",
    1: "forward",
    2: "forward_left",
    3: "handbrake",
    4: "reverse",
}

# Tạo thư mục lưu data
for cls in CLASSES.values():
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

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

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_features(lms):
    """21 landmarks → flat array (63,) normalized around wrist."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])
    # Normalize relative to wrist (index 0)
    coords -= coords[0]
    return coords.flatten()

def count_saved(cls_name):
    path = os.path.join(DATA_DIR, cls_name)
    return len([f for f in os.listdir(path) if f.endswith(".npy")])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No camera found.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state        = "IDLE"     # IDLE | COUNTDOWN | RECORDING
    active_cls   = None
    sequence     = []
    countdown    = 0
    ts_ms        = 0

    print("\n=== DATA COLLECTOR ===")
    print("Bam so [0-4] de bat dau ghi cu chi.")
    print("Moi sequence = 30 frames. Ghi toi thieu 30 sequence/class.")
    print("Bam [Q] de thoat.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        ts_ms += 33

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, ts_ms)

        hand_features = None
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            p   = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, p[a], p[b], (255,255,255), 2)
            for x, y in p:
                cv2.circle(frame, (x,y), 5, (0,255,180), -1)
            hand_features = extract_features(lms)

        # ── State machine ────────────────────────────────────────────────────
        if state == "IDLE":
            pass

        elif state == "COUNTDOWN":
            countdown -= 1
            cv2.putText(frame, f"Chuan bi... {countdown//10 + 1}", (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,220,255), 3)
            if countdown <= 0:
                state    = "RECORDING"
                sequence = []

        elif state == "RECORDING":
            if hand_features is not None:
                sequence.append(hand_features)

            # Progress bar
            progress = int((len(sequence) / SEQ_LEN) * (w - 40))
            cv2.rectangle(frame, (20, h-40), (20+progress, h-15), (80,255,80), -1)
            cv2.rectangle(frame, (20, h-40), (w-20, h-15), (200,200,200), 2)
            cv2.putText(frame, f"GHI: {len(sequence)}/{SEQ_LEN}", (w//2-60, h-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if len(sequence) >= SEQ_LEN:
                # Save
                cls_name  = CLASSES[active_cls]
                save_path = os.path.join(DATA_DIR, cls_name,
                                         f"{int(time.time()*1000)}.npy")
                np.save(save_path, np.array(sequence))
                n = count_saved(cls_name)
                print(f"[SAVED] {cls_name} → #{n}  ({save_path})")
                state = "IDLE"

        # ── HUD ──────────────────────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w, 130), (15,15,28), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "DATA COLLECTOR - LSTM TRAINING", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,220,255), 2)

        # Class buttons
        for idx, cls in CLASSES.items():
            n    = count_saved(cls)
            col  = (50,255,110) if n >= 30 else (0,180,255) if n >= 10 else (200,200,200)
            text = f"[{idx}] {cls:15s} {n:3d} seq"
            cv2.putText(frame, text, (14 + (idx % 3)*380, 65 + (idx//3)*35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        if active_cls is not None and state in ("COUNTDOWN","RECORDING"):
            cls_name = CLASSES[active_cls]
            col_map  = {0:(180,180,180),1:(80,255,80),2:(0,220,255),3:(50,80,255),4:(100,100,255)}
            cv2.putText(frame, f"CLASS: {cls_name.upper()}", (14, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_map.get(active_cls,(200,200,200)), 2)

        hand_status = "HAND DETECTED" if hand_features is not None else "NO HAND"
        hand_col    = (80,255,80) if hand_features is not None else (50,50,255)
        cv2.putText(frame, hand_status, (w-230, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_col, 2)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif chr(key).isdigit() and state == "IDLE":
            idx = int(chr(key))
            if idx in CLASSES:
                active_cls = idx
                state      = "COUNTDOWN"
                countdown  = 30  # ~1 second at 30fps

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\n=== DATA SUMMARY ===")
    for cls in CLASSES.values():
        n = count_saved(cls)
        print(f"  {cls:20s}: {n:3d} sequences")
    print("\nChay `python train_lstm.py` de train model!")

if __name__ == "__main__":
    main()
