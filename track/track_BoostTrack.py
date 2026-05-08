# ---------------------------------------------------------
# Appendix: Object Tracking and Counting Script (RT-DETR + BoostTrack)
# ---------------------------------------------------------

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import RTDETR
from boxmot import BoostTrack

# 1. Configuration and Parameter Settings
VIDEO_PATH = "path/to/input_video.mp4"
MODEL_PATH = "path/to/trained_rtdetr_weights.pt"
REID_WEIGHTS = "path/to/osnet_x0_25_msmt17.pt"
OUTPUT_PATH = "path/to/output_tracked_video.mp4"

# Class names and visualization colors for Manganji Togarashi
CLASS_NAMES = {0: "dried_flowers", 1: "flowers", 2: "fruits"} #
COLORS = {0: (92, 113, 166), 1: (220, 220, 255), 2: (0, 255, 0)}

# Confidence Threshold settings for experiments
CONF_THRESHOLD = 0.45
# CONF_THRESHOLD = 0.60
# CONF_THRESHOLD = 0.75
# CONF_THRESHOLD = 0.90

# 2. Initialize Models and Tracker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RTDETR(MODEL_PATH)

tracker = BoostTrack(
    reid_weights=Path(REID_WEIGHTS),
    device=device,
    half=False
)

# Sets to store unique track IDs for each class
unique_id_sets = {cid: set() for cid in CLASS_NAMES.keys()}

def run_tracking():
    # Video I/O Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    with torch.inference_mode():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- Object Detection using RT-DETR ---
            results = model(frame, verbose=False)[0]
            
            # Prepare detections for BoostTrack: [x1, y1, x2, y2, conf, cls_id]
            dets_list = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf >= CONF_THRESHOLD and cls_id in CLASS_NAMES:
                    dets_list.append([x1, y1, x2, y2, conf, cls_id])

            dets_array = np.array(dets_list, dtype=np.float32) if dets_list else np.empty((0, 6))

            # --- Tracker Update ---
            # Output format: [x1, y1, x2, y2, track_id, conf, cls_id, ind]
            tracks = tracker.update(dets_array, frame)
            tracks = np.asarray(tracks) if tracks is not None else np.empty((0, 8))

            # --- Visualization and Counting ---
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr[0:4])
                track_id = int(tr[4])
                cls_id = int(tr[6])

                if cls_id in unique_id_sets:
                    unique_id_sets[cls_id].add(track_id)

                    # Draw visualization
                    color = COLORS.get(cls_id, (255, 255, 255))
                    label = f"{CLASS_NAMES[cls_id]} ID:{track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            writer.write(frame)

    cap.release()
    writer.release()

    # Output final summary results
    print("\n--- Final Counts (Unique IDs) ---")
    for cid, name in CLASS_NAMES.items():
        print(f"{name}: {len(unique_id_sets[cid])}")

if __name__ == "__main__":
    run_tracking()

