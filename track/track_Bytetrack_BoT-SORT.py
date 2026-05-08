# ---------------------------------------------------------
# Appendix: Unified Script for Tracking Experiments (ByteTrack / BoT-SORT)
# ---------------------------------------------------------

import cv2
import os
import yaml
import numpy as np
from ultralytics import RTDETR
from pathlib import Path

# 1. Configuration and Parameter Settings
VIDEO_PATH = "path/to/input_video.mp4"
MODEL_PATH = "path/to/trained_rtdetr_weights.pt"
OUTPUT_DIR = "path/to/output_directory"

# Class names for Manganji Togarashi detection
CLASS_NAMES = {0: "dried_flowers", 1: "flowers", 2: "fruits"} #
COLORS = {0: (92, 113, 166), 1: (220, 220, 255), 2: (0, 255, 0)}

# Confidence Threshold settings for experiments
CONF_THRESHOLD = 0.45
# CONF_THRESHOLD = 0.60
# CONF_THRESHOLD = 0.75
# CONF_THRESHOLD = 0.90

# 2. Tracker Selection (Switch between 'bytetrack.yaml' or 'botsort.yaml')
TRACKER_YAML = "bytetrack.yaml" 
# TRACKER_YAML = "botsort.yaml"

# Generate custom BoT-SORT configuration if selected
if TRACKER_YAML == "botsort.yaml":
    botsort_config = {
        'tracker_type': 'botsort',
        'gmc_method': 'none',
        'track_high_thresh': 0.3,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.4,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'with_reid': False,
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.25,
        'fuse_score': True
    }
    with open(TRACKER_YAML, 'w') as f:
        yaml.dump(botsort_config, f)

# 3. Initialize Detection Model
model = RTDETR(MODEL_PATH)

# Sets to store unique track IDs for each class
unique_id_sets = {cid: set() for cid in CLASS_NAMES.keys()}

def run_tracking_experiment():
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"output_{TRACKER_YAML.split('.')[0]}.mp4")
    
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Tracking execution with the selected tracker ---
        results = model.track(
            source=frame,
            tracker=TRACKER_YAML,
            persist=True,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        # --- Process and count unique objects ---
        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            bboxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(bboxes, ids, classes):
                if cls_id in CLASS_NAMES:
                    unique_id_sets[cls_id].add(int(track_id))

                    # Visualization
                    x1, y1, x2, y2 = map(int, box)
                    color = COLORS.get(cls_id, (255, 255, 255))
                    label = f"{CLASS_NAMES[cls_id]} ID:{track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()

    # Final Summary Output
    print(f"\n--- Final Counts ({TRACKER_YAML}) ---")
    for cid, name in CLASS_NAMES.items():
        print(f"{name}: {len(unique_id_sets[cid])}")

if __name__ == "__main__":
    run_tracking_experiment()

