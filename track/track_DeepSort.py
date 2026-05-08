# ---------------------------------------------------------
# Appendix: Object Tracking and Counting Script (RT-DETR + DeepSORT)
# ---------------------------------------------------------

import cv2
import os
from ultralytics import RTDETR  # RTDETRを明示的にインポート
from deep_sort_realtime.deepsort_tracker import DeepSort

# 1. Configuration and Parameter Settings
VIDEO_PATH = "path/to/input_video.mp4"
MODEL_PATH = "path/to/trained_rtdetr_weights.pt"
OUTPUT_PATH = "path/to/output_tracked_video.mp4"

# Define class names and corresponding visualization colors
CLASS_NAMES = {0: "dried_flowers", 1: "flowers", 2: "fruits"}
COLORS = {0: (92, 113, 166), 1: (220, 220, 255), 2: (0, 255, 0)}

# Confidence Threshold settings for experiments
CONF_THRESHOLD = 0.45
# CONF_THRESHOLD = 0.60
# CONF_THRESHOLD = 0.75
# CONF_THRESHOLD = 0.90

# 2. Initialize Tracker and RT-DETR Model
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
model = RTDETR(MODEL_PATH)

# Sets to store unique track IDs for each class
unique_id_sets = {cid: set() for cid in CLASS_NAMES.keys()}

def run_tracking():
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Object Detection using RT-DETR ---
        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf >= CONF_THRESHOLD and cls_id in CLASS_NAMES:
                # DeepSORT format: ([left, top, w, h], confidence, class_id)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        # --- Tracker Update ---
        tracks = tracker.update_tracks(detections, frame=frame)

        # --- Visualization and Counting ---
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_id = track.get_det_class()
            
            if cls_id is not None and int(cls_id) in CLASS_NAMES:
                cls_id = int(cls_id)
                unique_id_sets[cls_id].add(track_id)

                ltrb = track.to_ltrb()
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), 
                              (int(ltrb[2]), int(ltrb[3])), COLORS[cls_id], 2)
                
        writer.write(frame)

    cap.release()
    writer.release()

    # Output final count results
    print("\n--- Final Counts (Unique IDs) ---")
    for cid, name in CLASS_NAMES.items():
        print(f"{name}: {len(unique_id_sets[cid])}")

if __name__ == "__main__":
    run_tracking()

