# ---------------------------------------------------------
# Appendix: Script for Comparative Analysis of YOLO Series
# ---------------------------------------------------------

from ultralytics import YOLO
import os

# 1. Configuration
# Note: Dataset and output paths should be adjusted to the execution environment.
DATASET_PATH = "path/to/dataset/data.yaml"
OUTPUT_DIR = "path/to/output/yolo_results"

# 2. Target YOLO Models
# Including v5 (Nano/Medium), v8 (Medium), v11 (Medium), and v12 (Medium)
target_yolo_models = [
    "yolov5nu.pt", 
    "yolov5mu.pt", 
    "yolov8m.pt", 
    "yolo11m.pt", 
    "yolo12m.pt"
]

def train_yolo_model(model_variant, data_path, epochs=200, imgsz=640):
    """
    Load a YOLO model variant and execute training and validation.
    """
    print(f"\n--- Initializing Experiment: {model_variant} ---")
    
    # Load model
    model = YOLO(model_variant)
    
    # Training process
    # Results are saved in the directory specified by 'project' and 'name'
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        project=OUTPUT_DIR,
        name=model_variant.split('.')[0],
        exist_ok=True
    )

    # Validation process
    metrics = model.val()
    
    # Summarize performance metrics
    print(f"Metrics for {model_variant}:")
    print(f"mAP 50-95: {metrics.box.map:.4f}")
    print(f"mAP 50: {metrics.box.map50:.4f}")

# 3. Main Execution Loop
if __name__ == "__main__":
    for variant in target_yolo_models:
        try:
            train_yolo_model(variant, DATASET_PATH)
        except Exception as e:
            print(f"Failed to process {variant}: {e}")



