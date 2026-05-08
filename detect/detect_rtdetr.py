# ---------------------------------------------------------
# Appendix: Script for Training and Inference of RT-DETR
# ---------------------------------------------------------

from ultralytics import RTDETR
import os

# 1. Configuration
# Note: Ensure the 'ultralytics' library is installed.
# Path to the dataset configuration file.
DATASET_PATH = "path/to/dataset/data.yaml"
OUTPUT_DIR = "path/to/output/rtdetr_results"

# 2. Model Training
def train_rtdetr_model(model_variant="rtdetr-l.pt", data_path=DATASET_PATH, epochs=200):
    """
    Initialize the RT-DETR model and execute the training process.
    """
    print(f"\n--- Initializing RT-DETR Training: {model_variant} ---")
    
    # Load the pre-trained RT-DETR model
    model = RTDETR(model_variant)
    
    # Start training
    # Batch size and image size can be adjusted based on GPU memory.
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        project=OUTPUT_DIR,
        name="rtdetr_experiment",
        plots=True
    )
    
    # Path to the best performing model weights
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"Training completed. Best model saved at: {best_model_path}")
    return model

# 3. Model Inference (Detection)
def run_rtdetr_inference(model_path, source_images):
    """
    Perform object detection on test images using the trained RT-DETR model.
    """
    # Load the trained weights
    model = RTDETR(model_path)
    
    # Run inference
    # Results include bounding boxes, class probabilities, and labels.
    results = model.predict(
        source=source_images,
        conf=0.25,
        save=True,
        save_txt=True,
        project=OUTPUT_DIR,
        name="inference_results"
    )
    print(f"Inference completed. Results saved in: {OUTPUT_DIR}/inference_results")

# 4. Main Execution
if __name__ == "__main__":
    # Example: Execute training
    # trained_model = train_rtdetr_model("rtdetr-l.pt")
    
    # Example: Execute inference using trained weights
    # run_rtdetr_inference("path/to/best.pt", "path/to/test/images")
    pass
# ---------------------------------------------------------