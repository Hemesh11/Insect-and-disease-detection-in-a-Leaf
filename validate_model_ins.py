from ultralytics import YOLO

def validate_my_model():
    """
    This function contains the core logic for loading and validating the model.
    """
    # --- CONFIGURATION ---
    # Path to your trained model weights
    MODEL_PATH = 'insect_dataset_split/runs/detect/yolov8s_insect_detection/weights/best.pt'

    # Path to your dataset configuration file
    DATA_YAML_PATH = 'insect_dataset_split/insect_dataset.yaml'

    # Load your custom trained model
    model = YOLO(MODEL_PATH)

    print("Starting validation...")

    # Validate the model on the 'val' set defined in your data.yaml
    metrics = model.val(
        data=DATA_YAML_PATH,
        split='val' # Explicitly specify the validation split
    )

    print("\nValidation Metrics:")
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p[0]:.4f}")
    print(f"Recall: {metrics.box.r[0]:.4f}")

# --- THIS IS THE FIX ---
# This block ensures the code only runs when the script is executed directly.
if __name__ == '__main__':
    validate_my_model()