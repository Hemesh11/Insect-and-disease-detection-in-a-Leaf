from ultralytics import YOLO
import torch

def main():
    """
    Trains a YOLOv8s model on the custom crop insect dataset.
    """
    # Check if a GPU is available and print the device information
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pre-trained YOLOv8s model.
    # Using a pre-trained model (transfer learning) is crucial for high accuracy.
    model = YOLO('yolov8s.pt')

    print("Starting YOLOv8s model training for crop insect detection...")
    
    # Train the model using the dataset configuration file
    # The model will be trained to detect 'crop insect' as specified in the hackathon.
    results = model.train(
        data='insect_dataset.yaml', # Path to your dataset config file
        epochs=100,                  # Number of training cycles. 100 is a good start.
        imgsz=640,                   # Image size for training
        batch=8,                     # Number of images to process at once. Adjust if you have memory errors.
        name='yolov8s_insect_detection' # Name for the output folder
    )

    # print("\nTraining complete!")
    # print("Results and trained model weights are saved in the 'runs/detect/yolov8s_insect_detection' folder.")
    print("\nTraining finished!")
    print(f"Model and results saved in 'runs/detect/{results.save_dir.name}'")
    # (Optional) You can print a summary of the results
    print(results)

if __name__ == '__main__':
    main()