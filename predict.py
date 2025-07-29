from ultralytics import YOLO
import cv2
import os

# --- CONFIGURATION ---
# Path to your trained model weights
MODEL_PATH = 'insect_dataset_split/runs/detect/yolov8s_insect_detection/weights/best.pt'

# Path to the FOLDER containing your test images
IMAGE_FOLDER_PATH = 'test_insect'
# --- END CONFIGURATION ---

# Load your custom trained model
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Run inference on the entire folder
# The 'stream=True' argument is efficient for handling multiple images or video frames.
print(f"Running inference on all images in: {IMAGE_FOLDER_PATH}")
results = model(IMAGE_FOLDER_PATH, stream=True)

# Process and display the results for each image
for r in results:
    # Get the original filename to display in the window title
    filename = os.path.basename(r.path)
    
    # 'plot()' creates the image with bounding boxes drawn on it
    annotated_frame = r.plot()
    
    # Display the annotated frame in a window titled with the filename
    cv2.imshow(f"Prediction: {filename}", annotated_frame)
    
    # Wait for a key press. Press 'q' to quit, any other key for the next image.
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

# Clean up and close all windows
cv2.destroyAllWindows()

print("Inference complete.")