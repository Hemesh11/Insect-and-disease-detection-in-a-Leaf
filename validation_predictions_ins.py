from ultralytics import YOLO
import cv2
import os

# --- CONFIGURATION ---
MODEL_PATH = 'insect_dataset_split/runs/detect/yolov8s_insect_detection/weights/best.pt'
# Path to your validation images folder
VAL_IMAGE_FOLDER = 'insect_dataset_split/images/val'
# --- END CONFIGURATION ---

model = YOLO(MODEL_PATH)

print(f"Running visual prediction on all images in: {VAL_IMAGE_FOLDER}")
results = model(VAL_IMAGE_FOLDER, stream=True)

for r in results:
    filename = os.path.basename(r.path)
    annotated_frame = r.plot()
    
    cv2.imshow(f"Validation Prediction: {filename}", annotated_frame)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()