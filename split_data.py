import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
# The folder containing your final augmented dataset (images and labels)
SOURCE_DIR = Path("insects_Aug/obj_train_data")

# The folder where the split dataset will be saved
OUTPUT_DIR = Path("insect_dataset_split")

# The ratio for the training set (e.g., 0.8 means 80% train, 20% validation)
TRAIN_RATIO = 0.8
# --- END CONFIGURATION ---

def split_dataset():
    """
    Splits a YOLO dataset into training and validation sets.
    """
    print("Setting up output directories...")
    # Create the output directory structure
    train_images_path = OUTPUT_DIR / "images" / "train"
    val_images_path = OUTPUT_DIR / "images" / "val"
    train_labels_path = OUTPUT_DIR / "labels" / "train"
    val_labels_path = OUTPUT_DIR / "labels" / "val"

    # Create directories, wiping them clean if they already exist
    for path in [train_images_path, val_images_path, train_labels_path, val_labels_path]:
        path.mkdir(parents=True, exist_ok=True)

    print("Finding all image files...")
    # Get all image files from the source directory
    image_files = list(SOURCE_DIR.glob("*.jpg")) + list(SOURCE_DIR.glob("*.png"))
    random.shuffle(image_files)

    # Calculate the split index
    split_index = int(len(image_files) * TRAIN_RATIO)

    # Divide the files into training and validation sets
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # Function to copy files
    def copy_files(file_list, image_dest, label_dest):
        for img_path in file_list:
            label_path = img_path.with_suffix(".txt")
            
            # Copy image file
            shutil.copy(img_path, image_dest)
            
            # Copy label file if it exists
            if label_path.exists():
                shutil.copy(label_path, label_dest)

    print("\nCopying training files...")
    copy_files(train_files, train_images_path, train_labels_path)
    
    print("Copying validation files...")
    copy_files(val_files, val_images_path, val_labels_path)
    
    print("\nDataset split successfully!")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    split_dataset()