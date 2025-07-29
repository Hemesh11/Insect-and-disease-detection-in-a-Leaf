
# # --- CONFIGURATION ---
# # The input directory is now correctly pointed to your image folder.
# INPUT_DIR = r"C:\Users\hemes\Desktop\AGRITHON\crop_insect_annotation\obj_train_data"
# # The output will be saved in a new sibling directory.
# OUTPUT_DIR =  r"C:\Users\hemes\Desktop\AGRITHON\augmented"
# import os
# import cv2
# import shutil
# import albumentations as A
# import numpy as np

# # --- CONFIGURATION
# AUGMENTATIONS_PER_IMAGE = 5
# # --- END CONFIGURATION ---

# # Create the output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Define the augmentation pipeline
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.Rotate(limit=30, p=0.8, border_mode=cv2.BORDER_CONSTANT),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
#     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
#     A.Blur(blur_limit=(3, 7), p=0.2),
#     # --- FIX 1: Corrected GaussNoise ---
#     # Removed the 'var_limit' argument to match your library version
#     A.GaussNoise(p=0.3),
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# # --- Main Logic ---

# image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')]
# print(f"Processing {len(image_files)} images. Copying originals and creating {AUGMENTATIONS_PER_IMAGE} augmentations each...")

# for i, image_name in enumerate(image_files, 1):
#     print(f"Processing {i}/{len(image_files)}: {image_name}")
    
#     image_path = os.path.join(INPUT_DIR, image_name)
#     label_path = os.path.join(INPUT_DIR, image_name.replace('.jpg', '.txt'))
    
#     shutil.copy(image_path, os.path.join(OUTPUT_DIR, image_name))
#     if os.path.exists(label_path):
#         shutil.copy(label_path, os.path.join(OUTPUT_DIR, image_name.replace('.jpg', '.txt')))
    
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     bboxes = []
#     class_labels = []
#     if os.path.exists(label_path):
#         with open(label_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 class_id = int(parts[0])
#                 coords = list(map(float, parts[1:]))
#                 bboxes.append(coords)
#                 class_labels.append(class_id)
    
#     for j in range(AUGMENTATIONS_PER_IMAGE):
#         augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
#         augmented_image = augmented['image']
#         augmented_bboxes = augmented['bboxes']

#         base_name = os.path.splitext(image_name)[0]
#         new_image_name = f"{base_name}_aug_{j}.jpg"
#         new_label_name = f"{base_name}_aug_{j}.txt"

#         new_image_path = os.path.join(OUTPUT_DIR, new_image_name)
#         new_label_path = os.path.join(OUTPUT_DIR, new_label_name)

#         # --- FIX 2: Corrected OpenCV typo ---
#         cv2.imwrite(new_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

#         with open(new_label_path, 'w') as f:
#             if augmented_bboxes:
#                 for k, bbox in enumerate(augmented_bboxes):
#                     class_id = augmented['class_labels'][k]
#                     x_center, y_center, width, height = bbox
#                     f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# print(f"\nProcessing complete.")
# print(f"The '{OUTPUT_DIR}' folder now contains the original files plus all augmented versions.")

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A

class SimpleYOLOAugmentation:
    def __init__(self, input_dir, output_dir, augmentation_factor=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augmentation_factor = augmentation_factor
        
        # Create output directory
        self.setup_output_directory()
        
        # --- MODIFIED: A single, powerful augmentation pipeline ---
        # This pipeline includes the required techniques: flipping, rotation,
        # contrast enhancement, and color jittering, plus other effective methods.
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.6),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.MotionBlur(p=0.5),
            ], p=0.2),
            A.GaussNoise(p=0.2)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))

    def setup_output_directory(self):
        """Create output directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "obj_train_data").mkdir(exist_ok=True)
        
        for file in ["obj.names", "obj.data"]:
            if (self.input_dir / file).exists():
                shutil.copy2(self.input_dir / file, self.output_dir / file)

    def read_yolo_annotation(self, annotation_path):
        """Read YOLO format annotation file"""
        annotations = []
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        annotations.append([class_id, bbox])
        return annotations

    def write_yolo_annotation(self, annotation_path, annotations):
        """Write YOLO format annotation file"""
        with open(annotation_path, 'w') as f:
            for class_id, bbox in annotations:
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    def get_image_files(self):
        """Get all image files from the dataset"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        obj_train_data = self.input_dir / "obj_train_data"
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(obj_train_data.glob(f"*{ext}")))
        
        return image_files

    def process_dataset(self):
        """Process entire dataset with augmentations"""
        image_files = self.get_image_files()
        total_images = len(image_files)
        
        print(f"Processing {total_images} images with {self.augmentation_factor} augmentations each...")
        
        all_image_paths = []
        
        for idx, image_path in enumerate(image_files):
            print(f"Processing {idx+1}/{total_images}: {image_path.name}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            annotation_path = image_path.with_suffix('.txt')
            annotations = self.read_yolo_annotation(annotation_path)
            
            if not annotations:
                print(f"No annotations found for {image_path.name}, skipping...")
                continue
            
            # Copy original image and annotation
            base_name = image_path.stem
            original_img_path = self.output_dir / "obj_train_data" / f"{base_name}.jpg" # Keep original name
            original_ann_path = self.output_dir / "obj_train_data" / f"{base_name}.txt"
            
            cv2.imwrite(str(original_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.write_yolo_annotation(original_ann_path, annotations)
            all_image_paths.append(str(original_img_path.relative_to(self.output_dir)))
            
            # Generate augmented images
            for aug_idx in range(self.augmentation_factor):
                try:
                    bboxes = [ann[1] for ann in annotations]
                    class_labels = [ann[0] for ann in annotations]

                    # --- MODIFIED: Apply the single transform pipeline ---
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    
                    if aug_bboxes:  # Only save if annotations exist after augmentation
                        aug_annotations = [[label, list(bbox)] for label, bbox in zip(transformed['class_labels'], aug_bboxes)]
                        
                        aug_img_path = self.output_dir / "obj_train_data" / f"{base_name}_aug_{aug_idx}.jpg"
                        aug_ann_path = self.output_dir / "obj_train_data" / f"{base_name}_aug_{aug_idx}.txt"
                        
                        cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                        self.write_yolo_annotation(aug_ann_path, aug_annotations)
                        all_image_paths.append(str(aug_img_path.relative_to(self.output_dir)))
                        
                except Exception as e:
                    print(f"Error processing augmentation {aug_idx} for {image_path.name}: {e}")
                    continue
        
        # Create updated train.txt
        train_txt_path = self.output_dir / "train.txt"
        with open(train_txt_path, 'w') as f:
            for img_path in all_image_paths:
                # On Windows, YOLO often works better with forward slashes
                f.write(f"{img_path.replace(os.sep, '/')}\n")
        
        print("\nAugmentation complete!")
        print(f"Original images processed: {total_images}")
        print(f"Total images in output dataset: {len(all_image_paths)}")
        print(f"Output saved to: {self.output_dir}")

def main():
    """Main execution function"""
    input_directory = r"C:\Users\hemes\Desktop\AGRITHON\crop_insect_annotation"
    output_directory = r"C:\Users\hemes\Desktop\AGRITHON\insects_Aug"
    augmentation_factor = 15
    
    augmenter = SimpleYOLOAugmentation(
        input_dir=input_directory,
        output_dir=output_directory,
        augmentation_factor=augmentation_factor
    )
    
    augmenter.process_dataset()

if __name__ == "__main__":
    main()