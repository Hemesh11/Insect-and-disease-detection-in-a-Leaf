# 🌾 Multimodal Crop Disease and Insect Detection System

A comprehensive AI-powered solution for agricultural pest and disease detection using computer vision and tabular data fusion.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)


## 🔍 Overview

This project implements a multimodal machine learning system that combines:
- **Computer Vision (YOLO)**: Image-based detection of crop diseases and insects
- **Tabular Data (TabNet)**: Symptom-based classification using farmer input
- **Fusion Models**: Integration of both modalities for enhanced accuracy

The system helps farmers identify crop threats through either visual inspection or symptom reporting, providing a comprehensive diagnostic tool for precision agriculture.

## ✨ Features

### 🖼️ Image-Based Detection
- **YOLOv8s** models for real-time object detection
- Disease detection on crop leaves with segmentation
- Insect detection with bounding box localization
- High accuracy with confidence scoring

### 📊 Symptom-Based Classification
- **TabNet** models for tabular data processing
- 30 binary symptom questions for each category
- Interpretable feature importance
- Robust to missing data

### 🔗 Multimodal Fusion
- **MLP-based fusion** combining YOLO and TabNet outputs
- Weighted prioritization (YOLO-priority by default)
- Enhanced accuracy through complementary information
- Real-time inference capability

## 🏗️ Architecture

### System overview showing how YOLO and TabNet outputs are fused.

Input Layer
├── Image Input → YOLOv8s → Confidence Scores
└── Symptom Input → TabNet → Probability Scores
                        ↓
                 Fusion Layer (MLP)
                        ↓
              Final Classification Output


### Model Components:
1. **YOLO Models**: 
   - Insect Detection: YOLOv8s (detection)
   - Disease Detection: YOLOv8s (segmentation)

2. **TabNet Models**:
   - Binary classification for presence/absence
   - 30 symptom features each

3. **Fusion Models**:
   - MLP with (16, 8) hidden layers
   - Weighted input combination
   - Logistic regression alternative available

## 📂 Dataset

### Image Data
- **Insect Dataset**: 800+ annotated images with bounding boxes
- **Disease Dataset**: 200+ annotated images with segmentation masks
- **Augmentation**: 5x factor using Albumentations library

### Symptom Data
- **Insect Symptoms**: 640 synthetic samples, 30 binary features
- **Disease Symptoms**: 200 synthetic samples, 30 binary features
- **Labels**: Generated using rule-based heuristics (5+ critical symptoms = positive)

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Dependencies
```bash
pip install ultralytics pytorch-tabnet scikit-learn pandas numpy opencv-python joblib
pip install albumentations matplotlib seaborn
```

### Setup
```bash
git clone <repository-url>
cd AGRITHON
```

## 💻 Usage

### 1. Train Individual Models

#### YOLO Training
```bash
# Train insect detection model
python insect_dataset_split/train_yolo.py

# Train disease detection model (update paths in script)
python train_disease_yolo.py
```

#### TabNet Training
```bash
python tabnet_Train.py
```

### 2. Train Fusion Models

#### Insect Fusion
```python
# Run fusion_insect.ipynb
jupyter notebook fusion_insect.ipynb
```

#### Disease Fusion
```python
# Run fusion_disease.ipynb
jupyter notebook fusion_disease.ipynb
```

### 3. Inference

#### Single Prediction Example
```python
from fusion_models import predict_insect, predict_disease

# Insect prediction
symptoms = [1, 0, 1, 0, ...]  # 30 binary values
image_path = "path/to/insect_image.jpg"
prediction, confidence = predict_insect(image_path, symptoms)

# Disease prediction
symptoms = [0, 1, 0, 1, ...]  # 30 binary values
image_path = "path/to/disease_image.jpg"
prediction, confidence = predict_disease(image_path, symptoms)
```

## 📊 Model Performance

### Individual Model Performance

| Model | Task | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|---------|----------|
| YOLO (Insect) | Detection | 92%+ | - | - | - |
| YOLO (Disease) | Segmentation | 90%+ | - | - | - |
| TabNet (Insect) | Classification | 76% | 0.75 | 0.78 | 0.76 |
| TabNet (Disease) | Classification | 90% | 0.89 | 0.91 | 0.90 |

### Fusion Model Performance

| Model | Task | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|---------|----------|
| MLP Fusion (Insect) | Classification | 85%+ | 0.84 | 0.86 | 0.85 |
| MLP Fusion (Disease) | Classification | 92%+ | 0.91 | 0.93 | 0.92 |

*Note: Performance may vary based on dataset quality and training parameters*

## 📁 Project Structure

```
AGRITHON/
├── README.md
├── requirements.txt
├── 
├── Models/
│   ├── tabnet_Train.py                 # TabNet training script
│   ├── tabnet_insect.zip              # Trained insect TabNet model
│   └── tabnet_disease.zip             # Trained disease TabNet model
│
├── YOLO_Training/
│   ├── insect_dataset_split/
│   │   ├── train_yolo.py              # YOLO training script
│   │   ├── insect_dataset.yaml        # Dataset configuration
│   │   └── runs/detect/               # Training outputs
│   └── Disease_training/              # Disease YOLO training
│
├── Fusion_Models/
│   ├── fusion_insect.ipynb           # Insect fusion pipeline
│   ├── fusion_disease.ipynb          # Disease fusion pipeline
│   ├── fusion_classifier.joblib      # Trained insect fusion model
│   └── fusion_disease_classifier.joblib # Trained disease fusion model
│
├── Data/
│   ├── 1_Crop_Disease_DS/            # Disease images
│   ├── 2_Crop_Insect_DS/             # Insect images
│   ├── insect_symptom_dataset_640.csv # Insect symptom data
│   └── disease_symptom_dataset_200.csv # Disease symptom data
│
├── Utils/
│   ├── aug.py                        # Data augmentation
│   ├── split_data.py                 # Dataset splitting
│   ├── predict.py                    # YOLO inference
│   └── validate_model_ins.py         # Model validation
│
└── Notebooks/
    └── agrithon-insect-pred-tabnet.ipynb # TabNet exploration
```

## 🎯 Results

### Key Achievements
- ✅ **Dual Modality Support**: Both image and symptom-based detection
- ✅ **Real-time Inference**: Fast prediction suitable for field use
- ✅ **High Accuracy**: 85%+ accuracy on fusion models
- ✅ **Scalable Architecture**: Easy to extend to new crops/pests
- ✅ **Production Ready**: Saved models ready for deployment

### Sample Outputs
```
🐛 Insect Detection Result:
YOLO confidence (weighted): 1.84
TabNet probability: 0.75
Fusion prediction: 1 (probability: 0.82)

🌿 Disease Detection Result:
YOLO confidence (weighted): 1.84
TabNet probability: 1.00
Fusion prediction: 1 (probability: 0.80)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

**Built with ❤️ for sustainable agriculture and food security**
