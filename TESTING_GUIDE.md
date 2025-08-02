# 📋 How to Test Your Multimodal System

## 🚀 Quick Testing Guide

### 1. Test Individual Predictions

#### For Insect Detection:
```python
# Example symptom input (30 binary values)
insect_symptoms = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
insect_image_path = "test_insect/14.png"

# Run prediction
prediction, confidence = predict_insect(insect_image_path, insect_symptoms)
```

#### For Disease Detection:
```python
# Example symptom input (30 binary values)
disease_symptoms = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
disease_image_path = "1_Crop_Disease_DS/disease1.jpg"

# Run prediction
prediction, confidence = predict_disease(disease_image_path, disease_symptoms)
```

### 2. Symptom Input Format

**Each symptom array should have exactly 30 binary values (0 or 1):**

#### Common Insect Symptoms:
1. Holes in leaves (0/1)
2. Chewed leaf edges (0/1)
3. Sticky honeydew on leaves (0/1)
4. Discolored spots (0/1)
5. Curling leaves (0/1)
... (25 more symptoms)

#### Common Disease Symptoms:
1. Yellow spots on leaves (0/1)
2. Brown patches (0/1)
3. Wilting (0/1)
4. Stunted growth (0/1)
5. Powdery coating (0/1)
... (25 more symptoms)

### 3. Expected Output Format

```
🐛 Insect Detection Result:
YOLO confidence (weighted): 1.84
YOLO confidence (unweighted): 0.92
TabNet probability: 0.75
Fusion prediction: 1 (probability: 0.82)

🌿 Disease Detection Result:
YOLO confidence (weighted): 1.40
YOLO confidence (unweighted): 0.70
TabNet probability: 0.90
Fusion prediction: 1 (probability: 0.85)
```

### 4. Image Requirements

- **Format**: JPG, PNG
- **Size**: Any size (YOLO will auto-resize)
- **Content**: Clear crop images showing potential insects/diseases
- **Quality**: Good lighting, focused images work best

### 5. Performance Testing

Run the `performance_evaluation.ipynb` notebook to get:
- ✅ Complete accuracy metrics
- ✅ Precision, Recall, F1-scores
- ✅ ROC-AUC scores
- ✅ Confusion matrices
- ✅ Performance visualizations
- ✅ Comprehensive report

### 6. Model Files Required

Make sure these files exist:
- `fusion_classifier.joblib` (insect fusion model)
- `fusion_disease_classifier.joblib` (disease fusion model)
- `tabnet_insect.zip.zip` (insect TabNet model)
- `tabnet_disease.zip.zip` (disease TabNet model)
- YOLO weight files in respective directories

### 7. Quick Performance Check

```python
# Load the performance evaluation notebook
jupyter notebook performance_evaluation.ipynb

# Or run individual cells to get specific metrics
```

## 🎯 Expected Performance Ranges

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|---------|----------|
| YOLO Only | 85-95% | 0.80-0.95 | 0.75-0.90 | 0.80-0.92 |
| TabNet Only | 75-90% | 0.70-0.90 | 0.70-0.95 | 0.72-0.92 |
| Fusion Model | 85-95% | 0.82-0.95 | 0.80-0.95 | 0.83-0.95 |



## 📊 Getting All Performance Metrics

Simply run the `performance_evaluation.ipynb` notebook - it will:
1. Load all your trained models
2. Calculate comprehensive metrics
3. Generate visualizations
4. Create a detailed performance report
5. Save everything as CSV and PNG files

This gives you publication-ready performance statistics for your multimodal system! 🎉
