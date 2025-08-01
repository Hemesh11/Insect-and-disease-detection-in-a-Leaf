# Step 1: Import required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import torch
# Step 2: Load the dataset
# Change path if needed
insect_df = pd.read_csv("insect_symptom_dataset_640.csv")
disease_df = pd.read_csv("disease_symptom_dataset_200.csv")

# Step 3: Function to train a TabNet model and print accuracy
def train_tabnet(df, label_col):
    # X = df.drop(columns=[label_col]).values
    # y = df[label_col].values
    X_df = df.drop(columns=[label_col])
    X = X_df.values
    y = df[label_col].values
    categorical_features_indices = list(range(X_df.shape[1]))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5, gamma=1.5,
        cat_idxs=categorical_features_indices,
        cat_dims=[2 for _ in categorical_features_indices],  # All features are binary (0, 1), so 2 unique values
        cat_emb_dim=1,  # Embedding dimension for categorical features
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',  # "sparsemax" or "entmax"
        verbose=1  # Set to 1 to see training progress
    )
    # model = TabNetClassifier()
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric=["accuracy"],
              max_epochs=100,
              patience=10,
              batch_size=32,
              virtual_batch_size=16,
              num_workers=0)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    print(f"✅ Accuracy on validation set: {round(acc * 100, 2)}%")
    print(f"🔎 Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}")
    return model

# Step 4: Train Insect TabNet
print("🐛 Training Insect TabNet")
insect_model = train_tabnet(insect_df, "Insect_Present_Label")

# Step 5: Train Disease TabNet
print("\n🌿 Training Disease TabNet")
disease_model = train_tabnet(disease_df, "Disease_Present_Label")

insect_model.save_model('tabnet_insect.zip')
disease_model.save_model('tabnet_disease.zip')

