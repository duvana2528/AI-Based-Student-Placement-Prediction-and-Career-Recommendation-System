"""
================================================================
   TRAIN PLACEMENT PREDICTION MODEL
   
   This script trains an XGBoost model on the placement dataset
   and saves the trained model for later use.
   
   RUN THIS FIRST:
   python train_model.py
================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

print("=" * 60)
print("  STEP 1: Load the dataset")
print("=" * 60)

# Load the CSV file with student data
df = pd.read_csv("./placementdata.csv")

print(f"\n✓ Loaded {df.shape[0]} students")
print(f"✓ Features: {df.shape[1]} columns")
print(f"\nFirst few rows:\n{df.head()}\n")


print("=" * 60)
print("  STEP 2: Clean the data")
print("=" * 60)

# Convert "Yes"/"No" → 1/0
df["ExtracurricularActivities"] = (df["ExtracurricularActivities"] == "Yes").astype(int)

# Create target column: "Placed" → 1, "NotPlaced" → 0
df["PlacementStatus_num"] = (df["PlacementStatus"] == "Placed").astype(int)

# Remove duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"✓ Removed {before - len(df)} duplicates")
print(f"✓ Final dataset size: {len(df)} students\n")

# Show class distribution
placed    = df["PlacementStatus_num"].sum()
not_placed = len(df) - placed

print(f"  Placement Status:")
print(f"    Placed     : {placed} students ({placed/len(df)*100:.1f}%)")
print(f"    Not Placed : {not_placed} students ({not_placed/len(df)*100:.1f}%)\n")


print("=" * 60)
print("  STEP 3: Prepare features and target")
print("=" * 60)

# Define the input features (what the model learns from)
FEATURE_COLUMNS = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "SSC_Marks",
    "HSC_Marks",
]

# X = Input features (student characteristics)
# y = Target label (Placed or Not Placed)
X = df[FEATURE_COLUMNS].astype(float)
y = df["PlacementStatus_num"]

print(f"✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}\n")


print("=" * 60)
print("  STEP 4: Train XGBoost model with 5-fold CV")
print("=" * 60)

# Create the XGBoost classifier
# This is a powerful machine learning algorithm that builds
# many decision trees and combines them for better predictions
scale_weight = (y == 0).sum() / (y == 1).sum()  # Handle class imbalance

model = XGBClassifier(
    n_estimators=300,           # Number of trees
    max_depth=6,                # Tree depth
    learning_rate=0.05,         # Learning speed
    subsample=0.8,              # Use 80% of data per tree
    colsample_bytree=0.8,       # Use 80% of features per tree
    scale_pos_weight=scale_weight,  # Balance the classes
    random_state=42,            # For reproducibility
    eval_metric="logloss",
    verbosity=0,
)

# 5-fold cross-validation: train on 4 folds, test on 1 fold (5 times)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running 5-fold cross-validation...\n")

# Calculate performance metrics
accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
f1       = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
auc_roc  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

print(f"Cross-Validation Results:")
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print(f"  AUC-ROC  : {auc_roc*100:.2f}% ← primary metric\n")


print("=" * 60)
print("  STEP 5: Train final model on full dataset")
print("=" * 60)

# Train the model on ALL data (after CV, we use the full dataset
# for the final model to maximize learning)
model.fit(X, y)
print("✓ Model trained on full dataset\n")


print("=" * 60)
print("  STEP 6: Save the trained model")
print("=" * 60)

# Save the trained model to a file so we can use it later
model_path = "./placement_model.pkl"
joblib.dump(model, model_path)

print(f"✓ Model saved to: {model_path}\n")

# Also save the feature names for later use
features_path = "./feature_columns.pkl"
joblib.dump(FEATURE_COLUMNS, features_path)

print(f"✓ Feature columns saved to: {features_path}\n")

# Save performance metrics
metrics = {
    "accuracy": round(accuracy, 4),
    "f1": round(f1, 4),
    "auc_roc": round(auc_roc, 4),
}

metrics_path = "./model_metrics.pkl"
joblib.dump(metrics, metrics_path)

print(f"✓ Metrics saved to: {metrics_path}\n")

print("=" * 60)
print("  ✓ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext step: Run predict.py to make predictions\n")
