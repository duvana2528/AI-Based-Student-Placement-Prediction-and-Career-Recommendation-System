"""
================================================================
   DATA ANALYSIS & VISUALIZATION (EDA)
   
   This script analyzes the placement dataset and creates
   visualizations to understand the data.
   
   RUN THIS ANYTIME:
   python data_analysis.py
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

print("=" * 60)
print("  LOAD AND CLEAN DATA")
print("=" * 60)

# Load dataset
df = pd.read_csv("./placementdata.csv")

print(f"✓ Loaded {df.shape[0]} students with {df.shape[1]} features\n")

# Clean data
df["ExtracurricularActivities"] = (df["ExtracurricularActivities"] == "Yes").astype(int)
df["PlacementStatus_num"] = (df["PlacementStatus"] == "Placed").astype(int)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

placed = df["PlacementStatus_num"].sum()
not_placed = len(df) - placed

print(f"Dataset Statistics:")
print(f"  Total Students  : {len(df)}")
print(f"  Placed          : {placed} ({placed/len(df)*100:.1f}%)")
print(f"  Not Placed      : {not_placed} ({not_placed/len(df)*100:.1f}%)\n")


print("=" * 60)
print("  DESCRIPTIVE STATISTICS")
print("=" * 60)

# Get numeric columns
numeric_cols = ["CGPA", "Internships", "Projects", "Workshops/Certifications",
                "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]

print("\nDataset Summary Statistics:\n")
print(df[numeric_cols].describe().round(2))


print("\n" + "=" * 60)
print("  CREATE VISUALIZATIONS")
print("=" * 60)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 14)

# Create a big figure with multiple subplots
fig = plt.figure(figsize=(16, 14))

colors = ["#E74C3C", "#27AE60"]  # Red = Not Placed, Green = Placed

print("\nCreating visualization 1/9: Placement Distribution...")
ax1 = plt.subplot(3, 3, 1)
ax1.pie([placed, not_placed], labels=["Placed", "Not Placed"],
        autopct="%1.1f%%", colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax1.set_title("Overall Placement Distribution", fontweight="bold", fontsize=11)

print("Creating visualization 2/9: CGPA Distribution...")
ax2 = plt.subplot(3, 3, 2)
df[df["PlacementStatus_num"] == 1]["CGPA"].plot.hist(
    ax=ax2, bins=15, color="#27AE60", alpha=0.7, label="Placed", edgecolor="black"
)
df[df["PlacementStatus_num"] == 0]["CGPA"].plot.hist(
    ax=ax2, bins=15, color="#E74C3C", alpha=0.7, label="Not Placed", edgecolor="black"
)
ax2.set_title("CGPA Distribution", fontweight="bold", fontsize=11)
ax2.set_xlabel("CGPA", fontsize=10)
ax2.set_ylabel("Number of Students", fontsize=10)
ax2.legend()

print("Creating visualization 3/9: Aptitude Score...")
ax3 = plt.subplot(3, 3, 3)
df[df["PlacementStatus_num"] == 1]["AptitudeTestScore"].plot.hist(
    ax=ax3, bins=15, color="#27AE60", alpha=0.7, label="Placed", edgecolor="black"
)
df[df["PlacementStatus_num"] == 0]["AptitudeTestScore"].plot.hist(
    ax=ax3, bins=15, color="#E74C3C", alpha=0.7, label="Not Placed", edgecolor="black"
)
ax3.set_title("Aptitude Test Score Distribution", fontweight="bold", fontsize=11)
ax3.set_xlabel("Score", fontsize=10)
ax3.set_ylabel("Number of Students", fontsize=10)
ax3.legend()

print("Creating visualization 4/9: Internships vs Placement...")
ax4 = plt.subplot(3, 3, 4)
internship_data = pd.crosstab(df["Internships"], df["PlacementStatus"])
internship_data.plot(kind="bar", ax=ax4, color=["#E74C3C", "#27AE60"], edgecolor="white")
ax4.set_title("Internships vs Placement", fontweight="bold", fontsize=11)
ax4.set_xlabel("Number of Internships", fontsize=10)
ax4.set_ylabel("Student Count", fontsize=10)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
ax4.legend(["Not Placed", "Placed"], fontsize=9)

print("Creating visualization 5/9: Projects vs Placement...")
ax5 = plt.subplot(3, 3, 5)
project_data = pd.crosstab(df["Projects"], df["PlacementStatus"])
project_data.plot(kind="bar", ax=ax5, color=["#E74C3C", "#27AE60"], edgecolor="white")
ax5.set_title("Projects vs Placement", fontweight="bold", fontsize=11)
ax5.set_xlabel("Number of Projects", fontsize=10)
ax5.set_ylabel("Student Count", fontsize=10)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
ax5.legend(["Not Placed", "Placed"], fontsize=9)

print("Creating visualization 6/9: Soft Skills Rating...")
ax6 = plt.subplot(3, 3, 6)
placed_skills = df[df["PlacementStatus_num"] == 1]["SoftSkillsRating"]
not_placed_skills = df[df["PlacementStatus_num"] == 0]["SoftSkillsRating"]
bp = ax6.boxplot(
    [not_placed_skills, placed_skills],
    labels=["Not Placed", "Placed"],
    patch_artist=True,
)
for patch, color in zip(bp["boxes"], [colors[0], colors[1]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_title("Soft Skills Rating by Placement", fontweight="bold", fontsize=11)
ax6.set_ylabel("Soft Skills Rating", fontsize=10)
ax6.grid(axis="y", alpha=0.3)

print("Creating visualization 7/9: Extracurricular Activities...")
ax7 = plt.subplot(3, 3, 7)
extra_data = pd.crosstab(df["ExtracurricularActivities"], df["PlacementStatus"])
extra_data.index = ["No Activities", "Has Activities"]
extra_data.plot(kind="bar", ax=ax7, color=["#E74C3C", "#27AE60"], edgecolor="white")
ax7.set_title("Extracurricular Activities vs Placement", fontweight="bold", fontsize=11)
ax7.set_xlabel("", fontsize=10)
ax7.set_ylabel("Student Count", fontsize=10)
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=15)
ax7.legend(["Not Placed", "Placed"], fontsize=9)

print("Creating visualization 8/9: SSC Marks...")
ax8 = plt.subplot(3, 3, 8)
df[df["PlacementStatus_num"] == 1]["SSC_Marks"].plot.hist(
    ax=ax8, bins=15, color="#27AE60", alpha=0.7, label="Placed", edgecolor="black"
)
df[df["PlacementStatus_num"] == 0]["SSC_Marks"].plot.hist(
    ax=ax8, bins=15, color="#E74C3C", alpha=0.7, label="Not Placed", edgecolor="black"
)
ax8.set_title("SSC Marks (10th Standard) Distribution", fontweight="bold", fontsize=11)
ax8.set_xlabel("Marks", fontsize=10)
ax8.set_ylabel("Number of Students", fontsize=10)
ax8.legend()

print("Creating visualization 9/9: Correlation Heatmap...")
ax9 = plt.subplot(3, 3, 9)
all_numeric = numeric_cols + ["PlacementStatus_num"]
corr_matrix = df[all_numeric].corr()
sns.heatmap(
    corr_matrix,
    ax=ax9,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 7},
)
ax9.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=11)
ax9.tick_params(axis="x", rotation=45, labelsize=8)
ax9.tick_params(axis="y", rotation=0, labelsize=8)

# Overall title
fig.suptitle("Student Placement Data Analysis (EDA)", fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()

# Save the figure
plt.savefig("eda_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: eda_analysis.png\n")
plt.close()


# ================================================================
#  CONFUSION MATRIX (Train a quick model to show evaluation)
# ================================================================

print("=" * 60)
print("  TRAIN MODEL & CREATE CONFUSION MATRIX")
print("=" * 60)

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

X = df[FEATURE_COLUMNS].astype(float)
y = df["PlacementStatus_num"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")

# Train quick model
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nModel Performance on Test Set:")
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print(f"  AUC-ROC  : {auc*100:.2f}%\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix Values:")
print(f"  True Negatives  (TN) : {tn}  → Correctly predicted Not Placed")
print(f"  True Positives  (TP) : {tp}  → Correctly predicted Placed")
print(f"  False Positives (FP) : {fp}  → Wrongly predicted Placed")
print(f"  False Negatives (FN) : {fn}  → Wrongly predicted Not Placed\n")

# Visualize Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Absolute values
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
disp1.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(
    f"Confusion Matrix (Counts)\nTN={tn} | FP={fp}\nFN={fn} | TP={tp}",
    fontweight="bold"
)

# Normalized (percentages)
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Not Placed", "Placed"])
disp2.plot(ax=axes[1], colorbar=False, cmap="Greens")
axes[1].set_title("Confusion Matrix (Normalized %)", fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("✓ Saved: confusion_matrix.png\n")
plt.close()


# ================================================================
#  FEATURE IMPORTANCE
# ================================================================

print("Creating Feature Importance chart...")

importance_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2980B9", edgecolor="white")
ax.set_title("Feature Importance\n(Which factors matter most for placement?)", 
             fontweight="bold", fontsize=12)
ax.set_xlabel("Importance Score", fontsize=11)

# Add score labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("✓ Saved: feature_importance.png\n")
plt.close()


print("=" * 60)
print("  ✓ DATA ANALYSIS COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  1. eda_analysis.png - Comprehensive data exploration")
print("  2. confusion_matrix.png - Model evaluation")
print("  3. feature_importance.png - Most important features\n")
