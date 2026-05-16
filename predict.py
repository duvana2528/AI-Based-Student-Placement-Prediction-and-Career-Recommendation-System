"""
================================================================
   MAKE PREDICTIONS & EXPLAIN RESULTS
   
   This script uses the trained model to make predictions
   for students and explain the results using SHAP.
   
   RUN THIS AFTER train_model.py:
   python predict.py
================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Skipping explanation plots.")

print("=" * 60)
print("  LOAD TRAINED MODEL")
print("=" * 60)

try:
    model = joblib.load("./placement_model.pkl")
    feature_cols = joblib.load("./feature_columns.pkl")
    metrics = joblib.load("./model_metrics.pkl")
    print("✓ Model loaded successfully\n")
except FileNotFoundError:
    print("ERROR: Model files not found!")
    print("Please run train_model.py first\n")
    exit()


print("=" * 60)
print("  SAMPLE STUDENT - PREDICTION")
print("=" * 60)

# Create a sample student profile
# You can change these values
sample_student = {
    "CGPA": 7.5,
    "Internships": 1,
    "Projects": 2,
    "Workshops/Certifications": 1,
    "AptitudeTestScore": 75,
    "SoftSkillsRating": 3.5,
    "ExtracurricularActivities": 1,  # 1 = Yes, 0 = No
    "SSC_Marks": 72,
    "HSC_Marks": 70,
}

# Convert to DataFrame (required for model)
student_df = pd.DataFrame([sample_student])

# Make prediction
predicted_class = model.predict(student_df)[0]  # 0 or 1
predicted_prob = model.predict_proba(student_df)[0]

print(f"\nStudent Profile:")
for key, value in sample_student.items():
    print(f"  {key:<30}: {value}")

print("\n" + "-" * 60)

# Probability of being Placed (class 1)
placed_prob = predicted_prob[1] * 100

print(f"\nPrediction Results:")
print(f"  Probability of Placement : {placed_prob:.1f}%")

if placed_prob >= 80:
    verdict = "✓ Strong Placement Likely"
elif placed_prob >= 65:
    verdict = "✓ Placement Likely"
elif placed_prob >= 50:
    verdict = "⚠ Could Go Either Way"
elif placed_prob >= 35:
    verdict = "⚠ Placement at Risk"
else:
    verdict = "✗ Unlikely to be Placed"

print(f"  Verdict                  : {verdict}")
print(f"\nModel Performance (from training):")
print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
print(f"  F1 Score : {metrics['f1']*100:.2f}%")
print(f"  AUC-ROC  : {metrics['auc_roc']*100:.2f}%")

print("\n" + "-" * 60)


# ================================================================
#  EXPLANATION USING SHAP (if available)
# ================================================================

if SHAP_AVAILABLE:
    print("\n[EXPLANATION] Using SHAP to understand the prediction")
    print("-" * 60)
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Get SHAP values for this student
        shap_values = explainer.shap_values(student_df)
        
        # For XGBoost with binary classification, shap_values is a list
        # We want the positive class (Placed = 1)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Positive class
        else:
            shap_vals = shap_values
        
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        # Create explanation dataframe
        explanation = pd.DataFrame({
            "Feature": feature_cols,
            "Your Value": student_df.iloc[0].values,
            "SHAP Value": shap_vals[0] if shap_vals.ndim > 1 else shap_vals,
        })
        
        explanation["Abs SHAP"] = np.abs(explanation["SHAP Value"])
        explanation = explanation.sort_values("Abs SHAP", ascending=False)
        
        print(f"\nBase Probability (average student): {base_value*100:.1f}%")
        print(f"Your Probability: {placed_prob:.1f}%")
        print(f"Difference: {(placed_prob - base_value*100):+.1f}pp\n")
        
        print("Feature Contributions (SHAP Values):")
        print(f"{'Feature':<30} {'Your Value':>12} {'SHAP Value':>12} {'Effect'}")
        print("-" * 70)
        
        for idx, row in explanation.iterrows():
            direction = "↑ Positive" if row["SHAP Value"] > 0 else "↓ Negative"
            print(f"{row['Feature']:<30} {str(row['Your Value']):>12} {row['SHAP Value']:>+12.4f} {direction}")
        
        # Try to create visualization
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by absolute value for better visualization
            plot_data = explanation.sort_values("SHAP Value")
            colors = ["#d62728" if x < 0 else "#2ca02c" for x in plot_data["SHAP Value"]]
            
            ax.barh(plot_data["Feature"], plot_data["SHAP Value"], color=colors, edgecolor="white")
            ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
            ax.set_title("Feature Contributions to Placement Prediction\n(Why this prediction was made)", 
                        fontweight="bold", fontsize=12)
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
            ax.grid(axis="x", alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("shap_explanation.png", dpi=150, bbox_inches="tight")
            print("\n✓ SHAP visualization saved as: shap_explanation.png")
            plt.close()
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Skipping SHAP analysis\n")


# ================================================================
#  RECOMMENDATIONS - How to improve placement chances
# ================================================================

print("\n" + "=" * 60)
print("  IMPROVEMENT RECOMMENDATIONS")
print("=" * 60)

# Features that can be improved
mutable_features = {
    "CGPA": {"min": 6.5, "max": 9.1, "step": 0.2, "action": "Focus on academics"},
    "Internships": {"min": 0, "max": 2, "step": 1, "action": "Complete internships (Internshala, LinkedIn)"},
    "Projects": {"min": 0, "max": 3, "step": 1, "action": "Build projects and publish on GitHub"},
    "Workshops/Certifications": {"min": 0, "max": 3, "step": 1, "action": "Earn certifications (Coursera, AWS)"},
    "AptitudeTestScore": {"min": 60, "max": 90, "step": 5, "action": "Practice daily (IndiaBix, PrepInsta)"},
    "SoftSkillsRating": {"min": 3.0, "max": 4.8, "step": 0.3, "action": "Join mock GD/PI sessions"},
    "ExtracurricularActivities": {"min": 0, "max": 1, "step": 1, "action": "Join clubs or volunteer"},
}

recommendations = []

for feature, config in mutable_features.items():
    current = sample_student.get(feature)
    
    if current is None:
        continue
    
    # Can we improve this feature?
    if current >= config["max"]:
        continue
    
    # Calculate new probability if we improve this feature
    improved_student = sample_student.copy()
    improved_student[feature] = min(config["max"], current + config["step"])
    
    improved_prob = model.predict_proba(pd.DataFrame([improved_student]))[0][1]
    gain = (improved_prob - predicted_prob[1]) * 100
    
    if gain >= 0.5:  # Only include if meaningful improvement
        recommendations.append({
            "Feature": feature,
            "Current": current,
            "Target": improved_student[feature],
            "Gain": gain,
            "Action": config["action"],
        })

# Sort by biggest gain
recommendations.sort(key=lambda x: -x["Gain"])

if recommendations:
    print(f"\nCurrent Placement Probability: {placed_prob:.1f}%\n")
    print("Top 5 Areas to Improve (ranked by impact):\n")
    
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec['Feature']}")
        print(f"   Current  : {rec['Current']}")
        print(f"   Target   : {rec['Target']:.2f}" if isinstance(rec['Target'], float) else f"   Target   : {rec['Target']}")
        print(f"   Gain     : +{rec['Gain']:.2f}pp")
        print(f"   Action   : {rec['Action']}")
        print()
else:
    print("\nYour profile is already very strong!")
    print("You're well-positioned for placement.\n")


print("=" * 60)
print("  ✓ PREDICTION COMPLETE")
print("=" * 60)
