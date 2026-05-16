# PlaceIQ — Student Placement Prediction System

A machine learning web app that predicts student placement probability using an XGBoost model, with SHAP-powered explainability and actionable improvement recommendations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![AUC--ROC](https://img.shields.io/badge/AUC--ROC-85.6%25-brightgreen)

---

## Features

- **Placement Prediction** — Probability score with verdict (e.g. "Strong Placement Likely")
- **SHAP Explainability** — Per-feature contribution breakdown using TreeSHAP
- **Recommendations** — Greedy counterfactual suggestions ranked by probability gain
- **Frontend UI** — Clean HTML/JS interface, no framework required
- **REST API** — FastAPI backend with auto-generated Swagger docs

---

## Model Performance (5-Fold CV)

| Metric   | Score  |
|----------|--------|
| Accuracy | 77.9%  |
| F1 Score | 73.9%  |
| AUC-ROC  | 85.6%  |

---

## Project Structure

```
placement-predictor/
├── main.py                  # FastAPI app (predict, explain, recommend endpoints)
├── train_model.py           # XGBoost training script
├── requirements.txt         # Python dependencies
├── index.html               # Frontend UI
├── data/
│   └── placementdata.csv    # Training dataset
└── model_artifacts/
    └── meta.json            # Feature config + CV metrics
```

> **Note:** `placement_model.pkl` is excluded from git (binary file). Run `train_model.py` to regenerate it locally.

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/placement-predictor.git
cd placement-predictor
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
# or specify paths:
python train_model.py --data_path ./data/placementdata.csv --output_dir ./model_artifacts
```

This generates `model_artifacts/placement_model.pkl`.

### 3. Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the UI

Open `index.html` in your browser, or visit `http://localhost:8000/docs` for the interactive Swagger API docs.

---

## API Endpoints

| Method | Endpoint     | Description                              |
|--------|--------------|------------------------------------------|
| GET    | `/`          | Health check + CV metrics                |
| GET    | `/meta`      | Feature config and metadata              |
| POST   | `/predict`   | Placement probability + verdict          |
| POST   | `/explain`   | SHAP feature-level explanation           |
| POST   | `/recommend` | Top 5 improvement recommendations        |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CGPA": 7.5,
    "Internships": 1,
    "Projects": 2,
    "Workshops_Certifications": 1,
    "AptitudeTestScore": 75,
    "SoftSkillsRating": 3.5,
    "ExtracurricularActivities": "Yes",
    "SSC_Marks": 72,
    "HSC_Marks": 70
  }'
```

### Example Response

```json
{
  "placement_probability": 0.7812,
  "placement_probability_pct": 78.1,
  "placed": true,
  "verdict": "Placement Likely"
}
```

---

## Input Features

| Feature                    | Type    | Range       |
|----------------------------|---------|-------------|
| CGPA                       | float   | 6.5 – 9.1   |
| Internships                | int     | 0 – 2       |
| Projects                   | int     | 0 – 3       |
| Workshops / Certifications | int     | 0 – 3       |
| Aptitude Test Score        | int     | 60 – 90     |
| Soft Skills Rating         | float   | 3.0 – 4.8   |
| Extracurricular Activities | Yes/No  | —           |
| SSC Marks (10th)           | int     | 55 – 90     |
| HSC Marks (12th)           | int     | 57 – 88     |

---

## Tech Stack

- **ML:** XGBoost, Scikit-learn, SHAP
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Data:** Pandas, NumPy
- **Frontend:** Vanilla HTML/CSS/JS

---

## License

MIT
