# 🧬 GlucoProfile

> **Clinical patient profiling using unsupervised clustering and supervised classification, served through FastAPI, Streamlit, and orchestrated with Apache Airflow.**

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement and Objectives](#problem-statement-and-objectives)
3. [Main Features](#main-features)
4. [Machine Learning Workflow](#machine-learning-workflow)
5. [Repository Structure](#repository-structure)
6. [Technology Stack](#technology-stack)
7. [Dataset Description](#dataset-description)
8. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
9. [K-Means Clustering Methodology](#k-means-clustering-methodology)
10. [Cluster Interpretation and Risk Categorisation](#cluster-interpretation-and-risk-categorisation)
11. [Classification Methodology](#classification-methodology)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Model Persistence and Loading](#model-persistence-and-loading)
14. [MLflow Experiment Tracking and Model Registry](#mlflow-experiment-tracking-and-model-registry)
15. [FastAPI Application](#fastapi-application)
16. [Streamlit Practitioner Interface](#streamlit-practitioner-interface)
17. [Apache Airflow Orchestration](#apache-airflow-orchestration)
18. [Docker Services and Architecture](#docker-services-and-architecture)
19. [Environment Setup](#environment-setup)
20. [Installation and Execution](#installation-and-execution)
21. [API Usage Examples](#api-usage-examples)
22. [Running Notebooks and Tests](#running-notebooks-and-tests)
23. [Known Limitations and Future Improvements](#known-limitations-and-future-improvements)
24. [Medical Disclaimer](#medical-disclaimer)

---

## Overview

GlucoProfile is a clinical data analysis and machine learning project developed in a simulated biomedical laboratory context.

The system analyses patient clinical measurements to identify patient profiles using **unsupervised K-Means clustering**, then classifies those profiles using a **supervised Logistic Regression pipeline**. The resulting risk assessment is exposed through a **FastAPI REST service** and an interactive **Streamlit practitioner dashboard**, with full **MLflow experiment tracking** and **Apache Airflow** orchestration for training and retraining.

---

## Problem Statement and Objectives

Clinical screening datasets often lack a labelled diabetes outcome suitable for direct supervised learning. GlucoProfile addresses this by:

1. **Discovering** natural patient profiles in unlabelled clinical data using K-Means clustering.
2. **Interpreting** each cluster against established clinical thresholds to assign a risk category (high/low).
3. **Training** a fast, interpretable Logistic Regression classifier on the derived labels for real-time inference.
4. **Serving** predictions through a production-grade REST API and a practitioner-facing dashboard.
5. **Automating** the full retraining pipeline using Apache Airflow, with MLflow experiment tracking and Model Registry integration.

---

## Main Features

- End-to-end ML pipeline from raw CSV to a served prediction endpoint.
- Deterministic data cleaning: zero-as-missing handling, IterativeImputer (MICE), type coercion.
- Feature transformation with `log1p` and `sqrt` for skewed clinical variables.
- K-Means clustering with RobustScaler, cluster relabelling by Glucose mean for determinism.
- Risk-category assignment based on three simultaneous clinical thresholds.
- Logistic Regression classification with 5-fold stratified cross-validation and C-grid search.
- MLflow tracking of all clustering and classification parameters, metrics, and artefacts.
- Automatic Model Registry registration with `production` alias promotion.
- FastAPI endpoints: single-patient prediction, batch prediction (up to 500), health/ready, model info, admin model reload.
- Streamlit interface: single-patient form, CSV batch upload, model information tab.
- Airflow DAG for weekly automated retraining with conditional API reload on promotion.
- Full Docker Compose orchestration of nine services.

---

## Machine Learning Workflow

```
Raw Data (diabetes.csv)
        │
        ▼
┌──────────────────┐
│ 1. Data Cleaning │  clean_column_names → validate_structure → convert_to_numeric
│  (src/cleaning)  │  replace_zeros_with_nan → remove_duplicates → IterativeImputer
│                  │  apply_transformations → detect_outliers
└────────┬─────────┘
         │  diabetes_cleaned.csv / diabetes_transformed.csv
         ▼
┌──────────────────┐
│ 2. Clustering    │  RobustScaler → KMeans(k=2) → relabel_by_glucose
│ (src/clustering) │  compute_risk_profile → add_risk_category → PCA viz → MLflow
└────────┬─────────┘
         │  diabetes_clustered.csv (+ Cluster + risk_category columns)
         ▼
┌──────────────────────┐
│ 3. Classification    │  train/test split (80/20, stratified)
│ (src/classification) │  LogisticRegression(C=10) → 5-fold CV → evaluate
│                      │  save pipeline → MLflow → Model Registry
└────────┬─────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ 4. Serving                                  │
│   FastAPI /predict  ←──  MLflow Registry   │
│   Streamlit dashboard                       │
│   Airflow DAG (weekly retraining)           │
└─────────────────────────────────────────────┘
```

---

## Repository Structure

```
GlucoProfile/
├── api/                             # FastAPI inference service
│   ├── __init__.py
│   ├── main.py                      # Endpoints: /predict, /health, /model/info, /model/reload
│   ├── schemas.py                   # Pydantic request / response models
│   └── utils.py                     # ModelService, RISK_THRESHOLDS, advice builder
│
├── app/                             # Streamlit practitioner dashboard
│   ├── __init__.py
│   └── main.py                      # Single-patient form, batch CSV, model info
│
├── dags/                            # Apache Airflow DAG
│   ├── __init__.py
│   └── main.py                      # gluco_profile_training DAG
│
├── data/
│   ├── raw/
│   │   └── diabetes.csv             # Original Pima Indians dataset
│   └── processed/
│       ├── diabetes_cleaned.csv     # After imputation, before transformation
│       ├── diabetes_transformed.csv # After log1p / sqrt transforms
│       └── diabetes_clustered.csv  # With Cluster and risk_category columns
│
├── docker/
│   ├── airflow/Dockerfile           # Airflow image with ML venv
│   ├── app/Dockerfile               # FastAPI / Streamlit / Python image
│   └── mlflow/Dockerfile            # MLflow tracking server image
│
├── models/
│   ├── kmeans_final.joblib          # Fitted KMeans (k=2)
│   ├── robust_scaler_final.joblib   # Fitted RobustScaler
│   ├── pca_final.joblib             # Fitted PCA (2 components)
│   ├── final_clustering_config.json # Cluster mapping, thresholds, metrics
│   ├── classification_final_pipeline.joblib  # Logistic Regression pipeline
│   └── classification_final_metadata.json    # Metrics, hyperparams, signature
│
├── notebooks/
│   ├── 01_cleaning.ipynb            # EDA and data cleaning walkthrough
│   ├── 02_clustering.ipynb          # Clustering experiments and selection
│   └── 03_classification.ipynb      # Classification experiments and evaluation
│
├── reports/
│   ├── final_cluster_profiles.csv
│   ├── final_risk_profile.csv
│   ├── final_pca_variance.csv
│   ├── clustering_pca_projection.png
│   ├── clustering_cluster_sizes.png
│   ├── classification_confusion_matrix.png
│   ├── classification_final_metrics.csv
│   ├── classification_coefficients.csv
│   ├── classification_test_predictions.csv
│   └── [exploration CSVs: kmeans_curves, dbscan_grid, gmm_covariance_grid, ...]
│
├── requirements/
│   ├── mlflow/requirements.txt      # ML + MLflow layer (shared)
│   └── app/requirements.txt         # FastAPI + Streamlit layer
│
├── src/                             # Reusable Python pipeline modules
│   ├── __init__.py
│   ├── cleaning.py                  # Data cleaning and transformation pipeline
│   ├── clustering.py                # K-Means clustering + MLflow tracking
│   └── classification.py            # Logistic Regression pipeline + MLflow tracking
│
├── docker-compose.yml               # Full nine-service orchestration
├── .env.example                     # Environment variable template (no secrets)
├── .gitignore
├── .dockerignore
├── tasks.csv                        # Jira-compatible task list
└── README.md                        # This file
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Data processing | pandas 2.2, numpy 2.2 |
| Machine learning | scikit-learn 1.6, imbalanced-learn 0.13, scipy 1.15 |
| Serialisation | joblib 1.4 |
| Visualisation | matplotlib 3.10, seaborn 0.13 |
| Experiment tracking | MLflow 2.22 (skinny) |
| REST API | FastAPI 0.115, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit 1.38 |
| Workflow orchestration | Apache Airflow 3.3.1 |
| Backend stores | PostgreSQL 17 (three isolated databases) |
| Database GUI | pgAdmin 4 |
| Containerisation | Docker, Docker Compose |

---

## Dataset Description

The dataset used is the **Pima Indians Diabetes Dataset**, a widely studied benchmark in the biomedical machine learning community.

| Property | Value |
|---|---|
| Records | 768 |
| Features | 8 clinical measurements |
| Source | `data/raw/diabetes.csv` |

### Clinical features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (µU/mL) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes hereditary predisposition score |
| `Age` | Age in years |

> **Note:** The original `Outcome` (diabetes diagnosis label) column is not used. GlucoProfile derives its own cluster-based risk labels.

---

## Data Preprocessing and Feature Engineering

Implemented in `src/cleaning.py` and documented in `notebooks/01_cleaning.ipynb`.

### Steps

1. **Column name normalisation** — strip whitespace, remove `*` characters, drop unnamed index columns.
2. **Structure validation** — verify all eight expected features are present.
3. **Type coercion** — convert all columns to numeric with `errors="coerce"`.
4. **Zero-as-missing replacement** — in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, a zero is physiologically impossible and is replaced by `NaN`.
5. **Validity rule enforcement** — apply per-feature rules (e.g. `Age > 0`) and replace violations with `NaN`.
6. **Duplicate removal** — drop duplicate rows.
7. **MICE imputation** — `IterativeImputer` (random_state=42) with observed min/max bounds.
8. **Dataset validation** — assert no remaining NaN or infinite values, all validity rules satisfied.
9. **Feature transformations** — `log1p` on `Insulin` and `Age`; `sqrt` on `DiabetesPedigreeFunction` to reduce skewness.
10. **Outlier reporting** — IQR-based outlier detection saved to `reports/outlier_report.csv` (outliers are reported, not removed).

---

## K-Means Clustering Methodology

Implemented in `src/clustering.py` and documented in `notebooks/02_clustering.ipynb`.

### Algorithm selection and parameter choice

- Multiple clustering algorithms were benchmarked: K-Means, DBSCAN, GMM, and Agglomerative Clustering.
- Candidate K values were evaluated with inertia (elbow), silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.
- **Final selection:** K-Means with **k=2**.

### Feature set

Clustering uses 5 of the 8 features: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.

### Scaling

RobustScaler (quantile range 25–75) is used to reduce the influence of outliers on distance calculations.

### Deterministic label ordering

Raw K-Means labels are relabelled in ascending order of mean `Glucose` per cluster:
- Cluster 0 = lower mean Glucose (low-risk profile)
- Cluster 1 = higher mean Glucose (high-risk profile)

### Clustering metrics (final run)

| Metric | Value |
|---|---|
| Inertia | 1545.42 |
| Silhouette score | 0.269 |
| Davies-Bouldin index | 1.364 |
| Calinski-Harabasz index | 344.28 |
| PC1 + PC2 explained variance | 72.6 % |
| Cluster balance | 49.7 % / 50.3 % |

---

## Cluster Interpretation and Risk Categorisation

Implemented in `src/clustering.py`.

### Risk thresholds

A cluster is labelled **Risque Élevé** (High Risk) when **all three** of the following conditions hold simultaneously on the cluster's feature means:

| Feature | Threshold |
|---|---|
| `Glucose` | > 126 mg/dL |
| `BMI` | > 30 kg/m² |
| `DiabetesPedigreeFunction` | > 0.5 |

> These thresholds are used to **interpret** cluster means, not to classify individual patients directly.

### Result

| Cluster | Risk Category | Interpretation |
|---|---|---|
| 0 | Risque Faible | Lower metabolic risk profile |
| 1 | Risque Élevé | Higher metabolic risk profile |

The final clustered dataset (`data/processed/diabetes_clustered.csv`) adds the `Cluster` (integer) and `risk_category` (string) columns for use as classification targets.

---

## Classification Methodology

Implemented in `src/classification.py` and documented in `notebooks/03_classification.ipynb`.

### Target

The `Cluster` column (0 or 1) derived from the unsupervised clustering stage.

### Pipeline

```
Input features (8 clinical variables)
        │
  passthrough (no scaling needed)
        │
  LogisticRegression(C=10, max_iter=5000, solver=lbfgs, penalty=l2, random_state=42)
        │
  Cluster prediction (0 or 1)  +  risk_category  +  prob_cluster_1  +  confidence
```

### Training procedure

1. Stratified 80/20 train/test split (`random_state=42`).
2. 5-fold stratified cross-validation on the training set.
3. C parameter grid search (notebooks); C=10 selected without preprocessing.
4. Final pipeline trained on full training set.

---

## Evaluation Metrics

All metrics are reported on the held-out test set (n=154, 20% of 768 records).

### Classification results (verified)

| Metric | Test set | CV mean ± std |
|---|---|---|
| Accuracy | **0.9870** | 0.9886 ± 0.0098 |
| Precision | **0.9747** | 0.9873 ± 0.0156 |
| Recall | **1.0000** | 0.9903 ± 0.0129 |
| F1 score | **0.9872** | 0.9887 ± 0.0097 |
| ROC-AUC | **0.9983** | — |

Confusion matrix (test set):

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 75 (TN) | 2 (FP) |
| **Actual 1** | 0 (FN) | 77 (TP) |

Zero false negatives: no high-risk patient was predicted as low-risk.

---

## Model Persistence and Loading

### Clustering artefacts

| File | Contents |
|---|---|
| `models/kmeans_final.joblib` | Fitted KMeans (k=2, n_init=10, random_state=42) |
| `models/robust_scaler_final.joblib` | Fitted RobustScaler |
| `models/pca_final.joblib` | Fitted PCA (2 components, for visualisation only) |
| `models/final_clustering_config.json` | Cluster mapping, risk thresholds, metrics, library versions |

### Classification artefacts

| File | Contents |
|---|---|
| `models/classification_final_pipeline.joblib` | Fitted sklearn Pipeline (passthrough + LogisticRegression) |
| `models/classification_final_metadata.json` | Hyperparams, test metrics, CV metrics, dataset signature |

### Loading (local)

```python
import joblib
pipeline = joblib.load("models/classification_final_pipeline.joblib")
predictions = pipeline.predict(X)
```

---

## MLflow Experiment Tracking and Model Registry

GlucoProfile logs two MLflow runs inside a single experiment (`GlucoProfile`):

### Clustering run (`clustering_kmeans_robust_k2`)

Logs: K, n_init, random_state, scaler parameters, risk thresholds, cluster-level statistics, silhouette/Davies-Bouldin/Calinski-Harabasz metrics, PCA variance, and dataset SHA-256 signatures. Registers the `ClusteringModel` pyfunc wrapper.

### Classification run (`classification_logreg_c10_noscaling`)

Logs: all hyperparameters, test set metrics, 5-fold CV metrics, confusion matrix image, coefficients CSV, and dataset signature. Registers `GlucoProfileClassifier` in the Model Registry and assigns the `production` alias if F1 ≥ 0.95 and recall = 1.0.

### Model Registry

The FastAPI service loads the production model from the registry at startup:

```
models:/GlucoProfileClassifier/<production-version>
```

Resolution order: stage `Production` → alias `production`.

> The MLflow server is only available when running with Docker Compose. For local development without MLflow, set `use_mlflow=False` when calling `classification_pipeline()`.

---

## FastAPI Application

**Source:** `api/main.py`, `api/schemas.py`, `api/utils.py`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model load status, name, version |
| GET | `/ready` | Returns 200 only if model is loaded; 503 otherwise |
| GET | `/model/info` | Full model metadata from the registry |
| POST | `/predict` | Single-patient risk prediction |
| POST | `/predict/batch` | Batch prediction (1–500 patients) |
| POST | `/model/reload` | Reload model from registry (admin key required) |

### Interactive documentation

Available at `http://localhost:<FASTAPI_PORT>/docs` when the service is running.

---

## API Usage Examples

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 219,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

**Response schema:**

```json
{
  "cluster": 1,
  "risk_category": "Risque Élevé",
  "risk_level": "high",
  "badge_color": "red",
  "risk_probability": 0.9942,
  "confidence": 0.9942,
  "alerts": [
    {
      "feature": "Glucose",
      "value": 148,
      "threshold": 126.0,
      "message": "Glycémie supérieure au seuil diabétique (126 mg/dL)"
    }
  ],
  "advice": ["..."],
  "model_name": "GlucoProfileClassifier",
  "model_version": "1",
  "predicted_at": "2026-10-09T10:00:00Z"
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
       "Insulin": 219, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50},
      {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
       "Insulin": 70, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}
    ]
  }'
```

---

## Streamlit Practitioner Interface

**Source:** `app/main.py`

The Streamlit dashboard provides three tabs:

| Tab | Feature |
|---|---|
| 👤 Patient | Single-patient clinical form → risk badge, probability bar, threshold alerts, follow-up advice |
| 📄 Fichier de patients | CSV upload → batch scoring → bar chart, results table, downloadable CSV |
| ⚙️ Modèle | Production model metadata, /ready check, admin model reload |

Available at `http://localhost:<STREAMLIT_PORT>` when the service is running.

---

## Apache Airflow Orchestration

**Source:** `dags/main.py`

The DAG `gluco_profile_training` runs **weekly** (configurable) and orchestrates the full retraining pipeline:

```
extract_data → run_cleaning → run_clustering → run_classification → reload_api
```

| Task | Description |
|---|---|
| `extract_data` | Validates the raw CSV and returns metadata |
| `run_cleaning` | Runs `cleaning_pipeline()`: imputation, transformation, outlier report |
| `run_clustering` | Runs `clustering_pipeline()`: K-Means + MLflow logging |
| `run_classification` | Runs `classification_pipeline()`: training, evaluation, MLflow + registry |
| `reload_api` | Calls `POST /model/reload` if the new version was promoted to production |

All ML tasks use `@task.external_python` with the isolated ML venv (`/home/airflow/ml-venv`) to avoid Airflow dependency conflicts.

The `reload_api` task raises `AirflowSkipException` if the new model version was not promoted, preserving the existing production model.

---

## Docker Services and Architecture

```
docker-compose.yml
│
├── postgres          (postgres:17-alpine)         — Application DB
├── airflow-postgres  (postgres:17-alpine)         — Airflow metadata DB
├── mlflow-postgres   (postgres:17-alpine)         — MLflow backend store
├── pgadmin           (dpage/pgadmin4)             — Database GUI
├── mlflow            (gluco-profile-mlflow)       — MLflow tracking server
├── python            (gluco-profile-app)          — Ad-hoc Python environment
├── fastapi           (gluco-profile-app)          — REST inference API
├── streamlit         (gluco-profile-app)          — Practitioner dashboard
├── airflow-init      (gluco-profile-airflow)      — DB migration + init
├── airflow-apiserver (gluco-profile-airflow)      — Airflow web UI
├── airflow-scheduler (gluco-profile-airflow)      — DAG scheduler
├── airflow-dag-processor (gluco-profile-airflow)  — DAG file parser
└── airflow-triggerer (gluco-profile-airflow)      — Deferred task triggerer
```

### Default service ports

| Service | Default port |
|---|---|
| FastAPI | 8000 |
| Streamlit | 8501 |
| MLflow | 5000 |
| Airflow | 8080 |
| pgAdmin | 5050 |
| App PostgreSQL | 5432 |
| Airflow PostgreSQL | 5433 |
| MLflow PostgreSQL | 5434 |

---

## Environment Setup

### Prerequisites

- Docker Engine ≥ 24 and Docker Compose V2
- Python 3.13 (for local development without Docker)

### Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your database passwords, API keys, and Fernet key
```

> Generate the Airflow Fernet key:
> ```bash
> python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
> ```

---

## Installation and Execution

### With Docker Compose (recommended)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your values

# 2. Build and start all services
docker compose up --build -d

# 3. Run the full training pipeline (first time)
docker compose exec python python -m src.cleaning
docker compose exec python python -m src.clustering
docker compose exec python python -m src.classification

# 4. Services are available at:
# FastAPI docs:    http://localhost:8000/docs
# Streamlit:       http://localhost:8501
# MLflow:          http://localhost:5000
# Airflow:         http://localhost:8080
# pgAdmin:         http://localhost:5050
```

### Local development (without Docker)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements/app/requirements.txt

# Run the cleaning pipeline
python -m src.cleaning

# Run the clustering pipeline (set MLFLOW_TRACKING_URI or disable MLflow)
python -m src.clustering

# Run the classification pipeline
python -m src.classification

# Start the FastAPI service
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the Streamlit dashboard (separate terminal)
API_URL=http://localhost:8000 streamlit run app/main.py
```

---

## Running Notebooks and Tests

### Notebooks

```bash
# Inside the virtual environment or Python container:
jupyter lab notebooks/
```

Run notebooks in order: `01_cleaning.ipynb` → `02_clustering.ipynb` → `03_classification.ipynb`.

### Syntax and import validation

```bash
# Validate Python source modules
python -m py_compile src/cleaning.py src/clustering.py src/classification.py
python -m py_compile api/main.py api/schemas.py api/utils.py
python -m py_compile app/main.py dags/main.py
```

### Docker Compose configuration validation

```bash
docker compose config --quiet && echo "Compose config OK"
```

---

## Known Limitations and Future Improvements

| Area | Limitation | Potential Improvement |
|---|---|---|
| Dataset | 768 records; Pima women only — limited diversity | Evaluate on larger, more diverse clinical datasets |
| Clustering | k=2 binary split; silhouette score 0.269 indicates overlapping clusters | Explore soft clustering (GMM), alternative feature subsets, or additional data |
| Risk thresholds | Thresholds applied to cluster means, not individual patients | Per-patient threshold comparison at inference time |
| API auth | Single shared admin key | Role-based access control, OAuth2 |
| Retraining | Weekly schedule; no automated data ingestion | Connect to a live data source with schema validation |
| Monitoring | No prediction drift detection | Integrate Evidently or a similar monitoring framework |
| Tests | No automated test suite | Add unit tests for cleaning, clustering, and API endpoints |

---

## Medical Disclaimer

> **This tool is a decision-support aid and does not replace a medical diagnosis.**
>
> GlucoProfile was developed for educational and portfolio purposes in a simulated biomedical laboratory context. The risk categories produced by the system are derived from unsupervised clustering of a limited benchmark dataset and have not been validated for clinical use.
>
> - Do not use this system to make or communicate clinical decisions to patients.
> - Always refer patients to a qualified healthcare professional for diagnosis and treatment.
> - Risk labels such as "Risque Élevé" are relative statistical groupings, not confirmed diagnoses.
> - Any clinical interpretation of the output must be performed by a licensed medical practitioner.

---

*Released for educational and portfolio purposes.*
