import os
from datetime import datetime, timedelta

import requests
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task

ML_PYTHON = "/home/airflow/ml-venv/bin/python"

API_URL = os.getenv("API_URL", "http://fastapi:8000")
API_ADMIN_KEY = os.getenv("API_ADMIN_KEY")

default_args = {
    "owner": "gluco-profile",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="gluco_profile_training",
    description="Nettoyage, clustering, classification et mise à jour du Model Registry",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["gluco-profile", "training", "mlflow"],
)
def gluco_profile_training():

    @task.external_python(python=ML_PYTHON)
    def extract_data():
        from src.cleaning import RAW_DATA_PATH, load_raw_data

        df = load_raw_data()
        return {
            "path": str(RAW_DATA_PATH),
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
        }

    @task.external_python(python=ML_PYTHON)
    def run_cleaning(extract_info: dict):
        from src.cleaning import cleaning_pipeline

        df_clean, df_transformed, report = cleaning_pipeline()
        return {
            "raw_rows": extract_info["n_rows"],
            "cleaned_shape": [int(df_clean.shape[0]), int(df_clean.shape[1])],
            "transformed_shape": [int(df_transformed.shape[0]), int(df_transformed.shape[1])],
            "outliers_per_feature": {
                row["feature"]: int(row["outlier_count"])
                for _, row in report.iterrows()
            },
        }

    @task.external_python(python=ML_PYTHON)
    def run_clustering(cleaning_info: dict):
        from src.clustering import clustering_pipeline

        df_clustered, summary = clustering_pipeline()
        return {
            "run_id": summary["run_id"],
            "metrics": {k: float(v) for k, v in summary["metrics"].items()},
            "risk_counts": {
                str(k): int(v)
                for k, v in df_clustered["risk_category"].value_counts().items()
            },
        }

    @task.external_python(python=ML_PYTHON)
    def run_classification(clustering_info: dict):
        from src.classification import classification_pipeline

        _, summary = classification_pipeline()
        metrics = summary["metrics"]
        return {
            "run_id": summary["run_id"],
            "metrics": {
                name: float(metrics[name])
                for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
            },
            "registry": summary["registry"],
        }
        
    @task
    def reload_api(classification_info: dict):
        registry = classification_info.get("registry") or {}
        if not registry.get("promoted"):
            raise AirflowSkipException(
                "Nouvelle version non promue en Production : l'API n'est pas rechargée."
            )

        headers = {"X-API-Key": API_ADMIN_KEY} if API_ADMIN_KEY else {}
        response = requests.post(f"{API_URL}/model/reload", headers=headers, timeout=60)
        response.raise_for_status()
        info = response.json()
        return {"model_name": info["model_name"], "model_version": info["model_version"]}

    extracted = extract_data()
    cleaned = run_cleaning(extracted)
    clustered = run_clustering(cleaned)
    classified = run_classification(clustered)
    reload_api(classified)


gluco_profile_training()