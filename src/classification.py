import hashlib
import json
import logging
import os
import platform
from importlib import metadata
from pathlib import Path
import dotenv

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("classification")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

dotenv.load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLUSTERED_DATA_PATH = PROCESSED_DIR / "diabetes_clustered.csv"

FINAL_PIPELINE_PATH = MODELS_DIR / "classification_final_pipeline.joblib"
FINAL_METADATA_PATH = MODELS_DIR / "classification_final_metadata.json"
FINAL_METRICS_REPORT_PATH = REPORTS_DIR / "classification_final_metrics.csv"
FINAL_COEFFICIENTS_REPORT_PATH = REPORTS_DIR / "classification_coefficients.csv"
FINAL_PREDICTIONS_REPORT_PATH = REPORTS_DIR / "classification_test_predictions.csv"
CONFUSION_MATRIX_PLOT_PATH = REPORTS_DIR / "classification_confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET = "Cluster"
RISK_COLUMN = "risk_category"
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Cluster 0", "Cluster 1"]

FINAL_MODEL_NAME = "LogisticRegression"
FINAL_SCALER_NAME = "NoScaling"
FINAL_SAMPLER_NAME = "None"
FINAL_C = 10.0
FINAL_MAX_ITER = 5000

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "GlucoProfile")
MLFLOW_RUN_NAME = "classification_logreg_c10_noscaling"
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "GlucoProfileClassifier")
MLFLOW_PRODUCTION_ALIAS = os.getenv("MLFLOW_PRODUCTION_ALIAS", "production")
DEFAULT_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = "model"

MIN_F1_FOR_PRODUCTION = 0.95
MIN_RECALL_FOR_PRODUCTION = 0.95

LIBRARIES_TO_LOG = [
    "numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "mlflow",
]


def load_data(path: Path = CLUSTERED_DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    logger.info("Données chargées : %s", df.shape)
    return df


def validate_data(df: pd.DataFrame) -> None:
    expected_columns = FEATURES + [TARGET, RISK_COLUMN]
    assert list(df.columns) == expected_columns, "Colonnes inattendues"
    assert df.isna().sum().sum() == 0, "Valeurs manquantes"
    assert np.isfinite(df[FEATURES].to_numpy()).all(), "Valeurs infinies"
    clusters = sorted(df[TARGET].astype(int).unique().tolist())
    assert clusters == CLASS_LABELS, f"Labels Cluster inattendus : {clusters}"
    logger.info("Validation des données OK")


def compute_dataset_signature(df: pd.DataFrame) -> dict:
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    content_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    schema_hash = hashlib.sha256(
        json.dumps({c: str(t) for c, t in df.dtypes.items()}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "content_sha256": content_hash,
        "schema_sha256": schema_hash,
    }


def build_risk_mapping(df: pd.DataFrame) -> dict[int, str]:
    pairs = df[[TARGET, RISK_COLUMN]].drop_duplicates().sort_values(TARGET)
    assert pairs[TARGET].is_unique, "Un cluster correspond à plusieurs risk_category"
    mapping = {int(c): str(r) for c, r in zip(pairs[TARGET], pairs[RISK_COLUMN])}
    for cluster, category in mapping.items():
        logger.info("Cluster %d -> %s", cluster, category)
    return mapping


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].astype("float64").copy()
    y = df[TARGET].astype(int).copy()
    assert TARGET not in X.columns
    assert RISK_COLUMN not in X.columns
    assert len(X) == len(y)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    assert len(np.intersect1d(X_train.index, X_test.index)) == 0
    assert len(X_train) + len(X_test) == len(X)
    logger.info("Split : train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def build_final_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("scaler", "passthrough"),
        ("classifier", LogisticRegression(
            C=FINAL_C,
            max_iter=FINAL_MAX_ITER,
            random_state=RANDOM_STATE,
        )),
    ])


def train_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    assert not X_train.isna().any().any()
    assert np.isfinite(X_train.to_numpy()).all()
    pipeline.fit(X_train, y_train)
    classifier = pipeline.named_steps["classifier"]
    assert list(classifier.classes_) == CLASS_LABELS
    logger.info("Pipeline entraîné : %s, C=%.1f, %d observations",
                FINAL_MODEL_NAME, FINAL_C, len(X_train))
    return pipeline


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=1,
    )
    cv_metrics = {}
    for name in ("accuracy", "precision", "recall", "f1"):
        values = scores[f"test_{name}"]
        cv_metrics[f"cv_{name}_mean"] = float(values.mean())
        cv_metrics[f"cv_{name}_std"] = float(values.std())
    logger.info("CV F1 = %.4f ± %.4f", cv_metrics["cv_f1_mean"], cv_metrics["cv_f1_std"])
    return cv_metrics


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    logger.info(
        "Test : accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | roc_auc=%.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"],
    )
    report = classification_report(
        y_test, y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    logger.info("Classification report :\n%s", report)
    return metrics, cm, y_pred, y_proba


def build_predictions_report(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    risk_mapping: dict[int, str],
) -> pd.DataFrame:
    report = X_test.copy()
    report["true_cluster"] = y_test.to_numpy()
    report["predicted_cluster"] = y_pred
    report["true_risk_category"] = report["true_cluster"].map(risk_mapping)
    report["predicted_risk_category"] = report["predicted_cluster"].map(risk_mapping)
    report["prob_cluster_1"] = y_proba
    report["confidence"] = np.maximum(y_proba, 1 - y_proba)
    report["correct"] = report["true_cluster"] == report["predicted_cluster"]
    return report


def compute_coefficients(pipeline: Pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    train_std = X_train[FEATURES].std(ddof=0).to_numpy()
    table = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": coefficients,
        "train_std": train_std,
        "standardized_coefficient": coefficients * train_std,
        "odds_ratio_per_unit": np.exp(coefficients),
    })
    table["abs_standardized_coefficient"] = table["standardized_coefficient"].abs()
    table["direction"] = np.where(table["coefficient"] > 0, "Vers Cluster 1", "Vers Cluster 0")
    table = table.sort_values("abs_standardized_coefficient", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path = CONFUSION_MATRIX_PLOT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=CLASS_LABELS,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Matrice de confusion — modèle final")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def get_library_versions(libraries: list[str] = LIBRARIES_TO_LOG) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for lib in libraries:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = "not-installed"
    return versions


def build_metadata(
    pipeline: Pipeline,
    metrics: dict[str, float],
    cv_metrics: dict[str, float],
    risk_mapping: dict[int, str],
    dataset_signature: dict,
    n_train: int,
    n_test: int,
) -> dict:
    classifier = pipeline.named_steps["classifier"]
    return {
        "model": {
            "name": FINAL_MODEL_NAME,
            "scaler": FINAL_SCALER_NAME,
            "sampler": FINAL_SAMPLER_NAME,
            "hyperparameters": {
                "C": float(classifier.C),
                "max_iter": int(classifier.max_iter),
                "penalty": str(classifier.penalty),
                "solver": str(classifier.solver),
                "random_state": int(RANDOM_STATE),
            },
        },
        "features": FEATURES,
        "target": TARGET,
        "classes": CLASS_LABELS,
        "cluster_mapping": {str(k): v for k, v in risk_mapping.items()},
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(n_train),
            "n_test": int(n_test),
        },
        "test_metrics": metrics,
        "cv_metrics": cv_metrics,
        "dataset_signature": dataset_signature,
        "library_versions": get_library_versions(),
    }


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_artifacts(
    pipeline: Pipeline,
    config: dict,
    metrics: dict[str, float],
    cv_metrics: dict[str, float],
    coefficients: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, FINAL_PIPELINE_PATH)

    with FINAL_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    metrics_table = pd.DataFrame(
        {"metric": list({**metrics, **cv_metrics}.keys()),
         "value": list({**metrics, **cv_metrics}.values())}
    )
    save_dataframe(metrics_table, FINAL_METRICS_REPORT_PATH)
    save_dataframe(coefficients, FINAL_COEFFICIENTS_REPORT_PATH)
    save_dataframe(predictions, FINAL_PREDICTIONS_REPORT_PATH, index=True)

    saved = {
        "Pipeline": FINAL_PIPELINE_PATH,
        "Metadata": FINAL_METADATA_PATH,
        "Metrics": FINAL_METRICS_REPORT_PATH,
        "Coefficients": FINAL_COEFFICIENTS_REPORT_PATH,
        "Predictions": FINAL_PREDICTIONS_REPORT_PATH,
    }
    for name, path in saved.items():
        assert path.exists(), f"Artefact manquant : {name} ({path})"
    return saved


def verify_saved_pipeline(X_test: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
    loaded = joblib.load(FINAL_PIPELINE_PATH)
    assert np.array_equal(loaded.predict(X_test), y_pred)
    assert np.allclose(loaded.predict_proba(X_test)[:, 1], y_proba)
    logger.info("Vérification par rechargement réussie")


def is_model_valid_for_production(metrics: dict[str, float]) -> bool:
    valid = (
        metrics["f1"] >= MIN_F1_FOR_PRODUCTION
        and metrics["recall"] >= MIN_RECALL_FOR_PRODUCTION
    )
    logger.info(
        "Validation production : f1=%.4f (min %.2f) | recall=%.4f (min %.2f) -> %s",
        metrics["f1"], MIN_F1_FOR_PRODUCTION,
        metrics["recall"], MIN_RECALL_FOR_PRODUCTION,
        "OK" if valid else "REFUSÉ",
    )
    return valid


