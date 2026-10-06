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


class ClassificationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        with open(context.artifacts["metadata"], encoding="utf-8") as file:
            config = json.load(file)
        self.features = config["features"]
        self.risk_map = {int(k): v for k, v in config["cluster_mapping"].items()}

    def predict(self, context, model_input, params=None):
        X = model_input[self.features].astype("float64")
        clusters = np.asarray(self.pipeline.predict(X)).astype("int64")
        proba = self.pipeline.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "Cluster": clusters,
            "risk_category": [self.risk_map[int(c)] for c in clusters],
            "prob_cluster_1": proba.astype("float64"),
            "confidence": np.maximum(proba, 1 - proba).astype("float64"),
        })


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


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
):
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow : uri=%s, experiment=%s", uri, experiment_name)
    return mlflow


def log_pyfunc_model(
    mlflow,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    risk_mapping: dict[int, str],
    saved_artifacts: dict[str, Path],
):
    input_example = X_train.head(5).astype("float64")
    example_clusters = np.asarray(pipeline.predict(input_example)).astype("int64")
    example_proba = pipeline.predict_proba(input_example)[:, 1].astype("float64")
    output_example = pd.DataFrame({
        "Cluster": example_clusters,
        "risk_category": [risk_mapping[int(c)] for c in example_clusters],
        "prob_cluster_1": example_proba,
        "confidence": np.maximum(example_proba, 1 - example_proba).astype("float64"),
    })
    signature = infer_signature(input_example, output_example)

    model_info = mlflow.pyfunc.log_model(
        artifact_path=MLFLOW_MODEL_NAME,
        python_model=ClassificationModel(),
        artifacts={
            "pipeline": str(saved_artifacts["Pipeline"]),
            "metadata": str(saved_artifacts["Metadata"]),
        },
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            f"numpy=={metadata.version('numpy')}",
            f"pandas=={metadata.version('pandas')}",
            f"scikit-learn=={metadata.version('scikit-learn')}",
            f"joblib=={metadata.version('joblib')}",
        ],
    )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test.astype("float64"))
    assert np.array_equal(predictions["Cluster"].to_numpy(), np.asarray(y_pred))
    assert np.allclose(predictions["prob_cluster_1"].to_numpy(), y_proba)
    expected_risk = [risk_mapping[int(c)] for c in y_pred]
    assert list(predictions["risk_category"]) == expected_risk
    logger.info("Vérification du modèle MLflow rechargé réussie : %s", model_info.model_uri)
    return model_info


def log_classification_run(
    mlflow,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics: dict[str, float],
    cv_metrics: dict[str, float],
    risk_mapping: dict[int, str],
    dataset_signature: dict,
    saved_artifacts: dict[str, Path],
    confusion_matrix_path: Path,
    n_train: int,
    n_test: int,
) -> tuple[str, str]:
    classifier = pipeline.named_steps["classifier"]

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
        mlflow.log_params({
            "model_name": FINAL_MODEL_NAME,
            "scaler_name": FINAL_SCALER_NAME,
            "sampler_name": FINAL_SAMPLER_NAME,
            "C": float(classifier.C),
            "max_iter": int(classifier.max_iter),
            "penalty": str(classifier.penalty),
            "solver": str(classifier.solver),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_splits": CV_SPLITS,
            "n_features": len(FEATURES),
            "features": ",".join(FEATURES),
            "n_train": n_train,
            "n_test": n_test,
            "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
        })

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        })
        mlflow.log_metrics(cv_metrics)

        mlflow.set_tags({
            "step": "classification",
            "target": TARGET,
            "dataset_content_sha256": dataset_signature["content_sha256"],
            "dataset_schema_sha256": dataset_signature["schema_sha256"],
            "dataset_n_rows": str(dataset_signature["n_rows"]),
            "cluster_mapping": json.dumps(
                {str(k): v for k, v in risk_mapping.items()}, ensure_ascii=False),
        })

        mlflow.log_dict({"features": FEATURES}, "traceability/features.json")
        mlflow.log_dict(get_library_versions(), "traceability/library_versions.json")
        mlflow.log_dict(dataset_signature, "traceability/dataset_signature.json")

        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
        mlflow.log_artifact(str(saved_artifacts["Pipeline"]), artifact_path="pipeline")
        mlflow.log_artifact(str(saved_artifacts["Metadata"]), artifact_path="config")
        for name in ("Metrics", "Coefficients", "Predictions"):
            mlflow.log_artifact(str(saved_artifacts[name]), artifact_path="reports")

        model_info = log_pyfunc_model(
            mlflow, pipeline, X_train, X_test, y_pred, y_proba,
            risk_mapping, saved_artifacts,
        )
        mlflow.set_tag("model_uri", model_info.model_uri)

        run_id = run.info.run_id
        logger.info("Run MLflow terminé : %s", run_id)
        return run_id, model_info.model_uri


def register_and_promote_model(
    mlflow,
    model_uri: str,
    run_id: str,
    metrics: dict[str, float],
    promote: bool = True,
) -> dict:
    client = mlflow.MlflowClient()
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MLFLOW_REGISTERED_MODEL_NAME,
    )
    version = str(model_version.version)
    logger.info("Modèle enregistré : %s v%s", MLFLOW_REGISTERED_MODEL_NAME, version)

    client.set_model_version_tag(MLFLOW_REGISTERED_MODEL_NAME, version, "run_id", run_id)
    client.set_model_version_tag(MLFLOW_REGISTERED_MODEL_NAME, version, "f1_score",
                                f"{metrics['f1']:.4f}")
    client.set_model_version_tag(MLFLOW_REGISTERED_MODEL_NAME, version, "recall",
                                f"{metrics['recall']:.4f}")

    promoted = False
    if promote and is_model_valid_for_production(metrics):
        try:
            client.set_registered_model_alias(
                MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_PRODUCTION_ALIAS, version)
            promoted = True
            logger.info("Alias '%s' -> version %s", MLFLOW_PRODUCTION_ALIAS, version)
        except Exception as error:
            logger.warning("Alias impossible : %s", error)
    elif promote:
        logger.warning("Version %s non promue : seuils de validation non atteints", version)

    return {"name": MLFLOW_REGISTERED_MODEL_NAME, "version": version, "promoted": promoted}


def classification_pipeline(
    use_mlflow: bool = True,
    tracking_uri: str | None = None,
    register_model: bool = True,
    promote_to_production: bool = True,
) -> tuple[Pipeline, dict]:
    df = load_data()
    validate_data(df)

    dataset_signature = compute_dataset_signature(df[FEATURES + [TARGET]])
    risk_mapping = build_risk_mapping(df)

    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_final_pipeline()
    cv_metrics = run_cross_validation(pipeline, X_train, y_train)

    pipeline = train_pipeline(pipeline, X_train, y_train)

    metrics, cm, y_pred, y_proba = evaluate_pipeline(pipeline, X_test, y_test)
    predictions = build_predictions_report(X_test, y_test, y_pred, y_proba, risk_mapping)
    coefficients = compute_coefficients(pipeline, X_train)
    confusion_matrix_path = plot_confusion_matrix(y_test, y_pred)

    config = build_metadata(
        pipeline, metrics, cv_metrics, risk_mapping,
        dataset_signature, len(X_train), len(X_test),
    )
    saved_artifacts = save_artifacts(
        pipeline, config, metrics, cv_metrics, coefficients, predictions,
    )
    verify_saved_pipeline(X_test, y_pred, y_proba)

    run_id = None
    registry_info = None
    if use_mlflow:
        mlflow_module = setup_mlflow(tracking_uri)
        run_id, model_uri = log_classification_run(
            mlflow_module, pipeline, X_train, X_test, y_pred, y_proba,
            metrics, cv_metrics, risk_mapping, dataset_signature,
            saved_artifacts, confusion_matrix_path, len(X_train), len(X_test),
        )
        if register_model:
            registry_info = register_and_promote_model(
                mlflow_module, model_uri, run_id, metrics, promote=promote_to_production,
            )

    logger.info("Tous les artefacts ont été sauvegardés :")
    for name, path in saved_artifacts.items():
        logger.info("  %-14s : %s", name, path)

    summary = {
        "run_id": run_id,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "registry": registry_info,
        "artifacts": saved_artifacts,
    }
    return pipeline, summary


if __name__ == "__main__":
    final_pipeline, summary = classification_pipeline()
    print(f"\nModèle final : {FINAL_MODEL_NAME} (C={FINAL_C})")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name:<10}: {summary['metrics'][name]:.4f}")
    if summary["run_id"]:
        print(f"MLflow run_id : {summary['run_id']}")
    if summary["registry"]:
        print(f"Registry      : {summary['registry']}")