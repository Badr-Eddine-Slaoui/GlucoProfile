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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("clustering")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

dotenv.load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEANED_DATA_PATH = PROCESSED_DIR / "diabetes_cleaned.csv"
TRANSFORMED_DATA_PATH = PROCESSED_DIR / "diabetes_transformed.csv"

FINAL_CLUSTERED_DATA_PATH = PROCESSED_DIR / "diabetes_clustered.csv"
FINAL_SCALER_PATH = MODELS_DIR / "robust_scaler_final.joblib"
FINAL_KMEANS_PATH = MODELS_DIR / "kmeans_final.joblib"
FINAL_PCA_PATH = MODELS_DIR / "pca_final.joblib"
FINAL_CONFIG_PATH = MODELS_DIR / "final_clustering_config.json"

FINAL_PROFILE_REPORT_PATH = REPORTS_DIR / "final_cluster_profiles.csv"
FINAL_RISK_REPORT_PATH = REPORTS_DIR / "final_risk_profile.csv"
FINAL_PCA_REPORT_PATH = REPORTS_DIR / "final_pca_variance.csv"
PCA_PLOT_PATH = REPORTS_DIR / "clustering_pca_projection.png"
SIZE_PLOT_PATH = REPORTS_DIR / "clustering_cluster_sizes.png"

RANDOM_STATE = 42

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

FINAL_MODEL_NAME = "KMeans"
FINAL_SCALER_NAME = "RobustScaler"
FINAL_K = 2
FINAL_N_INIT = 10
FINAL_FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RISK_THRESHOLDS = {
    "Glucose": 126,
    "BMI": 30,
    "DiabetesPedigreeFunction": 0.5,
}

RISK_HIGH_LABEL = "Risque Élevé"
RISK_LOW_LABEL = "Risque Faible"

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "GlucoProfile")
MLFLOW_RUN_NAME = "clustering_kmeans_robust_k2"
DEFAULT_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_PRODUCTION_ALIAS = os.getenv("MLFLOW_PRODUCTION_ALIAS", "production")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "glucoprofile-clustering")
MLFLOW_MODEL_NAME = "model"

LIBRARIES_TO_LOG = [
    "numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "mlflow",
]


def load_data(
    cleaned_path: Path = CLEANED_DATA_PATH,
    transformed_path: Path = TRANSFORMED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (cleaned_path, transformed_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    df_clean = pd.read_csv(cleaned_path)
    df_transformed = pd.read_csv(transformed_path)
    logger.info("Données chargées : clean=%s, transformed=%s",
                df_clean.shape, df_transformed.shape)
    return df_clean, df_transformed


def validate_data(df_clean: pd.DataFrame, df_transformed: pd.DataFrame) -> None:
    assert df_clean.shape == df_transformed.shape, "Dimensions différentes"
    assert list(df_clean.columns) == FEATURES, "Colonnes inattendues (clean)"
    assert list(df_transformed.columns) == FEATURES, "Colonnes inattendues (transformed)"
    assert df_transformed.isna().sum().sum() == 0, "Valeurs manquantes (transformed)"
    assert df_clean.isna().sum().sum() == 0, "Valeurs manquantes (clean)"
    assert np.isfinite(df_transformed.to_numpy()).all(), "Valeurs infinies"
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


def build_clustering_matrix(
    df_transformed: pd.DataFrame,
    features: list[str] = FINAL_FEATURES,
) -> pd.DataFrame:
    X_final = df_transformed[features].copy()
    assert X_final.isna().sum().sum() == 0
    assert np.isfinite(X_final.to_numpy()).all()
    return X_final


def fit_scaler(X: pd.DataFrame) -> tuple[RobustScaler, np.ndarray]:
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape
    assert np.isfinite(X_scaled).all()
    logger.info("RobustScaler ajusté")
    return scaler, X_scaled


def train_kmeans(X_scaled: np.ndarray) -> tuple[KMeans, np.ndarray]:
    kmeans = KMeans(
        n_clusters=FINAL_K,
        n_init=FINAL_N_INIT,
        random_state=RANDOM_STATE,
    )
    raw_labels = kmeans.fit_predict(X_scaled)
    logger.info("K-Means entraîné : k=%d, inertie=%.3f", FINAL_K, kmeans.inertia_)
    return kmeans, raw_labels


def relabel_by_glucose(
    raw_labels: np.ndarray,
    df_clean: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, int]]:
    glucose_means = df_clean.groupby(raw_labels)["Glucose"].mean().sort_values()
    mapping = {int(old): new for new, old in enumerate(glucose_means.index)}
    final_labels = np.array([mapping[int(label)] for label in raw_labels])
    assert set(final_labels) == set(range(FINAL_K))

    for old, new in mapping.items():
        logger.info("Cluster brut %d -> Cluster %d (Glucose moyen = %.2f)",
                    old, new, glucose_means.loc[old])
    return final_labels, mapping


def add_cluster_column(df_clean: pd.DataFrame, final_labels: np.ndarray) -> pd.DataFrame:
    df_clustered = df_clean.copy()
    df_clustered["Cluster"] = final_labels
    assert df_clustered["Cluster"].isna().sum() == 0
    return df_clustered


def compute_clustering_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    kmeans: KMeans,
) -> dict[str, float]:
    sizes_pct = pd.Series(labels).value_counts(normalize=True).sort_index() * 100
    metrics = {
        "inertia": float(kmeans.inertia_),
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        "min_cluster_pct": float(sizes_pct.min()),
        "max_cluster_pct": float(sizes_pct.max()),
        "n_iter": float(kmeans.n_iter_),
    }
    logger.info("Silhouette=%.3f | Davies-Bouldin=%.3f | Calinski-Harabasz=%.1f",
                metrics["silhouette"], metrics["davies_bouldin"],
                metrics["calinski_harabasz"])
    return metrics


def compute_cluster_profiles(df_clustered: pd.DataFrame) -> pd.DataFrame:
    profiles = df_clustered.groupby("Cluster")[FEATURES].mean()
    counts = df_clustered["Cluster"].value_counts().sort_index()
    profiles.insert(0, "count", counts)
    profiles.insert(1, "percentage", counts / len(df_clustered) * 100)
    return profiles


def compute_risk_profile(
    cluster_profiles: pd.DataFrame,
    thresholds: dict[str, float] = RISK_THRESHOLDS,
) -> pd.DataFrame:
    risk_profile = cluster_profiles.copy()
    for feature, threshold in thresholds.items():
        risk_profile[f"{feature}_threshold_exceeded"] = risk_profile[feature] > threshold
    risk_profile["criteres_depasses"] = sum(
        risk_profile[f"{feature}_threshold_exceeded"].astype(int) for feature in thresholds
    )
    return risk_profile


def identify_high_risk_cluster(
    risk_profile: pd.DataFrame,
    thresholds: dict[str, float] = RISK_THRESHOLDS,
) -> int:
    mask = pd.Series(True, index=risk_profile.index)
    for feature, threshold in thresholds.items():
        mask &= risk_profile[feature] > threshold
    high_risk = risk_profile.index[mask].tolist()

    if len(high_risk) != 1:
        raise ValueError(
            "Le projet suppose un unique cluster à haut risque, "
            f"mais {len(high_risk)} clusters satisfont les trois seuils : {high_risk}"
        )
    return int(high_risk[0])


def add_risk_category(
    df_clustered: pd.DataFrame,
    high_risk_cluster: int,
) -> tuple[pd.DataFrame, dict[int, str]]:
    risk_mapping = {
        int(cluster): (RISK_HIGH_LABEL if cluster == high_risk_cluster else RISK_LOW_LABEL)
        for cluster in sorted(df_clustered["Cluster"].unique())
    }
    result = df_clustered.copy()
    result["risk_category"] = result["Cluster"].map(risk_mapping)
    assert result["risk_category"].isna().sum() == 0
    for cluster, category in risk_mapping.items():
        logger.info("Cluster %d -> %s", cluster, category)
    return result, risk_mapping


def fit_pca(X_scaled: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    assert X_pca.shape == (X_scaled.shape[0], 2)
    logger.info("PCA : variance expliquée PC1+PC2 = %.2f %%",
                pca.explained_variance_ratio_.sum() * 100)
    return pca, X_pca


def compute_pca_variance(pca: PCA) -> pd.Series:
    return pd.Series(pca.explained_variance_ratio_ * 100, index=["PC1", "PC2"],
                     name="variance_expliquee_%")


def plot_pca_clusters(
    X_pca: np.ndarray,
    labels: np.ndarray,
    centroids_pca: np.ndarray,
    variance_pct: pd.Series,
    path: Path = PCA_PLOT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=35, alpha=0.65,
                   label=f"Cluster {cluster_id}")
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker="X", s=220,
               edgecolor="black", linewidth=1.5, label="Centroïdes")
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.axvline(0, linewidth=0.8, alpha=0.4)
    ax.set_xlabel(f"PC1 ({variance_pct['PC1']:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance_pct['PC2']:.1f}% variance)")
    ax.set_title("Projection PCA des profils patients selon leur cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cluster_sizes(
    df_clustered: pd.DataFrame,
    path: Path = SIZE_PLOT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = df_clustered["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    for cluster_id, count in counts.items():
        ax.text(str(cluster_id), count, f"{count}\n({count / len(df_clustered) * 100:.1f} %)",
                ha="center", va="bottom")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Nombre de patients")
    ax.set_title("Répartition des patients par cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


