import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from .schemas import PatientFeatures, PredictionResponse, ThresholdAlert
import dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

dotenv.load_dotenv(dotenv_path=ROOT_DIR / ".env", override=True)

logger = logging.getLogger("api.utils")

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

RISK_HIGH_LABEL = "Risque Élevé"
RISK_LOW_LABEL = "Risque Faible"
DEFAULT_CLUSTER_MAPPING = {0: RISK_LOW_LABEL, 1: RISK_HIGH_LABEL}

RISK_THRESHOLDS = {
    "Glucose": (126.0, "Glycémie supérieure au seuil diabétique (126 mg/dL)"),
    "BMI": (30.0, "IMC en zone d'obésité (> 30 kg/m²)"),
    "DiabetesPedigreeFunction": (0.5, "Prédisposition héréditaire élevée (pedigree > 0,5)"),
}

MLFLOW_PRODUCTION_ALIAS = "production"

@dataclass(frozen=True)
class Settings:
    tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    model_name: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_NAME", "GlucoProfileClassifier"))
    model_stage: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_STAGE", "Production"))
    model_alias: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_ALIAS", MLFLOW_PRODUCTION_ALIAS))
    admin_api_key: str | None = field(default_factory=lambda: os.getenv("API_ADMIN_KEY") or None)
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    )


def get_settings() -> Settings:
    return Settings()

class ModelNotAvailableError(RuntimeError):
    pass

@dataclass
class LoadedModel:
    pipeline: object
    name: str
    version: str
    stage: str
    uri: str
    run_id: str | None
    cluster_mapping: dict[int, str]
    loaded_at: datetime


class ModelService:

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        
    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded(self) -> LoadedModel:
        if self._loaded is None:
            self.load()
        assert self._loaded is not None
        return self._loaded

    def _resolve_production_version(self, client: MlflowClient):
        s = self.settings
        try:
            versions = client.get_latest_versions(s.model_name, stages=[s.model_stage])
            if versions:
                return versions[0]
        except Exception as error:
            logger.warning("Résolution par stage impossible : %s", error)

        try:
            return client.get_model_version_by_alias(s.model_name, s.model_alias)
        except Exception as error:
            logger.warning("Résolution par alias impossible : %s", error)

        raise ModelNotAvailableError(
            f"Aucune version de '{s.model_name}' en stage '{s.model_stage}' "
            f"ni avec l'alias '{s.model_alias}'. Lancez l'entraînement d'abord."
        )

    @staticmethod
    def _load_cluster_mapping(client: MlflowClient, run_id: str | None) -> dict[int, str]:
        if run_id:
            try:
                raw = client.get_run(run_id).data.tags.get("cluster_mapping")
                if raw:
                    return {int(k): str(v) for k, v in json.loads(raw).items()}
            except Exception as error:
                logger.warning("Lecture du cluster_mapping impossible : %s", error)
        logger.warning("Mapping par défaut utilisé : %s", DEFAULT_CLUSTER_MAPPING)
        return dict(DEFAULT_CLUSTER_MAPPING)

    def load(self) -> LoadedModel:
        with self._lock:
            mlflow.set_tracking_uri(self.settings.tracking_uri)
            client = MlflowClient()
            try:
                mv = self._resolve_production_version(client)
                uri = f"models:/{mv.name}/{mv.version}"
                pipeline = mlflow.sklearn.load_model(uri)
            except ModelNotAvailableError:
                raise
            except Exception as error:
                raise ModelNotAvailableError(f"Échec du chargement du modèle : {error}") from error

            self._loaded = LoadedModel(
                pipeline=pipeline,
                name=mv.name,
                version=str(mv.version),
                stage=self.settings.model_stage,
                uri=uri,
                run_id=getattr(mv, "run_id", None),
                cluster_mapping=self._load_cluster_mapping(client, getattr(mv, "run_id", None)),
                loaded_at=datetime.now(timezone.utc),
            )
            logger.info("Modèle chargé : %s v%s (%s)", mv.name, mv.version, uri)
            return self._loaded

    def predict(self, patients: list[PatientFeatures]) -> list[PredictionResponse]:
        model = self.loaded
        frame = build_dataframe(patients)

        probas = model.pipeline.predict_proba(frame)
        classes = [int(c) for c in model.pipeline.classes_]
        high_idx = [i for i, c in enumerate(classes)
                    if model.cluster_mapping.get(c) == RISK_HIGH_LABEL]

        results = []
        for patient, row in zip(patients, probas):
            best = int(np.argmax(row))
            cluster = classes[best]
            category = model.cluster_mapping.get(cluster, RISK_LOW_LABEL)
            is_high = category == RISK_HIGH_LABEL
            risk_probability = float(sum(row[i] for i in high_idx))

            alerts = compute_alerts(patient)
            results.append(
                PredictionResponse(
                    cluster=cluster,
                    risk_category=category,
                    risk_level="high" if is_high else "low",
                    badge_color="red" if is_high else "green",
                    risk_probability=round(risk_probability, 4),
                    confidence=round(float(row[best]), 4),
                    alerts=alerts,
                    advice=build_advice(is_high, alerts),
                    model_name=model.name,
                    model_version=model.version,
                    predicted_at=datetime.now(timezone.utc),
                )
            )
        return results

def build_dataframe(patients: list[PatientFeatures]) -> pd.DataFrame:
    records = [p.model_dump() for p in patients]
    return pd.DataFrame(records, columns=FEATURES).astype("float64")


def compute_alerts(patient: PatientFeatures) -> list[ThresholdAlert]:
    values = patient.model_dump()
    return [
        ThresholdAlert(feature=f, value=float(values[f]), threshold=thr, message=msg)
        for f, (thr, msg) in RISK_THRESHOLDS.items()
        if values[f] > thr
    ]


def build_advice(is_high_risk: bool, alerts: list[ThresholdAlert]) -> list[str]:
    advice: list[str] = []
    if is_high_risk:
        advice += [
            "Programmer une consultation avec un médecin / diabétologue dans les meilleurs délais.",
            "Prévoir un bilan de confirmation : glycémie à jeun, HbA1c, éventuellement test HGPO.",
            "Mettre en place un suivi rapproché (glycémie et poids) tous les 3 mois.",
            "Conseiller une alimentation pauvre en sucres rapides et une activité physique régulière "
            "(au moins 150 min/semaine).",
        ]
    else:
        advice += [
            "Maintenir une hygiène de vie équilibrée (alimentation variée, activité physique régulière).",
            "Renouveler le dépistage une fois par an.",
        ]

    flagged = {a.feature for a in alerts}
    if "BMI" in flagged:
        advice.append("IMC élevé : envisager un accompagnement nutritionnel pour une perte de poids progressive.")
    if "Glucose" in flagged and not is_high_risk:
        advice.append("Glycémie au-dessus du seuil : un contrôle de confirmation est recommandé malgré le profil global.")
    if "DiabetesPedigreeFunction" in flagged:
        advice.append("Antécédents familiaux marqués : renforcer la surveillance, même en l'absence de symptômes.")

    advice.append("Cet outil est une aide à la décision et ne remplace pas un diagnostic médical.")
    return advice