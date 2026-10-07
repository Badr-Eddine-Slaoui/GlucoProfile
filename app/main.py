import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")
TIMEOUT = 30
BATCH_SIZE = 500

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

STRICTLY_POSITIVE = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Age",
]

st.set_page_config(page_title="GlucoProfile", page_icon="🩺", layout="wide")


def call_api(method, path, headers=None, **kwargs):
    try:
        response = requests.request(
            method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs
        )
    except requests.exceptions.ConnectionError:
        return None, "Impossible de joindre l'API. Vérifiez que le service fastapi est démarré."
    except requests.exceptions.Timeout:
        return None, "L'API a mis trop de temps à répondre."

    if response.status_code == 200:
        return response.json(), None

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    if response.status_code == 401:
        return None, "Clé API invalide ou manquante."
    if response.status_code == 422:
        return None, "Valeurs invalides. Vérifiez les données envoyées."
    if response.status_code == 503:
        return None, f"Modèle indisponible : {detail}"
    return None, f"Erreur API ({response.status_code}) : {detail}"


def show_badge(risk_level, label):
    color = "#d62728" if risk_level == "high" else "#2ca02c"
    st.markdown(
        f"""
        <div style="background-color:{color}; color:white; padding:16px;
                    border-radius:10px; font-size:26px; font-weight:bold;
                    text-align:center;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sidebar():
    st.sidebar.title("🩺 GlucoProfile")
    st.sidebar.caption("Estimation du risque de diabète")
    st.sidebar.divider()
    st.sidebar.subheader("État du système")

    health, error = call_api("GET", "/health")
    if error:
        st.sidebar.error("API hors ligne")
        return

    st.sidebar.success("API en ligne")

    if health["model_loaded"]:
        st.sidebar.success("Modèle chargé")
        st.sidebar.write(f"**Modèle :** {health['model_name']}")
        st.sidebar.write(f"**Version :** {health['model_version']}")
    else:
        st.sidebar.warning("Aucun modèle de production chargé")

    st.sidebar.divider()
    st.sidebar.caption(f"API : {API_URL}")


def show_single_result(prediction):
    with st.container(border=True):
        show_badge(prediction["risk_level"], prediction["risk_category"])

        st.write("")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilité de risque élevé", f"{prediction['risk_probability'] * 100:.1f} %")
        col2.metric("Confiance du modèle", f"{prediction['confidence'] * 100:.1f} %")
        col3.metric("Cluster prédit", prediction["cluster"])

        st.progress(prediction["risk_probability"])

    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.subheader("Seuils dépassés")
            if prediction["alerts"]:
                for alert in prediction["alerts"]:
                    st.warning(
                        f"{alert['message']}\n\n"
                        f"Valeur : {alert['value']} | Seuil : {alert['threshold']}"
                    )
            else:
                st.success("Aucun seuil critique dépassé.")

    with col_right:
        with st.container(border=True):
            st.subheader("Conseils de suivi")
            for item in prediction["advice"]:
                st.write(f"- {item}")

    st.caption(
        f"Modèle {prediction['model_name']} v{prediction['model_version']} "
        f"— prédit le {prediction['predicted_at']}"
    )


def tab_single():
    st.subheader("Saisie des constantes cliniques")

    with st.form("patient_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pregnancies = st.number_input("Grossesses", 0.0, 20.0, 1.0, 1.0)
            insulin = st.number_input("Insuline (µU/mL)", 1.0, 1500.0, 100.0, 1.0)
        with col2:
            glucose = st.number_input("Glycémie (mg/dL)", 1.0, 600.0, 120.0, 1.0)
            bmi = st.number_input("IMC (kg/m²)", 1.0, 80.0, 28.0, 0.1)
        with col3:
            blood_pressure = st.number_input("Pression artérielle (mm Hg)", 1.0, 250.0, 70.0, 1.0)
            dpf = st.number_input("Pedigree du diabète", 0.0, 5.0, 0.4, 0.01)
        with col4:
            skin_thickness = st.number_input("Pli cutané (mm)", 1.0, 120.0, 25.0, 1.0)
            age = st.number_input("Âge (années)", 1.0, 120.0, 35.0, 1.0)

        submitted = st.form_submit_button("Évaluer le risque", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }
        with st.spinner("Analyse en cours..."):
            prediction, error = call_api("POST", "/predict", json=payload)

        if error:
            st.session_state.pop("single_prediction", None)
            st.error(error)
        else:
            st.session_state["single_prediction"] = prediction

    if "single_prediction" in st.session_state:
        st.divider()
        show_single_result(st.session_state["single_prediction"])


def prepare_csv(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, f"Colonnes manquantes : {', '.join(missing)}"

    df = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    invalid = df.isna().any(axis=1) | (df[STRICTLY_POSITIVE] <= 0).any(axis=1)
    if invalid.any():
        return None, df[invalid]

    return df, None


def run_batch(df):
    predictions = []
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start:start + BATCH_SIZE]
        payload = {"patients": chunk.astype(float).to_dict("records")}
        data, error = call_api("POST", "/predict/batch", json=payload)
        if error:
            return None, error
        predictions.extend(data["predictions"])
    return predictions, None


def tab_batch():
    st.subheader("Analyse d'un fichier de patients")
    st.write(
        "Importez un fichier CSV contenant les colonnes : "
        + ", ".join(f"`{c}`" for c in FEATURES)
        + "."
    )

    template = pd.DataFrame(
        [[6, 148, 72, 35, 219, 33.6, 0.627, 50], [1, 85, 66, 29, 70, 26.6, 0.351, 31]],
        columns=FEATURES,
    )
    st.download_button(
        "Télécharger un modèle de fichier",
        template.to_csv(index=False).encode("utf-8"),
        file_name="modele_patients.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception:
        st.error("Impossible de lire ce fichier CSV.")
        return

    df, problem = prepare_csv(raw)

    if df is None and isinstance(problem, str):
        st.error(problem)
        return

    if df is None:
        st.error(
            f"{len(problem)} ligne(s) contiennent des valeurs manquantes ou nulles "
            "dans des colonnes qui doivent être strictement positives."
        )
        st.dataframe(problem, use_container_width=True)
        return

    st.write(f"{len(df)} patients prêts à être analysés.")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Lancer l'analyse", type="primary"):
        with st.spinner("Analyse en cours..."):
            predictions, error = run_batch(df)

        if error:
            st.error(error)
            return

        results = pd.DataFrame(predictions)[
            ["risk_category", "risk_probability", "confidence", "cluster"]
        ]
        st.session_state["batch_results"] = pd.concat(
            [df.reset_index(drop=True), results], axis=1
        )

    if "batch_results" in st.session_state:
        show_batch_results(st.session_state["batch_results"])


def show_batch_results(results):
    st.divider()
    st.subheader("Résultats")

    total = len(results)
    high = int((results["risk_category"] == "Risque Élevé").sum())
    low = total - high

    col1, col2, col3 = st.columns(3)
    col1.metric("Patients analysés", total)
    col2.metric("Risque élevé", f"{high} ({high / total * 100:.1f} %)")
    col3.metric("Risque faible", f"{low} ({low / total * 100:.1f} %)")

    st.bar_chart(results["risk_category"].value_counts())

    st.dataframe(
        results,
        use_container_width=True,
        column_config={
            "risk_category": "Catégorie",
            "risk_probability": st.column_config.ProgressColumn(
                "Probabilité de risque", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "confidence": st.column_config.NumberColumn("Confiance", format="%.2f"),
            "cluster": "Cluster",
        },
    )

    st.download_button(
        "Télécharger les résultats (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="resultats_predictions.csv",
        mime="text/csv",
    )


def tab_model():
    st.subheader("Modèle en production")

    info, error = call_api("GET", "/model/info")
    if error:
        st.error(error)
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Modèle", info["model_name"])
        col2.metric("Version", info["model_version"])
        col3.metric("Stage", info["stage"])

        with st.container(border=True):
            st.write(f"**URI du modèle :** `{info['model_uri']}`")
            st.write(f"**Run MLflow :** `{info['run_id']}`")
            st.write(f"**Serveur de tracking :** `{info['tracking_uri']}`")
            st.write(f"**Chargé le :** {info['loaded_at']}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Variables utilisées**")
            st.dataframe(
                pd.DataFrame({"Variable": info["features"]}),
                use_container_width=True,
                hide_index=True,
            )
        with col_right:
            st.write("**Correspondance cluster → risque**")
            mapping = pd.DataFrame(
                {
                    "Cluster": list(info["cluster_mapping"].keys()),
                    "Catégorie": list(info["cluster_mapping"].values()),
                }
            )
            st.dataframe(mapping, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Vérification et rechargement")

    col_ready, col_reload = st.columns(2)

    with col_ready:
        with st.container(border=True):
            st.write("**Vérifier la disponibilité du modèle**")
            if st.button("Tester /ready"):
                ready, error = call_api("GET", "/ready")
                if error:
                    st.error(error)
                else:
                    st.success(f"Prêt : {ready['model_name']} v{ready['model_version']}")

    with col_reload:
        with st.container(border=True):
            st.write("**Recharger le modèle de production**")
            admin_key = st.text_input("Clé API admin (si définie)", type="password")
            if st.button("Recharger le modèle"):
                headers = {"X-API-Key": admin_key} if admin_key else None
                reloaded, error = call_api("POST", "/model/reload", headers=headers)
                if error:
                    st.error(error)
                else:
                    st.success(
                        f"Modèle rechargé : {reloaded['model_name']} v{reloaded['model_version']}"
                    )


def main():
    show_sidebar()

    st.title("Estimation du risque de diabète")
    st.caption("Outil d'aide à la décision pour les praticiens du laboratoire.")

    tab1, tab2, tab3 = st.tabs(["👤 Patient", "📄 Fichier de patients", "⚙️ Modèle"])

    with tab1:
        tab_single()
    with tab2:
        tab_batch()
    with tab3:
        tab_model()

    st.divider()
    st.caption("Cet outil ne remplace pas un diagnostic médical.")


main()