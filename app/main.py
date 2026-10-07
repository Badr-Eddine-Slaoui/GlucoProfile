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




def main():
    show_sidebar()
    tab_single()

if __name__ == '__main__':
    main()
